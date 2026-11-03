"""Policy-class ceiling (M3 diagnostic, iteration 005): the best return the flow policy family reaches when its
parameters are searched directly by the cross-entropy method (CEM) instead of learned by Map-STDP.
At the flow level pi(o) = M v(o) with M the n-hop operator of Eq. 1a, so the action shares are p = R v_C(o) with
R = H_A^T M[:, C] (actions x controller, nonnegative) and Eq. 1d is P(a | o) = (R v_C)_a / sum (R v_C).
  learned:  R of the W Map-STDP learned (flow, v3, beta_LA 0.3, --td-norm 0.04), frozen;
  level 2:  CEM over the controller -> action synapses of that W (bounds [w_min, w_max], mask, alpha, the latents and
            everything else as learned), Eq. 1d readout; the realistic ceiling;
  level 1:  CEM over any R = exp(theta), readouts Eq. 1d, Eq. 1e (beta<B>, e.g. beta4) and argmax; the generous upper
            bound. Iteration 010 (Proposal) adds a decision-bound readout on the embedded jump chain of the action
            spikes (each action spike lands on module a with probability p_a / sum p):
            db<T>: count-difference race (Eq. 1h), stop when one action leads every other by T spikes (K = 2: Eq. 1e,
            beta = T); the leader (ties uniform) at the deadline of --nmax action spikes. (The MSPRT readout
            msprt<T>c<C> was removed in iteration 013; its result files still load in --summary.)
Run from the repo root:
  python -m experiments.ceiling --task Acrobot --level {learned,2,1} [--readout {1d,beta<B>,argmax,db<T>}]
    [--nmax N] [--seed 0] [--encoder SPEC] [--no-train]
    -> experiments/results/ceiling_<task>_<level>[_<readout>][_enc<SPEC>]_s<seed>.json
  --encoder SPEC (iteration 012): the controller encoder tasks.with_encoder(task, SPEC). --no-train (level 1 only):
  skip the training run, which level 1 does not use (CEM starts from R = 1 with its own rng), so the result is the
  same; train_last100 is then null.
  python -m experiments.ceiling --summary"""
import argparse
import glob
import json
import pathlib
import re
import time
from functools import lru_cache

import gymnasium as gym
import numpy as np

from mapstdp import core
from models import basic
from models.basic import TASKS

POP, ELITE, ITERS, EPIS, FINAL = 64, 8, 80, 4, 100  # CEM population, elites, iterations, episodes/candidate, final
WMIN, WMAX, TD_NORM, LA_BETA = 1e-3, 1.0, 0.04, 0.3


def hop_operator(W, alpha=basic.ALPHA, n=basic.N_HOPS):
    """M with pi_n = M v for core.flow(W, v, alpha, n): M_0 = I, M_{k+1} = (1 - alpha) T M_k + alpha I (Eq. 1a)."""
    B, M = (1 - alpha) * W / W.sum(0), np.eye(len(W))
    for _ in range(n):
        M = B @ M + alpha * np.eye(len(W))
    return M


def action_matrix(W, agent):
    """R = H_A^T M[:, C]: action-module shares p_A = R v_C for any stimulus."""
    H = (agent.m[:, None] == agent.A[None, :]).astype(float)
    return H.T @ hop_operator(W)[:, agent.ctrl]


NMAX = 1024  # decision-bound readouts: deadline in action spikes (--nmax)


def jump_counts(p, nmax, rng):
    """Cumulative action-spike counts (nmax x K) of the embedded jump chain: spike s lands on a w.p. p_a / sum p."""
    idx = np.minimum(np.searchsorted(np.cumsum(p / p.sum()), rng.random(nmax), side="right"), len(p) - 1)
    return np.cumsum(np.eye(len(p), dtype=int)[idx], 0)


def bound_decision(p, theta, nmax, rng):
    """Decision-bound readout (Eq. 1h, module doc): (action, action spikes used, deadline hit); stop when
    n_a - max_{b != a} n_b >= theta. Deadline: leader, ties uniform."""
    n = jump_counts(p, nmax, rng)
    top2 = np.sort(n, 1)[:, -2:]
    hit = np.flatnonzero(top2[:, 1] - top2[:, 0] >= theta)
    if len(hit):
        return int(np.argmax(n[hit[0]])), int(hit[0]) + 1, False
    lead = np.flatnonzero(n[-1] == n[-1].max())
    return int(lead[0] if len(lead) == 1 else rng.choice(lead)), nmax, True


@lru_cache
def parse_readout(readout):
    """readout -> (kind, theta): ('1d'|'argmax', None), ('beta', B), ('db', T)."""
    m = re.fullmatch(r"(beta|db)([\d.]+)", readout)
    assert readout in ("1d", "argmax") or m, readout
    return (readout, None) if not m else (m[1], float(m[2]))


def act(p, readout, rng, nmax=NMAX, stats=None):
    """Action from the action shares p under the readout; stats (list) collects (spikes, deadline) of bound readouts."""
    kind, b = parse_readout(readout)
    if kind == "argmax":
        return int(np.argmax(p))
    if kind == "db":
        a, n, dl = bound_decision(p, b, nmax, rng)
        if stats is not None:
            stats.append((n, dl))
        return a
    q = p ** b if kind == "beta" else p
    if not q.sum() > 0 or not np.isfinite(q.sum()):  # p^beta under/overflow (iteration 011): rescale first
        q = (p / p.max()) ** b
    return int(rng.choice(len(q), p=q / q.sum()))


def rollouts(R, agent, env, readout, seeds, nmax=NMAX, stats=None):
    """Mean return of the frozen policy over episodes with the given env seeds (policy rng seeded likewise)."""
    out = []
    for s in seeds:
        o, _ = env.reset(seed=int(s))
        rng, total, done = np.random.default_rng(s), 0.0, False
        while not done:
            o, r, term, trunc, _ = env.step(act(R @ agent.stim(o)[agent.ctrl], readout, rng, nmax, stats))
            total, done = total + r, term or trunc
        out.append(total)
    return float(np.mean(out))


def trained_agent(task, seed, encoder=None, train=True):
    """The Map-STDP flow agent after the task's training episodes (the M3 flow reference with --td-norm)."""
    agent = basic.Agent(basic.Config("flow", seed=seed, la_beta=LA_BETA, task=task, td_norm=TD_NORM, encoder=encoder))
    env, R = gym.make(agent.task.env_id), []
    for _ in range(agent.task.episodes if train else 0):
        R.append(agent.episode(env))
    return agent, float(np.mean(R[-100:])) if train else None


def cem(score, mu, sd, rng, log):
    """Diagonal-Gaussian CEM maximising score(theta, seeds); common random numbers (one seed set) per iteration."""
    best = (-np.inf, mu)
    for it in range(ITERS):
        seeds = rng.integers(1 << 30, size=EPIS)
        th = mu + sd * rng.standard_normal((POP, len(mu)))
        f = np.array([score(t, seeds) for t in th])
        el = th[np.argsort(f)[-ELITE:]]
        mu, sd = el.mean(0), el.std(0) + 0.02  # noise floor keeps the search from freezing
        if f.max() > best[0]:
            best = (float(f.max()), th[np.argmax(f)])
        log(f"it {it} best {f.max():.1f} mean {f.mean():.1f} elite {np.sort(f)[-ELITE:].mean():.1f} sd {sd.mean():.3f}")
    return mu


def run(task, level, readout, seed, nmax=NMAX, encoder=None, train=True):
    t0 = time.perf_counter()
    agent, train_ret = trained_agent(task, seed, encoder, train)
    env, rng, W0 = gym.make(agent.task.env_id), np.random.default_rng(seed + 1000), agent.W.copy()
    R0 = action_matrix(W0, agent)
    v = agent.held[0]  # R reproduces the agent's own Eq. 1d policy
    assert np.allclose(R0 @ v[agent.ctrl] / (R0 @ v[agent.ctrl]).sum(),
                       core.policy(np.bincount(agent.m, core.flow(W0, v, basic.ALPHA, basic.N_HOPS), agent.K),
                                   agent.roles))
    log = lambda s: print(f"{task} L{level} {readout}: {s} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)  # noqa: E731
    log(f"trained, last100 training return {train_ret:.1f}" if train else "untrained (level 1)")
    if level == "learned":
        R = R0
    elif level == "2":
        sel = agent.mask & (agent.roles[agent.m] == "A")[:, None] & agent.ctrl[None, :]  # controller -> action
        def to_R(th):
            W = W0.copy()
            W[sel] = WMIN + (WMAX - WMIN) / (1 + np.exp(-th))
            return action_matrix(W, agent)
        x = np.clip((W0[sel] - WMIN) / (WMAX - WMIN), 1e-4, 1 - 1e-4)
        R = to_R(cem(lambda th, s: rollouts(to_R(th), agent, env, readout, s, nmax), np.log(x / (1 - x)),
                     np.ones(sel.sum()), rng, log))
    else:
        R = np.exp(cem(lambda th, s: rollouts(np.exp(th.reshape(R0.shape)), agent, env, readout, s, nmax),
                       np.zeros(R0.size), 2 * np.ones(R0.size), rng, log).reshape(R0.shape))
    st = []  # (action spikes, deadline hit) per decision of a bound readout
    final = [rollouts(R, agent, env, readout, [s], nmax, st)
             for s in np.random.default_rng(99).integers(1 << 30, size=FINAL)]
    res = dict(task=task, level=level, readout=readout, seed=seed, train_last100=train_ret,
               final_mean=float(np.mean(final)), final_sd=float(np.std(final)), final=final,
               minutes=(time.perf_counter() - t0) / 60,
               params=dict(pop=POP, elite=ELITE, iters=ITERS, epis=EPIS, td_norm=TD_NORM, la_beta=LA_BETA),
               **({"encoder_spec": encoder, "n_ctrl": agent.task.n_ctrl, "R": R.tolist()} if encoder else {}))
    if st:
        res.update(nmax=nmax, spikes_mean=float(np.mean([n for n, _ in st])),
                   deadline_frac=float(np.mean([d for _, d in st])), R=R.tolist())
    suffix = "" if level == "learned" else f"_{readout}" + (f"_n{nmax}" if st and nmax != NMAX else "")
    suffix += f"_enc{encoder}" if encoder else ""
    (basic.OUT / f"ceiling_{task}_L{level}{suffix}_s{seed}.json").write_text(json.dumps(res))
    log(f"final {res['final_mean']:.1f} +- {res['final_sd']:.1f} over {FINAL} fresh episodes")


def summary():
    rows = {}
    for f in sorted(glob.glob(str(basic.OUT / "ceiling_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        rows.setdefault(r["task"], []).append(r)
    for task, rs in rows.items():
        tr = [r["train_last100"] for r in rs if r["train_last100"] is not None]
        print(f"\n{task}: Map-STDP training last100 " + (f"{tr[0]:.1f}" if tr else "n/a"))
        for r in sorted(rs, key=lambda r: (r["level"] != "learned", r["level"], r["readout"])):
            extra = f"  spikes {r['spikes_mean']:.1f} deadline {r['deadline_frac']:.3f}" if "nmax" in r else ""
            print(f"  level {r['level']:7} {r['readout']:9} {r.get('encoder_spec') or '':9} s{r['seed']}  "
                  f"{r['final_mean']:8.1f} +- "
                  f"{r['final_sd']:6.1f}  ({r['minutes']:.0f} min)" + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS))
    ap.add_argument("--level", choices=("learned", "2", "1"), default="2")
    ap.add_argument("--readout", default="1d", help="1d, argmax, beta<B> or db<T> (module doc)")
    ap.add_argument("--nmax", type=int, default=NMAX, help="deadline (action spikes) of db")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default=None, help="controller encoder spec (iteration 012, tasks.with_encoder)")
    ap.add_argument("--no-train", action="store_true", help="level 1: skip the (unused) training run")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    assert a.level == "1" or a.readout == "1d", "levels learned and 2 use the model's own readout, Eq. 1d"
    parse_readout(a.readout)
    assert not a.no_train or a.level == "1", "--no-train: level 1 only"
    run(a.task, a.level, a.readout, a.seed, a.nmax, a.encoder, not a.no_train)


if __name__ == "__main__":
    main()
