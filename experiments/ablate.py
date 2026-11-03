"""Ablations of saved models (iteration 017): what a trained network loses when part of its structure is removed, with
plasticity off. One run = one saved model evaluated under every condition of its family, on common random numbers
(each condition restarts from the same evaluation stream), so the unit of the statistics is the model.

Family 'v4' (emergent-v4 models, models.basic.Agent; W[post, pre], modules C, L1, L2, A1, A2): every engine divides
a neuron's outgoing weights by their sum (T = W / d), so a cut either loses the drive (lesion: the cut neuron's own
outgoing weights go, its targets' input from it is gone, nobody else's d changes) or reroutes it (removal: cutting a
neuron's targets renormalises the rest of its outgoing weights). Conditions:
  intact            the saved model;
  lesion_L          the latents' outgoing synapses cut (W[:, L] = 0): they still receive and spike but drive nothing;
                    the controller's drive into them is lost;
  lesion_L1/2       the same for one latent module;
  remove_L          the latents disconnected (W[:, L] = W[L, :] = 0): the controller's outflow renormalises onto the
                    actions (the direct path only, at full gain);
  latent_path_only  controller -> action synapses cut (W[A, C] = 0): the controller's outflow renormalises onto the
                    latents, so actions are reached only through them.
Flow backend: a column of zeros has d = 0 (core.flow divides unguarded), so a cut latent keeps a 1e-12 self-loop: flow
that enters it stays there, which leaves the controller's and the actions' flow, hence the policy, as if it were
discarded. A frame with no action flow (flow) or no decision (spiking, the basic model's silent frames) takes a
uniform action.

Family 'ck' (race-study models, models.tdstdp.load_ck_model; HybridRec kicks b = chi R / d over the actor E
posts, modules = action columns): the lesions keep the intact d (the cut drive is lost, not rerouted):
  intact            the saved model;
  rec_off           every recurrent kick removed (b = 0);
  cross_lesion      the kicks between columns removed (a no-op on sealed columns: a built-in check);
  within_lesion     the kicks within columns removed;
  rec_untrained     the recurrence at its initial, untrained weights (R_ep0, its own d): the recurrent drive kept, what
                    the recurrence learned removed (a control for rec_off, which also removes drive).
A model without recurrent columns (a 'base' arm) gets the intact condition only: the reference of a policy trained
without columns, under the same evaluation.

Per condition: the returns of n episodes (ck: the first ceil(n / H) episodes of each copy, so short episodes are not
over-sampled), and monitors (v4: silent-frame share, flow or spike shares of the action and latent modules, spikes per
head and frame, the policy shift on the held-out stimuli; ck: actor spikes per decision and to the decision, race
decisions at the bound / deadline / tie). v4 also re-runs basic.frozen_returns on the intact model and compares it with
the training run's record (flow, branching and LIF reproduce it exactly; QIF's warm state is not saved).

Usage: python -m experiments.ablate --family v4|ck --model PATH [--run PATH] [--episodes N] [--seed S] --out PATH"""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import time

import gymnasium as gym
import numpy as np

from mapstdp import core
from models import basic, tdstdp

CONDITIONS = {"v4": ("intact", "lesion_L", "lesion_L1", "lesion_L2", "remove_L", "latent_path_only"),
              "ck": ("intact", "rec_off", "cross_lesion", "within_lesion", "rec_untrained")}
EPISODES = {"v4": 50, "CartPole": 50, "Acrobot": 50, "LunarLander": 96}  # per condition (ck: per task)
ABL_SEED, HELD_REPS, EPS_LOOP = 4721, 10, 1e-12


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---- family v4 ---------------------------------------------------------------------------------------------------

def v4_weights(W, m, roles, cond, flow):
    """The condition's W (module doc); flow: a 1e-12 self-loop on each latent left without outgoing weight."""
    W = W.copy()
    lat = np.flatnonzero(roles == "L")
    L = np.flatnonzero(np.isin(m, lat))
    if cond == "intact":
        return W
    if cond.startswith("lesion_L"):
        cut = L if cond == "lesion_L" else np.flatnonzero(m == lat[int(cond[-1]) - 1])
        W[:, cut] = 0
    elif cond == "remove_L":
        cut = L
        W[:, L] = 0
        W[L, :] = 0
    elif cond == "latent_path_only":
        A, C = np.isin(m, np.flatnonzero(roles == "A")), np.isin(m, np.flatnonzero(roles == "C"))
        W[np.ix_(A, C)] = 0
        return W
    else:
        raise ValueError(cond)
    if flow:
        W[cut, cut] = EPS_LOOP
    return W


def v4_agent(model, run):
    """(agent, final beta, the run's record) of an emergent-v4 model: the final snapshot and the last evaluation's
    beta (the adaptive beta of the readout)."""
    z = np.load(model, allow_pickle=False)
    a = basic.Agent(basic.Config(**json.loads(str(z["config"]))))
    assert np.array_equal(a.m, z["m"]) and np.array_equal(a.mask, z["mask"])
    last = max(int(k[4:]) for k in z.files if k.startswith("W_ep"))
    a.W = z[f"W_ep{last}"].astype(float)
    rec = json.loads(pathlib.Path(run).read_text())
    a.beta = float(rec["evals"][-1]["beta"])
    return a, a.beta, rec


def v4_frame(a, v):
    """basic.Agent.frame with plasticity off, plus (flow) a uniform action when no flow reaches the actions; returns
    (action, action share, latent share, spikes per head or None)."""
    m, roles = a.m, a.roles
    if a.backend in ("flow", "mpr"):
        pi = a.walk(v, warm=True)
        p = np.bincount(m, pi, a.K)
        pa = p[roles == "A"]
        if not pa.sum() > 0:
            a.silent += 1
            act = int(a.rng.integers(len(a.A)))
        else:
            act = int(a.rng.choice(len(a.A), p=core.policy(p, roles, a.beta)))
        tot = p.sum()
        return act, float(pa.sum() / tot), float(p[roles == "L"].sum() / tot), None
    pi, ob, act = a.frame(v)
    p = np.bincount(m, pi, a.K)
    return act, float(p[roles == "A"].sum()), float(p[roles == "L"].sum()), float(ob.n / a.heads)


def v4_episodes(a, n, seed):
    """n episodes; environment resets from their own stream (model seed, seed, ABL_SEED, 1), the agent's draws from
    (model seed, seed, ABL_SEED), so every condition faces the same initial states."""
    s = (a.cfg.seed, seed, ABL_SEED)
    a.rng, erng = np.random.default_rng(s), np.random.default_rng((*s, 1))
    env, ret, mon = gym.make(a.task.env_id), [], []
    a.silent = 0
    frames = 0
    for _ in range(n):
        o, _ = env.reset(seed=int(erng.integers(1 << 30)))
        a.qstate = a.lstate = None
        R, done = 0.0, False
        while not done:
            act, pa, pl, spk = v4_frame(a, a.stim(o))
            mon.append((pa, pl, np.nan if spk is None else spk))
            o, r, term, trunc, _ = env.step(act)
            R, done, frames = R + r, term or trunc, frames + 1
        ret.append(float(R))
    M = np.array(mon, float)
    pa, pl = M[:, :2].mean(0)
    spk = M[:, 2].mean()  # NaN on flow (no spikes)
    return ret, dict(silent_frac=a.silent / max(frames, 1), action_share=float(pa), latent_share=float(pl),
                     spikes_per_head=None if np.isnan(spk) else float(spk), frames=frames)


def v4_held_policy(a, seed, reps=HELD_REPS):
    """P(action) on each held-out stimulus: flow exact (Eq. 1d / 1e), spiking the action frequencies of reps frames
    (cold-started per frame), drawn from (model seed, seed, ABL_SEED, 2)."""
    a.rng = np.random.default_rng((a.cfg.seed, seed, ABL_SEED, 2))
    nA, out = len(a.A), []
    for v in a.held:
        if a.backend in ("flow", "mpr"):
            p = np.bincount(a.m, a.walk(v), a.K)
            pa = p[a.roles == "A"]
            out.append(core.policy(p, a.roles, a.beta) if pa.sum() > 0 else np.full(nA, 1 / nA))
            continue
        cnt = np.zeros(nA)
        for _ in range(reps):
            a.qstate = a.lstate = None
            cnt[v4_frame(a, v)[0]] += 1
        out.append(cnt / reps)
    return np.array(out)


def run_v4(model, run, n, seed):
    base, beta, rec = v4_agent(model, run)
    flow = base.backend in ("flow", "mpr")
    W0 = base.W.copy()
    chk = copy.deepcopy(base)
    fr = basic.frozen_returns(chk)
    out = dict(family="v4", task=base.task.env_id.split("-")[0], backend=base.backend, engine=base.cfg.engine,
               lam=base.cfg.lam, beta=beta, seed=base.cfg.seed, frozen_check=dict(
                   returns=fr, recorded=rec.get("frozen_returns"), reproduced=fr == rec.get("frozen_returns")),
               blocks=basic.block_shares(W0, base.m, base.roles), conditions={})
    P0 = None
    for cond in CONDITIONS["v4"]:
        a = copy.deepcopy(base)
        a.W = v4_weights(W0, a.m, a.roles, cond, flow)
        t0 = time.time()
        ret, mon = v4_episodes(a, n, seed)
        P = v4_held_policy(a, seed)
        if cond == "intact":
            P0 = P
            Pn = v4_held_policy(copy.deepcopy(a), seed + 1)  # noise floor: a second intact stream
            mon["held_shift_floor"] = float(0.5 * np.abs(Pn - P0).sum(1).mean())
        mon["held_shift"] = float(0.5 * np.abs(P - P0).sum(1).mean())  # total variation, mean over held stimuli
        out["conditions"][cond] = dict(returns=ret, mean=float(np.mean(ret)), **mon, wall=time.time() - t0)
        print(f"{cond:17s} return {np.mean(ret):8.2f}  silent {mon['silent_frac']:.3f}  "
              f"p_A {mon['action_share']:.3f}  shift {mon['held_shift']:.3f}  {time.time() - t0:.0f}s", flush=True)
    return out


# ---- family ck ---------------------------------------------------------------------------------------------------

def ck_kicks(hr, cond, R0=None):
    """The condition's kicks b ([post, pre], nR x nR): the lesions with the intact d, rec_untrained R0's own (module
    doc)."""
    if cond == "intact":
        return None
    if cond == "rec_untrained":
        b0 = hr.chi * R0 / np.maximum(R0.sum(0), 1e-12)
        return lambda: b0
    keep = {"rec_off": np.zeros_like(hr.R, bool), "cross_lesion": hr.m[:, None] == hr.m[None, :],
            "within_lesion": hr.m[:, None] != hr.m[None, :]}[cond]
    b = hr.chi * hr.R * keep / np.maximum(hr.R.sum(0), 1e-12)
    return lambda: b


def run_ck(model, n, seed):
    z = np.load(model, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    cols = "R" in z.files
    out = dict(family="ck", task=cfg["task"], seed=cfg["seed"], readout=cfg["readout"], rec=cfg["rec"],
               rec_seal=cfg["rec_seal"], conditions={})
    for cond in CONDITIONS["ck"] if cols else ("intact",):
        r = tdstdp.load_ck_model(model, rng_seed=(seed, ABL_SEED))
        b = ck_kicks(r.hr, cond, z["R_ep0"] if cols else None)
        if b is not None:
            r.hr.b = b  # the instance attribute shadows HybridRec.b, which bt() calls
        H = len(r.returns_heads)
        k = math.ceil(n / H)
        t0 = time.time()
        while min(len(x) for x in r.returns_heads) < k:
            r.tick()
        ret = [x for h in range(H) for x in r.returns_heads[h][:k]]
        dec = r.rdec.sum()
        mon = dict(actor_spikes_per_decision=float(r.espk[0] / max(r.espk[1], 1)),
                   actor_spikes_to_decision=float(r.ebnd[0] / max(r.ebnd[1], 1)) if r.race else None,
                   deadline_frac=float(r.rdec[1] / dec) if r.race and dec else None,
                   tie_frac=float(r.rdec[2] / dec) if r.race and dec else None)
        out["conditions"][cond] = dict(returns=ret, mean=float(np.mean(ret)), **mon, wall=time.time() - t0)
        print(f"{cond:14s} return {np.mean(ret):8.2f}  spikes/decision {mon['actor_spikes_per_decision']:.1f}  "
              f"deadline {mon['deadline_frac']}  {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=tuple(CONDITIONS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", help="v4: the training run's result JSON (final beta, frozen record); default: the model "
                    "path with models/ -> runs/ and .npz -> .json")
    ap.add_argument("--episodes", type=int, default=None, help="per condition (default EPISODES)")
    ap.add_argument("--seed", type=int, default=0, help="evaluation stream, with ABL_SEED")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.family == "v4":
        run = a.run or str(pathlib.Path(a.model.replace("/models/", "/runs/")).with_suffix(".json"))
        res = run_v4(a.model, run, a.episodes or EPISODES["v4"], a.seed)
    else:
        task = json.loads(str(np.load(a.model, allow_pickle=False)["config"]))["task"]
        res = run_ck(a.model, a.episodes or EPISODES[task], a.seed)
    res |= dict(model=a.model, model_sha256=sha256(a.model), eval_seed=a.seed, abl_seed=ABL_SEED,
                episodes=a.episodes or EPISODES["v4" if a.family == "v4" else res["task"]], wall=time.time() - t0)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)


if __name__ == "__main__":
    main()
