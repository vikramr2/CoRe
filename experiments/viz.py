"""Replay recorder: one frozen episode of a saved model, spike by spike, with one sampled cause per spike (the walk).

  python -m experiments.viz record --model PATH [--run PATH] [--tries N] --out FILE.json
  python -m experiments.viz all --out DIR [--workers N]     # every model under models/emergent-v4 and models/race-*

The episode is the best of `tries` frozen episodes (plasticity off) on environment seeds 1000 + k: the first CartPole
episode to reach 500, otherwise the highest return. It is the episode actually played: the spikes are captured by
wrapping module functions for the duration of the recording (sp.run_heads, sp.diffbound, lif.run_lif, lif._warmup,
qif.draw_events, qif.run_qif, td.draw, td.simulate_nlif). The wrappers pass every argument and return value through, so
no random draw moves (tests/test_viz.py checks that recorded and unrecorded episodes match). Causes are sampled after
the episode from their own stream (CAUSE_SEED); flow walkers from WALK_SEED.

Cause of a spike of i at step t (head 0 of a v4 model, copy 0 of a race model): c >= 0 a parent neuron (a walk step),
-1 stimulus, -2 background. A spiking model's walk step is a sample of the direct cause, drawn from the parent posterior
of a linear cluster (Hawkes) process: P(cause) = phi / Z, Z = lam0 dt + sum of the phi.
  branching  exact: parents among the spikes at t - 1, phi = T_ij (the linear escape step); external spikes (the
             stimulus) force their neuron and get no parent; no background (p(0) = 0).
  lif (v4)   phi_j = G_i tau_eff b_ij k(t - t_j), t_j >= t_last,i + nref (a reset erases older kicks), lags 1..10 steps;
             stimulus as a constant-rate source G_i tau_eff a_ext ext_rate v_i dt; background lam0 dt. k: the measured
             response kernel normalised to sum 1, so a kick's expected hops are the counter c (Eq. 11b).
  qif (v4)   phi_j = b_ij k_rec(t - t_j), lags 1..5 bins; each external pulse to i at its actual bin, k_ext(t - t_e),
             lags 0..5; background lam0 bin. Both kernels measured (absolute, per unit kick / per pulse).
  nlif       (race models) as lif with the input current s_in W_ij / 13 and the recurrent kicks chi R_ij / d_j, lags
             1..6 ms, t_j >= t_last + nref; signed inputs: the parents are the positive terms E, the background share is
             lam0 dt / (lam0 dt + max(E - I, 0)) with I the summed negative terms. Input parents are shown as stimulus;
             only R -> R parents are walk steps. Input neurons are stored as per-window spike counts (inp).
  flow       no spikes: M_WALKERS walkers of Eq. 1 per frame (hop 0 from v; then restart from v with probability alpha,
             else a step along T), from WALK_SEED; the action is drawn from the exact pi.
The kernels and background rates are pre-set constants, measured once on the models named in their comments (review of
2026-10-31, preliminary)."""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import pathlib
import time

import gymnasium as gym
import numpy as np

from experiments import ablate, m2, tdstdp
from mapstdp import core, lif, qif
from mapstdp import spiking as sp
from mapstdp import tdstdp as td

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIZ_SEED, CAUSE_SEED, WALK_SEED, ENV_SEED0 = 8101, 8102, 8103, 1000
M_WALKERS, MAX_EDGES = 32, 800
# response kernels (lagged excess spikes per presynaptic spike, regressed on the kick; preliminary, one model each)
K_LIF = np.array([1.0, .254, .152, .089, .056, .041, .025, .016, .011, .005])  # lags 1..10 x 0.5 ms, shape;
#                                                       models/emergent-v4/CartPole_lif_on/s2017.npz
K_QIF_REC = np.array([.0078, .0106, .0117, .0073, .0063])  # lags 1..5 bins, per unit kick b = gain W / d;
#                                                       models/emergent-v4/CartPole_qif_on/s2004.npz
K_QIF_EXT = np.array([.008, .034, .062, .197, .107, .028])  # lags 0..5 bins, excess spikes per external pulse; s2004
K_NLIF = np.array([.888, .131, .088, .058, .054, .058])  # lags 1..6 ms, shape; models/race-cartpole/
#                                                       CartPole_cols_on_race_rp/s7010.npz
LAM0 = {"lif": 2.53, "qif": 0.65, "nlif": 2.54}  # background-only rate (Hz): Siegert at (mu_bg, sigma_V) for lif;
#                                         an isolated neuron with noise only (s2004) for qif; markov_rate(mu_bg) nlif
WINDOW = {"flow": 1, "branching": 1, "lif": len(K_LIF), "qif": len(K_QIF_REC), "nlif": len(K_NLIF)}  # steps
ACTIONS = {"CartPole": ["push left", "push right"], "Acrobot": ["torque −1", "torque 0", "torque +1"],
           "LunarLander": ["noop", "left engine", "main engine", "right engine"]}
ENGINE = {"numpy": "branching", "lif": "lif", "qif": "qif"}
ENG_LABEL = {"flow": "flow", "branching": "branching", "lif": "LIF", "qif": "QIF", "nlif": "noisy LIF"}
V4_ARM = {"on": "map term on", "off": "map term off", "sealed": "sealed modules"}
RACE_ARM = {"base_ck": "CK softmax", "base_nl": "normalised softmax", "base_race": "no columns · race",
            "cols_lam0": "implicit columns · race", "cols_on": "learned columns (map term) · race",
            "cols_sealed": "sealed columns · race"}


@contextlib.contextmanager
def patched(pairs):
    """Replace module (or instance) attributes by wrappers make(orig) for the duration; restore them afterwards."""
    saved = []
    try:
        for obj, name, make in pairs:
            orig = getattr(obj, name)
            saved.append((obj, name, orig, name in vars(obj)))
            setattr(obj, name, make(orig))
        yield
    finally:
        for obj, name, orig, own in reversed(saved):
            if own:
                setattr(obj, name, orig)
            else:
                delattr(obj, name)


def tap(store, key, pick):
    """Wrapper factory: call through, store pick(result, args, kwargs) under store[key], return the result."""
    def make(orig):
        def f(*a, **k):
            out = orig(*a, **k)
            store[key] = pick(out, a, k)
            return out
        return f
    return make


def bound_step(S, groups, theta, rng=None, burn=0, cap=0, heads=1):
    """The step at which sp.diffbound stops (same arithmetic, no draws): burn + first hit, k - 1 at the deadline, -1 if
    no group spikes."""
    if cap > 0:
        cnt = np.stack([np.minimum(S[burn:, g].reshape(len(S) - burn, heads, -1).sum(2), cap).sum(1)
                        for g in groups], 1)
    else:
        cnt = np.stack([S[burn:, g].sum(1) for g in groups], 1)
    n = np.cumsum(cnt, 0)
    if not len(n) or not n[-1].any():
        return -1
    top2 = np.sort(n, 1)[:, -2:]
    hit = np.flatnonzero(top2[:, 1] - top2[:, 0] >= theta)
    return int(burn + hit[0]) if len(hit) else len(S) - 1


def rnd(x, d=4):
    return [round(float(v), d) for v in np.ravel(x)]


def draw_cause(rng, phi, null, tags):
    """Index into tags of the sampled cause (weights phi, then the null weight); tags[-1] belongs to the null."""
    w = np.r_[phi, null]
    tot = w.sum()
    if not tot > 0:
        return tags[-1]
    return tags[int(np.searchsorted(np.cumsum(w), rng.random() * tot, side="right").clip(0, len(w) - 1))]


# ---- v4 (emergent-v4 models, m2.Agent) ---------------------------------------------------------------------------

def v4_paths(model, run=None):
    model = pathlib.Path(model)
    run = run or ROOT / "reports" / "emergent-v4" / "runs" / model.parent.name / (model.stem + ".json")
    return model, pathlib.Path(run)


def play_v4(a, k, max_frames=None, store=None):
    """One frozen episode (ablate.v4_frame) on env seed ENV_SEED0 + k, agent stream (seed, VIZ_SEED, k). store: a dict
    the recording wrappers fill per frame; returns (return, raw frames)."""
    a.rng = np.random.default_rng((a.cfg.seed, VIZ_SEED, k))
    a.qstate = a.lstate = None
    env = gym.make(a.task.env_id)
    o, _ = env.reset(seed=ENV_SEED0 + k)
    R, done, raw = 0.0, False, []
    while not done and (max_frames is None or len(raw) < max_frames):
        if store is not None:
            store.clear()
        v = a.stim(o)
        act = ablate.v4_frame(a, v)[0]
        raw.append(dict(obs=np.asarray(o, float).copy(), a=act, v=v, **(store or {})))
        o, r, term, trunc, _ = env.step(act)
        R, done = R + float(r), term or trunc
    return R, raw


def v4_wrappers(store, N):
    """The capture of one v4 frame, head 0 only (columns 0 .. N - 1 of the k x (H N) rasters)."""
    def ev0(o):  # head 0's external pulses: (bin substep, neuron)
        ptr, s, i = o[:3]
        return s[ptr[0]:ptr[1]].copy(), i[ptr[0]:ptr[1]].copy()

    def raster(o):
        return (o[0] if isinstance(o, tuple) else o)[:, :N].copy()
    return [(sp, "run_heads", tap(store, "rh", lambda o, a, k: (o[0][:, :N].copy(), o[2][:, :N].copy()))),
            (sp, "diffbound", tap(store, "dec", lambda o, a, k: bound_step(*a, **k))),
            (lif, "run_lif", tap(store, "lif", lambda o, a, k: (o[0][:, :N].copy(), o[1]["G"].copy()))),
            (lif, "_warmup", tap(store, "warm_tail", lambda o, a, k: a[2]["tail"][:, 0].copy())),
            (qif, "draw_events", tap(store, "ev", lambda o, a, k: ev0(o))),
            (qif, "run_qif", tap(store, "qS", lambda o, a, k: raster(o)))]


def attribute_v4(a, eng, raw):
    """Frames of the page from the raw frames of one episode (module doc): s = [t, n, c] of head 0."""
    N, rng, wrng = len(a.W), np.random.default_rng(CAUSE_SEED), np.random.default_rng(WALK_SEED)
    T = a.W / a.W.sum(0)
    frames, cnt = [], np.zeros(3)  # walk, stim, none
    hist, last, ext_hist, t0 = None, np.full(N, -10 ** 9), [], 0
    if eng == "lif":
        p = a.lp
        b, nref = lif.kicks(a.W, p), p.nref
        kh = K_LIF / K_LIF.sum()
    if eng == "qif":
        p = a.qp
        b = p.gain * a.W / np.maximum(a.W.sum(0), 1e-12)
    for fr in raw:
        s, P, dec = [], None, -1
        if eng == "flow":
            pi = core.flow(a.W, fr["v"], m2.ALPHA, m2.N_HOPS)
            P = rnd(np.bincount(a.m, pi, a.K))
            cv = np.cumsum(fr["v"]) / fr["v"].sum()
            pos = np.searchsorted(cv, wrng.random(M_WALKERS), side="right").clip(0, N - 1)
            s += [[0, int(x), -1] for x in pos]
            cnt[1] += M_WALKERS
            CT = np.cumsum(T, 0)
            for hop in range(1, m2.N_HOPS):
                tel = wrng.random(M_WALKERS) < m2.ALPHA
                u = wrng.random(M_WALKERS)
                new = np.where(tel, np.searchsorted(cv, u, side="right"),
                               [np.searchsorted(CT[:, j], u_ * CT[-1, j], side="right") for j, u_ in zip(pos, u)])
                new = new.clip(0, N - 1)
                s += [[hop, int(n), -1 if tl else int(j)] for n, j, tl in zip(new, pos, tel)]
                cnt[0], cnt[1] = cnt[0] + (~tel).sum(), cnt[1] + tel.sum()
                pos = new
            frames.append(dict(obs=rnd(fr["obs"]), a=fr["a"], dec=-1, P=P, s=s, pose=None))
            continue
        dec = fr.get("dec", -1)
        if eng == "branching":
            S0, X0 = fr["rh"]
            for t, i in zip(*np.nonzero(S0)):
                if X0[t, i] or t == 0:
                    c = -1 if X0[t, i] else -2
                else:
                    pre = np.flatnonzero(S0[t - 1] & (T[i] > 0))
                    c = draw_cause(rng, T[i, pre], 0.0, [*map(int, pre), -2])
                s.append([int(t), int(i), c])
        elif eng == "lif":
            S0, G = fr["lif"]
            k = len(S0)
            if fr.get("warm_tail") is not None:  # first frame: the discarded warm-up's last steps are the history
                hist = fr["warm_tail"]
                last = np.full(N, -10 ** 9)
                for tt, i in zip(*np.nonzero(hist)):
                    last[i] = tt - len(hist) + t0
            full = np.concatenate([hist[-len(kh):], S0]) if hist is not None else S0
            off = len(full) - k
            stim = G * p.tau_eff * p.a_ext * p.ext_rate * fr["v"] * p.dt
            null = LAM0["lif"] * p.dt
            for t, i in zip(*np.nonzero(S0)):
                ta = t0 + t
                lo = max(int(last[i]) + nref - t0 + off, t + off - len(kh), 0)
                tj, j = np.nonzero(full[lo:t + off])
                tj = tj + lo
                phi = G[i] * p.tau_eff * b[i, j] * kh[t + off - tj - 1]
                ok = phi > 0
                c = draw_cause(rng, np.r_[phi[ok], stim[i]], null, [*map(int, j[ok]), -1, -2])
                s.append([int(t), int(i), c])
                last[i] = ta
            hist, t0 = full[-len(kh):], t0 + k
        else:  # qif
            S0 = fr["qS"]
            k = len(S0)
            evb, evi = fr["ev"][0] // p.nsub + t0, fr["ev"][1]
            ext_hist = [(tb, i) for tb, i in ext_hist if tb >= t0 - len(K_QIF_EXT)] + list(zip(evb, evi))
            E = {}
            for tb, i in ext_hist:
                E.setdefault(int(i), []).append(int(tb))
            L = len(K_QIF_REC)
            full = np.concatenate([hist, S0]) if hist is not None else np.concatenate([np.zeros((L, N), bool), S0])
            for t, i in zip(*np.nonzero(S0)):
                tj, j = np.nonzero(full[t:t + L])  # rows t .. t + L - 1 of full = bins t - L .. t - 1
                lag = L - tj
                phi = b[i, j] * K_QIF_REC[lag - 1]
                ok = phi > 0
                te = np.array([t0 + t - x for x in E.get(int(i), []) if 0 <= t0 + t - x < len(K_QIF_EXT)], int)
                c = draw_cause(rng, np.r_[phi[ok], K_QIF_EXT[te]], LAM0["qif"] * p.bin * 1e-3,
                               [*map(int, j[ok]), *[-1] * len(te), -2])
                s.append([int(t), int(i), c])
            hist, t0 = full[-L:], t0 + k
        n0 = np.bincount(a.m, S0.sum(0), a.K).astype(float)
        P = rnd(n0 / max(n0.sum(), 1))
        for e in s:
            cnt[0 if e[2] >= 0 else 1 if e[2] == -1 else 2] += 1
        frames.append(dict(obs=rnd(fr["obs"]), a=fr["a"], dec=int(dec), P=P, s=s, pose=None))
    return frames, cnt


def record_v4(model, run=None, tries=10, max_frames=None):
    model, run = v4_paths(model, run)
    a0, beta, rec = ablate.v4_agent(model, run)
    eng = "flow" if a0.backend == "flow" else ENGINE[a0.cfg.engine]
    task = a0.task.env_id.split("-")[0]
    best = None
    store = {}
    for k in range(tries):
        a = a0  # the frozen agent carries its estimators across episodes as in ablate.v4_episodes
        with patched(v4_wrappers(store, len(a.W))):
            R, raw = play_v4(a, k, max_frames, store)
        if best is None or R > best[0]:
            best = (R, raw, k)
        print(f"{model.parent.name}/{model.name} try {k}: return {R:.0f} ({len(raw)} frames)", flush=True)
        if task == "CartPole" and R >= 500:
            break
    R, raw, k = best
    frames, cnt = attribute_v4(a0, eng, raw)
    a = a0
    names = {"C": "controller", "L": "latent", "A": "action"}
    modules, nl = [], 0
    for q, role in enumerate(a.roles):
        d = dict(name=names[role], role={"C": "ctrl", "L": "lat", "A": "act"}[role], action=None, grid=None,
                 grid_labels=None)
        if role == "L":
            nl += 1
            d["name"] = f"latent {nl}"
        if role == "A":
            d["action"] = int(np.flatnonzero(a.A == q)[0])
            d["name"] = f"action: {ACTIONS[task][d['action']]}"
        if role == "C" and (a.m == q).sum() == 36 and task == "CartPole":
            d["grid"], d["grid_labels"] = [6, 6], ["θ", "θ̇"]
        modules.append(d)
    T = a.W / a.W.sum(0)
    cross = (a.m[:, None] != a.m[None, :]) & (a.W > 0)
    edges = top_edges([(i, j, T[i, j]) for i, j in zip(*np.nonzero(cross))], MAX_EDGES)
    returns = rec.get("returns", [])
    train = dict(frozen_mean=float(np.mean(rec["frozen_returns"])) if rec.get("frozen_returns") else None,
                 last100=float(np.mean(returns[-100:])) if returns else None,
                 mean=float(np.mean(returns)) if returns else None, episodes=len(returns), beta=beta)
    k_ = {"flow": m2.N_HOPS, "branching": m2.K_SPK, "lif": a.lp.k if a.lp else 0, "qif": a.qp.k if a.qp else 0}[eng]
    dt = {"flow": 1.0, "branching": 1.0, "lif": a.lp.dt * 1e3 if a.lp else 0, "qif": a.qp.bin if a.qp else 0}[eng]
    arm = model.parent.name.split("_", 2)[2]
    return dict(id=f"v4-{task}-{eng}-{arm}".lower(), task=task, family="v4", engine=eng, arm=arm,
                label=f"v4 · {ENG_LABEL[eng]} · {V4_ARM.get(arm, arm)}", seed=int(a.cfg.seed),
                model=str(model.relative_to(ROOT)) if model.is_absolute() else str(model), dt_ms=dt, k=int(k_),
                burn=0 if eng == "flow" else int(a.burn), modules=modules, m=a.m.tolist(), edges=edges,
                actions=ACTIONS[task], note=NOTES[eng] + ("" if eng == "flow" else HEAD_NOTE.format(H=a.heads)),
                shares=shares(cnt), **{"return": R}, train=train, env={}, frames=frames, env_seed=ENV_SEED0 + k)


NOTES = {
    "flow": "The flow engine has no spikes: it computes the walk's distribution pi exactly (Eq. 1). Dots are 32 "
            "walkers sampled from that chain (restart at the stimulus with probability 0.2, else a step along T); a "
            "walk step is one walker's hop, an exact sample. The action is drawn from the exact pi, not from the "
            "walkers.",
    "branching": "One time step is one hop: each non-stimulus spike gets one parent among the previous step's spikes, "
                 "drawn in proportion to T_ij, the exact posterior of the branching step. Stimulus spikes "
                 "(teleportation) force their neuron and have no parent.",
    "lif": "A walk step is a sampled direct cause: a presynaptic spike in the last 5 ms after the neuron's last reset,"
           " weighted by kick x gain x the measured response kernel, against a constant stimulus source and a 2.5 Hz "
           "background. Linear-response approximation at the scale of the counter c (Eq. 11b), about 1.5x the effect "
           "of removing the kick.",
    "qif": "A walk step is a sampled direct cause: a presynaptic spike 1-5 ms earlier, weighted by kick x the measured"
           " response kernel (peak at 3 ms), against stimulus pulses at their actual times and a 0.65 Hz background. "
           "Approximation (kernels measured on one model, indirect correlation included); the inhibitory pools are not"
           " shown and are not causes.",
}
HEAD_NOTE = " Head 0 of {H} parallel heads is shown; the decision pools all {H} heads."
RACE_NOTE = ("A walk step is a recurrent hop between actor posts (column to column or within one): a sampled direct "
             "cause in the last 6 ms after the post's last reset, weighted by kick x gain x the measured response "
             "kernel. Input spikes, and post spikes caused by an input, are stimulus; 2.5 Hz background; net "
             "inhibitory input lowers the explained share. Linear-response approximation at the scale of the counter "
             "c. Shares count post spikes only.")
RACE_NOTE_BASE = ("No recurrence, so there are no walk steps: every post spike is caused by an input (stimulus) or by "
                  "the 2.5 Hz background, sampled from the inputs of the last 6 ms after the post's last reset, "
                  "weighted by weight x gain x the measured response kernel; net inhibitory input lowers the explained"
                  " share. Linear-response approximation at the scale of the counter c. Shares count post spikes only.")


def shares(cnt):
    tot = max(cnt.sum(), 1)
    return {k: round(float(c / tot), 4) for k, c in zip(("walk", "stim", "none"), cnt)}


def top_edges(cands, n):
    """[[post, pre, w]] of the n strongest candidates (post, pre, w), w normalised to max 1."""
    cands = sorted(cands, key=lambda e: -e[2])[:n]
    if not cands:
        return []
    mx = cands[0][2]
    return [[int(i), int(j), round(float(w / mx), 3)] for i, j, w in cands]


# ---- race models (TD-STDP backbone, experiments.tdstdp.CKRunner) ----------------------------------------------------

class EnvTap:
    """Copy 0's environment: resets on the seeds ENV_SEED0, ENV_SEED0 + 1, ... whatever seed the runner draws (the
    runner's draw still happens, so no stream moves); marks steps."""

    def __init__(self, env):
        self.env, self.n_reset, self.stepped = env, 0, False

    def reset(self, seed=None, **kw):
        out = self.env.reset(seed=ENV_SEED0 + self.n_reset, **kw)
        self.n_reset += 1
        return out

    def step(self, a):
        self.stepped = True
        return self.env.step(a)

    def __getattr__(self, name):
        return getattr(self.env, name)


def lander(env):
    u = env.unwrapped
    leg = [[*rnd(lg.position, 3), round(float(lg.angle), 3)] for lg in u.legs]
    return dict(x=round(float(u.lander.position[0]), 3), y=round(float(u.lander.position[1]), 3),
                ang=round(float(u.lander.angle), 3), legs=leg, contact=[bool(lg.ground_contact) for lg in u.legs])


def terrain(env):
    u = env.unwrapped
    pts = [p[0] for p in u.sky_polys] + [u.sky_polys[-1][1]]
    return dict(terrain=[rnd(q, 3) for q in pts], helipad=rnd([u.helipad_x1, u.helipad_x2, u.helipad_y], 3),
                W=20.0, H=13.333)


def race_runner(model):
    r = tdstdp.load_ck_model(str(model), rng_seed=VIZ_SEED)
    r.envs[0] = EnvTap(r.envs[0])
    r.obs[0] = r.envs[0].reset(seed=0)[0]  # copy 0's first episode on ENV_SEED0 (no runner draw moves)
    return r


def play_race(r, tries, max_frames=None, store=None, stop500=False):
    """Tick the runner until copy 0 has finished `tries` episodes (CartPole with stop500: or one reaches 500), or for
    max_frames live frames of its first episode. Returns [(return, ticks, env)] per episode of copy 0: ticks are dicts
    per 20-ms window from the end of the previous episode on (raw capture if store is given; live = copy 0 stepped
    its environment in that window), env the LunarLander terrain of the episode."""
    env0, eps, ticks = r.envs[0], [], []
    ll = r.cfg.task == "LunarLander"
    env_info, resets = (terrain(env0) if ll else {}), env0.n_reset
    while len(eps) < tries:
        if store is not None:
            store.clear()
        n_ret = len(r.returns_heads[0])
        pre = dict(obs=np.asarray(r.obs[0], float).copy(), code=int(r.code[0]), any_reset=bool((r.code == 2).any()),
                   timer=float(r.timer[0]), pose=lander(env0) if ll else None)
        env0.stepped = False
        r.tick()
        ticks.append(dict(pre, live=env0.stepped, a=int(r.action[0]), **(store or {})))
        if ll and env0.n_reset > resets:  # copy 0 reset in this window: its new episode's terrain
            env_info, resets = terrain(env0), env0.n_reset
        done = len(r.returns_heads[0]) > n_ret
        cut = max_frames is not None and sum(t["live"] for t in ticks) >= max_frames
        if done or cut:
            R = float(r.returns_heads[0][-1]) if done else float(r.c_ret[0])
            eps.append((R, ticks, env_info))
            ticks = []
            if cut or (stop500 and R >= 500):
                break
    return eps


def race_wrappers(r, store):
    def bnd(orig):
        def f(X):
            out = orig(X)
            if r.was[0] and r.bnd[0] >= 0:
                store["bnd"] = int(r.bnd[0])
            store["was"] = bool(r.was[0])
            return out
        return f

    def acc(key, pick):
        def make(orig):
            def f(*a, **k):
                out = orig(*a, **k)
                store.setdefault(key, []).append(pick(out))
                return out
            return f
        return make
    return [(td, "draw", acc("x", lambda o: o[0][:, 0].astype(bool))),
            (td, "simulate_nlif", acc("X", lambda o: (o[0][:, 0].copy(), o[1][0].copy(), o[2][0].copy()))),
            (r, "race_block", bnd)]


def attribute_race(r, ticks):
    """Page frames of copy 0 (module doc, nlif): display index = input neurons, then posts."""
    lay, p, cfg = r.lay, r.lp, r.cfg
    n_in, n_post = lay.n_in, lay.n_post
    rng = np.random.default_rng(CAUSE_SEED)
    kh = K_NLIF / K_NLIF.sum()
    L = len(kh)
    Win = cfg.s_in / td.THETA_L * r.W  # [post, input], threshold units
    ridx = r.hr.ridx if r.hr is not None else np.zeros(0, int)
    pos = {int(q): z for z, q in enumerate(ridx)}
    B = r.hr.b() if r.hr is not None else None
    null = LAM0["nlif"] * p.dt
    xs = [np.concatenate(t["x"]) for t in ticks]
    Xs = [np.concatenate([q[0] for q in t["X"]]) for t in ticks]
    x_all, X_all = np.concatenate(xs), np.concatenate(Xs)
    last, block_in = np.full(n_post, -10 ** 9), set()
    frames, cnt, t0 = [], np.zeros(3), 0
    for tk, x, X in zip(ticks, xs, Xs):
        k = len(X)
        Lb = k // len(tk["X"])
        G = np.concatenate([np.repeat(lif.gain_markov(p.mu_bg + xs_ / Lb / (1 - p.a),
                                                      np.sqrt(np.maximum(xq / Lb - (xs_ / Lb) ** 2, 0)), p)[None],
                                      Lb, 0) for _, xs_, xq in tk["X"]])
        if tk["code"] == 2:  # copy 0 reset: membranes, in-flight spikes and previous inputs cleared
            last[:] = t0 - p.nref
        if tk["any_reset"]:
            block_in.add(t0 - 1)  # every copy's previous input spikes are cleared
        s, nx = [], x.sum(0)
        inp = [[int(j), int(nx[j])] for j in np.flatnonzero(nx)]  # input neurons: counts only (always stimulus)
        for t, i in zip(*np.nonzero(X)):
            ta = t0 + t
            lo = max(int(last[i]) + p.nref, ta - L, 0)
            tq, jq = np.nonzero(x_all[lo:ta])
            tq = tq + lo
            keep = np.array([q not in block_in for q in tq], bool) if len(tq) else np.zeros(0, bool)
            tq, jq = tq[keep], jq[keep]
            phi_in = G[t, i] * p.tau_eff * Win[i, jq] * kh[ta - tq - 1]
            phi_r, jr = np.zeros(0), np.zeros(0, int)
            if int(i) in pos:
                tr, qr = np.nonzero(X_all[lo:ta][:, ridx])
                tr = tr + lo
                phi_r, jr = G[t, i] * p.tau_eff * B[pos[int(i)], qr] * kh[ta - tr - 1], ridx[qr]
            phi = np.r_[phi_in, phi_r]
            tags = [-1] * len(phi_in) + [n_in + int(q) for q in jr]
            M = max(phi[phi > 0].sum() + phi[phi < 0].sum(), 0.0)  # max(E - I, 0)
            ok = phi > 0
            if M > 0:
                pn = null / (null + M)
                c = -2 if rng.random() < pn else draw_cause(rng, phi[ok], 0.0, [*np.array(tags)[ok].tolist(), -2])
            else:
                c = -2
            s.append([int(t), n_in + int(i), int(c)])
            cnt[0 if c >= 0 else 1 if c == -1 else 2] += 1
            last[i] = ta
        t0 += k
        if tk["live"]:
            dec = tk.get("bnd", -1)
            nb = tk["timer"] + tdstdp.MS >= cfg.freeze
            if dec < 0 and nb and (not r.race or tk.get("was", True)):
                dec = k - 1
            frames.append(dict(obs=rnd(tk["obs"]), a=tk["a"], dec=int(dec), P=None, s=sorted(s), inp=inp,
                               pose=tk["pose"]))
        else:
            frames.append(None)  # not an env step (warm-up or post-episode window): history only
    return frames, cnt


def input_grid(cfg, n_in):
    if cfg.f_cross:  # CartPole: index digits (dim 0 fastest), 3 x 3 x 3 x 3 -> columns (dim 2, 3), rows (dim 0, 1)
        k = cfg.f_order + 1
        return [k * k, k * k], ["θ · θ̇ order", "x · ẋ order"]
    per = n_in // cfg.in_rep  # [cos]+ per dim, [-cos]+ per dim, bias; repeated in_rep times
    return [cfg.in_rep, per], ["repeat", "feature (cos+ per dim, cos− per dim, bias)"]


def record_race(model, tries=10, max_frames=None):
    model = pathlib.Path(model)
    r = race_runner(model)
    cfg, lay = r.cfg, r.lay
    task = cfg.task
    store = {}
    with patched(race_wrappers(r, store)):
        eps = play_race(r, tries, max_frames, store, stop500=task == "CartPole")
    k = int(np.argmax([e[0] for e in eps]))
    R, ticks, envd = eps[k]
    for z, e in enumerate(eps):
        print(f"{model.parent.name}/{model.name} try {z}: return {e[0]:.0f} ({sum(t['live'] for t in e[1])} frames)",
              flush=True)
    frames, cnt = attribute_race(r, ticks)
    frames = [f for f in frames if f is not None]
    nz = [c for f in frames for _, c in f["inp"]]
    inp_max = float(np.percentile(nz, 99)) if nz else 1.0
    n_in, K = lay.n_in, r.K
    modules = [dict(name="inputs", role="stim", action=None, grid=input_grid(cfg, n_in)[0],
                    grid_labels=input_grid(cfg, n_in)[1])]
    m = [0] * n_in
    blocks = [("critic E", "critic", None, cfg.n_ce), ("critic I", "inh", None, cfg.n_ci)]
    for a in range(K):
        blocks += [(f"{ACTIONS[task][a]} E", "act", a, cfg.n_ae), (f"{ACTIONS[task][a]} I", "inh", a, cfg.n_ai)]
    for name, role, act, n in blocks:
        if n:
            m += [len(modules)] * n
            modules.append(dict(name=name, role=role, action=act, grid=None, grid_labels=None))
    edges = []
    if r.hr is not None:
        b = r.hr.b()
        cr = [(n_in + int(r.hr.ridx[i]), n_in + int(r.hr.ridx[j]), b[i, j])
              for i, j in zip(*np.nonzero((r.hr.m[:, None] != r.hr.m[None, :]) & (b > 0)))]
        edges = top_edges(cr, MAX_EDGES // 2)
    Win = cfg.s_in / td.THETA_L * r.W
    ci = [(n_in + i, j, Win[i, j]) for i, j in zip(*np.nonzero(Win > 0))]
    edges += top_edges(ci, MAX_EDGES - len(edges))
    run = ROOT / "reports" / model.parent.parent.name / "runs" / model.parent.name / (model.stem + ".json")
    train = {}
    if run.exists():
        rec = json.loads(run.read_text())
        ret = rec.get("returns", [])
        train = dict(last100=float(np.mean(ret[-100:])) if ret else None, mean=float(np.mean(ret)) if ret else None,
                     episodes=len(ret), T_f=rec.get("T_f"), T_s=rec.get("T_s"))
    arm = model.parent.name[len(task) + 1:]
    key = next((q for q in RACE_ARM if arm.startswith(q)), arm)
    note = (RACE_NOTE if r.hr is not None else RACE_NOTE_BASE) + \
        (f" Copy 0 of {cfg.heads} environment copies run in lockstep is shown." if cfg.heads > 1 else "")
    out = dict(id=f"race-{task}-nlif-{arm}".lower(), task=task, family="race", engine="nlif", arm=arm,
               label=f"TD-STDP · {RACE_ARM.get(key, arm)}", seed=int(cfg.seed),
               model=str(model.relative_to(ROOT)) if model.is_absolute() else str(model), dt_ms=r.lp.dt * 1e3,
               k=tdstdp.MS, burn=0, modules=modules, m=m, edges=edges, actions=ACTIONS[task], note=note,
               shares=shares(cnt), **{"return": R}, train=train, env=envd or {}, inp_max=inp_max, frames=frames,
               env_seed=ENV_SEED0 + k)
    return out


# ---- CLI ----------------------------------------------------------------------------------------------------------

def family(model):
    """'race' (a save_ck_model file: W) or 'v4' (W_ep<k> snapshots)."""
    return "race" if "W" in np.load(model).files else "v4"


def record(model, run=None, tries=10, max_frames=None):
    model = pathlib.Path(model).resolve()
    out = record_race(model, tries, max_frames) if family(model) == "race" else record_v4(model, run, tries, max_frames)
    return out


def dump(out, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    path.write_text(txt)
    return len(txt.encode())


def _one(args):
    model, out_dir, tries = args
    t = time.time()
    out = record(model, tries=tries)
    size = dump(out, pathlib.Path(out_dir) / f"{out['id']}.json")
    return dict(id=out["id"], file=f"{out['id']}.json", task=out["task"], family=out["family"], engine=out["engine"],
                arm=out["arm"], label=out["label"], seed=out["seed"], **{"return": out["return"]},
                frames=len(out["frames"]), bytes=size, shares=out["shares"], wall=round(time.time() - t, 1))


def all_models():
    return sorted(map(str, (ROOT / "models" / "emergent-v4").glob("*/*.npz"))) + \
        sorted(map(str, (ROOT / "models").glob("race-*/*/*.npz")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("record", "all"))
    ap.add_argument("--model")
    ap.add_argument("--run", help="v4: the training run's JSON (default reports/emergent-v4/runs/<group>/<seed>.json)")
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "record":
        out = record(a.model, a.run, a.tries)
        print(f"{out['id']}: return {out['return']}, {len(out['frames'])} frames, shares {out['shares']}, "
              f"{dump(out, a.out) / 1e6:.2f} MB -> {a.out}")
        return
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    jobs = [(mdl, a.out, a.tries) for mdl in all_models()]
    if a.workers > 1:
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            idx = pool.map(_one, jobs, chunksize=1)
    else:
        idx = [_one(j) for j in jobs]
    out = pathlib.Path(a.out)
    (out / "index.json").write_text(json.dumps(dict(models=[{k: e[k] for k in (
        "id", "file", "task", "family", "engine", "arm", "label", "seed", "return", "frames")} for e in idx]),
        ensure_ascii=False, indent=1))
    for e in idx:
        print(f"{e['id']:45s} return {e['return']:8.1f} frames {e['frames']:5d} {e['bytes'] / 1e6:6.2f} MB "
              f"shares {e['shares']} ({e['wall']:.0f}s)")


if __name__ == "__main__":
    main()
