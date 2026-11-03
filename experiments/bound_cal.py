"""Calibration of the diffbound readout (Eq. 1h) for a spiking engine: the table beta_eff(theta) that models.basic
inverts to set the bound from beta (basic.BOUND_CAL, BOUND_CAL_QIF, BOUND_CAL_LIF), by the method of iteration 010 §4.3.
  python -m experiments.bound_cal --engine lif --heads 16 MODEL.npz [MODEL.npz ...] [--frames 200] [--stimuli 20]
For each trained model (basic.save_model .npz; its last W snapshot, task from its config): on the first --stimuli
held-out stimuli, a fresh warm state per stimulus (qif: one warm-up frame; lif: the engine's in-frame warm-up), then
--frames consecutive warm frames on that stimulus, each read by sp.diffbound (cap 1 per module, head and step, burn-in
the engine's) at every theta of THETAS. beta_eff(theta) is the maximum-likelihood beta of Eq. 1e, P(a) ~ p_a^beta with p
the flow walk's action-module shares on the same W, over all decisions of the model; the table is the mean over
models. Also reported: the deadline fraction (frames with no decision before the frame ends). The frames draw from
default_rng(seed); nothing is written except the --out JSON (default: printed only)."""
import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from mapstdp import core, lif, qif
from mapstdp import spiking as sp
from models import basic

THETAS = (1, 2, 4, 8, 16, 32, 64)


def frames(agent, v, n, rng):
    """n consecutive warm frames of agent's engine on stimulus v from a fresh state (rasters k x H N)."""
    H = agent.heads
    if agent.engine == "lif":
        st = lif.init_state(len(agent.m), H, agent.lp)
        return [lif.run_lif(agent.W, v, H, rng, st, agent.lp, return_stats=False) for _ in range(n)]
    if agent.engine == "qif":
        st = qif.init_state(agent.eta_q, rng, agent.qp)
        run = lambda: qif.run_qif(agent.W, v, H, rng, agent.eta_q, st, agent.qp, agent.m, agent.roles)  # noqa: E731
        run()
        return [run() for _ in range(n)]
    return [sp.run_heads(agent.W, v, H, basic.K_SPK, basic.RATE, rng, basic.THETA, 1 - basic.ALPHA) for _ in range(n)]


def beta_ml(counts, P1):
    """ML beta of P ~ p^beta: counts (stimuli x |A|), P1 (stimuli x |A|) the beta = 1 shares."""
    lp = np.log(np.maximum(P1, 1e-300))

    def nll(lb):
        z = np.exp(lb) * lp
        return -(counts * (z - np.logaddexp.reduce(z, 1, keepdims=True))).sum()
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(200)), method="bounded").x))


def calibrate(path, engine, heads, n_frames, n_stim, seed):
    z = np.load(path)
    cfg = json.loads(str(z["config"]))
    snaps = sorted((k for k in z.files if k.startswith("W_ep")), key=lambda k: int(k[4:]))
    extra = dict(rule="emergent") if engine == "lif" else {}
    agent = basic.Agent(basic.Config("spiking", task=cfg["task"], heads=heads, engine=engine, p_cc=cfg["p_cc"],
                               seed=cfg["seed"], **extra))
    agent.W = np.array(z[snaps[-1]], float)
    rng = np.random.default_rng(seed)
    K = len(agent.A)
    counts, dead, P1 = np.zeros((len(THETAS), n_stim, K)), np.zeros(len(THETAS)), []
    for s, v in enumerate(agent.held[:n_stim]):
        p = np.bincount(agent.m, agent.walk(v), agent.K)
        P1.append(core.policy(p, agent.roles))
        for S in frames(agent, v, n_frames, rng):
            for t, th in enumerate(THETAS):
                r = sp.diffbound(S, agent.race_groups, th, rng, agent.burn, cap=1, heads=heads)
                if r is not None:
                    counts[t, s, r[0]] += 1
                    dead[t] += r[1]
    P1 = np.array(P1)
    return dict(task=cfg["task"], beta_eff=[beta_ml(c, P1) for c in counts],
                deadline=(dead / (n_stim * n_frames)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--engine", choices=("numpy", "qif", "lif"), default="lif")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stimuli", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [calibrate(f, a.engine, a.heads, a.frames, a.stimuli, a.seed) for f in a.models]
    for r in rows:
        print(r["task"], "beta_eff", np.round(r["beta_eff"], 2).tolist(),
              "deadline", np.round(r["deadline"], 3).tolist())
    table = tuple((th, round(float(b), 2)) for th, b in zip(THETAS, np.mean([r["beta_eff"] for r in rows], 0)))
    print(f"table {a.heads}: {table}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(rows=rows, table=table, args=vars(a)), fh, indent=1)


if __name__ == "__main__":
    main()
