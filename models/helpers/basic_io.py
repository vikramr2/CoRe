"""Run plumbing of models/basic.py: result names, checkpoints, the frozen end-of-run episodes and --save-model."""
import copy
import dataclasses
import json
import pathlib

import gymnasium as gym
import numpy as np

from models.helpers.agent import FROZEN_EPIS, FROZEN_SEED, MODEL


def default_name(cfg):
    """The m2 result name of cfg (m2_<backend>_<routing>_lam<lam>[...]_<MODEL>_s<seed>)."""
    b = (f"_b{cfg.beta:g}" if cfg.beta != 1 else "") + (f"_eta{cfg.eta:g}" if cfg.eta else "")
    b += f"_at{cfg.eta_tau:g}" if cfg.eta_tau is not None else ""
    b += f"_lb{cfg.la_beta:g}" if cfg.la_beta else ""
    return f"m2_{cfg.backend}_{cfg.routing}_lam{cfg.lam:g}{b}_{MODEL}_s{cfg.seed}"


def save_checkpoint(ck, agent, returns, evals, snaps, resumes, wall):
    """Write <out>.ckpt.npz atomically: the agent's state(), the returns and evaluations so far, the model snapshots,
    the resume count and the wall time."""
    arr, meta = agent.state()
    tmp = ck.with_name(ck.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, meta=meta, run=json.dumps(dict(returns=returns, evals=evals, resumes=resumes, wall=wall)),
                 **arr, **{f"snap_{k}": x for k, x in snaps.items()})
    tmp.replace(ck)


def load_checkpoint(ck, agent):
    """Restore save_checkpoint's file into a fresh agent; returns (returns, evals, snaps, resumes, wall)."""
    with np.load(ck) as z:
        arr = {k: z[k] for k in z.files if k not in ("meta", "run") and not k.startswith("snap_")}
        agent.load_state(arr, str(z["meta"]))
        r = json.loads(str(z["run"]))
        snaps = {k[5:]: z[k] for k in z.files if k.startswith("snap_")}
    return r["returns"], r["evals"], snaps, r["resumes"], r["wall"]


def frozen_returns(agent, n=FROZEN_EPIS):
    """Returns of n episodes with the final W frozen (rule 'emergent'; a training return can hide a W that has just
    collapsed onto one action, iteration 014 §7): a deep copy of the agent acts through frame() as in episode(), with
    no learning (W, critic and beta fixed), its frames and environment seeds from (seed, FROZEN_SEED); the agent itself
    is untouched."""
    a = copy.deepcopy(agent)
    a.rng = np.random.default_rng((a.cfg.seed, FROZEN_SEED))
    env, out = gym.make(a.task.env_id), []
    for _ in range(n):
        o, _ = env.reset(seed=int(a.rng.integers(1 << 30)))
        a.qstate = a.lstate = None
        R, done = 0.0, False
        while not done:
            o, r, term, trunc, _ = env.step(a.frame(a.stim(o))[2])
            R, done = R + r, term or trunc
        out.append(float(R))
    return out


def save_model(path, agent, snaps, cfg):
    """--save-model: compressed float32 .npz with W_ep<k> (episode 0, every SAVE_EVERY and the end), mask, m, roles,
    the Config as JSON and the final critic u."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: x.astype(np.float32) for k, x in snaps.items()}, mask=agent.mask, m=agent.m,
                        roles=agent.roles, config=json.dumps(dataclasses.asdict(cfg)), u=agent.u.astype(np.float32))
