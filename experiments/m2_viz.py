"""Record one CartPole episode of a trained spiking Map-STDP agent (model v3) for the replay page.
Run from the repo root:
  python -m experiments.m2_viz train  --seed 3 [--la-beta 0.3]   -> experiments/results/viz_m2_W_s<seed>[_lb<B>].npz
  python -m experiments.m2_viz record --seed 3 [--la-beta 0.3]   -> experiments/results/viz_m2_episode_s<seed>[...].json
  python -m experiments.m2_viz page --seed 3 [--la-beta 0.3] --out replay/replay.html  -> the page, js/ and data/
Recording freezes the weights and the critic. Every frame keeps its spike raster, which spikes were stimulus input,
the walk's hops (one sampled parent per non-input spike), the race winner and the cart state, so the page can replay
the per-frame walk next to the pole."""
import argparse
import json
import pathlib

import gymnasium as gym
import numpy as np

from experiments import m2
from mapstdp import core
from mapstdp import spiking as sp


def path(kind, seed, la_beta=0.0):
    lb = f"_lb{la_beta:g}" if la_beta else ""
    return m2.OUT / (f"viz_m2_W_s{seed}{lb}.npz" if kind == "W" else f"viz_m2_episode_s{seed}{lb}.json")


def train(seed, episodes, la_beta=0.0):
    agent, env, R = m2.Agent(m2.Config("spiking", seed=seed, la_beta=la_beta)), gym.make("CartPole-v1"), []
    for ep in range(1, episodes + 1):
        R.append(agent.episode(env))
        if ep % 50 == 0:
            print(f"ep {ep} last50 {np.mean(R[-50:]):.1f}", flush=True)
    np.savez(path("W", seed, la_beta), W=agent.W, mask=agent.mask, m=agent.m, roles=agent.roles, returns=R)


def first_race_step(S, groups, burn):
    """Step of the first action spike at or after burn (where sp.race decides), -1 if none."""
    cnt = np.stack([S[burn:, g].sum(1) for g in groups], 1).sum(1)
    t = np.flatnonzero(cnt)
    return int(t[0]) + burn if len(t) else -1


def walk_hops(S, X, T, rng):
    """One parent per non-input spike: post i at step t gets a pre j that spiked at t - 1, drawn with probability
    proportional to T[i, j]. In branching mode each presynaptic spike adds W_snm = (1 - alpha) T to the escape
    probability linearly, so this is the exact posterior over which walker step caused the spike. Input spikes
    (X) force their neuron to spike and get no parent. Returns hops as [t, pre, post]."""
    hops = []
    for t in range(1, len(S)):
        pre = np.flatnonzero(S[t - 1])
        for i in np.flatnonzero(S[t] & ~X[t]):
            w = T[i, pre]
            if w.sum() > 0:
                hops.append([t, int(rng.choice(pre, p=w / w.sum())), int(i)])
    return hops


def record(seed, la_beta=0.0, tries=20):
    """Best of `tries` frozen-weight episodes (the policy is stochastic), so the page shows a long balance."""
    z = np.load(path("W", seed, la_beta))
    agent = m2.Agent(m2.Config("spiking", seed=seed, la_beta=la_beta))
    agent.W = z["W"]
    T = agent.W / agent.W.sum(0)
    hop_rng = np.random.default_rng(seed)
    env, best = gym.make("CartPole-v1"), []
    for k in range(tries):
        o, _ = env.reset(seed=1000 + k)
        frames, done = [], False
        while not done:
            v = agent.stim(o)
            sp.set_weights(agent.snn, agent.W, 1 - m2.ALPHA)
            S, X = sp.run_frame(agent.snn, v, m2.K_SPK, m2.RATE, agent.rng, m2.THETA, return_ext=True)
            a = sp.race(S, agent.groups, agent.rng, agent.burn)
            a = int(agent.rng.integers(2)) if a is None else a
            p = core.module_stats(agent.W, core.flow(agent.W, v, m2.ALPHA, m2.N_HOPS), agent.m, agent.K)[0]
            frames.append({"obs": [round(float(x), 4) for x in o], "a": a,
                           "race_t": first_race_step(S, agent.groups, agent.burn),
                           "P": [round(float(x), 3) for x in core.policy(p, agent.roles)],
                           "S": [np.flatnonzero(s).tolist() for s in S],
                           "X": [np.flatnonzero(x).tolist() for x in X & S],
                           "H": walk_hops(S, X, T, hop_rng)})
            o, _, term, trunc, _ = env.step(a)
            done = term or trunc
        print(f"try {k}: {len(frames)} frames", flush=True)
        if len(frames) > len(best):
            best = frames
        if len(best) >= 500:
            break
    i, j = np.nonzero(z["mask"])
    out = {"seed": seed, "la_beta": la_beta, "k": m2.K_SPK, "burn": agent.burn, "alpha": m2.ALPHA,
           "m": agent.m.tolist(),
           "roles": agent.roles.tolist(), "train_returns": z["returns"].tolist(),
           "edges": [[int(a), int(b), round(float(T[a, b]), 4)] for a, b in zip(i, j)],  # [post, pre, T]
           "frames": best}
    path("E", seed, la_beta).write_text(json.dumps(out, separators=(",", ":")))
    print(f"recorded {len(best)} frames -> {path('E', seed, la_beta)}")


def page(seed, out, la_beta=0.0):
    """The page at out, its script at js/<stem>.js and the recorded episode at data/<stem>.json beside it."""
    out, here = pathlib.Path(out), pathlib.Path(__file__).parent
    for sub in ("js", "data"):
        (out.parent / sub).mkdir(parents=True, exist_ok=True)
    out.write_text((here / "m2_viz_template.html").read_text().replace("__STEM__", out.stem))
    (out.parent / "js" / f"{out.stem}.js").write_text((here / "m2_viz_template.js").read_text())
    (out.parent / "data" / f"{out.stem}.json").write_text(path("E", seed, la_beta).read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "record", "page"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", default="replay/replay.html")
    ap.add_argument("--la-beta", type=float, default=0.0, help="latent-output barrier (Eq. 4e, Proposal)")
    a = ap.parse_args()
    {"train": lambda: train(a.seed, a.episodes, a.la_beta), "record": lambda: record(a.seed, a.la_beta),
     "page": lambda: page(a.seed, a.out, a.la_beta)}[a.cmd]()
