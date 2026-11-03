"""The task registry shared by both models (docs/SPEC.md §4, docs/model.md §1): per Gymnasium task the module
sizes, the stimulus encoder v(o) over the controller and the reward r_f fed to the TD error (Eq. 4c); the
iteration 012 encoders (grid, Fourier) and the separate critic layer. Details in models/basic.py's docstring."""
import dataclasses
import re
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from mapstdp import core

LATENTS, ACTION_SIZE, ENC_SEED, POOL = (16, 16), 16, 777, 20000


@dataclasses.dataclass
class Task:
    env_id: str
    n_actions: int
    n_ctrl: int
    episodes: int  # default training length (M3 reporting unit: 5 seeds x episodes)
    reward: Callable  # (r, term) -> r_f of Eq. 4c
    encoder: str  # 'cartpole' | 'grid' | 'pool' | 'fourier' (with_encoder)
    enc: dict = dataclasses.field(default_factory=dict)  # fitted encoder state (centers, sigma, normalisation)

    @property
    def sizes(self):
        return (self.n_ctrl, *LATENTS, *(ACTION_SIZE,) * self.n_actions)

    @property
    def roles(self):
        return "C" + "L" * len(LATENTS) + "A" * self.n_actions

    def stim(self, o, m, roles):
        """v(o) over the controller neurons."""
        if self.encoder == "cartpole":
            return core.cartpole_stimulus(o, m, roles)
        if self.encoder == "fourier":
            e = self.enc
            return core.fourier_stimulus(unit(o, e["lo"], e["hi"]), e["C"], m, roles, e["onoff"], e["bias"])
        if not self.enc:
            self.enc = fit_encoder(self)
        e = self.enc
        return core.rbf_stimulus(np.clip((np.asarray(o, float) - e["shift"]) / e["scale"], *e["clip"]),
                                 e["centers"], e["sigma"], m, roles)


def random_states(env_id, n_actions, n, seed):
    """n states visited by a uniform-random policy, episodes from consecutive resets."""
    env, rng, obs = gym.make(env_id), np.random.default_rng(seed), []
    while len(obs) < n:
        o, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            obs.append(o)
            o, _, term, trunc, _ = env.step(int(rng.integers(n_actions)))
            done = term or trunc
    return np.array(obs[:n])


def fit_encoder(task):
    """Encoder state for the 'grid' and 'pool' encoders (fixed seed: a sensory front-end shared by every agent seed)."""
    if task.encoder == "grid":
        sp = gym.make(task.env_id).observation_space
        g = np.linspace(-1, 1, 6)
        return dict(shift=(sp.high + sp.low) / 2, scale=(sp.high - sp.low) / 2, clip=(-1, 1), sigma=0.4,
                    centers=np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2))
    X = random_states(task.env_id, task.n_actions, POOL, ENC_SEED)
    shift, scale = X.mean(0), np.maximum(X.std(0), 1e-6)
    Z = (X - shift) / scale
    C = Z[np.random.default_rng(ENC_SEED).choice(len(Z), task.n_ctrl, replace=False)]
    D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1)) + np.diag(np.full(len(C), np.inf))
    return dict(shift=shift, scale=scale, clip=(-4, 4), centers=C, sigma=float(np.median(D.min(1))))


TASKS = {t.env_id.split("-")[0]: t for t in (
    Task("CartPole-v1", 2, 36, 1500, lambda r, term: -1.0 if term else 0.0, "cartpole"),
    Task("MountainCar-v0", 3, 36, 1500, lambda r, term: 0.1 * r, "grid"),
    Task("Acrobot-v1", 3, 64, 1000, lambda r, term: 0.1 * r, "pool"),
    Task("LunarLander-v3", 4, 64, 1000, lambda r, term: 0.01 * r, "pool"),
)}
# CartPole with the 'pool' encoder over all four observations (iteration 006): the M2 encoder sees only theta and
# theta_dot, so neither the policy nor a network-activity critic can see the cart position that ends sharp-policy runs.
TASKS["CartPoleFull"] = Task("CartPole-v1", 2, 64, 1500, TASKS["CartPole"].reward, "pool")


# Iteration 012 (Proposal): encoder and critic-layer specs.
#   g<n>[s<sigma>]          n x n Gaussian grid over a 2-D observation normalised by its bounds to [-1, 1]
#                           (sigma default 0.8 x the grid spacing 2 / (n - 1)); MountainCar only.
#   f<order>c<conj>[h][b]   Fourier basis over the observation scaled to [0, 1] by FOURIER_BOUNDS: coefficient vectors
#                           in {0..order}^d with 1..conj nonzero entries, ON/OFF split ([cos]+, [-cos]+) or, with h,
#                           (1 + cos) / 2; b appends a constant unit. One feature per controller neuron.
FOURIER_BOUNDS = {  # (lo, hi) per observation dimension, clipped outside (hand-set from the observation physics)
    "CartPole-v1": ([-2.4, -2.0, -0.21, -2.5], [2.4, 2.0, 0.21, 2.5]),
    "Acrobot-v1": ([-1, -1, -1, -1, -2 * np.pi, -4 * np.pi], [1, 1, 1, 1, 2 * np.pi, 4 * np.pi]),
    "LunarLander-v3": ([-1, -0.1, -1.5, -2.0, -1.5, -2.0, 0, 0], [1, 1.5, 1.5, 0.5, 1.5, 2.0, 1, 1]),
}


def unit(o, lo, hi):
    """Observation scaled to [0, 1] by (lo, hi), clipped."""
    return np.clip((np.asarray(o, float) - lo) / (hi - lo), 0, 1)


def grid_centers(n):
    g = np.linspace(-1, 1, n)
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def parse_spec(spec):
    """'g12s0.15' -> ('grid', 12, 0.15); 'f2c2' -> ('fourier', 2, 2, onoff, bias)."""
    g = re.fullmatch(r"g(\d+)(?:s([\d.]+))?", spec)
    if g:
        n = int(g[1])
        return "grid", n, float(g[2]) if g[2] else 0.8 * 2 / (n - 1)
    f = re.fullmatch(r"f(\d+)c(\d+)(h?)(b?)", spec)
    assert f, f"bad encoder spec {spec!r}"
    return "fourier", int(f[1]), int(f[2]), not f[3], bool(f[4])


def with_encoder(task, spec):
    """The task with the controller encoder spec (n_ctrl = feature count, enc prebuilt)."""
    kind, *a = parse_spec(spec)
    if kind == "grid":
        sp = gym.make(task.env_id).observation_space
        assert sp.shape == (2,), "grid encoder specs are for 2-D observations"
        enc = dict(shift=(sp.high + sp.low) / 2, scale=(sp.high - sp.low) / 2, clip=(-1, 1), sigma=a[1],
                   centers=grid_centers(a[0]))
        return dataclasses.replace(task, n_ctrl=a[0] ** 2, encoder="grid", enc=enc)
    lo, hi = (np.array(x, float) for x in FOURIER_BOUNDS[task.env_id])
    C = core.fourier_coeffs(len(lo), a[0], a[1])
    enc = dict(lo=lo, hi=hi, C=C, onoff=a[2], bias=a[3])
    return dataclasses.replace(task, n_ctrl=len(C) * (2 if a[2] else 1) + a[3], encoder="fourier", enc=enc)


def critic_layer(task, spec):
    """Separate critic (iteration 012 B, Proposal): a fixed nonnegative RF layer y = phi(o) outside the actor.
    Grid specs: Gaussian RFs normalised to sum 1 (as the controller's v); Fourier specs: raw rates (no normalisation),
    so V = u . y is a linear function of the Fourier features (with b: plus a constant)."""
    kind, *a = parse_spec(spec)
    if kind == "grid":
        sp = gym.make(task.env_id).observation_space
        shift, scale, C, s = (sp.high + sp.low) / 2, (sp.high - sp.low) / 2, grid_centers(a[0]), a[1]

        def phi(o):
            d2 = ((C - np.clip((np.asarray(o, float) - shift) / scale, -1, 1)) ** 2).sum(1)
            b = np.exp(-(d2 - d2.min()) / (2 * s ** 2))
            return b / b.sum()
        return phi
    lo, hi = (np.array(x, float) for x in FOURIER_BOUNDS[task.env_id])
    C = core.fourier_coeffs(len(lo), a[0], a[1])
    return lambda o: core.fourier_features(unit(o, lo, hi), C, a[2], a[3])
