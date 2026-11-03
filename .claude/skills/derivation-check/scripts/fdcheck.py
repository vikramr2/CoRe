#!/usr/bin/env python3
"""Numerical checks required by CLAUDE.md before changing docs/derivation.md.

Import it (``sys.path.insert(0, <this dir>)``; ``import fdcheck as fc``) or run
``python fdcheck.py --selftest`` to reproduce the reference numbers.

Conventions follow the derivation: W[i, j] is the synapse pre j -> post i,
T = W / d with d_j = sum_i W[i, j], J = pi_j T[i, j], modules m[i] in 0..K-1.
"""
import argparse

import numpy as np


def make_net(sizes=(4, 3, 3, 3), alpha=0.2, seed=0, within=0.6):
    """Random modular network. Module 0 is the controller; the rest are latent/action."""
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(len(sizes)), sizes)
    N = len(m)
    W = rng.uniform(0.05, 0.4, (N, N))
    W[m[:, None] == m[None, :]] += within
    np.fill_diagonal(W, 0)
    return dict(W=W, m=m, K=len(sizes), N=N, alpha=alpha, rng=rng)


def T_of(W):
    return W / W.sum(0, keepdims=True)


def flow(W, v, alpha):
    """Exact per-frame flow pi = alpha (I - (1 - alpha) T)^-1 v."""
    return alpha * np.linalg.solve(np.eye(len(v)) - (1 - alpha) * T_of(W), v)


def stimulus(net, module=0, conc=0.5):
    """Random teleportation vector concentrated on one module (the controller by default)."""
    v = np.zeros(net["N"])
    idx = net["m"] == module
    v[idx] = net["rng"].dirichlet(np.ones(idx.sum()) * conc)
    return v


def module_sums(x, m, K):
    return np.array([x[m == k].sum() for k in range(K)])


def map_L(W, pi, m, K):
    """Map equation L(M) in nats, with pi held fixed (EM split)."""
    T = T_of(W)
    ebar = (T * (m[:, None] != m[None, :])).sum(0)
    q, p = module_sums(pi * ebar, m, K), module_sums(pi, m, K)

    def xlx(x):
        return x * np.log(x)

    return xlx(q.sum()) - 2 * xlx(q).sum() - (pi * np.log(pi)).sum() + xlx(p + q).sum()


def map_modulator(W, pi, m, K):
    """g_ij = M*_{m(j)} I[i not in m(j)]: the map equation's marginal cost (Eq. 6)."""
    T = T_of(W)
    cross = (m[:, None] != m[None, :]).astype(float)
    q = module_sums(pi * (T * cross).sum(0), m, K)
    p = module_sums(pi, m, K)
    Ms = np.log(q.sum() * (p + q) / q**2)
    return Ms[m][None, :] * cross


def lemma_grad(W, pi, g):
    """Eq. 5: dD/dW_ij = (pi_j / d_j) (g_ij - gbar_j), gbar_j = sum_k T_kj g_kj."""
    T, d = T_of(W), W.sum(0)
    return (pi / d)[None, :] * (g - (T * g).sum(0)[None, :])


def fd(f, W, h=1e-6):
    """Central finite differences of scalar f(W) over off-diagonal entries."""
    g = np.zeros_like(W)
    for i, j in zip(*np.nonzero(~np.eye(len(W), dtype=bool))):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += h
        Wm[i, j] -= h
        g[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return g


def off(W):
    return ~np.eye(len(W), dtype=bool)


def check_gradient(D, grad, W):
    """Max |analytic - finite-difference| over off-diagonal entries. D: W -> float."""
    return float(np.abs(grad - fd(D, W))[off(W)].max())


def check_three_factor(W, pis, g_fn, target, frames=4000, steps=200, rng=None):
    """Monte Carlo of the three-factor rule averaged over frames.

    pis: per-frame flows; g_fn(W, pi) -> modulator matrix g_ij; target: matrix to compare
    with (usually -W * grad of the frame-averaged D). Pairings are sampled at rate J_ij.
    Returns the correlation between the average update and the target.
    """
    rng = rng or np.random.default_rng(1)
    acc = np.zeros_like(W)
    T = T_of(W)
    for _ in range(frames):
        pi = pis[rng.integers(len(pis))]
        J = pi[None, :] * T
        g = g_fn(W, pi)
        cnt = rng.multinomial(steps, (J / J.sum()).ravel()).reshape(W.shape) * (J.sum() / steps)
        acc += -(g - (T * g).sum(0)[None, :]) * cnt
    acc /= frames
    o = off(W)
    return float(np.corrcoef(acc[o], target[o])[0, 1])


# ---------------------------------------------------------------- observed walk (Eqs. 8-10)

def make_obs(net, n_inhib=0, seed=None, chance=(0.3, 0.6), indirect=0.02, h_sd=0.5):
    """Observed-walk ingredients for `net` (Eq. 8): presynaptic sign, target sensitivity
    h_i = chi omega_i G_i, chance term a_i = r_i w, indirect term X (held fixed), flow pi.

    n_inhib: inhibitory neurons per module (the last ones of each module). chance=None or
    indirect=0 switch those terms off; h_sd=0 gives equal sensitivities."""
    rng = np.random.default_rng(seed if seed is not None else 7)
    m, N = net["m"], net["N"]
    sgn = np.ones(N)
    for k in range(net["K"]):
        idx = np.flatnonzero(m == k)
        sgn[idx[len(idx) - n_inhib:]] = -1.0 if n_inhib else 1.0
    h = 0.3 * rng.lognormal(0, h_sd, N) if h_sd else np.full(N, 0.3)
    a = rng.uniform(*chance, N) if chance else np.zeros(N)
    X = rng.normal(0, indirect, (N, N)) if indirect else np.zeros((N, N))
    pi = rng.dirichlet(3 * np.ones(N))
    return dict(sgn=sgn, h=h, a=a, X=X, pi=pi)


def obs_E(W, sgn, h, a, X):
    """Eq. 8: E_ij = a_i + sgn_j h_i T_ij + X_ij (chance + direct + indirect)."""
    return a[:, None] + sgn[None, :] * h[:, None] * T_of(W) + X


def obs_walk(E, pi):
    """Eq. 8: T^obs = E / E_j (column-normalised) and J^obs = pi_j T^obs_ij."""
    Tobs = E / E.sum(0, keepdims=True)
    return Tobs, pi[None, :] * Tobs


def map_L_J(J, pi, m, K):
    """Map equation (Eq. 3) for an arbitrary link flow J (pi fixed). Returns L, p, q."""
    cross = m[:, None] != m[None, :]
    q, p = module_sums((J * cross).sum(0), m, K), module_sums(pi, m, K)

    def xlx(x):
        return x * np.log(x)

    return xlx(q.sum()) - 2 * xlx(q).sum() - (pi * np.log(pi)).sum() + xlx(p + q).sum(), p, q


def obs_L(W, o, m, K):
    """L on the observed walk, as a function of W with pi, a, X, h held fixed (local split)."""
    E = obs_E(W, o["sgn"], o["h"], o["a"], o["X"])
    return map_L_J(obs_walk(E, o["pi"])[1], o["pi"], m, K)[0]


def obs_modulator(W, o, m, K):
    """g_ij = M*_{m(j)} I[i not in m(j)] (Eq. 6) on the observed walk, and ebar^obs_j."""
    E = obs_E(W, o["sgn"], o["h"], o["a"], o["X"])
    Tobs, J = obs_walk(E, o["pi"])
    _, p, q = map_L_J(J, o["pi"], m, K)
    cross = (m[:, None] != m[None, :]).astype(float)
    Ms = np.log(q.sum() * (p + q) / q**2)
    return Ms[m][None, :] * cross, (Tobs * cross).sum(0), Ms


def lemma_E_grad(W, o, g):
    """Eq. 9 (Lemma E): dL/dW_ij = pi_j sgn_j / (E_j d_j) [h_i (g_ij - gbar^obs_j) - C_j],
    gbar^obs_j = sum_k T^obs_kj g_kj, C_j = sum_k h_k T_kj (g_kj - gbar^obs_j)."""
    E = obs_E(W, o["sgn"], o["h"], o["a"], o["X"])
    Tobs = E / E.sum(0, keepdims=True)
    T, d, h = T_of(W), W.sum(0), o["h"]
    gbar = (Tobs * g).sum(0)
    C = (h[:, None] * T * (g - gbar[None, :])).sum(0)
    return (o["pi"] * o["sgn"] / (E.sum(0) * d))[None, :] * (h[:, None] * (g - gbar[None, :]) - C[None, :])


def emergent_update(W, o, m, K):
    """Eq. 10 expected update: -M*_{m(j)} (I[i not in m(j)] - ebar^obs_j) E[c_ij], with the
    signed counter's mean E[c_ij] = zeta pi_j sgn_j h_i T_ij (zeta = T_f sum r set to 1)."""
    g, ebar, Ms = obs_modulator(W, o, m, K)
    cross = (m[:, None] != m[None, :]).astype(float)
    Ec = o["pi"][None, :] * o["sgn"][None, :] * o["h"][:, None] * T_of(W)
    return -Ms[m][None, :] * (cross - ebar[None, :]) * Ec


def remove_col_rescaling(V, W):
    """Project out the per-column component proportional to W (a column rescaling)."""
    return V - W * (V.sum(0) / W.sum(0))[None, :]


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def emergent_check(sizes=(4, 4, 4, 4), n_inhib=0, seed=0):
    """(a) Lemma E vs FD (relative), (b) column sums of W*grad L, (c) Theorem E cosine."""
    net = make_net(sizes, seed=seed)
    W, m, K = net["W"], net["m"], net["K"]
    o = make_obs(net, n_inhib=n_inhib, seed=seed + 11)
    assert (obs_E(W, o["sgn"], o["h"], o["a"], o["X"]) > 0).all()
    g = obs_modulator(W, o, m, K)[0]
    grad = lemma_E_grad(W, o, g)
    o_ = off(W)
    rel = float(np.abs(grad - fd(lambda X: obs_L(X, o, m, K), W))[o_].max() / np.abs(grad[o_]).max())
    Wg = W * grad
    colsum = float(np.abs(Wg.sum(0)).max() / np.abs(Wg).max())
    Ej = obs_E(W, o["sgn"], o["h"], o["a"], o["X"]).sum(0)
    u = remove_col_rescaling(emergent_update(W, o, m, K), W)
    return dict(N=net["N"], n_inhib=n_inhib * K, lemma_rel_err=rel, colsum=colsum,
                cos_theoremE=cosine(u[o_], (-Ej[None, :] * Wg)[o_]))


def emergent_reduction(seed=0):
    """(d) a = X = 0, sgn = +1, equal h: Lemma E equals the Lemma (Eq. 5). Max abs difference."""
    net = make_net((4, 4, 4, 4), seed=seed)
    W, m, K = net["W"], net["m"], net["K"]
    o = make_obs(net, n_inhib=0, seed=seed + 11, chance=None, indirect=0, h_sd=0)
    g = obs_modulator(W, o, m, K)[0]
    return float(np.abs(lemma_E_grad(W, o, g) - lemma_grad(W, o["pi"], g))[off(W)].max())


def emergent_selftest():
    ok = True
    for n_inhib in (0, 1):
        r = emergent_check(n_inhib=n_inhib)
        print(f"N={r['N']}, {r['n_inhib']} inhibitory: Lemma E vs FD rel err {r['lemma_rel_err']:.1e} (<= 1e-6); "
              f"col sums of W*gradL {r['colsum']:.1e}; Theorem E cosine {r['cos_theoremE']:.7f} (>= 0.999)")
        ok &= r["lemma_rel_err"] <= 1e-6 and r["colsum"] <= 1e-10 and r["cos_theoremE"] >= 0.999
    red = emergent_reduction()
    print(f"reduction a = X = 0, sgn = +1, equal h: max |Lemma E - Lemma| {red:.1e} (<= 1e-10)")
    ok &= red <= 1e-10
    print("EMERGENT", "PASS" if ok else "FAIL")
    return ok


def selftest():
    net = make_net()
    W, m, K, a = net["W"], net["m"], net["K"], net["alpha"]
    pis = [flow(W, stimulus(net), a) for _ in range(6)]
    errs = [check_gradient(lambda X, pi=pi: map_L(X, pi, m, K), lemma_grad(W, pi, map_modulator(W, pi, m, K)), W)
            for pi in pis]
    gbar = np.mean([lemma_grad(W, pi, map_modulator(W, pi, m, K)) for pi in pis], 0)
    corr = check_three_factor(W, pis, lambda X, pi: map_modulator(X, pi, m, K), -W * gbar)
    print(f"map-equation gradient vs FD, max error over 6 frames: {max(errs):.1e}   (expect <= 1e-8)")
    print(f"three-factor Monte Carlo vs -W*grad D (frame-averaged): corr {corr:.5f}   (expect >= 0.999)")
    ok = max(errs) <= 1e-8 and corr >= 0.999
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emergent", action="store_true", help="observed-walk checks (Eqs. 8-10)")
    args = ap.parse_args()
    if args.selftest or args.emergent:
        ok = (selftest() if args.selftest else True) and (emergent_selftest() if args.emergent else True)
        raise SystemExit(0 if ok else 1)
    ap.print_help()
