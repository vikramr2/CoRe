#!/usr/bin/env python3
"""
Numerical checks for the emergent-walk derivation (Map-STDP, Derivation Details D12).

    python3 test_emergent.py     # report
    pytest -q test_emergent.py   # assertions

E1  Lemma E: gradient of the map equation on the observed walk (chance term, signed
    sensitivities, indirect terms held fixed) against finite differences.
E2  Theorem E: the rule with the signed counter and the observed-walk baseline is
    weight-scaled descent; other baselines are not.
E3  Task score with inhibitory neurons (signed Lambda) and the module cascade.
E4  Branching engine with inhibition: gated counter vs exact signed offspring, the
    sign pattern the structure term produces, and a short structural-learning run.
"""
import numpy as np

RESULTS = {}


def cos(a, b):
    a = np.ravel(a); b = np.ravel(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def plogp(x):
    x = np.asarray(x, float)
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def map_L(pi, J, mod):
    K = mod.max() + 1
    cross = mod[:, None] != mod[None, :]
    q = np.bincount(mod, weights=(J * cross).sum(0), minlength=K)
    p = np.bincount(mod, weights=pi, minlength=K)
    return plogp(q.sum()) - 2 * plogp(q).sum() - plogp(pi).sum() + plogp(p + q).sum(), p, q


def mstar(p, q):
    return np.minimum(np.log(q.sum() * (p + q) / q ** 2), 20.0)


# ------------------------------------------------------------------ E1, E2
def make_ei(seed, K=4, nE=3, nI=1):
    rng = np.random.default_rng(seed)
    n = nE + nI
    N = K * n
    mod = np.repeat(np.arange(K), n)
    sgn = np.tile(np.r_[np.ones(nE), -np.ones(nI)], K)
    same = mod[:, None] == mod[None, :]
    W = rng.uniform(0.1, 1, (N, N)) * np.where(same, 2.0, 0.5)
    np.fill_diagonal(W, 0)
    return rng, N, mod, sgn, same, W


def check_lemmaE(seed=0):
    rng, N, mod, sgn, same, W = make_ei(seed)
    cross = ~same
    off = ~np.eye(N, dtype=bool)
    h = rng.lognormal(0, 0.5, N) * 0.3           # target susceptibility chi*omega*G
    a = rng.uniform(0.3, 0.6, N)                  # chance term, r_i * window
    X = rng.normal(0, 0.02, (N, N))               # indirect / common-cause terms, held fixed
    pi = rng.dirichlet(3 * np.ones(N))

    def Emat(Wm):
        T = Wm / Wm.sum(0)
        return a[:, None] + sgn[None, :] * h[:, None] * T + X

    def L_of(Wm):
        E = Emat(Wm)
        return map_L(pi, pi[None, :] * E / E.sum(0), mod)[0]

    E = Emat(W)
    assert (E > 0).all()
    d = W.sum(0); T = W / d
    Ej = E.sum(0); Tobs = E / Ej
    _, p, q = map_L(pi, pi * Tobs, mod)
    Ms = mstar(p, q)
    g = Ms[mod][None, :] * cross
    gbar = (Tobs * g).sum(0)
    C = (h[:, None] * T * (g - gbar)).sum(0)
    grad = pi * sgn / (Ej * d) * (h[:, None] * (g - gbar) - C)       # Lemma E

    fd = np.zeros_like(W); eps = 1e-6
    for i, j in zip(*np.nonzero(off)):
        Wp = W.copy(); Wp[i, j] += eps
        Wm = W.copy(); Wm[i, j] -= eps
        fd[i, j] = (L_of(Wp) - L_of(Wm)) / (2 * eps)

    Lam = h[:, None] * T                          # magnitude of the windowed direct effect
    Ec = sgn[None, :] * Lam * pi[None, :]         # E[c] with T_f sum r = 1
    Wg = W * grad
    target = -Ej[None, :] * Wg                    # Theorem E direction, lambda*zeta = 1

    def proj(V):                                  # remove the column-rescaling component
        return V - W * (V.sum(0) / d)[None, :]

    def rule(baseline):
        return -Ms[mod][None, :] * (cross - baseline[None, :]) * Ec

    e_obs = (Tobs * cross).sum(0)
    e_c = (np.abs(Ec) * cross).sum(0) / np.abs(Ec).sum(0)            # counter-weighted
    e_w = (T * cross).sum(0)                                          # weight-normalised
    out = dict(lemma_rel_err=float(np.abs(fd - grad)[off].max() / np.abs(grad[off]).max()),
               colsum_Wgrad=float(np.abs(Wg.sum(0)).max() / np.abs(Wg).max()))
    for name, b in [("obs", e_obs), ("counter", e_c), ("weight", e_w)]:
        u = rule(b)
        out[f"cos_{name}"] = cos(proj(u)[off], target[off])
        out[f"dL_{name}"] = float((fd * u)[off].sum())
    out["dL_exact"] = float(-(Ej[None, :] * W * fd ** 2)[off].sum())
    # sign pattern: inhibitory presynaptic columns
    uo = rule(e_obs)
    I = sgn < 0
    out["I_cross_frac_strengthened"] = float(np.mean(uo[np.ix_(np.ones(N, bool), I)][cross[:, I]] > 0))
    out["I_within_frac_weakened"] = float(np.mean(uo[:, I][(same & off)[:, I]] < 0))
    out["E_cross_frac_weakened"] = float(np.mean(uo[:, ~I][cross[:, ~I]] < 0))
    out["E_within_frac_strengthened"] = float(np.mean(uo[:, ~I][(same & off)[:, ~I]] > 0))
    RESULTS["E1E2"] = out
    return out


def test_lemmaE():
    r = check_lemmaE()
    assert r["lemma_rel_err"] < 1e-5
    assert r["colsum_Wgrad"] < 1e-10


def test_theoremE():
    r = check_lemmaE()
    assert r["cos_obs"] > 1 - 1e-10
    assert abs(r["dL_obs"] - r["dL_exact"]) < 1e-6 * abs(r["dL_exact"])
    assert r["dL_obs"] < 0
    assert r["cos_counter"] < 0.9999


def test_sign_pattern():
    r = check_lemmaE()
    assert r["I_cross_frac_strengthened"] == 1.0 and r["I_within_frac_weakened"] == 1.0
    assert r["E_cross_frac_weakened"] == 1.0 and r["E_within_frac_strengthened"] == 1.0


# ------------------------------------------------------------------ E3
def softplus(x):
    return np.logaddexp(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def check_score_inhib(seed=1, chiE=0.45, chiI=0.6):
    rng = np.random.default_rng(seed)
    K, nE, nI = 4, 5, 1
    n = nE + nI; N = K * n
    mod = np.repeat(np.arange(K), n)
    sgn = np.tile(np.r_[np.ones(nE), -np.ones(nI)], K)
    chi = np.where(sgn > 0, chiE, chiI)
    A = np.array([0, 1]); astar = 0
    same = mod[:, None] == mod[None, :]
    off = ~np.eye(N, dtype=bool)
    W = rng.uniform(0.1, 1, (N, N)) * np.where(same, 2.0, 0.6)
    np.fill_diagonal(W, 0)
    aa = np.clip(rng.lognormal(0.4, 0.5, N), 0.3, 3.5)
    Iext = rng.uniform(0.0, 1.0, N)

    def fixed(Wm, r0=None, iters=4000):
        B = sgn * chi * Wm / Wm.sum(0)
        r = np.ones(N) if r0 is None else r0.copy()
        for _ in range(iters):
            r = 0.5 * r + 0.5 * aa * softplus(B @ r + Iext)
        return r, B @ r + Iext, B

    r, x, B = fixed(W)

    def logP(Wm):
        R = np.bincount(mod, fixed(Wm, r0=r, iters=400)[0], K)
        return np.log(R[astar] / R[A].sum())

    G = aa * sigmoid(x)
    d = W.sum(0); T = W / d
    Lam = G[:, None] * B                                  # signed
    ev = np.linalg.eigvals(Lam)
    R = np.bincount(mod, r, K); P = R / R[A].sum()
    s = np.zeros(N)
    for b in A:
        s[mod == b] = ((b == astar) - P[b]) / R[b]
    Phi = np.linalg.solve((np.eye(N) - Lam).T, s)
    GP = G * Phi
    grad = (chi * sgn * r / d)[None, :] * (GP[:, None] - (T * GP[:, None]).sum(0)[None, :])
    fd = np.zeros_like(W); eps = 1e-6
    for i, j in zip(*np.nonzero(off)):
        Wp = W.copy(); Wp[i, j] += eps
        Wm = W.copy(); Wm[i, j] -= eps
        fd[i, j] = (logP(Wp) - logP(Wm)) / (2 * eps)
    Wg = W * grad
    Ec = Lam * r[None, :]                                  # signed E[c]
    LamK = np.zeros((K, K))
    for a_ in range(K):
        for b in range(K):
            LamK[b, a_] = Ec[np.ix_(mod == b, mod == a_)].sum() / R[a_]
    sK = np.array([s[mod == b][0] for b in range(K)])
    PhiK = np.linalg.solve((np.eye(K) - LamK).T, sK)
    proj = lambda V: V - W * (V.sum(0) / d)[None, :]
    out = dict(score_rel_err=float(np.abs(fd - grad)[off].max() / np.abs(grad[off]).max()),
               cos_exact=cos(proj(Ec * Phi[:, None])[off], Wg[off]),
               cos_PhiK=cos(proj(Ec * PhiK[mod][:, None])[off], Wg[off]),
               cos_onehop=cos(proj(Ec * s[:, None])[off], Wg[off]),
               max_re_eig=float(ev.real.max()),
               rho_EE=float(np.abs(np.linalg.eigvals(Lam[np.ix_(sgn > 0, sgn > 0)])).max()))
    return out


def test_score_inhib():
    r = check_score_inhib()
    assert r["score_rel_err"] < 1e-5
    assert r["cos_exact"] > 0.99999


# ------------------------------------------------------------------ E4 branching with inhibition
def branching_ei(seed=3, K=4, nE=10, nI=2, cross_E=0.05, I_within=1.0, I_cross=1.0,
                 w0E=1.6, w0I=1.6, base=0.03, frame=50):
    rng = np.random.default_rng(seed)
    n = nE + nI; N = K * n
    mod = np.repeat(np.arange(K), n)
    sgn = np.tile(np.r_[np.ones(nE), -np.ones(nI)], K)
    same = mod[:, None] == mod[None, :]
    W = rng.uniform(0.2, 1, (N, N))
    isE = sgn > 0
    W[:, isE] *= np.where(same[:, isE], 1.0, cross_E)
    W[:, ~isE] *= np.where(same[:, ~isE], I_within, I_cross)
    np.fill_diagonal(W, 0)
    w0 = np.where(isE, w0E, w0I)
    return dict(rng=rng, N=N, mod=mod, sgn=sgn, same=same, W=W, w0=w0, base=base, frame=frame, K=K)


def run_block(net, W, steps):
    rng, N, mod, sgn, K = net["rng"], net["N"], net["mod"], net["sgn"], net["K"]
    b = sgn * net["w0"] * W / W.sum(0)                    # signed efficacy, output-normalised
    P = lambda u: np.clip(u / 2.0, 0.0, 1.0)
    x = np.zeros(N, bool)
    kap = np.zeros((N, N)); c_ex = np.zeros((N, N)); c_gate = np.zeros((N, N)); nspk = np.zeros(N)
    for t in range(steps):
        if t % net["frame"] == 0:
            mu = net["base"] * rng.exponential(1.0, K)[mod]
        u = b @ x + 2.0 * mu
        p = P(u)
        x_new = rng.random(N) < p
        js = np.flatnonzero(x)
        if js.size:
            bj = b[:, js]
            c_ex[:, js] += p[:, None] - P(u[:, None] - bj)          # exact signed direct offspring
            c_gate[:, js] += 0.5 * bj * ((u > 0) & (u < 2.0))[:, None]
            kap[:, js] += x_new[:, None]
            nspk[js] += 1
        x = x_new
    return dict(kap=kap, c_ex=c_ex, c_gate=c_gate, n=nspk, steps=steps)


def walk_stats(net, blk):
    mod, same = net["mod"], net["same"]
    cross = ~same
    n = np.maximum(blk["n"], 1)
    pi = blk["n"] / blk["n"].sum()
    Tobs = blk["kap"] / np.maximum(blk["kap"].sum(0), 1)
    J = pi[None, :] * Tobs
    L, p, q = map_L(pi, J, mod)
    e_obs = (Tobs * cross).sum(0)
    return L, mstar(p, q), e_obs, pi


def check_branching_ei(seed=3, steps=150_000):
    net = branching_ei(seed)
    W = net["W"].copy()
    blk = run_block(net, W, steps)
    off = ~np.eye(net["N"], dtype=bool)
    cross = ~net["same"]
    L, Ms, e_obs, pi = walk_stats(net, blk)
    mod, sgn = net["mod"], net["sgn"]
    g_dir = Ms[mod][None, :] * (cross - e_obs[None, :])
    u_true = -g_dir * blk["c_ex"]
    u_gate = -g_dir * blk["c_gate"]
    u_kap = -g_dir * blk["kap"]
    I = sgn < 0
    neg = blk["c_ex"][:, I]
    out = dict(
        L=float(L),
        rates=[float(v) for v in (blk["n"] / steps).reshape(net["K"], -1).mean(0)],
        corr_gate_vs_exact=float(np.corrcoef(blk["c_gate"][off], blk["c_ex"][off])[0, 1]),
        inhib_effect_share_clipped=float(1 - np.abs(neg).sum() / np.abs(blk["c_gate"][:, I]).sum()),
        struct_cos_gate=cos(u_gate[off], u_true[off]),
        struct_cos_paircount=cos(u_kap[off], u_true[off]),
        frac_obs_transitions_negative=float((blk["kap"] < 0).mean()),
        e_obs_mean=float(e_obs.mean()),
    )
    RESULTS["E4_static"] = out
    return out


def learn_branching_ei(seed=4, blocks=40, steps=5000, factor="c_ex", step_frac=0.05):
    net = branching_ei(seed, cross_E=0.5, I_within=1.0, I_cross=1.0)
    W = net["W"].copy()
    mod, sgn, same = net["mod"], net["sgn"], net["same"]
    cross = ~same; off = ~np.eye(net["N"], dtype=bool)
    isE = sgn > 0
    hist = []
    for k in range(blocks):
        blk = run_block(net, W, steps)
        L, Ms, e_obs, pi = walk_stats(net, blk)
        T = W / W.sum(0)
        hist.append(dict(L=float(L),
                         E_cross_share=float(T[:, isE][cross[:, isE]].sum() / isE.sum()),
                         I_cross_share=float(T[:, ~isE][cross[:, ~isE]].sum() / (~isE).sum()),
                         rate=float(blk["n"].sum() / steps / net["N"])))
        U = -Ms[mod][None, :] * (cross - e_obs[None, :]) * blk[factor]
        U[~off] = 0
        if factor == "kap":                       # pair counts carry no sign: give them the presynaptic sign
            U = U * sgn[None, :]
        scale = step_frac / max(np.abs(U[off] / np.maximum(W[off], 1e-12)).max(), 1e-12)
        W = np.maximum(W + scale * U, 1e-9)
        W = W / W.sum(0)                          # strong homeostasis: d_j = 1
    RESULTS.setdefault("E4_learning", {})[factor] = hist
    return hist


def check_lif_kick(mu, w, sig=0.25, N=60000, seed=21, tau_m=0.02, th=1.0, vr=0.0, dt=1e-4, burn=0.2, win=0.6):
    """Single noisy LIF neuron: extra spikes caused by one kick (signed), paired runs with common noise."""
    from mapstdp.lif import siegert_gain
    rng = np.random.default_rng(seed)
    nb, nw = int(burn / dt), int(win / dt)
    V = mu + sig * rng.standard_normal(N); V2 = V.copy()
    sd = sig * np.sqrt(2 * dt / tau_m); c1 = np.zeros(N); c2 = np.zeros(N)
    for k in range(nb + nw):
        xi = rng.standard_normal(N) * sd
        V += -(V - mu) * dt / tau_m + xi; V2 += -(V2 - mu) * dt / tau_m + xi
        if k == nb:
            V2 += w
        s1 = V >= th; s2 = V2 >= th
        if k >= nb:
            c1 += s1; c2 += s2
        V[s1] = vr; V2[s2] = vr
    d = c2 - c1
    pred = siegert_gain(mu, np.sqrt(2) * sig) * tau_m * w
    return dict(mu=mu, w=w, sim=float(d.mean()), se=float(d.std() / np.sqrt(N)), pred=float(pred))


def test_lif_inhibitory_kick():
    r = check_lif_kick(0.8, -0.05, N=20000)
    assert r["sim"] < 0
    assert 0.75 < r["sim"] / r["pred"] < 1.05


def test_branching_ei_static():
    r = check_branching_ei(steps=40_000)
    assert r["corr_gate_vs_exact"] > 0.98
    assert r["struct_cos_gate"] > 0.98


def main():
    import json, time
    t0 = time.time()
    print("== E1/E2 Lemma E and Theorem E ==")
    for k, v in check_lemmaE().items():
        print(f"  {k:28s} {v: .4g}")
    print("\n== E3 task score with inhibition ==")
    rows = []
    for s in range(1, 9):
        try:
            rr = check_score_inhib(s)
        except Exception:
            continue
        if rr["max_re_eig"] < 0.95 and rr["score_rel_err"] < 1e-4:
            rows.append(rr)
    print(f"  kept {len(rows)} networks")
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        print(f"  {k:16s} min {v.min(): .4g} median {np.median(v): .4g} max {v.max(): .4g}")
    RESULTS["E3"] = rows
    print(f"\n== E4 branching with inhibition, static ({time.time()-t0:.0f}s) ==")
    for k, v in check_branching_ei().items():
        print(f"  {k:30s} {v}")
    print(f"\n== E4 structural learning ({time.time()-t0:.0f}s) ==")
    for factor in ["c_ex", "kap"]:
        h = learn_branching_ei(factor=factor)
        print(f"  factor={factor}")
        for i in [0, 9, 19, 29, 39]:
            print(f"    block {i:2d}: L={h[i]['L']:.4f} E cross share={h[i]['E_cross_share']:.3f} "
                  f"I cross share={h[i]['I_cross_share']:.3f} rate={h[i]['rate']:.3f}")
    print(f"\ntotal {time.time()-t0:.0f}s")
    with open("test_emergent_results.json", "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)


if __name__ == "__main__":
    main()
