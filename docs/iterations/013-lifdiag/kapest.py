"""kappa estimator variants on one network/op: whole reset frame, reset after burn-in, long-run stationary."""
import pickle, sys, numpy as np
D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D); import eng, analyze as an
from mapstdp import core
task, wn, opn = sys.argv[1:4]
R = pickle.load(open(f"{D}out/{task}_{wn}_{opn}.pkl", "rb"))
net = eng.network(task, wn == "trained"); W, mk, m, K, roles, N = net["W"], net["mask"], net["m"], net["K"], net["roles"], net["N"]
op = R["op"]; lif = op["kind"] == "lif"; B = eng.B_of(net, op); T = W / W.sum(0); cr = an.cross_ne(m, roles); A = roles == "A"
sim = eng.Sim(net, op); rng = np.random.default_rng(99)
Tf, burn = (300, 100) if lif else (100, 20)
res = []
for s in range(2):
    st = R["stat"][s]; r = st["r"]; G = st["G"][0]; v = R["V"][s]; Lam = G[:, None] * B; pi = r / r.sum(); nu = Lam.sum(0)
    Te, S_ = an.teff(W, G); g = an.mod_g(pi, Te * pi, m, K, cr); gbar = (Te * g).sum(0)
    tgt = -nu[None] * W * pi[None] * G[:, None] / np.where(S_ > 0, S_, 1)[None] * (g - gbar[None])
    rb = np.bincount(m, r, K); a = int(R["agreedy"][s])
    sv = np.where(m == a, 1 / rb[a], 0.0) - np.where(A[m], 1 / rb[A].sum(), 0.0)
    Phi = np.linalg.solve((np.eye(N) - Lam).T, sv); psim = G * Phi
    k8 = an.colfree(op["chi"] * T * r[None] * (psim[:, None] - (T * psim[:, None]).sum(0)[None]), W)
    JL = Lam * pi[None]
    lam = (op["R"] if lif else op["rate"]) * v
    H = 100; kw, kb_ = np.zeros((N, N)), np.zeros((N, N))
    for b in range(10):
        stt = sim.fresh(1, H); X = np.empty((Tf, H, N))
        for t in range(Tf):
            ext, nz = sim.draws(rng, H, np.broadcast_to(lam, (H, N))); X[t] = sim.step(stt, ext, nz, None)[0][0]
        for (Y, acc) in ((X, kw), (X[burn:], kb_)):
            k = len(Y); c = Y.sum(0)
            C = np.einsum("thi,thj->ij", Y[1:], Y[:-1])
            acc += C - np.einsum("hi,hj->ij", c, c) * (k - 1) / k ** 2
    # long-run stationary: lag-1 covariance with long-run rates
    H2, T2 = 16, 20000; stt = sim.fresh(1, H2); C = np.zeros((N, N)); cnt = np.zeros(N); prev = None
    for t in range(300 + T2):
        ext, nz = sim.draws(rng, H2, np.broadcast_to(lam, (H2, N))); z = sim.step(stt, ext, nz, None)[0][0].astype(float)
        if t >= 300:
            if prev is not None: C += z.T @ prev
            cnt += z.sum(0)
        prev = z if t >= 300 else None
    rl = cnt / (T2 * H2); klr = C / ((T2 - 1) * H2) - np.outer(rl, rl)
    for name, kb in (("whole", kw), ("burn", kb_), ("longrun", klr)):
        crs = (m[:, None] != m[None]) & mk
        res.append((s, name, np.corrcoef(JL[mk], kb[mk])[0, 1], an.cos(-(g - gbar[None]) * kb, tgt, mk), an.cos(an.colfree(kb * Phi[:, None], W), k8, mk),
                    kb[mk & (W < 0.01)].mean() / (kb[mk].sum() / JL[mk].sum()), JL[mk & (W < 0.01)].mean()))
for x in res:
    print(f"{task} {wn} {opn} stim {x[0]} {x[1]:8s} corr(kappa,JL) {x[2]:.3f} rule cos {x[3]:.3f} kappa*Phi vs K8 {x[4]:.3f} | weak-syn kappa/zeta {x[5]:.2e} vs JL {x[6]:.2e}")
