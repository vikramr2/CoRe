import pickle, sys, numpy as np
D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D); import eng, analyze as an
from mapstdp import core
for f in sys.argv[1:]:
    R = pickle.load(open(D + "out/" + f + ".pkl", "rb"))
    net = eng.network(R["task"], R["W"] == "trained"); W, mk, m, K, roles, N = net["W"], net["mask"], net["m"], net["K"], net["roles"], net["N"]
    op = R["op"]; B = eng.B_of(net, op); T = W / W.sum(0); cr = an.cross_ne(m, roles)
    s = 0; st = R["stat"][s]; r = st["r"]; G = st["G"][0]; v = R["V"][s]; Lam = G[:, None] * B
    pk = np.linalg.solve(np.eye(N) - Lam, G * eng.iext(net, op, v))
    f_ = lambda x: (np.bincount(m, x, K) / x.sum()).round(3)
    print(f, "\n  module p stat", f_(r), "\n  module p K3  ", f_(pk), "\n  flow(1)      ", f_(core.flow(W, v, 0.2)))
    print("  module mean G", (np.bincount(m, G, K) / np.bincount(m)).round(4), " module mean x", (np.bincount(m, st["x"], K) / np.bincount(m)).round(2))
    pi = r / r.sum(); Te, S_ = an.teff(W, G); g = an.mod_g(pi, Te * pi, m, K, cr); gbar = (Te * g).sum(0)
    gradK = pi[None] * G[:, None] / np.where(S_ > 0, S_, 1)[None] * (g - gbar[None]); nu = Lam.sum(0)
    tgt = -nu[None] * W * gradK
    for mode in ("reset", "persist"):
        kb = R[mode]["kap"][1][s] / R[mode]["n"][s]; dw = -(g - gbar[None]) * kb
        JL = Lam * pi[None]; crs = (m[:, None] != m[None, :]) & mk; wk = mk & (W < 0.01)
        print(f"  {mode}: cos all {an.cos(dw, tgt, mk):.3f} within {an.cos(dw, tgt, mk & ~crs):.3f} cross {an.cos(dw, tgt, crs):.3f}",
              f"| share of |tgt|^2 on weak (W<0.01) {np.sum(tgt[wk]**2)/np.sum(tgt[mk]**2):.2f}, on cross {np.sum(tgt[crs]**2)/np.sum(tgt[mk]**2):.2f}",
              f"| weak synapses: mean kappa/zeta {kb[wk].mean() / (kb[mk].sum()/JL[mk].sum()):.2e} vs mean JL {JL[wk].mean():.2e}, n={wk.sum()}")
