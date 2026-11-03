import pickle, sys, numpy as np
D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D); import eng
for f in sys.argv[1:]:
    R = pickle.load(open(D + "out/" + f + ".pkl", "rb"))
    net = eng.network(R["task"], R["W"] == "trained"); W, mk, m = net["W"], net["mask"], net["m"]
    op = R["op"]; B = eng.B_of(net, op)
    out = []
    for s in range(4):
        st = R["stat"][s]; r = st["r"]; G = st["G"][0]; pi = r / r.sum()
        JL = G[:, None] * B * pi[None, :]
        for mode in ("reset", "persist"):
            kb = R[mode]["kap"][1][s] / R[mode]["n"][s]
            # fit kappa_ij ~ a_i b_j JL_ij (rank-1 multiplicative correction), alternating LS in log-free form
            a = np.ones(len(W)); b = np.ones(len(W))
            for _ in range(50):
                a = (kb * JL * b[None]).sum(1) / np.maximum((JL * b[None]) ** 2 * mk, 0).sum(1).clip(1e-30)
                b = (kb * JL * a[:, None]).sum(0) / ((JL * a[:, None]) ** 2 * mk).sum(0).clip(1e-30)
            fit_ab = (a[:, None] * b[None] * JL)[mk]
            fit_a = None
            a1 = (kb * JL).sum(1) / (JL ** 2 * mk).sum(1).clip(1e-30)
            b1 = (kb * JL).sum(0) / (JL ** 2 * mk).sum(0).clip(1e-30)
            c0 = np.corrcoef(JL[mk], kb[mk])[0, 1]
            ca = np.corrcoef((a1[:, None] * JL)[mk], kb[mk])[0, 1]
            cb = np.corrcoef((b1[None] * JL)[mk], kb[mk])[0, 1]
            cab = np.corrcoef(fit_ab, kb[mk])[0, 1]
            act = r > 0.005
            cva = a1[act].std() / a1[act].mean(); cvb = b1[act].std() / b1[act].mean()
            ra = np.corrcoef(a1[act], G[act])[0, 1]; rb = np.corrcoef(b1[act], st["fano"][act])[0, 1]
            rbr = np.corrcoef(b1[act], r[act])[0, 1]; rar = np.corrcoef(a1[act], r[act])[0, 1]
            out.append((mode, c0, ca, cb, cab, cva, cvb, ra, rar, rb, rbr))
    for mode in ("reset", "persist"):
        o = np.array([x[1:] for x in out if x[0] == mode]).mean(0)
        print(f"{f:28s} {mode:7s} corr direct {o[0]:.3f} +post factor {o[1]:.3f} +pre factor {o[2]:.3f} +both {o[3]:.3f} | CV post {o[4]:.2f} pre {o[5]:.2f} | corr(post f, G) {o[6]:.2f} (post f, r) {o[7]:.2f} corr(pre f, Fano) {o[8]:.2f} (pre f, r) {o[9]:.2f}")
