import sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); import eng
net = eng.network("CartPole", False); v = net["held"][0]; N = net["N"]
op = dict(kind="lif", chi=10.4, w_in=4.0, R=21.0)
t0 = time.time()
o = eng.stationary(net, op, v, np.random.default_rng(1), H=8, T=5000, perturb_R=True, Hr=8, Tr=5000, deltas=(0, 0.05, 0.2))
print("time", time.time() - t0, "base_err", o["base_err"])
r, G = o["r"], o["G"]
act = r > 0.005
print("rate mean", r.mean(), "G(0.05) mean/cv", G[0][act].mean(), G[0][act].std()/G[0][act].mean(), "G(0.2)", G[1][act].mean())
print("1/13 =", 1/13, "kickG mean", o["kickG"][act].mean(), "ksz", o["ksz"], "fano mean", o["fano"][act].mean(), "sync", o["sync"].round(2))
B = eng.B_of(net, op); L = G[0][:, None] * B
print("rho(Lambda)", np.abs(np.linalg.eigvals(L)).max())
Rm = o["R"]; h0, h1 = o["R_halves"]
Rp = np.linalg.solve(np.eye(N) - L, np.diag(G[0]))
mk = net["mask"]
print("R split-half corr all", np.corrcoef(h0.ravel(), h1.ravel())[0,1], "diag", np.corrcoef(np.diag(h0), np.diag(h1))[0,1])
print("R vs model corr", np.corrcoef(Rm.ravel(), Rp.ravel())[0,1], "rel err", np.linalg.norm(Rm-Rp)/np.linalg.norm(Rp))
print("diag R mean", np.diag(Rm).mean(), "model", np.diag(Rp).mean())
