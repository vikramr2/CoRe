import sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); import eng
for task in ["CartPole", "LunarLander"]:
    net = eng.network(task, False)
    v = net["held"][0]
    print(task, "vmax", v.max(), "N", net["N"])
    for R in [5, 10, 20]:
        op = dict(kind="lif", chi=10.4, w_in=4.0, R=R)
        sim = eng.Sim(net, op); rng = np.random.default_rng(0); H=8
        st = sim.fresh(1, H); cnt = 0; lam = np.broadcast_to(R * v, (H, net["N"]))
        for t in range(1500):
            ext, nz = sim.draws(rng, H, lam); z, _ = sim.step(st, ext, nz, None)
            if t >= 500: cnt = cnt + z[0]
        r = cnt.sum(0) / (1000 * H)
        print(" R", R, "mean rate", r.mean().round(4), "per module", np.bincount(net["m"], r, net["K"]) / np.bincount(net["m"]), "max", r.max().round(3), "silent<0.005", (r < 0.005).mean().round(2))
