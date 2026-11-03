"""R (external spikes per ms per unit v) giving mean rate 0.15/ms, deterministic LIF, init W, 4 held stimuli."""
import json, sys, numpy as np
D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D); import eng
def rate(net, chi, R):
    op = dict(kind="lif", chi=chi, w_in=4.0, R=R); sim = eng.Sim(net, op); rng = np.random.default_rng(0); out = []
    for v in net["held"][:4]:
        H = 16; st = sim.fresh(1, H); lam = np.broadcast_to(R * v, (H, net["N"])); c = 0
        for t in range(2500):
            ext, nz = sim.draws(rng, H, lam); z, _ = sim.step(st, ext, nz, None)
            if t >= 500: c = c + z[0].sum()
        out.append(c / (2000 * H * net["N"]))
    return float(np.mean(out))
cal = {}
for task in ["CartPole", "LunarLander"]:
    net = eng.network(task, False)
    for chi in [6.5, 10.4, 12.35]:
        lo, hi = 1.0, 200.0
        for _ in range(14):
            mid = np.sqrt(lo * hi)
            (lo, hi) = (mid, hi) if rate(net, chi, mid) < 0.15 else (lo, mid)
        R = float(np.sqrt(lo * hi)); cal[f"{task}_{chi}"] = R
        print(task, chi, R, rate(net, chi, R), flush=True)
json.dump(cal, open(D + "calib.json", "w"), indent=1)
