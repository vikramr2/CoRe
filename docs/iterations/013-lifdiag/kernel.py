"""Open-loop single-kick response kernel h(s): extra spikes of a neuron at lag s after one extra recurrent-size kick,
on its recorded stationary input (CRN replay). python kernel.py TASK W OP -> out/kernel_TASK_W_OP.json"""
import json
import sys

import numpy as np

D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D)
import eng  # noqa: E402
import run  # noqa: E402

task, wname, opname = sys.argv[1:4]
net = eng.network(task, wname == "trained")
op = run.ops(task)[opname]
N, H, T, P, LAG = net["N"], 8, 40000, 200, 150
v = np.array(net["held"])[65 if task == "CartPole" else 98]
rng = np.random.default_rng(7)
sim = eng.Sim(net, op)
lam = np.broadcast_to((op["R"] if op["kind"] == "lif" else op["rate"]) * v, (H, N))
st = sim.fresh(1, H)
for _ in range(300):
    ext, nz = sim.draws(rng, H, lam)
    sim.step(st, ext, nz, None)
v0 = st["v"][0].copy()
Irec = np.empty((T, H, N), np.float32)
NZ = np.empty((T, H, N), np.float32)
for t in range(T):
    ext, nz = sim.draws(rng, H, lam)
    z, inp = sim.step(st, ext, nz, None)
    Irec[t] = inp[0]
    NZ[t] = 0 if nz is None else nz
if op["kind"] == "lif" and not op.get("Delta") and not op.get("sigma"):
    NZ = None
B = eng.B_of(net, op)
ksz = float(np.median(B[B > 0]))
# kicks at t0 = P k + jitter (independent per head/neuron), one per period
kick = np.zeros((T, H, N), np.float32)
off = rng.integers(0, P - LAG, (T // P, H, N))
for k_ in range(T // P - 1):
    t0 = k_ * P + off[k_]
    for h in range(H):
        kick[t0[h], h, np.arange(N)] = ksz


def raster(kk):
    vv = np.broadcast_to(v0, (H, N)).copy()
    S = np.zeros((T, H, N), bool)
    for t in range(T):
        inp = Irec[t].astype(float) + (0 if kk is None else kk[t])
        if op["kind"] == "lif":
            vv = eng.A_LEAK * vv + inp
            if op.get("Delta"):
                z = NZ[t] < 1 - np.exp(-np.exp(np.minimum((vv - eng.TH) / op["Delta"], 30)))
            else:
                if NZ is not None:
                    vv = vv + NZ[t]
                z = vv >= eng.TH
            vv = np.where(z, 0.0, vv)
        else:
            z = (NZ[t] + inp) > 0.5
        S[t] = z
    return S


S0, S1 = raster(None), raster(kick)
dS = S1.astype(float) - S0
h = np.zeros(LAG)
nk = 0
r = S0.mean((0, 1))
act = r > 0.005
for k_ in range(T // P - 1):
    t0 = k_ * P + off[k_]
    for lag in range(LAG):
        h[lag] += dS[t0 + lag, np.arange(H)[:, None], np.arange(N)[None, :]][:, act].sum()
    nk += H * act.sum()
h /= nk  # extra spikes per kick at lag s (lag 0 = same step as the kick arrives)
tot = h.sum()
cum = np.cumsum(h) / tot
out = dict(task=task, W=wname, op=opname, ksz=ksz, h=h.tolist(), total=tot, gain_per_mV=tot / ksz,
           omega={L: float(cum[L]) for L in (0, 1, 4, 19, 49, 99)}, median_lag=float(np.searchsorted(cum, 0.5)),
           neg_mass=float(-h[h < 0].sum() / np.abs(h).sum()), rate=float(r[act].mean()),
           pos=float(h[h > 0].sum()), neg=float(-h[h < 0].sum()), lag0=float(h[0]),
           mean_lag_pos=float((np.arange(LAG) * np.maximum(h, 0)).sum() / np.maximum(h, 0).sum()))
json.dump(out, open(f"{D}out/kernel_{task}_{wname}_{opname}.json", "w"))
print({k: v for k, v in out.items() if k != "h"})
