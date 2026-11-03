"""One (task, W, operating point) measurement. python run.py TASK {init,trained} OP -> out/TASK_W_OP.pkl"""
import json
import zlib
import pickle
import sys
import time

import numpy as np

D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D)
import eng  # noqa: E402
from mapstdp import core  # noqa: E402

S_STIM, WINS = 4, (1, 5, 20)


def ops(task):
    cal = json.load(open(D + "calib.json"))
    R = lambda chi: cal[f"{task}_{chi}"]  # noqa: E731
    base = dict(kind="lif", w_in=4.0)
    return {
        "branch": dict(kind="branch", chi=0.8, rate=1.0, R=1.0, w_in=1.0),
        "lif_det": dict(base, chi=10.4, R=R(10.4)),
        "lif_s1": dict(base, chi=10.4, R=R(10.4), sigma=1.0),
        "lif_s2": dict(base, chi=10.4, R=R(10.4), sigma=2.0),
        "lif_s4": dict(base, chi=10.4, R=R(10.4), sigma=4.0),
        "lif_esc2": dict(base, chi=10.4, R=R(10.4), Delta=2.0),
        "lif_chi6.5": dict(base, chi=6.5, R=R(10.4)),
        "lif_chi12.35": dict(base, chi=12.35, R=R(10.4)),
    }


def frame_stats(X, net, a_greedy, W, Wins=WINS):
    """X: (k, H, N) bool rasters of H frames with stimulus index sid (H,). Returns per-head dict of arrays."""
    k, H, N = X.shape
    m, roles, K = net["m"], net["roles"], net["K"]
    Xf = X.astype(np.float64)
    c = Xf.sum(0)  # H x N
    cs = np.concatenate([np.zeros((1, H, N)), np.cumsum(Xf, 0)])
    kap = {}
    for L0 in Wins:
        L = min(L0, k - 1)
        # Y_t = sum_{lag=1..L} X_{t-lag}
        idx = np.arange(k)
        Y = cs[idx] - cs[np.maximum(idx - L, 0)]
        C = np.einsum("thi,thj->hij", Xf, Y)
        wsum = sum(k - lag for lag in range(1, L + 1))
        kap[L0] = C - c[:, :, None] * c[:, None, :] * wsum / k ** 2
    tot = c.sum(1)
    pih = c / np.maximum(tot, 1)[:, None]
    T = W / W.sum(0)
    H1 = np.eye(K)[m]
    out = dict(c=c, pih=pih, kap=kap, tot=tot, dw={}, e={}, ok=np.ones(H, bool))
    Gc = np.zeros((H, N, N))
    Ec = np.zeros((H, N, N))
    for h in range(H):
        p, q, JK, _ = core.module_stats(W, pih[h], m, K)
        g = core.modulator(p, q, roles, routing="exempt")
        Gc[h] = core.centered(g, W, m)
        Jin = core.action_inflow(W, pih[h], m, K)
        a = a_greedy[h]
        if (Jin[roles == "A"] <= 0).any():
            out["ok"][h] = False
            continue
        Ec[h] = core.centered(core.action_score(W, pih[h], m, roles, a), W, m)
    for L in kap:
        n = np.maximum(tot, 1)[:, None, None]
        out["dw"][L] = -Gc * kap[L] / n
        out["e"][L] = Ec * kap[L] / n
    return out


class Acc:
    def __init__(self, S, N, K):
        z = lambda: np.zeros((S, N, N))  # noqa: E731
        self.n = np.zeros(S)
        self.nok = np.zeros(S)
        self.c, self.pih, self.pih2 = np.zeros((S, N)), np.zeros((S, N)), np.zeros((S, N))
        self.pmod, self.pmod2 = np.zeros((S, K)), np.zeros((S, K))
        self.kap = {L: z() for L in WINS}
        self.kap_h = {L: [z(), z()] for L in WINS}
        self.dw = {L: z() for L in WINS}
        self.e = {L: z() for L in WINS}
        self.spk = np.zeros(S)

    def add(self, st, sid, m, K):
        for h, s in enumerate(sid):
            self.n[s] += 1
            self.c[s] += st["c"][h]
            self.pih[s] += st["pih"][h]
            self.pih2[s] += st["pih"][h] ** 2
            pm = np.bincount(m, st["pih"][h], K)
            self.pmod[s] += pm
            self.pmod2[s] += pm ** 2
            self.spk[s] += st["tot"][h]
            half = int(self.n[s]) % 2
            for L in WINS:
                self.kap[L][s] += st["kap"][L][h]
                self.kap_h[L][half][s] += st["kap"][L][h]
                self.dw[L][s] += st["dw"][L][h]
                if st["ok"][h]:
                    self.e[L][s] += st["e"][L][h]
            self.nok[s] += st["ok"][h]


def main():
    task, wname, opname = sys.argv[1], sys.argv[2], sys.argv[3]
    quick = len(sys.argv) > 4
    t0 = time.time()
    net = eng.network(task, wname == "trained")
    op = ops(task)[opname]
    N, K, m, roles, W = net["N"], net["K"], net["m"], net["roles"], net["W"]
    Vall = np.array(net["held"])
    tvv = 0.5 * np.abs(Vall[:, None] - Vall[None]).sum(-1)
    sel = [int(np.argmax(tvv.sum(1)))]
    while len(sel) < S_STIM:
        sel.append(int(np.argmax(tvv[:, sel].min(1))))
    V = Vall[sel]
    rng = np.random.default_rng(zlib.crc32(f"{task}{wname}{opname}".encode()))
    lif = op["kind"] == "lif"
    Tst = 6000 if quick else 60000
    # 1. stationary per stimulus
    stat = []
    for s in range(S_STIM):
        stat.append(eng.stationary(net, op, V[s], rng, H=8, T=Tst, deltas=(0.0, 0.05, 0.2), perturb_R=True,
                                   Hr=8, Tr=Tst))
        print(f"stat {s} {time.time() - t0:.0f}s rate {stat[-1]['r'].mean():.4f}", flush=True)
    A = np.flatnonzero(roles == "A")
    agreedy = []
    for s in range(S_STIM):
        pm = np.bincount(m, stat[s]["r"], K)
        agreedy.append(int(A[np.argmax(pm[A])]))
    agreedy = np.array(agreedy)
    sim = eng.Sim(net, op)
    res = dict(op=op, stat=stat, agreedy=agreedy, task=task, W=wname, V=V)
    # 2. reset frames
    Tf_reset = 200 if lif else 100
    Hb, nb = (100, 2 if quick else 20)
    acc = Acc(S_STIM, N, K)
    for s in range(S_STIM):
        lam = (op["R"] * V[s] if lif else op["rate"] * V[s])[None].repeat(Hb, 0)
        for b in range(nb):
            st = sim.fresh(1, Hb)
            X = np.empty((Tf_reset, Hb, N), bool)
            for t in range(Tf_reset):
                ext, nz = sim.draws(rng, Hb, lam)
                X[t] = sim.step(st, ext, nz, None)[0][0]
            acc.add(frame_stats(X, net, np.full(Hb, agreedy[s]), W), np.full(Hb, s), m, K)
    res["reset"] = acc.__dict__
    res["Tf_reset"] = Tf_reset
    print(f"reset done {time.time() - t0:.0f}s", flush=True)
    # 3. persistent 20-step frames, iid switching among stimuli
    Hp, nf = 64, (100 if quick else 1000)
    acc = Acc(S_STIM, N, K)
    st = sim.fresh(1, Hp)
    sid = rng.integers(S_STIM, size=Hp)
    for f in range(-20, nf):
        lam = (op["R"] if lif else op["rate"]) * V[sid]
        X = np.empty((20, Hp, N), bool)
        for t in range(20):
            ext, nz = sim.draws(rng, Hp, lam)
            X[t] = sim.step(st, ext, nz, None)[0][0]
        if f >= 0:
            acc.add(frame_stats(X, net, agreedy[sid], W), sid, m, K)
        sid = rng.integers(S_STIM, size=Hp)
    res["persist"] = acc.__dict__
    print(f"persist done {time.time() - t0:.0f}s", flush=True)
    # 4. relaxation: switch s -> s+1 from stationary, and cold start at s+1
    Hr, Tw, Tm = (400 if quick else 2000), 400, 300
    relax = []
    for s in range(S_STIM):
        b = (s + 1) % S_STIM
        curves = {}
        for mode in ("switch", "cold"):
            st = sim.fresh(1, Hr)
            if mode == "switch":
                lamA = ((op["R"] if lif else op["rate"]) * V[s])[None].repeat(Hr, 0)
                for t in range(Tw):
                    ext, nz = sim.draws(rng, Hr, lamA)
                    sim.step(st, ext, nz, None)
            lamB = ((op["R"] if lif else op["rate"]) * V[b])[None].repeat(Hr, 0)
            cnt = np.zeros((Tm, N))
            for t in range(Tm):
                ext, nz = sim.draws(rng, Hr, lamB)
                cnt[t] = sim.step(st, ext, nz, None)[0][0].sum(0)
            curves[mode] = cnt
        relax.append(dict(a=s, b=b, **curves))
    res["relax"] = relax
    pickle.dump(res, open(f"{D}out/{task}_{wname}_{opname}{'_quick' if quick else ''}.pkl", "wb"))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
