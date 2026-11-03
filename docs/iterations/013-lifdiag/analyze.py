"""Checks 1-6 of PREREG.md for out/*.pkl. python analyze.py [files...] -> prints rows, writes out/metrics.json"""
import glob
import json
import pickle
import sys

import numpy as np

D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
sys.path.insert(0, D)
import eng  # noqa: E402
from mapstdp import core  # noqa: E402

PRIM = 1


def xlx(x):
    x = np.asarray(x, float)
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1)), 0.0)


def ex_table(roles):
    return core.route_pairs(roles)


def cross_ne(m, roles):
    ex = ex_table(roles)
    return (m[:, None] != m[None, :]) & ~ex[m][:, m]


def mapD(pi, J, m, K, cr):
    q = np.bincount(m, (J * cr).sum(0), K)
    p = np.bincount(m, pi, K)
    return float(xlx(q.sum()) - 2 * xlx(q).sum() - xlx(pi).sum() + xlx(p + q).sum())


def mod_g(pi, J, m, K, cr):
    q = np.bincount(m, (J * cr).sum(0), K)
    p = np.bincount(m, pi, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ms = np.where(q > 0, np.log(q.sum() * (p + q) / np.where(q > 0, q, 1) ** 2), 0.0)
    return Ms[m][None, :] * cr


def teff(W, G):
    GW = G[:, None] * W
    S = GW.sum(0)
    return GW / np.where(S > 0, S, 1), S


def cos(a, b, mk):
    a, b = a[mk], b[mk]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / na / nb) if na > 0 and nb > 0 else np.nan


def colfree(X, W):
    return X - W * X.sum(0) / W.sum(0)


def fd(f, W, mk, h=1e-6):
    g = np.zeros_like(W)
    for i, j in zip(*np.nonzero(mk)):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += h
        Wm[i, j] -= h
        g[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return g


def tv(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def analyze(path, do_fd=True):
    R = pickle.load(open(path, "rb"))
    task, wname, op = R["task"], R["W"], R["op"]
    net = eng.network(task, wname == "trained")
    W, m, roles, mk, K, N = net["W"], net["m"], net["roles"], net["mask"], net["K"], net["N"]
    T = W / W.sum(0)
    chi = op["chi"]
    B = chi * T
    cr = cross_ne(m, roles)
    A = roles == "A"
    out = dict(task=task, W=wname, op=path.split("_", 2)[-1][:-4])
    per = []
    for s, st in enumerate(R["stat"]):
        v = R["V"][s]
        r = st["r"]
        G = st["G"][0]
        act = r > 0.005 * (1 if op["kind"] == "lif" else 1)
        Lam = G[:, None] * B
        nu = Lam.sum(0)
        Iext = eng.iext(net, op, v)
        pi = r / r.sum()
        piK3 = np.linalg.solve(np.eye(N) - Lam, G * Iext)
        pi1 = core.flow(W, v, 0.2)
        mu = r - Lam @ r
        Rm = st["R"]
        Rmod = np.linalg.solve(np.eye(N) - Lam, np.diag(G))
        d = dict(rate=r.mean(), rate_max=r.max(), silent=float((r < 0.005).mean()),
                 G_mean=G[act].mean(), G_cv=G[act].std() / G[act].mean(),
                 G_nonlin=float(np.mean(st["G"][1][act]) / np.mean(G[act])),
                 kick_ratio=float(np.mean(st["kickG"][act]) / np.mean(G[act])),
                 rho=float(np.abs(np.linalg.eigvals(Lam)).max()), nu_mean=float(nu[act].mean()),
                 fano=float(np.median(st["fano"][act])), sync=float(st["sync"].max()),
                 base_err=st["base_err"],
                 mu_neg=float(-mu[mu < 0].sum() / np.abs(mu).sum()),
                 mu_corr=float(np.corrcoef(mu, G * Iext)[0, 1]),
                 tv_stat_K3=tv(r, piK3), tv_stat_1=tv(r, pi1),
                 K2_relerr=float(np.linalg.norm(Rm - Rmod) / np.linalg.norm(Rmod)),
                 K2_corr=float(np.corrcoef(Rm.ravel(), Rmod.ravel())[0, 1]),
                 R_rel=float(np.corrcoef(st["R_halves"][0].ravel(), st["R_halves"][1].ravel())[0, 1]))
        # targets for check 4
        Te, S_ = teff(W, G)
        Je = Te * pi
        g = mod_g(pi, Je, m, K, cr)
        gbar = (Te * g).sum(0)
        gradK = (pi * 1.0)[None, :] * G[:, None] / np.where(S_ > 0, S_, 1)[None, :] * (g - gbar[None, :])
        if do_fd:
            fK = lambda W_: mapD(pi, teff(W_, G)[0] * pi, m, K, cr)  # noqa: E731
            gfd = fd(fK, W, mk)
            d["lemmaK_err"] = float(np.abs(gfd - gradK)[mk].max() / np.abs(gradK[mk]).max())
            Iv = G * Iext

            def ffull(W_):
                Bw = chi * W_ / W_.sum(0)
                rr = np.linalg.solve(np.eye(N) - G[:, None] * Bw, Iv)
                p_ = rr / rr.sum()
                return mapD(p_, teff(W_, G)[0] * p_, m, K, cr)
            gfull = fd(ffull, W, mk, 1e-6)
        else:
            gfd, gfull = gradK, gradK
        tgtK = -nu[None, :] * W * gfd
        tgtF = -nu[None, :] * W * gfull
        # implemented Eq. 4 target: G = 1, J = T pi
        g1 = mod_g(pi, T * pi, m, K, cr)
        grad1 = (pi / W.sum(0))[None, :] * (g1 - (T * g1).sum(0)[None, :])
        tgt1 = -W * grad1
        # score
        rb = np.bincount(m, r, K)
        a = int(R["agreedy"][s])
        sv = np.where(m == a, 1 / rb[a], 0.0) - np.where(A[m], 1 / rb[A].sum(), 0.0)
        psi = Rm.T @ sv
        exact = chi * T * r[None, :] * (psi[:, None] - (T * psi[:, None]).sum(0)[None, :])
        exh = [chi * T * r[None, :] * ((h_.T @ sv)[:, None] - (T * (h_.T @ sv)[:, None]).sum(0)[None, :])
               for h_ in st["R_halves"]]
        Phi = np.linalg.solve((np.eye(N) - Lam).T, sv)
        psim = G * Phi
        k8 = chi * T * r[None, :] * (psim[:, None] - (T * psim[:, None]).sum(0)[None, :])
        H1 = np.eye(K)[m]
        Lpop = (H1.T @ Lam @ H1) / H1.sum(0)[None, :]
        sK = np.where(np.arange(K) == a, 1 / rb[a], 0.0) - np.where(A, 1 / rb[A].sum(), 0.0)
        PhiK = np.linalg.solve((np.eye(K) - Lpop).T, sK)
        exf = colfree(exact, W)
        d["score_rel"] = cos(colfree(exh[0], W), colfree(exh[1], W), mk)
        d["score_K8_vs_exact"] = cos(colfree(k8, W), exf, mk)
        for mode in ("reset", "persist"):
            ac = R[mode]
            n = ac["n"][s]
            pbar = ac["pih"][s] / n
            pm = ac["pmod"][s] / n
            pm_sd = np.sqrt(np.maximum(ac["pmod2"][s] / n - pm ** 2, 0))
            pmK3 = np.bincount(m, piK3 / piK3.sum(), K)
            d[f"{mode}_tv_K3"] = tv(pbar, piK3)
            d[f"{mode}_tv_1"] = tv(pbar, pi1)
            d[f"{mode}_tv_stat"] = tv(pbar, r)
            d[f"{mode}_modbias"] = float(np.sqrt(np.mean((pm - pmK3) ** 2)))
            d[f"{mode}_modsd"] = float(np.sqrt(np.mean(pm_sd ** 2)))
            d[f"{mode}_spk"] = ac["spk"][s] / n
            JL = Lam * pi[None, :]
            for L in (1, 5, 20):
                kb = ac["kap"][L][s] / n
                x, y = JL[mk], kb[mk]
                zeta = y.sum() / x.sum()
                d[f"{mode}_k{L}_corr"] = float(np.corrcoef(x, y)[0, 1])
                d[f"{mode}_k{L}_slope"] = float(np.polyfit(x, y / zeta, 1)[0])
                d[f"{mode}_k{L}_rel"] = float(np.corrcoef(ac["kap_h"][L][0][s][mk], ac["kap_h"][L][1][s][mk])[0, 1])
                xf = (Lam * (pi * st["fano"])[None, :])[mk]
                d[f"{mode}_k{L}_corrF"] = float(np.corrcoef(xf, y)[0, 1])
                ch = ((Lam @ Lam) * pi[None, :])[mk]
                ci = ((Lam * pi[None, :]) @ Lam.T)[mk]
                X3 = np.stack([x, ch, ci, np.ones_like(x)], 1)
                res3 = np.linalg.lstsq(X3, y, rcond=None)
                yhat = X3 @ res3[0]
                d[f"{mode}_k{L}_R2direct"] = float(np.corrcoef(x, y)[0, 1] ** 2)
                d[f"{mode}_k{L}_R2all"] = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
                dwK = -(g - gbar[None, :]) * kb
                d[f"{mode}_k{L}_rule"] = cos(dwK, tgtK, mk)
                kp = np.where(mk, kb, 0.0)
                gb_emp = (kp * g).sum(0) / np.where(np.abs(kp.sum(0)) > 0, kp.sum(0), 1)
                d[f"{mode}_k{L}_rule_pb"] = cos(-(g - gb_emp[None, :]) * kb, tgtK, mk)
                d[f"{mode}_k{L}_rulefull"] = cos(dwK, tgtF, mk)
                d[f"{mode}_k{L}_ruleimpl"] = cos(ac["dw"][L][s] / n, tgt1, mk)
                e_impl = ac["e"][L][s] / max(ac["nok"][s], 1)
                d[f"{mode}_k{L}_s4b"] = cos(colfree(e_impl, W), exf, mk)
                kh0, kh1 = ac["kap_h"][L][0][s], ac["kap_h"][L][1][s]
                rr = cos(-(g - gbar[None, :]) * kh0, -(g - gbar[None, :]) * kh1, mk)
                d[f"{mode}_k{L}_rule_rel"] = rr
                d[f"{mode}_k{L}_rule_corr"] = d[f"{mode}_k{L}_rule"] / np.sqrt(max(2 * rr / (1 + rr), 1e-9)) if rr > 0 else np.nan
                rp = cos(colfree(kh0 * Phi[:, None], W), colfree(kh1 * Phi[:, None], W), mk)
                d[f"{mode}_k{L}_sPhi_rel"] = rp
                d[f"{mode}_k{L}_sPhi_vsK8"] = cos(colfree(kb * Phi[:, None], W), colfree(k8, W), mk)
                d[f"{mode}_k{L}_sPhiK_vsK8"] = cos(colfree(kb * PhiK[m][:, None], W), colfree(k8, W), mk)
                hst = core.centered(core.action_score(W, pi, m, roles, a), W, m)
                d[f"{mode}_k{L}_s4bstat"] = cos(colfree(kb * hst, W), exf, mk)
                d[f"{mode}_k{L}_s4b_vsK8"] = cos(colfree(e_impl, W), colfree(k8, W), mk)
                d[f"{mode}_k{L}_sPhi"] = cos(colfree(kb * Phi[:, None], W), exf, mk)
                d[f"{mode}_k{L}_sPhiK"] = cos(colfree(kb * PhiK[m][:, None], W), exf, mk)
        # relaxation
        rl = R["relax"][s]
        b = rl["b"]
        rB = R["stat"][b]["r"]
        for mode in ("switch", "cold"):
            c = rl[mode]
            cs = np.array([c[max(t - 1, 0):t + 2].sum(0) for t in range(len(c))])
            tvs = np.array([tv(x + 1e-12, rB) for x in cs])
            floor = float(np.median(tvs[-100:]))
            hit = np.flatnonzero(tvs <= 0.05)
            d[f"t95_{mode}"] = float(hit[0] + 1) if len(hit) else np.inf
            d[f"floor_{mode}"] = floor
            d[f"tv20_{mode}"] = tv(c[:20].sum(0) + 1e-12, rB)
            nc = roles[m] != "C"
            tvn = np.array([tv(x[nc] + 1e-12, rB[nc]) for x in cs])
            hit = np.flatnonzero(tvn <= 0.05)
            d[f"t95nc_{mode}"] = float(hit[0] + 1) if len(hit) else np.inf
            d[f"floornc_{mode}"] = float(np.median(tvn[-100:]))
            d[f"tvAB_nc"] = tv(R["stat"][rl["a"]]["r"][nc], rB[nc])
            d[f"tvAB"] = tv(R["stat"][rl["a"]]["r"], rB)
        per.append(d)
    keys = per[0].keys()
    for k in keys:
        out[k] = float(np.mean([p[k] for p in per]))
        out[k + "_max"] = float(np.max([p[k] for p in per]))
        out[k + "_min"] = float(np.min([p[k] for p in per]))
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(D + "out/*.pkl"))
    res = []
    for f in files:
        o = analyze(f, do_fd="--nofd" not in f)
        res.append(o)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()
                          if not k.endswith(("_max", "_min"))}), flush=True)
    tag = "metrics" if len(sys.argv) == 1 else "metrics_partial"
    json.dump(res, open(D + f"out/{tag}.json", "w"), indent=1)
