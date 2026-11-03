"""Summaries and diagnostics of models/basic.py runs: the M2 and M3 tables, the ablation table, the
learning-signal fidelity check and the iteration 012 diagnostics."""
import glob
import json
import pathlib

import gymnasium as gym
import numpy as np

from mapstdp import core, qif
from mapstdp import spiking as sp
from models.helpers.agent import ALPHA, ETA, GAMMA, K_SPK, N_HOPS, OUT, QIF_SEED, RATE, TAU_E, THETA, Agent, Config
from models.helpers.basic_cli import HEADS, LA_BETA
from models.helpers.tasks import TASKS


def random_return(episodes=1000, seed=0, task=TASKS["CartPole"]):
    """Mean and sd of the uniform-random-policy return (the --summary baseline)."""
    env, rng, R = gym.make(task.env_id), np.random.default_rng(seed), []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 30)))
        n, done = 0, False
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(task.n_actions)))
            n, done = n + r, term or trunc
        R.append(n)
    return float(np.mean(R)), float(np.std(R))


def r2_cv(X, y, ridge=1e-3):
    """Two-fold cross-validated R^2 of a linear (ridge on z-scored X, with intercept) fit of y on X (fixed split)."""
    idx = np.random.default_rng(0).permutation(len(y))
    X1 = np.c_[np.ones(len(y)), (X - X.mean(0)) / (X.std(0) + 1e-12)]
    pred = np.zeros(len(y))
    for tr, te in ((idx[::2], idx[1::2]), (idx[1::2], idx[::2])):
        A = X1[tr].T @ X1[tr] + ridge * len(tr) * np.eye(X1.shape[1])
        pred[te] = X1[te] @ np.linalg.solve(A, X1[tr].T @ y[tr])
    return float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300))


def diag012(agent, n=2000):
    """Iteration-012 end-of-run diagnostics on up to n observations from the last REC_EPIS episodes (never touches
    agent.rng or W): controller density (share of controller neurons with v_i > 10% of max v), the critic's V on those
    states, and how well V decodes linearly from module flow shares: p_b over latent and action modules (R2_LA), the
    controller-share features of the default critic (R2_ctrl, the V of a pi_C critic) and neuron-level L/A shares
    (R2_LA_neurons, ridge)."""
    obs = agent.rec or []
    obs = [obs[i] for i in np.linspace(0, len(obs) - 1, min(n, len(obs))).astype(int)] if obs else []
    agent.rec = None
    if len(obs) < 20:
        return {}
    LA, nLA = np.isin(agent.roles, ["L", "A"]), np.isin(agent.roles[agent.m], ["L", "A"])
    dens, P, PN, PC, V = [], [], [], [], []
    for o in obs:
        v = agent.stim(o)
        vc = v[agent.ctrl]
        dens.append(float((vc > 0.1 * vc.max()).mean()))
        pi = core.flow(agent.W, v, ALPHA, N_HOPS)
        P.append(np.bincount(agent.m, pi, agent.K)[LA])
        PN.append(pi[nLA])
        PC.append(pi[agent.ctrl] / pi[agent.ctrl].sum())
        V.append(float(agent.u @ agent.feats(pi, o)))
    V = np.array(V)
    held = [float((v[agent.ctrl] > 0.1 * v[agent.ctrl].max()).mean()) for v in agent.held]
    return dict(n=len(obs), density=float(np.mean(dens)), density_held=float(np.mean(held)), V_sd=float(V.std()),
                R2_LA=r2_cv(np.array(P), V), R2_ctrl=r2_cv(np.array(PC), V), R2_LA_neurons=r2_cv(np.array(PN), V, 1e-2))


def summary_m2():
    """Results table over seeds of every m2_*.json in OUT (v1 and v2 files included), grouped by model, backend,
    routing, lam and variant."""
    rand = random_return()
    print(f"random policy: {rand[0]:.1f} +- {rand[1]:.1f} (1000 episodes)")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "m2_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        var = f"b{r.get('beta', 1.0):g}" + (f",lb{r['la_beta']:g}" if r.get("la_beta") else "")
        if r.get("eta_tau") is not None and r["model"] != "v3":
            var += f",eta{r['eta_task']:g}" + (f",at{r['eta_tau']:g}" if r["eta_tau"] else "")
        groups.setdefault((r.get("model", "v1"), r["backend"], r["routing"], r["lam"], var), []).append(r)
    cols = ("D", "J_route", "p_A", "persist_LA", "persist_C", "d_max", "KL_1d_race", "KL_floor")
    print("return: mean +- sd over seeds of per-seed mean return; metrics: seed-mean at episode 0 > final eval (sd)")
    print(f"{'model':5} {'backend':8} {'routing':7} {'lam':>5} {'variant':>14} {'n':>2} {'epis':>5} "
          f"{'ret first100':>13} {'ret last100':>13} " + " ".join(f"{c:>18}" for c in cols))
    ms = lambda x: f"{np.mean(x):6.1f}+-{np.std(x):5.1f}"  # noqa: E731
    for (mo, b, ro, lam, be), rs in sorted(groups.items()):
        line = f"{mo:5} {b:8} {ro:7} {lam:5g} {be:>14} {len(rs):2} {min(r['episodes'] for r in rs):5} " \
               f"{ms([np.mean(r['returns'][:100]) for r in rs])} {ms([np.mean(r['returns'][-100:]) for r in rs])}"
        for c in cols:
            if c in rs[0]["evals"][-1]:
                s, e = [r["evals"][0][c] for r in rs], [r["evals"][-1][c] for r in rs]
                line += f" {np.mean(s):5.3f}>{np.mean(e):5.3f}({np.std(e):.3f})"
        print(line)


def row_corr(K, J, mask):
    """Mean over postsynaptic rows (>= 3 synapses, non-constant) of corr(K[i], J[i]) on the mask: insensitive to a
    per-postsynaptic gain."""
    c = [np.corrcoef(K[i, mask[i]], J[i, mask[i]])[0, 1] for i in range(len(K)) if mask[i].sum() >= 3]
    return float(np.nanmean(c))


def fidelity(task, Hs=HEADS, reps=20, n_obs=10, seed=0, engine="numpy", qp=None):
    """Learning-signal error of H pooled heads against the flow on the initial W, over held-out stimuli: L1 of pi-hat
    to the n-hop flow, and 1 - corr of kappa with J (the flow's expected pairings), mean over reps frames.
    engine='qif' (qp: qif.QIFParams): warm frames after one warm-up frame per stimulus (cold if not qp.warm), eta from
    the (seed, QIF_SEED) stream as the agent draws it; adds the mean per-row 1 - corr (1-corr_row), the
    spikes per module per frame (S_mod) and 1 - corr(mean kappa, T * mean pi-hat) (1-corr_Jhat: kappa against the
    pairings the network's own rates predict). All engines: the bias 1 - corr(mean kappa over reps, J) (1-corr_bias)."""
    agent = Agent(Config("flow", seed=seed, la_beta=LA_BETA, task=task))
    rows, N = [], len(agent.W)
    for H in Hs:
        rng, l1, rk, rr, rb, rh, smod = np.random.default_rng(seed), [], [], [], [], [], []
        if engine == "qif":
            qrng = np.random.default_rng((seed, QIF_SEED))
            eta = qif.lorentzian(qrng, (H, qif.n_neurons(N, agent.m, agent.roles, qp)), qp)
        for v in agent.held[:n_obs]:
            pi = core.flow(agent.W, v, ALPHA, N_HOPS)
            J = agent.W / agent.W.sum(0) * pi
            if engine == "qif":
                st = qif.init_state(eta, qrng, qp)
                if qp.warm:
                    qif.run_qif(agent.W, v, H, rng, eta, st, qp, agent.m, agent.roles)
            ks, phs = [], []
            for _ in range(reps):
                if engine == "qif":
                    if not qp.warm:
                        st = qif.init_state(eta, qrng, qp)
                    S = qif.run_qif(agent.W, v, H, rng, eta, st, qp, agent.m, agent.roles)
                    ph, kap, _ = qif.pooled(S, H, qp)
                    smod.append(np.bincount(agent.m, S.reshape(len(S), H, N).sum((0, 1)), agent.K))
                    rr.append(1 - row_corr(kap, J, agent.mask))
                else:
                    S = sp.run_heads(agent.W, v, H, K_SPK, RATE, rng, THETA, 1 - ALPHA)
                    ph, kap, _ = sp.pooled(S, H)
                l1.append(np.abs(ph - pi).sum())
                rk.append(1 - np.corrcoef(kap[agent.mask], J[agent.mask])[0, 1])
                ks.append(kap)
                phs.append(ph)
            rb.append(1 - np.corrcoef(np.mean(ks, 0)[agent.mask], J[agent.mask])[0, 1])
            Jh = agent.W / agent.W.sum(0) * np.mean(phs, 0)
            rh.append(1 - np.corrcoef(np.mean(ks, 0)[agent.mask], Jh[agent.mask])[0, 1])
        rows.append({"H": H, "L1_pi": float(np.mean(l1)), "1-corr_kappa_J": float(np.mean(rk)),
                     "1-corr_bias": float(np.mean(rb))})
        if engine == "qif":
            rows[-1].update({"1-corr_row": float(np.mean(rr)), "1-corr_Jhat": float(np.mean(rh)),
                             "S_mod": np.mean(smod, 0).tolist()})
        print(f"{task:12} H={H:2d}  " + "  ".join(f"{k} {x:.4f}" for k, x in rows[-1].items()
                                                if k not in ("H", "S_mod")) +
              (f"  S_mod {np.round(rows[-1]['S_mod']).astype(int).tolist()}" if engine == "qif" else ""), flush=True)
    return rows


def nondefault(r):
    """True for a variant run (any option off its M3 default). Also reads the keys of the variants removed in
    iteration 013 (beta_end, score_unit, beta_eps, beta_signal, beta_step_cap, critic_lambda, c_use), so their
    committed files stay out of the baseline groups."""
    eta0 = ETA if r["backend"] == "flow" else r["params"]["eta_task"]
    return r.get("lam", 0.05) != 0.05 or r.get("p_cc", 0.1) != 0.1 or r.get("p_cross", 0.05) != 0.05 or \
        r.get("beta", 1.0) != 1.0 or \
        r.get("beta_end") is not None or \
        r.get("eta_task", eta0) != eta0 or bool(r.get("beta_kappa")) or bool(r.get("score_unit")) or \
        bool(r.get("beta_eps")) or r.get("beta_signal", "episode") != "episode" or bool(r.get("beta_step_cap")) or \
        bool(r.get("critic_lambda")) or r.get("gamma", GAMMA) != GAMMA or \
        r.get("ablate", {}).get("tau_e", TAU_E) != TAU_E or r.get("readout", "race") != "race" or \
        bool(r.get("encoder_spec") or r.get("critic_layer") or r.get("c_ent") or r.get("c_use")) or \
        r.get("engine") in ("qif", "lif") or r.get("rule", "v4") != "v4" or r.get("lam_scale", 1.0) != 1.0 or \
        r.get("eta_scale", 1.0) != 1.0 or \
        r["episodes"] != TASKS[r.get("task", "CartPole")].episodes


def summary():
    """Per task: random-policy return, then flow and spiking per H as mean +- sd over seeds of the last-100 return,
    and the share of the flow-over-spiking(H = 1) gap that H heads close."""
    print("return: mean +- sd over seeds of the per-seed mean of the last 100 episodes; "
          "closed = (spiking_H - spiking_1) / (flow - spiking_1)")
    for task in TASKS:
        runs = {}
        for f in sorted(glob.glob(str(OUT / f"m3_{task}_*.json"))):
            r = json.loads(pathlib.Path(f).read_text())
            if r.get("ablate") or r.get("eta_tau", 300.0) != 300.0 or nondefault(r):
                continue  # learning-rule ablations and variants: see ablation_summary()
            runs.setdefault((r["backend"], r.get("heads", 1), r["la_beta"], r.get("td_norm", 0.0)), []).append(r)
        if not runs:
            continue
        rand = random_return(200, task=TASKS[task])
        print(f"\n{task}: random {rand[0]:.1f} +- {rand[1]:.1f}")
        last = {k: [np.mean(r["returns"][-100:]) for r in rs] for k, rs in runs.items()}
        for (b, H, lb, tn), rs in sorted(runs.items(), key=lambda kv: (kv[0][0] != "flow", kv[0][3], *kv[0][1:3])):
            R = last[(b, H, lb, tn)]
            line = f"  {b:8} H={H:2d} lb={lb:g} tn={tn:g} n={len(rs)} epis={min(r['episodes'] for r in rs):5} " \
                   f"ret {np.mean(R):8.1f} +- {np.std(R):6.1f}  ms/fr {np.mean([r['ms_per_frame'] for r in rs]):5.1f}"
            fl, s1 = last.get(("flow", 1, lb, tn)), last.get(("spiking", 1, lb, tn))
            if b == "spiking" and fl and s1 and abs(np.mean(fl) - np.mean(s1)) > 1e-9:
                line += f"  closed {(np.mean(R) - np.mean(s1)) / (np.mean(fl) - np.mean(s1)):+.2f}"
            print(line)


def ablation_summary():
    """Flow learning-rule ablations (iteration 005) against the --td-norm 0.04 baseline: last-100 return over seeds,
    final policy entropy and share of synapses at the weight floor."""
    groups = {}
    for f in sorted(glob.glob(str(OUT / "m3_*_flow_lb0.3_tn0.04*_s*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        var = pathlib.Path(f).stem.split("_tn0.04")[1].rsplit("_s", 1)[0].lstrip("_") or "baseline"
        groups.setdefault((r.get("task", "CartPole"), var), []).append(r)
    for (task, var), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "baseline", kv[0][1])):
        R, ev = [np.mean(r["returns"][-100:]) for r in rs], [r["evals"][-1] for r in rs]
        print(f"{task:11} {var:16} n={len(rs)} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}  "
              f"[{' '.join(f'{x:.0f}' for x in R)}]  H(P) {np.mean([e['P_entropy'] for e in ev]):.2f}  "
              f"P_max {np.mean([e['P_max'] for e in ev]):.2f}  at_floor {np.mean([e['frac_wmin'] for e in ev]):.3f}")
