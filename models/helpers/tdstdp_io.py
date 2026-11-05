"""Results and saved models of the TD-STDP backbone: result records, solve times, save_ck_model and
load_ck_model."""
import dataclasses
import json
import pathlib
import time

import numpy as np

from models.helpers.runner import SOLVE, TDSTDP_SEED, CKRunner, Config, trailing_mean_time


def rec_line(hr):
    """Log-line monitors of rec over the last logged bins (HybridRec.log)."""
    lg = hr.log
    if not lg["episode"]:
        return ""
    k = slice(-10, None)
    mx = lambda key: float(np.mean(lg[key][k]))  # noqa: E731
    return (f" | R: maxRe {mx('maxre'):.2f} (max {max(lg['maxre_max'][k]):.2f}) chiG {mx('chiG'):.2f} causal "
            f"{mx('causal'):.4f} within {mx('within'):.3f} L_obs {mx('L_obs'):.3f} rate {mx('rate'):.0f} Hz repeat "
            f"{mx('repeat'):.2f}/{mx('repeat_chance'):.2f}" + (f" lat {mx('rate_lat0'):.0f}/{mx('rate_lat1'):.0f} Hz "
                                                               f"lat_out {mx('lat_out'):.3f}" if hr.Kr > hr.K else ""))


def save_ck_model(path, cfg, r, eta_sgd=None, s_cal=None, rec_cal=None):
    """The final model of a CK run as one .npz (--save-model): the input weights W (n_post x n_in, CK units), the
    calibrated config as JSON (s_in, alphas, etas after --eta-cal, race beta), the Layout sizes, the post signs and
    the race's beta; with recurrence also R, its mask and module labels and the R snapshots (the keys
    experiments.structure reads). load_ck_model() rebuilds a frozen runner from it (models/evaluate.py)."""
    meta = dict(W=r.W, config=json.dumps(dataclasses.asdict(cfg)), n_in=r.lay.n_in, n_ce=cfg.n_ce, n_ci=cfg.n_ci,
                n_ae=cfg.n_ae, n_ai=cfg.n_ai, K=r.K, n_lat=r.lay.n_lat, sign=r.lay.sign, episodes_run=len(r.returns),
                beta=getattr(r, "beta", 0.0), eta_sgd_mV=np.asarray(eta_sgd if eta_sgd is not None else []),
                s_cal=np.asarray(s_cal if s_cal is not None else []), rec_cal=json.dumps(rec_cal or {}))
    if r.hr is not None:
        meta.update(R=r.hr.R, m=r.hr.m, roles=r.hr.roles, mask=r.hr.mask, ridx=r.hr.ridx,
                    episodes=np.array(sorted(r.snaps)), **{f"R_ep{k}": v for k, v in r.snaps.items()})
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **meta)


def load_ck_model(path, rng_seed=TDSTDP_SEED):
    """A frozen CKRunner (no plasticity) holding the saved W (and R, mask) of save_ck_model: its tick() runs the policy
    in the environment and appends episode returns to .returns. The saved config carries every calibrated value, so no
    calibration runs."""
    z = np.load(path, allow_pickle=False)
    d = json.loads(str(z["config"]))
    cfg = Config(**{"rec_full" if k == "rec_seal" else k: v for k, v in d.items()})  # rec_seal: rec_full's old name
    r = CKRunner(cfg, rng_seed=rng_seed, frozen=True)
    r.W[:] = z["W"]
    if r.hr is not None and "R" in z:
        r.hr.R[:], r.hr.mask[:] = z["R"], z["mask"]
    if "beta" in z and hasattr(r, "beta"):
        r.beta = float(z["beta"])
    return r


def result_ck(cfg, r, t0, eta_sgd, resumes, s_cal=None, rec_cal=None):
    Tf, Ts = pooled_times(cfg.task, r.returns)
    wall = time.perf_counter() - t0
    out = {"config": dataclasses.asdict(cfg), "returns": r.returns, "T_f": Tf, "T_s": Ts, "frames": r.frames,
           "ms_per_frame": wall / max(r.frames, 1) * 1e3, "wall_s": wall, "W_abs_mean": float(np.abs(r.W).mean()),
           "W_neg_frac": float((r.W < 0).mean()), "n_in": r.lay.n_in, "mv": r.mv, "resumes": resumes}
    if cfg.heads > 1:
        out["returns_heads"] = r.returns_heads
    if eta_sgd is not None:
        out["eta_sgd_mV"] = eta_sgd.tolist()
    if cfg.engine == "nlif":
        out["s_in"] = cfg.s_in  # also in config; s_in_cal: [lif, nlif] initial post rate (absent after a resume)
        if s_cal is not None:
            out["s_in_cal"] = s_cal
    if r.lowrate is not None:
        out["low_rate"] = r.lowrate
        if "race" in r.lowrate:
            out["race_cal"] = r.lowrate["race"]
    # energy monitor (016 §14.1): mean actor spikes (E and I posts) per decision window per copy, over the run
    out["actor_spikes_per_decision"] = float(r.espk[0] / max(r.espk[1], 1))
    if cfg.readout == "race":
        out["race_decisions"] = dict(theta=r.theta, at_bound=int(r.rdec[0]), at_deadline=int(r.rdec[1]),
                                     ties=int(r.rdec[2]))
        # actor spikes up to the decision (the bound ms, or the whole window at the deadline), per copy-decision: the
        # energy a loop that stopped at the bound would spend (016 §18); actor_spikes_per_decision is the full window
        out["actor_spikes_to_decision"] = float(r.ebnd[0] / max(r.ebnd[1], 1))
        if cfg.race_reprobe:  # the re-probe records (reprobe_race) and the surrogate's final gain
            out["race_cal_history"], out["race_beta_final"] = r.race_cal_history, float(r.beta)
        out["race_log"] = r.slog  # per log bin: score_abs_mean, score_clip_frac (score credit), race_entropy,
        # col_rate_min, col_rate_min_ms, action_repeat_probability (SLOG_KEYS; score_monitors)
        if cfg.credit != "surrogate":
            out["c_t_eff"] = float(r.c_t)
    if r.hr is not None:  # iteration 015: monitors per flush (HybridRec.log), calibration (absent after a resume)
        hr = r.hr
        out["rec"] = {"eta_rec": cfg.eta_rec, "lam": cfg.lam * cfg.lam_rec, "log": hr.log, "n_R": hr.nR,
                      "n_syn": int(hr.mask.sum()),
                      "within_final": float((hr.R * (hr.m[:, None] == hr.m[None, :])).sum() / hr.R.sum())}
        H = np.eye(hr.m.max() + 1)[hr.m]  # module weight table [post, pre] / total R, per snapshot (SNAP_EVERY)
        out["rec"]["RK"] = {str(k): (H.T @ v @ H / v.sum()).tolist() for k, v in sorted(r.snaps.items())}
        # the stimulus-conditioned walk (016 §15): per snapshot and executed action, the module-level pair table
        # [post, pre] as a share of that class's pairs since the previous snapshot, and the class's pair count
        out["rec"]["RKc"] = {str(k): {x: np.asarray(y).tolist() for x, y in v.items()}
                             for k, v in sorted(r.snaps_kc.items())}
        if rec_cal is not None:
            out["rec"]["cal"] = rec_cal
    return out


def pooled_times(task, ret):
    """main.py's t_f and t_s on the pooled episode sequence (1-indexed; None if never): T_f = first episode with return
    >= the perfect score, T_s = last episode of the first 100-episode window whose mean reaches it."""
    perf, ret = SOLVE[task][0], np.asarray(ret, float)
    hit = np.flatnonzero(ret >= perf)
    return (int(hit[0]) + 1 if len(hit) else None), trailing_mean_time(ret, perf)


def solve_times(task, returns_heads):
    """(T_f, T_s, per-episode return): returns_heads is a list of per-copy return lists. T_f: first episode in which
    some copy reaches the perfect score. T_s: first episode e >= 100 at which the last 100 (copy-mean) returns are all
    perfect (CartPole) or average >= 200 (LunarLander). 1-indexed; None if never."""
    perf, rule = SOLVE[task]
    E = min(len(r) for r in returns_heads)
    M = np.array([r[:E] for r in returns_heads])
    ret = M.mean(0)
    hit = np.flatnonzero((M >= perf).any(0))
    Tf = int(hit[0]) + 1 if len(hit) else None
    ok = M.min(0) >= perf if rule == "all" else None
    Ts = None
    for e in range(100, E + 1):
        if (ok[e - 100:e].all() if rule == "all" else ret[e - 100:e].mean() >= perf):
            Ts = e
            break
    return Tf, Ts, ret.tolist()


def result(cfg, r, t0, eta_sgd):
    Tf, Ts, ret = solve_times(cfg.task, r.returns)
    wall = time.perf_counter() - t0
    out = {"config": dataclasses.asdict(cfg), "returns": ret, "T_f": Tf, "T_s": Ts, "frames": r.frames,
           "ms_per_frame": wall / max(r.frames, 1) * 1e3, "wall_s": wall, "W_abs_mean": float(np.abs(r.W).mean()),
           "W_neg_frac": float((r.W < 0).mean()), "n_in": r.lay.n_in, "mv": r.mv}
    if cfg.heads > 1:
        out["returns_heads"] = r.returns
    if eta_sgd is not None:
        out["eta_sgd_ck_units"] = eta_sgd.tolist()
    return out
