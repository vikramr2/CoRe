"""Iteration 013 step 1: the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044,
mapstdp/tdstdp.py) in closed loop on CartPole and LunarLander. This file holds the training loops (run_ck, run) and the
command line; the objects and methods they use are in models/helpers/ (runner: parameters, Config, presets, Runner,
CKRunner; calibration; tdstdp_io; tdstdp_stats) and are re-exported here (__all__). Run from the repo root:
  python -m models.tdstdp --task {CartPole,LunarLander} --seed 0 [--adam | --no-adam | --eta-cal] [--heads H]
    [--block B] [--episodes E] [--engine {numpy,snm,lif,flow,nlif}] [--s-in S] [--legacy [--rate0 p]
    [--engine lif-snm]] [--rec {cols,lat} [--chi-rec X] [--lam L] [--lam-ratio r] [--save-model PATH] ...]
    -> experiments/results/tdstdp_<task>[_ck][_<tokens>]_s<seed>.json   (tokens: options off the preset, in order
       adam | noadam | sgd (= --eta-cal), B<B>, h<H>, ep<E>, r<rate0> (legacy), fast, si<S>, engine snm | lif |
       lifsnm | flow | nlif)
    [--out PATH]   (result path; the default above; experiments.report uses it to write reports/<study>/runs/)
  python -m models.tdstdp --summary   -> per group: T_f, T_s, solved fraction, last-100 return, ms per frame,
       next to CK's Table 2 and the model v4 (flow, spiking) returns
Default (token ck): the authors' code, reference/td_stdp (commit a5cb7f4, BindsNET 0.2.7), where it disagrees with
the paper (iteration 013 §8). PRESETS are config/config_cp.ini and config_ll.ini (eta in mV); weights convert to
branching units by MV = 13 (a BindsNET spike through w adds w mV against a 13 mV gap), engine lif is BindsNET's
LIF with W in mV. CKRunner transcribes main.py, bindsnet_add.Snn_actor / Reward_fn and util.batch_envs: the action
is sampled from the response to the current observation, p is fixed at its sampling probability, V' := 0 only on true
termination, 6 presentations of s_0 per episode, the update is the mean over all copies, and episodes are pooled
over copies (min_eps = episodes, max_eps). Block B: W frozen for B ms (tdstdp.block_learn_ck), Adam then steps once
per block with per-ms-matched moments (tdstdp.Adam); the authors step per ms (B = 1).
--fast (iteration 013 §10, S2): the update summed over copies in one matrix product (tdstdp.block_learn_ck_fast);
same rng order, equal up to floating-point summation order (not bit-identical). --engine flow (S3): the branching
step's noise-free limit (tdstdp.flow_step): rates instead of spikes, plus the input autocorrelation term of z.
--engine nlif (iteration 015): the noisy LIF of mapstdp.lif on the 1 ms grid (tdstdp.simulate_nlif) on every post, the
critic included; W, init and eta in mV as engine lif. The input scale s_in is set before the run, not tuned on return
(calibrate_s_in: the mean initial post rate equals engine lif's on the same stream); --s-in fixes it instead. Rates rho,
value, softmax and the rule are unchanged; the membranes are carried across blocks and set to mu_bg (no draw) on an
episode reset. The neuron noise has its own stream, so the main rng (inputs, actions, env seeds) is consumed as
engine lif's.
--rate-target R (iteration 015 step 3, engine nlif; token rt<R>): the low-rate regime, tau_ref 2 ms (lif.DT1R), s_in
set so the mean initial post rate is R Hz, the softmax and critic gains alpha_a, alpha_c times k = r_CK / R (r_CK:
BindsNET's initial rate on the same probe) and, without --eta-cal or Adam, eta_c, eta_a matched to the RMS of
BindsNET's initial per-ms update (calibrate_low; the result's 'low_rate' records it). tau_n stays CK's 20 ms.
--rec {cols,lat} (iteration 015 step 2, engine nlif, --block 20, no Adam; derivation §8.1, Proposal): recurrent E->E
modules in the actor (tdstdp.HybridRec): R among the actor E posts, one module per action ('cols'), plus --n-lat-mod
latent modules of n_ae posts ('lat'; input->latent W frozen at init, latent->action R learned, latent-output barrier
--la-beta). Kicks chi_rec R / d_j (--chi-rec). R learns by the one-hop task credit (eta_rec) and the map term (lam_rec,
off with --lam 0) with d_j homeostasis; the counter's gain is the kernel's own transfer (lif.gain_markov). eta_rec and
lam_rec are pre-set by calibrate_rec unless given (--eta-rec, --lam-rec); s_in and --eta-cal are calibrated without
recurrence, so every arm of a seed shares them. --rec-seal drops R's cross-module synapses from the same draws
(enforced modularity); --p-cross sets R's cross-module connection probability (0.05; within 0.5; iteration 018),
0 being the sealed graph. Tokens after the engine: cols | lat, xr<chi_rec>, seal, px<p_cross>, lam0 | lam<l>,
lr<lam_ratio>, lx<lam_rec>, er<eta_rec>, sw<stat_windows>, pw<pair_ms>, nl<n_lat_mod>, lb<la_beta> (off-default
only, except the first two). The result's 'rec' holds the calibration and the monitors (HybridRec.log, one row per
pooled-episode event); --save-model PATH writes R snapshots (episode 0, every SNAP_EVERY, end) with m, roles, mask.
--readout race (iteration 016 §14.1 Stage B, engine nlif with --rate-target; token race): the softmax is replaced by a
race on the columns' E spike counts (CKRunner.Mcnt: the excitatory posts of each column, §18; the opponent I posts
shape the dynamics but are not tallied) at 1 ms resolution: the first column whose count since the copy's last
resample reaches theta wins (ties: one uniform draw), else the leader at the deadline (the freeze window), Eq. 1h.
theta = round(n_ae rate_target freeze), the mean count of one column at the target rate over a decision window
(--race-theta overrides; token th<theta>). Learning uses the population surrogate p~ = softmax(beta log max(r_k,
eps_r)) of the columns' E rates r_k (the quantity the race tallies, CKRunner.MaE, 016 §21; eps_r = 1 / (tau_n n_ae),
one spike's filtered rate) with its exact gradient as the per-column eligibility factor (race_factor) on the E actor
rows (the I rows get no policy credit and no entropy term; a no-op where n_ai = 0); beta is pre-set by calibrate_race
(a held-observation probe: the surrogate's mean per-state entropy at the mean column rates equals the realised
race's) unless --race-beta is given
(token rb<beta>); --race-reprobe N (016 §18 fix a; token rp<N>) re-runs the probe at the current W (and R) once every
N pooled episodes on its own stream (reprobe_race) and the new beta applies from the next window on if the record's
fit is within RACE_FIT_GATE (reprobe_step; a rejected record is marked 'accepted': False in the result's
'race_cal_history'). --freeze MS (token fz<ms> off the preset) sets the decision window, the race's deadline and the
hold of the sampled action over freeze / 20 env steps; the decision hold of 016 §21 is theta from the 20-ms rule
(--race-theta round(n_ae r0 20 ms)) with the preset's freeze (see Config.freeze).
--norm-logits and --ent-match-rec are softmax-only. Both readouts log the actor spikes per decision window per copy
(the result's 'actor_spikes_per_decision'; the race adds 'race_decisions').
--credit {hazard,noise} (016 §20 Stage B', readout race, --eta-cal; tokens sch | scn): the surrogate is replaced by the
spike score (lif.post_scores): every actor E post's spike or silence at ms t, up to the copy's bound ms (the whole
window at a deadline decision), is credited by its own score Y_i(t) = d log P(S_i(t)) / d(input) (hazard: (S - h) psi
clipped at h = 1e-3; noise: the exact observed-membrane score), and the actor rows learn by z <- a z + Y x(t-1)^T /
tau_z, q <- b q + z / tau_q, W += eta delta q (c = 1 on E actor posts, 0 on I posts, no sign, no beta, no 1 / r): the
score pairs with x(t-1), the input the kernel paired with the step (d m(t) / dW_ij = s_in x_j(t-1) / theta_L, V(t-1)
being conditioned on; the P trace would count inputs whose effect is already in V(t-1) and over-counts the gradient
1.7-1.9x on the one-neuron toy; tdstdp self-check (h4)). Two learner calls per block: the critic rows as today (X, P;
bit-identical to the surrogate run), the actor rows on an actor-only Layout. The rate term is applied rank-1 per
block, -c_t sum_t (rho_bar - rho_hat) rho_i P_j (homeostatic, on the P trace), with c_t scaled by eta_a(surrogate) /
eta_a(score) (both from calibrate_ck) so its effective step is the
surrogate arm's; c_e is forced to 0; race_reprobe must be 0 (the surrogate's beta is still calibrated, for the
reference eta_a only). R's credit (HybridRec) is the column's summed score up to the bound, paired with the delivered
current b n_j (credit called with G = 1 / tau_eff). Monitors per log bin (the result's 'race_log'): mean |Y| over
active actor-E ms, the fraction of actor-E spikes at h < 1e-3, the entropy of the realised actions per copy, and
(every race run) the columns' mean E rates: col_rate_min, the minimum over columns of the bin-averaged column rate,
and col_rate_min_ms, the bin average of the per-ms, per-copy minimum over columns (Hz).
--rate-floor F (016 §20, the neuroscientist's contrast floor; requires --rate-cost; token rf<F>; pre-set 0.5): a
one-sided per-column rate floor added to the mean pin, a band (copy mean at r0, each column >= F r0). Per ms, with
rho_k(t) the mean rate of column k's E posts and rho_min = F rate_target, column k's E actor rows get
+c_t (rho_min - rho_k(t))_+: potentiation only, zero at or above rho_min, never lowers a winner (the deprivation side
of rate homeostasis). Surrogate arms: on kap per ms (it rides z, like the mean term); score arms: rank-1 per block,
+c_t sum_t (rho_min - rho_k(t))_+ rho_i P_j on column k's E rows, with the mean term's effective step c_t_eff.
Latents and I posts untouched; off in the frozen calibration probes."""
import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

from mapstdp import tdstdp as td
from models.helpers.calibration import (
    RACE_DELTAS,
    RACE_FIT_GATE,
    _race_probe,
    calibrate,
    calibrate_ck,
    calibrate_low,
    calibrate_race,
    calibrate_rec,
    calibrate_s_in,
    probe_entropy,
    probe_rate,
    race_fit,
    reprobe_race,
    reprobe_step,
)
from models.helpers.runner import (
    CAL_SEED,
    MS,
    OUT,
    PRESETS,
    R_CK_CARTPOLE,
    SLOG_KEYS,
    TD_TASKS,
    WARM_N,
    CKRunner,
    Config,
    Runner,
    ck_encoder,
    name,
    preset,
    race_theta,
    tokens,
    trailing_mean_time,
)
from models.helpers.tdstdp_io import load_ck_model, rec_line, result, result_ck, save_ck_model
from models.helpers.tdstdp_stats import M_THR, PAPER, group_stats, summary

__all__ = [  # the model's API (objects and methods from helpers/, re-exported)
    "CAL_SEED", "CKRunner", "Config", "MS", "M_THR", "OUT", "PAPER", "PRESETS", "RACE_DELTAS", "RACE_FIT_GATE",
    "R_CK_CARTPOLE", "Runner", "SLOG_KEYS", "TD_TASKS", "WARM_N", "_race_probe", "argparse", "calibrate",
    "calibrate_ck", "calibrate_low", "calibrate_race", "calibrate_rec", "calibrate_s_in", "ck_encoder", "dataclasses",
    "group_stats", "json", "load_ck_model", "main", "name", "np", "pathlib", "preset", "probe_entropy", "probe_rate",
    "race_fit", "race_theta", "rec_line", "reprobe_race", "reprobe_step", "result", "result_ck", "run", "run_ck",
    "save_ck_model", "summary", "td", "time", "tokens", "trailing_mean_time",
]


def run_ck(cfg, log_every=50, out=None, save_model=None):
    eta_sgd, s_cal, rec_cal = None, None, None
    fn = pathlib.Path(out) if out else OUT / f"{name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    ck = fn.with_suffix(".ckpt.npz")
    low, score = None, cfg.credit != "surrogate"
    if score:
        assert cfg.readout == "race" and cfg.engine == "nlif" and cfg.rate_target and cfg.sgd \
            and not cfg.race_reprobe, "credit: --readout race --engine nlif --rate-target --eta-cal, no --race-reprobe"
        if cfg.c_e:
            print(f"credit {cfg.credit}: the preset's entropy bonus c_e {cfg.c_e:g} is forced to 0 (no policy under "
                  "the score; the rate term is the exploration floor)", flush=True)
            cfg = dataclasses.replace(cfg, c_e=0.0)
    if cfg.engine == "nlif" and cfg.rate_target:
        if ck.exists():
            low = json.loads(str(np.load(ck)["lowrate"]))
            cfg = dataclasses.replace(cfg, **low["cfg"])
        else:
            cfg, low = calibrate_low(cfg)
            print("low-rate calibration: " + ", ".join(f"{k} {v}" for k, v in low.items()), flush=True)
    elif cfg.engine == "nlif" and not cfg.s_in:
        if ck.exists():
            cfg = dataclasses.replace(cfg, s_in=float(np.load(ck)["s_in"]))
        else:
            s_in, target, got, _ = calibrate_s_in(cfg)
            cfg, s_cal = dataclasses.replace(cfg, s_in=s_in), [target, got]
            print(f"s_in calibration: initial post rate (spikes / ms) lif {target:.4f}, nlif {got:.4f} at s_in "
                  f"{s_in:.4f}", flush=True)
    if cfg.sgd:
        eta_sgd, rms = calibrate_ck(cfg)
        eta_sgd = eta_sgd * np.array([1.0, cfg.eta_mult_a])  # the per-task actor step multiplier (1 by default)
        if score:  # the rate term keeps the surrogate arm's effective step: c_t x eta_a(surrogate) / eta_a(score),
            eta_ref, rms_ref = calibrate_ck(dataclasses.replace(cfg, credit="surrogate"))  # both at the same W
            low["score"] = dict(credit=cfg.credit, eta_a=float(eta_sgd[1]), eta_a_ref=float(eta_ref[1]),
                                rms_a=float(rms[1]), rms_a_ref=float(rms_ref[1]), c_t=cfg.c_t,
                                c_t_eff=float(cfg.c_t * eta_ref[1] / eta_sgd[1]))
            print("score credit: " + ", ".join(f"{k} {v}" for k, v in low["score"].items()), flush=True)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"eta calibration: per-ms update RMS {rms}, eta (mV) {eta_sgd}", flush=True)
    if cfg.rec != "none" and not (cfg.eta_rec and cfg.lam_rec):
        if ck.exists():
            c = np.load(ck)
            cfg = dataclasses.replace(cfg, eta_rec=float(c["eta_rec"]), lam_rec=float(c["lam_rec"]))
        else:
            eta_rec, lam_rec, rec_cal = calibrate_rec(cfg)
            cfg = dataclasses.replace(cfg, eta_rec=eta_rec, lam_rec=lam_rec)
            print(f"rec calibration: eta_rec {eta_rec:.4g}, lam_rec {lam_rec:.4g} (lam {cfg.lam:g}); "
                  + ", ".join(f"{k} {v:.3g}" for k, v in rec_cal.items()), flush=True)
    r, wall0, resumes = CKRunner(cfg), 0.0, 0
    r.lowrate = low
    if score:
        r.c_t = low["score"]["c_t_eff"]
    if ck.exists():
        wall0, resumes = r.load(ck)
        print(f"resumed from {ck.name} at {len(r.returns)} episodes", flush=True)
    t0, nxt = time.perf_counter() - wall0, (len(r.returns) // log_every + 1) * log_every
    rp = cfg.race_reprobe if cfg.readout == "race" else 0
    nxt_rp = (len(r.returns) // rp + 1) * rp if rp else None
    while not r.done:
        r.tick()
        e = len(r.returns)
        if rp and e >= nxt_rp:  # 016 §18 fix (a): the contrast probe at the current W (and R); beta from the next
            nxt_rp = (e // rp + 1) * rp  # window on if the fit passes the gate; the probe draws from its own stream
            reprobe_step(r, e)
        if e >= nxt:
            nxt = (e // log_every + 1) * log_every
            st = r.stats / max(r.stats[4], 1)
            r.stats[:] = 0
            sm = r.score_monitors(e) if r.race else None
            print(f"ep {e} last{log_every} {np.mean(r.returns[-log_every:]):.1f} rate_c {st[0]:.3f} rate_a "
                  f"{st[1]:.3f} H(s) {st[2]:.3f} V {st[3]:.3f} |W| {np.abs(r.W).mean():.4f} spk/dec "
                  f"{st[5] / max(st[6], 1e-12):.1f} {(time.perf_counter() - t0) / 60:.1f} min"
                  + (f" | score |Y| {sm['score_abs_mean']:.3f} clip {sm['score_clip_frac']:.4f} H_race "
                     f"{sm['race_entropy']:.3f}" if score else "")
                  + (f" | colmin {sm['col_rate_min']:.1f}/{sm['col_rate_min_ms']:.1f} Hz"  # bin-mean / per-ms
                     + ("" if score else f" H_race {sm['race_entropy']:.3f}") if cfg.rate_floor else "")
                  + (f" rep {sm['action_repeat_probability']:.2f}" if r.race else "")
                  + (rec_line(r.hr) if r.hr else ""), flush=True)
            fn.with_suffix(".partial").write_text(json.dumps(result_ck(cfg, r, t0, eta_sgd, resumes, s_cal,
                                                                       rec_cal)))
            r.save(ck, time.perf_counter() - t0, resumes)
    if r.hr is not None:
        r.snaps[len(r.returns)] = r.hr.R.copy()
        if len(r.returns) not in r.snaps_kc:  # else the SNAP_EVERY snapshot of this episode already holds the pairs
            r.snaps_kc[len(r.returns)] = r.hr.class_table()
    if save_model:
        save_ck_model(save_model, cfg, r, eta_sgd, s_cal, rec_cal)
    res = result_ck(cfg, r, t0, eta_sgd, resumes, s_cal, rec_cal)
    fn.write_text(json.dumps(res))
    fn.with_suffix(".partial").unlink(missing_ok=True)
    ck.unlink(missing_ok=True)
    print(f"T_f {res['T_f']} T_s {res['T_s']} episodes {len(res['returns'])} last100 "
          f"{np.mean(res['returns'][-100:]):.1f} ms/frame {res['ms_per_frame']:.2f}")
    return res


def run(cfg, log_every=50, out=None, save_model=None):
    """Run cfg; the result goes to `out` (a .json path; .partial and .ckpt.npz alongside) or, by default, to
    experiments/results/<name(cfg)>.json. save_model (rec only): R snapshots."""
    if cfg.ck:
        return run_ck(cfg, log_every, out, save_model)
    eta_sgd = None
    if cfg.sgd:
        eta_sgd, rms = calibrate(cfg)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"eta calibration: per-ms update RMS {rms}, eta (CK units) {eta_sgd}", flush=True)
    r, t0, nxt = Runner(cfg), time.perf_counter(), log_every
    fn = pathlib.Path(out) if out else OUT / f"{name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    while not r.done.all():
        r.tick()
        e = min(len(x) for x in r.returns)
        if e >= nxt:
            nxt += log_every
            st = r.stats / max(r.stats[4], 1)
            r.stats[:] = 0
            last = np.mean([np.mean(x[-log_every:]) for x in r.returns])
            print(f"ep {e} last{log_every} {last:.1f} rate_c {st[0]:.3f} rate_a {st[1]:.3f} H(s) {st[2]:.3f} "
                  f"V {st[3]:.3f} |W| {np.abs(r.W).mean():.4f} {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            res = result(cfg, r, t0, eta_sgd)
            fn.with_suffix(".partial").write_text(json.dumps(res))
    res = result(cfg, r, t0, eta_sgd)
    fn.write_text(json.dumps(res))
    fn.with_suffix(".partial").unlink(missing_ok=True)
    print(f"T_f {res['T_f']} T_s {res['T_s']} last100 {np.mean(res['returns'][-100:]):.1f} "
          f"ms/frame {res['ms_per_frame']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(PRESETS), default="CartPole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="authors' code: min_eps = max_eps (pooled)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adam", action="store_true", default=None)
    g.add_argument("--no-adam", dest="adam", action="store_false")
    g.add_argument("--eta-cal", dest="sgd", action="store_true",
                   help="E4: no Adam; the local rule with eta calibrated to Adam's step at init")
    g.add_argument("--sgd", dest="sgd", action="store_true", help=argparse.SUPPRESS)  # old name, in recorded commands
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--legacy", action="store_true", help="pre-audit implementation (E1, E2, first E3/E4)")
    ap.add_argument("--rate0", type=float, default=None, help="legacy: initial spike probability per ms (post hoc)")
    ap.add_argument("--engine", choices=("numpy", "snm", "lif", "lif-snm", "flow", "nlif"), default="numpy",
                    help="lif: BindsNET's LIF (legacy: CK Eq. 1); lif-snm (legacy): SuperNeuroMAT's subtractive leak; "
                         "flow: the branching step's noise-free limit; nlif: the noisy LIF of mapstdp.lif (dt 1 ms)")
    ap.add_argument("--s-in", type=float, default=None, help="nlif: input scale (default: calibrate_s_in)")
    ap.add_argument("--rate-target", type=float, default=None,
                    help="nlif: low-rate regime: initial post rate (Hz; tau_ref 2 ms; calibrate_low)")
    ap.add_argument("--fast", action="store_true", help="update summed over copies in one product (not bit-identical)")
    ap.add_argument("--beta-c", type=float, default=None, help="critic bias (preset override; 016 §20 Acrobot rule)")
    ap.add_argument("--race-fit-gate", type=float, default=None,
                    help="race: accept a re-probed beta only if the probe fit <= this (default 0.05; inf = no gate)")
    ap.add_argument("--eta-mult-a", type=float, default=None, help="--eta-cal: actor step multiplier (per-task grid)")
    ap.add_argument("--rate-cost", type=float, default=None,
                    help="CK's target-rate term c_t on the actor E rows, target rate_target (016 §18; pre-set 5e-4)")
    ap.add_argument("--rate-floor", type=float, default=None,
                    help="one-sided per-column rate floor, fraction of rate_target, +c_t (rho_min - rho_k)_+ on column "
                         "k's E rows (016 §20; needs --rate-cost; pre-set 0.5)")
    ap.add_argument("--rec", choices=("none", "cols", "lat"), default="none",
                    help="nlif: recurrent E->E actor modules (iteration 015): cols, one per action; lat, + latents")
    for f, t, h in (("chi-rec", float, "rec: total kick per spike (threshold units)"),
                    ("p-cross", float, "rec: R's cross-module connection probability (0.05; 0 = --rec-seal)"),
                    ("lam", float, "rec: map term multiplier (0 = off)"),
                    ("lam-ratio", float, "rec: lam_rec = ratio x RMS(task step) / RMS(structural step)"),
                    ("lam-rec", float, "rec: structural rate (default: calibrate_rec)"),
                    ("eta-rec", float, "rec: task rate on R (default: calibrate_rec)"),
                    ("stat-windows", int, "rec: windows per statistics frame"),
                    ("pair-ms", int, "rec: pair-count window (ms)"),
                    ("n-lat-mod", int, "rec lat: latent modules"),
                    ("la-beta", float, "rec lat: latent-output barrier")):
        ap.add_argument(f"--{f}", type=t, default=None, help=h)
    ap.add_argument("--rec-seal", action="store_true", help="rec: no cross-module synapses in R (enforced modularity)")
    ap.add_argument("--rate-match-rec", action="store_true",
                    help="nlif: calibrate s_in with the arm's recurrence on (every arm at the same initial rate)")
    ap.add_argument("--ent-match-rec", action="store_true",
                    help="nlif low-rate: alpha_a per arm so the initial policy entropy is the base run's")
    ap.add_argument("--norm-logits", action="store_true",
                    help="nlif low-rate: policy scores normalised by the pooled actor rate against --rate-target")
    ap.add_argument("--readout", choices=("softmax", "race"), default="softmax",
                    help="nlif low-rate: race (016 §14.1 Stage B), the first column whose net count reaches theta")
    ap.add_argument("--freeze", type=int, default=None,
                    help="ms between action samples (preset 20; LunarLander 40): the decision window and the hold of "
                         "the action over freeze / 20 env steps (016 §21: --freeze 40 --race-theta 16 decides within "
                         "20 ms and holds for 2 env steps)")
    ap.add_argument("--race-theta", type=float, default=None, help="race: bound (default n_ae rate_target freeze)")
    ap.add_argument("--race-beta", type=float, default=None, help="race: surrogate sharpness (default calibrate_race)")
    ap.add_argument("--race-reprobe", type=int, default=None,
                    help="race: re-run the contrast probe at the current W every N episodes (016 §18 fix a; preset 50)")
    ap.add_argument("--credit", choices=("surrogate", "hazard", "noise"), default="surrogate",
                    help="race, --eta-cal: the actor rows' credit, the surrogate or the spike score (016 §20)")
    ap.add_argument("--save-model", default=None,
                    help="final model .npz: W, config, beta; with rec also R and its snapshots")
    ap.add_argument("--out", default=None, help="result .json path (default experiments/results/<name>.json)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    kw = {k: v for k, v in (("episodes", a.episodes), ("heads", a.heads), ("block", a.block), ("rate0", a.rate0),
                            ("s_in", a.s_in)) if v is not None}
    if a.fast:
        kw["fast"] = True
    kw.update({k: v for k, v in (("rate_target", a.rate_target), ("c_t", a.rate_cost), ("rate_floor", a.rate_floor),
                                 ("chi_rec", a.chi_rec), ("p_cross", a.p_cross),
                                 ("beta_c", a.beta_c), ("eta_mult_a", a.eta_mult_a), ("race_fit_gate", a.race_fit_gate),
                                 ("lam", a.lam), ("lam_ratio", a.lam_ratio),
                                 ("lam_rec", a.lam_rec), ("eta_rec", a.eta_rec), ("stat_windows", a.stat_windows),
                                 ("pair_ms", a.pair_ms), ("n_lat_mod", a.n_lat_mod), ("la_beta", a.la_beta))
               if v is not None})
    if a.rec != "none":
        kw["rec"] = a.rec
    if a.rec_seal:
        kw["rec_seal"] = True
    if a.rate_match_rec:
        kw["rate_match_rec"] = True
    if a.ent_match_rec:
        kw["ent_match_rec"] = True
    if a.norm_logits:
        kw["norm_logits"] = True
    if a.readout != "softmax":
        kw["readout"] = a.readout
    if a.credit != "surrogate":
        kw["credit"] = a.credit
    kw.update({k: v for k, v in (("freeze", a.freeze), ("race_theta", a.race_theta), ("race_beta", a.race_beta),
                                 ("race_reprobe", a.race_reprobe)) if v is not None})
    if a.episodes is not None and not a.legacy:
        kw["max_eps"] = a.episodes
    if a.sgd:
        kw.update(adam=False, sgd=True)
    elif a.adam is not None:
        kw["adam"] = a.adam
    run(preset(a.task, legacy=a.legacy, seed=a.seed, engine=a.engine, **kw), out=a.out, save_model=a.save_model)


if __name__ == "__main__":
    main()
