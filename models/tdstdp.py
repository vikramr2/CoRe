"""Iteration 013 step 1: the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044,
mapstdp/tdstdp.py) in closed loop on CartPole and LunarLander. Run from the repo root:
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
import copy
import dataclasses
import glob
import itertools
import json
import pathlib
import re
import time

import gymnasium as gym
import numpy as np

from mapstdp import lif
from mapstdp import tdstdp as td
from models.basic import FOURIER_BOUNDS, TASKS, Task, critic_layer, random_states

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "results"  # the intermediates store
TDSTDP_SEED, CAL_SEED, CAL_TICKS, FBAR_STATES, MV_LEGACY, MS, WARM_TICKS = 1302, 1303, 200, 5000, 1300.0, 20, 5
MV, WARM_N = td.THETA_L, 5  # authors' code: mV per unit of branching weight; batch_envs warm_n (windows)
PAPER = {"CartPole": dict(T_f=(57.0, 23.97), T_s=(169.5, 23.47)),
         "LunarLander": dict(T_f=(383.8, 71.49), T_s=(2575.2, 666.51))}  # CK Table 2 (10 runs)
SOLVE = {"CartPole": (500.0, "all"), "LunarLander": (200.0, "mean"),  # perfect score; solved = last 100 all / mean
         "Acrobot": (-100.0, "mean"), "Taxi": (8.0, "mean")}  # iteration 016: Gymnasium's reward_threshold
M_THR = {"CartPole": 475.0, "LunarLander": 200.0, "Acrobot": -100.0, "Taxi": 8.0}  # secondary: trailing-100 mean
SNAP_EVERY = {"CartPole": 100, "LunarLander": 500, "Acrobot": 100, "Taxi": 200}  # rec: R snapshots (pooled episodes)
TD_TASKS = TASKS | {"Taxi": Task("Taxi-v4", 6, 0, 2000, None, "onehot")}  # CKRunner's tasks (Taxi: tdstdp only)
R_CK_CARTPOLE = 140.0  # Hz: BindsNET's initial post rate on CartPole (hybrid-columns, 20 seeds: 138-142)
SLOG_KEYS = ("episode", "score_abs_mean", "score_clip_frac", "race_entropy",  # race_log (score_monitors)
             "col_rate_min", "col_rate_min_ms", "action_repeat_probability")


@dataclasses.dataclass(frozen=True)
class Config:
    task: str = "CartPole"
    seed: int = 0
    episodes: int = 400
    adam: bool = False
    sgd: bool = False  # E4 (--eta-cal): no Adam; the local rule with eta from the Adam-equivalent step at init
    heads: int = 1
    block: int = 1
    engine: str = "numpy"  # numpy | snm (SuperNeuroMAT, slow; smoke runs only) | lif | lif-snm (true-LIF control)
    spec: str = "f2c4hb"  # tasks.critic_layer Fourier spec of the input features
    in_rep: int = 1  # input neurons per feature
    n_ce: int = 40
    n_ci: int = 0
    n_ae: int = 20
    n_ai: int = 0
    eta_c: float = 2.5e-3  # CK units (mV); with adam: the Adam step
    eta_a: float = 1e-2
    adam_step_c: float = 0.0  # --eta-cal numerator when the preset's eta are SGD rates, not Adam steps (CartPole;
    adam_step_a: float = 0.0  # 016 §15, §20); 0: use eta_c / eta_a (the presets whose eta are Adam steps)
    eta_mult_a: float = 1.0  # --eta-cal: multiplier on the calibrated actor step (per-task grid, 016 §20)
    alpha_c: float = 2.0
    beta_c: float = -0.2
    alpha_a: float = 25.0
    tau_n: float = 20.0
    tau_p: float = 20.0
    tau_z: float = 20.0
    tau_q: float = 40.0
    c_e: float = 0.0
    c_w: float = 0.0  # CK units (1 / mV)
    c_t: float = 0.0  # CK's target-rate term [CK Eq. 31] on actor E rows; CKRunner: --rate-cost (016 §18 fix b)
    rho_hat: float = 0.0  # its target (per ms); 0 with c_t > 0 means rate_target (CKRunner)
    rate_floor: float = 0.0  # one-sided per-column rate floor, a fraction of rate_target (016 §20; pre-set 0.5): per
    # ms +c_t (rho_min - rho_k)_+ on column k's E rows, rho_k its E posts' mean rate; needs c_t > 0 (--rate-floor)
    tau_d: float = 1000.0  # discount time constant (ms)
    r_scale: float = 0.02
    resample: int = 1  # env steps per action sample
    adam_b1: float = 0.995
    adam_b2: float = 0.99995
    rate0: float = 0.2  # initial spike probability per ms (pre-registered weight init; legacy)
    # authors' code (ck = True); the defaults above are the legacy CartPole configuration
    ck: bool = False
    max_eps: int = 0  # pooled episodes cap (main.py max_eps); episodes is min_eps
    init_scale: float = 0.1  # W ~ U(0, init_scale) mV
    freeze: int = 20  # ms between action samples (freeze_action; --freeze): the decision window, the race's deadline
    # and the hold of the sampled action over freeze / 20 env steps. The hold rule (016 §21): the race decides within
    # 20 ms at theta = round(n_ae r0 x 20 ms) (--race-theta; r0 = rate_target per ms) and holds the decision for the
    # preset's freeze / 20 env steps, i.e. the race runs at the backbone's own control rate. The recipe is thus
    # LunarLander --race-theta 26 with the preset freeze 40 (2 env steps), CartPole the preset freeze 20 (1 step; the
    # 20-ms rule is the default theta there), Acrobot --freeze 40 (2 steps) with --race-theta 16.
    f_order: int = 2  # Fourier encoder (util.State_to_spike_Fourier): order, cross terms, ON/OFF split, constant unit
    f_cross: bool = True
    f_double: bool = False
    f_basis: bool = False
    adam_eps: float = 1e-8
    fast: bool = False  # authors' code: update summed over copies in one product (block_learn_ck_fast; §10 S2)
    s_in: float = 0.0  # engine nlif: feed-forward input scale (0 = calibrate_s_in before the run)
    rate_target: float = 0.0  # engine nlif, low-rate regime (Hz; 0 = CK-matched): calibrate_low before the run
    # iteration 015 (engine nlif only; tdstdp.HybridRec): recurrent modules in the actor
    rec: str = "none"  # 'none' | 'cols' (E->E matrix R on the actor E posts, a module per action) | 'lat' (+ latents)
    chi_rec: float = 0.3  # total kick of one spike over its targets (threshold units)
    lam: float = 1.0  # map term: lambda = lam lam_rec (0 = off)
    lam_ratio: float = 1.0  # rec_cal: lam_rec = lam_ratio RMS(task step on R) / RMS(structural step), per frame
    lam_rec: float = 0.0  # structural rate (0 = calibrate_rec before the run)
    eta_rec: float = 0.0  # task rate on R (0 = calibrate_rec: the actor W's relative step at init)
    stat_windows: int = 5  # windows (20 ms) per statistics frame
    pair_ms: int = 20  # pair-count window (ms)
    n_lat_mod: int = 2  # rec 'lat': latent modules of n_ae posts each (input->latent W frozen at init)
    la_beta: float = 0.3  # rec 'lat': latent-output barrier (core.latent_output_barrier)
    rec_seal: bool = False  # enforced modularity: R without cross-module synapses (same draws; iteration 015)
    p_cross: float = 0.05  # R's cross-module connection probability (within: 0.5); 0 = rec_seal (iteration 018)
    # iteration 016: tasks without the authors' configs
    onehot: bool = False  # ck_encoder: one-hot of a Discrete observation (n states), repeated in_rep times
    r_ck_hz: float = 0.0  # calibrate_low: k = r_ck_hz / rate_target (0 = BindsNET's rate measured on this task)
    rate_match_rec: bool = False  # probe_rate with the arm's R at init: every arm starts at the target rate
    ent_match_rec: bool = False  # calibrate_low: alpha_a per arm so its initial policy entropy is the base run's
    norm_logits: bool = False  # policy scores normalised by the pooled actor rate against rate_target (016 §13)
    # iteration 016 §14.1 Stage B (engine nlif, rate_target): the race readout in place of the softmax
    readout: str = "softmax"  # 'softmax' (CK Eq. 12) | 'race' (Eq. 1h on the columns' E counts, CKRunner.race_block)
    race_theta: float = 0.0  # race: bound on a column's E count (0 = round(n_ae rate_target freeze), CKRunner)
    race_beta: float = 0.0  # race: sharpness of the surrogate softmax(beta log r_E) of the E rates (0 = calibrate_race)
    race_reprobe: int = 0  # race: re-run the contrast probe at the current W (and R) every N episodes (0 = never)
    race_fit_gate: float = 0.05  # reprobe_step: accept a re-probed beta only if the probe's fit <= this (inf: no gate;
    # the pre-gate behaviour of study race-cartpole, models/reproduce.sh)
    # iteration 016 §20 Stage B' (race, --eta-cal): the actor rows' credit, 'surrogate' (p~ = softmax(beta log r)) or
    # the spike score of lif.post_scores, 'hazard' ((S - h) psi, clipped) | 'noise' (the exact observed-membrane score)
    credit: str = "surrogate"


PRESETS = {  # the authors' configs, reference/td_stdp/config/config_{cp,ll}.ini (eta in mV, before trace scaling)
    "CartPole": dict(ck=True, episodes=400, max_eps=400, eta_c=0.005, eta_a=0.01, alpha_a=50.0, init_scale=0.1,
                     adam_eps=1e-9, adam_step_c=1.25e-4, adam_step_a=6.25e-5),  # eta: SGD rates; adam_step: LL's
    "LunarLander": dict(ck=True, episodes=3000, max_eps=10000, adam=True, heads=16, block=20, in_rep=16, f_order=1,
                        f_cross=False, f_double=True, f_basis=True, n_ce=128, n_ci=128, n_ae=32, n_ai=32,
                        eta_c=1.25e-4, eta_a=6.25e-5, init_scale=0.125, alpha_c=4.0, beta_c=-2.0, alpha_a=15.0,
                        tau_q=20.0, c_e=1e-4, tau_d=2000.0, r_scale=0.012, freeze=40, adam_eps=1e-9),
}  # LunarLander: weight_reg = 1e-8 is never passed to the rule (main.py), so c_w = 0; block 20 is ours (B = 1 theirs)
# Iteration 016, pre-set rules (no authors' config; never tuned on return): CartPole's populations, alpha_a, tau_d,
# freeze and init; LunarLander's Adam steps as the --eta-cal step (the authors' only Adam steps); k from CartPole's
# BindsNET rate (R_CK_CARTPOLE), so the policy logits keep CartPole's range; r_scale = 0.02 / per-step reward RMS of a
# random policy (CartPole's RMS is 1); beta_c = min(-0.2, 1.2 V_rand), V_rand = r_scale E_rand[r] tau_d / MS the value
# of a random policy, so V = alpha_c rate + beta_c >= beta_c can represent it (LunarLander's -2.0 vs V_rand -2.31;
# CartPole's floor -0.2; test_tasks016). Acrobot (6-dim): LunarLander's encoder (CartPole's order-2
# cross terms would give 3^6 features). Taxi (500 states): one-hot, in_rep 1 (the smallest in_rep; calibrate_s_in
# reaches 40 Hz).
_NEW = dict(ck=True, eta_c=1.25e-4, eta_a=6.25e-5, init_scale=0.1, alpha_a=50.0, adam_eps=1e-9, r_ck_hz=R_CK_CARTPOLE,
            c_e=1e-4)
# c_e: LunarLander's entropy bonus (Eq. 26), the authors' exploration setting beyond CartPole. On an E-only actor the
# bonus's update lowers the total actor drive (sum_k g_k r_k = -alpha Var(r) <= 0), a silencing drift; the authors'
# pairing (opponent I posts, n_ai = n_ae) was piloted and withdrawn (base at floor); the fix is --norm-logits
# (iteration 016 §13).
PRESETS["Acrobot"] = dict(_NEW, episodes=500, max_eps=500, in_rep=16, f_order=1, f_cross=False, f_double=True,
                          f_basis=True, r_scale=0.02, beta_c=-1.2)
PRESETS["Taxi"] = dict(_NEW, episodes=2000, max_eps=2000, onehot=True, in_rep=1, r_scale=0.0035, beta_c=-0.82)
CK_BOUNDS = {"CartPole": ([-2.4, -3.2, -np.pi / 12, -3.2], [2.4, 3.2, np.pi / 12, 3.2]),  # main.py state_lim
             "LunarLander": ([-1, -0.2, -1, -1, -1, -1, 0, 0], [1, 2, 1, 1, 1, 1, 1, 1]),
             "Acrobot": FOURIER_BOUNDS["Acrobot-v1"]}  # iteration 016: the M3 bounds
LEGACY = {  # pre-audit (E1, E2, first E3/E4): CK Table 1 and §4 in CK units, divided by MV_LEGACY
    "CartPole": {},
    "LunarLander": dict(episodes=3000, adam=True, heads=16, block=20, spec="f1c1b", in_rep=16, n_ce=128, n_ci=128,
                        n_ae=32, n_ai=32, eta_c=1.25e-4, eta_a=6.25e-5, alpha_c=4.0, beta_c=-2.0, alpha_a=15.0,
                        tau_q=20.0, c_e=1e-4, c_w=1e-8, tau_d=2000.0, r_scale=0.012, resample=2),
}


def preset(task, legacy=False, **kw):
    return dataclasses.replace(Config(task=task, **(LEGACY if legacy else PRESETS)[task]), **kw)


def race_theta(cfg):
    """readout race: the bound theta = round(n_ae r0 T), the mean E count of one column at the target rate r0 =
    rate_target (per ms) over a decision window T = freeze (ms): CartPole and Acrobot 16, LunarLander 51 (016 §14.1
    Stage B, no free number; the tally is the E count on every task, §18); race_theta overrides."""
    return float(cfg.race_theta or round(cfg.n_ae * cfg.rate_target * 1e-3 * cfg.freeze))


def tokens(cfg):
    p = preset(cfg.task, legacy=not cfg.ck)
    t = ["ck"] if cfg.ck else []
    t += ["sgd"] if cfg.sgd else ["adam"] if cfg.adam and not p.adam else ["noadam"] if p.adam and not cfg.adam else []
    t += [f"B{cfg.block}"] if cfg.block != p.block else []
    t += [f"h{cfg.heads}"] if cfg.heads != p.heads else []
    t += [f"ep{cfg.episodes}"] if cfg.episodes != p.episodes else []
    t += [f"mx{cfg.max_eps}"] if cfg.max_eps != p.max_eps else []
    t += [f"r{cfg.rate0:g}"] if cfg.rate0 != p.rate0 else []
    t += ["fast"] if cfg.fast else []
    t += [f"si{cfg.s_in:g}"] if cfg.s_in else []
    t += [cfg.engine.replace("-", "")] if cfg.engine != "numpy" else []
    t += [f"rt{cfg.rate_target:g}"] if cfg.rate_target else []
    t += ["rm"] if cfg.rate_match_rec else []
    t += ["em"] if cfg.ent_match_rec else []
    t += ["nl"] if cfg.norm_logits else []
    t += [f"rc{cfg.c_t:g}"] if cfg.c_t else []
    t += [f"fg{cfg.race_fit_gate:g}"] if cfg.race_fit_gate != 0.05 else []
    t += [f"rf{cfg.rate_floor:g}"] if cfg.rate_floor else []
    t += [f"em{cfg.eta_mult_a:g}"] if cfg.eta_mult_a != 1.0 else []
    t += [f"bc{cfg.beta_c:g}"] if cfg.beta_c != p.beta_c else []
    t += [f"fz{cfg.freeze}"] if cfg.freeze != p.freeze else []
    if cfg.readout == "race":
        t += ["race"] + ([f"th{cfg.race_theta:g}"] if cfg.race_theta else [])
        t += [f"rb{cfg.race_beta:g}"] if cfg.race_beta else []
        t += [f"rp{cfg.race_reprobe}"] if cfg.race_reprobe else []
        t += ["sch"] if cfg.credit == "hazard" else ["scn"] if cfg.credit == "noise" else []
    if cfg.rec != "none":
        d = Config()
        t += [cfg.rec, f"xr{cfg.chi_rec:g}"] + (["seal"] if cfg.rec_seal else [])
        t += [f"px{cfg.p_cross:g}"] if cfg.p_cross != d.p_cross else []
        t += ["lam0"] if cfg.lam == 0 else [f"lam{cfg.lam:g}"] if cfg.lam != 1 else []
        t += [f"lr{cfg.lam_ratio:g}"] if cfg.lam_ratio != d.lam_ratio else []
        t += [f"lx{cfg.lam_rec:g}"] if cfg.lam_rec else []
        t += [f"er{cfg.eta_rec:g}"] if cfg.eta_rec else []
        t += [f"sw{cfg.stat_windows}"] if cfg.stat_windows != d.stat_windows else []
        t += [f"pw{cfg.pair_ms}"] if cfg.pair_ms != d.pair_ms else []
        if cfg.rec == "lat":
            t += [f"nl{cfg.n_lat_mod}"] if cfg.n_lat_mod != d.n_lat_mod else []
            t += [f"lb{cfg.la_beta:g}"] if cfg.la_beta != d.la_beta else []
    return t


def name(cfg):
    return "_".join(["tdstdp", cfg.task, *tokens(cfg)]) + f"_s{cfg.seed}"


_FBAR = {}


def features(cfg):
    """(phi, fbar): input rates phi(o) (each feature repeated in_rep times) and their mean over FBAR_STATES states a
    random policy visits (encoder seed of tasks.py), used by the weight init only."""
    task = TASKS[cfg.task]
    base = critic_layer(task, cfg.spec)
    phi = lambda o: np.repeat(base(o), cfg.in_rep)  # noqa: E731
    key = (cfg.task, cfg.spec, cfg.in_rep)
    if key not in _FBAR:
        _FBAR[key] = np.mean([phi(o) for o in random_states(task.env_id, task.n_actions, FBAR_STATES, 777)], 0)
    return phi, _FBAR[key]


class Runner:
    """Legacy (pre-audit) loop: H env copies on shared W (Layout of mapstdp.tdstdp), stepped in 20 ms ticks."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        self.cfg, self.frozen = cfg, frozen
        task = TASKS[cfg.task]
        self.K, self.H = task.n_actions, cfg.heads
        self.phi, fbar = features(cfg)
        self.lay = lay = td.Layout(len(fbar), cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, self.K)
        self.rng = np.random.default_rng((cfg.seed, rng_seed))
        assert cfg.engine != "nlif", "engine nlif: authors' code only"
        self.lif = cfg.engine.startswith("lif")  # true-LIF control: W in CK units, Table 1 values verbatim
        self.mv = 1.0 if self.lif else MV_LEGACY
        self.W = td.init_weights(lay, fbar, cfg.rate0, self.rng, td.THETA_L * td.TAU_M if self.lif else 1.0)
        nc, H = lay.n_crit, self.H
        self.eta = np.r_[np.full(nc, cfg.eta_c), np.full(lay.n_post - nc, cfg.eta_a)][:, None] / self.mv
        self.c_w = cfg.c_w * self.mv
        self.v = np.zeros((H, lay.n_post))  # LIF membranes
        self.opt = td.Adam(self.W.shape, cfg.adam_b1, cfg.adam_b2) if cfg.adam else None
        self.envs = [gym.make(task.env_id) for _ in range(H)]
        self.obs = [None] * H
        self.ret, self.steps, self.warm, self.action = np.zeros(H), np.zeros(H, int), np.zeros(H, int), np.zeros(H, int)
        self.returns, self.done = [[] for _ in range(H)], np.zeros(H, bool)
        self.z, self.q = np.zeros((H, lay.n_post, lay.n_in)), np.zeros((H, lay.n_post - nc, lay.n_in))
        self.P, self.xprev, self.rho = np.zeros((H, lay.n_in)), np.zeros((H, lay.n_in)), np.zeros((H, lay.n_post))
        self.Vprev, self.rprev = np.zeros(H), np.zeros(H)
        self.zprev, self.cact = np.zeros(H, bool), np.zeros(H, bool)
        self.s_last = np.full((H, self.K), 1 / self.K)
        self.frames, self.snn = 0, [None] * H
        self.gsq, self.nupd = np.zeros(2), 0  # calibration: summed squared block update per population
        self.stats = np.zeros(5)  # sums of: critic rate, actor rate, policy entropy, V, ticks
        for h in range(H):
            self.reset(h)

    def reset(self, h):
        self.obs[h] = self.envs[h].reset(seed=int(self.rng.integers(1 << 31)))[0]
        self.ret[h], self.steps[h], self.warm[h] = 0.0, 0, WARM_TICKS
        for a in (self.z, self.q, self.P, self.xprev, self.rho, self.v):
            a[h] = 0
        self.cact[h] = False

    def tick(self):
        cfg, lay, H, K = self.cfg, self.lay, self.H, self.K
        f, r_ms = np.zeros((H, lay.n_in)), np.zeros(H)
        final, learn, A = np.zeros(H, bool), np.zeros(H, bool), np.zeros((H, K))
        for h in range(H):
            if self.done[h]:
                continue
            f[h] = self.phi(self.obs[h])
            if self.warm[h]:
                self.warm[h] -= 1
                continue
            if self.steps[h] % cfg.resample == 0:
                self.action[h] = self.rng.choice(K, p=self.s_last[h])
            o2, r, term, trunc, _ = self.envs[h].step(int(self.action[h]))
            r_ms[h], final[h], learn[h], A[h, self.action[h]] = cfg.r_scale * r / MS, term or trunc, True, 1
            self.ret[h] += r
            self.steps[h] += 1
            self.obs[h] = o2
        self.frames += int(learn.sum())
        for b in range(MS // cfg.block):
            self.learn_block(f, r_ms, final, learn, A, np.arange(b * cfg.block, (b + 1) * cfg.block))
        for h in np.flatnonzero(final):
            self.returns[h].append(float(self.ret[h]))
            if len(self.returns[h]) >= cfg.episodes:
                self.done[h] = True
            else:
                self.reset(h)

    def learn_block(self, f, r_ms, final, learn, A, ms):
        cfg, lay, L, nc = self.cfg, self.lay, len(ms), self.lay.n_crit
        x, noise = td.draw(f, L, lay.n_post, self.rng)
        if self.lif:
            X, self.v = td.simulate_lif(self.W, self.xprev, x, self.v, "exp" if cfg.engine == "lif" else "snm")
        elif cfg.engine == "snm":
            X = np.concatenate([self._snm(h, x[:, h:h + 1], noise[:, h:h + 1]) for h in range(self.H)], 1)
        else:
            X = td.simulate(self.W, self.xprev, x, noise)
        P = td.pre_trace(self.P, x, cfg.tau_p)
        R = td.rates(self.rho, X, cfg.tau_n)
        V = cfg.alpha_c * R @ lay.Mv + cfg.beta_c  # Eq. 4
        s = td.softmax(cfg.alpha_a * R @ lay.Ma.T)  # Eq. 12
        zero = final[None] & (ms >= MS - 2)[:, None]  # target V := 0 in the last 2 ms
        eV, e2 = np.exp(-1 / cfg.tau_d), np.exp(-1 / (2 * cfg.tau_d))
        d = np.zeros((L + 1, self.H))  # Eq. 69 for t = -1..L-1
        d[0] = self.cact * (eV * V[0] * ~self.zprev + e2 * self.rprev - self.Vprev)
        d[1:L] = learn * (eV * V[1:] * ~zero[:-1] + e2 * r_ms - V[:-1])
        d[L] = (learn & zero[-1]) * (e2 * r_ms - V[-1])
        act, sa = lay.act, lay.sign[nc:]
        c = learn[:, None] * (A[:, act] - s[..., act])  # Eq. 77 feedback A_k - s_k
        kap = learn[:, None] * (cfg.c_e * sa * td.entropy_coef(s)[..., act]
                                - cfg.c_t * (R[..., nc:][..., lay.ae].mean(-1, keepdims=True) - cfg.rho_hat) * lay.ae)
        G, self.z, self.q = td.block_learn(self.z, self.q, P, X, d, c, kap, lay, cfg.tau_z, cfg.tau_q)
        n = int((learn | self.cact).sum())
        if n:
            g = G.sum(0) / n - 0.5 * self.c_w * L * self.W  # Eqs. 73, 78, 26, 27, 31
            if self.frozen:
                self.gsq += [(g[:nc] ** 2).mean(), (g[nc:] ** 2).mean()]
                self.nupd += 1
            elif self.opt is not None:
                self.W += self.opt.step(g, self.eta, L)
            else:
                self.W += self.eta * g
        self.xprev, self.P, self.rho, self.s_last = x[-1], P[-1], R[-1], s[-1]
        self.Vprev, self.rprev, self.zprev, self.cact = V[-1], r_ms, zero[-1], learn & ~zero[-1]
        m = learn | self.cact
        if m.any():
            self.stats += [R[..., :nc][:, m].mean(), R[..., nc:][:, m].mean(),
                           -(s[:, m] * np.log(np.maximum(s[:, m], 1e-300))).sum(-1).mean(), V[:, m].mean(), 1]

    def _snm(self, h, x, noise):
        X, self.snn[h] = td.simulate_snm(self.W, self.xprev[h:h + 1], x, noise, self.snn[h])
        return X


def calibrate(cfg):
    """Legacy E4 rule (Adam-equivalent step at init): eta per population (critic, actor) = Adam step per ms / RMS of the
    per-ms update (averaged over heads), measured per ms (block 1) with the initial W frozen over CAL_TICKS ticks of a
    run on the (seed, CAL_SEED) stream: an RMS-sized per-ms update moves W by the step per-ms Adam takes at init
    (|m_hat| / sqrt(v_hat) = 1). Returns (eta, rms) in CK units."""
    r = Runner(dataclasses.replace(cfg, adam=False, sgd=False, block=1), rng_seed=CAL_SEED, frozen=True)
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    return np.array([cfg.eta_c, cfg.eta_a]) / rms, rms


def ck_encoder(cfg):
    """util.State_to_spike_Fourier: the observation scaled to [0, 1] by CK_BOUNDS (clipped), coefficient vectors
    {0..order}^d (cross terms, the constant included) or order 1..f_order per dimension; (1 + cos) / 2 or the ON/OFF
    split ([cos]+, [-cos]+); a constant unit if f_basis; the whole vector repeated in_rep times. Returns
    (phi: H x d observations -> H x n_in input rates, n_in). onehot: a Discrete observation's one-hot (iteration
    016)."""
    if cfg.onehot:
        n = gym.make(TD_TASKS[cfg.task].env_id).observation_space.n
        return lambda o: np.tile(np.eye(n)[np.asarray(o).astype(int)], cfg.in_rep), int(n) * cfg.in_rep
    lo, hi = (np.array(v, float) for v in CK_BOUNDS[cfg.task])
    n, k = len(lo), cfg.f_order
    if cfg.f_cross:
        C = np.array(list(itertools.product(range(k + 1), repeat=n)), float)[:, ::-1].T  # column i: digits of i
    else:
        C = np.zeros((n, n * k))
        for j, m in itertools.product(range(n), range(k)):
            C[j, j * k + m] = m + 1

    def phi(o):
        c = np.cos(np.pi * np.clip((np.asarray(o, float) - lo) / (hi - lo), 0, 1) @ C)
        f = np.concatenate([np.maximum(c, 0), np.maximum(-c, 0)], -1) if cfg.f_double else (1 + c) / 2
        if cfg.f_basis:
            f = np.concatenate([f, np.ones((*f.shape[:-1], 1))], -1)
        return np.tile(f, cfg.in_rep)
    return phi, C.shape[1] * (2 if cfg.f_double else 1) * cfg.in_rep + cfg.f_basis * cfg.in_rep


class CKRunner:
    """The authors' closed loop (reference/td_stdp: main.py, bindsnet_add.Snn_actor / Reward_fn, util.batch_envs):
    H env copies on shared W in lockstep, one 20 ms window per tick. Per window: copies whose env was just reset
    (code 2) clear traces, rates and membranes (every copy's previous input spikes too, as BindsNET's Input layer
    reset ignores the index); every copy presents its current observation with (A_(k-1), p_(k-1), r_(k-1)); delta = 0
    in reset (2) and warm-up (3) windows; V' := 0 in the last 2 ms only after a true termination (bootstrapped at
    truncation); the entropy term applies in every window; the update is the mean over all H copies (BindsNET's batch
    reduction). Then p = softmax at the window's end, the action is resampled when the copy's timer reaches `freeze`
    ms, and the env copies step: an ended copy resets, then WARM_N warm-up windows, so s_0 is presented 6 times
    (5 + 1 on the first episode, whose first window is live as in main.py). Episodes are pooled over copies in the
    order they end; the run stops when solved with >= `episodes` pooled episodes, or at `max_eps`.
    Engine nlif: the membranes (lif state lst: V, ref, z) are reset to mu_bg on code 2 like BindsNET's to rest; the
    neuron noise draws from nrng. probe = True (calibrate_s_in) skips the rule and counts spikes in spk.
    rng_seed (an int, or a tuple of ints for a probe's own stream, reprobe_race) follows cfg.seed in every stream's
    seed: rng (cfg.seed, *rng_seed), nrng (..., 1), HybridRec (..., 2)."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        assert cfg.ck and cfg.engine in ("numpy", "snm", "lif", "flow", "nlif") and MS % cfg.block == 0
        assert cfg.engine != "nlif" or cfg.s_in > 0, "engine nlif needs s_in (run_ck calibrates it)"
        assert not cfg.rate_target or cfg.engine == "nlif", "rate_target: engine nlif"
        assert cfg.rec in ("none", "cols", "lat")
        assert cfg.rec == "none" or (cfg.engine == "nlif" and cfg.block == MS and not cfg.adam), \
            "rec: engine nlif, block 20, no Adam"
        self.cfg, self.frozen = cfg, frozen
        task = TD_TASKS[cfg.task]
        self.K, self.H = K, H = task.n_actions, cfg.heads
        self.phi, n_in = ck_encoder(cfg)
        n_lat = cfg.n_lat_mod * cfg.n_ae if cfg.rec == "lat" else 0
        self.lay = lay = td.Layout(n_in, cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, K, n_lat)
        rs = (cfg.seed, *(rng_seed if isinstance(rng_seed, tuple) else (rng_seed,)))
        self.rng = np.random.default_rng(rs)
        self.lif = cfg.engine == "lif"  # BindsNET LIF, W in mV
        self.nlif = cfg.engine == "nlif"  # noisy LIF (mapstdp.lif, dt 1 ms), W in mV
        self.mv = 1.0 if self.lif or self.nlif else MV
        self.probe, self.spk = False, np.zeros(2)  # s_in calibration: summed post spikes, spike slots
        if self.nlif:
            self.lp = dataclasses.replace(lif.LIFParams(), **(lif.DT1R if cfg.rate_target else lif.DT1))
            self.lst = lif.init_state(lay.n_post, H, self.lp)
            self.lst["V"] = np.full((H, lay.n_post), self.lp.mu_bg)
            self.nrng = np.random.default_rng((*rs, 1))
        self.W = self.rng.uniform(0, cfg.init_scale / self.mv, (lay.n_post, lay.n_in))
        nc, na = lay.n_crit, lay.n_post - lay.n_crit
        self.nu = np.r_[np.full(nc, cfg.eta_c), np.full(na, cfg.eta_a)][:, None]  # mV
        # the code folds nu into the pre trace, so its Adam sees nu g: eps on our unit-trace gradient g is eps / nu
        self.opt = td.Adam(self.W.shape, cfg.adam_b1, cfg.adam_b2, cfg.adam_eps / self.nu) if cfg.adam else None
        self.envs = [gym.make(task.env_id) for _ in range(H)]
        self.obs = np.array([e.reset(seed=int(self.rng.integers(1 << 31)))[0] for e in self.envs], float)
        self.reward, self.is_end, self.trunc = np.zeros(H), np.zeros(H, bool), np.zeros(H, bool)
        self.code, self.warm, self.rest = np.zeros(H, int), np.zeros(H, int), np.zeros(H, int)
        self.p, self.action, self.timer = np.full((H, K), 1 / K), self.rng.choice(K, size=H), np.zeros(H)
        assert not cfg.norm_logits or cfg.rate_target > 0, "norm_logits: the reference r0 is rate_target"
        self.apost = np.arange(nc, nc + K * (cfg.n_ae + cfg.n_ai))  # actor posts (latents excluded)
        self.r0 = cfg.rate_target * 1e-3  # norm_logits reference, per ms
        # CK's target-rate term [CK Eq. 31; bindsnet_add targ_adj] (016 §18 fix b): per ms, on the copy's actor E rows,
        # kap -= c_t (mean actor-E rate - rho_hat); rho_hat defaults to rate_target, the rate every arm is matched to
        # (off in the frozen calibration probes, which measure CK's backbone at W fixed)
        self.c_t = 0.0 if frozen else cfg.c_t
        self.rho_hat = cfg.rho_hat if cfg.rho_hat > 0 else cfg.rate_target * 1e-3
        assert self.c_t >= 0 and (not self.c_t or self.rho_hat > 0), "c_t: rho_hat or rate_target"
        # the one-sided per-column rate floor (016 §20, the neuroscientist's contrast floor, added to the mean pin):
        # per ms, column k's E rows get +c_t (rho_min - rho_k)_+, rho_k the mean rate of its E posts and rho_min =
        # rate_floor x rate_target; zero at or above rho_min, so it potentiates a deprived column and never lowers a
        # winner. It uses the mean term's coefficient (self.c_t: c_t_eff under a score credit; 0 when frozen).
        assert cfg.rate_floor >= 0 and (not cfg.rate_floor or frozen or (cfg.c_t > 0 and cfg.rate_target > 0)), \
            "rate_floor: c_t > 0 (--rate-cost) and rate_target"  # frozen probes (engine lif, rate_target 0): exempt
        self.rho_min = cfg.rate_floor * cfg.rate_target * 1e-3
        self.floor = bool(cfg.rate_floor) and self.c_t > 0
        self.eps_n = cfg.alpha_a * self.r0 / (cfg.tau_n * cfg.n_ae)  # norm_logits floor: one spike moves <= one nat
        self.prk, self.prbar = np.zeros((H, K)), np.zeros(H)  # norm_logits / race: column rates (pooled rate) of self.p
        # readout race (016 §14.1 Stage B): theta by the pre-set rule, the surrogate's floor eps_r, the columns' signed
        # count matrix, per copy the net counts since the last resample and the pending winner (-1: none yet)
        assert cfg.readout in ("softmax", "race"), cfg.readout
        self.race = cfg.readout == "race"
        assert not self.race or (cfg.engine == "nlif" and cfg.rate_target > 0), "readout race: engine nlif, rate_target"
        assert not (self.race and (cfg.norm_logits or cfg.ent_match_rec)), "norm_logits, ent_match_rec: softmax only"
        self.theta = race_theta(cfg)
        self.eps_r = 1.0 / (cfg.tau_n * cfg.n_ae)
        # the surrogate's gain: cfg.race_beta at init (calibrate_race's); with race_reprobe, reprobe_race replaces it
        # every N episodes and appends its record to race_cal_history (both carried by the checkpoint)
        self.beta, self.race_cal_history = cfg.race_beta, []
        # the tally counts a column's E posts only (user decision 2026-10-29, 016 §18: the I posts shape the dynamics
        # but are not counted, so theta means the same on every task; without I posts it is the net count of Eq. 1h)
        self.Mcnt = (lay.Ma[:, self.apost] > 0).astype(float)  # K x n_act
        # the surrogate reads what the race tallies (016 §21): the column's E rate r_E,k = mean rate of its E posts,
        # R @ MaE.T with MaE = Ma with its I entries (-1 / n_ai) zeroed, i.e. (R[apost] @ Mcnt.T) / n_ae up to
        # rounding. Where n_ai = 0 MaE is Ma itself, so the product is bit-identical to the net rate R @ Ma.T (CK's
        # score, which the softmax readout keeps). With I posts the net rate E - I sits near 0, below eps_r, and the
        # credit factor was mostly dead (LunarLander).
        self.MaE = np.where(lay.Ma > 0, lay.Ma, 0.0)  # K x n_post
        self.cnt, self.pend = np.zeros((H, K)), np.full(H, -1)
        # rate_floor: the column -> actor-row expansion (K x n_act + n_lat; 1 on column k's E posts, 0 on I posts and
        # latents), and the race monitor's sums: per-column rate, per-ms minimum over columns, ms x copies in an episode
        self.Mfl = np.zeros((K, na))
        self.Mfl[:, :len(self.apost)] = self.Mcnt
        self.fsum = np.zeros(K + 2)
        # the score credit (016 §20, --credit hazard | noise): the actor E posts' own scores replace the surrogate
        assert cfg.credit in ("surrogate", "hazard", "noise"), cfg.credit
        self.score = cfg.credit != "surrogate"
        assert not self.score or (self.race and self.nlif), "credit hazard | noise: readout race, engine nlif"
        assert not self.score or cfg.race_reprobe == 0, "credit hazard | noise: race_reprobe must be 0"
        assert not self.score or frozen or cfg.sgd, "credit hazard | noise: --eta-cal"
        self.c_e = 0.0 if self.score else cfg.c_e  # no policy under the score: the entropy bonus is off
        self.cE = (lay.sign[self.apost] > 0).astype(float)  # the post factor's mask: E actor posts (I posts: 0)
        # under the score the learner is called twice: the critic rows as today (X, P), the actor rows with Y paired
        # with x(t - 1) and no sign (Y carries the post's sign; the pre's enters through W's sign convention)
        if self.score:  # the learner reads n_crit, n_post and sign only
            self.lay_c, self.lay_a = copy.copy(lay), copy.copy(lay)
            self.lay_c.n_post, self.lay_c.sign = nc, lay.sign[:nc]
            self.lay_a.n_crit, self.lay_a.n_post, self.lay_a.sign = 0, na, np.ones(na)
        self.was, self.bnd = np.ones(H, bool), np.full(H, -1)  # race_block: no decision pending before the block;
        self.ssum = np.zeros(4)  # the bound ms within it (-1: none). Sums: |Y| and count over active actor-E ms,
        self.abin = np.zeros((H, K))  # actor-E spikes at h < H_CLIP and all; the bin's realised actions per copy
        self.arep = np.zeros((H, 2))  # per copy in the bin: resamples whose action repeats the previous one, resamples
        self.slog = {k: [] for k in SLOG_KEYS}
        self.rdec = np.zeros(3)  # race decisions: at the bound, at the deadline, tie draws
        self.wspk = np.zeros(H)  # actor spikes (E and I) per copy since its window started (spikes to decision)
        self.ebnd = np.zeros(2)  # energy: summed actor spikes up to the decision (bound ms, else the window), decisions
        self.hold = False  # calibrate_race: a held observation, no env step, no reset; decisions tallied in rwin
        self.rwin, self.rsum, self.rn = np.zeros(K), np.zeros(K), 0  # hold: winners, summed column rates, ticks
        self.espk = np.zeros(2)  # energy: summed actor spikes (E and I posts, all copies), copy-decision windows
        self.z, self.q = np.zeros((H, lay.n_post, lay.n_in)), np.zeros((H, na, lay.n_in))
        self.P, self.xprev = np.zeros((H, lay.n_in)), np.zeros((H, lay.n_in))
        self.rho, self.v, self.Vprev = np.zeros((H, lay.n_post)), np.zeros((H, lay.n_post)), np.zeros(H)
        self.c_ret, self.returns, self.returns_heads = np.zeros(H), [], [[] for _ in range(H)]
        self.done, self.solved, self.frames, self.snn = False, False, 0, [None] * H
        self.gsq, self.nupd = np.zeros(2), 0  # calibration: summed squared per-ms update per population
        # sums of: critic rate, actor rate, policy entropy, V, ticks; then (016 §14.1) actor spikes, copy-decision
        self.stats = np.zeros(7)  # windows
        self.lowrate = None  # rate_target: calibrate_low's record (run_ck sets it; saved in checkpoints)
        self.hr = None  # rec: tdstdp.HybridRec (own stream (seed, rng_seed, 2)); W rows of latent posts frozen
        if cfg.rec != "none":
            self.hr = td.HybridRec(lay, K, cfg.n_ae, cfg.n_lat_mod if n_lat else 0, cfg.chi_rec, self.lp, H,
                                   (*rs, 2), cfg.stat_windows, cfg.pair_ms, cfg.lam * cfg.lam_rec,
                                   cfg.la_beta if n_lat else 0.0, p_cross=cfg.p_cross, seal=cfg.rec_seal)
            self.hs = np.zeros((H, self.hr.Kr))  # module scores of the window's credit
            self.snaps = {0: self.hr.R.copy()}
            self.snaps_kc = {}  # per snapshot: HybridRec.class_table() of the pairs since the previous snapshot
            self.rsq = np.zeros(5)  # rec_cal sums: actor g^2, actor W^2 (per entry), task dir^2, frame task^2, dS^2
            self.rcnt = np.zeros(3)  # counts: windows, frames (task), frames (structural)
            self.ftask = 0.0  # rec_cal: task direction summed over the current frame

    def tick(self):
        cfg, lay, K = self.cfg, self.lay, self.K
        rs = self.code == 2
        if rs.any():
            for a in (self.z, self.q, self.P, self.rho, self.v):
                a[rs] = 0
            self.xprev[:] = 0
            if self.nlif:
                self.lst["V"][rs], self.lst["ref"][rs], self.lst["z"][rs] = self.lp.mu_bg, 0, False
            if self.hr is not None:
                self.hr.reset_heads(rs)
            if self.race:
                self.cnt[rs], self.pend[rs], self.wspk[rs] = 0, -1, 0
        act, sa = lay.act, lay.sign[lay.n_crit:]
        Ap = np.eye(K)[self.action] - self.p  # (A - p) with p fixed at the sampling probability
        ge = td.entropy_coef(self.p)  # Eq. 25, from the window's p
        if cfg.norm_logits:  # the exact gradient through the normalisation (016 §13): fac (A - p - u), fac (g - ug)
            fac, u, ug = self.norm_factors(ge)
            Ap, ge = fac[:, None] * (Ap - u[:, None]), fac[:, None] * (ge - ug[:, None])
        elif self.race:  # the surrogate's exact gradient (016 §14.1): the per-column factor on (A - p~) and on g(p~)
            fr = self.race_factor(self.prk)
            Ap, ge = fr * Ap, fr * ge
        c = Ap[:, act]
        kap = self.c_e * sa * ge[:, act]  # Eq. 26 (c_e = 0 under the score credit)
        if self.race:  # the surrogate is a function of the E rates only (MaE), so its credit and entropy term go to
            c, kap = np.where(lay.ae, c, 0.0), np.where(lay.ae, kap, 0.0)  # the E actor rows (I posts, latents: 0)
        if self.hr is not None:  # CK's action score per module (latents 0; their W rows get no update)
            if not self.score:  # the score credit sets hs from the window's scores in learn_block
                self.hs[:, :K] = cfg.alpha_a * Ap / cfg.n_ae
            if lay.n_lat:
                c[:, lay.lat], kap[:, lay.lat] = 0.0, 0.0
        f, r_ms = self.phi(self.obs), cfg.r_scale * self.reward / MS
        end, learn = self.is_end & ~self.trunc, self.code < 2
        for b in range(MS // cfg.block):
            self.learn_block(f, r_ms, end, learn, c, kap, np.arange(b * cfg.block, (b + 1) * cfg.block))
        if self.race:  # p~ of the window's column E rates; the action is the race's pending winner, else the leader now
            self.prk = self.rho @ self.MaE.T
            self.p = self.surrogate(self.prk)
        else:
            sc, self.prk, self.prbar = self.scores(self.rho)
            self.p = td.softmax(sc)
        self.timer += MS
        nb = self.timer >= cfg.freeze
        prev = self.action.copy()
        if self.race:
            for h in np.flatnonzero(nb):
                if self.pend[h] < 0:  # the deadline: the leader at the end of the decision window (Eq. 1h)
                    self.pend[h] = self._leader(self.cnt[h])
                    self.rdec[1] += 1
                    self.ebnd += [self.wspk[h], 1]
                self.action[h] = self.pend[h]
            ok = nb & (self.code < 2)  # the bin's realised actions of copies in an episode (race_entropy) and
            self.abin[ok, self.action[ok]] += 1  # whether each resample repeats the previous decision (the held
            self.arep[ok, 0] += self.action[ok] == prev[ok]  # steps between resamples are not counted)
            self.arep[ok, 1] += 1
            self.cnt[nb], self.pend[nb], self.wspk[nb] = 0, -1, 0
        else:
            for h in np.flatnonzero(nb):
                self.action[h] = self.rng.choice(K, p=self.p[h])
        self.timer[nb] = 0
        if self.hold:  # calibrate_race: tally the decisions and the column rates; the env is neither stepped nor reset
            self.rwin += np.bincount(self.action[nb], minlength=K)
            self.rsum += self.prk.sum(0)
            self.rn += self.H
            return
        if self.hr is not None and nb.any():  # perseveration monitor (copies in an episode)
            ok = nb & (self.code < 2)
            if ok.any():
                self.hr.log_actions(float((self.action[ok] == prev[ok]).mean()), float((self.p[ok] ** 2).sum(1).mean()))
        was_end = self.is_end.copy()  # batch_envs.step (rest_n = 0)
        self.rest[self.is_end] += 1
        self.warm += 1
        for h in np.flatnonzero(self.rest > 0):
            self.obs[h] = self.envs[h].reset(seed=int(self.rng.integers(1 << 31)))[0]
            self.reward[h], self.rest[h], self.warm[h], self.trunc[h], self.is_end[h] = 0, 0, 0, False, False
        live = (self.warm > WARM_N) & ~self.is_end
        self.reward[~live] = 0
        for h in np.flatnonzero(live):
            o, r, term, trunc, _ = self.envs[h].step(int(self.action[h]))
            self.obs[h], self.reward[h], self.is_end[h], self.trunc[h] = o, r, term or trunc, trunc and not term
        self.frames += int(live.sum())
        self.code[live] = 0
        self.code[self.rest >= 1] = 1
        self.code[self.warm <= WARM_N] = 3
        self.code[self.warm == 0] = 2
        self.c_ret += self.reward  # main.py: pooled episode returns
        new = ~was_end & self.is_end
        for h in np.flatnonzero(new):
            self.returns.append(float(self.c_ret[h]))
            self.returns_heads[h].append(float(self.c_ret[h]))
        self.c_ret[new] = 0
        if new.any():
            n = len(self.returns)
            if self.hr is not None:
                self.hr.flush(n)
                if n // SNAP_EVERY[cfg.task] > (n - int(new.sum())) // SNAP_EVERY[cfg.task]:
                    k = n // SNAP_EVERY[cfg.task] * SNAP_EVERY[cfg.task]
                    self.snaps[k], self.snaps_kc[k] = self.hr.R.copy(), self.hr.class_table()
            if not self.solved and n > 100:
                self.solved = trailing_mean_time(self.returns, SOLVE[cfg.task][0]) is not None
            self.done = (self.solved and n >= cfg.episodes) or n >= cfg.max_eps

    def learn_block(self, f, r_ms, end, learn, c, kap, ms):
        cfg, lay, L, nc, H = self.cfg, self.lay, len(ms), self.lay.n_crit, self.H
        K = None
        if cfg.engine == "flow":  # noise-free limit: no spike draws; z gets the input autocorrelation term
            x, X, K0 = td.flow_step(self.W, self.xprev, f, L)
            K = np.exp(-1 / cfg.tau_p) * K0
        else:
            x, noise = td.draw(f, L, lay.n_post, self.rng)
            if self.lif:
                X, self.v = td.simulate_bindsnet(self.W, self.xprev, x, self.v)
            elif self.nlif:
                rec = (self.hr.bt(), self.hr.ridx) if self.hr is not None else (None, ())
                out = td.simulate_nlif(self.W, self.xprev, x, self.lst, self.nrng, self.lp, cfg.s_in, *rec, self.score)
                X, xs, xq = out[:3]
            elif cfg.engine == "snm":
                X = np.concatenate([Runner._snm(self, h, x[:, h:h + 1], noise[:, h:h + 1]) for h in range(H)], 1)
            else:
                X = td.simulate(self.W, self.xprev, x, noise)
        P = td.pre_trace(self.P, x, cfg.tau_p)
        R = td.rates(self.rho, X, cfg.tau_n)
        V = cfg.alpha_c * R @ lay.Mv + cfg.beta_c  # Eq. 4
        if self.race:
            self.race_block(X)
        ea = [X[..., self.apost].sum(), H * L / cfg.freeze]  # energy monitor (016 §14.1), both readouts: actor spikes
        self.espk += ea  # (E and I posts, all copies) and copy-decision windows; espk over the run, stats per log bin
        self.stats[5:] += ea
        if self.probe:  # calibrate_s_in: rates only, W frozen
            self.spk += [X.sum(), X.size]
            self.xprev, self.P, self.rho, self.Vprev = x[-1], P[-1], R[-1], V[-1]
            return
        live = (ms < MS - 2)[:, None] | ~end  # Reward_fn.isLive
        Vp = np.concatenate([self.Vprev[None], V[:-1]])
        d = learn * (np.exp(-1 / (2 * cfg.tau_d)) * r_ms + np.exp(-1 / cfg.tau_d) * V * live - Vp)  # Eq. 69
        bc = lambda a: np.broadcast_to(a, (L, *a.shape))  # noqa: E731
        kapb = bc(kap)  # L x H x n_act
        rdev = R[..., nc:][..., lay.ae].mean(-1) - self.rho_hat  # L x H: the copy's actor-E mean rate against rho_hat
        if self.c_t and not self.score:  # CK's target-rate term, per ms, E actor rows only
            kapb = kapb - self.c_t * rdev[..., None] * lay.ae
        if self.floor:  # the per-column floor (module doc): per ms and copy, c_t (rho_min - rho_k)_+ for column k
            flc = self.c_t * np.maximum(self.rho_min - R[..., self.apost] @ self.Mcnt.T / cfg.n_ae, 0.0)  # L x H x K
            if not self.score:  # on column k's E rows (0 on I posts and latents), riding z like the mean term
                kapb = kapb + flc @ self.Mfl
        if self.score:  # the score credit (module doc): Y on the actor E posts up to the bound, paired with x(t - 1)
            ap = self.apost  # the scores of the actor posts only (the critic's are not used)
            Y, hz = lif.post_scores(X[..., ap], out[3][..., ap], out[4][..., ap], self.lp, cfg.credit)
            Ya = Y * self.live_mask(L)[..., None] * self.cE
            na, ne = lay.n_post - nc, len(self.cE)
            Yf = np.zeros((L, H, na))  # latent posts 0
            Yf[..., :ne] = Ya
            cl = np.zeros(na)  # c = 1 on E actor posts, 0 on I posts and latents
            cl[:ne] = self.cE
            xsh = np.concatenate([self.xprev[None], x[:-1]])  # x(t - 1): the input the kernel paired with step t
            fn = td.block_learn_ck_fast if cfg.fast else td.block_learn_ck
            Gc, zc, _ = fn(self.z[:, :nc], self.q[:, :0], P, X[..., :nc], d, np.zeros((L, H, 0)), np.zeros((L, H, 0)),
                           self.lay_c, cfg.tau_z, cfg.tau_q)
            Ga, za, self.q = fn(self.z[:, nc:], self.q, xsh, Yf, d, bc(np.broadcast_to(cl, (H, na))),
                                np.zeros((L, H, na)), self.lay_a, cfg.tau_z, cfg.tau_q)
            G, self.z = np.concatenate([Gc, Ga], 0 if cfg.fast else 1), np.concatenate([zc, za], 1)
            g = G / H if cfg.fast else G.mean(0)
            actE = ~np.isnan(out[3][..., ap]) & (self.cE > 0)
            SE = X[..., ap] & (self.cE > 0)
            self.ssum += [np.abs(Y[actE]).sum(), actE.sum(), (hz[SE] < lif.H_CLIP).sum(), SE.sum()]
        elif cfg.fast or K is not None:
            G, self.z, self.q = td.block_learn_ck_fast(self.z, self.q, P, X, d, bc(c), kapb, lay, cfg.tau_z,
                                                       cfg.tau_q, K)
            g = G / H
        else:
            G, self.z, self.q = td.block_learn_ck(self.z, self.q, P, X, d, bc(c), kapb, lay, cfg.tau_z, cfg.tau_q)
            g = G.mean(0)
        if self.score and self.c_t:  # the rate term, rank-1 per block (no per-synapse state): E[z_ij] ~ rho_i P_j
            g[self.apost] -= self.c_t / H * np.einsum("lhi,lhj->ij", rdev[..., None] * R[..., self.apost] * self.cE, P)
            if self.floor:  # the floor, rank-1 in the same form: +c_t sum_t (rho_min - rho_k)_+ rho_i P_j, column k's E
                g[self.apost] += np.einsum("lhi,lhj->ij", (flc @ self.Mcnt) * R[..., self.apost], P) / H
        if lay.n_lat:
            g[nc:][lay.lat] = 0.0  # input -> latent W frozen at init
        if self.frozen:
            # the actor RMS over the credited rows: under the race only the E actor rows receive the policy credit
            # (016 §21), so the I rows would dilute the E4' rule's RMS and inflate eta_a by sqrt(2) where n_ai > 0
            ga = g[nc:][self.lay.ae] if self.race else g[nc:]
            self.gsq += [(g[:nc] ** 2).mean(), (ga ** 2).mean()]
            self.nupd += 1
        elif self.opt is not None:
            self.W += self.opt.step(g, self.nu / self.mv, L)
        else:
            self.W += self.nu / self.mv * g
        if self.hr is not None:
            if self.score:  # R's credit: the column's summed score up to the bound (latents 0), used from the next
                hs = np.zeros_like(self.hs)  # window on, when its counter is the previous window's (rec_block)
                hs[:, :self.K] = np.einsum("lhi,ki->hk", Ya, self.Mcnt) / cfg.n_ae
            self.rec_block(X, xs, xq, d, g)
            if self.score:
                self.hs = hs
        self.xprev, self.P, self.rho, self.Vprev = x[-1], P[-1], R[-1], V[-1]
        if learn.any():
            if self.race:
                s = self.surrogate(R[:, learn] @ self.MaE.T)  # H(s) monitor: the surrogate of the E rates
                rk = R[:, learn][..., self.apost] @ self.Mcnt.T / cfg.n_ae  # L x h x K: the columns' mean E rates
                self.fsum += [*rk.sum((0, 1)), rk.min(-1).sum(), rk[..., 0].size]  # col_rate_min(_ms) monitors
            else:
                s = td.softmax(self.scores(R[:, learn])[0])
            self.stats[:5] += [R[..., :nc][:, learn].mean(), R[..., nc:][:, learn].mean(),
                               -(s * np.log(np.maximum(s, 1e-300))).sum(-1).mean(), V[:, learn].mean(), 1]

    def scores(self, R):
        """Policy scores of rates R (..., n_post): alpha_a x column rates (Eq. 12); with norm_logits alpha_a r_k r0 /
        max(rbar, eps_n), rbar the pooled actor rate and r0 = rate_target, so the policy is invariant to a common change
        of all rates (iteration 016 §13). Returns (scores, column rates, rbar); the default path is CK's expression."""
        if not self.cfg.norm_logits:
            return self.cfg.alpha_a * R @ self.lay.Ma.T, None, None
        rk, rbar = R @ self.lay.Ma.T, R[..., self.apost].mean(-1)
        return self.cfg.alpha_a * (self.r0 / np.maximum(rbar, self.eps_n))[..., None] * rk, rk, rbar

    def norm_factors(self, ge):
        """norm_logits: of the window's p (self.prk, self.prbar) and action, the eligibility factor fac = r0 / max(rbar,
        eps_n) and the common corrections u = (r_a - sum_k p_k r_k) / (K rbar), ug = sum_k g_k r_k / (K rbar) (the
        exact gradients of log p_a and of H through the normalisation; 0 below the floor, where the normaliser is the
        constant eps_n). Broadcast scalars per copy."""
        K, rk, rbar = self.K, self.prk, self.prbar
        above = rbar >= self.eps_n
        den = K * np.maximum(rbar, self.eps_n)
        u = np.where(above, (rk[np.arange(self.H), self.action] - (self.p * rk).sum(1)) / den, 0.0)
        ug = np.where(above, (ge * rk).sum(1) / den, 0.0)
        return self.r0 / np.maximum(rbar, self.eps_n), u, ug

    def surrogate(self, rk):
        """readout race: the population surrogate of the race, p~ = softmax(beta log max(r_k, eps_r)) of column rates
        rk (..., K), the columns' E rates R @ MaE.T (what the race tallies, 016 §21); eps_r = 1 / (tau_n n_ae), the
        filtered rate one spike gives, so the log never diverges (016 §14.1). beta is self.beta (cfg.race_beta, or
        the last re-probe's)."""
        return td.softmax(self.beta * np.log(np.maximum(rk, self.eps_r)))

    def race_factor(self, rk):
        """readout race: the per-column eligibility factor beta / (alpha_a max(r_j, eps_r)) (H x K) of the window's
        column E rates rk, so that factor x (A_j - p~_j) x alpha_a / n_ae = d log p~_a / d rho_i for every E post i
        of column j (the exact gradient of the surrogate; test_race_gradient; the I posts do not enter r_j and get no
        credit, tick), and likewise for the entropy bonus with g(p~) (Eq. 25/26). Zero below the floor, where the
        surrogate is constant in r_j. A K-vector broadcast per copy."""
        fr = self.beta / (self.cfg.alpha_a * np.maximum(rk, self.eps_r))
        return np.where(rk > self.eps_r, fr, 0.0)

    def race_block(self, X):
        """readout race, Eq. 1h at 1 ms resolution on the block's raster X (L x H x n_post): each column's E count (its
        excitatory posts, Mcnt; the I posts are not tallied) accumulates in self.cnt since the copy's last resample;
        for a copy without a pending decision, the first ms at which any column's count reaches theta decides it, the
        leader at that ms winning (ties: one uniform draw). The actor spikes (E and I) up to the bound ms are the
        energy spent on that decision (ebnd); wspk carries the window's running total for the deadline case."""
        A = X[..., self.apost].astype(float)
        cum = self.cnt[None] + np.cumsum(A @ self.Mcnt.T, 0)  # L x H x K
        tot = self.wspk[None] + np.cumsum(A.sum(-1), 0)  # L x H
        self.cnt, self.wspk = cum[-1], tot[-1]
        self.was, self.bnd = self.pend < 0, np.full(self.H, -1)  # live_mask: pending before, bound ms in this block
        for h in np.flatnonzero(self.was):
            hit = np.flatnonzero((cum[:, h] >= self.theta).any(-1))
            if len(hit):
                self.pend[h] = self._leader(cum[hit[0], h])
                self.bnd[h] = hit[0]
                self.rdec[0] += 1
                self.ebnd += [tot[hit[0], h], 1]

    def live_mask(self, L):
        """The score credit's truncation at the decision (016 §20): ms t of the last race_block's block counts for a
        copy iff no decision was pending before the block (pend persists until the resample) and t is at or before the
        copy's bound ms in it (the bound ms counts; no bound: the whole block, so a deadline decision counts the whole
        window). L x H bool."""
        t = np.arange(L)[:, None]
        return self.was[None, :] & ((self.bnd[None, :] < 0) | (t <= self.bnd[None, :]))

    def score_monitors(self, episode):
        """The score credit's monitors of the bin since the last call, appended to self.slog (the result's 'race_log'):
        score_abs_mean (mean |Y| over active actor-E ms, truncation not applied), score_clip_frac (actor-E spikes at
        h < lif.H_CLIP), race_entropy (entropy, nats, of the realised actions of copies in an episode, per copy,
        averaged over the copies with a decision), col_rate_min (Hz; the minimum over columns of the column's mean E
        rate averaged over the bin's ms x copies in an episode: the marginal, perseveration monitor of the rate floor)
        and col_rate_min_ms (Hz; the bin average of the per-ms, per-copy minimum over columns: what the floor sees),
        action_repeat_probability (016 §21: the fraction of resamples, of copies in an episode, at which the new
        decision equals the previous one, per copy, averaged over the copies with a resample; the held env steps
        between resamples are not counted, so under a hold it is the race's own persistence). Resets the sums.
        Returns the row."""
        s, ab, fs, K, ar = self.ssum, self.abin, self.fsum, self.K, self.arep
        n = ab.sum(1)
        q = ab[n > 0] / n[n > 0, None]
        ent = float(-(q * np.log(np.maximum(q, 1e-300))).sum(1).mean()) if len(q) else float("nan")
        nf = fs[K + 1]
        row = dict(episode=int(episode), score_abs_mean=float(s[0] / max(s[1], 1)),
                   score_clip_frac=float(s[2] / max(s[3], 1)), race_entropy=ent,
                   col_rate_min=float(1e3 * fs[:K].min() / nf) if nf else float("nan"),
                   col_rate_min_ms=float(1e3 * fs[K] / nf) if nf else float("nan"),
                   action_repeat_probability=float((ar[ar[:, 1] > 0, 0] / ar[ar[:, 1] > 0, 1]).mean())
                   if (ar[:, 1] > 0).any() else float("nan"))
        for k, v in row.items():
            self.slog[k].append(v)
        self.ssum[:], self.abin[:], self.fsum[:], self.arep[:] = 0, 0, 0, 0
        return row

    def _leader(self, cnt):
        """The column with the largest count; an exact tie is broken by one uniform draw from self.rng."""
        tied = np.flatnonzero(cnt == cnt.max())
        if len(tied) == 1:
            return int(tied[0])
        self.rdec[2] += 1
        return int(self.rng.choice(tied))

    def hold_reset(self):
        """calibrate_race: a fresh observation (env copy 0's reset, broadcast to every copy), then hold_clear; the
        copies' membranes and traces carry on (the warm-up ticks remove the transient)."""
        self.obs[:] = self.envs[0].reset(seed=int(self.rng.integers(1 << 31)))[0]
        self.hold_clear()

    def hold_clear(self):
        """calibrate_race: the race, the timers and the tallies reset (a new probe condition)."""
        self.cnt[:], self.pend[:], self.timer[:] = 0, -1, 0
        self.rwin[:], self.rsum[:], self.rn = 0, 0, 0

    def rec_block(self, X, xs, xq, d, g):
        """rec (iteration 015, tdstdp.HybridRec), once per 20 ms window: the window's statistics, the one-hop credit
        of the previous window's counter with this window's (A - p) (the action sampled from that window's rates), the
        task step eta_rec mean_h (sum_t delta_ht) e_h on R, and at the end of a statistics frame the structural update
        and homeostasis. Frozen (calibrate_rec): R is not changed; the RMS sums of the actor's step, the task direction
        and the structural increment are accumulated instead."""
        hr, cfg, nc, lay = self.hr, self.cfg, self.lay.n_crit, self.lay
        ri = hr.ridx
        G, n = hr.window(X[..., ri], xs[:, ri], xq[:, ri], self.action)
        # under the score credit hs is a current-derivative score (sum_t Y, per theta of input), so it pairs with the
        # delivered current b n_j: credit's tau_eff b G_i n_j (s_m(i) - sbar_j) with G = 1 / tau_eff; the walk's
        # counter (hr.window) keeps the kernel's gain G
        hr.credit(self.hs, np.full_like(G, 1.0 / hr.p.tau_eff) if self.score else G, n, np.exp(-MS / cfg.tau_q))
        u = hr.task_dir(d.sum(0))
        if self.frozen:
            act = nc + np.flatnonzero(~lay.lat)
            self.rsq[:3] += [(g[act] ** 2).mean(), (self.W[act] ** 2).mean(), (u[hr.mask] ** 2).mean()]
            self.rcnt[0] += 1
            self.ftask = self.ftask + u
        else:
            hr.apply_task(cfg.eta_rec * u)
        dS = hr.frame()
        if dS is not None and self.frozen:
            self.rsq[3:] += [(self.ftask[hr.mask] ** 2).mean() if np.ndim(self.ftask) else 0.0,
                             (dS[hr.mask] ** 2).mean()]
            self.rcnt[1:] += 1
            self.ftask = 0.0

    def save(self, fn, wall, resumes):
        o = self.opt
        nl = dict(V_lif=self.lst["V"], ref_lif=self.lst["ref"], z_lif=self.lst["z"], s_in=self.cfg.s_in,
                  nrng=json.dumps(self.nrng.bit_generator.state)) if self.nlif else {}
        if self.lowrate is not None:
            nl["lowrate"] = json.dumps(self.lowrate)
        if self.race:  # the surrogate's current gain, the re-probe records (race_reprobe) and the bin monitors
            nl |= dict(race_beta=self.beta, race_cal_history=json.dumps(self.race_cal_history),
                       race_log=json.dumps(self.slog))
        if self.hr is not None:
            nl |= self.hr.state() | dict(eta_rec=self.cfg.eta_rec, lam_rec=self.cfg.lam_rec,
                                        snaps=json.dumps({k: v.tolist() for k, v in self.snaps.items()}),
                                        snaps_kc=json.dumps({k: {x: np.asarray(y).tolist() for x, y in v.items()}
                                                             for k, v in self.snaps_kc.items()}))
        np.savez(fn, W=self.W, m=o.m if o else 0, v=o.v if o else 0, t=o.t if o else 0, frames=self.frames,
                 wall=wall, resumes=resumes, returns=json.dumps([self.returns, self.returns_heads]),
                 rng=json.dumps(self.rng.bit_generator.state), espk=self.espk, rdec=self.rdec, ebnd=self.ebnd, **nl)

    def load(self, fn):
        """Resume from a checkpoint: W, Adam moments, pooled returns, rng; every copy starts a fresh episode with
        cleared traces (not bit-identical to an uninterrupted run); engine nlif also restores the membranes and the
        neuron-noise stream. Returns (wall, resumes)."""
        c = np.load(fn)
        self.W = c["W"]
        if self.nlif:
            assert float(c["s_in"]) == self.cfg.s_in, (float(c["s_in"]), self.cfg.s_in)
            self.lst["V"], self.lst["ref"], self.lst["z"] = c["V_lif"].copy(), c["ref_lif"].copy(), c["z_lif"].copy()
            self.nrng.bit_generator.state = json.loads(str(c["nrng"]))
        if self.hr is not None:
            assert (float(c["eta_rec"]), float(c["lam_rec"])) == (self.cfg.eta_rec, self.cfg.lam_rec)
            self.hr.load_state(c)
            self.snaps = {int(k): np.array(v) for k, v in json.loads(str(c["snaps"])).items()}
            if "snaps_kc" in c:
                self.snaps_kc = {int(k): {x: np.array(y) for x, y in v.items()}
                                 for k, v in json.loads(str(c["snaps_kc"])).items()}
        if self.opt is not None:
            self.opt.m, self.opt.v, self.opt.t = c["m"], c["v"], int(c["t"])
        self.returns, self.returns_heads = json.loads(str(c["returns"]))
        self.rng.bit_generator.state = json.loads(str(c["rng"]))
        self.frames = int(c["frames"])
        if "espk" in c:
            self.espk, self.rdec = c["espk"].copy(), c["rdec"].copy()
        if "ebnd" in c:
            self.ebnd = c["ebnd"].copy()
        if "race_beta" in c:
            self.beta, self.race_cal_history = float(c["race_beta"]), json.loads(str(c["race_cal_history"]))
        if "race_log" in c:
            self.slog = json.loads(str(c["race_log"]))
            for k in SLOG_KEYS:  # a checkpoint from before a monitor existed: NaN for its bins
                self.slog.setdefault(k, [float("nan")] * len(self.slog["episode"]))
        return float(c["wall"]), int(c["resumes"]) + 1


def calibrate_ck(cfg):
    """E4' rule (the E4 rule on the authors' loop): eta per population (critic, actor) = Adam step per ms / RMS of the
    per-ms update g (mean over all copies, the authors' units), measured per ms (block 1) with the initial W frozen
    over CAL_TICKS ticks after the first WARM_N + 1 (no env step yet) of a run on the (seed, CAL_SEED) stream: an
    RMS-sized per-ms update moves W by the step per-ms Adam takes at init. Without recurrence (rec 'none'), so every rec
    arm gets the base run's eta. Returns (eta, rms), eta in mV."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=1, rec="none"), rng_seed=CAL_SEED, frozen=True)
    for _ in range(WARM_N + 1):
        r.tick()
    r.gsq[:], r.nupd = 0, 0
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    step = np.array([cfg.adam_step_c or cfg.eta_c, cfg.adam_step_a or cfg.eta_a])  # Adam steps (016 §20: CartPole's
    return step / rms, rms  # preset eta are SGD rates, 68-280x too large as a per-ms step; its adam_step is LL's)


def probe_rate(cfg):
    """Mean post spike probability per ms (all posts and copies) over CAL_TICKS ticks after the first WARM_N + 1, W
    frozen at its init, on the (seed, CAL_SEED) stream; rule off (CKRunner.probe). Without recurrence (rec 'none'), so
    every rec arm gets the base run's s_in; with rate_match_rec (engine nlif) with the arm's R at init (its kicks,
    sealed or not), so every arm starts at the same post rate (iteration 016 §10). The BindsNET reference probes of
    the calibrations (engine lif, rate_target 0) run without norm_logits: they measure CK's backbone."""
    rec = cfg.rec if cfg.rate_match_rec and cfg.engine == "nlif" else "none"
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=MS, rec=rec), rng_seed=CAL_SEED, frozen=True)
    r.probe = True
    for _ in range(WARM_N + 1):
        r.tick()
    r.spk[:] = 0
    for _ in range(CAL_TICKS):
        r.tick()
    return r.spk[0] / r.spk[1]


def probe_entropy(cfg):
    """Mean policy entropy H(p) (nats, over live copies) over CAL_TICKS ticks after the first WARM_N + 1, W and the
    arm's R frozen at init, on the (seed, CAL_SEED) stream; rule off (CKRunner.probe)."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=MS), rng_seed=CAL_SEED, frozen=True)
    r.probe = True
    for _ in range(WARM_N + 1):
        r.tick()
    h, n = 0.0, 0  # pooled over the live copies of every tick (a tick with none, e.g. CartPole's one copy in reset,
    for _ in range(CAL_TICKS):  # contributes nothing; a per-tick mean would be NaN there and derail match_entropy)
        r.tick()
        p = r.p[r.code < 2]
        h += float(-(p * np.log(np.maximum(p, 1e-300))).sum())
        n += len(p)
    return h / n


def match_entropy(cfg, rtol=2e-3):
    """Pre-set actor gain of a rec arm (iteration 016 §11; not tuned on return): the factor f on alpha_a such that
    probe_entropy with the arm's R equals the base run's (rec 'none', same s_in and alpha_a), by bisection on log f in
    [1/8, 8]. The within-column loop amplifies rate differences between columns, so at a matched mean rate the arm's
    policy is sharper; f < 1 undoes that gain. Returns (f, target entropy, entropy at f)."""
    target = probe_entropy(dataclasses.replace(cfg, rec="none"))
    at = lambda f: probe_entropy(dataclasses.replace(cfg, alpha_a=f * cfg.alpha_a))  # noqa: E731
    lo, hi = -3.0, 3.0  # log2 f; H falls with f
    while hi - lo > rtol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at(2 ** mid) > target else (lo, mid)
    f = 2 ** ((lo + hi) / 2)
    return f, target, at(f)


def calibrate_s_in(cfg, rtol=2e-3):
    """Pre-set input scale of engine nlif (not tuned on return): s_in such that probe_rate equals engine lif's
    (BindsNET, deterministic) on the same stream and initial W, or with rate_target that rate (Hz), by bisection (the
    noise stream is common to every probe; the closed loop makes the rate only approximately monotone in s_in).
    Returns (s_in, target, rate, BindsNET's rate), rates per ms."""
    target = r_ck = probe_rate(dataclasses.replace(cfg, engine="lif", s_in=0.0, rate_target=0.0, norm_logits=False,
                                                   readout="softmax", credit="surrogate"))
    if cfg.rate_target:  # low-rate regime: the target is rate_target, not BindsNET's rate
        target = cfg.rate_target * 1e-3
    lo, hi = 0.0, 1.0
    while probe_rate(dataclasses.replace(cfg, s_in=hi)) < target:
        lo, hi = hi, 2 * hi
        assert hi < 1e3, "nlif cannot reach the target rate"
    while hi - lo > rtol * hi:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if probe_rate(dataclasses.replace(cfg, s_in=mid)) < target else (lo, mid)
    return hi, float(target), float(probe_rate(dataclasses.replace(cfg, s_in=hi))), float(r_ck)


def calibrate_low(cfg):
    """Low-rate nlif regime (--rate-target R Hz; iteration 015 step 3), pre-set, not tuned on return, all without
    recurrence (so every rec arm of a seed shares it): tau_ref 2 ms (lif.DT1R); s_in by calibrate_s_in so that the mean
    initial post rate is R (unless --s-in is given); k = r_CK / R with r_CK BindsNET's initial rate on the same probe;
    alpha_a and alpha_c times k (policy logits and V keep CK's range; beta_c kept), and with ent_match_rec alpha_a
    times match_entropy's f (the arm's initial policy entropy is the base run's); without --eta-cal or Adam, eta_c
    and eta_a times RMS(g_CK) / RMS(g_low), the per-ms update RMS at init with W frozen on the CAL stream
    (calibrate_ck's measurement) of BindsNET's engine at CK's alphas and of the low-rate engine at the scaled ones
    (with --eta-cal, calibrate_ck normalises eta by RMS(g) itself). With readout race and race_beta 0, calibrate_race
    sets beta after the alphas and before the eta matching (whose actor update needs beta); its record is the returned
    dict's 'race' and race_beta joins 'cfg', so a checkpoint's 'lowrate' restores it. Returns (cfg with s_in, alphas,
    etas and race_beta replaced, record)."""
    assert not (cfg.readout == "race" and cfg.ent_match_rec), "ent_match_rec: softmax only"
    if cfg.s_in:
        s_in, target, r_ck = cfg.s_in, cfg.rate_target * 1e-3, probe_rate(dataclasses.replace(
            cfg, engine="lif", s_in=0.0, rate_target=0.0, norm_logits=False, readout="softmax", credit="surrogate"))
        got = probe_rate(cfg)
    else:
        s_in, target, got, r_ck = calibrate_s_in(cfg)
    if cfg.r_ck_hz:  # iteration 016: CartPole's BindsNET rate, not this task's
        r_ck = cfg.r_ck_hz * 1e-3
    k = r_ck / target
    low = dataclasses.replace(cfg, s_in=s_in, alpha_a=k * cfg.alpha_a, alpha_c=k * cfg.alpha_c)
    em = dict(ent_scale=1.0)
    if cfg.ent_match_rec and cfg.rec != "none":
        f, h_base, h_got = match_entropy(low)
        low = dataclasses.replace(low, alpha_a=f * low.alpha_a)
        em = dict(ent_scale=float(f), h_base=h_base, h_got=h_got)
    fields = ["s_in", "alpha_a", "alpha_c", "eta_c", "eta_a"]
    if cfg.readout == "race" and not cfg.race_beta:
        beta, rec = calibrate_race(low)
        low = dataclasses.replace(low, race_beta=beta)
        em["race"] = dict(theta=race_theta(low), **rec)
        fields.append("race_beta")
    es = np.ones(2)
    if not (cfg.sgd or cfg.adam):
        rms_ck = calibrate_ck(dataclasses.replace(cfg, engine="lif", rate_target=0.0, s_in=0.0, norm_logits=False,
                                                  readout="softmax", credit="surrogate"))[1]
        rms_low = calibrate_ck(low)[1]
        es = rms_ck / rms_low
        low = dataclasses.replace(low, eta_c=float(cfg.eta_c * es[0]), eta_a=float(cfg.eta_a * es[1]))
    rep = {f: getattr(low, f) for f in fields}
    return low, dict(k=float(k), r_ck_hz=1e3 * float(r_ck), r_target_hz=cfg.rate_target, r_got_hz=1e3 * float(got),
                     s_in=float(s_in), alpha_scale=float(k), eta_scale=[float(x) for x in es], cfg=rep, **em)


def entropy_mm(counts):
    """Entropy (nats) of a histogram with the Miller-Madow bias correction (m - 1) / (2 n), m the occupied bins: the
    plug-in estimate of a K-outcome distribution from n draws is low by about (K - 1) / (2 n) [Miller 1955]."""
    n = counts.sum()
    q = counts / n
    return float(-(q * np.log(np.maximum(q, 1e-300))).sum() + (np.count_nonzero(counts) - 1) / (2 * n))


RACE_DELTAS = (-0.2, -0.1, 0.0, 0.1, 0.2)  # calibrate_race: contrast levels on column 0's E rows (a convention)


def calibrate_race(cfg, n_states=8, rtol=2e-3):
    """Pre-set sharpness beta of the race's surrogate policy (016 §14.1 Stage B; fixed before any return is seen): the
    race's sharpness as a function of rate contrast, the quantity the surrogate p~ = softmax(beta log r) has to
    reproduce, measured by a contrast probe. A held-observation probe (CKRunner.hold: the env is not stepped and no
    copy resets, so every decision window is an independent draw of the spike noise at a fixed observation; W and R
    frozen at init, rule off, the (seed, CAL_SEED) stream): for n_states fresh observations (env copy 0's reset,
    broadcast to every copy) and each contrast delta in RACE_DELTAS, column 0's excitatory input rows of W are scaled
    by 1 + delta (WARM_N + 1 warm-up ticks, then CAL_TICKS / n_states ticks), and the closed loop's own race decisions
    (once per freeze window) give the winners and the mean window-end column E rates rbar(state, delta) (CKRunner.prk,
    the quantity the surrogate reads). Conventions: the perturbed column (0), its E rows (so the contrast is in the
    E rate the surrogate reads and the race tallies) and the delta set.
    beta is the maximum-likelihood fit of the winners under softmax(beta log max(rbar(state, delta), eps_r)) over all
    states and conditions (one scalar; the log-likelihood is concave in beta, so log2 beta is bisected on the sign of
    its slope over [-3, 6]). Biologically the probe is a psychometric curve of the action channel: the win probability
    against the log-rate contrast, whose slope is the decision's sharpness (a drift-diffusion gain set by the bound).
    Returns (beta, record): beta, loglik (at beta, nats), deltas, win0 (column 0's realised win rate per delta),
    win0_surrogate (the surrogate's at beta), n_decisions, and the diagnostics H_race (Miller-Madow entropy of the
    delta = 0 winners, mean over states), H_surrogate (at beta on the delta = 0 rates), beta_poisson = 1.6 sqrt(theta
    / 2) (the slope at zero contrast of a race between Poisson counts to theta, logit for probit)."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=MS, race_beta=0.0), rng_seed=CAL_SEED,
                 frozen=True)
    return _race_probe(r, n_states, rtol)


def reprobe_race(runner, episode, n_states=8, rtol=2e-3):
    """The contrast probe of calibrate_race re-run at the runner's current state (016 §18 fix a, --race-reprobe): a
    frozen probe runner from the runner's cfg (its calibrated s_in and alphas) takes the runner's current W and, with
    recurrence, its R (and mask), on its own stream (cfg.seed, CAL_SEED, episode), so the runner's rng and nrng draw
    nothing and the probe is reproducible at a given state and episode. The runner is not changed: the caller sets
    runner.beta from the record. Returns calibrate_race's record plus theta and episode (the race_cal shape)."""
    cfg = runner.cfg
    p = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=MS, race_beta=0.0),
                 rng_seed=(CAL_SEED, int(episode)), frozen=True)
    p.W = runner.W.copy()
    if runner.hr is not None:
        p.hr.R, p.hr.mask = runner.hr.R.copy(), runner.hr.mask.copy()
    beta, rec = _race_probe(p, n_states, rtol)
    return dict(theta=runner.theta, episode=int(episode), **rec)


RACE_FIT_GATE = 0.05  # reprobe_step: a re-probed beta is accepted only if its record's fit is within it (§18, §21)


def race_fit(rec):
    """The surrogate's fit in a probe record: max |win0 - win0_surrogate| over the contrasts (the gate, RACE_FIT_GATE,
    is 0.05)."""
    return float(np.abs(np.array(rec["win0"]) - np.array(rec["win0_surrogate"])).max())


def reprobe_step(runner, episode):
    """run_ck's re-probe hook (--race-reprobe): reprobe_race at the runner's current state, the new beta accepted only
    if the record's fit is within RACE_FIT_GATE (016 §21: a poor fit means the surrogate does not describe the race,
    and the bisection's beta is then arbitrary); a rejected record keeps the current beta and is marked
    'accepted': False (absent on an accepted record, so runs whose re-probes all pass are unchanged). The record goes
    to race_cal_history either way, and the line printed. Returns the record. The init probe (calibrate_race) is not
    gated: its fit is reported."""
    rec = reprobe_race(runner, episode)
    fit = race_fit(rec)
    if fit <= runner.cfg.race_fit_gate:
        runner.beta = rec["beta"]
    else:
        rec["accepted"] = False
    runner.race_cal_history.append(rec)
    print(f"reprobe ep {episode} beta {rec['beta']:.4g} fit {fit:.3f}" + ("" if fit <= RACE_FIT_GATE else " rejected"),
          flush=True)
    return rec


def _race_probe(r, n_states, rtol):
    """The contrast probe on a frozen probe runner r (calibrate_race's body, shared with reprobe_race): r.W is the
    probed W; returns (beta, record)."""
    cfg = r.cfg
    r.probe = r.hold = True
    W0, nc, n_ae, K, nd = r.W.copy(), r.lay.n_crit, cfg.n_ae, r.K, len(RACE_DELTAS)
    per, win, lr = CAL_TICKS // n_states, np.zeros((n_states, nd, K)), np.zeros((n_states, nd, K))
    for s in range(n_states):
        r.hold_reset()
        for i, d in enumerate(RACE_DELTAS):
            r.W = W0.copy()
            r.W[nc:nc + n_ae] *= 1 + d
            r.hold_clear()
            for _ in range(WARM_N + 1):
                r.tick()
            r.hold_clear()
            for _ in range(per):
                r.tick()
            assert r.rwin.sum() > 0, "no race decision in the condition's block"
            win[s, i], lr[s, i] = r.rwin, np.log(np.maximum(r.rsum / r.rn, r.eps_r))
    r.W = W0

    def slope(b):  # d log-likelihood / d beta, falls with beta
        return float((win * (lr - (td.softmax(b * lr) * lr).sum(-1, keepdims=True))).sum())
    lo, hi = -3.0, 6.0  # log2 beta
    while hi - lo > rtol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if slope(2 ** mid) > 0 else (lo, mid)
    beta = float(2 ** ((lo + hi) / 2))
    p = td.softmax(beta * lr)
    i0 = RACE_DELTAS.index(0.0)
    p0 = td.softmax(beta * lr[:, i0])
    rec = dict(beta=beta, loglik=float((win * np.log(np.maximum(p, 1e-300))).sum()), deltas=list(RACE_DELTAS),
               win0=(win[..., 0].sum(0) / win.sum((0, 2))).tolist(),
               win0_surrogate=((p[..., 0] * win.sum(-1)).sum(0) / win.sum((0, 2))).tolist(),
               n_decisions=int(win.sum()),
               H_race=float(np.mean([entropy_mm(w) for w in win[:, i0]])),
               H_surrogate=float(-(p0 * np.log(np.maximum(p0, 1e-300))).sum(-1).mean()),
               beta_poisson=float(1.6 * np.sqrt(r.theta / 2)))
    return beta, rec


def calibrate_rec(cfg):
    """Pre-set rates of rec (iteration 015; not tuned on return), on a run with W and R frozen at init on the
    (seed, CAL_SEED) stream, over CAL_TICKS windows after the first WARM_N + 1: eta_rec gives R the actor W's relative
    step per window, eta_a RMS(g_actor) / RMS(W_actor) = eta_rec RMS(u) / RMS(R) (u = mean_h (sum_t delta) e_h, the
    task direction of a window; RMS over existing synapses); lam_rec = lam_ratio RMS(eta_rec x u summed over a frame) /
    RMS(structural increment of a frame) (lam = 1). Measured on the unsealed default draws (rec_seal off, p_cross
    0.05), so every arm of a seed shares the rates: with sealed R every target of j is in m(j), the one-hop credit
    h_m(i) - hbar_j is exactly 0, and RMS(u) is rounding noise. Returns (eta_rec, lam_rec, dict of the RMS values)."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, eta_rec=1.0, lam_rec=0.0, rec_seal=False,
                                     p_cross=Config().p_cross),
                 rng_seed=CAL_SEED, frozen=True)
    for _ in range(WARM_N + 1):
        r.tick()
    r.rsq[:], r.rcnt[:], r.ftask, r.hr.acc = 0, 0, 0.0, r.hr._fresh()
    for _ in range(CAL_TICKS):
        r.tick()
    hr, q = r.hr, np.sqrt(r.rsq / r.rcnt[[0, 0, 0, 1, 2]])
    rel_w = cfg.eta_a / r.mv * q[0] / q[1]
    eta = cfg.eta_rec or rel_w * float(np.sqrt((hr.R[hr.mask] ** 2).mean())) / q[2]
    lam = cfg.lam_rec or cfg.lam_ratio * eta * q[3] / q[4]
    return float(eta), float(lam), dict(rel_step_W=float(rel_w), rms_g_actor=q[0], rms_W_actor=q[1], rms_u=q[2],
                                        rms_u_frame=q[3], rms_dS=q[4])


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
    cfg = Config(**json.loads(str(z["config"])))
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


def v4_returns(task):
    """Last-100 returns over seeds of the model v4 runs (flow; spiking per heads and variant), from models.basic."""
    pat = re.compile(rf"m3_{task}_(flow|spiking_h\d+)_lb0\.3_tn0\.04_pcc0_bk0\.01m16(_\w+?)?_eta5_s\d+")
    out = {}
    for f in sorted(glob.glob(str(OUT / f"m3_{task}_*_eta5_s*.json"))):
        m = pat.fullmatch(pathlib.Path(f).stem)
        if m and (m[1] != "flow" or not m[2]):
            k = "v4 " + m[1].replace("_", " ") + (" " + m[2][1:] if m[2] else "")
            out.setdefault(k, []).append(np.mean(json.loads(pathlib.Path(f).read_text())["returns"][-100:]))
    return out


def trailing_mean_time(ret, thr):
    """First episode e >= 100 whose trailing-100 mean return reaches thr (1-indexed), or None."""
    c = np.cumsum(np.r_[0.0, ret])
    hit = np.flatnonzero((c[100:] - c[:-100]) / 100 >= thr)
    return int(hit[0]) + 100 if len(hit) else None


def group_stats(task, rs, ref=None):
    """Summary statistics of one group of runs (result dicts) on `task`, shared by --summary and experiments.report:
    T_f / T_s (CK definitions) mean, sd (ddof 0) and count over runs that reach them; median T_s with unsolved =
    infinite; T_m = first episode whose trailing-100 mean reaches M_THR; last-100 return mean, sd and the count >=
    M_THR; mean ms per frame. With ref (another group's runs): two-sided Mann-Whitney p on T_f, T_s (never reached =
    episode cap + 1) and the last-100 return."""
    from scipy.stats import mannwhitneyu
    Tf = [r["T_f"] for r in rs if r["T_f"] is not None]
    Ts = [r["T_s"] for r in rs if r["T_s"] is not None]
    Tm = [t for t in (trailing_mean_time(r["returns"], M_THR[task]) for r in rs) if t is not None]
    R = [float(np.mean(r["returns"][-100:])) for r in rs]
    ms = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (float("nan"),) * 2  # noqa: E731
    out = {"n": len(rs), "solved": len(Ts), "T_f_mean": ms(Tf)[0], "T_f_sd": ms(Tf)[1], "T_f_n": len(Tf),
           "T_s_mean": ms(Ts)[0], "T_s_sd": ms(Ts)[1],
           "T_s_median": float(np.median([r["T_s"] if r["T_s"] is not None else np.inf for r in rs])),
           "T_m_mean": ms(Tm)[0], "T_m_sd": ms(Tm)[1], "T_m_n": len(Tm), "ret_mean": float(np.mean(R)),
           "ret_sd": float(np.std(R)), "ret_ge_M": sum(x >= M_THR[task] for x in R),
           "ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rs]))}
    if ref is not None:
        cap = lambda r: (r["config"].get("max_eps") or r["config"]["episodes"]) + 1  # noqa: E731
        cens = lambda rr, k: [r[k] or cap(r) for r in rr]  # noqa: E731
        R0 = [np.mean(r["returns"][-100:]) for r in ref]
        out |= {"mw_T_f": float(mannwhitneyu(cens(rs, "T_f"), cens(ref, "T_f")).pvalue),
                "mw_T_s": float(mannwhitneyu(cens(rs, "T_s"), cens(ref, "T_s")).pvalue),
                "mw_ret": float(mannwhitneyu(R, R0).pvalue)}
    return out


def summary():
    print("T_f / T_s (CK definitions, pre-registered): mean +- sd over runs that reach it [n]; median T_s counts "
          "unsolved as infinite. Authors'-code groups (ck) count episodes pooled over copies (main.py); legacy "
          "LunarLander groups count per copy (copy-mean returns). Secondary: T_m = first episode whose trailing-100 "
          "mean reaches M_THR (CartPole 475, Gymnasium's threshold; LunarLander 200 = T_s). ret: last-100 return, "
          "mean +- sd over seeds; >=M: runs with ret >= M_THR. MW: Mann-Whitney p against the family's base group "
          "(the task preset), on T_f / T_s (never reached = episode cap + 1) / ret.")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "tdstdp_*.json"))):
        groups.setdefault(pathlib.Path(f).stem.rsplit("_s", 1)[0], []).append(json.loads(pathlib.Path(f).read_text()))
    ms = lambda m, s: f"{m:6.1f} +- {s:5.1f}" if m == m else "   nan          "  # noqa: E731
    for task in PRESETS:
        if not any(g.split("_")[1] == task for g in groups):
            continue
        p = PAPER.get(task)
        print(f"\n{task}: " + (f"CK 2020 T_f {p['T_f'][0]:.1f} +- {p['T_f'][1]:.1f}  "
                                f"T_s {p['T_s'][0]:.1f} +- {p['T_s'][1]:.1f}" if p else "(no CK 2020 result)"))
        for k, R in v4_returns(task).items():
            print(f"  {k:28} n={len(R):2d} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}")
        for ck, base in ((False, f"tdstdp_{task}"), (True, f"tdstdp_{task}_ck")):
            gs = {g: rs for g, rs in groups.items() if g.split("_")[1] == task and (g.split("_")[2:3] == ["ck"]) == ck}
            if not gs:
                continue
            print("  authors' code (default):" if ck else "  pre-audit code (--legacy):")
            for g, rs in sorted(gs.items(), key=lambda kv: (kv[0] != base, kv[0])):
                st = group_stats(task, rs, gs[base] if g != base and base in gs else None)
                line = (f"  {g:28} n={st['n']:2d} solved {st['solved']:2d}/{st['n']:2d}  T_f "
                        f"{ms(st['T_f_mean'], st['T_f_sd'])} [{st['T_f_n']:2d}]  T_s {ms(st['T_s_mean'], st['T_s_sd'])}"
                        f" median {st['T_s_median']:5.0f}  T_m {ms(st['T_m_mean'], st['T_m_sd'])} [{st['T_m_n']:2d}]  "
                        f"ret {st['ret_mean']:6.1f} +- {st['ret_sd']:5.1f} >=M {st['ret_ge_M']:2d}  "
                        f"ms/fr {st['ms_per_frame']:5.2f}")
                if "mw_T_f" in st:
                    line += f"  MW {st['mw_T_f']:.2g} / {st['mw_T_s']:.2g} / {st['mw_ret']:.2g}"
                print(line)


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
