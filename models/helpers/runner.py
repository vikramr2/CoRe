"""The TD-STDP backbone's parameters, Config, presets and runners (Runner: the branching / flow / BindsNET-LIF
loop; CKRunner: the authors' loop with the race readout and recurrent columns). models/tdstdp.py's
docstring describes every option."""
import copy
import dataclasses
import itertools
import json
import pathlib

import gymnasium as gym
import numpy as np

from mapstdp import lif
from mapstdp import tdstdp as td
from models.helpers.tasks import FOURIER_BOUNDS, TASKS, Task, critic_layer, random_states

OUT = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "results"  # the intermediates store
TDSTDP_SEED, CAL_SEED, CAL_TICKS, FBAR_STATES, MV_LEGACY, MS, WARM_TICKS = 1302, 1303, 200, 5000, 1300.0, 20, 5
MV, WARM_N = td.THETA_L, 5  # authors' code: mV per unit of branching weight; batch_envs warm_n (windows)
SOLVE = {"CartPole": (500.0, "all"), "LunarLander": (200.0, "mean"),  # perfect score; solved = last 100 all / mean
         "Acrobot": (-100.0, "mean"), "Taxi": (8.0, "mean")}  # iteration 016: Gymnasium's reward_threshold
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


def trailing_mean_time(ret, thr):
    """First episode e >= 100 whose trailing-100 mean return reaches thr (1-indexed), or None."""
    c = np.cumsum(np.r_[0.0, ret])
    hit = np.flatnonzero((c[100:] - c[:-100]) / 100 >= thr)
    return int(hit[0]) + 100 if len(hit) else None
