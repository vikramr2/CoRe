"""The basic Map-STDP model: models v3 and v4 and the emergent rule (docs/derivation.md), the model of study
emergent-v4. This file holds the training loop (run) and the command line; the objects and methods it uses are in
models/helpers/ (tasks: the task registry; agent: parameters, Config, Agent; basic_io, basic_stats, basic_cli) and are
re-exported here (__all__), so callers use models.basic. This docstring documents all of them.

The runner (M3, docs/SPEC.md §11): model v3 with the opened latents (latent-output barrier Eq. 4e, beta_LA = 0.3) on
CartPole, MountainCar, Acrobot and LunarLander, flow against spiking with H parallel heads (Proposal). Run from the
repo root:
  python -m models.basic --task {CartPole,MountainCar,Acrobot,LunarLander} --backend {flow,spiking,mpr} [--heads H]
    --seed 0 [--episodes E] [--td-norm s] [variant options]
    -> experiments/results/m3_<task>_<backend>[_h<H>]_lb<beta_LA>[_tn<s>][_<variant>]_s<seed>.json
  Every option maps onto a Config field (The agent, below). Variant tokens, in file-name order:
    ablations (iteration 005) wmin<w>, tau_e<T>, at<T> (--wmin, --tau-e, --eta-tau);
    lam<l>, pcc<P> (--lam, --p-cc); b<B> (--beta, Eq. 1e); bk<k>m<B> (--beta-kappa, --beta-max: adaptive beta,
    Eq. 1g); g<g> (--gamma, iteration 009); dbc (--readout diffbound, Eq. 1h, spiking, iteration 010);
    enc<SPEC>, crit<SPEC>, ce<c>[H<f>] (--encoder, --critic-layer, --c-ent [--ent-target]: iteration 012, flow);
    eta<E> (--eta); qif[-<field><value>...] (--engine qif and its QIFParams overrides, iteration 013, e.g. qif-tau_s0
    -inhma-warm0-lags3); lif[-<field><value>...] (--engine lif, rule emergent only, and its LIFParams overrides
    --lif-<field>); emergent[-ot<T>][-ebmod][-stk] (--rule emergent, Eqs. 8-12, and its options --obs-tau,
    --ebar module, --step-stat kappa); ls<x>, es<x> (--lam-scale, --eta-scale), ms (--match-scales); ep<E> (--episodes
    off the task's default).
  Run plumbing: --out PATH (result file; its .ckpt.npz checkpoint alongside, resumed if present), --save-model PATH
    (compressed float32 .npz of W at episode 0, every 250 episodes and the end, with mask, m, roles, config, u),
    --match-scales (match_scales: lam and eta scales from the RMS of the structural and task increments at the
    initial W against flow's; printed, applied and recorded in the result).
  Model v4 is the flag set --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 (docs/model.md).
  The beta ramp, the beta mean reversion / trace signal / step cap, the score-unit ablation, the TD(lambda) critic,
  the rich and net critics and the usage term (former Eq. 4i) were removed in iteration 013 (not adopted); their result
  files still load in --summary and --ablations.
  python -m models.basic --summary    -> per task: random, flow and spiking per H (last-100 return over seeds)
  python -m models.basic --ablations  -> flow learning-rule ablations against the --td-norm 0.04 baseline
  python -m models.basic --fidelity [--engine qif ...]  -> per task and H: error of the pooled pi-hat and kappa
    against the flow (qif: also the per-postsynaptic-row 1 - corr and the bias 1 - corr(E[kappa], J))
Heads: H independent spiking replicas of the network on the same W (tasks share one block-diagonal SuperNeuroMAT
network). Their spikes are pooled: pi-hat from summed counts, kappa the sum of per-head covariance counts divided by
the total spikes, one race over all heads (still Eq. 1d). H = 1 is the M2 spiking agent.
  python -m models.basic --m2-summary -> the M2 table (CartPole, model v3) over seeds of the m2_*.json files

The task registry (docs/SPEC.md §4, docs/model.md §1). per Gymnasium task, the module sizes, the
stimulus encoder v(o) over the controller and the reward r_f fed to the TD error (Eq. 4c).
CartPole keeps its M2 encoder (6 x 6 grid over theta, theta_dot) and reward (-1 on termination), bit for bit.
The M3 encoders (Proposal) are conjunctive Gaussian RFs over the whole normalised observation:
  grid:  a 6 x 6 grid over a 2-D observation normalised by its bounds (MountainCar);
  pool:  RF centers drawn from states a random policy visits, observation z-scored by that pool, sigma the median
         nearest-center distance (Acrobot, LunarLander: the 'random sparse conjunctive fields' of model.md §1).
Rewards are the environment's, scaled so that a failing episode's TD errors are O(1) like CartPole's.

The agent (M2, docs/SPEC.md §11): CartPole in closed loop with Map-STDP, model v3 (docs/model.md), and its extensions.
Model v3: d_j homeostasis as its own term (Eq. 4d, EPS_H, every frame), race burn-in BURN steps, task rate eta
annealed as eta / (1 + episode / ETA_TAU). Models v1 and v2 and the dual routing term (Eq. 4a) were removed in
iteration 013; their committed results still load in --summary (git history has the code).
Config (one per run) holds every option; its defaults reproduce v3 on CartPole:
  --eta-tau T: task rate annealed as eta / (1 + episode / T) (ablation; 0 = constant).
  --beta B: sharpened policy P ~ p_a^beta with the matched score (Eq. 1e, Proposal); flow, or spiking with diffbound.
  --la-beta B: latent-output barrier (Eq. 4e, Proposal) subtracted from g on latent->action pairs; 0 is off.
  p_cc: controller->controller connection probability of the SBM (core.build_network; 0.1 = M2). p_cc = 0 removes
    the C->C recurrence, whose neuron-level credit the module-level Eq. 4b cannot represent (iteration 006).
  p_cross: cross-module connection probability of the SBM outside the routing pairs (0.05 = M2). p_cross = 0 builds
    the modules with full modularity, the structure the map term converges to (iteration 014): the control for "the
    map term helps" versus "full modularity helps". The SBM draws are the same, so only the mask changes.
  beta_kappa k, beta_max B: causal adaptive gain (Eq. 1g, Proposal), one global scalar: after each episode
    log beta += k z, z = (R_fast - R_slow) / sd(R) clipped to [-2, 2] (fast / slow running returns over BETA_TAU_F /
    BETA_TAU_S episodes), beta kept in [beta, B]. It rises while return improves and falls when it drops.
  gamma (iteration 009): discount of the critic and of delta (Eq. 4c); default GAMMA = 0.99.
  readout='diffbound' (Proposal, iteration 010; spiking only): the action is the count-difference race to a bound
    (Eq. 1h, sp.diffbound, at most one spike per action module and head per step) instead of the race. beta (fixed,
    or adaptive with beta_kappa) is the target sharpness: theta = BOUND_CAL^-1(beta), log-log interpolation of the
    measured beta_eff(theta) table for the agent's heads, clipped to the table, randomised between floor and ceil for
    a fractional theta. The Eq. 4b score uses beta (matched where the table reaches beta; below its first entry the
    readout is sharper than the score). Draws one extra uniform per frame, only on this path.
  Iteration 012 (Proposals, flow; all off by default, none draws from rng):
  encoder SPEC: the controller encoder replaced by with_encoder(task, SPEC) (grid g<n>s<sigma> or Fourier
    f<order>c<conj>[h][b]); the controller size becomes the feature count.
  critic_layer SPEC: separate critic, V = u . phi(o) on its own fixed RF layer (critic_layer), NLMS (Eq. 4c'');
    the actor's flow is not used by the critic.
  c_ent c[, ent_target f]: entropy term (Eq. 4h), a separate per-frame ungated update W += eta_e c kappa
    (g_b / J^in_b - gbar_j) (core.entropy_score; average W * grad H(P~), P~ the beta = 1 action shares, pi fixed);
    eta_e the annealed task rate. ent_target adapts c SAC-style, log c += ENT_K (f log|A| - H(P~)) per frame (nats).
  backend='mpr' (Proposal, iteration 013, Phase B): no spikes; pi is the normalised neuron-level MPR fixed point of the
    QIF network (qif.rates, at qif.MPR_POINT with Config.qif overrides, warm-started from the previous frame), kappa the
    expected pairings T * pi at those rates, the action from Eq. 1d / 1e as in flow; evaluate() uses the same pi.
  engine='qif' (Proposal, iteration 013; spiking): mapstdp.qif, QIF neurons with fixed Lorentzian excitabilities drawn
    at init from their own rng stream (seed, QIF_SEED), Config.qif overriding qif.QIFParams fields. The state is carried
    across frames (warm start, reset at episode start; QIFParams.warm = False: cold start every frame), the race
    burn-in is QIFParams.burn bins and kappa uses the QIF pairing window (qif.pairings). The teleportation pulses draw
    from self.rng as the other engines do; evaluate() races from a reset state per held-out stimulus, after one
    warm-up frame.
Both backends share one frame update; they differ only in where pi, kappa and the action come from:
  flow:    pi = n-hop flow, kappa = J (expected pairings), action ~ Eq. 1d;
  spiking: pi = spike shares, kappa = covariance count / spikes in the frame, action = race readout.
Dividing kappa by the frame's spike count is the 1/(spikes per frame) scaling of lambda and eta (model.md §4).
rule='emergent' (derivation Eqs. 8-12): the walk is observed from spikes (pair counts kappa_raw, Eq. 8) and sets the
  baselines and M*; a signed counter c (Eq. 11a for branching, engine='numpy') sets each synapse's step: structural
  -lam (g - gbar^obs) c (Eq. 4), eligibility c (h - hbar^obs) (Eq. 4b), homeostasis -eps_h (d_j - 1) W pi_j / d_j
  (Eq. 4d). Flow (kappa_raw = c = J) runs the v4 code, bit for bit. Options obs_tau, ebar, step_stat; per-episode
  monitors in the result's 'emergent'; evaluate() adds L_obs and block_shares. Engines plug in at
  Agent.counter_frame (its docstring is the contract); qif: Eq. 11b with a regression gain (Agent.qif_counter_frame);
  lif: mapstdp.lif (Agent.lif_counter_frame; Config.lif overriding LIFParams; rule 'emergent' only, race or
  diffbound with BOUND_CAL_LIF, burn-in LIF_BURN steps).
run(): checkpoints every LOG_EVERY episodes (<out>.ckpt.npz, resumed, counted in 'resumes'), optional --save-model;
match_scales(): the pre-set step-size match of lambda and eta to flow (Config.lam_scale, eta_scale)."""
import argparse
import dataclasses
import json
import pathlib
import time

import gymnasium as gym
import numpy as np

from mapstdp import core, lif, qif
from models.helpers.agent import (
    ALPHA,
    BOUND_CAL,
    ETA,
    ETA_V,
    FROZEN_EPIS,
    GAMMA,
    K_SPK,
    LOG_EVERY,
    MODEL,
    N_HOPS,
    OUT,
    RATE,
    REC_EPIS,
    SAVE_EVERY,
    TAU_E,
    THETA,
    Agent,
    Config,
    block_shares,
    match_scales,
)
from models.helpers.basic_cli import (
    HEADS,
    LA_BETA,
    LIF_FLAGS,
    QIF_FLAGS,
    lif_overrides,
    name,
    qif_overrides,
    qif_token,
    variant_tokens,
)
from models.helpers.basic_io import default_name, frozen_returns, load_checkpoint, save_checkpoint, save_model
from models.helpers.basic_stats import ablation_summary, diag012, fidelity, random_return, summary, summary_m2
from models.helpers.tasks import TASKS

__all__ = [  # the model's API (objects and methods from helpers/, re-exported)
    "ALPHA", "Agent", "BOUND_CAL", "Config", "ETA", "ETA_V", "FROZEN_EPIS", "GAMMA", "HEADS", "K_SPK", "LA_BETA",
    "LIF_FLAGS", "LOG_EVERY", "MODEL", "N_HOPS", "OUT", "QIF_FLAGS", "RATE", "REC_EPIS", "SAVE_EVERY", "TASKS",
    "TAU_E", "THETA", "ablation_summary", "argparse", "block_shares", "core", "dataclasses", "default_name",
    "diag012", "fidelity", "frozen_returns", "gym", "json", "lif", "lif_overrides", "load_checkpoint", "main",
    "match_scales", "name", "np", "pathlib", "qif", "qif_overrides", "qif_token", "random_return", "run",
    "save_checkpoint", "save_model", "summary", "summary_m2", "time", "variant_tokens",
]


def run(cfg, episodes, name=None, out=None, save_model_path=None, scales=None):
    """Train for `episodes`, evaluating every LOG_EVERY (evaluate() plus diagnostics()), and write the result file
    (`out`, a .json path, or OUT/<name>.json, default name m2_*.json). Iteration-012 runs add end-of-run diagnostics
    (diag012) over the last REC_EPIS episodes. Checkpoints: every LOG_EVERY episodes <out>.ckpt.npz (save_checkpoint);
    a run that finds one resumes from it (not bit-identical in general; counted in the result's `resumes`) and deletes
    it on completion. save_model_path: save_model at the end. scales: match_scales' record, stored in the result."""
    agent = Agent(cfg)
    new012 = bool(cfg.encoder or cfg.critic_layer or cfg.c_ent or cfg.ent_target)
    env = gym.make(agent.task.env_id)
    fn = pathlib.Path(out) if out else OUT / f"{name or default_name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    ck = fn.with_suffix(".ckpt.npz")
    if ck.exists():
        returns, evals, snaps, resumes, wall = load_checkpoint(ck, agent)
        resumes += 1
        print(f"resumed from {ck.name} at episode {len(returns)}", flush=True)
    else:
        returns, evals, resumes, wall = [], [{"episode": 0, **agent.evaluate(), **agent.diagnostics()}], 0, 0.0
        snaps = {"W_ep0": agent.W.astype(np.float32)} if save_model_path else {}
    t0 = time.perf_counter() - wall
    for ep in range(len(returns) + 1, episodes + 1):
        if new012 and agent.rec is None and ep >= episodes - REC_EPIS + 1:
            agent.rec = []
        returns.append(agent.episode(env))
        if save_model_path and (ep % SAVE_EVERY == 0 or ep == episodes):
            snaps[f"W_ep{ep}"] = agent.W.astype(np.float32)
        if ep % LOG_EVERY == 0:
            evals.append({"episode": ep, **agent.evaluate(), **agent.diagnostics()})
            if cfg.beta_kappa:
                evals[-1]["beta"] = agent.beta
            ev = evals[-1]
            kl = f"KL {ev['KL_1d_race']:.3f} " if cfg.backend == "spiking" else ""
            print(f"ep {ep} last50 {np.mean(returns[-50:]):.1f} D {ev['D']:.3f} p_A {ev['p_A']:.3f} "
                  f"persist_LA {ev['persist_LA']:.3f} H(P) {ev['P_entropy']:.2f} |td| {ev['td_abs']:.3f} "
                  f"d_max {ev['d_max']:.2f} " + kl + f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if ep < episodes:
                save_checkpoint(ck, agent, returns, evals, snaps, resumes, time.perf_counter() - t0)
    frames = agent.frames
    res = {"backend": cfg.backend, "routing": cfg.routing, "lam": cfg.lam, "seed": cfg.seed, "episodes": episodes,
           "returns": returns, "model": MODEL, "beta": cfg.beta, "eta_tau": agent.eta_tau, "eta_task": agent.eta,
           "la_beta": cfg.la_beta, "td_norm": cfg.td_norm, "ablate": cfg.ablate, "p_cc": cfg.p_cc,
           **({"p_cross": cfg.p_cross} if cfg.p_cross != 0.05 else {}),
           "beta_kappa": cfg.beta_kappa, "beta_max": cfg.beta_max,
           **({"gamma": cfg.gamma} if cfg.gamma != GAMMA else {}),
           **({"readout": cfg.readout, "bound_deadline_frac": agent.bound_dl / max(agent.bound_n, 1),
               "bound_theta_mean": agent.bound_th / max(agent.bound_n, 1)} if cfg.readout != "race" else {}),
           "evals": evals, "frames": frames, "ms_per_frame": (time.perf_counter() - t0) / frames * 1e3,
           "silent_race_frames": agent.silent, "resumes": resumes,
           "params": dict(alpha=ALPHA, n=N_HOPS, k=K_SPK, rate=RATE, theta=THETA, eta=ETA, tau_e=TAU_E,
                          gamma=cfg.gamma, eta_v=ETA_V, eps=0.0, eps_h=agent.eps_h, burn=agent.burn, eta_task=agent.eta,
                          beta=cfg.beta, eta_tau=agent.eta_tau, sizes=agent.task.sizes)}
    if (cfg.task, cfg.heads) != ("CartPole", 1):
        res.update(task=cfg.task, heads=cfg.heads, engine=cfg.engine, encoder=agent.task.encoder,
                   n_ctrl=agent.task.n_ctrl)
    if agent.qp is not None:
        res.update(qif=dataclasses.asdict(agent.qp))
    if agent.lp is not None:
        res.update(lif=dataclasses.asdict(agent.lp))
    if cfg.backend == "mpr":
        res.update(mpr_stats=agent.mpr_stats)
    if cfg.rule != "v4":
        res.update(rule=cfg.rule, obs_tau=cfg.obs_tau, ebar=cfg.ebar, step_stat=cfg.step_stat, emergent=agent.ep_log,
                   frozen_returns=frozen_returns(agent))
    if cfg.lam_scale != 1.0 or cfg.eta_scale != 1.0 or scales:
        res.update(lam_scale=cfg.lam_scale, eta_scale=cfg.eta_scale, **({"scales": scales} if scales else {}))
    if new012:
        res.update({k: v for k, v in (("encoder_spec", cfg.encoder), ("critic_layer", cfg.critic_layer),
                                      ("c_ent", cfg.c_ent), ("ent_target", cfg.ent_target)) if v},
                   n_ctrl=agent.task.n_ctrl, diag012=diag012(agent))
    fn.write_text(json.dumps(res))
    if save_model_path:
        save_model(save_model_path, agent, snaps, cfg)
    ck.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), default=None, help="required to train; --fidelity: default all")
    ap.add_argument("--backend", choices=("flow", "spiking", "mpr"), default="flow",
                    help="mpr: rates of the QIF network's MPR fixed point, no spikes (Phase B); --q-<field> overrides")
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="default: the task's (tasks.TASKS)")
    ap.add_argument("--engine", choices=("numpy", "snm", "qif", "lif"), default="numpy",
                    help="spiking engine: numpy = sp.run_heads (SuperNeuroMAT's raster, batched over heads); "
                         "qif = mapstdp.qif (Proposal, iteration 013); lif = mapstdp.lif (noisy LIF, --rule emergent)")
    for k, typ in LIF_FLAGS.items():
        ap.add_argument(f"--lif-{k.replace('_', '-')}", dest=f"lif_{k}", type=typ, default=None,
                        help=f"--engine lif: LIFParams.{k} (default {getattr(lif.LIFParams(), k)!r})")
    for k, typ in QIF_FLAGS.items():
        ap.add_argument(f"--q-{k.replace('_', '-')}", dest=f"q_{k}", type=typ, default=None,
                        help=f"--engine qif: QIFParams.{k} (default {getattr(qif.QIFParams(), k)!r})")
    ap.add_argument("--la-beta", type=float, default=LA_BETA, help="latent-output barrier (Eq. 4e, Proposal)")
    ap.add_argument("--td-norm", type=float, default=0.0, help="TD-error normalisation to this RMS (Proposal; 0 = off)")
    ap.add_argument("--eta-tau", type=float, default=None, help="ablation: task-rate annealing (0 = constant)")
    ap.add_argument("--wmin", type=float, default=None, help="ablation: weight floor (default 1e-3)")
    ap.add_argument("--tau-e", type=float, default=None, help="ablation: eligibility time constant (default 3)")
    ap.add_argument("--lam", type=float, default=0.05, help="structural (map) term rate; 0 = map-term ablation")
    ap.add_argument("--p-cc", type=float, default=0.1, help="controller->controller connection probability")
    ap.add_argument("--p-cross", type=float, default=0.05,
                    help="cross-module (non-routing) connection probability; 0 = full modularity (iteration 014)")
    ap.add_argument("--beta", type=float, default=1.0, help="sharpened policy exponent (Eq. 1e, Proposal; flow)")
    ap.add_argument("--eta", type=float, default=None, help="task rate (default: model v3's)")
    ap.add_argument("--beta-kappa", type=float, default=0.0, help="causal adaptive beta gain (Eq. 1g; 0 = off)")
    ap.add_argument("--beta-max", type=float, default=4.0, help="upper bound of the adaptive beta")
    ap.add_argument("--gamma", type=float, default=GAMMA, help="discount of the critic and delta (Eq. 4c)")
    ap.add_argument("--readout", choices=("race", "diffbound"), default="race",
                    help="spiking action readout: race (Eq. 1d) or count-difference bound (Eq. 1h, iteration 010)")
    ap.add_argument("--encoder", default=None, help="controller encoder spec (iteration 012): g<n>s<sigma> | "
                    "f<order>c<conj>[h][b]")
    ap.add_argument("--critic-layer", default=None, help="separate critic RF layer spec (Eq. 4c'', iteration 012)")
    ap.add_argument("--c-ent", type=float, default=0.0, help="entropy term coefficient (Eq. 4h; 0 = off)")
    ap.add_argument("--ent-target", type=float, default=None, help="adapt c_ent toward H(P~) = this x log|A|")
    ap.add_argument("--rule", choices=("v4", "emergent"), default="v4",
                    help="plasticity rule: v4 (models v3/v4) or emergent (observed walk + signed counter, Eqs. 8-12)")
    ap.add_argument("--obs-tau", type=float, default=0.0,
                    help="emergent, spiking: running averages of the walk statistics over this many frames (0 = frame)")
    ap.add_argument("--ebar", choices=("obs", "module"), default="obs", help="emergent ablation: structural baseline")
    ap.add_argument("--step-stat", choices=("c", "kappa"), default="c",
                    help="emergent ablation: structural step factor")
    ap.add_argument("--lam-scale", type=float, default=1.0, help="structural rate x this (any engine)")
    ap.add_argument("--eta-scale", type=float, default=1.0, help="task rate x this (any engine)")
    ap.add_argument("--match-scales", action="store_true",
                    help="set --lam-scale/--eta-scale by match_scales (RMS match to flow at the initial W)")
    ap.add_argument("--out", default=None, help="result path (default experiments/results/<name>.json)")
    ap.add_argument("--save-model", default=None, help="write the trained model (.npz, save_model) here")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--m2-summary", action="store_true", help="the M2 results table (m2_*.json, CartPole v3)")
    ap.add_argument("--ablations", action="store_true", help="summary of the flow learning-rule ablations")
    ap.add_argument("--fidelity", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.m2_summary:
        return summary_m2()
    if a.ablations:
        return ablation_summary()
    if a.fidelity:
        qp = qif.QIFParams(**qif_overrides(a)) if a.engine == "qif" else None
        Hs = (a.heads,) if a.heads != 1 else HEADS
        out = {t: fidelity(t, Hs, engine=a.engine, qp=qp) for t in ([a.task] if a.task else TASKS)}
        tag = f"_{qif_token(a)}" if a.engine == "qif" else ""
        (OUT / f"m3_fidelity{tag}.json").write_text(json.dumps(out, indent=1))
        return
    if a.task is None:
        ap.error("--task is required to train")
    ablate = {k: v for k, v in (("wmin", a.wmin), ("tau_e", a.tau_e)) if v is not None}
    variant = [f"{k}{v:g}" for k, v in ablate.items()] + ([f"at{a.eta_tau:g}"] if a.eta_tau is not None else [])
    variant = "_".join(variant + variant_tokens(a))
    cfg = Config(a.backend, "exempt", a.lam, a.seed, beta=a.beta, eta=a.eta, eta_tau=a.eta_tau, la_beta=a.la_beta,
                    task=a.task, heads=a.heads, engine=a.engine, td_norm=a.td_norm, ablate=ablate, p_cc=a.p_cc,
                    p_cross=a.p_cross, beta_kappa=a.beta_kappa, beta_max=a.beta_max, gamma=a.gamma, readout=a.readout,
                    encoder=a.encoder, critic_layer=a.critic_layer, c_ent=a.c_ent, ent_target=a.ent_target,
                    qif=qif_overrides(a) if a.engine == "qif" or a.backend == "mpr" else {}, rule=a.rule,
                    lif=lif_overrides(a) if a.engine == "lif" else {},
                    lam_scale=a.lam_scale, eta_scale=a.eta_scale, obs_tau=a.obs_tau, ebar=a.ebar,
                    step_stat=a.step_stat)
    scales = None
    if a.match_scales:
        assert (a.q_lam_scale or 1.0) == 1.0 and (a.q_eta_scale or 1.0) == 1.0, "--match-scales: QIFParams scales 1"
        assert (a.lif_lam_scale or 1.0) == 1.0 and (a.lif_eta_scale or 1.0) == 1.0, "--match-scales: LIFParams scales 1"
        scales = match_scales(cfg)
        cfg = dataclasses.replace(cfg, lam_scale=cfg.lam_scale * scales["lam_scale"],
                                  eta_scale=cfg.eta_scale * scales["eta_scale"])
        print("match_scales: " + " ".join(f"{k} {v:.4g}" for k, v in scales.items()), flush=True)
    r = run(cfg, a.episodes or TASKS[a.task].episodes,
               name=name(a.task, a.backend, a.heads, a.la_beta, a.seed, a.td_norm, variant), out=a.out,
               save_model_path=a.save_model, scales=scales)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
