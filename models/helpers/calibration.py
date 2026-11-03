"""Pre-run calibrations of the TD-STDP backbone: rates, input scale, entropy match, race bound and its
re-probe, and the recurrent columns' eta_rec / lambda_rec (all on frozen probes, own rng streams)."""
import dataclasses

import numpy as np

from mapstdp import tdstdp as td
from models.helpers.runner import CAL_SEED, CAL_TICKS, MS, WARM_N, CKRunner, Config, Runner, race_theta


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
