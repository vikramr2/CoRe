"""Noisy LIF engine for the emergent-walk Map-STDP rule (derivation §2.1-2.2, §4.1-4.2, §5.3, §7; details D8).

Wiring (for models/basic.py; nothing here imports it)
--------------------------------------------------------
The agent's contract (models/basic.py) for rule='emergent' engines is Agent.counter_frame(v, rng) -> (S, kappa_raw,
c), all raw and summed over heads; obs_frame divides kappa_raw and c by the frame's total spikes. For engine='lif':

    p = lif.LIFParams()                          # or dataclasses.replace(LIFParams(), ...) from Config fields
    # episode start (every reset of the environment):
    agent.lstate = lif.init_state(N, H, p)       # no rng draws; the warm-up runs inside the first frame
    # each frame (warm state carried; p.warm=False re-initialises every frame):
    S, kappa_raw, c = lif.counter_frame(agent.W, v, agent.heads, rng, agent.lstate, p)
    # S          p.k x (H N) bool raster (dt = 0.5 ms bins, head h in columns h N .. (h + 1) N - 1), so sp.pi_hat,
    #            sp.race, sp.diffbound and sp.head_groups apply unchanged; race burn-in is in steps of p.dt;
    # kappa_raw  N x N [post, pre] pair counts in the window w = p.window (Eq. 8);
    # c          N x N signed counter (Eq. 11b by default); its expectation is the weight-scaled derivative
    #            n_j W_ij dE_ij / dW_ij (= n_j D_ij in linear response; see "What the counter estimates" below).
    # Learning-rate scales: lam * p.lam_scale, eta * p.eta_scale, eps_h * p.eps_h_scale (all 1 by default).
    # Optional monitor (Eq. 12, continuous form): lif.stability(c, n, m, K)[1] < 1, with n the per-neuron spike counts
    # of the frame summed over heads (S.reshape(k, H, N).sum((0, 1))).
    # Readout caveat: every neuron fires at its background rate (~5-10 Hz), so with H = 16 heads the action
    # modules emit a spike every ~0.2 ms by chance and sp.race after a short burn-in is close to a coin flip; a
    # count-based readout over the frame (sp.diffbound, or the spike-share policy) is the natural one here (untested).

All randomness comes from the rng passed per frame (evaluate() and match_scales() may pass their own streams); the
state carries no generator. The first frame of an episode (state['f'] == 0) and every frame with p.warm = False first
runs p.warmup steps on that frame's stimulus (default 3 tau_m), discarded, from V ~ N(mu_bg, sigma_V) clipped below
threshold: this removes the onset transient from the frame's statistics (warm state).

Model (normalised units: rest 0, threshold theta = 1, reset V_r = 0; time in seconds; rates in Hz)
-----------------------------------------------------------------------------------------------
    V(t+dt) = mu_bg + a (V(t) - mu_bg) + sigma_V sqrt(1 - a^2) xi + sum_j b_ij x_j(t) + a_ext e_i(t),  a = e^{-dt/tau_m}

the exact OU step of tau_m dV = -(V - mu_bg) dt + sigma_V sqrt(2 tau_m) dW (stationary s.d. sigma_V: the background
input of derivation §5.3) plus pulses. Spike when V >= theta (p.delta_V = 0), or with escape noise (p.delta_V > 0) with
probability 1 - exp(-rho0 dt exp((V - theta) / delta_V)) per step; then V = V_r, clamped for tau_ref (p.reset = False,
escape only: no reset or clamp, a spike only blocks the next spike for tau_ref: the reset-free Hawkes-LIF in which the
psi counter is first-order exact). Synaptic delay is one step (a spike at step t kicks its targets at step t + 1).
  Kicks: b_ij = w0 W_ij / d_j (output-normalised, as branching; excitatory only), w0 = chi (1 - a) / dt ~ chi / tau_m,
    so one spike of j injects a total input-pulse area chi (theta s) across its targets (derivation §2: chi = w0 tau_m).
  External drive: e_i(t) ~ Poisson(ext_rate v_i dt) pulses of amplitude a_ext (ext_rate: Hz per head; v sums to 1).
  Defaults (calibrate()): sigma_V = 0.2, background 5 Hz in a homogeneous network with first-order offspring
    chi G = 0.5 there (mu_bg = 0.567, chi = 0.0157 theta s, G = 31.8 Hz / theta, w0 = 0.78 theta). Not 0.8 as in
    branching: the LIF transfer is convex, rates also raise the shot-noise variance, and at chi G = 0.8 the
    effective offspring is ~0.99 and strong stimuli ignite a high-rate state (bistability; see calibrate,
    mean_field). ext_rate 1500 Hz, a_ext = 0.1: a controller neuron with v_i = 0.15 (the CartPole grid peak) gets
    225 Hz of 0.1-theta pulses, +0.45 theta of mean drive (~40 Hz); v4 L / A modules then fire at 8-12 Hz.

Statistics per frame, pooled over heads (cross-head pairs excluded: heads share weights, not synapses)
-----------------------------------------------------------------------------------------------------
  kappa_raw[i, j] = sum over spikes of j at step t of the spikes of i in steps (t, t + W], W = window / dt (Eq. 8).
    Straddling: with warm state the last W steps of the raster are carried (state['tail']); pairs whose presynaptic
    spike is in the previous frame and postsynaptic spike in this one are credited to this frame, so over an episode
    every pair is counted once. Cold frames (warm=False) and the first frame after init carry the warm-up's tail.
    The diagonal (j's own later spikes) is kept, as in the branching engine.
  n_j: spikes of j; mu_i, sig_i: mean and s.d. of i's free membrane potential, from its own summed input pulses
    (mu = mu_bg + <x> / (1 - a), sig^2 = sigma_V^2 + var(x) / (1 - a^2), x = per-step input; white-input
    approximation), the operating point the Siegert gain needs.
  Counter c (p.counter):
    'siegert'    c_ij = omega tau_eff G_i b_ij n_j (Eq. 11b), G from (mu_i, sqrt 2 sig_i) through Eq. 11c with the
                 discrete-time threshold correction (siegert_rate, dt); tau_eff = dt / (1 - a) is the area of one
                 kick's voltage trace. Optional EMA of G over p.gain_tau frames. Requires delta_V = 0.
    'psi'        the time-resolved escape-noise counter c_ij = b_ij sum_{t_f} sum_{t in (t_f, t_f + W]}
                 a^{t - t_f - 1} psi_i(t), psi_i = dP(spike at t) / dV (zero while refractory); presynaptic trace times
                 postsynaptic psi, crossbar-native. Windows straddle frames the same way as kappa. Requires delta_V > 0.
                 Only first-order exact without reset (p.reset = False): with the LIF reset it overstates the truth
                 4-9x (each extra spike's reset cancels most of the hazard increase; cutting the kernel at i's next
                 spike, a per-synapse trace, still leaves 4-7x). Use 'siegert' or 'regression' for a resetting LIF.
    'regression' c_ij = omega tau_eff G_i b_ij n_j with G_i the slope of i's frame rate on its frame input mean mu_i,
                 EMA moments over p.reg_tau frames (Siegert until p.reg_min frames have passed or the input variance
                 is degenerate). Per-neuron slopes are noisy (frame-level mu varies little across stimuli for L / A
                 neurons); derivation §5.3 also expects an upward bias from recurrent feedback.
  omega (the in-window share, ~0.95 at w = 3 tau_m) defaults to 1 as in Eq. 11b.

What the counter estimates (self-check (c))
  Two ground truths by open-loop replay with common random numbers: the removal effect n(b) - n(0) (the walk's
  finite direct part D_ij of Eq. 8) and the marginal effect b dn/db (the weight-scaled derivative W dE/dW that Lemma E
  and Theorem E use). They coincide in linear response, but here one kick is 0.04-0.23 theta against sigma ~0.2-0.3
  and the response is strongly convex on that scale: removal is ~0.65 of marginal (as low as 0.48 for single
  neurons). The Siegert counter tracks the marginal effect (corr 0.95, ratio 1.04), which is the quantity the rule's
  descent needs; it overstates the removal effect by ~1.5x, so D_ij in Eq. 8 (and the stability monitor built from c)
  is the linearised one, not the finite one.

Self-check: python -m mapstdp.lif (~20 s): (a, b) single-neuron Siegert rate and gain against simulation; (c) the
counters against the replay truths on a frozen 30-neuron network; (d) pair counts against the definition, across frame
boundaries and over warm engine frames; (e) operating point, stability monitor and ms/frame on the v4 CartPole and
LunarLander networks at H = 16; (f) at dt = 1 ms: the DT1 calibration, Siegert against simulation, simulate_ext.

Kernel: with numba (and delta_V = 0, the default), _simulate runs the compiled loop _kernel_nb, which draws the
background noise and the external pulses inside the loop from numba's generator, seeded once per call from the frame's
rng (one rng.integers draw). It has the same model and distributions as the numpy loop, but not the same random
stream: the external pulses are drawn as one Poisson count per head and step, spread over the support by v (Poisson
splitting). The numpy loop (escape noise, record=True, or kernel='np') is the reference.

External current (simulate_ext; the noisy-LIF engine of models.tdstdp, --engine nlif): _kernel_ext_{py,nb} take a
given per-step current I[t, h, i] (signed; the caller's feed-forward input) in place of the teleport pulses, plus
recurrent kicks on an index set (none = an empty set); same noise, leak, threshold and reset as _kernel_nb. At dt = 1 ms
(CK's grid) the background point is DT1. With sigma_V = 0, mu_bg = 0, no refractory period and CK's tau and threshold
it is BindsNET's LIF (tdstdp self-check (h1)). Its gain at that grid is gain_markov (the kernel's own transfer from the
Markov chain of V, markov_rate), not Siegert: the posts there run mean-driven at 100-500 Hz, where the diffusion limit
misses the drift regime and the mode locking of the 1 ms grid (rates locked at 500 Hz; iteration 015)."""
import dataclasses
import sys
import time
from math import comb

import numpy as np
from scipy.special import erfcx, erfi, ndtr

try:
    import numba
except ImportError:  # numpy loop only
    numba = None

_XG, _WG = np.polynomial.legendre.leggauss(48)
RHO_OVERSHOOT = 0.5826  # -zeta(1/2) / sqrt(2 pi): expected overshoot of a Gaussian random walk, in step s.d.


# ---------------------------------------------------------------- Siegert (Eq. 11c)
def _int_erfcx(a, b):
    """int_a^b erfcx(x) dx (a <= b; bounded smooth integrand), 48-point Gauss-Legendre."""
    h = (b - a) / 2
    x = ((a + b) / 2)[..., None] + h[..., None] * _XG
    return h * (erfcx(x) * _WG).sum(-1)


def _int_g(lo, hi):
    """int_lo^hi g(x) dx, g(x) = e^{x^2} (1 + erf x) = erfcx(-x), without overflow: split at 0; on x > 0,
    g = 2 e^{x^2} - erfcx(x) and int 2 e^{x^2} = sqrt(pi) erfi."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    nl, nh, pl, ph = np.minimum(lo, 0), np.minimum(hi, 0), np.maximum(lo, 0), np.maximum(hi, 0)
    return _int_erfcx(-nh, -nl) + np.sqrt(np.pi) * (erfi(np.minimum(ph, 26.0)) - erfi(np.minimum(pl, 26.0))) \
        - _int_erfcx(pl, ph)


def _ys(mu, sigma_B, tau_m, th, vr, dt):
    mu, sB = np.asarray(mu, float), np.maximum(np.asarray(sigma_B, float), 1e-9)
    th_e = th + RHO_OVERSHOOT * sB * np.sqrt(dt / tau_m) if dt else th  # step s.d. = sB sqrt(dt / tau)
    return (th_e - mu) / sB, (vr - mu) / sB, sB


def siegert_rate(mu, sigma_B, tau_m=0.02, th=1.0, vr=0.0, tau_ref=0.0, dt=0.0):
    """Stationary rate (Hz) of an LIF neuron with white-noise input [Siegert 1951; Brunel & Hakim 1999] (Eq. 11c):
    1 / r = tau_ref + tau_m sqrt(pi) int_{y_r}^{y_th} g(x) dx, y = (V - mu) / sigma_B, sigma_B = sqrt(2) x the
    stationary s.d. of the free membrane potential. dt > 0: discrete-time simulation with the threshold checked once
    per step; the threshold is raised by the expected overshoot 0.5826 x the per-step s.d. (Siegmund's continuity
    correction for discretely monitored barriers, as in Broadie, Glasserman & Kou 1997)."""
    yt, yr, _ = _ys(mu, sigma_B, tau_m, th, vr, dt)
    return 1.0 / (tau_ref + tau_m * np.sqrt(np.pi) * _int_g(yr, yt))


def siegert_gain(mu, sigma_B, tau_m=0.02, th=1.0, vr=0.0, tau_ref=0.0, dt=0.0):
    """G = dr / dmu (Hz per unit of mean drive; Eq. 11c): r^2 tau_m sqrt(pi) [g(y_th) - g(y_r)] / sigma_B."""
    yt, yr, sB = _ys(mu, sigma_B, tau_m, th, vr, dt)
    r = 1.0 / (tau_ref + tau_m * np.sqrt(np.pi) * _int_g(yr, yt))
    return r ** 2 * tau_m * np.sqrt(np.pi) / sB * (erfcx(-np.minimum(yt, 26.0)) - erfcx(-yr))


# ---------------------------------------------------------------- parameters and state
@dataclasses.dataclass(frozen=True)
class LIFParams:
    tau_m: float = 0.02  # membrane time constant (s)
    dt: float = 5e-4  # step (s)
    th: float = 1.0  # threshold (normalised units)
    vr: float = 0.0  # reset
    tau_ref: float = 0.002  # absolute refractory period (s), clamped at vr
    mu_bg: float = 0.5666  # background mean drive (free-membrane mean without recurrent or external input)
    sigma_V: float = 0.2  # background noise: stationary s.d. of the free membrane potential
    delta_V: float = 0.0  # escape-noise width (0 = hard threshold); counter 'psi' needs > 0
    rho0: float = 200.0  # escape hazard at threshold (Hz)
    reset: bool = True  # False (escape noise only): no reset or clamp, spikes only gate the hazard for tau_ref
    #                    (Hawkes-LIF; the case in which the psi counter is first-order exact)
    chi: float = 0.01572  # input-pulse area of one spike's total output (theta s); chi G = 0.5 at the background point
    a_ext: float = 0.1  # external pulse amplitude
    ext_rate: float = 1500.0  # external pulses per second per head, spread by v (v sums to 1)
    k: int = 200  # steps per frame (100 ms)
    window: float = 0.06  # walk window w (s), 3 tau_m
    counter: str = "siegert"  # 'siegert' | 'psi' | 'regression'
    omega: float = 1.0  # in-window share applied to the mean counters (Eq. 11b takes 1)
    gain_tau: float = 0.0  # EMA of the Siegert gain over frames (0 = this frame only)
    reg_tau: float = 100.0  # regression counter: EMA time constant of the moments (frames)
    reg_min: int = 20  # regression counter: frames before the regression replaces Siegert
    warm: bool = True  # carry V, refractory, in-flight spikes and the window tail across frames
    warmup: int = 120  # discarded steps on the first frame of an episode (and every cold frame), 3 tau_m
    lam_scale: float = 1.0  # agent: structural rate lambda x this
    eta_scale: float = 1.0  # agent: task rate eta x this
    eps_h_scale: float = 1.0  # agent: d_j homeostasis rate x this
    kernel: str = "nbf"  # numba loop when importable and delta_V = 0 (module doc): 'nb' Gaussian noise, 'nbf' fast
    #                     variance-matched noise; 'np': the numpy loop

    @property
    def a(self):
        return float(np.exp(-self.dt / self.tau_m))

    @property
    def tau_eff(self):
        """Area of one unit kick's voltage trace, sum_s a^s dt = dt / (1 - a) (~ tau_m + dt / 2)."""
        return self.dt / (1 - self.a)

    @property
    def w0(self):
        """Total kick of one spike over its targets (theta): its voltage-trace area is chi."""
        return self.chi / self.tau_eff

    @property
    def W_steps(self):
        return int(round(self.window / self.dt))

    @property
    def nref(self):
        return int(round(self.tau_ref / self.dt))


# dt = 1 ms (CK's grid; models.tdstdp --engine nlif): calibrate(LIFParams(dt=1e-3, tau_ref=0)), the defaults'
# operating point (5 Hz, chi G = 0.5 in the homogeneous recurrent network); without recurrent input a neuron fires at
# 2.5 Hz there. No refractory period, as CK's BindsNET posts: a 2 ms clamp caps the rate at 1/3 per ms, below their
# initial LunarLander rate (0.36 per ms) and the critic's value range on CartPole (V = 2 rho - 0.2 needs rho ~ 0.6)
DT1 = dict(dt=1e-3, tau_ref=0.0, mu_bg=0.5778, chi=0.01551)
# the low-rate nlif regime (models.tdstdp --rate-target, iteration 015 step 3): the default 2 ms refractory period
# at dt = 1 ms, calibrate(LIFParams(dt=1e-3)); rate ceiling 333 Hz; at 40 Hz the kernel's gain is 41 Hz / theta
# (mu ~ 1.48), near its peak of ~46 at 15-20 Hz
DT1R = dict(dt=1e-3, tau_ref=0.002, mu_bg=0.5778, chi=0.01575)


def kicks(W, p):
    """b_ij = w0 W_ij / d_j (post, pre)."""
    return p.w0 * W / np.maximum(W.sum(0), 1e-12)


def init_state(N, H, p):
    """Per-episode state (no rng draws: the warm-up of the first frame draws from the frame's rng).
    V, ref, z: (H, N) membrane, refractory steps left, spikes of the last step (in flight); tail: the last W steps of
    the raster (W, H, N) for straddling windows; G, reg: gain EMA and regression moments (per neuron, pooled heads);
    f: frames run."""
    return dict(V=None, ref=np.zeros((H, N), np.int64), z=np.zeros((H, N), bool),
                tail=np.zeros((p.W_steps, H, N), bool), G=None, reg=None, f=0, N=N, H=H)


def _draws(v, H, N, k, rng, p):
    """Per-step random inputs: background noise (k, H, N), Poisson external pulse counts on the support of v,
    escape uniforms if delta_V > 0. Draw order: noise, external, escape."""
    noise = rng.standard_normal((k, H, N)) * (p.sigma_V * np.sqrt(1 - p.a ** 2))
    sup = np.flatnonzero(v > 0)
    ext = rng.poisson(p.ext_rate * p.dt * v[sup], (k, H, len(sup))) * p.a_ext
    U = rng.random((k, H, N)) if p.delta_V > 0 else None
    return noise, sup, ext, U


def _step(V, ref, drive, u, p):
    """One step, in place on V and ref: OU leak plus drive, refractory clamp, threshold or escape, reset.
    Returns (spikes, psi); psi = dP(spike) / dV for escape noise (0 while refractory), else None."""
    V *= p.a
    V += (1 - p.a) * p.mu_bg + drive
    act = ref == 0
    np.maximum(ref - 1, 0, out=ref)
    if p.reset:
        V[~act] = p.vr
    psi = None
    if p.delta_V > 0:
        rho = p.rho0 * p.dt * np.exp(np.minimum((V - p.th) / p.delta_V, 30.0))
        q = np.exp(-rho)  # survival of the step
        z = act & (u < 1 - q)
        psi = np.where(act, q * rho / p.delta_V, 0.0)
    else:
        z = act & (V >= p.th)
    if p.reset:
        V[z] = p.vr
    ref[z] = p.nref
    return z, psi


def _simulate(Bt, v, st, k, rng, p, record=False):
    """k steps of H heads from state st (mutated: V, ref, z). Returns (S, xsum, xsq, Psi, rec) where S (k, H, N) bool,
    xsum/xsq (H, N) sums of the per-step input pulses (recurrent + external) and their squares, Psi (k, H, N) or
    None, rec (record=True) = dict(drive, V, ref, U) per step for the open-loop replay of the self-check."""
    H, N = st["z"].shape
    if _kernel_nb is not None and p.delta_V == 0 and not record and p.kernel in ("nb", "nbf"):
        return _simulate_nb(Bt, v, st, k, rng, p)
    noise, sup, ext, U = _draws(v, H, N, k, rng, p)
    V, ref, z = st["V"], st["ref"], st["z"]
    S = np.zeros((k, H, N), bool)
    Psi = np.zeros((k, H, N)) if p.delta_V > 0 else None
    xsum, xsq = np.zeros((H, N)), np.zeros((H, N))
    if record:
        rec = dict(drive=np.zeros((k, H, N)), V=np.zeros((k, H, N)), ref=np.zeros((k, H, N), np.int64), U=U)
    for t in range(k):
        x = z @ Bt
        if len(sup):
            x[:, sup] += ext[t]
        xsum += x
        xsq += x * x
        drive = x + noise[t]
        z, psi = _step(V, ref, drive, None if U is None else U[t], p)
        S[t] = z
        if psi is not None:
            Psi[t] = psi
        if record:
            rec["drive"][t], rec["V"][t], rec["ref"][t] = drive, V, ref
    st["z"] = z
    return S, xsum, xsq, Psi, (rec if record else None)


def _kernel_py(Bt, sup, cv, lam, V, ref, z, k, seed, a, drift, sig_n, a_ext, th, vr, nref, S, xsum, xsq, fast):
    """The numpy loop's step (_step, delta_V = 0) per head and neuron, compiled by numba: recurrent input summed over
    the presynaptic spikes only; external pulses as one Poisson(lam) count per head and step, each sent to sup[s] with
    probability v_s / sum v (cv: cumulative shares); noise drawn per neuron and step, Gaussian, or with fast=True the
    variance-matched sum of four uniforms from a xorshift64* generator (module doc). Mutates V, ref, z, S, xsum,
    xsq."""
    np.random.seed(seed)
    xs = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) | np.uint64(1)
    H, N = V.shape
    x = np.empty(N)
    spk = np.empty(N, np.int64)
    for t in range(k):
        for h in range(H):
            x[:] = 0.0
            ns = 0
            for j in range(N):
                if z[h, j]:
                    spk[ns] = j
                    ns += 1
            for q in range(ns):
                j = spk[q]
                for i in range(N):
                    x[i] += Bt[j, i]
            for _ in range(np.random.poisson(lam)):
                s = np.searchsorted(cv, np.random.random(), side="right")
                x[sup[min(s, len(sup) - 1)]] += a_ext
            for i in range(N):
                xi = x[i]
                xsum[h, i] += xi
                xsq[h, i] += xi * xi
                if fast:
                    xs ^= xs >> np.uint64(12)
                    xs ^= xs << np.uint64(25)
                    xs ^= xs >> np.uint64(27)
                    y = xs * np.uint64(2685821657736338717)
                    m16 = np.uint64(0xFFFF)
                    u4 = (y & m16) + ((y >> np.uint64(16)) & m16) + ((y >> np.uint64(32)) & m16) + (y >> np.uint64(48))
                    g = ((float(u4) + 2.0) / 65536.0 - 2.0) * 1.7320508075688772  # Irwin-Hall(4), mean 0, var 1
                else:
                    g = np.random.standard_normal()
                Vi = V[h, i] * a + drift + xi + g * sig_n
                act = ref[h, i] == 0
                if ref[h, i] > 0:
                    ref[h, i] -= 1
                if not act:
                    Vi = vr
                fire = act and Vi >= th
                if fire:
                    Vi = vr
                    ref[h, i] = nref
                V[h, i] = Vi
                z[h, i] = fire
                S[t, h, i] = fire


_kernel_nb = numba.njit(cache=True)(_kernel_py) if numba is not None else None


def _simulate_nb(Bt, v, st, k, rng, p):
    """_simulate through _kernel_nb (module doc): returns (S, xsum, xsq, None, None); one rng draw (the seed)."""
    H, N = st["z"].shape
    sup = np.flatnonzero(v > 0)
    vs = v[sup]
    cv = np.cumsum(vs) / max(vs.sum(), 1e-300)
    S = np.zeros((k, H, N), bool)
    xsum, xsq = np.zeros((H, N)), np.zeros((H, N))
    seed = int(rng.integers(1 << 31))
    _kernel_nb(np.ascontiguousarray(Bt), sup, cv, p.ext_rate * p.dt * float(vs.sum()), st["V"], st["ref"], st["z"], k,
               seed, p.a, (1 - p.a) * p.mu_bg, p.sigma_V * np.sqrt(1 - p.a ** 2), p.a_ext, p.th, p.vr, p.nref, S,
               xsum, xsq, p.kernel == "nbf")
    return S, xsum, xsq, None, None


_NO_SCORE = np.empty((1, 1, 1))  # _kernel_ext_py: the score outputs when score=False (never written)


def _kernel_ext_py(Iext, Bt, ridx, V, ref, z, k, seed, a, drift, sig_n, th, vr, nref, S, xsum, xsq, fast, score=False,
                   M=_NO_SCORE, GN=_NO_SCORE):
    """_kernel_py with a given external current instead of the teleport pulses: neuron i of head h gets Iext[t, h, i]
    (signed, any units of th) at step t, plus the recurrent kicks Bt[q, r] from spikes of ridx[q] at step t - 1 onto
    ridx[r] (an index set; empty = none). Same noise (Gaussian, or fast=True Irwin-Hall(4) from xorshift64*), leak,
    threshold, reset and refractory clamp. Mutates V, ref, z, S, xsum, xsq. score=True (post_scores): also writes
    M[t, h, i] = the pre-noise membrane m = a V + drift + I (NaN while refractory: the inactive mask) and GN[t, h, i] =
    the standardised noise draw g of the step (V' = m + g sig_n); no extra draw, the default path is unchanged."""
    np.random.seed(seed)
    xs = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) | np.uint64(1)
    H, N = V.shape
    nr = len(ridx)
    x = np.empty(N)
    spk = np.empty(max(nr, 1), np.int64)
    for t in range(k):
        for h in range(H):
            for i in range(N):
                x[i] = Iext[t, h, i]
            ns = 0
            for q in range(nr):
                if z[h, ridx[q]]:
                    spk[ns] = q
                    ns += 1
            for s in range(ns):
                q = spk[s]
                for r in range(nr):
                    x[ridx[r]] += Bt[q, r]
            for i in range(N):
                xi = x[i]
                xsum[h, i] += xi
                xsq[h, i] += xi * xi
                if fast:
                    xs ^= xs >> np.uint64(12)
                    xs ^= xs << np.uint64(25)
                    xs ^= xs >> np.uint64(27)
                    y = xs * np.uint64(2685821657736338717)
                    m16 = np.uint64(0xFFFF)
                    u4 = (y & m16) + ((y >> np.uint64(16)) & m16) + ((y >> np.uint64(32)) & m16) + (y >> np.uint64(48))
                    g = ((float(u4) + 2.0) / 65536.0 - 2.0) * 1.7320508075688772  # Irwin-Hall(4), mean 0, var 1
                else:
                    g = np.random.standard_normal()
                m = V[h, i] * a + drift + xi
                Vi = m + g * sig_n
                act = ref[h, i] == 0
                if score:
                    M[t, h, i] = m if act else np.nan
                    GN[t, h, i] = g
                if ref[h, i] > 0:
                    ref[h, i] -= 1
                if not act:
                    Vi = vr
                fire = act and Vi >= th
                if fire:
                    Vi = vr
                    ref[h, i] = nref
                V[h, i] = Vi
                z[h, i] = fire
                S[t, h, i] = fire


_kernel_ext_nb = numba.njit(cache=True)(_kernel_ext_py) if numba is not None else None


def simulate_ext(Iext, Bt, ridx, st, rng, p, score=False):
    """k = len(Iext) steps of H heads from state st (mutated: V, ref, z) driven by the external current Iext (k x H x N,
    signed, units of th; the caller's feed-forward input) plus recurrent kicks Bt (len(ridx) x len(ridx), [pre, post]
    within the index set ridx; pass np.zeros((0, 0)) and an empty ridx for none). Hard threshold only (delta_V = 0).
    Kernel by p.kernel: 'nbf' (numba, fast noise), 'nb' (numba, Gaussian), 'np' (the same loop interpreted, Gaussian;
    the reference). One rng draw (the seed). Returns (S (k, H, N) bool, xsum, xsq (H, N) sums of the per-step input
    and its square, as _simulate); with score=True also (M, GN), the pre-noise membrane (NaN while refractory) and the
    standardised noise draw per step (post_scores), from the same draws."""
    assert p.delta_V == 0
    k, H, N = Iext.shape
    S = np.zeros((k, H, N), bool)
    xsum, xsq = np.zeros((H, N)), np.zeros((H, N))
    M, GN = (np.empty((k, H, N)), np.empty((k, H, N))) if score else (_NO_SCORE, _NO_SCORE)
    seed = int(rng.integers(1 << 31))
    kern = _kernel_ext_nb if _kernel_ext_nb is not None and p.kernel in ("nb", "nbf") else _kernel_ext_py
    with np.errstate(over="ignore"):  # the interpreted xorshift wraps uint64 by design
        kern(np.ascontiguousarray(Iext, float), np.ascontiguousarray(Bt, float), np.asarray(ridx, np.int64), st["V"],
             st["ref"], st["z"], k, seed, p.a, (1 - p.a) * p.mu_bg, p.sigma_V * np.sqrt(1 - p.a ** 2), p.th, p.vr,
             p.nref, S, xsum, xsq, p.kernel == "nbf", score, M, GN)
    return (S, xsum, xsq, M, GN) if score else (S, xsum, xsq)


def _warmup(Bt, v, st, rng, p):
    """Fresh V ~ N(mu_bg, sigma_V) clipped below threshold, then p.warmup discarded steps; the tail is the warm-up's
    last W steps (zero-padded if warmup < W)."""
    H, N = st["z"].shape
    st["V"] = np.minimum(p.mu_bg + p.sigma_V * rng.standard_normal((H, N)), p.th - 1e-3)
    st["ref"][:] = 0
    st["z"] = np.zeros((H, N), bool)
    tail = np.zeros((p.W_steps, H, N), bool)
    if p.warmup > 0:
        S = _simulate(Bt, v, st, p.warmup, rng, p)[0]
        n = min(p.W_steps, p.warmup)
        tail[p.W_steps - n:] = S[-n:]
    st["tail"] = tail


def pair_counts(full, W_steps, start):
    """kappa[i, j] = sum over (pre j at u, post i at s) with u < s <= u + W_steps and s >= start, same head, of a
    (T, H, N) raster whose rows before `start` are the carried tail (Eq. 8; window counts by cumulative sums).
    With numba: _pair_counts_nb, the same integer counts, summed over the presynaptic spikes only."""
    if _pair_counts_nb is not None:
        return _pair_counts_nb(np.ascontiguousarray(full), W_steps, start)
    return _pair_counts_np(full, W_steps, start)


def _pair_counts_py(full, W_steps, start):
    """pair_counts per presynaptic spike: kappa[:, j] += the window's spike counts of every i (cumulative sums)."""
    T, H, N = full.shape
    cs = np.zeros((T + 1, H, N), np.int64)
    for t in range(T):
        for h in range(H):
            for i in range(N):
                cs[t + 1, h, i] = cs[t, h, i] + full[t, h, i]
    kappa = np.zeros((N, N))
    for u in range(T):
        lo, hi = max(u + 1, start), min(u + W_steps, T - 1)
        if hi < lo:
            continue
        for h in range(H):
            for j in range(N):
                if full[u, h, j]:
                    for i in range(N):
                        kappa[i, j] += cs[hi + 1, h, i] - cs[lo, h, i]
    return kappa


_pair_counts_nb = numba.njit(cache=True)(_pair_counts_py) if numba is not None else None


def _pair_counts_np(full, W_steps, start):
    """pair_counts in numpy (the reference)."""
    T, H, N = full.shape
    cs = np.concatenate([np.zeros((1, H, N)), np.cumsum(full, 0, dtype=float)])  # cs[s] = sum of rows < s
    u = np.arange(T)
    lo, hi = np.maximum(u + 1, start), np.minimum(u + W_steps, T - 1)
    ok = hi >= lo
    F = np.where(ok[:, None, None], cs[np.minimum(hi + 1, T)] - cs[np.minimum(lo, T)], 0.0)
    return F.reshape(-1, N).T @ full.reshape(-1, N).astype(float)


def presyn_trace(full, W_steps, a, start):
    """xbar(t) = sum_{s=0}^{W-1} a^s x(t - 1 - s) for the rows t >= start of a (T, H, N) raster (the psi counter's
    presynaptic trace: the voltage kernel of each spike in its window)."""
    T = len(full)
    X = np.zeros((T - start,) + full.shape[1:])
    xb = np.zeros(full.shape[1:])
    aW = a ** W_steps
    for t in range(1, T):  # recursion over the whole buffer (the tail rows rebuild the carried trace)
        xb = a * xb + full[t - 1]
        if t - 1 - W_steps >= 0:
            xb = xb - aW * full[t - 1 - W_steps]
        if t >= start:
            X[t - start] = xb
    return X


def input_stats(xsum, xsq, steps, p):
    """Free-membrane mean and s.d. per neuron from the pooled per-step input sums (white-input approximation)."""
    m1, m2 = xsum / steps, xsq / steps
    mu = p.mu_bg + m1 / (1 - p.a)
    var = p.sigma_V ** 2 + np.maximum(m2 - m1 ** 2, 0.0) / (1 - p.a ** 2)
    return mu, np.sqrt(var)


def gain(mu, sig, p):
    """Siegert gain (Hz per theta) at the operating point (mu, sig), discrete-time corrected."""
    return siegert_gain(mu, np.sqrt(2) * sig, p.tau_m, p.th, p.vr, p.tau_ref, p.dt)


def _ih4_cdf(x):
    """CDF of the 'nbf' kernel noise, (U_1 + ... + U_4 - 2) sqrt(3) (Irwin-Hall(4), mean 0, variance 1)."""
    y = np.clip(np.asarray(x, float) / np.sqrt(3) + 2, 0, 4)
    return sum((-1) ** k * comb(4, k) * np.where(y > k, y - k, 0.0) ** 4 for k in range(5)) / 24


def _ih4_pdf(x):
    """Density of the 'nbf' kernel noise (the derivative of _ih4_cdf)."""
    y = np.clip(np.asarray(x, float) / np.sqrt(3) + 2, 0, 4)
    return sum((-1) ** k * comb(4, k) * np.where(y > k, y - k, 0.0) ** 3 for k in range(5)) / (6 * np.sqrt(3))


def _ih4_dpdf(x):
    """Derivative of _ih4_pdf."""
    y = np.clip(np.asarray(x, float) / np.sqrt(3) + 2, 0, 4)
    return sum((-1) ** k * comb(4, k) * np.where(y > k, y - k, 0.0) ** 2 for k in range(5)) / 6


H_CLIP = 1e-3  # post_scores: the hazard score's psi is clipped at its value where the hazard is H_CLIP
_IH4_C = [(-1) ** k * comb(4, k) for k in range(5)]


def _ih4_fused(x, need_df=False):
    """(F, f, f') of the Irwin-Hall(4) noise from one evaluation of the (y - k)_+ terms (post_scores; the same
    arithmetic as _ih4_cdf, _ih4_pdf, _ih4_dpdf term by term, so the values are theirs)."""
    y = np.clip(np.asarray(x, float) / np.sqrt(3) + 2, 0, 4)
    t = [np.maximum(y - k, 0.0) for k in range(5)]
    F = sum(c * tk ** 4 for c, tk in zip(_IH4_C, t)) / 24
    f = sum(c * tk ** 3 for c, tk in zip(_IH4_C, t)) / (6 * np.sqrt(3))
    return F, f, (sum(c * tk ** 2 for c, tk in zip(_IH4_C, t)) / 6 if need_df else None)


def _gauss_fused(x, need_df=False):
    f = _gauss_pdf(x)
    return ndtr(x), f, (-x * f if need_df else None)


def noise_dist(p):
    """(F, f, f') of the kernel's standardised noise: Irwin-Hall(4) for p.kernel 'nbf', Gaussian otherwise."""
    if p.kernel == "nbf":
        return _ih4_cdf, _ih4_pdf, _ih4_dpdf
    return ndtr, _gauss_pdf, lambda u: -u * _gauss_pdf(u)


def _gauss_pdf(u):
    return np.exp(-0.5 * np.square(u)) / np.sqrt(2 * np.pi)


_PSI_MAX = {}


def psi_max(p):
    """The hazard score's clip: psi = f(z) / (sig_n h (1 - h)) at the z where the hazard h = 1 - F(z) is H_CLIP
    (cached per kernel and noise scale)."""
    key = (p.kernel == "nbf", p.sigma_V, p.a, p.th)
    if key not in _PSI_MAX:
        F, f, _ = noise_dist(p)
        sig_n = p.sigma_V * np.sqrt(1 - p.a ** 2)
        if p.kernel == "nbf":
            from scipy.optimize import brentq
            zc = brentq(lambda z: 1 - F(z) - H_CLIP, 0.0, 2 * np.sqrt(3) - 1e-9)
        else:
            from scipy.special import ndtri
            zc = float(ndtri(1 - H_CLIP))
        _PSI_MAX[key] = float(f(zc) / (sig_n * H_CLIP * (1 - H_CLIP)))
    return _PSI_MAX[key]


def post_scores(S, M, GN, p, kind):
    """Per-step score of every post's spike or silence with respect to its input current (the derivative of log P of
    the step given the past, exact for the kernel: the likelihood factorises per step; reset and refractoriness are
    deterministic given the spike history), from simulate_ext(score=True)'s raster S, pre-noise membrane M (NaN while
    refractory) and noise draw GN (all L x H x N). With z = (th - m) / sig_n, the hazard h = 1 - F(z) = P(spike | past):
      kind 'hazard': Y = (S - h) psi, psi = f(z) / (sig_n h (1 - h)), the derivative of log P(S) under the Bernoulli
        of the step (Pfister et al. 2006's per-neuron score for this noise model); psi is clipped at psi_max(p) (its
        value at h = H_CLIP) on both branches, so E[Y | past] stays 0;
      kind 'noise': the observed-membrane score, exact in the finite differences of E[R] (self-check (h)): a spike step
        f(z) / (sig_n h), an active silent step -f'(g) / (sig_n f(g)) (= g / sig_n for Gaussian noise), no clip.
    Returns (Y, h), both 0 on refractory steps. Y is a per-post scalar (the post's own membrane and spike), so
    Y_i P_j is a product of two local factors."""
    fused = _ih4_fused if p.kernel == "nbf" else _gauss_fused
    sig_n = p.sigma_V * np.sqrt(1 - p.a ** 2)
    act = ~np.isnan(M)
    z = np.where(act, (p.th - M) / sig_n, 0.0)
    Fz, fz, _ = fused(z)
    h = np.where(act, np.clip(1 - Fz, 0.0, 1.0), 0.0)  # the IH4 polynomial rounds to 1 +- eps at its edges
    Sf = S.astype(float)
    if kind == "hazard":
        ok = act & (h > 0) & (h < 1)
        psi = np.minimum(np.where(ok, fz / (sig_n * np.where(ok, h * (1 - h), 1.0)), 0.0), psi_max(p))
        Y = (Sf - h) * psi
    elif kind == "noise":
        spk = S & act
        _, fg, dfg = fused(GN, need_df=True)
        sil = act & ~S & (fg > 0)
        Y = np.where(spk, fz / (sig_n * np.where(h > 0, h, 1.0)), 0.0)
        Y = np.where(sil, -dfg / (sig_n * np.where(fg > 0, fg, 1.0)), Y)
    else:
        raise ValueError(kind)
    return Y, h


def markov_rate(mu, sig_x, p, nbins=400, lo=-1.5):
    """Stationary spike probability per step of the discrete-time kernel (_step, delta_V = 0) at free-membrane mean mu
    with per-step input noise of s.d. sig_x on top of the background (V' = a V + (1 - a) mu + background noise +
    N(0, sig_x^2); spike if V' >= th, then vr and p.nref clamped steps), from the transition matrix of V on nbins bins
    over [lo, th] (below lo lumped into the lowest bin) plus p.nref refractory states. Exact for the kernel up to the
    binning; the background noise is Irwin-Hall(4) for p.kernel 'nbf', else Gaussian. Siegert's diffusion limit misses
    the drift-dominated regime of the 1 ms grid: at mu = 5 and 8 the kernel's gain is 41 and 31 Hz / theta (no
    refractory period), Siegert (dt-corrected) gives 48 at both."""
    assert p.delta_V == 0
    sb, nr = p.sigma_V * np.sqrt(1 - p.a ** 2), p.nref
    edges = np.linspace(lo, p.th, nbins + 1)
    dz = (p.th - lo) / nbins / 4
    zmax = 6 * np.hypot(sb, sig_x)
    zc = np.arange(-zmax - dz / 2, zmax + dz, dz)  # knots of the increment pmf
    cdf = _ih4_cdf if p.kernel == "nbf" else ndtr
    pz = np.diff(cdf(zc / sb))
    if sig_x > 1e-9:
        pz = np.convolve(pz, np.diff(ndtr(zc / sig_x)), "same")
    Fz = np.r_[0.0, np.cumsum(pz / pz.sum())]
    m = p.a * (edges[:-1] + edges[1:]) / 2 + (1 - p.a) * mu
    C = np.interp(edges[None, :] - m[:, None], zc, Fz, left=0.0, right=1.0)
    n = nbins + nr
    P = np.zeros((n, n))
    P[:nbins, :nbins] = np.diff(C, axis=1)
    P[:nbins, 0] += C[:, 0]
    spk = 1 - C[:, -1]
    r0 = np.searchsorted(edges, p.vr) - 1  # the reset bin: V = vr at the start of the next active step
    P[:nbins, nbins if nr else r0] += spk  # a spike enters the refractory chain (clamped at vr), or resets
    for k in range(nr):
        P[nbins + k, nbins + k + 1 if k + 1 < nr else r0] = 1.0
    A = P.T - np.eye(n)
    A[-1] = 1
    return float(np.linalg.solve(A, np.r_[np.zeros(n - 1), 1.0])[:nbins] @ spk)


_MU_GRID = np.round(np.r_[np.arange(-0.5, 4, 0.05), np.arange(4, 25.01, 0.2)], 10)  # free-membrane mean (theta)
_SX_GRID = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.09, 0.13])  # per-step input s.d. (theta)
_TABLES = {}


def gain_table(p):
    """(rate Hz, gain Hz / theta) of markov_rate on the grid _MU_GRID x _SX_GRID (sig_x per step), the gain by central
    differences of the rate along mu; computed once per parameter set (~15 s)."""
    key = (p.tau_m, p.dt, p.th, p.vr, p.mu_bg, p.sigma_V, p.nref, p.kernel == "nbf")
    if key not in _TABLES:
        r = np.array([[markov_rate(mu, sx, p) for sx in _SX_GRID] for mu in _MU_GRID]) / p.dt
        _TABLES[key] = (r, np.gradient(r, _MU_GRID, axis=0))
    return _TABLES[key]


def gain_markov(mu, sig_x, p):
    """Gain dr / dmu (Hz per theta) of the discrete-time kernel at (mu, sig_x) (markov_rate; bilinear in gain_table,
    clipped to the grid): what the nlif counter uses (iteration 015). An abstraction of what a neuron knows about its
    own operating point, as the Siegert gain."""
    G = gain_table(p)[1]
    mu = np.clip(np.asarray(mu, float), _MU_GRID[0], _MU_GRID[-1])
    i = np.clip(np.searchsorted(_MU_GRID, mu) - 1, 0, len(_MU_GRID) - 2)
    fx = (mu - _MU_GRID[i]) / (_MU_GRID[i + 1] - _MU_GRID[i])
    s = np.clip(np.asarray(sig_x, float), 0, _SX_GRID[-1])
    j = np.clip(np.searchsorted(_SX_GRID, s) - 1, 0, len(_SX_GRID) - 2)
    fs = (s - _SX_GRID[j]) / (_SX_GRID[j + 1] - _SX_GRID[j])
    return ((1 - fx) * ((1 - fs) * G[i, j] + fs * G[i, j + 1]) + fx * ((1 - fs) * G[i + 1, j] + fs * G[i + 1, j + 1]))


def _regression_gain(st, mu, r, Gs, p):
    """EMA moments of (mu_i, r_i) across frames; slope cov / var, Siegert until reg_min frames or degenerate."""
    g = 1.0 / max(p.reg_tau, 1.0)
    if st["reg"] is None:
        st["reg"] = dict(m=mu.copy(), r=r.copy(), mm=mu * mu, mr=mu * r, n=0)
    R = st["reg"]
    for key, val in (("m", mu), ("r", r), ("mm", mu * mu), ("mr", mu * r)):
        R[key] += g * (val - R[key])
    R["n"] += 1
    var, cov = R["mm"] - R["m"] ** 2, R["mr"] - R["m"] * R["r"]
    ok = var > 1e-6
    G = np.where(ok, cov / np.where(ok, var, 1.0), Gs)
    return np.maximum(G, 0.0) if R["n"] >= p.reg_min else Gs


def run_lif(W, v, H, rng, state, p, return_stats=True):
    """One frame of H heads on the current W. Returns the p.k x (H N) bool raster, and with return_stats a dict:
    kappa (N x N raw pair counts, straddling pairs from the carried tail included), n (spikes per neuron, summed over
    heads), c (N x N counter of p.counter), G (gain used), G_siegert, mu, sig (operating point), b (kicks).
    state from init_state (None = a one-off cold frame); it is mutated."""
    N, k, Ws = len(W), p.k, p.W_steps
    assert p.counter in ("siegert", "psi", "regression"), p.counter
    assert (p.counter == "psi") == (p.delta_V > 0) or p.counter == "regression", "psi <=> escape noise (delta_V > 0)"
    if state is None:
        state = init_state(N, H, p)
    b = kicks(W, p)
    Bt = b.T.copy()
    if state["V"] is None or not p.warm:
        _warmup(Bt, v, state, rng, p)
    S, xsum, xsq, Psi, _ = _simulate(Bt, v, state, k, rng, p)
    state["f"] += 1
    if not return_stats:
        state["tail"] = S[k - Ws:].copy() if k >= Ws else np.concatenate([state["tail"][k:], S])
        return S.reshape(k, H * N)
    full = np.concatenate([state["tail"], S])
    state["tail"] = full[-Ws:].copy()
    kappa = pair_counts(full, Ws, Ws)
    n = S.sum((0, 1)).astype(float)
    mu, sig = input_stats(xsum.sum(0), xsq.sum(0), k * H, p)
    Gs = gain(mu, sig, p)
    G = Gs
    if p.counter == "siegert" and p.gain_tau > 1:
        state["G"] = Gs.copy() if state["G"] is None else state["G"] + (Gs - state["G"]) / p.gain_tau
        G = state["G"]
    if p.counter == "regression":
        G = _regression_gain(state, mu, n / (H * k * p.dt), Gs, p)
    if p.counter == "psi":
        X = presyn_trace(full, Ws, p.a, Ws)
        c = b * (Psi.reshape(-1, N).T @ X.reshape(-1, N))
    else:
        c = p.omega * p.tau_eff * G[:, None] * b * n[None, :]
    stats = dict(kappa=kappa, n=n, c=c, G=G, G_siegert=Gs, mu=mu, sig=sig, b=b)
    return S.reshape(k, H * N), stats


def counter_frame(W, v, H, rng, state, p):
    """The agent's engine contract (Agent.counter_frame, rule 'emergent'): (S, kappa_raw, c), raw, summed over heads."""
    S, st = run_lif(W, v, H, rng, state, p)
    return S, st["kappa"], st["c"]


# ---------------------------------------------------------------- stability and calibration
def stability(c, n, m, K):
    """Module-level signed offspring Lambda^K[b, a] = sum_{i in b, j in a} c_ij / sum_{j in a} n_j (derivation §7,
    monitoring; excitatory only, so K x K) and its largest real eigenvalue (Eq. 12, continuous form: stable < 1)."""
    na = np.bincount(m, n, K)
    Ck = np.zeros((K, K))
    np.add.at(Ck, (m[:, None], m[None, :]), c)
    L = Ck / np.maximum(na, 1e-12)[None, :]
    return L, float(np.linalg.eigvals(L).real.max())


def mean_field(W, v, p, iters=400):
    """Diffusion mean field of the network at stimulus v: r_i = F(mu_i(r), sig_i(r)) (Siegert, discrete-time corrected)
    with mu_i = mu_bg + tau_eff (sum_j b_ij r_j + a_ext ext_rate v_i) and sig_i^2 = sigma_V^2 + dt / (1 - a^2)
    (sum_j b_ij^2 r_j + a_ext^2 ext_rate v_i) (shot noise), by damped iteration from 5 Hz (the low state). Returns
    dict(r, G, resid, eig1 = max Re eig of the first-order offspring tau_eff G_i b_ij (what the counters measure),
    eig = the same with the variance channel dF/dsig^2 dt / (1 - a^2) b_ij^2 added: the actual linear stability)."""
    b, lam = kicks(W, p), p.ext_rate * np.asarray(v, float)
    c2 = p.dt / (1 - p.a ** 2)

    def F(mu, var):
        return siegert_rate(mu, np.sqrt(2 * var), p.tau_m, p.th, p.vr, p.tau_ref, p.dt)

    def ms(r):
        return p.mu_bg + p.tau_eff * (b @ r + p.a_ext * lam), p.sigma_V ** 2 + c2 * ((b * b) @ r + p.a_ext ** 2 * lam)

    r = np.full(len(W), 5.0)
    for _ in range(iters):
        r = 0.7 * r + 0.3 * F(*ms(r))
    mu, var = ms(r)
    G, h = gain(mu, np.sqrt(var), p), 1e-6
    Gv = (F(mu, var + h) - F(mu, var - h)) / (2 * h)
    J1 = p.tau_eff * G[:, None] * b
    return dict(r=r, G=G, resid=float(np.abs(F(mu, var) - r).max()), eig1=float(np.linalg.eigvals(J1).real.max()),
                eig=float(np.linalg.eigvals(J1 + c2 * Gv[:, None] * b * b).real.max()))


def calibrate(p, r0=5.0, offspring=0.5, q=1 / 12):
    """(mu_bg, chi) putting a homogeneous network (every neuron with in-weight sum_j T_ij = 1 and sum_j T_ij^2 = q)
    at rate r0 with first-order offspring chi G = offspring. The defaults are calibrate(LIFParams(sigma_V=0.2)):
    offspring 0.8 is near-critical here because rates also raise the shot-noise variance (the convexity of the
    transfer; mean_field's eig ~ 0.99) and strong stimuli then ignite a high-rate state; 0.5 keeps eig <= ~0.7 on the
    v4 networks up to a peak stimulus share of 0.75."""
    from scipy.optimize import brentq
    chi = p.chi
    for _ in range(50):
        pp = dataclasses.replace(p, chi=chi)
        var = p.sigma_V ** 2 + pp.w0 ** 2 * q * r0 * p.dt / (1 - p.a ** 2)
        sB = np.sqrt(2 * var)
        mu = brentq(lambda x, sB=sB: float(siegert_rate(x, sB, p.tau_m, p.th, p.vr, p.tau_ref, p.dt)) - r0, -2, 3)
        chi = offspring / float(siegert_gain(mu, sB, p.tau_m, p.th, p.vr, p.tau_ref, p.dt))
    return mu - chi * r0, chi


# ---------------------------------------------------------------- self-check
def _single_neuron(mu, sig, p, H=4000, T=4.0, seed=0, dmu=0.0):
    """Spike rate of H independent noisy LIF neurons (no recurrence) at constant mean drive mu (+ dmu), CRN in seed."""
    pp = dataclasses.replace(p, mu_bg=mu + dmu, sigma_V=sig, delta_V=0.0)
    rng = np.random.default_rng(seed)
    st = init_state(1, H, pp)
    st["V"] = np.minimum(mu + sig * rng.standard_normal((H, 1)), pp.th - 1e-3)
    Bt, cnt = np.zeros((1, 1)), 0.0
    for _ in range(int(round(0.1 / pp.dt))):  # burn-in
        _step(st["V"], st["ref"], sig * np.sqrt(1 - pp.a ** 2) * rng.standard_normal((H, 1)), None, pp)
    nblk = int(round(T / (pp.k * pp.dt)))
    for _ in range(nblk):
        S = _simulate(Bt, np.zeros(1), st, pp.k, rng, pp)[0]
        cnt += S.sum()
    return cnt / (H * nblk * pp.k * pp.dt)


def _replay_truth(rec, S, b, p, eps=0.25, chunk=30000):
    """Ground truths of the counter by open-loop replay with common random numbers: for every spike of j at step t
    (with a full window) and every target i of j, replay neuron i alone from its recorded state at t over (t, t + W]
    with the recorded drive, its kick b_ij at t + 1 scaled by f in (1, 0, 1 + eps, 1 - eps), and the same escape
    uniforms. removal = n(1) - n(0) is the walk's finite direct effect D_ij; marginal = (n(1 + eps) - n(1 - eps)) /
    (2 eps) is the weight-scaled derivative W_ij dE_ij / dW_ij that Lemma E and Theorem E use (they differ when the
    transfer is curved on the scale of one kick). Returns (removal, marginal) N x N sums over events, n_used per
    presynaptic neuron, and the max mismatch of the f = 1 replay against the recorded raster (must be 0)."""
    k, H, N = S.shape
    Ws = p.W_steps
    t_, h_, j_ = np.nonzero(S[:k - Ws])
    tgt = [np.flatnonzero(b[:, j]) for j in range(N)]
    deg = np.array([len(x) for x in tgt])[j_]
    ev_t, ev_h, ev_j = np.repeat(t_, deg), np.repeat(h_, deg), np.repeat(j_, deg)
    ev_i = np.concatenate([tgt[j] for j in j_])
    fac = np.array([1.0, 0.0, 1 + eps, 1 - eps])[:, None]
    rem, mar, mism = np.zeros((N, N)), np.zeros((N, N)), 0
    for s0 in range(0, len(ev_t), chunk):
        sl = slice(s0, s0 + chunk)
        t, h, j, i = ev_t[sl], ev_h[sl], ev_j[sl], ev_i[sl]
        V = np.repeat(rec["V"][t, h, i][None], 4, 0)
        ref = np.repeat(rec["ref"][t, h, i][None], 4, 0)
        cnt = np.zeros((4, len(t)))
        for s in range(1, Ws + 1):
            dr = rec["drive"][t + s, h, i][None] + ((fac - 1) * b[i, j] if s == 1 else 0.0)
            u = None if rec["U"] is None else np.repeat(rec["U"][t + s, h, i][None], 4, 0)
            cnt += _step(V, ref, dr, u, p)[0]
        act = S[t[None] + np.arange(1, Ws + 1)[:, None], h, i].sum(0)
        mism = max(mism, int(np.abs(cnt[0] - act).max()) if len(t) else 0)
        np.add.at(rem, (i, j), cnt[0] - cnt[1])
        np.add.at(mar, (i, j), (cnt[2] - cnt[3]) / (2 * eps))
    return rem, mar, np.bincount(j_, minlength=N).astype(float), mism


def _frozen_check(p, seed=3, H=16, k=12000, stim=0.3, G=None):
    """One long warm frame (k steps, H heads) on a frozen 3-module x 10 network (C, L, A; p_cc = 0.1), recorded; the
    counter of p.counter over the presynaptic spikes with a full window against the marginal and removal truths
    (_replay_truth). Returns dict(corr, ratio (c / marginal), corr_rem, ratio_rem (c / removal), rate, max_re)."""
    from mapstdp import core
    W, m, roles, _ = core.build_network((10, 10, 10), "CLA", p_cc=0.1, seed=seed)
    v, N, Ws = core.stimulus(stim, m, roles), len(W), p.W_steps
    rng = np.random.default_rng(seed)
    st = init_state(N, H, p)
    b = kicks(W, p)
    _warmup(b.T.copy(), v, st, rng, p)
    S, xsum, xsq, Psi, rec = _simulate(b.T.copy(), v, st, k, rng, p, record=True)
    rem, mar, n_used, mism = _replay_truth(rec, S, b, p)
    assert mism == 0, f"open-loop replay does not reproduce the raster ({mism})"
    if p.counter == "psi":  # presynaptic spikes with a full window only (as the truths)
        Su = S.copy()
        Su[k - Ws:] = False
        X = presyn_trace(np.concatenate([np.zeros((Ws, H, N), bool), Su]), Ws, p.a, Ws)
        c = b * (Psi.reshape(-1, N).T @ X.reshape(-1, N))
    else:
        mu, sig = input_stats(xsum.sum(0), xsq.sum(0), k * H, p)
        G = gain(mu, sig, p) if G is None else G
        c = p.omega * p.tau_eff * G[:, None] * b * n_used[None, :]
    sel = b > 0
    n = S.sum((0, 1)).astype(float)
    return dict(corr=float(np.corrcoef(c[sel], mar[sel])[0, 1]), ratio=float(c[sel].sum() / mar[sel].sum()),
                corr_rem=float(np.corrcoef(c[sel], rem[sel])[0, 1]), ratio_rem=float(c[sel].sum() / rem[sel].sum()),
                rate=float(S.mean() / p.dt), max_re=stability(c * (n / np.maximum(n_used, 1))[None, :], n, m, 3)[1],
                max_re_true=stability(mar * (n / np.maximum(n_used, 1))[None, :], n, m, 3)[1],
                n_syn=int(sel.sum()), pairs=float(n_used.sum()))


def _regression_check(p, frames=300, H=16, seed=3):
    """Regression gain on the frozen 3 x 10 network: frames with a random stimulus position each (warm state), then
    the counter with the regressed G against the truths of _frozen_check (stimulus 0.3). Returns (_frozen_check dict,
    corr and mean ratio of G_reg against the Siegert gain at that stimulus)."""
    from mapstdp import core
    W, m, roles, _ = core.build_network((10, 10, 10), "CLA", p_cc=0.1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    st = init_state(len(W), H, p)
    for _ in range(frames):
        _, sts = run_lif(W, core.stimulus(rng.random(), m, roles), H, rng, st, p)
    Gr = sts["G"]
    r = _frozen_check(dataclasses.replace(p, counter="siegert"), seed=seed, G=Gr, k=6000)
    _, sts = run_lif(W, core.stimulus(0.3, m, roles), H, rng, st, p)
    Gs = sts["G_siegert"]
    return r, float(np.corrcoef(Gr, Gs)[0, 1]), float(Gr.mean() / Gs.mean())


def _check_pairs(p, seed=0):
    """kappa against a brute-force loop over the definition, across a frame boundary: two warm frames' kappa (the
    second with the straddling pairs) sum to the concatenated raster's kappa."""
    rng = np.random.default_rng(seed)
    N, H, Ws = 5, 2, 7
    A = rng.random((40, H, N)) < 0.15
    brute = np.zeros((N, N))
    for h in range(H):
        for u in range(40):
            for j in np.flatnonzero(A[u, h]):
                brute[:, j] += A[u + 1:u + 1 + Ws, h].sum(0)
    tail0 = np.zeros((Ws, H, N), bool)
    k1 = pair_counts(np.concatenate([tail0, A[:25]]), Ws, Ws)
    k2 = pair_counts(np.concatenate([A[25 - Ws:25], A[25:]]), Ws, Ws)
    assert np.array_equal(k1 + k2, brute) and (brute >= 0).all()
    # psi trace: xbar(t) = sum_{s<W} a^s x(t-1-s)
    X = presyn_trace(np.concatenate([tail0, A]), Ws, 0.9, Ws)
    t = 30
    ref = sum(0.9 ** s * A[t - 1 - s] for s in range(Ws))
    assert np.allclose(X[t], ref)


def _score_check(p=None, H=200000, T=20, seed=7, verbose=True):
    """Self-check (h) of post_scores (models.tdstdp --credit): (i) on a frozen random-input run of both numba
    kernels, each neuron's summed hazard equals its spike count within the martingale's s.d. (sum h (1 - h)); (ii) on a
    one-neuron toy (H trajectories of T steps, Bernoulli input spikes through one weight w, the free-membrane mean 1.3
    at the DT1R point), the 'noise' score's REINFORCE estimate of d E[R] / d w for R = the T-step spike count,
    sum_t s_t x(t-1) (R - mean R), matches the central finite difference of E[R] with common random numbers within
    5%; (iii) the 'hazard' score's sum over the toy has zero mean within 4 s.e. (the clip keeps it). Returns the
    ratios."""
    p = dataclasses.replace(LIFParams(), **DT1R) if p is None else p
    out = {}
    # (i) sum of h vs the spike count per neuron, both kernels
    for kern in ("nbf", "nb"):
        pk = dataclasses.replace(p, kernel=kern)
        rng = np.random.default_rng(seed)
        N, Hh, k = 12, 8, 2000
        st = init_state(N, Hh, pk)
        st["V"] = np.full((Hh, N), pk.mu_bg)
        Ix = (1 - pk.a) * rng.uniform(0.3, 1.2, N)[None, None, :] + rng.normal(0, 0.02, (k, Hh, N))
        S, _, _, M, GN = simulate_ext(Ix, np.zeros((0, 0)), [], st, rng, pk, score=True)
        Y, h = post_scores(S, M, GN, pk, "hazard")
        n, hs, sd = S.sum((0, 1)), h.sum((0, 1)), np.sqrt((h * (1 - h)).sum((0, 1)))
        dev = np.abs(n - hs) / sd
        assert (dev < 4).all() and (n > 50).all(), (kern, n, hs, sd)
        assert np.isnan(M[S.astype(bool)]).sum() == 0 and (Y[np.isnan(M)] == 0).all() and (h[np.isnan(M)] == 0).all()
        out[f"h_{kern}"] = float(dev.max())
        if verbose:
            print(f"(h) kernel {kern}: sum h vs spike count per neuron, max |dev| {dev.max():.2f} s.d. "
                  f"(rates {n.min() / (k * Hh * pk.dt):.0f}-{n.max() / (k * Hh * pk.dt):.0f} Hz)")
    # (ii, iii) the one-neuron toy: three common-random-number runs (w - dw, w, w + dw)
    rng = np.random.default_rng(seed + 1)
    frate, mu_free = 0.2, 1.3
    x = (rng.random(T) < frate).astype(float)
    w = (mu_free - p.mu_bg) * (1 - p.a) / frate
    V0 = np.minimum(p.mu_bg + p.sigma_V * rng.standard_normal((H, 1)), p.th - 1e-3)
    xin = np.r_[0.0, x[:-1]]  # x(t - 1)

    def run(ww):
        st = init_state(1, H, p)
        st["V"] = V0.copy()
        Ix = np.broadcast_to((ww * xin)[:, None, None], (T, H, 1))
        return simulate_ext(Ix, np.zeros((0, 0)), [], st, np.random.default_rng(seed + 2), p, score=True)
    dw = 0.02 * w
    Rp, Rm = run(w + dw)[0].sum((0, 2)), run(w - dw)[0].sum((0, 2))
    S, _, _, M, GN = run(w)
    R = S.sum((0, 2)).astype(float)
    fd, fd_se = (Rp.mean() - Rm.mean()) / (2 * dw), np.std(Rp - Rm) / (2 * dw) / np.sqrt(H)
    Yn, _ = post_scores(S, M, GN, p, "noise")
    est = (Yn[:, :, 0] * xin[:, None]).sum(0) * (R - R.mean())
    Yh, h = post_scores(S, M, GN, p, "hazard")
    ysum = Yh[:, :, 0].sum(0)
    out["fd"], out["est"], out["est_se"] = float(fd), float(est.mean()), float(est.std() / np.sqrt(H))
    out["EY"], out["EY_se"] = float(ysum.mean()), float(ysum.std() / np.sqrt(H))
    out["psi_max"], out["clip_frac"] = psi_max(p), float((h[S] < H_CLIP).mean())
    if verbose:
        print(f"    toy (mu 1.3, {H} trajectories x {T} ms, mean count {R.mean():.2f}): d E[count] / dw by finite "
              f"differences {fd:.3f} +- {fd_se:.3f}, by the noise score {est.mean():.3f} +- {out['est_se']:.3f} "
              f"(ratio {est.mean() / fd:.3f}); hazard score: E[sum_t Y] {ysum.mean():.4f} +- {out['EY_se']:.4f}, "
              f"psi_max {out['psi_max']:.1f}, spikes at h < {H_CLIP:g}: {out['clip_frac']:.4f}")
    assert abs(est.mean() / fd - 1) < 0.05, (est.mean(), fd)
    assert abs(ysum.mean()) < 4 * out["EY_se"], (ysum.mean(), out["EY_se"])
    return out


def _main():
    t0 = time.time()
    p = LIFParams()
    out = {}
    # (a, b) Siegert rate and gain against a single simulated neuron (no recurrence), dt = 0.5 ms
    print("(a, b) single neuron: Siegert (dt-corrected) vs simulation, sigma_V = 0.2")
    for mu in (0.5, 0.7, 0.9, 1.2):
        r_sim = _single_neuron(mu, 0.2, p, seed=1)
        r_th = float(siegert_rate(mu, np.sqrt(2) * 0.2, p.tau_m, p.th, p.vr, p.tau_ref, p.dt))
        r_c = float(siegert_rate(mu, np.sqrt(2) * 0.2, p.tau_m, p.th, p.vr, p.tau_ref))
        dm = 0.04
        g_sim = (_single_neuron(mu, 0.2, p, seed=2, dmu=dm) - _single_neuron(mu, 0.2, p, seed=2, dmu=-dm)) / (2 * dm)
        g_th = float(gain(mu, 0.2, p))
        print(f"  mu={mu:.1f}: rate sim {r_sim:6.2f} Hz, Siegert {r_th:6.2f} (ratio {r_th / r_sim:.3f}; "
              f"uncorrected {r_c / r_sim:.3f}); gain sim {g_sim:6.1f}, Siegert {g_th:6.1f} (ratio {g_th / g_sim:.3f})")
        out[f"a_mu{mu}"] = (r_th / r_sim, g_th / g_sim)
        if r_sim > 2:
            assert abs(r_th / r_sim - 1) < 0.05, (mu, r_sim, r_th)
            assert abs(g_th / g_sim - 1) < 0.10, (mu, g_sim, g_th)
    # (c) counters vs ground truth on a frozen recurrent network (open-loop replay with common random numbers)
    print("(c) counter vs truth (frozen 3 x 10 network, H = 16, 6 s per head); truth = marginal b dE/db (what the rule"
          " needs) and removal (the walk's finite D)")
    psi_p = dataclasses.replace(p, counter="psi", delta_V=0.2, sigma_V=0.1, mu_bg=0.35, rho0=100.0, reset=False,
                                chi=0.003, tau_ref=0.0, a_ext=0.05)
    rs = {}
    for name, pp in (("siegert", p), ("siegert omega=0.95", dataclasses.replace(p, omega=0.95)),
                     ("psi, reset-free escape", psi_p),
                     ("psi, escape LIF w/ reset", dataclasses.replace(p, counter="psi", delta_V=0.2, sigma_V=0.1,
                                                                     mu_bg=0.6, rho0=100.0))):
        r = rs[name] = _frozen_check(pp, k=6000)
        print(f"  {name:25s} corr {r['corr']:.3f} ratio c / marginal {r['ratio']:.3f} | corr {r['corr_rem']:.3f} "
              f"ratio c / removal {r['ratio_rem']:.3f} | rate {r['rate']:.1f} Hz, max Re eig Lambda^K from c "
              f"{r['max_re']:.2f}, from truth {r['max_re_true']:.2f} ({r['n_syn']} synapses, {r['pairs']:.0f} spikes)")
    assert rs["siegert"]["corr"] >= 0.9 and 0.85 < rs["siegert"]["ratio"] < 1.2, rs["siegert"]
    assert rs["psi, reset-free escape"]["corr"] >= 0.85 and 0.8 < rs["psi, reset-free escape"]["ratio"] < 1.25
    r, cg, rg = _regression_check(dataclasses.replace(p, counter="regression"))
    print(f"  {'regression (300 frames)':25s} corr {r['corr']:.3f} ratio c / marginal {r['ratio']:.3f} | corr "
          f"{r['corr_rem']:.3f} ratio c / removal {r['ratio_rem']:.3f} | G_reg vs Siegert: corr {cg:.2f}, "
          f"mean ratio {rg:.2f}")
    out["c"] = rs
    # (d) pair counts: definition, window, straddling
    _check_pairs(p)
    from mapstdp import core
    W, m, roles, _ = core.build_network((10, 10, 10), "CLA", p_cc=0.1, seed=3)
    v, rng, H, Ws = core.stimulus(0.3, m, roles), np.random.default_rng(9), 4, p.W_steps
    st = init_state(len(W), H, p)
    run_lif(W, v, H, rng, st, p)
    tail, Ss, ks, ns = st["tail"].copy(), [], 0.0, 0.0
    for _ in range(3):  # the engine's carried tail: per-frame kappas sum to the concatenated raster's
        S, sts = run_lif(W, v, H, rng, st, p)
        Ss.append(S.reshape(p.k, H, -1))
        ks, ns = ks + sts["kappa"], ns + sts["n"]
    full = np.concatenate([tail] + Ss)
    assert np.array_equal(ks, pair_counts(full, Ws, Ws)) and (ks >= 0).all()
    E = ks / np.maximum(ns, 1)[None, :]  # E_ij estimate: spikes of i per window after a spike of j
    a_chance = ns / (3 * p.k * H) * Ws  # r_i w
    print(f"(d) pair counts == brute-force definition across a frame boundary and over 3 warm engine frames; psi trace "
          f"== definition; E_ij = kappa / n_j >= 0, mean E_ij / (r_i w) = {np.mean(E / a_chance[:, None]):.3f} "
          f"(1 = chance only)")
    # (e) operating point and stability on the v4 networks (p_cc = 0, flow-v4 sizes); ms/frame at H = 16
    from models.basic import TASKS
    print("(e) v4 networks, H = 16, 30 warm frames, stimulus: a ring bump over the controller (peak share ~0.28), "
          "then 5 frames at peak share 0.75")
    for task in ("CartPole", "LunarLander"):
        T = TASKS[task]
        W, m, roles, _ = core.build_network(T.sizes, T.roles, p_cc=0.0, seed=0)
        N, K, H = len(W), len(roles), 16
        rng = np.random.default_rng(0)
        st = init_state(N, H, p)
        ctrl = np.flatnonzero(roles[m] == "C")
        rates, mres, tms = [], [], []
        for f in range(35):
            v = np.zeros(N)
            if f < 30:
                v[ctrl] = np.exp(-((np.arange(len(ctrl)) - rng.integers(len(ctrl))) % len(ctrl)) ** 2 / 4.0)
            else:
                v[ctrl[:4]] = (0.75, 0.25 / 3, 0.25 / 3, 0.25 / 3)
            v /= v.sum()
            t1 = time.perf_counter()
            S, sts = run_lif(W, v, H, rng, st, p)
            tms.append(time.perf_counter() - t1)
            rates.append(np.bincount(m, sts["n"], K) / np.bincount(m, minlength=K) / (H * p.k * p.dt))
            mres.append(stability(sts["c"], sts["n"], m, K)[1])
        rmax = sts["n"].max() / (H * p.k * p.dt)
        R, Rp = np.mean(rates[1:30], 0), np.mean(rates[30:], 0)
        mf = mean_field(W, v, p)
        ms = 1e3 * np.median(tms[1:])
        tps = []
        pp, stp = dataclasses.replace(p, counter="psi", delta_V=0.2, sigma_V=0.1, mu_bg=0.6), init_state(N, H, p)
        for _ in range(6):
            t1 = time.perf_counter()
            run_lif(W, v, H, rng, stp, pp)
            tps.append(time.perf_counter() - t1)
        print(f"  {task} (N = {N}): module rates {np.round(R, 1)} Hz (peak 0.75: {np.round(Rp, 1)}; max neuron "
              f"{rmax:.0f} Hz); max Re eig Lambda^K from c: mean {np.mean(mres):.2f}, max {np.max(mres):.2f}; "
              f"mean field at peak 0.75: eig1 {mf['eig1']:.2f}, eig with variance channel {mf['eig']:.2f}; "
              f"{ms:.0f} ms/frame (siegert), {1e3 * np.median(tps[1:]):.0f} ms/frame (psi)")
        assert (R > 2).all() and (R < 50).all() and max(mres) < 1 and mf["eig"] < 1, (R, mres, mf["eig"])
        out[task] = ms
    # (f) dt = 1 ms (DT1): calibration, Siegert against simulation, the external-current kernel at zero input
    p1 = dataclasses.replace(p, **DT1)
    mu1, chi1 = calibrate(p1)
    assert abs(mu1 - DT1["mu_bg"]) < 1e-3 and abs(chi1 - DT1["chi"]) < 1e-4, (mu1, chi1)
    print(f"(f) dt = 1 ms: calibrate -> mu_bg {mu1:.4f}, chi {chi1:.5f} (DT1 {DT1['mu_bg']}, {DT1['chi']})")
    mu1r, chi1r = calibrate(dataclasses.replace(p, **DT1R))
    assert abs(mu1r - DT1R["mu_bg"]) < 1e-3 and abs(chi1r - DT1R["chi"]) < 1e-4, (mu1r, chi1r)
    print(f"  with tau_ref 2 ms: mu_bg {mu1r:.4f}, chi {chi1r:.5f} (DT1R {DT1R['mu_bg']}, {DT1R['chi']})")
    for mu in (0.5, 0.7, 0.9, 1.2):
        r_sim = _single_neuron(mu, 0.2, p1, seed=1)
        r_th = float(siegert_rate(mu, np.sqrt(2) * 0.2, p1.tau_m, p1.th, p1.vr, p1.tau_ref, p1.dt))
        dm = 0.04
        g_sim = (_single_neuron(mu, 0.2, p1, seed=2, dmu=dm) - _single_neuron(mu, 0.2, p1, seed=2, dmu=-dm)) / (2 * dm)
        g_th = float(gain(mu, 0.2, p1))
        print(f"  mu={mu:.1f}: rate sim {r_sim:6.2f} Hz, Siegert {r_th:6.2f} (ratio {r_th / r_sim:.3f}); gain sim "
              f"{g_sim:6.1f}, Siegert {g_th:6.1f} (ratio {g_th / g_sim:.3f})")
        out[f"f_mu{mu}"] = (r_th / r_sim, g_th / g_sim)
        if r_sim > 2:
            assert abs(r_th / r_sim - 1) < 0.05, (mu, r_sim, r_th)
            assert abs(g_th / g_sim - 1) < 0.10, (mu, g_sim, g_th)
    H1, rng1 = 2000, np.random.default_rng(4)
    st1 = init_state(1, H1, p1)
    st1["V"] = np.full((H1, 1), p1.mu_bg)
    simulate_ext(np.zeros((200, H1, 1)), np.zeros((0, 0)), [], st1, rng1, p1)  # burn-in
    r_ext = simulate_ext(np.zeros((2000, H1, 1)), np.zeros((0, 0)), [], st1, rng1, p1)[0].mean() / p1.dt
    r_bg = float(siegert_rate(p1.mu_bg, np.sqrt(2) * 0.2, p1.tau_m, p1.th, p1.vr, p1.tau_ref, p1.dt))
    print(f"  simulate_ext, zero input: {r_ext:.2f} Hz, Siegert at mu_bg {r_bg:.2f} Hz (background without "
          f"recurrence; the recurrent operating point adds chi r0 = {DT1['chi'] * 5:.3f} theta for 5 Hz)")
    assert abs(r_ext / r_bg - 1) < 0.05, (r_ext, r_bg)
    # (g) dt = 1 ms, mean-driven (the nlif posts of models.tdstdp): the kernel's Markov-chain gain (gain_markov)
    # against simulate_ext with common random numbers, both noise modes; Siegert for contrast
    for kern, pg, mus in (("nbf", p1, (0.7, 3.0, 8.0)), ("nb", p1, (0.7, 3.0, 8.0)),
                          ("nbf", dataclasses.replace(p, **DT1R), (1.0, 1.5, 3.0))):
        pk = dataclasses.replace(pg, kernel=kern)
        line = []
        for mu in mus:
            rs = []
            for dm in (0.05, -0.05):
                stg = init_state(1, 2000, pk)
                stg["V"] = np.full((2000, 1), 0.5)
                Ig = np.full((2500, 2000, 1), (1 - pk.a) * (mu + dm - pk.mu_bg))
                rs.append(simulate_ext(Ig, np.zeros((0, 0)), [], stg, np.random.default_rng(5), pk)[0][500:].mean())
            g_sim, g_mk = (rs[0] - rs[1]) / 0.1 / pk.dt, float(gain_markov(mu, 0.0, pk))
            g_sg = float(gain(mu, pk.sigma_V, pk))
            line.append(f"mu={mu:.1f}: sim {g_sim:5.1f}, Markov {g_mk:5.1f}, Siegert {g_sg:5.1f}")
            assert abs(g_mk / g_sim - 1) < 0.06, (kern, mu, g_sim, g_mk)
        print(f"(g) gain at dt = 1 ms, tau_ref {pk.tau_ref * 1e3:.0f} ms, kernel {kern}: " + "; ".join(line))
    # (h) the post scores (post_scores; models.tdstdp --credit hazard | noise) at the DT1R point
    out["score"] = _score_check()
    print(f"lif self-check ok ({time.time() - t0:.0f} s)")
    return out


if __name__ == "__main__":
    sys.exit(0 if _main() is not None else 1)
