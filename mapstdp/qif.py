"""QIF spiking engine for Map-STDP (Proposal, iteration 013): quadratic integrate-and-fire neurons in place of the
branching-mode SuperNeuroMAT neurons, with a module-level mean field (Montbrio, Pazo & Roxin 2015).

Neuron: tau dV/dt = V^2 + eta_i + I_s, integrated in theta form (V = tan(theta / 2), V_peak = inf), so
  tau dtheta/dt = (1 + eta + I_s) + (eta + I_s - 1) cos(theta), a spike when theta crosses pi (theta -= 2 pi).
  RK2 (midpoint) with step dt; eta_i ~ Lorentzian(eta_bar, delta), quenched per neuron and head.
Pulses: a pulse of size w is the exact jump V -> V + w, i.e. theta -> 2 arctan(tan(theta / 2) + w), applied after the
  step's threshold and reset. Teleportation: Bernoulli pulses of amplitude a_ext to E neuron i with probability
  rate v_i per bin, at a uniform substep of the bin.
Synapses: delay `delay`; recurrent spikes either jump V (tau_s = 0) or kick an exponential current I_s by w tau / tau_s
  (same time integral as the jump). Weights [pre, post] = gain W / d_j (as sp.set_weights) on E -> E; optional fixed
  inhibitory pools (`inh`: m = one per module, a = one over the action modules, g = global), E -> I weight w_ei / n_E
  and I -> E weight -w_ie / n_inh per synapse. The I neurons are extra neurons after the N E neurons, never in W or the
  raster.
Frames: k bins of `bin` ms; the raster is binned (k x (H N) bool, E only), so sp.pooled, pi_hat, race, diffbound and
  head_groups apply unchanged. Warm start carries the state (theta, I_s, in-flight spikes) across frames.
Order inside one step s (matches Brian2's schedule groups -> thresholds -> synapses -> resets -> end): integrate
  (theta clamped at -pi, _rk2), threshold/reset (theta wrapped mod 2 pi, at most one spike per step), deliver recurrent
  spikes emitted at s - D and the external pulses of step s, apply the jumps.
Rule 'emergent' (derivation Eqs. 8, 11b): run_qif(inputs=True) also returns each E neuron's measured frame input
  (frame_inputs, no extra draws, same raster); window_pairs gives the raw pair counts of the walk in the pairing window
  (E neurons only: the pools are fixed and outside W, so not in the walk), mean_counter the counter
  c = tau G_i b_ij n_j with G from GainEstimator (regression of rate on the fresh-noise current across frames).
Only numpy is required; numba (if importable) runs the batched kernel, brian2 is imported by brian_frame only.
Self-check: python -m mapstdp.qif [--brian2] (the Brian2 part runs in the map-stdp-brian2 env)."""
import dataclasses
import math
import sys
import time

import numpy as np

try:
    import numba
except ImportError:  # the Brian2 reference env has no numba: the numpy kernel is used
    numba = None


@dataclasses.dataclass(frozen=True)
class QIFParams:
    tau: float = 10.0  # membrane time constant (ms)
    dt: float = 0.1  # integration step (ms)
    eta_bar: float = -2.0  # Lorentzian centre of the excitabilities (rest at -sqrt(2), threshold +sqrt(2))
    delta: float = 0.05  # Lorentzian half-width (0.25: 4% tonic neurons, 2/3 of the spikes spontaneous)
    eta_max: float = 10.0  # Lorentzian truncated to [2 eta_bar - eta_max, eta_max] (rates <= 100 Hz; RK2 resolves them)
    delay: float = 1.0  # synaptic delay (ms)
    tau_s: float = 2.0  # exponential synaptic current (ms); 0 = pulses
    gain: float = 30.0  # recurrent output gain: [pre, post] = gain W / d_j (calibrated: m_eff 0.85, CartPole)
    a_ext: float = 5.0  # teleportation pulse amplitude (V units; forces a spike from rest unless eta < -6.25)
    rate: float = 0.5  # teleportation: P(pulse to E neuron i in one bin) = rate v_i
    bin: float = 1.0  # raster bin (ms)
    k: int = 100  # bins per frame
    burn: int = 20  # race burn-in (bins)
    warm: bool = True  # carry the state across frames (reset at episode start); False = cold start every frame
    lags: int = 5  # pairing window: causal lags 1..lags bins (flat)
    tau_plus: float = 0.0  # > 0: exponential pairing window exp(-(lag - 1) / tau_plus) (bins) instead of the flat one
    inh: str = "m"  # inhibitory pools: any of 'm' (per module), 'a' (action modules), 'g' (global); 'none' = no pool
    n_inh: int = 4  # I neurons per pool (per head)
    w_ei: float = 30.0  # E -> I total weight (per-synapse w_ei / n_E of the pool)
    w_ie: float = 60.0  # I -> E total weight (per-synapse -w_ie / n_inh)
    lam_scale: float = 1.0  # agent: structural rate lambda x this (learning-rate match to branching, Phase A)
    eta_scale: float = 1.0  # agent: task rate eta x this
    eps_h_scale: float = 1.0  # agent: d_j homeostasis rate eps_h x this
    sigma: float = 1.0  # fresh noise: Gaussian current held per bin, sd sigma sqrt(tau / bin) (diffusion sigma^2 / tau)

    @property
    def nsub(self):
        return int(round(self.bin / self.dt))

    @property
    def D(self):
        return int(round(self.delay / self.dt))


# Operating point of the rates backend (Phase B): near the MPR-exact regime. No fresh noise (the Lorentzian width
# supplies the irregularity), slow synaptic currents and many weak IPSPs, so that recurrent input acts as a current
# (where the neuron-level MPR fixed point matches the QIF network), and a unique, Newton-reachable fixed point.
MPR_POINT = dict(sigma=0.0, eta_bar=-2.0, delta=0.5, gain=20.0, w_ie=15.0, rate=1.0, n_inh=32, tau_s=20.0)


def lorentzian(rng, shape, p):
    """eta ~ Lorentzian(eta_bar, delta) truncated symmetrically at eta_max (inverse-CDF draw, one uniform per neuron).
    The untruncated tail is unresolvable at step dt (an eta of 1e3 advances theta by tens of radians per step, so the
    neuron spikes every step) and dominates the spontaneous rate (MPR's r for eta_bar < 0)."""
    lo = np.arctan((p.eta_max - p.eta_bar) / p.delta) / np.pi if p.eta_max > p.eta_bar else 0.5
    return p.eta_bar + p.delta * np.tan(np.pi * lo * (2 * rng.random(shape) - 1))


def pools(m, roles, inh):
    """E-index arrays of the inhibitory pools for the flags in inh (module doc)."""
    roles = np.asarray(roles)
    out = [np.flatnonzero(m == a) for a in range(len(roles))] if "m" in inh else []
    out += [np.flatnonzero(roles[m] == "A")] if "a" in inh else []
    return out + ([np.arange(len(m))] if "g" in inh else [])


def weights(W, p, pls=()):
    """[pre, post] matrix of one head, (N + n_inh len(pls)) square: gain W / d_j transposed on E -> E, then pools."""
    N = len(W)
    M = N + p.n_inh * len(pls)
    A = np.zeros((M, M))
    A[:N, :N] = p.gain * (W / np.maximum(W.sum(0), 1e-12)).T
    for k, g in enumerate(pls):
        idx = N + k * p.n_inh + np.arange(p.n_inh)
        A[np.ix_(g, idx)] += p.w_ei / len(g)
        A[np.ix_(idx, g)] -= p.w_ie / p.n_inh
    return A


def init_state(eta, rng, p):
    """Stationary state of the uncoupled neurons: rest 2 arctan(-sqrt(-eta)) for eta < 0, a uniform oscillation phase
    (V = sqrt(eta) tan(u), u ~ U(-pi/2, pi/2)) for eta >= 0; no synaptic current, no spikes in flight.
    Returns (theta, I_s, buf), buf (D, H, M) bool the spikes in flight by emission substep mod D."""
    u = np.pi * (rng.random(eta.shape) - 0.5)
    se = np.sqrt(np.abs(eta))
    theta = np.where(eta < 0, 2 * np.arctan(-se), 2 * np.arctan(se * np.tan(u)))
    return theta, np.zeros_like(eta), np.zeros((p.D, *eta.shape), bool)


def draw_events(v, H, N, M, rng, p):
    """Inputs of one frame. Teleportation pulses: Bernoulli rate v_i per bin (k x H N draw, as sp.run_heads), each at a
    uniform substep of its bin, as per-head CSR (ptr (H + 1,), substep, neuron) sorted by head then substep. Noise
    current (k x H x M, all neurons) if sigma > 0, else a 1 x H x M zero array."""
    ext = rng.random((p.k, H * N)) < p.rate * np.tile(v, H)
    t, col = np.nonzero(ext)
    s = t * p.nsub + rng.integers(0, p.nsub, len(t))
    h, i = col // N, col % N
    o = np.lexsort((s, h))
    xn = rng.standard_normal((p.k, H, M)) * (p.sigma * np.sqrt(p.tau / p.bin)) if p.sigma > 0 else np.zeros((1, H, M))
    return np.searchsorted(h[o], np.arange(H + 1)), s[o].astype(np.int64), i[o].astype(np.int64), xn


_C = [(-1) ** n / math.factorial(2 * n) for n in range(16)]  # Taylor to x^30: error < 1e-15 on [-pi, pi]
_C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _C10, _C11, _C12, _C13, _C14, _C15 = _C


def _cos(x):
    """cos for the numba kernel: x reduced into [-pi, pi], then the degree-30 Taylor polynomial in Horner form
    (vectorisable, unlike libm's cos; within 1e-15 of math.cos)."""
    x = x - 2 * math.pi * math.floor(x / (2 * math.pi) + 0.5)
    y = x * x
    return _C0 + y * (_C1 + y * (_C2 + y * (_C3 + y * (_C4 + y * (_C5 + y * (_C6 + y * (_C7 + y * (
        _C8 + y * (_C9 + y * (_C10 + y * (_C11 + y * (_C12 + y * (_C13 + y * (_C14 + y * _C15))))))))))))))


def _rk2(th, Ic, et, xn, M, rt, dec, syn):
    """One RK2 (midpoint) step of theta (and of I_s if syn) for one head, in place (the kernel's vectorised loop).
    theta is clamped at -pi: the flow cannot cross -pi (dtheta/dt = 2 / tau there for any input), but a midpoint step
    under strong inhibition (I_s of order -100) overshoots it."""
    if syn:
        for i in range(M):
            c = et[i] + xn[i] + Ic[i]
            f1 = (1.0 + c) + (c - 1.0) * _cos(th[i])
            Im = Ic[i] - 0.5 * dec * Ic[i]
            cm = et[i] + xn[i] + Im
            th[i] = max(th[i] + rt * ((1.0 + cm) + (cm - 1.0) * _cos(th[i] + 0.5 * rt * f1)), -math.pi)
            Ic[i] = Ic[i] - dec * Im
    else:
        for i in range(M):
            c = et[i] + xn[i]
            f1 = (1.0 + c) + (c - 1.0) * _cos(th[i])
            th[i] = max(th[i] + rt * ((1.0 + c) + (c - 1.0) * _cos(th[i] + 0.5 * rt * f1)), -math.pi)


def _kernel_py(theta, Is, buf, eta, A, ptr, ev_s, ev_i, xn, S, nsub, D, dt, tau, tau_s, a_ext):
    """Batched QIF frame in loops, compiled by numba when available (module doc for the step order; _kernel_np is the
    vectorised twin). Mutates theta, Is, buf and S (k x H x N bool, E only)."""
    H, M = theta.shape
    N = S.shape[2]
    n_steps = S.shape[0] * nsub
    syn = tau_s > 0
    kick = tau / tau_s if syn else 0.0
    dec = dt / tau_s if syn else 0.0
    rt = dt / tau
    nx, nb = xn.shape[0], n_steps // xn.shape[0]  # noise bins and substeps per noise bin
    for h in numba_prange(H):
        th, Ic, et = theta[h], Is[h], eta[h]
        dv = np.zeros(M)
        new = np.zeros(M, np.bool_)
        e = ptr[h]
        for s in range(n_steps):
            slot = s % D
            _rk2(th, Ic, et, xn[s // nb if nx > 1 else 0, h], M, rt, dec, syn)
            for i in range(M):  # threshold and reset
                if th[i] > math.pi:  # at most one spike per step
                    th[i] -= 2 * math.pi * math.floor((th[i] + math.pi) / (2 * math.pi))
                    new[i] = True
                    if i < N:
                        S[s // nsub, h, i] = True
            for j in range(M):  # recurrent spikes emitted at s - D
                if buf[slot, h, j]:
                    for i in range(M):
                        w = A[j, i]
                        if w != 0.0:
                            if syn:
                                Ic[i] += w * kick
                            else:
                                dv[i] += w
                buf[slot, h, j] = new[j]
                new[j] = False
            while e < ptr[h + 1] and ev_s[e] == s:  # teleportation pulses of step s
                dv[ev_i[e]] += a_ext
                e += 1
            for i in range(M):  # exact jumps V -> V + dv
                if dv[i] != 0.0:
                    th[i] = 2 * math.atan(math.tan(th[i] / 2) + dv[i])
                    dv[i] = 0.0


def _kernel_np(theta, Is, buf, eta, A, ptr, ev_s, ev_i, xn, S, nsub, D, dt, tau, tau_s, a_ext):
    """Vectorised numpy twin of _kernel_py (the reference in envs without numba); same arithmetic, same order."""
    H, M = theta.shape
    syn = tau_s > 0
    hh = np.repeat(np.arange(H), np.diff(ptr))
    n_steps = S.shape[0] * nsub
    nb = n_steps // xn.shape[0]
    for s in range(n_steps):
        slot = s % D
        x = xn[s // nb]
        c = eta + x + Is
        f1 = (1.0 + c) + (c - 1.0) * np.cos(theta)
        Im = Is - 0.5 * (dt / tau_s) * Is if syn else 0.0
        cm = eta + x + Im
        theta += (dt / tau) * ((1.0 + cm) + (cm - 1.0) * np.cos(theta + 0.5 * (dt / tau) * f1))
        np.maximum(theta, -np.pi, out=theta)
        if syn:
            Is -= (dt / tau_s) * Im
        new = theta > np.pi
        theta[new] -= 2 * np.pi * np.floor((theta[new] + np.pi) / (2 * np.pi))
        S[s // nsub] |= new[:, :S.shape[2]]
        dv = np.zeros((H, M))
        hs, js = np.nonzero(buf[slot])
        if len(hs):
            rec = np.zeros((H, M))
            np.add.at(rec, hs, A[js])
            if syn:
                Is += rec * (tau / tau_s)
            else:
                dv += rec
        buf[slot] = new
        sel = ev_s == s
        np.add.at(dv, (hh[sel], ev_i[sel]), a_ext)
        nz = dv != 0
        theta[nz] = 2 * np.arctan(np.tan(theta[nz] / 2) + dv[nz])


if numba is not None:  # fastmath without nnan / ninf: lets LLVM vectorise the RK2 loop (4x)
    numba_prange = numba.prange
    _FM = {"nsz", "arcp", "contract", "afn", "reassoc"}
    _cos = numba.njit(inline="always")(_cos)
    _rk2 = numba.njit(fastmath=_FM)(_rk2)
    _kernel = numba.njit(parallel=True, cache=True, fastmath=_FM)(_kernel_py)
else:
    numba_prange = range
    _kernel = _kernel_np


def run_qif(W, v, H, rng, eta, state, p, m=None, roles=None, kernel=None, inputs=False):
    """Frame of H heads on the current W (k x H N bool E raster, head h in columns h N .. (h + 1) N - 1, as
    sp.run_heads). eta, state: (H, M) arrays of the agent (M = N + inhibitory neurons; m, roles needed for pools).
    Draws the teleportation pulses from rng; mutates state.
    inputs=True: returns (S, inp) with inp = frame_inputs(...) (the measured input of every E neuron; the same draws and
    the same raster: the kernel records all M neurons and the E columns are returned)."""
    N = len(W)
    A = weights(W, p, pools(m, roles, p.inh) if p.inh else ())
    ev = draw_events(v, H, N, len(A), rng, p)
    if not inputs:
        S = np.zeros((p.k, H, N), bool)
        (kernel or _kernel)(*state, eta, A, *ev, S, p.nsub, p.D, p.dt, p.tau, p.tau_s, p.a_ext)
        return S.reshape(p.k, H * N)
    I0, b0 = state[1].copy(), state[2].sum(0)
    S = np.zeros((p.k, H, len(A)), bool)
    (kernel or _kernel)(*state, eta, A, *ev, S, p.nsub, p.D, p.dt, p.tau, p.tau_s, p.a_ext)
    return S[:, :, :N].reshape(p.k, H * N), frame_inputs(S, A, ev, I0, b0, state, N, p)


def frame_inputs(S, A, ev, I0, b0, state, N, p):
    """Measured input of a frame per head and E neuron, from quantities the neuron receives (no rng):
      x_rec: frame-mean recurrent + fresh-noise current, [tau_s (I_s(0) - I_s(T)) + tau sum_j A_ji n^del_j +
             integral of the noise] / T, with n^del the spikes delivered in the frame (in flight at the start, plus
             emitted, minus in flight at the end; E and I neurons). The RK2 update of I_s makes the first two terms
             exactly the summed midpoint current the theta step integrates; with tau_s = 0 the kicks are jumps of
             area tau w, the same second term;
      n_ext: teleportation pulses received;  x: x_rec + tau a_ext n_ext / T (each pulse a jump of area tau a_ext);
      xi: the frame-mean fresh-noise current alone (zero mean, variance sigma^2 tau / T; GainEstimator's regressor);
      n: E spike counts (H, N);  T: frame length (ms)."""
    H, M = S.shape[1], S.shape[2]
    T = p.k * p.bin
    n_all = S.sum(0).astype(float)
    delivered = b0 + n_all - state[2].sum(0)
    area = np.einsum("hj,ji->hi", delivered, A) * p.tau
    if p.tau_s > 0:
        area += p.tau_s * (I0 - state[1])
    ptr, _, ev_i, xn = ev
    xi = xn.sum(0) / len(xn)
    area += xi * T
    n_ext = np.zeros((H, M))
    np.add.at(n_ext, (np.repeat(np.arange(H), np.diff(ptr)), ev_i), 1.0)
    x_rec, n_ext = area[:, :N] / T, n_ext[:, :N]
    return dict(x_rec=x_rec, n_ext=n_ext, x=x_rec + p.tau * p.a_ext * n_ext / T, xi=xi[:, :N], n=n_all[:, :N], T=T)


def n_neurons(N, m, roles, p):
    """M = N + the inhibitory neurons of p.inh."""
    return N + p.n_inh * (len(pools(m, roles, p.inh)) if p.inh else 0)


def pairings(S, p):
    """Pairing counts of a k x N raster for the QIF window: flat lags 1..p.lags (sp.pairings), or with tau_plus > 0
    the exponential window w_l = exp(-(l - 1) / tau_plus), l >= 1, through a presynaptic trace (one decaying pulse
    count per neuron, crossbar-native). Returns (causal, causal - acausal, causal - chance) as sp.pairings."""
    from mapstdp import spiking as sp
    if p.tau_plus <= 0:
        return sp.pairings(S, p.lags)
    S = S.astype(float)
    k, c, g = len(S), S.sum(0), np.exp(-1 / p.tau_plus)
    tr = np.zeros_like(S)
    for t in range(1, k):
        tr[t] = S[t - 1] + g * tr[t - 1]
    C = S.T @ tr
    lag = np.arange(1, k)
    return C, C - C.T, C - np.outer(c, c) * (g ** (lag - 1) * (k - lag)).sum() / k ** 2


def pooled(S, H, p):
    """sp.pooled with the QIF pairing window."""
    from mapstdp import spiking as sp
    Sh = S.reshape(len(S), H, -1)
    return sp.pi_hat(Sh.sum(1)), sum(pairings(Sh[:, h], p)[2] for h in range(H)), int(S.sum())


GAIN_TAU = 500.0  # frames: memory of the gain regression's running sums (exponential)
GAIN_SHRINK = 1000.0  # frames: prior weight of the module gain in each neuron's gain (per-neuron data overtakes it
#   after ~GAIN_SHRINK frames; measured: per-frame slope noise sd 0.04 / ms against a between-neuron sd of 0.001)
GAIN_PRIOR = 5.0  # neuron-frames: prior weight of the MPR transfer gain in the module gain (used for the first frame)


class GainEstimator:
    """Per-neuron gain G = dr/dI of the E neurons of every head (rule 'emergent', Eq. 11b), by regression of the frame
    rate r on the frame-mean fresh-noise current xi across frames (running sums over ~GAIN_TAU frames).
    Why the noise and not the whole measured input: at the default operating point the mean input is dominated by the
    pools' inhibition, which the neuron's own spikes recruit, so rate regressed on total input has the wrong sign
    (-0.0004 against a finite-difference f-I slope of +0.0029 / ms on CartPole's initial W; E-input only: +18% bias,
    corr 0.70 across neurons). The noise is exogenous and zero-mean, so its slope is unbiased (within 1%, corr 0.95).
    Estimate, hierarchical shrinkage (both crossbar-native: per-neuron sums and a K-vector of module sums):
      module  G_k  = (sum_{h, i in k} Sxr + GAIN_PRIOR v <G0>_k) / (sum_{h, i in k} Sxx + GAIN_PRIOR v),
      neuron  G_hi = (Sxr_hi + GAIN_SHRINK v G_k) / (Sxx_hi + GAIN_SHRINK v),   clipped at 0,
    Sxr = sum xi (r - rbar), Sxx = sum xi^2 (discounted by 1 - 1/GAIN_TAU per frame), v = sigma^2 tau / T the
    expected per-frame variance of xi, G0 the MPR transfer gain at the measured input (transfer; with sigma = 0 there
    is no instrument and G = the module mean of G0). G0 is a poor prior at fluctuation-driven points (~0 at the
    default QIFParams, where the mean input is far below rheobase), but it carries weight for one frame only."""

    def __init__(self, H, N, m, K, p):
        self.m, self.K, self.n = np.asarray(m), K, 0
        self.v = p.sigma ** 2 * p.tau / (p.k * p.bin)
        self.sxr, self.sxx, self.rbar = np.zeros((H, N)), np.zeros((H, N)), np.zeros((H, N))

    def gains(self, g0):
        """(H, N) gain estimate from the sums so far, g0 (H, N) the prior (MPR transfer gain)."""
        msum = lambda x: np.bincount(self.m, x.sum(0), self.K)  # noqa: E731
        nk = np.bincount(self.m, minlength=self.K) * len(g0)
        Gk = msum(g0) / nk  # module mean of the prior
        if self.v == 0:  # no fresh noise: no instrument, the prior alone
            return np.maximum(np.broadcast_to(Gk[self.m], g0.shape).copy(), 0.0)
        Gk = (msum(self.sxr) + GAIN_PRIOR * self.v * Gk) / (msum(self.sxx) + GAIN_PRIOR * self.v)
        return np.maximum((self.sxr + GAIN_SHRINK * self.v * Gk[self.m]) / (self.sxx + GAIN_SHRINK * self.v), 0.0)

    def update(self, xi, r):
        """Add one frame: xi (H, N) frame-mean noise current, r (H, N) frame rate (per ms)."""
        if self.n == 0:  # the first frame only sets the rate baseline (an Sxx without Sxr would bias G to 0)
            self.rbar = r.copy()
        else:  # centred on the earlier frames' mean (independent of this frame's xi, so unbiased)
            d = 1 - 1 / GAIN_TAU
            self.sxr = d * self.sxr + xi * (r - self.rbar)
            self.sxx = d * self.sxx + xi ** 2
            self.rbar = self.rbar + (r - self.rbar) / min(self.n + 1, GAIN_TAU)
        self.n += 1

    STATE = ("sxr", "sxx", "rbar", "n")  # checkpoint arrays (m2.Agent.state)


def prior_gain(inp, p):
    """MPR transfer gain at the frame's measured input (transfer: phi at x_rec plus the pulses' forcing term)."""
    return transfer(inp["x_rec"], inp["n_ext"] / inp["T"], p)[1]


def window_pairs(S, H, p, tail=None):
    """Raw pair counts of the QIF window (Eq. 8), summed over heads, cross-head pairs excluded: kappa[i, j] =
    sum_t x_j(t) N_i(t, t + w] with the window of pairings (flat lags 1..p.lags, or with tau_plus > 0 the exponential
    weights exp(-(lag - 1) / tau_plus)). tail carries the previous frame of a warm-started head: the last p.lags bins
    (flat) or the presynaptic trace (exponential), so pairs that straddle the frame boundary are counted; None = no
    earlier spikes (cold start). Returns (kappa (N, N) [post, pre], the tail for the next frame)."""
    k = len(S)
    S3 = S.reshape(k, H, -1).astype(float)
    N = S3.shape[2]
    if p.tau_plus <= 0:
        L = p.lags
        X = np.concatenate([np.zeros((L, H, N)) if tail is None else tail, S3])
        kap = sum(np.tensordot(S3, X[L - lag:L - lag + k], ([0, 1], [0, 1])) for lag in range(1, L + 1))
        return kap, X[-L:].copy()
    g = np.exp(-1 / p.tau_plus)
    tr = np.empty_like(S3)
    tr[0] = 0.0 if tail is None else tail
    for t in range(1, k):
        tr[t] = S3[t - 1] + g * tr[t - 1]
    return np.tensordot(S3, tr, ([0, 1], [0, 1])), S3[-1] + g * tr[-1]


def mean_counter(S, W, H, G, p):
    """Eq. 11b summed over heads: c_ij = tau b_ij sum_h G_hi n_hj, with b_ij = gain W_ij / d_j the kick (V units) j
    gives i (weights) and n_hj the frame's spike counts. G (H, N). Excitatory W only (the pools are outside W)."""
    n = S.reshape(len(S), H, -1).sum(0).astype(float)
    b = p.gain * W / np.maximum(W.sum(0), 1e-12)
    return p.tau * b * (G.T @ n)


def coupling(A, groups):
    """Population coupling of the [pre, post] matrix A: J[a, b] = mean over post i in group a of sum_{j in b} A[j, i]
    (the mean summed weight one neuron of a receives per unit rate of b)."""
    return np.array([[A[np.ix_(gb, ga)].sum(0).mean() for gb in groups] for ga in groups])


def mpr(J, nu, p, T=300.0, dt=0.001, ext="force", I_t=None):
    """Mean field of Montbrio, Pazo & Roxin (2015) for populations with Lorentzian eta (eta_bar, delta), in
    W = pi tau r + i v form: tau dW/dt = delta + i (eta_bar + I) - i W^2, with the recurrent input I = I_s,
    tau_s dI_s/dt = -I_s + tau J r(t - delay) (I_s = tau J r(t - delay) if tau_s = 0), plus I_t(t) (P,) if given.
    External pulses of amplitude a_ext at nu[a] per ms per neuron (the MPR reduction is exact for none of these):
      'force': each pulse is a spike: r = r_MPR + nu P(a_ext > 2 sqrt(-eta)) (large pulses from rest, low nu);
      'z':     dZ/dt = nu (Z(W + i a_ext) - Z) on the Kuramoto order parameter Z = (1 - conj W) / (1 + conj W), i.e. the
               jumped mixture projected back onto the Lorentzian family through Z;
      'mean':  the mean current tau a_ext nu (diffusion limit: small pulses at high rate).
    The fresh noise (sigma) is not represented. Euler, step dt (ms). Returns rates (1/ms), (steps x P) every 0.1 ms."""
    nu = np.asarray(nu, float)
    P, tau, a = len(nu), p.tau, p.a_ext
    W = np.full(P, np.pi * tau * 1e-4 + 1j * -np.sqrt(abs(p.eta_bar)) * (p.eta_bar < 0), complex)
    D = max(int(round(p.delay / dt)), 1)
    hist, Is = np.zeros((D, P)), np.zeros(P)
    every, out = int(round(0.1 / dt)), []
    I_ext = tau * a * nu if ext == "mean" else 0.0
    r_force = nu * (0.5 + np.arctan((p.eta_bar + a * a / 4) / p.delta) / np.pi) if ext == "force" else 0.0
    for s in range(int(round(T / dt))):
        r = W.real / (np.pi * tau) + r_force
        drive = tau * J @ hist[s % D]
        hist[s % D] = r  # read at s + D
        Is = Is + dt * (drive - Is) / p.tau_s if p.tau_s > 0 else drive
        It = I_ext + (I_t(s * dt) if I_t is not None else 0.0)
        W = W + dt / tau * (p.delta + 1j * (p.eta_bar + Is + It) - 1j * W ** 2)
        if ext == "z":
            Z = (1 - W.conj()) / (1 + W.conj())
            Zj = (1 - (W + 1j * a).conj()) / (1 + (W + 1j * a).conj())
            Z = Z + dt * nu * (Zj - Z)
            W = ((1 - Z) / (1 + Z)).conj()
        if s % every == 0:
            out.append(r)
    return np.array(out)


_PHI_CACHE = {}


def _phi_table(p):
    """Stationary MPR rate of a population with eta ~ the engine's truncated Lorentzian under a constant input I:
    r(I) = E_eta[sqrt(max(eta + I, 0))] / (pi tau) (the uncoupled stationary rate of every neuron, averaged; equals the
    MPR fixed point Re sqrt(eta_bar + I + i delta) / (pi tau) when eta_max = inf). Tabulated on I in [-60, 400] by
    quadrature in u = arctan((eta - eta_bar) / delta), which is uniform under the Lorentzian. Also the share of the
    population a teleportation pulse forces to spike at input I, P_f(I) = P(-a_ext^2 / 4 < eta + I < 0) (silent
    neurons whose jump crosses threshold; tonic neurons only shift phase). Returns (I, r, dr/dI, P_f, dP_f/dI)."""
    key = (p.eta_bar, p.delta, p.eta_max, p.tau, p.a_ext)
    if key not in _PHI_CACHE:
        u0 = np.arctan((p.eta_max - p.eta_bar) / p.delta) if p.eta_max > p.eta_bar else np.pi / 2
        u = (np.arange(4000) + 0.5) / 4000 * 2 * u0 - u0
        eta = p.eta_bar + p.delta * np.tan(u)
        grid = np.linspace(-60.0, 400.0, 46001)
        r = np.empty_like(grid)
        for k in range(0, len(grid), 2000):
            r[k:k + 2000] = np.sqrt(np.maximum(eta[None, :] + grid[k:k + 2000, None], 0)).mean(1)
        r /= np.pi * p.tau
        cdf = lambda x: np.clip((np.arctan((np.minimum(x, p.eta_max) - p.eta_bar) / p.delta) + u0) / (2 * u0), 0, 1)  # noqa: E731
        pf = cdf(-grid) - cdf(-p.a_ext ** 2 / 4 - grid)  # pulse forces a spike: -a^2/4 < eta + I < 0
        _PHI_CACHE[key] = (grid, r, np.gradient(r, grid), pf, np.gradient(pf, grid))
    return _PHI_CACHE[key]


def phi(cur, p):
    """(rate per ms, gain dr/dI) of the populations at input cur (_phi_table, linear interpolation)."""
    grid, r, dr = _phi_table(p)[:3]
    return np.interp(cur, grid, r), np.interp(cur, grid, dr)


def pforce(cur, p):
    """(P_f, dP_f/dI) at input cur (_phi_table)."""
    grid, pf, dpf = _phi_table(p)[0], *_phi_table(p)[3:]
    return np.interp(cur, grid, pf), np.interp(cur, grid, dpf)


def force_frac(p):
    """Share of neurons a teleportation pulse forces to spike from rest: eta > -a_ext^2 / 4 (truncated Lorentzian)."""
    u0 = np.arctan((p.eta_max - p.eta_bar) / p.delta) if p.eta_max > p.eta_bar else np.pi / 2
    uc = np.arctan((-p.a_ext ** 2 / 4 - p.eta_bar) / p.delta)
    return float(np.clip((u0 - max(uc, -u0)) / (2 * u0), 0, 1))


def transfer(x, nu, p):
    """Rate per ms and gain of neurons with input x and teleportation pulse rate nu (per ms): phi(x) + nu P_f(x)."""
    r, g = phi(x, p)
    pf, dpf = pforce(x, p)
    return r + nu * pf, g + nu * dpf


def _newton(r, tA, nu, p, tol, max_iter):
    """Damped Newton on F(r) = r - transfer(tA r, nu) (backtracking on the max-norm residual). Returns
    (r, |F|_max, its)."""
    F = lambda r_: r_ - transfer(tA @ r_, nu, p)[0]  # noqa: E731
    res, it = F(r), 0
    nrm = np.abs(res).max()
    while nrm > tol and it < max_iter:
        G = transfer(tA @ r, nu, p)[1]
        step = np.linalg.solve(np.eye(len(r)) - G[:, None] * tA, -res)
        lam = 1.0
        while lam > 1e-4:
            r_new = np.maximum(r + lam * step, 0.0)
            res_new = F(r_new)
            if np.abs(res_new).max() < (1 - 0.25 * lam) * nrm:
                break
            lam /= 2
        r, res, it = r_new, res_new, it + 1
        nrm = np.abs(res).max()
    return r, nrm, it


def rates(W, v, p, m=None, roles=None, r0=None, tol=1e-9, max_iter=60, ode=True):
    """Neuron-level MPR fixed point of the QIF network (Phase B, step 4). Every neuron is a population with the engine's
    eta distribution (the stationary MPR rate phi, _phi_table), and its teleportation pulses (rate nu_i = rate v_i / bin
    per ms, amplitude a_ext) add nu_i P_f(I_i) forced spikes:
      r = phi(tau A^T r) + nu P_f(tau A^T r),   A = weights(W, p, pools) (E -> E gain W / d_j and the pools).
    Delay and tau_s do not move a fixed point. Valid where the recurrent input acts as a current (many small or slow
    PSPs: tau_s >> 1 ms, many weak IPSPs); the fresh noise sigma is not represented. Damped Newton from r0 (warm start)
    or the uncoupled rates; if it fails, continuation in the coupling (c A, c = 0.1 .. 1), then damped iteration, then
    the time average of the neuron-level ODE (mpr; skipped if not ode).
    Returns (r (M,) per ms, G (M,) = dr/dI (total gain incl. the forcing term), info dict(it, converged, fallback))."""
    N = len(W)
    A = weights(W, p, pools(m, roles, p.inh) if p.inh else ())
    M = len(A)
    nu = np.zeros(M)
    nu[:N] = p.rate * v / p.bin
    tA = p.tau * A.T
    r = transfer(np.zeros(M), nu, p)[0] if r0 is None else np.asarray(r0, float).copy()
    r, nrm, it = _newton(r, tA, nu, p, tol, max_iter)
    info = dict(it=it, converged=bool(nrm <= tol), fallback="")
    if not info["converged"]:  # continuation in the coupling (follows the branch connected to the uncoupled state)
        r = transfer(np.zeros(M), nu, p)[0]
        for c in np.linspace(0.1, 1.0, 10):
            r, nrm, k = _newton(r, c * tA, nu, p, tol, max_iter)
            it += k
        info.update(it=it, converged=bool(nrm <= tol), fallback="continuation")
    if not info["converged"]:  # damped fixed-point iteration, then the ODE time average
        for _ in range(5000):
            r = 0.9 * r + 0.1 * transfer(tA @ r, nu, p)[0]
        if np.abs(r - transfer(tA @ r, nu, p)[0]).max() <= 1e-6:
            info.update(fallback="damped", converged=True)
        elif not ode:
            info.update(fallback="none")
        else:
            r = mpr(A.T.copy(), nu, p, T=300.0)[-1000:].mean(0)
            info.update(fallback="ode")
    return r, transfer(tA @ r, nu, p)[1], info


def brian_frame(A, eta, state, events, p):
    """Brian2 reference of one frame from the same state and pre-drawn events (no spikes in flight: cold buffer).
    Returns the k x H x N E raster and the spike (substep, flat neuron) list."""
    import brian2 as b2
    b2.prefs.codegen.target = "numpy"
    b2.BrianLogger.suppress_name("resolution_conflict")
    b2.start_scope()
    b2.defaultclock.dt = p.dt * b2.ms
    theta0, Is0, buf = state
    assert not buf.any(), "brian_frame starts with no spikes in flight"
    H, M = eta.shape
    ns = dict(tau=p.tau * b2.ms, tau_s=max(p.tau_s, 1e-9) * b2.ms, a_ext=p.a_ext, kick=p.tau / max(p.tau_s, 1e-9))
    syn = p.tau_s > 0
    ptr, ev_s, ev_i, xn = events
    ns["xn"] = b2.TimedArray(xn.reshape(len(xn), H * M), dt=(p.k * p.bin / len(xn)) * b2.ms)
    eqs = ("dtheta/dt = ((1 + c + Is) + (c + Is - 1) * cos(theta)) / tau : 1\ndIs/dt = -Is / tau_s : 1\n"
           if syn else "dtheta/dt = ((1 + c) + (c - 1) * cos(theta)) / tau : 1\nIs : 1\n") + \
        "c = eta + xn(t, i) : 1\neta : 1 (constant)\ndV : 1"
    G = b2.NeuronGroup(H * M, eqs, threshold="theta > pi", reset="theta -= 2 * pi * floor((theta + pi) / (2 * pi))",
                       method="rk2", namespace=ns)
    G.run_regularly("theta = clip(theta, -pi, inf)", when="before_thresholds")
    G.theta, G.Is, G.eta = theta0.ravel(), Is0.ravel(), eta.ravel()
    G.run_regularly("theta = theta + int(dV != 0) * (2 * arctan(tan(theta / 2) + dV) - theta)\ndV = 0", when="end")
    pre, post = np.nonzero(A)
    off = (np.arange(H)[:, None] * M)
    R = b2.Synapses(G, G, "w : 1", on_pre=("Is_post += w * kick" if syn else "dV_post += w"), namespace=ns)
    R.connect(i=(pre + off).ravel(), j=(post + off).ravel())
    R.w = np.tile(A[pre, post], H)
    R.delay = p.delay * b2.ms
    hh = np.repeat(np.arange(H), np.diff(ptr))
    nets = [G, R]
    if len(ev_s):
        X = b2.SpikeGeneratorGroup(H * M, hh * M + ev_i, ev_s * p.dt * b2.ms)
        E = b2.Synapses(X, G, on_pre="dV_post += a_ext", namespace=ns)
        E.connect(j="i")
        nets += [X, E]
    mon = b2.SpikeMonitor(G)
    net = b2.Network(*nets, mon)
    net.run(p.k * p.nsub * p.dt * b2.ms)
    s = np.round(np.asarray(mon.t / b2.ms) / p.dt).astype(int)
    i = np.asarray(mon.i)
    S = np.zeros((p.k, H, M), bool)
    S[s // p.nsub, i // M, i % M] = True
    return S, (s, i)


def match_frac(s1, i1, s2, i2, tol):
    """Share of the spikes of list 1 matched one-to-one (same neuron, |substep difference| <= tol) by list 2."""
    hit = 0
    for n in np.union1d(i1, i2):
        a, b = np.sort(s1[i1 == n]), list(np.sort(s2[i2 == n]))
        for t in a:
            j = next((q for q, u in enumerate(b) if abs(u - t) <= tol), None)
            if j is not None:
                hit += 1
                b.pop(j)
    return hit / max(len(s1), 1)


def spike_list(kernel, A, eta, state, events, p):
    """(substep, flat neuron) spike list of one frame: the kernel run with bin = dt (one bin per substep)."""
    q = dataclasses.replace(p, bin=p.dt, k=p.k * p.nsub)
    H, M = eta.shape
    S = np.zeros((q.k, H, M), bool)
    kernel(*state, eta, A, *events, S, 1, q.D, q.dt, q.tau, q.tau_s, q.a_ext)
    s, h, i = np.nonzero(S)
    return s, h * M + i


def _check_fi(p, kernel):
    """f-I of isolated neurons (eta = I, no input): rate against sqrt(I) / (pi tau)."""
    cur = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    q = dataclasses.replace(p, k=2000)
    eta = cur[:, None].copy()
    st = (np.full((len(cur), 1), -np.pi / 2), np.zeros((len(cur), 1)), np.zeros((q.D, len(cur), 1), bool))
    S = np.zeros((q.k, len(cur), 1), bool)
    ev = (np.zeros(len(cur) + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((1, len(cur), 1)))
    kernel(*st, eta, np.zeros((1, 1)), *ev, S, q.nsub, q.D, q.dt, q.tau, q.tau_s, q.a_ext)
    t_spk = [np.flatnonzero(S[:, j, 0]) for j in range(len(cur))]  # bins; rate from the inter-spike interval
    r = np.array([(len(t) - 1) / ((t[-1] - t[0]) * q.bin) * 1e3 for t in t_spk])
    r0 = np.sqrt(cur) / (np.pi * q.tau) * 1e3
    return cur, r, r0


def _demo_net(p, H=2, seed=1):
    """Small CLAA network with drawn eta, state and one frame of events (self-checks and Brian2 comparison)."""
    from mapstdp import core
    W, m, roles, _ = core.build_network((12, 8, 8, 8), "CLAA", seed=seed)
    rng = np.random.default_rng(seed)
    v = core.stimulus(0.4, m, roles)
    M = n_neurons(len(W), m, roles, p)
    eta = lorentzian(rng, (H, M), p)
    A = weights(W, p, pools(m, roles, p.inh) if p.inh else ())
    return A, eta, init_state(eta, rng, p), draw_events(v, H, len(W), M, rng, p)


def _copy(st):
    return tuple(x.copy() for x in st)


def _check_counter(p, H=32, F=300, F_test=20, n_syn=40, n_fd=20, dI=0.5, seed=0):
    """Rule 'emergent' self-check on a frozen CartPole-sized W (CLLAA, p_cc = 0), H heads, F warm frames on random
    ring stimuli, the gain estimate updated every frame as in training. On the last F_test frames, with common random
    numbers (stored state and events): the counter c (Eq. 11b, the estimate of that frame) against the direct offspring
    of n_syn random synapses by removal (A[j, i] = 0, the change of i's spikes in the frame); the final gain estimate
    against the finite-difference f-I slope (eta_i +- dI, all heads) of n_fd neurons; rates per module; the stability
    monitor max Re lambda (Eq. 12, continuous form) of the module offspring Lambda^K. Returns a dict."""
    from mapstdp import core
    W, m, roles, mask = core.build_network((36, 16, 16, 16, 16), "CLLAA", p_cc=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    N, K, T = len(W), len(roles), p.k * p.bin
    pls = pools(m, roles, p.inh) if p.inh else ()
    A = weights(W, p, pls)
    M = len(A)
    eta = lorentzian(rng, (H, M), p)
    st, est, tail = init_state(eta, rng, p), GainEstimator(H, N, m, K, p), None
    keep, ns, lam = [], np.zeros(N), []
    for f in range(F):
        ev = draw_events(core.stimulus(rng.random(), m, roles), H, N, M, rng, p)
        s0, I0, b0 = _copy(st), st[1].copy(), st[2].sum(0)
        S = np.zeros((p.k, H, M), bool)
        _kernel(*st, eta, A, *ev, S, p.nsub, p.D, p.dt, p.tau, p.tau_s, p.a_ext)
        inp = frame_inputs(S, A, ev, I0, b0, st, N, p)
        SE = S[:, :, :N].reshape(p.k, H * N)
        kraw, tail = window_pairs(SE, H, p, tail)
        G = est.gains(prior_gain(inp, p))
        est.update(inp["xi"], inp["n"] / T)
        if f >= F - F_test:
            c = mean_counter(SE, W, H, G, p)
            nf = inp["n"].sum(0)
            pk = np.bincount(m, nf, K) / nf.sum()
            L = core.module_offspring(c / nf.sum(), pk, m, K)[0]
            lam.append(np.linalg.eigvals(L).real.max())
            keep.append((s0, ev, c, inp["n"].sum(0)))
            ns += nf
    pre, post = np.nonzero(mask.T)  # mask [post, pre]: pairs (j, i)
    pick = rng.choice(len(pre), n_syn, replace=False)
    cs, truth = np.zeros(n_syn), np.zeros(n_syn)
    for q, (j, i) in enumerate(zip(pre[pick], post[pick])):
        A2 = A.copy()
        A2[j, i] = 0.0
        for s0, ev, c, n0 in keep:
            S = np.zeros((p.k, H, M), bool)
            _kernel(*_copy(s0), eta, A2, *ev, S, p.nsub, p.D, p.dt, p.tau, p.tau_s, p.a_ext)
            truth[q] += n0[i] - S[:, :, i].sum()
            cs[q] += c[i, j]
    Gh = est.gains(np.zeros((H, N)))
    fd_idx = rng.choice(N, n_fd, replace=False)
    Gfd = np.zeros(n_fd)
    for q, i in enumerate(fd_idx):
        for s0, ev, _, _ in keep:
            for sgn in (1, -1):
                et = eta.copy()
                et[:, i] += sgn * dI
                S = np.zeros((p.k, H, M), bool)
                _kernel(*_copy(s0), et, A, *ev, S, p.nsub, p.D, p.dt, p.tau, p.tau_s, p.a_ext)
                Gfd[q] += sgn * S[:, :, i].sum() / (2 * dI * T * H * len(keep))
    rate = np.bincount(m, ns, K) / np.bincount(m, minlength=K) / (H * len(keep) * T) * 1e3
    return dict(corr_c=float(np.corrcoef(cs, truth)[0, 1]), ratio_c=float(cs.sum() / truth.sum()),
                corr_G=float(np.corrcoef(Gh[:, fd_idx].mean(0), Gfd)[0, 1]),
                ratio_G=float(Gh[:, fd_idx].mean() / Gfd.mean()), G_fd=float(Gfd.mean()), rate_hz=rate,
                lam_max=float(np.max(lam)), mpr_prior=float(prior_gain(inp, p).mean()))


if __name__ == "__main__":
    p0 = QIFParams()
    cur, r, r0 = _check_fi(p0, _kernel)
    print("f-I (eta = I, Hz): " + "  ".join(f"I={a:g}: {b:.3f} vs {c:.3f}" for a, b, c in zip(cur, r, r0)))
    assert np.allclose(r, r0, rtol=2e-3), (r, r0)
    for kern in (_kernel, _kernel_np):  # stiff case: strong inhibition must leave theta finite and near -pi
        st = (np.zeros((1, 2)), np.array([[-500.0, 0.0]]), np.zeros((p0.D, 1, 2), bool))
        ev = (np.array([0, 0]), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((1, 1, 2)))
        kern(*st, np.full((1, 2), -2.0), np.zeros((2, 2)), *ev, np.zeros((5, 1, 2), bool), p0.nsub, p0.D, p0.dt,
             p0.tau, p0.tau_s, p0.a_ext)
        assert np.isfinite(st[0]).all() and -np.pi <= st[0].min() and st[0][0, 0] < -2.5, st[0]
    pm = dataclasses.replace(p0, **MPR_POINT)  # rates (Phase B): phi = closed-form MPR, residual, gain, ODE average
    cur = np.array([-6.0, -2.0, 0.0, 2.0, 10.0])
    pb = dataclasses.replace(pm, eta_max=1e6)
    assert np.allclose(phi(cur, pb)[0], np.sqrt(pb.eta_bar + cur + 1j * pb.delta).real / (np.pi * pb.tau), atol=2e-4)
    from mapstdp import core as _core
    Wd, md, rd, _ = _core.build_network((12, 8, 8, 8), "CLAA", seed=1)
    vd = _core.stimulus(0.4, md, rd)
    r, G, info = rates(Wd, vd, pm, md, rd)
    Ad = weights(Wd, pm, pools(md, rd, pm.inh))
    nud = np.zeros(len(Ad))
    nud[:len(Wd)] = pm.rate * vd
    x = pm.tau * Ad.T @ r
    assert info["converged"] and np.abs(r - transfer(x, nud, pm)[0]).max() < 1e-9
    dG = (transfer(x + 1e-3, nud, pm)[0] - transfer(x - 1e-3, nud, pm)[0]) / 2e-3  # table: O(grid) agreement
    assert np.allclose(G, dG, rtol=2e-2, atol=1e-6), np.abs(G - dG).max()
    r0, _, _ = rates(Wd, 0 * vd, pb, md, rd)  # no pulses, full Lorentzian: fixed point = ODE time average
    rode = mpr(Ad.T.copy(), np.zeros(len(Ad)), pb, T=200.0, dt=0.002)[-500:].mean(0)
    assert np.abs(r0 - rode).max() < 5e-4, np.abs(r0 - rode).max()  # 0.5 Hz: the quadrature of the untruncated tail
    print(f"rates: residual ok, gain ok, fixed point vs MPR ODE max diff {np.abs(r0 - rode).max() * 1e3:.4f} Hz")
    CASES = (dict(), dict(tau_s=0.0), dict(inh="ma"), dict(sigma=0.0), dict(inh="", gain=8.0))
    for kw in CASES:  # numba kernel against the numpy kernel, same inputs
        p = dataclasses.replace(p0, **kw)
        A, eta, st, ev = _demo_net(p)
        l1 = spike_list(_kernel, A, eta, _copy(st), ev, p)
        l2 = spike_list(_kernel_np, A, eta, _copy(st), ev, p)
        f = match_frac(*l1, *l2, tol=p.nsub)
        print(f"kernel vs numpy {kw}: {len(l1[0])} / {len(l2[0])} spikes, "
              f"matched within 1 bin {f:.4f}")
        assert len(l1[0]) > 50 and f >= 0.99 and abs(len(l1[0]) - len(l2[0])) <= 0.01 * len(l1[0])
    t0 = time.time()
    ck = _check_counter(p0)
    print(f"emergent counter (Eq. 11b) vs synapse-removal offspring: corr {ck['corr_c']:.3f}, sum ratio "
          f"{ck['ratio_c']:.3f}; gain vs FD f-I slope: corr {ck['corr_G']:.3f}, mean ratio {ck['ratio_G']:.3f} "
          f"(FD {ck['G_fd'] * 1e3:.2f} Hz per unit, MPR prior {ck['mpr_prior'] * 1e3:.2g}); max Re lambda(Lambda^K) "
          f"{ck['lam_max']:.3f}; rates (Hz) {np.round(ck['rate_hz'], 1).tolist()} [{time.time() - t0:.0f} s]")
    assert ck["corr_c"] >= 0.75 and 0.7 <= ck["ratio_c"] <= 1.4 and 0.7 <= ck["ratio_G"] <= 1.4, ck
    assert ck["lam_max"] < 1 and (ck["rate_hz"] > 0.5).all() and (ck["rate_hz"] < 100).all(), ck
    if "--brian2" in sys.argv:
        for kw in CASES:
            p = dataclasses.replace(p0, **kw)
            A, eta, st, ev = _demo_net(p)
            l1 = spike_list(_kernel, A, eta, _copy(st), ev, p)
            _, l2 = brian_frame(A, eta, _copy(st), ev, p)
            f12, f21 = match_frac(*l1, *l2, tol=p.nsub), match_frac(*l2, *l1, tol=p.nsub)
            ex = match_frac(*l1, *l2, tol=0)
            print(f"numpy vs Brian2 {kw}: {len(l1[0])} / {len(l2[0])} spikes, matched within "
                  f"1 bin {f12:.4f} / {f21:.4f}, same substep {ex:.4f}")
            assert min(f12, f21) >= 0.99
    print("qif self-check ok")
