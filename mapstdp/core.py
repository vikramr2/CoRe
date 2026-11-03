"""Map-STDP numpy flow-level reference.

Conventions (docs/derivation.md): W[i, j] is the synapse pre j -> post i, T = W / d with
d_j = sum_i W[i, j], J[i, j] = pi_j T[i, j], module tables are indexed [b, a] = (post module, pre module).
Logs are base 2 (bits). Task term: action-gated eligibility (Eq. 4b) here; TD critic in models/basic.py.
"""
import itertools

import numpy as np


def build_network(sizes=(16, 16, 16, 16, 16), roles="CLLAA", p_same=0.5, p_cc=0.1,
                  p_route=0.3, p_cross=0.05, seed=0):
    """SBM: dense within modules, sparse controller recurrence, controller->L/A routing links.
    Returns W (columns normalised to d_j = 1), module labels m, role per module, synapse mask."""
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(len(sizes)), sizes)
    roles = np.array(list(roles))
    P = np.full((len(sizes),) * 2, float(p_cross))
    np.fill_diagonal(P, p_same)
    P[route_pairs(roles)] = p_route
    c = int(np.flatnonzero(roles == "C")[0])
    P[c, c] = p_cc
    mask = rng.random((len(m),) * 2) < P[m][:, m]
    np.fill_diagonal(mask, False)
    assert mask.any(0).all(), "neuron without outgoing synapse"
    W = mask * rng.uniform(0.5, 1.5, mask.shape)
    return W / W.sum(0), m, roles, mask


def route_pairs(roles):
    """K x K bool table [b, a]: a is the controller and b is a latent or action module (Eq. 4a)."""
    return (roles == "C")[None, :] & np.isin(roles, ["A", "L"])[:, None]


def stimulus(o, m, roles, width=0.1):
    """v(o): Gaussian receptive fields on a ring over the controller neurons, o in [0, 1)."""
    ctrl = np.flatnonzero(roles[m] == "C")
    dist = np.abs((np.arange(len(ctrl)) / len(ctrl) - o + 0.5) % 1 - 0.5)
    v = np.zeros(len(m))
    v[ctrl] = np.exp(-dist ** 2 / (2 * width ** 2))
    return v / v.sum()


def cartpole_stimulus(o, m, roles, n=6, sigma=0.4, scale=(0.21, 3.5)):
    """Conjunctive Gaussian RFs on an n x n grid over (theta / 0.21, theta_dot / 3.5) clipped to [-1, 1];
    one controller neuron per cell (needs n^2 controller neurons), v sums to 1."""
    x = np.clip(np.asarray(o, float)[2:] / scale, -1, 1)
    g = np.linspace(-1, 1, n)
    c = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    b = np.exp(-((c - x) ** 2).sum(1) / (2 * sigma ** 2))
    v = np.zeros(len(m))
    v[roles[m] == "C"] = b / b.sum()
    return v


def rbf_stimulus(x, centers, sigma, m, roles):
    """Conjunctive Gaussian RFs at arbitrary centers (one controller neuron per row of centers), x already
    normalised; the exponent is shifted by its minimum so far-from-every-center inputs in high dimension stay finite.
    v sums to 1. cartpole_stimulus is the 6 x 6 grid case (kept separate: it is bit-locked by tests/golden.json)."""
    d2 = ((centers - x) ** 2).sum(1)
    b = np.exp(-(d2 - d2.min()) / (2 * sigma ** 2))
    v = np.zeros(len(m))
    v[roles[m] == "C"] = b / b.sum()
    return v


def fourier_coeffs(d, order, conj):
    """Fourier-basis coefficient vectors c in {0..order}^d with 1..conj nonzero entries (conjunction order <= conj;
    the constant c = 0 excluded), in itertools.product order (iteration 012, Proposal)."""
    return np.array([c for c in itertools.product(range(order + 1), repeat=d) if 0 < np.count_nonzero(c) <= conj],
                     float)


def fourier_features(x, C, onoff=True, bias=False):
    """Nonnegative Fourier-basis rates of x in [0, 1]^d: f = cos(pi C x), ON/OFF split ([f]+, [-f]+) or (1 + f) / 2,
    plus a constant unit if bias (Chung & Kozma 2020, arXiv:2008.13044, with conjunction order capped)."""
    f = np.cos(np.pi * C @ np.asarray(x, float))
    b = np.concatenate([np.maximum(f, 0), np.maximum(-f, 0)]) if onoff else (1 + f) / 2
    return np.append(b, 1.0) if bias else b


def fourier_stimulus(x, C, m, roles, onoff=True, bias=False):
    """v(o) from fourier_features, one feature per controller neuron, v sums to 1 (iteration 012, Proposal)."""
    b = fourier_features(x, C, onoff, bias)
    v = np.zeros(len(m))
    v[roles[m] == "C"] = b / max(b.sum(), 1e-12)
    return v


def flow(W, v, alpha=0.2, n=None):
    """Per-frame flow (Eq. 1). n=None: exact solve; else n hops of power iteration from v (Eq. 1a)."""
    T = W / W.sum(0)
    if n is None:
        return alpha * np.linalg.solve(np.eye(len(v)) - (1 - alpha) * T, v)
    pi = v.copy()  # the iterates x_k of derivation §2.3 (Eq. 1a), converging to pi
    for _ in range(n):
        pi = (1 - alpha) * T @ pi + alpha * v
    return pi


def module_stats(W, pi, m, K):
    """p (visit), q (exit), JK[b, a] (module flow), TK = JK / p (Eq. 1c)."""
    H = np.eye(K)[m]
    J = W / W.sum(0) * pi
    JK = H.T @ J @ H
    p = H.T @ pi
    q = JK.sum(0) - np.diag(JK)
    with np.errstate(divide="ignore", invalid="ignore"):
        TK = JK / p
    return p, q, JK, TK


def policy(p, roles, beta=1.0):
    """Eq. 1d: action-module flow shares; beta != 1 is the sharpened policy p_a^beta / sum p^beta (Proposal, Eq. 1e)."""
    pa = p[roles == "A"] ** beta
    return pa / pa.sum()


def _plogp(x):
    """x log2 x elementwise, with 0 log 0 = 0."""
    x = np.asarray(x, float)
    return np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.0)


def map_equation(p, q, pi):
    """L(M), Eq. 3, in bits."""
    return float(_plogp(q.sum()) - 2 * _plogp(q).sum() - _plogp(pi).sum() + _plogp(p + q).sum())


def mstar(p, q, clip=20.0):
    """M*_m = log2(q_tot (p_m + q_m) / q_m^2) >= 0, clipped to [0, clip] (diverges as q_m -> 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log2(q.sum() * (p + q) / q ** 2)
    return np.clip(np.nan_to_num(M, nan=clip, posinf=clip), 0, clip)


def modulator(p, q, roles, clip=20.0, routing="none", mu=0.0):
    """K x K table g[b, a] = M*_a I[b != a] (Eq. 6). routing: 'none' | 'exempt' (g = 0 on routing
    pairs) | 'dual' (subtract mu on routing pairs, Eq. 4a). 'exempt' is the agent's default
    (m2.Config, derivation §4.5); 'dual' is an m1-only ablation."""
    g = mstar(p, q, clip)[None, :] * (1 - np.eye(len(p)))
    if routing == "exempt":
        g[route_pairs(roles)] = 0.0
    elif routing == "dual":
        g[route_pairs(roles)] -= mu
    return g


def latent_output_barrier(p, JK, roles, beta):
    """Latent-output barrier (Eq. 4e, Proposal): K x K table with beta p_a / J_{A a} on pairs a in L, b in A, else 0.
    Subtracted from g, it is the marginal cost of -beta sum_{l in L} p_l log J_{A l} (pi fixed), so a latent module's
    flow into the action modules is pushed up as it nears 0 and never reaches it."""
    L, A = roles == "L", roles == "A"
    b = np.where(L, beta * p / np.maximum(JK[A].sum(0), 1e-12), 0.0)
    return A[:, None] * b[None, :]


def j_route(JK, roles):
    """Controller routing flow J_route = sum over b in A, L of J_{bC} (Eq. 4a)."""
    return float(JK[route_pairs(roles)].sum())


def centered(tab, W, m):
    """tab[m(i), m(j)] - baseline_j, baseline_j = sum_k T[k, j] tab[m(k), m(j)] (Eqs. 4, 6).
    tab = g gives the modulator minus its per-neuron baseline; tab = 1 - I gives G(i, j) (Eq. 7)."""
    T = W / W.sum(0)
    G = tab[m][:, m]
    return G - (T * G).sum(0)


def structural_update(W, mask, g, kappa, m, lam, eps=0.0, wmin=1e-3, wmax=1.0):
    """Eq. 4: W += -lam (g_ij - gbar_j + eps (d_j - 1)) kappa_ij, then sign mask and bounds on existing synapses.
    eps: d_j homeostasis in the baseline (derivation §2.2). kappa: expected (c * J), sampled (sample_pairings)
    or spiking (causal, balanced or covariance) pairings."""
    return clip_weights(W - lam * (centered(g, W, m) + eps * (W.sum(0) - 1)) * kappa, mask, wmin, wmax)


def action_score(W, pi, m, roles, a, beta=1.0):
    """h_b of Eq. 4b (score of the surrogate policy (J^in_a)^beta / sum_A (J^in)^beta) for chosen action module a:
    h_b = beta (I[b = a] - P_beta(b)) / J^in_b on action modules, 0 elsewhere; beta = 1 is Eq. 4b. Returned as a
    K x K table tab[b, a'] = h_b, ready for centered()."""
    K, A = len(roles), roles == "A"
    Jin = np.bincount(m, (W / W.sum(0) * pi).sum(1), K)
    return _score_table(Jin, K, A, a, beta)


def _score_table(Jin, K, A, a, beta):
    """The K x K score table of action_score from the module inflows J^in (shared by Eqs. 4b and the emergent 4b)."""
    P = np.zeros(K)
    P[A] = Jin[A] ** beta / (Jin[A] ** beta).sum()
    h = np.where(A, -beta * P / np.where(A, Jin, 1.0), 0.0)
    h[a] += beta / Jin[a]
    return np.repeat(h[:, None], K, 1)


def action_inflow(W, pi, m, K):
    """J^in_b = sum_{i in b} sum_j T_ij pi_j per module (action shares at beta = 1 are J^in_A / sum J^in_A)."""
    return np.bincount(m, (W / W.sum(0) * pi).sum(1), K)


def action_entropy(W, pi, m, roles):
    """H(P~) in nats, P~ the beta = 1 action shares J^in_A / sum J^in_A, pi held fixed (iteration 012)."""
    J = action_inflow(W, pi, m, len(roles))[roles == "A"]
    P = J / J.sum()
    return float(-(P * np.log(P)).sum())


def entropy_score(W, pi, m, roles):
    """Entropy modulator (Eq. 4h, iteration 012, Proposal) as a K x K table tab[b, a'] = g_b / J^in_b with
    g_b = -P~(b) (log P~(b) - sum_a P~(a) log P~(a)) on action modules, 0 elsewhere (P~: beta = 1 shares).
    centered(tab) * kappa with kappa = J averages to W * dH(P~)/dW, pi fixed (self-check)."""
    K, A = len(roles), roles == "A"
    Jin = action_inflow(W, pi, m, K)
    P = Jin[A] / Jin[A].sum()
    h = np.zeros(K)
    h[A] = -P * (np.log(P) - (P * np.log(P)).sum()) / Jin[A]
    return np.repeat(h[:, None], K, 1)


def eligibility_step(e, W, kappa, pi, m, roles, a, tau_e=3.0, beta=1.0):
    """Eq. 4b: e <- (1 - 1/tau_e) e + kappa_ij (h_m(i) - hbar_j)."""
    return (1 - 1 / tau_e) * e + kappa * centered(action_score(W, pi, m, roles, a, beta), W, m)


def homeostasis_update(W, mask, kappa, eps_h, wmin=1e-3, wmax=1.0):
    """d_j homeostasis as its own term on all plasticity (model v2): W -= eps_h (d_j - 1) kappa_ij. Its expectation
    -eps_h (d_j - 1) pi_j W_ij / d_j rescales column j uniformly, so T, pi and D are unchanged.
    Abstraction: presynaptic (outgoing) conservation, not postsynaptic synaptic scaling (docs/model.md)."""
    return clip_weights(W - eps_h * (W.sum(0) - 1) * kappa, mask, wmin, wmax)


def additive_G_update(W, mask, m, roles, eta, routing="none"):
    """Exact-gradient baseline (Eq. 7 with positive factors dropped): W += -eta G(i, j) on synapses."""
    c = 1 - np.eye(len(roles))
    if routing == "exempt":
        c[route_pairs(roles)] = 0.0
    return clip_weights(W - eta * centered(c, W, m) * mask, mask)


def clip_weights(W, mask, wmin=1e-3, wmax=1.0):
    """Explicit bounds; wmin > 0 is the additive floor that keeps multiplicative updates from freezing."""
    return np.where(mask, np.clip(W, wmin, wmax), 0.0)


def grad_L(W, pi, m, K):
    """Exact dL/dW with pi fixed (Eq. 7): M*_{m(j)} pi_j / d_j G(i, j); M* unclipped."""
    p, q, _, _ = module_stats(W, pi, m, K)
    g = mstar(p, q, np.inf)[None, :] * (1 - np.eye(K))
    return centered(g, W, m) * pi / W.sum(0)


def sample_pairings(J, n, rng):
    """n causal pairings drawn with probability proportional to J_ij (mean-field assumption)."""
    return rng.multinomial(n, (J / J.sum()).ravel()).reshape(J.shape).astype(float)


# ---- emergent walk (Eqs. 8-12): the walk observed from spikes, a signed counter for the step ----------------------
# Every statistic of the rule is needed only at module resolution: R[b, j] = sum_{i in b} kappa_ij (K x N, j's pair
# counts by target module: a per-neuron count of a K-number module-activity broadcast, derivation Eq. 8). The
# baselines, M*, e^obs and the running averages (obs_tau, models/basic.py) therefore need no per-synapse state.


def module_rows(X, m, K):
    """K x N aggregate of an N x N [post, pre] matrix over postsynaptic modules: [b, j] = sum_{i in b} X[i, j]."""
    return np.eye(K)[m].T @ X


def observed_walk_modules(R, pi, m, K, fallback=None):
    """Eq. 8 at module resolution. R: K x N pair counts by target module (any normalisation; module_rows of kappa_raw,
    or a running average of it). Returns p, q, JK[b, a] (module link flow of J^obs_ij = pi_j T^obs_ij), tK (K x N,
    tK[b, j] = sum_{i in b} T^obs_ij) and e_obs (N, the exit share of j's observed transitions). Columns with no counts
    (a silent j, or one whose spikes nobody followed; 0/0) take fallback[:, j] (e.g. module_rows of T), else 0."""
    tot = R.sum(0)
    live = tot > 0
    tK = np.where(live, R / np.where(live, tot, 1.0), 0.0)
    if fallback is not None:
        tK[:, ~live] = fallback[:, ~live]
    JK = (tK * pi) @ np.eye(K)[m]
    p = np.bincount(m, pi, K)
    q = JK.sum(0) - np.diag(JK)
    return p, q, JK, tK, tK.sum(0) - tK[m, np.arange(len(m))]


def observed_walk(kappa_raw, pi, m, K, fallback=None):
    """Eq. 8, the observed walk: T^obs = kappa_raw / column sums, J^obs = pi_j T^obs_ij, module tables p, q, JK as
    module_stats, e_obs_j = sum_{i not in m(j)} T^obs_ij. Silent columns (0/0) take fallback[:, j] (an N x N
    column-stochastic matrix, e.g. T), else 0. Returns J^obs, p, q, JK, T^obs, e_obs. With kappa_raw = J (flow) it is
    module_stats(W, pi) exactly in exact arithmetic (T^obs = T, derivation: the flow engine is the a = X = 0 case)."""
    tot = kappa_raw.sum(0)
    live = tot > 0
    Tobs = np.where(live, kappa_raw / np.where(live, tot, 1.0), 0.0)
    if fallback is not None:
        Tobs[:, ~live] = fallback[:, ~live]
    Jobs = Tobs * pi
    H = np.eye(K)[m]
    JK = H.T @ Jobs @ H
    p = H.T @ pi
    q = JK.sum(0) - np.diag(JK)
    return Jobs, p, q, JK, Tobs, (Tobs * (m[:, None] != m[None, :])).sum(0)


def centered_walk(tab, tK, m):
    """tab[m(i), m(j)] - gbar_j with the observed-walk baseline gbar_j = sum_k T^obs_kj tab[m(k), m(j)]
    = sum_b tK[b, j] tab[b, m(j)] (Eq. 4; centered() with T^obs in place of T). tK: K x N module shares of T^obs."""
    return tab[m][:, m] - (tK * tab[:, m]).sum(0)


def structural_increment_emergent(g, tK, c, m):
    """Eq. 4 without the rate: -(g_ij - gbar^obs_j) c_ij (structural_update_emergent adds lam times this and clips)."""
    return -centered_walk(g, tK, m) * c


def structural_update_emergent(W, mask, g, tK, c, m, lam, wmin=1e-3, wmax=1.0):
    """Eq. 4 (Theorem E, Eq. 10: it descends D): W += -lam (g_ij - gbar^obs_j) c_ij, then mask and bounds. For the map
    table g = M*_a I[b != a] this is -lam M*_{m(j)} (I[i not in m(j)] - e^obs_j) c_ij. c: the signed counter
    (Eq. 11a/11b), per frame divided by the frame's spikes; tK: module shares of the observed walk
    (observed_walk_modules)."""
    return clip_weights(W + lam * structural_increment_emergent(g, tK, c, m), mask, wmin, wmax)


def homeostasis_update_emergent(W, mask, pi, eps_h, wmin=1e-3, wmax=1.0):
    """Eq. 4d: W -= eps_h (d_j - 1) W_ij pi_j / d_j, pi_j = n_j / total spikes (flow: the flow). An exact column
    rescaling, so T, the walk and L are unchanged; written in homeostasis_update's operation order so that pi = flow
    gives its result bit for bit (kappa = J there)."""
    return clip_weights(W - eps_h * (W.sum(0) - 1) * (W / W.sum(0) * pi), mask, wmin, wmax)


def score_table_obs(JK, roles, a, beta=1.0):
    """The Eq. 4b score table (action_score) from the observed walk's module inflow J^in_b = sum_a JK[b, a]. Guarded for
    spiking frames: h_b = 0 where J^in_b = 0 (a silent action module), and all 0 if every action module is silent."""
    K, A = len(roles), roles == "A"
    Jin = JK.sum(1)
    if (Jin[A] > 0).all():
        return _score_table(Jin, K, A, a, beta)
    h = np.zeros(K)
    if Jin[A].sum() > 0:
        P = np.where(A, np.maximum(Jin, 0) ** beta, 0.0)
        P /= P.sum()
        live = A & (Jin > 0)
        h[live] = -beta * P[live] / Jin[live]
        if Jin[a] > 0:
            h[a] += beta / Jin[a]
    return np.repeat(h[:, None], K, 1)


def eligibility_step_emergent(e, c, JK, tK, m, roles, a, tau_e=3.0, beta=1.0):
    """Eq. 4b with the signed counter: e <- (1 - 1/tau_e) e + c_ij (h_m(i) - hbar^obs_j), h from the observed walk's
    module inflow (score_table_obs), hbar^obs_j = sum_k T^obs_kj h_m(k) (one-hop credit, the analogue of Lemma E for
    log P; derivation Eq. 4b)."""
    return (1 - 1 / tau_e) * e + c * centered_walk(score_table_obs(JK, roles, a, beta), tK, m)


def module_offspring(c, p, m, K):
    """Eq. 12 monitor: Lambda^K[b, a] = sum_{i in b, j in a} c_ij / (spikes of a), from c divided by the frame's spikes
    (so the spikes of a are p_a in the same units); columns of silent modules are 0. Returns (Lambda^K, spectral
    radius max |lambda|)."""
    CK = module_rows(c, m, K) @ np.eye(K)[m]
    L = np.where(p > 0, CK / np.where(p > 0, p, 1.0), 0.0)
    return L, float(np.abs(np.linalg.eigvals(L)).max())


def mstar_ceiling(p, q, q_chance, clip=20.0):
    """M* with the module exit rate at its chance floor q_chance (derivation D2 step 5):
    log2(q_tot (p + q) / q_chance^2), the largest M* the chance coincidences allow in an excitatory
    network; clipped to [0, clip]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log2(q.sum() * (p + q) / q_chance ** 2)
    return np.clip(np.nan_to_num(M, nan=clip, posinf=clip), 0, clip)


def lemma_e_grad(W, E, h, pi, m, K, sgn=None, clip=np.inf):
    """Eq. 9 (Lemma E): dL/dW_ij of the map equation of the observed walk T^obs = E / E_j (pi, the chance and indirect
    parts of E, and the gains held fixed; the local split), for the walk E_kj = a_k + sgn_j h_k T_kj + X_kj:
    pi_j sgn_j / (E_j d_j) [h_i (g_ij - gbar^obs_j) - C_j], g_ij = M*_{m(j)} I[i not in m(j)],
    gbar^obs_j = sum_k T^obs_kj g_kj, C_j = sum_k h_k T_kj (g_kj - gbar^obs_j)."""
    sgn = np.ones(len(W)) if sgn is None else sgn
    d, Ej = W.sum(0), E.sum(0)
    T, Tobs = W / d, E / Ej
    _, p, q, _, _, _ = observed_walk(E, pi, m, K)
    g = mstar(p, q, clip)[m][None, :] * (m[:, None] != m[None, :])
    gbar = (Tobs * g).sum(0)
    C = (h[:, None] * T * (g - gbar)).sum(0)
    return pi * sgn / (Ej * d) * (h[:, None] * (g - gbar) - C)


def _check_emergent():
    """Self-check of Eqs. 8-10: Lemma E (Eq. 9) against central differences on a 16-neuron E-I network with a chance
    term and fixed indirect terms (derivation Appendix A, E1); Theorem E's expected update (E[c] = pi_j sgn_j h_i T_ij)
    against -E_j W * grad L after removing the column rescaling (E2); and the flow case (kappa_raw = c = J): the
    emergent functions reproduce the v4 functions to 1e-12. Returns (Theorem E cosine, Lemma E relative error)."""
    rng = np.random.default_rng(0)
    K, n = 4, 4
    N, m = K * n, np.repeat(np.arange(K), n)
    sgn = np.tile([1.0, 1.0, 1.0, -1.0], K)
    same = m[:, None] == m[None, :]
    W = rng.uniform(0.1, 1, (N, N)) * np.where(same, 2.0, 0.5)
    np.fill_diagonal(W, 0)
    off = ~np.eye(N, dtype=bool)
    h, a, X = rng.lognormal(0, 0.5, N) * 0.3, rng.uniform(0.3, 0.6, N), rng.normal(0, 0.02, (N, N))
    pi = rng.dirichlet(3 * np.ones(N))
    Emat = lambda W_: a[:, None] + sgn[None, :] * h[:, None] * W_ / W_.sum(0) + X  # noqa: E731
    Lobs = lambda W_: map_equation(*observed_walk(Emat(W_), pi, m, K)[1:3], pi)  # noqa: E731
    gr, eps = lemma_e_grad(W, Emat(W), h, pi, m, K, sgn), 1e-6
    fd = np.zeros_like(W)
    for i, j in zip(*np.nonzero(off)):
        E = np.zeros_like(W)
        E[i, j] = eps
        fd[i, j] = (Lobs(W + E) - Lobs(W - E)) / (2 * eps)
    err = np.abs(fd - gr)[off].max() / np.abs(gr[off]).max()
    assert err < 1e-5, err
    assert np.abs((W * gr).sum(0)).max() < 1e-12  # column sums of W * grad L vanish
    E = Emat(W)
    _, p, q, JK, Tobs, _ = observed_walk(E, pi, m, K)
    g = mstar(p, q, np.inf)[None, :] * (1 - np.eye(K))
    tK = module_rows(Tobs, m, K)
    Ec = sgn * h[:, None] * W / W.sum(0) * pi  # E[c_ij] per unit time
    u = -centered_walk(g, tK, m) * Ec * off
    u = u - W * (u.sum(0) / W.sum(0))  # remove the column rescaling
    target = -E.sum(0) * W * gr
    cosE = float(u[off] @ target[off] / np.sqrt((u[off] @ u[off]) * (target[off] @ target[off])))
    assert cosE > 1 - 1e-10, cosE
    W, m, roles, mask = build_network((4, 3, 3, 3), "CLAA", p_same=0.8, p_cross=0.2, seed=1)  # flow == v4
    K, pi = len(roles), flow(W, stimulus(0.3, m, roles), 0.2)
    J = W / W.sum(0) * pi
    p0, q0, JK0, _ = module_stats(W, pi, m, K)
    Jo, p1, q1, JK1, To, eo = observed_walk(J, pi, m, K)
    p2, q2, JK2, tK2, eo2 = observed_walk_modules(module_rows(J, m, K), pi, m, K)
    tK = module_rows(To, m, K)
    for x, y in ((Jo, J), (p1, p0), (q1, q0), (JK1, JK0), (p2, p0), (q2, q0), (JK2, JK0), (tK2, tK), (eo2, eo)):
        assert np.abs(x - y).max() < 1e-12
    g = modulator(p0, q0, roles, routing="exempt") - latent_output_barrier(p0, JK0, roles, 0.3)
    v4 = structural_update(W, mask, g, J, m, 0.05)
    em = structural_update_emergent(W, mask, g, tK, J, m, 0.05)
    assert np.abs(v4 - em).max() < 1e-12
    W13 = W * 1.3
    assert (homeostasis_update(W13, mask, W13 / W13.sum(0) * pi, 5.0) ==
            homeostasis_update_emergent(W13, mask, pi, 5.0)).all()  # same operations: bit-identical
    a = int(np.flatnonzero(roles == "A")[1])
    for beta in (1.0, 3.0):
        e4 = eligibility_step(0.5, W, J, pi, m, roles, a, 3.0, beta)
        ee = eligibility_step_emergent(0.5, J, JK1, tK, m, roles, a, 3.0, beta)
        assert np.abs(e4 - ee).max() < 1e-12
    return cosE, float(err)


if __name__ == "__main__":  # self-check: Eq. 1a bound, Eq. 7 vs finite differences, Theorem 3 by Monte Carlo
    W, m, roles, mask = build_network((4, 3, 3, 3), "CLAA", p_same=0.8, p_cross=0.2, seed=1)
    K, alpha, rng = len(roles), 0.2, np.random.default_rng(0)
    v = stimulus(0.3, m, roles)
    pi = flow(W, v, alpha)
    for n in (5, 10, 20):
        assert np.abs(flow(W, v, alpha, n) - pi).sum() <= 2 * (1 - alpha) ** n
    L = lambda W_: map_equation(*module_stats(W_, pi, m, K)[:2], pi)  # noqa: E731 (pi held fixed)
    gr, h = grad_L(W, pi, m, K), 1e-6
    for i, j in zip(*np.nonzero(mask)):
        E = np.zeros_like(W)
        E[i, j] = h
        assert abs((L(W + E) - L(W - E)) / (2 * h) - gr[i, j]) < 1e-6
    p, q, JK, _ = module_stats(W, pi, m, K)
    g = modulator(p, q, roles, clip=np.inf)
    assert np.allclose((W * centered(g, W, m)).sum(0), 0)  # conserves d_j in expectation
    J = W / W.sum(0) * pi
    mean_dW = sum(-centered(g, W, m) * sample_pairings(J, 500, rng) for _ in range(4000)) / 4000
    r = np.corrcoef(mean_dW[mask], (-W * gr * 500)[mask])[0, 1]
    assert r > 0.99, r
    a = int(np.flatnonzero(roles == "A")[0])  # Eq. 4b: expected increment (kappa = J) = W * dlog P~/dW, pi fixed
    for beta in (1.0, 3.0):  # beta = 3: the sharpened score (Eq. 1e, Proposal)
        def logP(W_, beta=beta):
            Jin = np.bincount(m, (W_ / W_.sum(0) * pi).sum(1), K)[roles == "A"] ** beta
            return np.log(Jin[0] / Jin.sum())
        inc = eligibility_step(0, W, J, pi, m, roles, a, tau_e=1.0, beta=beta)
        for i, j in zip(*np.nonzero(mask)):
            E = np.zeros_like(W)
            E[i, j] = h
            assert abs(W[i, j] * (logP(W + E) - logP(W - E)) / (2 * h) - inc[i, j]) < 1e-6
    def D_lb(W_, beta=0.3):  # Eq. 4e: L - beta sum_L p_l ln J_{A l}, pi fixed; Eq. 5 with g - latent_output_barrier
        p_, q_, JK_, _ = module_stats(W_, pi, m, K)
        return L(W_) - beta * (p_ * np.log(np.maximum(JK_[roles == "A"].sum(0), 1e-300)))[roles == "L"].sum()
    gl = modulator(p, q, roles, clip=np.inf) - latent_output_barrier(p, JK, roles, 0.3)
    glb = centered(gl, W, m) * pi / W.sum(0)
    for i, j in zip(*np.nonzero(mask)):
        E = np.zeros_like(W)
        E[i, j] = h
        assert abs((D_lb(W + E) - D_lb(W - E)) / (2 * h) - glb[i, j]) < 1e-6
    W2 = structural_update(W * 1.5, mask, 0 * g, np.ones_like(W), m, 0.1, eps=1.0)  # homeostasis pulls d_j to 1
    assert (np.abs(W2.sum(0) - 1) < np.abs(1.5 * W.sum(0) - 1)).all()
    W3 = homeostasis_update(W * 1.5, mask, np.ones_like(W), 0.1)
    assert (np.abs(W3.sum(0) - 1) < np.abs(1.5 * W.sum(0) - 1)).all()
    W5, m5, r5, mk5 = build_network((4, 3, 3, 3, 3), "CLAAA", p_same=0.8, p_cross=0.2, seed=2)  # iteration 012
    pi5 = flow(W5, stimulus(0.6, m5, r5), alpha)
    gH = centered(entropy_score(W5, pi5, m5, r5), W5, m5) * pi5 / W5.sum(0)  # Eq. 5 form: dH/dW, pi fixed
    for i, j in zip(*np.nonzero(mk5)):
        E = np.zeros_like(W5)
        E[i, j] = h
        assert abs((action_entropy(W5 + E, pi5, m5, r5) - action_entropy(W5 - E, pi5, m5, r5)) / (2 * h) - gH[i, j]) \
            < 1e-6
    Cf = fourier_coeffs(4, 2, 2)
    assert len(Cf) == 32 and fourier_features(np.full(4, 0.5), Cf).min() >= 0
    vf = fourier_stimulus(np.zeros(4), Cf, *build_network((64, 4, 4), "CLA")[1:3])
    assert np.isclose(vf.sum(), 1)
    Wc, mc, rc, _ = build_network((36, 4, 4), "CLA")
    vc = cartpole_stimulus([0, 0, 0.21, -3.5], mc, rc)  # theta at +max, theta_dot at -max: grid cell (5, 0)
    assert np.isclose(vc.sum(), 1) and vc.argmax() == 30
    cosE, errE = _check_emergent()
    print(f"core self-check ok (Monte Carlo corr with -W*grad = {r:.4f}; Eq. 4b score, homeostasis, encoder; "
          f"Lemma E rel. err {errE:.1e}, Theorem E cosine {cosE:.6f}; emergent == v4 on flow)")
