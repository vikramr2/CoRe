"""LIF (BindsNET/TD-STDP neuron) and branching engines for the v4 map network, with common random numbers.
State arrays are (C, H, N): C conditions sharing random draws (CRN), H independent heads, N neurons.
W[i, j] pre j -> post i; the recurrent matrix in engine units is B = chi * T (T = W / d, presynaptic normalisation)."""
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")
TAU, TH = 100.0, 13.0
A_LEAK = np.exp(-1 / TAU)
SP = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/"


def network(task, trained):
    from models import basic
    cfg = basic.Config("flow", seed=0, la_beta=0.3, task=task, td_norm=0.04, p_cc=0.0, beta_kappa=0.01,
                    beta_max=16.0, eta=5.0)
    ag = basic.Agent(cfg)
    W = pickle.load(open(f"{SP}qif/trained/{task}_s0.pkl", "rb"))["W"] if trained else ag.W
    return dict(W=W, m=ag.m, roles=ag.roles, mask=ag.mask, held=ag.held, K=len(ag.roles), N=len(W))


def B_of(net, op):
    T = net["W"] / net["W"].sum(0)
    return op["chi"] * T


def iext(net, op, v):
    """External drive in engine units per step: LIF mV/ms (= w_in R v); branching: spikes/step (rate v)."""
    return (op["w_in"] * op["R"] if op["kind"] == "lif" else op.get("rate", 1.0)) * v


class Sim:
    def __init__(self, net, op):
        self.op, self.net = op, net
        self.Bt = B_of(net, op).T.copy()  # [pre, post]
        self.N = net["N"]

    def draws(self, rng, H, lam):
        """lam: (H, N) external rate per step. Returns ext (H,N) and noise (H,N)."""
        op = self.op
        if op["kind"] == "lif":
            ext = rng.poisson(lam).astype(float)
            if op.get("Delta"):
                nz = rng.random(lam.shape)
            elif op.get("sigma"):
                nz = rng.standard_normal(lam.shape) * op["sigma"]
            else:
                nz = None
        else:
            ext = (rng.random(lam.shape) < lam).astype(float)
            nz = rng.uniform(-0.5, 0.5, lam.shape)
        return ext, nz

    def step(self, st, ext, nz, dI, kick=None):
        """st = dict(v, z) arrays (C,H,N). Returns new spikes (C,H,N) bool and the deterministic input (C,H,N)."""
        op = self.op
        rec = st["z"] @ self.Bt
        if op["kind"] == "lif":
            inp = rec + op["w_in"] * ext
            if dI is not None:
                inp = inp + dI
            if kick is not None:
                inp = inp + kick
            v = A_LEAK * st["v"] + inp
            if op.get("Delta"):
                p = 1 - np.exp(-np.exp(np.minimum((v - TH) / op["Delta"], 30)))
                z = nz < p
            else:
                if nz is not None:
                    v = v + nz
                z = v >= TH
            v = np.where(z, 0.0, v)
            st["v"] = v
        else:  # branching: P(spike) = clip(x + 1/2): noise U(-1/2, 1/2) + input > 1/2; ext forces (amplitude 1)
            inp = rec + ext
            if dI is not None:
                inp = inp + dI
            if kick is not None:
                inp = inp + kick
            z = (nz + inp) > 0.5
        st["z"] = z.astype(float)
        return z, inp

    def fresh(self, C, H):
        return dict(v=np.zeros((C, H, self.N)), z=np.zeros((C, H, self.N)))


def replay(op, Irec, NZ, deltas, kick=None):
    """Open-loop replay of isolated neurons on recorded deterministic input Irec (T,H,N) with the same noise NZ:
    for each delta (constant current) count spikes. kick: optional (T,H,N) extra input. Returns counts (len(deltas),H,N)."""
    T, H, N = Irec.shape
    D = np.asarray(deltas, float)[:, None, None]
    v = np.zeros((len(D), H, N))
    cnt = np.zeros((len(D), H, N))
    for t in range(T):
        inp = Irec[t] + D
        if kick is not None:
            inp = inp + kick[t]
        if op["kind"] == "lif":
            v = A_LEAK * v + inp
            if op.get("Delta"):
                z = NZ[t] < 1 - np.exp(-np.exp(np.minimum((v - TH) / op["Delta"], 30)))
            else:
                if NZ is not None:
                    v = v + NZ[t]
                z = v >= TH
            v = np.where(z, 0.0, v)
        else:
            z = (NZ[t] + inp) > 0.5
        cnt += z
    return cnt


def stationary(net, op, v, rng, H=8, T=20000, warm=300, deltas=(0.0, 0.05), perturb_R=True, Hr=None,
               Tr=None):
    """Long persistent run at stimulus v. Returns rates r (per step), G (open-loop replay tangent), kick gain,
    Fano of 20 ms counts, and optionally R = dr/dI_ext by network perturbation (CRN)."""
    sim, N = Sim(net, op), net["N"]
    lam = np.broadcast_to(op["R"] * v if op["kind"] == "lif" else op.get("rate", 1.0) * v, (H, N))
    st = sim.fresh(1, H)
    for _ in range(warm):
        ext, nz = sim.draws(rng, H, lam)
        sim.step(st, ext, nz, None)
    Irec = np.empty((T, H, N), np.float32)
    NZ = None if (op["kind"] == "lif" and not op.get("Delta") and not op.get("sigma")) else np.empty((T, H, N), np.float32)
    S = np.empty((T, H, N), bool)
    v0 = st["v"][0].copy()
    for t in range(T):
        ext, nz = sim.draws(rng, H, lam)
        z, inp = sim.step(st, ext, nz, None)
        S[t], Irec[t] = z[0], inp[0]
        if NZ is not None:
            NZ[t] = nz
    r = S.mean((0, 1))
    dl = np.array(deltas) * (1.0 if op["kind"] == "lif" else 0.2)  # branching: delta 0.01
    cnt = replay_v0(op, Irec, NZ, dl, v0)
    base_ok = float(np.abs(cnt[0] - S.sum(0)).sum()) / max(S.sum(), 1)
    G = (cnt[1:].sum(1) - cnt[0].sum(0)) / (dl[1:, None] * T * H)  # per delta (nd-1, N)
    # kick gain: extra kicks of typical recurrent size at rate 1/50 per step
    B = B_of(net, op)
    ksz = float(np.median(B[B > 0]))
    kr = np.random.default_rng(12345)
    kick = (kr.random((T, H, N)) < 0.02).astype(np.float32) * ksz
    ck = replay_v0(op, Irec, NZ, [0.0], v0, kick)[0]
    kickG = (ck.sum(0) - cnt[0].sum(0)) / (kick.sum((0, 1)))  # extra spikes per mV of kick
    c20 = S[:T // 20 * 20].reshape(-1, 20, H, N).sum(1)
    fano = c20.var((0, 1)) / np.maximum(c20.mean((0, 1)), 1e-12)
    xmean = Irec.mean((0, 1))
    out = dict(r=r, G=G, Gall=G, kickG=kickG, ksz=ksz, fano=fano, base_err=base_ok, x=xmean,
               dl=dl, sync=pop_sync(S, net))
    if perturb_R:
        out["R"], out["R_halves"] = network_R(net, op, v, rng, Hr or 8, Tr or T, warm, 2 * dl[1])
    return out


def replay_v0(op, Irec, NZ, deltas, v0, kick=None):
    """replay() starting from the recorded initial membrane state v0 (H,N)."""
    T, H, N = Irec.shape
    D = np.asarray(deltas, float)[:, None, None]
    v = np.broadcast_to(v0, (len(D), H, N)).copy() if op["kind"] == "lif" else None
    cnt = np.zeros((len(D), H, N))
    for t in range(T):
        inp = Irec[t].astype(float) + D
        if kick is not None:
            inp = inp + kick[t]
        if op["kind"] == "lif":
            v = A_LEAK * v + inp
            if op.get("Delta"):
                z = NZ[t] < 1 - np.exp(-np.exp(np.minimum((v - TH) / op["Delta"], 30)))
            else:
                if NZ is not None:
                    v = v + NZ[t]
                z = v >= TH
            v = np.where(z, 0.0, v)
        else:
            z = (NZ[t] + inp) > 0.5
        cnt += z
    return cnt


def network_R(net, op, v, rng, H, T, warm, delta):
    """R[k, i] = d r_k / d I_ext_i by perturbing each neuron's input by delta, CRN over N+1 conditions."""
    sim, N = Sim(net, op), net["N"]
    lam = np.broadcast_to(op["R"] * v if op["kind"] == "lif" else op.get("rate", 1.0) * v, (H, N))
    dI = np.zeros((N + 1, 1, N))
    dI[1:, 0, :] = np.eye(N) * delta
    st = sim.fresh(N + 1, H)
    cnt = np.zeros((2, N + 1, N))
    for t in range(warm + T):
        ext, nz = sim.draws(rng, H, lam)
        z, _ = sim.step(st, ext, nz, dI if t >= 0 else None)
        if t >= warm:
            hh = H // 2
            cnt[0] += z[:, :hh].sum(1)
            cnt[1] += z[:, hh:].sum(1)
    halves = []
    for c in cnt:
        halves.append(((c[1:] - c[0][None]) / (delta * T * (H // 2))).T)  # [k, i]
    return (halves[0] + halves[1]) / 2, halves


def pop_sync(S, net):
    """Synchrony: variance of module population count per ms over sum of single-neuron variances (1 = independent)."""
    m, K = net["m"], net["K"]
    out = []
    for b in range(K):
        X = S[:, :, m == b].astype(float)
        num = X.sum(2).var()
        den = X.var((0, 1)).sum()
        out.append(num / max(den, 1e-12))
    return np.array(out)
