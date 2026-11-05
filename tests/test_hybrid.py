"""The noisy-LIF engine of the TD-STDP backbone (models.tdstdp --engine nlif, iteration 015): the external-current
kernel compiled == interpreted (lif._kernel_ext_{nb,py}, both noise modes, recurrent kicks on an index set), its
noise-free limit == BindsNET's LIF (tdstdp self-check (h1)), the episode reset of the membranes, and a checkpoint round
trip of an nlif run."""
import dataclasses

import numpy as np
import pytest

from mapstdp import lif
from mapstdp import tdstdp as td
from models import tdstdp as ex


@pytest.mark.parametrize("fast", [True, False])
def test_ext_kernel_compiled_equals_interpreted(fast):
    if lif._kernel_ext_nb is None:
        pytest.skip("numba not installed")
    p = dataclasses.replace(lif.LIFParams(), **lif.DT1)
    rng = np.random.default_rng(0)
    k, H, N = 60, 3, 9
    Iext, ridx, Bt = rng.normal(0.05, 0.1, (k, H, N)), np.array([1, 3, 4, 7]), 0.3 * rng.random((4, 4))
    outs = []
    for kern in (lif._kernel_ext_py, lif._kernel_ext_nb):
        V, ref, z = np.full((H, N), p.mu_bg), np.zeros((H, N), np.int64), np.zeros((H, N), bool)
        S, xs, xq = np.zeros((k, H, N), bool), np.zeros((H, N)), np.zeros((H, N))
        with np.errstate(over="ignore"):
            kern(Iext, Bt, ridx, V, ref, z, k, 123, p.a, 3 * (1 - p.a) * p.mu_bg, p.sigma_V * np.sqrt(1 - p.a ** 2),
                 p.th, p.vr, p.nref, S, xs, xq, fast)
        outs.append((S, V, ref, z, xs, xq))
    assert all(np.array_equal(a, b) for a, b in zip(*outs)) and 0.02 < outs[0][0].mean() < 0.5
    # the recurrent kicks reach exactly the index set: xsum differs from the external sum only on ridx
    extra = outs[0][4] - Iext.sum(0)
    assert np.abs(np.delete(extra, ridx, 1)).max() < 1e-12 and (extra[:, ridx] > 0).any()


def test_nlif_noise_free_limit_is_bindsnet():
    assert td.check_bindsnet_limit(seed=7, L=200)[:2] == (0, 0)


def test_nlif_checkpoint_round_trip(tmp_path):
    cfg = ex.preset("CartPole", seed=3, engine="nlif", s_in=1.3, block=20, fast=True)
    r = ex.CKRunner(cfg)
    for _ in range(40):
        r.tick()
    assert r.returns and np.isfinite(r.W).all() and 0 < r.stats[1] / r.stats[4] < 1
    fn = tmp_path / "c.ckpt.npz"
    r.save(fn, 1.0, 0)
    r2 = ex.CKRunner(cfg)
    assert r2.load(fn) == (1.0, 1)
    assert (r2.W == r.W).all() and r2.returns == r.returns and r2.frames == r.frames
    assert all((r2.lst[k] == r.lst[k]).all() for k in ("V", "ref", "z"))
    assert r2.nrng.bit_generator.state == r.nrng.bit_generator.state
    assert r2.rng.bit_generator.state == r.rng.bit_generator.state
    with pytest.raises(AssertionError):  # a checkpoint of another input scale is refused
        ex.CKRunner(dataclasses.replace(cfg, s_in=1.0)).load(fn)
    # episode reset (code 2): membranes to mu_bg with no neuron-noise draw (the window itself is stubbed out)
    r.learn_block = lambda *a: None
    nst = r.nrng.bit_generator.state
    r.lst["V"][:], r.lst["ref"][:], r.lst["z"][:], r.code[:] = 0.3, 1, True, 2
    r.tick()
    assert (r.lst["V"] == r.lp.mu_bg).all() and (r.lst["ref"] == 0).all() and not r.lst["z"].any()
    assert nst == r.nrng.bit_generator.state


# ---- iteration 015 step 2: recurrent columns (tdstdp.HybridRec, --rec)
def _rec_cfg(**kw):
    return ex.preset("CartPole", seed=3, engine="nlif", s_in=0.94, block=20, fast=True, **kw)


def test_rec_full_drops_cross_module_synapses():
    """--rec-full: the partial draws restricted to within-module pairs (no cross-module synapse, same within mask)."""
    a, b = (ex.CKRunner(_rec_cfg(rec="cols", eta_rec=1.0, lam_rec=1.0, rec_full=s)).hr for s in (False, True))
    same = a.m[:, None] == a.m[None, :]
    assert not b.mask[~same].any() and a.mask[~same].any() and np.array_equal(a.mask & same, b.mask)
    assert "full" in ex.tokens(_rec_cfg(rec="cols", rec_full=True))
    cal = [ex.calibrate_rec(_rec_cfg(rec="cols", rec_full=s)) for s in (False, True)]  # shared rates per seed
    assert cal[0][:2] == cal[1][:2] and cal[0][2]["rms_u"] > 1e-6


def test_rec_p_cross_sweep():
    """--p-cross (iteration 018): 0 is --rec-full's R; the masks of one seed are nested in p_cross with the same
    within-module draws; the default keeps the name; calibrate_rec's rates do not depend on p_cross."""
    hr = {p: ex.CKRunner(_rec_cfg(rec="cols", eta_rec=1.0, lam_rec=1.0, p_cross=p)).hr for p in (0.0, 0.05, 0.2, 0.5)}
    full = ex.CKRunner(_rec_cfg(rec="cols", eta_rec=1.0, lam_rec=1.0, rec_full=True)).hr
    assert np.array_equal(hr[0.0].mask, full.mask) and np.array_equal(hr[0.0].R, full.R)
    same = hr[0.5].m[:, None] == hr[0.5].m[None, :]
    ps = sorted(hr)
    for a, b in zip(ps, ps[1:]):
        assert np.array_equal(hr[a].mask & same, hr[b].mask & same)
        assert (hr[b].mask | ~hr[a].mask).all() and hr[b].mask.sum() > hr[a].mask.sum()
    assert ex.tokens(_rec_cfg(rec="cols")) == ex.tokens(_rec_cfg(rec="cols", p_cross=0.05))
    assert "px0.2" in ex.tokens(_rec_cfg(rec="cols", p_cross=0.2))
    cal = [ex.calibrate_rec(_rec_cfg(rec="cols", p_cross=p)) for p in (0.05, 0.5)]
    assert cal[0][:2] == cal[1][:2]


@pytest.mark.parametrize("heads", [1, 3])
def test_rec_chi0_equals_nlif(heads):
    """(h2) rec 'cols' with chi_rec = 0 (kicks of zero) is the nlif baseline bit for bit: same raster, W, returns and
    streams (the window statistics, the R rule and the monitors run, but nothing reaches the posts)."""
    base = ex.CKRunner(_rec_cfg(heads=heads))
    rec = ex.CKRunner(_rec_cfg(heads=heads, rec="cols", chi_rec=0.0, eta_rec=50.0, lam_rec=5.0))
    for _ in range(120):
        base.tick()
        rec.tick()
    assert np.array_equal(base.W, rec.W) and base.returns == rec.returns and base.frames == rec.frames
    assert all(np.array_equal(base.lst[k], rec.lst[k]) for k in ("V", "ref", "z"))
    assert base.rng.bit_generator.state == rec.rng.bit_generator.state
    assert base.nrng.bit_generator.state == rec.nrng.bit_generator.state
    hr = rec.hr  # the walk, the monitors and the R rule ran (the counter is proportional to chi_rec, so R stays)
    assert hr.log["episode"] and np.isfinite(hr.log["L_obs"]).all() and (hr.R[~hr.mask] == 0).all()


def test_rec_credit_matches_definition():
    """HybridRec.credit: e_h += c^h_ij (s_h,m(i) - sum_b T_bj s_hb), c^h_ij = tau_eff b_ij G_hi n_hj, per synapse."""
    rng = np.random.default_rng(0)
    p = dataclasses.replace(lif.LIFParams(), **lif.DT1)
    lay = td.Layout(5, 0, 0, 4, 0, 2, n_lat=8)
    hr = td.HybridRec(lay, 2, 4, 2, 0.3, p, 3, 1, p_same=0.9)
    G, n, s = rng.random((3, 16)) * 50, rng.integers(0, 5, (3, 16)).astype(float), rng.normal(size=(3, 4))
    hr.credit(s, G, n, 0.5)  # pending <- (G, n); e stays 0
    assert not hr.e.any()
    hr.credit(s, G, n, 0.5)
    T, b = hr.R / hr.R.sum(0), p.tau_eff * hr.b()
    ref = np.zeros_like(hr.e)
    for h, i, j in np.ndindex(ref.shape):
        ref[h, i, j] = b[i, j] * G[h, i] * n[h, j] * (s[h, hr.m[i]] - sum(T[k, j] * s[h, hr.m[k]] for k in range(16)))
    assert np.allclose(hr.e, ref)
    assert list(hr.roles) == list("AALL") and (hr.m == np.repeat(np.arange(4), 4)).all()


def test_rec_lat_and_checkpoint(tmp_path):
    """rec 'lat': input -> latent W rows stay at their init, latents are excluded from the readout; checkpoint round
    trip of R, the eligibility, the walk EMA, the window tail and the logs."""
    cfg = _rec_cfg(rec="lat", eta_rec=20.0, lam_rec=2.0)
    r = ex.CKRunner(cfg)
    lat = r.lay.n_crit + np.flatnonzero(r.lay.lat)
    W0 = r.W[lat].copy()
    assert (r.lay.Ma[:, lat] == 0).all() and (r.lay.Mv[lat] == 0).all() and len(lat) == 2 * cfg.n_ae
    for _ in range(60):
        r.tick()
    assert np.array_equal(r.W[lat], W0) and not np.array_equal(r.hr.R, r.snaps[0])
    fn = tmp_path / "c.ckpt.npz"
    r.save(fn, 1.0, 0)
    r2 = ex.CKRunner(cfg)
    r2.load(fn)
    for k, v in r.hr.state().items():
        w = r2.hr.state()[k]
        assert (v == w) if isinstance(v, str) else np.array_equal(np.asarray(v), np.asarray(w)), k
    assert r2.snaps.keys() == r.snaps.keys()
    RK = ex.result_ck(cfg, r, 0.0, None, 0, None, None)["rec"]["RK"]  # module tables per snapshot, shares of R
    K = r.hr.m.max() + 1
    assert RK.keys() == {"0"} and np.allclose(np.sum(RK["0"]), 1) and np.shape(RK["0"]) == (K, K)
    with pytest.raises(AssertionError):  # a checkpoint of other rates is refused
        ex.CKRunner(dataclasses.replace(cfg, eta_rec=1.0)).load(fn)


# ---- iteration 015 step 3: the low-rate nlif regime (--rate-target, calibrate_low)
def test_default_s_in_unchanged():
    """The CK-matched default: calibrate_s_in on CartPole seed 0 gives pilot A's s_in (0.9414, a dyadic bisection
    value) and the runner keeps lif.DT1 (no refractory period)."""
    cfg = ex.preset("CartPole", seed=0, engine="nlif", block=20, fast=True)
    assert ex.calibrate_s_in(cfg)[0] == 0.94140625
    assert ex.CKRunner(dataclasses.replace(cfg, s_in=0.94)).lp.tau_ref == 0.0 and "rt" not in "".join(ex.tokens(cfg))


def test_low_rate_calibration_and_resume(tmp_path):
    """--rate-target 40: s_in hits 40 Hz within 5% (re-probed), tau_ref 2 ms, alphas times k = r_CK / R; a checkpoint
    carries the record and a resume reuses it without calibrating."""
    cfg0 = ex.preset("CartPole", seed=1, engine="nlif", block=20, fast=True, rate_target=40.0, episodes=3, max_eps=3)
    cfg, low = ex.calibrate_low(cfg0)
    assert abs(ex.probe_rate(cfg) / 0.04 - 1) < 0.05 and abs(low["r_got_hz"] / 40 - 1) < 0.05
    assert cfg.alpha_a == pytest.approx(low["k"] * cfg0.alpha_a) and cfg.alpha_c == pytest.approx(low["k"] * 2.0)
    assert low["k"] == pytest.approx(low["r_ck_hz"] / 40)
    assert cfg.eta_a == pytest.approx(cfg0.eta_a * low["eta_scale"][1])
    r = ex.CKRunner(cfg)
    assert r.lp.tau_ref == 0.002 and r.lp.nref == 2 and ex.tokens(cfg0)[-1] == "rt40"
    r.lowrate = low
    for _ in range(20):
        r.tick()
    r.save(tmp_path / "x.ckpt.npz", 1.0, 0)
    res = ex.run_ck(cfg0, out=tmp_path / "x.json")
    assert res["low_rate"] == low and res["config"]["alpha_a"] == cfg.alpha_a and res["config"]["s_in"] == cfg.s_in
    assert res["resumes"] == 1
