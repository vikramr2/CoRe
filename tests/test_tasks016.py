"""Iteration 016: the TD-STDP backbone on Acrobot and Taxi (models.tdstdp PRESETS by pre-set rules): the one-hot
encoder, the presets' sizes, a short closed loop with recurrent columns, k from CartPole's BindsNET rate, the r_scale
rule and the hybrid-columns-2 study declaration; the race readout of §14.1 Stage B (the decision, the theta rule, the
surrogate's gradient, closed loops, calibrate_race, tokens); the score credit of §20 Stage B' (lif.post_scores, the
default path's lock, the truncation at the bound, the critic rows' identity, the rank-1 rate term, R's credit,
tokens)."""
import dataclasses
import hashlib
import sys

import gymnasium as gym
import numpy as np
import pytest

from experiments import report
from mapstdp import lif
from models import tdstdp as ex
from models.helpers import calibration


def test_onehot_encoder():
    cfg = ex.preset("Taxi", seed=0, in_rep=2)
    phi, n = ex.ck_encoder(cfg)
    f = phi(np.array([3.0, 499.0]))
    assert n == 1000 and f.shape == (2, 1000) and f.sum() == 4
    assert f[0, 3] == f[0, 503] == f[1, 499] == f[1, 999] == 1


@pytest.mark.parametrize("task, n_in, K", [("Acrobot", 208, 3), ("Taxi", 500, 6)])
def test_preset_closed_loop(task, n_in, K):
    """rec 'cols' on the new tasks: sizes, a short run, the module table of R."""
    cfg = ex.preset(task, seed=0, engine="nlif", s_in=1.0, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0)
    r = ex.CKRunner(cfg)
    assert r.lay.n_in == n_in and r.K == K and r.hr.Kr == K
    assert r.lay.n_post == cfg.n_ce + K * cfg.n_ae and cfg.n_ai == 0
    for _ in range(300):
        r.tick()
    assert r.frames > 0 and np.isfinite(r.W).all() and np.isfinite(r.hr.R).all()
    RK = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]["RK"]["0"]
    assert np.shape(RK) == (K, K) and np.isclose(np.sum(RK), 1)


def test_k_from_cartpole_rate():
    """r_ck_hz: k = CartPole's BindsNET rate / rate_target, not this task's."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.18, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, norm_logits=True)  # norm_logits: the BindsNET reference probes run without it
    low, rec = ex.calibrate_low(cfg)
    assert rec["k"] == pytest.approx(ex.R_CK_CARTPOLE / 40) and low.alpha_a == pytest.approx(rec["k"] * 50.0)


@pytest.mark.parametrize("task", ["CartPole", "Acrobot", "Taxi"])
def test_reward_rules(task):
    """r_scale = 0.02 / RMS of the per-step reward under a random policy (CartPole: RMS 1, the preset 0.02); beta_c =
    min(-0.2, 1.2 V_rand), V_rand = r_scale E[r] tau_d / MS (CartPole: -0.2)."""
    env, rng, r = gym.make(ex.TD_TASKS[task].env_id), np.random.default_rng(0), []
    env.reset(seed=0)
    while len(r) < 20000:
        _, x, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        r.append(x)
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 31)))
    p = ex.preset(task)
    assert p.r_scale == pytest.approx(0.02 / np.sqrt(np.mean(np.square(r))), rel=0.03)
    assert p.beta_c == pytest.approx(min(-0.2, 1.2 * p.r_scale * np.mean(r) * p.tau_d / ex.MS), rel=0.05)


def test_hybrid_columns_2_declared():
    s = report.STUDIES["hybrid-columns-2"]
    assert len(s["groups"]) == 4 and all(list(g["seeds"]) == list(range(4000, 4020)) for g in s["groups"].values())
    assert all({"--rate-match-rec", "--ent-match-rec"} <= set(g["flags"]) for g in s["groups"].values())
    heads = {p[0] for p in s["pairs"] if len(p) == 2}
    assert heads == {f"Acrobot_{a}" for a in ("cols_lam0", "cols_on", "cols_sealed")}
    r = dict(config=dataclasses.asdict(ex.preset("Taxi")), returns=list(range(2000)))
    assert report.TD_PAIR_METRICS["primary"](r) == report._auc_min(r)


def test_rate_match_rec():
    """--rate-match-rec: probe_rate runs with the arm's recurrence (the kicks raise the rate at a shared s_in), and only
    on engine nlif (BindsNET's probe stays without recurrence); default off, token 'rm'."""
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols")
    on = dataclasses.replace(cfg, rate_match_rec=True)
    assert ex.probe_rate(on) > ex.probe_rate(cfg) and not ex.Config().rate_match_rec
    assert ex.probe_rate(dataclasses.replace(on, engine="lif", s_in=0.0, rate_target=0.0)) > 0
    assert ex.tokens(on)[-1] != "rm" and "rm" in ex.tokens(on) and "rm" not in ex.tokens(cfg)


def test_ent_match_rec():
    """--ent-match-rec: at a shared s_in the columns' loop sharpens the initial policy (probe_entropy lower than the
    base run's), which match_entropy undoes with f < 1 on alpha_a; default off, token 'em'."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", alpha_a=175.0)
    assert ex.probe_entropy(cfg) < ex.probe_entropy(dataclasses.replace(cfg, rec="none"))
    on = dataclasses.replace(cfg, ent_match_rec=True)
    assert not ex.Config().ent_match_rec and "em" in ex.tokens(on) and "em" not in ex.tokens(cfg)
    cp = ex.preset("CartPole", seed=0, engine="nlif", s_in=0.28, rate_target=40.0, block=20, fast=True, adam=False,
                   alpha_a=175.0)  # one copy, short episodes: ticks with no live copy must not make the probe NaN
    assert 0 < ex.probe_entropy(cp) <= np.log(2) + 1e-9


def test_norm_logits():
    """--norm-logits (016 §13): the policy is invariant to a common scaling of the rates above the floor, the scores
    are CK's at rbar = r0, the eligibility factors are the exact gradient of log p and H through the normalisation
    (finite differences), the floor rule, default off, token 'nl'."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, norm_logits=True)
    r = ex.CKRunner(cfg)
    K, H, rng = r.K, r.H, np.random.default_rng(0)
    assert r.eps_n == pytest.approx(cfg.alpha_a * 0.04 / (cfg.tau_n * cfg.n_ae))
    R = rng.uniform(0.02, 0.08, (H, r.lay.n_post))
    sc, rk, rbar = r.scores(R)
    assert np.allclose(ex.td.softmax(sc), ex.td.softmax(r.scores(0.5 * R)[0]))  # scale-free above the floor
    R0 = R * 0.04 / rbar[:, None]  # pooled rate = r0: CK's scores
    assert np.allclose(r.scores(R0)[0], cfg.alpha_a * R0 @ r.lay.Ma.T)
    assert np.allclose(r.scores(R * 1e-3)[0], cfg.alpha_a * (0.04 / r.eps_n) * (R * 1e-3) @ r.lay.Ma.T)  # floored
    # the exact gradient: d log p_a / d r_j and dH / d r_j vs finite differences on the column rates
    r.prk, r.prbar, r.p, r.action = rk, rbar, ex.td.softmax(sc), np.zeros(H, int)
    ge = ex.td.entropy_coef(r.p)
    fac, u, ug = r.norm_factors(ge)
    claim = fac[:, None] * (np.eye(K)[r.action] - r.p - u[:, None]) * cfg.alpha_a
    claimH = fac[:, None] * (ge - ug[:, None]) * cfg.alpha_a

    def f(rk_, h):
        s = cfg.alpha_a * 0.04 / max(rk_.mean(), r.eps_n) * rk_
        p = ex.td.softmax(s)
        return np.log(p[0]), -(p * np.log(p)).sum()
    for h in range(H):
        for j in range(K):
            d = 1e-6 * np.eye(K)[j]
            lp1, H1 = f(rk[h] + d, h)
            lp0, H0 = f(rk[h] - d, h)
            assert claim[h, j] == pytest.approx((lp1 - lp0) / 2e-6, rel=1e-4, abs=1e-6)
            assert claimH[h, j] == pytest.approx((H1 - H0) / 2e-6, rel=1e-4, abs=1e-6)
    for _ in range(200):
        r.tick()
    assert np.isfinite(r.W).all() and np.isfinite(r.p).all()
    assert not ex.Config().norm_logits and "nl" in ex.tokens(cfg) and "nl" not in ex.tokens(ex.preset("Acrobot"))


def test_hybrid_columns_3_declared():
    s = report.STUDIES["hybrid-columns-3"]
    assert len(s["groups"]) == 12 and all(list(g["seeds"]) == list(range(5000, 5020)) for g in s["groups"].values())
    flags = {"--rate-match-rec", "--ent-match-rec", "--norm-logits"}
    assert all(flags <= set(g["flags"]) for g in s["groups"].values())
    assert all(("--eta-cal" in g["flags"]) == (g["task"] != "CartPole") for g in s["groups"].values())
    assert {g["task"] for g in s["groups"].values()} == {"CartPole", "LunarLander", "Acrobot"}
    assert len([p for p in s["pairs"] if len(p) == 2]) == 9 and len(s["pairs"]) == 18


def test_class_pair_counts():
    """The stimulus-conditioned walk (016 §15): the per-class split of the pair counts sums to the one call exactly,
    class_table gives per-class shares that sum to 1 (0 for an empty class) and resets, and the result and checkpoint
    carry it (rec.RKc)."""
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0, heads=4)
    r = ex.CKRunner(cfg)
    hr, rng = r.hr, np.random.default_rng(1)
    XR = rng.random((20, 4, hr.nR)) < 0.05
    xs, xq = rng.random((4, hr.nR)), rng.random((4, hr.nR)) + 1
    cls = np.array([0, 2, 2, 3])
    hr.window(XR, xs, xq, cls)
    split = hr.acc["kappa"].copy()
    hr.acc, hr.kc[:] = hr._fresh(), 0
    hr.tail[:] = False
    hr.window(XR, xs, xq)
    assert np.array_equal(split, hr.acc["kappa"])
    hr.tail[:] = False
    hr.window(XR, xs, xq, cls)
    csplit = hr.cc.sum(0)
    t = hr.class_table()
    assert t["share"].shape == t["cshare"].shape == (r.K, hr.Kr, hr.Kr)
    assert t["n"][1] == 0 and (t["n"][[0, 2, 3]] > 0).all()
    assert np.allclose(t["share"].sum((1, 2)), [1, 0, 1, 1]) and np.allclose(t["cshare"].sum((1, 2)), [1, 0, 1, 1])
    assert not hr.kc.any() and not hr.cc.any() and np.allclose(csplit, hr.acc["c"])  # the counter's split sums too
    for _ in range(60):
        r.tick()
    out = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]
    k = max(int(x) for x in out["RK"])
    assert str(k) in out["RK"] and set(out["RKc"]) <= set(out["RK"]) or out["RKc"] == {}
    assert "rec_kc" in hr.state()


def _race_runner(**kw):
    return ex.CKRunner(ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True,
                                 sgd=True, adam=False, readout="race", race_beta=6.0, **kw))


def test_race_decision():
    """The race (Eq. 1h, CKRunner.race_block) on a synthetic raster: the first column whose E count reaches theta
    wins at that ms (an earlier lead does not), a copy with no column at theta takes the leader at the deadline, an
    exact tie costs exactly one uniform draw (none otherwise), the actor spikes up to the bound ms (or the window at
    the deadline) are tallied as the energy to decide, and I posts are not counted."""
    r = _race_runner()
    L, n, nc, n_ae = 20, r.lay.n_post, r.lay.n_crit, r.cfg.n_ae
    assert r.theta == 16 and r.pend.tolist() == [-1] and not r.cnt.any()
    X = np.zeros((L, 1, n), bool)
    X[:, 0, nc + n_ae:nc + 2 * n_ae] = True  # column 1: 20 spikes per ms, at theta in the first ms
    X[:, 0, nc] = True  # column 0: 1 spike per ms
    r.race_block(X)
    assert r.pend[0] == 1 and r.cnt[0].tolist() == [20, 400, 0] and r.rdec[0] == 1
    assert r.ebnd.tolist() == [21, 1] and r.wspk[0] == 420  # 21 actor spikes in the bound ms; 420 in the block
    r.cnt[:], r.pend[:], r.wspk[:] = 0, -1, 0
    X = np.zeros((L, 1, n), bool)
    X[:5, 0, nc] = True  # column 0 leads early (5 spikes), column 2 reaches theta at ms 7 (2 spikes per ms)
    X[:, 0, nc + 2 * n_ae] = True
    X[:16, 0, nc + 2 * n_ae + 1] = True
    r.race_block(X)
    assert r.pend[0] == 2 and r.cnt[0].tolist() == [5, 0, 36] and r.ebnd.tolist() == [42, 2]  # 5 + 16 by ms 7
    # the deadline: no column at theta by the end of the decision window (theta out of reach), the leader wins
    r = _race_runner(race_theta=1e9)
    r.cnt[0], r.pend[0], r.timer[:] = [0.0, 1000.0, 0.0], -1, r.cfg.freeze - ex.MS
    r.wspk[0] = 7.0
    st = r.rng.bit_generator.state
    r.tick()
    assert r.action[0] == 1 and r.pend[0] == -1 and not r.cnt.any() and r.rdec[1] == 1 and r.rdec[2] == 0
    assert r.ebnd[1] == 1 and r.ebnd[0] >= 7 and r.wspk[0] == 0  # the window's spikes (7 carried in plus the tick's)
    # I posts (LunarLander: n_ai = n_ae) are not tallied: a raster of I spikes only moves no count
    q = ex.CKRunner(ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True,
                              sgd=True, adam=False, heads=1, readout="race", race_beta=4.0))
    ipost = np.flatnonzero((q.lay.Ma < 0).any(0))  # the columns' I posts
    assert len(ipost) == q.K * q.cfg.n_ae and np.isin(ipost, q.apost).all()
    Xi = np.zeros((L, 1, q.lay.n_post), bool)
    Xi[:, 0, ipost] = True  # every I post spikes every ms
    q.race_block(Xi)
    assert not q.cnt.any() and q.pend[0] == -1 and q.wspk[0] == L * len(ipost)
    # ties: one draw among the tied columns; no draw without a tie
    r = _race_runner()
    st = r.rng.bit_generator.state
    assert r._leader(np.array([3.0, 7.0, 7.0])) in (1, 2) and r.rdec[2] == 1
    ref = np.random.default_rng(0)
    ref.bit_generator.state = st
    ref.choice(np.array([1, 2]))
    assert r.rng.bit_generator.state == ref.bit_generator.state
    st = r.rng.bit_generator.state
    assert r._leader(np.array([3.0, 7.0, 2.0])) == 1 and r.rng.bit_generator.state == st


def test_race_theta_rule():
    """theta = round(n_ae r0 freeze): Acrobot 20 x 0.04 x 20 = 16, LunarLander 32 x 0.04 x 40 = 51; --race-theta
    overrides."""
    assert ex.race_theta(ex.preset("Acrobot", rate_target=40.0)) == 16
    assert ex.race_theta(ex.preset("LunarLander", rate_target=40.0)) == 51
    assert ex.race_theta(ex.preset("CartPole", rate_target=40.0, race_theta=9)) == 9


def test_race_gradient():
    """The per-column factor (race_factor) times (A - p~) and times g(p~) is the exact gradient of log p~_a and of H
    with respect to the column rates (central differences; alpha_a / n_ae absorbed into eta as CK's rule), and the
    common mode sum_j r_j dH / dr_j is zero (H is invariant to a common scaling of the rates above the floor)."""
    r = _race_runner()
    K, cfg, rng = r.K, r.cfg, np.random.default_rng(3)
    assert r.eps_r == pytest.approx(1 / (cfg.tau_n * cfg.n_ae))
    for _ in range(50):
        rk = rng.uniform(0.005, 0.1, K)
        p, a = r.surrogate(rk), int(rng.integers(K))
        fr = r.race_factor(rk[None])[0]
        claim = fr * (np.eye(K)[a] - p) * cfg.alpha_a
        claimH = fr * ex.td.entropy_coef(p) * cfg.alpha_a

        def f(x, a=a):
            q = r.surrogate(x)
            return np.log(q[a]), -(q * np.log(q)).sum()
        for j in range(K):
            d = 1e-6 * np.eye(K)[j]
            (lp1, H1), (lp0, H0) = f(rk + d), f(rk - d)
            assert claim[j] == pytest.approx((lp1 - lp0) / 2e-6, rel=1e-4, abs=1e-6)
            assert claimH[j] == pytest.approx((H1 - H0) / 2e-6, rel=1e-4, abs=1e-6)
        assert abs((rk * claimH).sum()) < 1e-9
    assert np.allclose(r.surrogate(np.full(K, 1e-9)), 1 / K) and not r.race_factor(np.full((1, K), 1e-9)).any()


def test_race_closed_loop():
    """readout race in closed loop: Acrobot (one copy) and LunarLander (4 copies, rec cols) give finite W, R and p; the
    result carries the energy monitor and the decision counts; the softmax path still reports the energy."""
    r = _race_runner()
    for _ in range(100):
        r.tick()
    assert r.frames > 0 and np.isfinite(r.W).all() and np.isfinite(r.p).all() and r.rdec[:2].sum() > 0
    out = ex.result_ck(r.cfg, r, 0.0, None, 0)
    assert out["actor_spikes_per_decision"] > 0 and out["race_decisions"]["theta"] == 16
    assert out["race_decisions"]["at_bound"] + out["race_decisions"]["at_deadline"] == r.rdec[:2].sum()
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0, heads=4, readout="race", race_beta=4.0)
    r = ex.CKRunner(cfg)
    assert r.theta == 51
    for _ in range(60):
        r.tick()
    assert np.isfinite(r.W).all() and np.isfinite(r.hr.R).all() and np.isfinite(r.p).all()
    s = ex.CKRunner(dataclasses.replace(cfg, readout="softmax", race_beta=0.0, rec="none"))
    for _ in range(10):
        s.tick()
    assert "race_decisions" not in ex.result_ck(s.cfg, s, 0.0, None, 0) and s.espk[0] > 0 and s.stats[6] > 0


def test_calibrate_race():
    """calibrate_race on Acrobot (the contrast probe): a finite beta strictly inside the bracket, the surrogate at beta
    reproducing column 0's realised win rate within 0.1 at every contrast, the record's diagnostics."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, readout="race")
    beta, rec = ex.calibrate_race(cfg)
    assert np.isfinite(beta) and 2 ** -3 * 1.05 < beta < 2 ** 6 / 1.05 and rec["beta"] == beta
    assert rec["deltas"] == list(ex.RACE_DELTAS) and rec["n_decisions"] == 8 * len(ex.RACE_DELTAS) * 25
    w, ws = np.array(rec["win0"]), np.array(rec["win0_surrogate"])
    assert np.abs(w - ws).max() < 0.1 and w[-1] > w[0]  # the contrast moves column 0's win rate
    assert rec["loglik"] < 0 and 0 < rec["H_race"] < np.log(3) + 0.2 and 0 < rec["H_surrogate"] <= np.log(3) + 1e-9
    assert rec["beta_poisson"] == pytest.approx(1.6 * np.sqrt(8))


def test_race_tokens_and_defaults():
    d = ex.Config()
    assert d.readout == "softmax" and d.race_theta == 0.0 and d.race_beta == 0.0
    cfg = ex.preset("Acrobot", engine="nlif", rate_target=40.0, readout="race")
    assert ex.tokens(cfg)[-1] == "race" and "race" not in ex.tokens(ex.preset("Acrobot", engine="nlif"))
    assert ex.tokens(dataclasses.replace(cfg, race_theta=12.0, race_beta=2.5))[-3:] == ["race", "th12", "rb2.5"]
    with pytest.raises(AssertionError):
        ex.CKRunner(dataclasses.replace(cfg, s_in=0.16, norm_logits=True))
    with pytest.raises(AssertionError):
        ex.CKRunner(dataclasses.replace(cfg, s_in=0.16, rate_target=0.0))


def test_rate_cost_term(monkeypatch):
    """CK's target-rate term (016 §18 fix b, --rate-cost): per ms the copy's actor-E mean rate against rho_hat
    (rate_target by default) enters kap on the E actor rows only, the same value on every E row of a copy; it draws
    nothing (same raster at the same seed) and with c_t = 0 kap is what it was."""
    cap, orig = [], ex.td.block_learn_ck_fast

    def spy(z, q, P, X, d, c, kap, *a):
        cap.append((X.copy(), np.array(kap)))
        return orig(z, q, P, X, d, c, kap, *a)
    monkeypatch.setattr(ex.td, "block_learn_ck_fast", spy)
    runs = {}
    mk = lambda **kw: ex.CKRunner(ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20,  # noqa: E731
                                            fast=True, sgd=True, adam=False, **kw))
    for key, kw in (("off", {}), ("lo", dict(c_t=1.0)), ("hi", dict(c_t=1.0, rho_hat=10.0))):
        q = mk(**kw)
        assert q.rho_hat == (10.0 if key == "hi" else 0.04) and q.c_t == (0.0 if key == "off" else 1.0)
        assert ex.CKRunner(q.cfg, frozen=True).c_t == 0.0  # the calibration probes measure the backbone
        cap.clear()
        q.tick()
        runs[key] = cap[0]
    (X0, k0), (X1, k1), (X2, k2) = runs["off"], runs["lo"], runs["hi"]
    assert np.array_equal(X0, X1) and np.array_equal(X0, X2) and k0.ndim == 3 and k1.shape == k0.shape
    ae = q.lay.ae
    d1, d2 = k1 - k0, k2 - k0
    assert np.allclose(d1[..., ae], d1[..., ae][..., :1]) and np.allclose(d2[..., ae], d2[..., ae][..., :1])
    rbar = 0.04 - d1[..., ae][..., 0]  # c_t = 1: d = -(rbar - rho_hat)
    assert (rbar >= 0).all() and rbar.max() > 0 and np.allclose(d2[..., ae][..., 0] - d1[..., ae][..., 0], 10.0 - 0.04)
    assert "rc0.0005" in ex.name(dataclasses.replace(q.cfg, c_t=5e-4)) and "rc" not in ex.name(q.cfg.__class__(**{
        **dataclasses.asdict(q.cfg), "c_t": 0.0}))
    # I posts (LunarLander) get no term
    cap.clear()
    q = ex.CKRunner(ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True,
                              sgd=True, adam=False, heads=1, c_t=1.0))
    q.tick()
    kl = cap[0][1]
    ai = q.lay.sign[q.lay.n_crit:] < 0
    assert ai.any() and np.allclose(kl[..., ai], kl[..., ai][..., :1]) and not np.allclose(kl[..., ai][..., 0],
                                                                                            kl[..., ~ai][..., 0])


def test_race_reprobe(monkeypatch, tmp_path):
    """--race-reprobe (016 §18 fix a): default 0 and token rp<N>; reprobe_race on a frozen runner returns a record with
    race_cal's keys plus episode and a finite beta, the probe draws nothing from the runner's rng and nrng and leaves
    its W, membranes and R untouched, two re-probes at the same state and episode agree exactly (own stream), setting
    the runner's beta changes surrogate() and race_factor(), and a runner's streams after k windows are the same with
    and without a re-probe between them. calibrate_race's init probe is byte-identical to the unshared body."""
    d = ex.Config()
    assert d.race_reprobe == 0 and "rp" not in "".join(ex.tokens(ex.preset("Acrobot", engine="nlif",
                                                                               rate_target=40.0, readout="race")))
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, readout="race", race_beta=6.0, race_reprobe=50)
    assert ex.tokens(cfg)[-2:] == ["rb6", "rp50"]
    cfg = dataclasses.replace(cfg, rec="cols", eta_rec=1.0, lam_rec=1.0)
    r = ex.CKRunner(cfg, frozen=True)
    for _ in range(30):
        r.tick()
    st, nst, W, V, R = (r.rng.bit_generator.state, r.nrng.bit_generator.state, r.W.copy(), r.lst["V"].copy(),
                        r.hr.R.copy())
    rec = ex.reprobe_race(r, 30)
    assert r.rng.bit_generator.state == st and r.nrng.bit_generator.state == nst  # own stream
    assert np.array_equal(r.W, W) and np.array_equal(r.lst["V"], V) and np.array_equal(r.hr.R, R)
    keys = {"theta", "beta", "loglik", "deltas", "win0", "win0_surrogate", "n_decisions", "H_race", "H_surrogate",
            "beta_poisson"}
    assert set(rec) == keys | {"episode"} and rec["episode"] == 30 and rec["theta"] == 16
    assert np.isfinite(rec["beta"]) and 2 ** -3 < rec["beta"] < 2 ** 6 and 0 <= ex.race_fit(rec) <= 1
    assert ex.reprobe_race(r, 30) == rec and ex.reprobe_race(r, 31)["beta"] != rec["beta"]
    rk = np.array([[0.03, 0.05, 0.04]])
    p0, f0 = r.surrogate(rk), r.race_factor(rk)
    r.beta = 2 * rec["beta"]
    assert not np.allclose(r.surrogate(rk), p0) and np.allclose(r.race_factor(rk), 2 * rec["beta"] / 6.0 * f0)
    r.race_cal_history.append(rec)  # the checkpoint carries the current beta and the history
    r.save(tmp_path / "c.npz", 0.0, 0)
    q = ex.CKRunner(cfg, frozen=True)
    q.load(tmp_path / "c.npz")
    assert q.beta == 2 * rec["beta"] and q.race_cal_history == [rec]
    assert "race_cal_history" in ex.result_ck(cfg, q, 0.0, None, 0) and ex.result_ck(
        dataclasses.replace(cfg, race_reprobe=0), q, 0.0, None, 0).get("race_cal_history") is None
    # the run's streams: k windows, a re-probe, k windows, against the same without the re-probe
    a, b = ex.CKRunner(cfg, frozen=True), ex.CKRunner(cfg, frozen=True)
    for _ in range(20):
        a.tick()
        b.tick()
    a.beta = ex.reprobe_race(a, 20)["beta"]
    for _ in range(20):
        a.tick()
        b.tick()
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert a.nrng.bit_generator.state == b.nrng.bit_generator.state and np.array_equal(a.W, b.W)
    # calibrate_race builds the (seed, CAL_SEED) runner and hands it to the shared body unchanged
    seen = []
    orig = ex._race_probe

    def spy(q, n_states, rtol):
        seen.append((q.rng.bit_generator.state, q.cfg.race_beta, q.cfg.block, q.frozen))
        return orig(q, n_states, rtol)
    monkeypatch.setattr(calibration, "_race_probe", spy)  # where calibrate_race looks it up
    base = dataclasses.replace(cfg, rec="none", race_beta=0.0, race_reprobe=0)
    ex.calibrate_race(base)
    ref = ex.CKRunner(dataclasses.replace(base, adam=False, sgd=False, block=ex.MS), rng_seed=ex.CAL_SEED, frozen=True)
    assert seen == [(ref.rng.bit_generator.state, 0.0, ex.MS, True)]


_SC = dict(seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True, adam=False, c_t=5e-4)
# the default path's fingerprints (test_credit_default_path) at commit 4a74319, before --credit existed: the learner's
# inputs (raster, P, delta, c, kap, signs) over 20 windows, W, and the rng / nrng states. Regenerate (print the
# fingerprints) only for an intended change of the race or softmax path, and say so in the CHANGELOG.
_LOCK = {"race": {"inputs": "c040f0a8b244eb50", "W": "0e3a63e3f786d10c", "rng": 176107796947934727478485008267928337156,
                  "nrng": 203759873000756180637601557732678366663, "n": 20},
         "softmax": {"inputs": "fd4ebd1b9474826f", "W": "3b94ac7a1ac18b07",
                     "rng": 157346277774273280172225960227273939189, "nrng": 203759873000756180637601557732678366663,
                     "n": 20}}


def _spy_fingerprint(monkeypatch, cfg, ticks=20):
    cap, orig = [], ex.td.block_learn_ck_fast

    def spy(z, q, P, X, d, c, kap, lay, *a):
        cap.append(hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(P).tobytes()
                                  + np.ascontiguousarray(d).tobytes() + np.ascontiguousarray(c).tobytes()
                                  + np.ascontiguousarray(kap).tobytes() + lay.sign.tobytes()).hexdigest())
        return orig(z, q, P, X, d, c, kap, lay, *a)
    monkeypatch.setattr(ex.td, "block_learn_ck_fast", spy)
    r = ex.CKRunner(cfg)
    for _ in range(ticks):
        r.tick()
    return dict(inputs=hashlib.sha256("".join(cap).encode()).hexdigest()[:16],
                W=hashlib.sha256(r.W.tobytes()).hexdigest()[:16], rng=r.rng.bit_generator.state["state"]["state"],
                nrng=r.nrng.bit_generator.state["state"]["state"], n=len(cap))


def test_post_scores():
    """lif.post_scores on a frozen random-input run: Y and h are 0 exactly on the refractory steps (the NaN mask of M,
    nref steps after each spike), the hazard score's mean over active steps is 0 within 4 s.e. (both kernels), the
    noise score's silent-step value is g / sig_n for Gaussian noise, and the self-check (finite differences of the
    one-neuron toy) passes at a reduced size."""
    for kern in ("nbf", "nb"):
        p = dataclasses.replace(lif.LIFParams(), **lif.DT1R, kernel=kern)
        rng = np.random.default_rng(5)
        k, H, N = 1500, 6, 10
        st = lif.init_state(N, H, p)
        st["V"] = np.full((H, N), p.mu_bg)
        Ix = (1 - p.a) * rng.uniform(0.4, 1.1, N) + rng.normal(0, 0.02, (k, H, N))
        S, _, _, M, GN = lif.simulate_ext(Ix, np.zeros((0, 0)), [], st, rng, p, score=True)
        nan = np.isnan(M)
        ref = np.zeros_like(S)  # the refractory steps: nref after each spike
        for s in range(1, p.nref + 1):
            ref[s:] |= S[:-s]
        assert np.array_equal(nan, ref) and nan.any() and not nan[S].any()
        for kind in ("hazard", "noise"):
            Y, h = lif.post_scores(S, M, GN, p, kind)
            assert (Y[nan] == 0).all() and (h[nan] == 0).all() and np.isfinite(Y).all() and (0 <= h[~nan]).all()
            assert (h[~nan] <= 1).all() and h[S].min() > 0 and (Y[S] > 0).all()
        Y, h = lif.post_scores(S, M, GN, p, "hazard")
        se = Y[~nan].std() / np.sqrt((~nan).sum())
        assert abs(Y[~nan].mean()) < 4 * se and h.sum() == pytest.approx(S.sum(), rel=0.1)
        assert (Y[~nan] * (1 - 2 * S[~nan]) <= 0).all()  # sign: positive on spikes, negative on silence
        if kern == "nb":
            Yn, _ = lif.post_scores(S, M, GN, p, "noise")
            sil = ~nan & ~S
            assert np.allclose(Yn[sil], GN[sil] / (p.sigma_V * np.sqrt(1 - p.a ** 2)))
    out = lif._score_check(H=40000, verbose=False)
    assert abs(out["est"] / out["fd"] - 1) < 0.05 and out["psi_max"] > 10


def test_score_eligibility_rule():
    """The rule the code applies (tdstdp self-check (h4)): block_learn_ck_fast driven with Y and x(t - 1) as learn_block
    drives it gives, per input, d E[count] / dW within 5% of central finite differences (Appendix A)."""
    r = ex.td.check_score_eligibility(H=100000, seed=3)
    assert (np.abs(r["ratio"] - 1) < 0.05).all() and r["count"] > 0.3, r


def test_credit_default_path(monkeypatch):
    """credit='surrogate' (the default) is today's path: the learner's inputs over 20 windows, W and the rng / nrng
    states of a race runner (with the rate term) and of a softmax runner are the fingerprints recorded before --credit
    existed (_LOCK)."""
    assert ex.Config().credit == "surrogate"
    got = {"race": _spy_fingerprint(monkeypatch, ex.preset("Acrobot", readout="race", race_beta=6.0, **_SC)),
           "softmax": _spy_fingerprint(monkeypatch, ex.preset("Acrobot", **_SC))}
    assert got == _LOCK, got


def test_credit_live_mask():
    """race_block records, per copy, whether a decision was pending before the block and the bound ms within it, for
    any block size; live_mask counts the ms up to and including the bound, the whole block without a bound, nothing
    for a copy decided in an earlier block; pend persists until the resample."""
    r = ex.CKRunner(ex.preset("LunarLander", heads=3, readout="race", race_beta=4.0, credit="noise",
                              **dict(_SC, s_in=0.1)))
    nc, n_ae, L = r.lay.n_crit, r.cfg.n_ae, 5
    X = np.zeros((L, 3, r.lay.n_post), bool)
    X[2:, 0, nc:nc + n_ae] = True  # copy 0: column 0 at 32 per ms from ms 2 (theta 51: bound at ms 3)
    r.pend[1] = 2  # copy 1: decided in an earlier block
    r.race_block(X)
    assert r.was.tolist() == [True, False, True] and r.bnd.tolist() == [3, -1, -1] and r.pend.tolist() == [0, 2, -1]
    m = r.live_mask(L)
    assert m.shape == (L, 3) and m[:, 0].tolist() == [True] * 4 + [False] and not m[:, 1].any() and m[:, 2].all()
    r.race_block(np.zeros((L, 3, r.lay.n_post), bool))  # the next block: copy 0's decision persists
    assert r.was.tolist() == [False, False, True] and not r.live_mask(L)[:, :2].any() and r.live_mask(L)[:, 2].all()


def test_credit_score_run(monkeypatch):
    """A 20-window credit='noise' run against the surrogate run on a held observation (LunarLander, I posts present):
    the critic rows of W are bit-identical, the actor I rows stay at their init under both credits (the surrogate's
    credit goes to the E rows only, 016 §21; c_e = 0 and the rate term is E-only here), the actor E rows differ; the
    learner is called twice per block, the critic rows with (X, P) as before and the actor rows with Y (float, 0
    after each copy's bound ms and on I posts) paired with x(t - 1) (a 0/1 input, not the P trace), c = 1 on E
    posts, 0 on I posts, no sign and kap = 0; the rank-1 rate term on a block equals -c_t sum_t (rho_bar - rho_hat)
    rho_i P_j; the checkpoint carries the bin monitors. (Tie draws are pinned: they come from the main rng.)"""
    cfg = ex.preset("LunarLander", heads=2, readout="race", race_beta=4.0, c_e=0.0, **dict(_SC, s_in=0.1))
    monkeypatch.setattr(ex.CKRunner, "_leader", lambda self, cnt: int(np.argmax(cnt)))  # a tie's draw would shift
    runs, orig = {}, ex.td.block_learn_ck_fast  # the input stream of one run and not the other
    for credit in ("surrogate", "noise"):
        cap = []

        def spy(z, q, P, X, d, c, kap, lay, *a, cap=cap, orig=orig):
            cap.append((X.copy(), np.array(c), np.array(kap), lay.sign.copy(), runs["r"].was.copy(),
                        runs["r"].bnd.copy(), P.copy(), lay.n_crit))
            return orig(z, q, P, X, d, c, kap, lay, *a)
        monkeypatch.setattr(ex.td, "block_learn_ck_fast", spy)
        r = runs["r"] = ex.CKRunner(dataclasses.replace(cfg, credit=credit))
        r.hold = True
        W0 = r.W.copy()
        for _ in range(20):
            r.tick()
        runs[credit] = (W0, r.W.copy(), cap, r)
    nc, ap, cE = runs["r"].lay.n_crit, runs["r"].apost, runs["r"].cE > 0
    (W0s, Ws, caps, rs), (W0n, Wn, capn, rn) = runs["surrogate"], runs["noise"]
    assert np.array_equal(W0s, W0n) and np.array_equal(Ws[:nc], Wn[:nc]) and not np.array_equal(Ws[nc:], Wn[nc:])
    assert np.array_equal(Wn[ap][~cE], W0n[ap][~cE]) and np.array_equal(Ws[ap][~cE], W0s[ap][~cE])
    assert (Wn[ap][cE] != W0n[ap][cE]).any() and (Wn[ap][cE] != Ws[ap][cE]).any()
    assert len(capn) == 40 and len(caps) == 20 and caps[0][0].dtype == bool
    capc, capa = capn[0::2], capn[1::2]  # per block: the critic call, then the actor call
    ia = ap - nc
    for (Xc, cc, _kc, sc, _, _, Pc, ncc), (X, c, kap, sign, was, bnd, P, nca) in zip(capc, capa):
        L = len(X)
        assert ncc == nc and nca == 0 and Xc.dtype == bool and Xc.shape[-1] == nc and cc.shape[-1] == 0
        assert np.array_equal(sc, rs.lay.sign[:nc]) and X.dtype == float and (X[..., ia][..., ~cE] == 0).all()
        assert (c[..., :len(cE)] == cE).all() and not kap.any() and (sign == 1).all() and X.shape[-1] == len(ia)
        assert np.isin(P, (0.0, 1.0)).all() and P.any() and not np.isin(Pc, (0.0, 1.0)).all()  # x(t - 1), not P
        assert np.allclose(P[2:], Pc[1:-1] - np.exp(-1 / cfg.tau_p) * Pc[:-2])  # P(t) = a P(t-1) + x(t): the
        # actor call's P argument at ms t is x(t - 1) = P(t - 1) - a P(t - 2)
        live = was[None] & ((bnd[None] < 0) | (np.arange(L)[:, None] <= bnd[None]))
        assert not X[..., ia][~live].any() and (X[..., ia][live][:, cE] != 0).any()
    for _X, c, kap, sign, _, _, _, ncs in caps[:3]:
        assert np.array_equal(sign, rs.lay.sign) and (c != 1).any() and kap.any() and ncs == nc  # as before
    assert rn.rdec[0] > 0 and rn.c_e == 0.0 and rs.c_e == 0.0
    # the rank-1 rate term: one window at c_t = 1 against c_t = 0 from the same state, R from td.rates
    Rc = []
    monkeypatch.setattr(ex.td, "rates", lambda *a, _o=ex.td.rates: Rc.append(_o(*a)) or Rc[-1])
    Ws = {}
    for ct in (0.0, 1.0):
        q = ex.CKRunner(dataclasses.replace(cfg, credit="noise", c_t=ct))
        q.hold = True
        Rc.clear()
        q.tick()
        Ws[ct] = q.W.copy()
    dW = Ws[1.0] - Ws[0.0]
    R = Rc[-1]  # the c_t = 1 run's rates (same draws as c_t = 0: identical rasters)
    q = ex.CKRunner(dataclasses.replace(cfg, credit="noise", c_t=1.0))  # the same first window's inputs
    q.hold = True
    f = q.phi(q.obs)
    x, _ = ex.td.draw(f, 20, q.lay.n_post, q.rng)
    P = ex.td.pre_trace(q.P, x, cfg.tau_p)
    rdev = R[..., nc:][..., q.lay.ae].mean(-1) - q.rho_hat
    exp = -cfg.eta_a / q.H * np.einsum("lhi,lhj->ij", rdev[..., None] * R[..., ap] * q.cE, P)
    assert np.allclose(dW[ap], exp, atol=1e-12) and np.abs(exp).max() > 0 and not dW[:nc].any()
    assert not dW[ap][~cE].any()
    # monitors and the checkpoint
    row = rn.score_monitors(7)
    assert row["score_abs_mean"] > 0 and 0 <= row["score_clip_frac"] < 0.05 and 0 <= row["race_entropy"] <= np.log(4)
    assert rn.slog["episode"] == [7] and ex.result_ck(rn.cfg, rn, 0.0, None, 0)["race_log"] == rn.slog
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rn.save(d + "/c.npz", 0.0, 0)
        q = ex.CKRunner(rn.cfg)
        q.load(d + "/c.npz")
        assert q.slog == rn.slog


def test_credit_rec_hs(monkeypatch):
    """R's credit under the score: hs per window is the column's summed Y over its E posts up to the bound / n_ae
    (latents 0), used at the next window (credit pairs it with the previous window's counter), and credit receives
    G = 1 / tau_eff in place of the kernel's gain; calibrate_rec runs the frozen runner with the arm's credit."""
    cfg = ex.preset("Acrobot", readout="race", race_beta=6.0, credit="hazard", rec="cols", eta_rec=1.0, lam_rec=1.0,
                    **_SC)
    r = ex.CKRunner(cfg)
    seen, cap = [], []
    orig_c, orig_l = r.hr.credit, ex.td.block_learn_ck_fast
    monkeypatch.setattr(r.hr, "credit", lambda s, G, n, decay: seen.append((s.copy(), G.copy())) or orig_c(s, G, n,
                                                                                                             decay))
    monkeypatch.setattr(ex.td, "block_learn_ck_fast", lambda z, q, P, X, *a: cap.append(X.copy()) or orig_l(z, q, P,
                                                                                                             X, *a))
    for _ in range(3):
        r.tick()
    K, ia, n_ae = r.K, r.apost - r.lay.n_crit, cfg.n_ae
    cap = [X for X in cap if X.dtype == float]  # the actor calls (the critic calls pass the bool raster)
    assert len(seen) == 3 and len(cap) == 3 and np.allclose(seen[0][1], 1 / r.hr.p.tau_eff) and not seen[0][0].any()
    for t in (1, 2):  # window t's credit uses window t - 1's summed scores
        hs = np.einsum("lhi,ki->hk", cap[t - 1][..., ia], r.Mcnt) / n_ae
        assert np.allclose(seen[t][0][:, :K], hs) and hs.any()
    assert np.allclose(r.hs[:, :K], np.einsum("lhi,ki->hk", cap[2][..., ia], r.Mcnt) / n_ae)
    s = ex.CKRunner(dataclasses.replace(cfg, credit="surrogate"))  # the surrogate's hs as before: from the window's
    s.tick()  # (action, p~, rates) at the start of the next tick
    a0, p0, rk0 = s.action.copy(), s.p.copy(), s.prk.copy()
    s.tick()
    assert np.allclose(s.hs[:, :K], cfg.alpha_a * (np.eye(K)[a0] - p0) * s.race_factor(rk0) / n_ae)
    eta, lam, rec = ex.calibrate_rec(dataclasses.replace(cfg, eta_rec=0.0, lam_rec=0.0))
    assert np.isfinite([eta, lam]).all() and eta > 0 and rec["rms_u"] > 0


def test_credit_tokens_and_asserts():
    cfg = ex.preset("Acrobot", readout="race", race_beta=6.0, **_SC)
    assert ex.tokens(dataclasses.replace(cfg, credit="hazard"))[-1] == "sch"
    assert ex.tokens(dataclasses.replace(cfg, credit="noise", rec="cols"))[-4:] == ["rb6", "scn", "cols", "xr0.3"]
    assert not {"sch", "scn"} & set(ex.tokens(cfg)) and ex.name(dataclasses.replace(cfg, credit="noise")).endswith(
        "_race_rb6_scn_s0")
    ex.CKRunner(dataclasses.replace(cfg, credit="noise"))
    for bad in (dict(credit="noise", readout="softmax", race_beta=0.0), dict(credit="hazard", sgd=False),
                dict(credit="noise", race_reprobe=50), dict(credit="score")):
        with pytest.raises((AssertionError, ValueError)):
            ex.CKRunner(dataclasses.replace(cfg, **bad))
    ex.CKRunner(dataclasses.replace(cfg, credit="hazard", sgd=False), frozen=True)  # the calibration probes
    with pytest.raises(AssertionError):
        ex.CKRunner(dataclasses.replace(cfg, credit="noise", engine="lif", rate_target=0.0, s_in=0.0))


def test_eta_cal_cartpole_uses_adam_steps(monkeypatch):
    """--eta-cal on CartPole calibrates against the authors' Adam steps (adam_step_c/a, the LunarLander config's), not
    the preset's SGD rates, which gave a 68-280x step (016 §20); presets whose eta are Adam steps are unchanged."""
    monkeypatch.setattr(calibration, "CAL_TICKS", 20)  # where calibrate_ck reads it
    cp = ex.preset("CartPole", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                   adam=False)
    assert (cp.adam_step_c, cp.adam_step_a) == (1.25e-4, 6.25e-5) and (cp.eta_c, cp.eta_a) == (0.005, 0.01)
    eta, rms = ex.calibrate_ck(cp)
    assert np.allclose(eta, np.array([1.25e-4, 6.25e-5]) / rms) and eta[0] < 0.1 and eta[1] < 0.1
    ac = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                   adam=False)
    assert (ac.adam_step_c, ac.adam_step_a) == (0.0, 0.0)
    eta2, rms2 = ex.calibrate_ck(ac)
    assert np.allclose(eta2, np.array([ac.eta_c, ac.eta_a]) / rms2)


def test_beta_c_and_eta_mult_flags():
    """--beta-c overrides the preset's critic bias (token bc), --eta-mult-a multiplies the calibrated actor step (token
    em); both default to the preset / 1 with no token (016 §20 per-task grid)."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False)
    assert cfg.beta_c == -1.2 and cfg.eta_mult_a == 1.0 and "bc" not in ex.name(cfg) and "em" not in ex.name(cfg)
    c2 = dataclasses.replace(cfg, beta_c=-1.5, eta_mult_a=3.0)
    assert "bc-1.5" in ex.name(c2) and "em3" in ex.name(c2)
    a = ex.build_parser().parse_args(["--task", "Acrobot", "--beta-c", "-1.5", "--eta-mult-a", "3"]) \
        if hasattr(ex, "build_parser") else None
    if a is not None:
        assert a.beta_c == -1.5 and a.eta_mult_a == 3.0


_ORIG_RATES = ex.td.rates


def _craft_rates(monkeypatch, nc, apost, Mcnt, low_col, lo=0.01, hi=0.04):
    """td.rates returns R with column low_col's E posts at `lo` per ms and every other actor E post at `hi` (I posts and
    the critic as computed), so the rate floor's term is known: c_t (rho_min - lo)_+ on low_col's E rows only."""
    orig = ex.td.rates
    cap = []

    def rates(*a):
        R = orig(*a).copy()
        E = Mcnt.sum(0) > 0
        R[..., apost[E]] = hi
        R[..., apost[Mcnt[low_col] > 0]] = lo
        cap.append(R)
        return R
    monkeypatch.setattr(ex.td, "rates", rates)
    return cap


def test_rate_floor_term(monkeypatch):
    """The one-sided per-column rate floor (016 §20, --rate-floor): with rate_floor 0.5 at rate_target 40 Hz (rho_min
    0.02 per ms) and column 0's E rate crafted at 0.01 (the others at 0.04), the surrogate path's kap gains +c_t (0.02
    - 0.01) on column 0's E rows only (0 on the other columns, on I posts and on the mean term, which is the same in
    both runs); the rasters are unchanged (no draw) and the frozen probes have it off."""
    cap, orig = [], ex.td.block_learn_ck_fast

    def spy(z, q, P, X, d, c, kap, *a):
        cap.append((X.copy(), np.array(kap)))
        return orig(z, q, P, X, d, c, kap, *a)
    monkeypatch.setattr(ex.td, "block_learn_ck_fast", spy)
    for task, s_in in (("Acrobot", 0.16), ("LunarLander", 0.1)):
        cfg = ex.preset(task, seed=0, engine="nlif", s_in=s_in, rate_target=40.0, block=20, fast=True, sgd=True,
                        adam=False, heads=2, c_t=1.0, readout="race", race_beta=6.0)
        r0 = ex.CKRunner(cfg)
        nc, ap, Mcnt = r0.lay.n_crit, r0.apost, r0.Mcnt
        _craft_rates(monkeypatch, nc, ap, Mcnt, 0)
        runs = {}
        for rf in (0.0, 0.5):
            q = ex.CKRunner(dataclasses.replace(cfg, rate_floor=rf))
            assert q.rho_min == rf * 0.04 and q.floor == (rf > 0) and not ex.CKRunner(q.cfg, frozen=True).floor
            cap.clear()
            q.tick()
            runs[rf] = cap[0]
        (X0, k0), (X1, k1) = runs[0.0], runs[0.5]
        assert np.array_equal(X0, X1) and k1.shape == k0.shape
        d = k1 - k0
        exp = np.zeros(d.shape[-1])
        exp[:len(ap)] = (Mcnt[0] > 0) * (0.02 - 0.01)
        assert np.allclose(d, exp) and d.max() == pytest.approx(0.01)
        if task == "LunarLander":  # I posts of column 0 untouched
            assert (Mcnt.sum(0) == 0).any() and not d[..., :len(ap)][..., Mcnt.sum(0) == 0].any()
        monkeypatch.setattr(ex.td, "rates", _ORIG_RATES)  # the next task crafts from the original


def test_rate_floor_score_rank1(monkeypatch):
    """Under a score credit the floor is rank-1 per block: one window at rate_floor 0.5 against 0 from the same state
    (credit noise, c_t 1, rates crafted as test_rate_floor_term, column 0 deprived): dW on the actor rows equals
    eta_a / H sum_t c_t (rho_min - rho_0)_+ rho_i P_j on column 0's E rows and 0 elsewhere; the critic rows are
    unchanged."""
    cfg = ex.preset("LunarLander", heads=2, readout="race", race_beta=4.0, c_e=0.0, credit="noise",
                    **dict(_SC, s_in=0.1, c_t=1.0))
    monkeypatch.setattr(ex.CKRunner, "_leader", lambda self, cnt: int(np.argmax(cnt)))
    r0 = ex.CKRunner(cfg)
    nc, ap, Mcnt = r0.lay.n_crit, r0.apost, r0.Mcnt
    Rc = _craft_rates(monkeypatch, nc, ap, Mcnt, 0)
    Ws = {}
    for rf in (0.0, 0.5):
        q = ex.CKRunner(dataclasses.replace(cfg, rate_floor=rf))
        q.hold = True
        Rc.clear()
        q.tick()
        Ws[rf] = q.W.copy()
    R = Rc[-1]
    dW = Ws[0.5] - Ws[0.0]
    q = ex.CKRunner(dataclasses.replace(cfg, rate_floor=0.5))  # the same first window's inputs
    q.hold = True
    f = q.phi(q.obs)
    x, _ = ex.td.draw(f, 20, q.lay.n_post, q.rng)
    P = ex.td.pre_trace(q.P, x, cfg.tau_p)
    flc = 1.0 * np.maximum(0.02 - R[..., ap] @ Mcnt.T / cfg.n_ae, 0.0)  # L x H x K; column 0: 0.01, others 0
    assert np.allclose(flc[..., 0], 0.01) and not flc[..., 1:].any()
    exp = cfg.eta_a / q.H * np.einsum("lhi,lhj->ij", (flc @ Mcnt) * R[..., ap], P)
    assert np.allclose(dW[ap], exp, atol=1e-12) and np.abs(exp).max() > 0 and not dW[:nc].any()
    assert not dW[ap][Mcnt[0] == 0].any() and (dW[ap][Mcnt[0] > 0] != 0).any()
    monkeypatch.setattr(ex.td, "rates", _ORIG_RATES)
    # the monitors: col_rate_min is the minimum over columns of the bin mean, col_rate_min_ms the mean per-ms minimum
    q = ex.CKRunner(dataclasses.replace(cfg, rate_floor=0.5))
    q.hold = True
    for _ in range(3):
        q.tick()
    row = q.score_monitors(3)
    assert 0 < row["col_rate_min_ms"] <= row["col_rate_min"] < 1e3 * q.rho.max() and q.fsum.sum() == 0
    assert set(q.slog) == set(ex.SLOG_KEYS) and ex.result_ck(q.cfg, q, 0.0, None, 0)["race_log"] == q.slog


def test_rate_floor_tokens_and_asserts():
    """rate_floor 0 is the default (no token; the fingerprints of test_credit_default_path hold); rf<frac> after rc;
    the floor needs c_t > 0 and rate_target; the CLI flag maps to it."""
    assert ex.Config().rate_floor == 0.0
    cfg = ex.preset("Acrobot", readout="race", race_beta=6.0, **_SC)
    assert "rf" not in ex.name(cfg)
    t = ex.tokens(dataclasses.replace(cfg, rate_floor=0.5))
    assert t[t.index("rc0.0005") + 1] == "rf0.5" and ex.name(dataclasses.replace(cfg, rate_floor=0.5, credit="noise")) \
        .endswith("_rc0.0005_rf0.5_race_rb6_scn_s0")
    ex.CKRunner(dataclasses.replace(cfg, rate_floor=0.5))
    for bad in (dict(rate_floor=0.5, c_t=0.0), dict(rate_floor=-0.1),
                dict(rate_floor=0.5, readout="softmax", race_beta=0.0, rate_target=0.0, rho_hat=0.04)):
        with pytest.raises(AssertionError):
            ex.CKRunner(dataclasses.replace(cfg, **bad))
    assert ex.CKRunner(dataclasses.replace(cfg, rate_floor=0.5), frozen=True).floor is False


class _SpyRng:
    """A delegating wrapper of a Generator that counts the policy draws (choice with p=)."""

    def __init__(self, rng):
        self._rng, self.n_policy = rng, 0

    def choice(self, *a, **k):
        self.n_policy += "p" in k
        return self._rng.choice(*a, **k)

    def __getattr__(self, k):
        return getattr(self._rng, k)


def test_freeze_hold(monkeypatch):
    """--freeze MS (016 §21, the decision hold): with freeze 40 and race_theta 16 on Acrobot theta is 16 (the 20-ms
    rule, not the window's 32), the race's deadline is the 40-ms window (a copy with no column at theta takes the
    leader at ms 40, not 20), the action is resampled once per 40 ms while the env steps every 20 ms (two env steps
    per decision, the action held over both), the energy denominator counts 40-ms windows, the softmax path with
    freeze 40 resamples every 40 ms too, the token is fz40 (none at the preset's; LunarLander's preset is 40) and
    the CLI flag maps to Config.freeze."""
    r = _race_runner(freeze=40, race_theta=16.0)
    assert r.cfg.freeze == 40 and r.theta == 16
    assert ex.race_theta(dataclasses.replace(r.cfg, race_theta=0.0)) == 32  # the rule over the full window
    # the deadline at ms 40: theta out of reach, column 1 leads, no decision at ms 20
    r = _race_runner(freeze=40, race_theta=1e9)
    r.cnt[0], r.pend[0], r.timer[:] = [0.0, 1000.0, 0.0], -1, 0
    a0 = int(r.action[0])
    r.tick()
    assert r.timer[0] == 20 and r.pend[0] == -1 and r.rdec[1] == 0 and r.action[0] == a0 and r.cnt[0, 1] >= 1000
    r.tick()
    assert r.timer[0] == 0 and r.action[0] == 1 and r.pend[0] == -1 and r.rdec[1] == 1 and not r.cnt.any()
    assert r.espk[1] == pytest.approx(1.0)  # two 20-ms blocks of one copy: one 40-ms decision window
    # the hold: the env steps every tick, the action is resampled every second tick and held over both steps
    r = _race_runner(freeze=40, race_theta=16.0)
    steps, orig = [], r.envs[0].step
    r.envs[0].step = lambda a: steps.append(int(a)) or orig(a)
    timers, ticks = [], 2 * (ex.WARM_N + 1) + 20
    for _ in range(ticks):
        r.tick()
        timers.append(int(r.timer[0]))
    assert timers == [20, 0] * (ticks // 2) and not r.is_end[0]
    assert len(steps) == ticks - ex.WARM_N  # one env step per tick once live (the first window is live, WARM_N after)
    held = [steps[i] == steps[i + 1] for i in range(len(steps) - 1)]
    assert all(held[(ex.WARM_N + 1) % 2::2]) and r.arep[0, 1] > 0  # pairs aligned to the resample ticks (odd ticks)
    # the softmax path: a policy draw every 40 ms
    s = ex.CKRunner(ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True,
                              sgd=True, adam=False, freeze=40))
    s.rng = _SpyRng(s.rng)
    for _ in range(10):
        s.tick()
    assert s.rng.n_policy == 5 and s.timer[0] == 0 and s.espk[1] == pytest.approx(5.0)
    # tokens and the CLI
    cfg = ex.preset("Acrobot", engine="nlif", rate_target=40.0, readout="race", freeze=40, race_theta=16.0)
    t = ex.tokens(cfg)
    assert t[-3:] == ["fz40", "race", "th16"] and ex.name(cfg).endswith("_fz40_race_th16_s0")
    assert "fz" not in "".join(ex.tokens(ex.preset("Acrobot", engine="nlif", rate_target=40.0, readout="race")))
    assert "fz" not in "".join(ex.tokens(ex.preset("LunarLander"))) and "fz20" in ex.tokens(ex.preset("LunarLander",
                                                                                                        freeze=20))
    got = []
    monkeypatch.setattr(ex, "run", lambda cfg, **kw: got.append(cfg))
    base = ["tdstdp", "--task", "Acrobot", "--seed", "3", "--engine", "nlif", "--rate-target", "40", "--readout",
            "race"]
    monkeypatch.setattr(sys, "argv", base + ["--freeze", "40", "--race-theta", "16"])
    ex.main()
    monkeypatch.setattr(sys, "argv", base)
    ex.main()
    assert (got[0].freeze, got[0].race_theta, got[1].freeze, got[1].race_theta) == (40, 16.0, 20, 0.0)
    assert ex.name(got[0]).endswith("_fz40_race_th16_s3") and "fz" not in ex.name(got[1])


def test_reprobe_fit_gate(monkeypatch, capsys):
    """The re-probe gate (016 §21, reprobe_step, RACE_FIT_GATE 0.05): a record whose fit exceeds the gate leaves beta
    unchanged and is marked accepted False; a record within the gate (or exactly at it) sets beta and carries no mark;
    every record goes to race_cal_history and the result; run_ck's hook is reprobe_step."""
    import inspect
    assert ex.RACE_FIT_GATE == 0.05 and "reprobe_step(r, e)" in inspect.getsource(ex.run_ck)
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, readout="race", race_beta=6.0, race_reprobe=50)
    r = ex.CKRunner(cfg, frozen=True)

    def fake(fit, beta):
        def probe(runner, episode):
            return dict(theta=runner.theta, episode=int(episode), beta=beta, win0=[0.0] * 5,
                        win0_surrogate=[fit, 0.0, 0.0, 0.0, 0.0])
        return probe
    monkeypatch.setattr(calibration, "reprobe_race", fake(0.3, 9.0))
    rec = ex.reprobe_step(r, 50)
    assert r.beta == 6.0 and rec["accepted"] is False and ex.race_fit(rec) == pytest.approx(0.3)
    assert "reprobe ep 50 beta 9 fit 0.300 rejected" in capsys.readouterr().out
    monkeypatch.setattr(calibration, "reprobe_race", fake(0.01, 9.0))
    rec2 = ex.reprobe_step(r, 100)
    assert r.beta == 9.0 and "accepted" not in rec2 and r.race_cal_history == [rec, rec2]
    out = capsys.readouterr().out
    assert "reprobe ep 100 beta 9 fit 0.010" in out and "rejected" not in out
    monkeypatch.setattr(calibration, "reprobe_race", fake(0.05, 3.0))
    assert "accepted" not in ex.reprobe_step(r, 150) and r.beta == 3.0
    hist = ex.result_ck(cfg, r, 0.0, None, 0)["race_cal_history"]
    assert [x.get("accepted", True) for x in hist] == [False, True, True]


def test_action_repeat_monitor():
    """action_repeat_probability (016 §21, race_log): the fraction of resamples at which the decision repeats the
    previous one, per copy averaged over copies in an episode; the held steps between resamples are not counted, so
    the count of resamples in a bin is the number of 40-ms windows under a hold of 2, not the env steps."""
    assert "action_repeat_probability" in ex.SLOG_KEYS
    r = _race_runner(freeze=40, race_theta=16.0)
    for _ in range(2 * (ex.WARM_N + 1)):
        r.tick()
    r.arep[:] = 0
    n, reps = 0, 0
    for _ in range(20):
        prev = int(r.action[0])
        r.tick()
        if r.timer[0] == 0:  # a resample: the decision against the previous one (the held tick between is not counted)
            n, reps = n + 1, reps + (int(r.action[0]) == prev)
    assert r.arep[0].tolist() == [reps, n] and n == 10 and r.code[0] == 0
    rep = reps / n
    row = r.score_monitors(1)
    assert row["action_repeat_probability"] == pytest.approx(rep) and not r.arep.any()
    assert np.isnan(r.score_monitors(2)["action_repeat_probability"])  # an empty bin
    assert set(r.slog) == set(ex.SLOG_KEYS) and len(r.slog["action_repeat_probability"]) == 2


_LL = dict(seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True, adam=False, heads=1)


def test_race_surrogate_e_rate():
    """The surrogate reads what the race tallies (016 §21): on LunarLander (n_ai = 32) prk after a tick is the
    columns' E rate (rho @ Mcnt.T) / n_ae, non-negative and not CK's net rate rho @ Ma.T (which sits near 0, below
    eps_r); calibrate_race's probe fits the race (fit <= RACE_FIT_GATE) with a sharp beta (> 3), where the net-rate
    surrogate (MaE = Ma, today's) gives a floor artefact: a worse fit and a smaller beta. The probe's record carries
    the E rates (rsum / rn >= 0)."""
    cfg = ex.preset("LunarLander", readout="race", race_beta=4.0, c_t=5e-4, **_LL)
    r = ex.CKRunner(cfg)
    assert cfg.n_ai > 0 and (r.MaE >= 0).all() and (r.MaE > 0).sum() == r.K * cfg.n_ae and (r.lay.Ma < 0).any()
    for _ in range(3):
        r.tick()
    e = r.rho[..., r.apost] @ r.Mcnt.T / cfg.n_ae
    assert np.allclose(r.prk, e) and (r.prk >= 0).all() and (r.prk > r.eps_r).all()
    assert not np.allclose(r.prk, r.rho @ r.lay.Ma.T) and np.allclose(r.surrogate(r.prk), r.p)
    assert (r.race_factor(r.prk) > 0).all()  # the credit active on every column
    beta, rec = ex.calibrate_race(dataclasses.replace(cfg, race_beta=0.0))
    fit = ex.race_fit(rec)
    assert fit <= ex.RACE_FIT_GATE and beta > 3 and rec["beta"] == beta
    p = ex.CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=ex.MS, race_beta=0.0), rng_seed=ex.CAL_SEED,
                    frozen=True)
    p.MaE = p.lay.Ma  # today's surrogate of the net rate, on the same probe runner
    beta_net, rec_net = ex._race_probe(p, 8, 2e-3)
    assert ex.race_fit(rec_net) > fit and beta_net < beta and ex.race_fit(rec_net) > ex.RACE_FIT_GATE
    q = ex.CKRunner(dataclasses.replace(cfg, race_beta=beta), rng_seed=ex.CAL_SEED, frozen=True)
    q.probe = q.hold = True
    for _ in range(3):
        q.tick()
    assert q.rn > 0 and (q.rsum / q.rn >= 0).all() and np.allclose(q.rsum / q.rn, q.prk.sum(0) / q.H, atol=1e-2)


def test_race_credit_e_rows(monkeypatch):
    """The policy credit goes to the E rows only under the race (016 §21): the learner's c and kap (the entropy term)
    are zero on the I actor rows of a LunarLander race runner and non-zero on E rows; the softmax path still credits
    the I rows (with the sign flip)."""
    cap, orig = [], ex.td.block_learn_ck_fast

    def spy(z, q, P, X, d, c, kap, *a):
        cap.append((np.array(c), np.array(kap)))
        return orig(z, q, P, X, d, c, kap, *a)
    monkeypatch.setattr(ex.td, "block_learn_ck_fast", spy)
    r = ex.CKRunner(ex.preset("LunarLander", readout="race", race_beta=4.0, **_LL))
    ai, ae = r.lay.sign[r.lay.n_crit:] < 0, r.lay.ae
    assert ai.any() and ai.sum() == ae.sum() and r.cfg.c_e > 0 and r.c_t == 0
    for _ in range(3):
        r.tick()
    c, kap = cap[-1]
    assert not c[..., ai].any() and not kap[..., ai].any() and c[..., ae].any() and kap[..., ae].any()
    assert not np.signbit(c[..., ai]).any() and not np.signbit(kap[..., ai]).any()  # +0.0, not -0.0
    cap.clear()
    s = ex.CKRunner(ex.preset("LunarLander", **_LL))
    for _ in range(3):
        s.tick()
    c, kap = cap[-1]
    assert c[..., ai].any() and kap[..., ai].any()


def test_race_e_rate_invariance_n_ai0():
    """Where n_ai = 0 (CartPole, Acrobot) the fix is a no-op: MaE is Ma itself (bit-identical products, so the race
    studies on those tasks are unchanged; test_credit_default_path locks the Acrobot race fingerprint), the E rate
    equals (R @ Mcnt.T) / n_ae up to rounding, and the E masks (lay.ae, cE) are all ones."""
    for task, s_in in (("Acrobot", 0.16), ("CartPole", 0.16)):
        r = ex.CKRunner(ex.preset(task, readout="race", race_beta=6.0, **dict(_LL, s_in=s_in)))
        assert r.cfg.n_ai == 0 and np.array_equal(r.MaE, r.lay.Ma) and r.MaE.flags.c_contiguous
        assert r.lay.ae.all() and (r.cE == 1).all() and len(r.lay.ae) == len(r.cE)
        R = np.random.default_rng(1).uniform(0, 0.1, (5, 2, r.lay.n_post))
        assert np.array_equal(R @ r.MaE.T, R @ r.lay.Ma.T)
        assert np.allclose(R @ r.MaE.T, R[..., r.apost] @ r.Mcnt.T / r.cfg.n_ae)


def test_save_and_load_ck_model(tmp_path):
    """--save-model writes the final W, the calibrated config, the race beta and (with rec) R; load_ck_model rebuilds a
    frozen runner that holds them and runs the policy (returns appended) without any plasticity."""
    cfg = ex.preset("CartPole", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, readout="race",
                    race_beta=6.0, rec="cols", chi_rec=0.3, lam=0.0, rec_seal=True, eta_rec=0.1, lam_rec=0.0)
    r = ex.CKRunner(cfg)
    for _ in range(30):
        r.tick()
    r.snaps[len(r.returns)] = r.hr.R.copy()
    p = tmp_path / "m.npz"
    ex.save_ck_model(p, cfg, r, eta_sgd=np.array([0.01, 0.02]))
    z = np.load(p, allow_pickle=False)
    assert z["W"].shape == r.W.shape and np.array_equal(z["W"], r.W) and np.array_equal(z["R"], r.hr.R)
    assert z["K"] == 2 and z["n_in"] == r.lay.n_in and float(z["beta"]) == 6.0
    assert z["eta_sgd_mV"].tolist() == [0.01, 0.02]
    q = ex.load_ck_model(p)
    assert q.frozen and np.array_equal(q.W, r.W) and np.array_equal(q.hr.R, r.hr.R) and q.beta == 6.0
    W0 = q.W.copy()
    for _ in range(40):
        q.tick()
    assert np.array_equal(q.W, W0) and np.array_equal(q.hr.R, r.hr.R)  # frozen: no plasticity on W or R


def test_race_fit_gate_option(monkeypatch):
    """--race-fit-gate sets the re-probe acceptance threshold (default 0.05, token fg<v> otherwise; inf accepts every
    re-probe, the behaviour before the gate existed, which models/reproduce.sh uses for race-cartpole)."""
    r = _race_runner()
    assert r.cfg.race_fit_gate == 0.05 and "fg" not in ex.name(r.cfg)
    q = _race_runner(race_fit_gate=float("inf"))
    assert "fginf" in ex.name(q.cfg)
    rec = {"beta": 9.0, "win0": [0.0, 0.5, 1.0], "win0_surrogate": [0.3, 0.5, 0.7], "episode": 50}  # fit 0.3
    monkeypatch.setattr(calibration, "reprobe_race", lambda runner, ep: dict(rec))
    b0 = r.beta
    ex.reprobe_step(r, 50)
    assert r.beta == b0 and r.race_cal_history[-1].get("accepted", True) is False
    ex.reprobe_step(q, 50)
    assert q.beta == 9.0 and q.race_cal_history[-1].get("accepted", True) is True
