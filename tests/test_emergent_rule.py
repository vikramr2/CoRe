"""The emergent-walk Map-STDP rule (basic.Config.rule = 'emergent', derivation Eqs. 8-12) and its run plumbing: flow
emergent == v4 bit for bit, the general formulas reproduce v4 on flow inputs, the exact branching counter (Eq. 11a)
against a loop, the QIF and LIF frames (Eq. 11b: shapes, signs, state isolation; QIF: v4's raster), checkpoint round
trips and resume, --save-model, and experiments.report (tdstdp summaries unchanged, the m3 runner's summary)."""
import csv
import json
import shutil

import gymnasium as gym
import numpy as np
import pytest

from experiments import report
from mapstdp import core
from mapstdp import spiking as sp
from models import basic

V4 = dict(td_norm=0.04, p_cc=0.0, beta_kappa=0.01, beta_max=16.0, eta=5.0, la_beta=0.3)


def train(cfg, episodes=3):
    agent = basic.Agent(cfg)
    env = gym.make(agent.task.env_id)
    return agent, [agent.episode(env) for _ in range(episodes)]


def test_flow_emergent_is_v4_bit_for_bit():
    a, ra = train(basic.Config(**V4))
    b, rb = train(basic.Config(rule="emergent", **V4))
    assert ra == rb and (a.W == b.W).all() and (a.u == b.u).all() and a.beta == b.beta
    ea, eb = a.evaluate(), b.evaluate()
    assert all(ea[k] == eb[k] for k in ea) and eb["L_obs"] == eb["D"]
    assert len(b.ep_log["L_obs"]) == 3 and np.allclose(b.ep_log["causal_share"], 1.0)


def test_emergent_formulas_reproduce_v4_on_flow():
    agent, _ = train(basic.Config(rule="emergent", ebar="obs", **V4), 2)
    W, m, roles = agent.W, agent.m, agent.roles
    for f, v in enumerate(agent.held[:5]):
        pi = agent.walk(v)
        J = W / W.sum(0) * pi
        a = f % len(agent.A)
        dS, de = agent.increments(pi, J, a)[:2]
        p, q, JK, _ = core.module_stats(W, pi, m, agent.K)
        g = core.modulator(p, q, roles, routing="exempt") - core.latent_output_barrier(p, JK, roles, 0.3)
        assert np.abs(dS - (-core.centered(g, W, m) * J)).max() < 1e-12
        assert np.abs(de - core.eligibility_step(0.0, W, J, pi, m, roles, agent.A[a], 1.0, agent.beta)).max() < 1e-12
        assert (core.homeostasis_update_emergent(W, agent.mask, pi, 5.0) ==
                core.homeostasis_update(W, agent.mask, J, 5.0)).all()


def brute(S, ext, W, H, theta=0.5, gain=0.8):
    k, N = len(S), len(W)
    b = gain * W / W.sum(0)
    p = lambda u: min(max(u / (2 * theta), 0.0), 1.0)  # noqa: E731
    kap, c = np.zeros((N, N)), np.zeros((N, N))
    for h in range(H):
        x = S[:, h * N:(h + 1) * N]
        e = ext[:, h * N:(h + 1) * N]
        for t in range(k - 1):
            for i in range(N):
                u = sum(b[i, j] * x[t, j] for j in range(N)) + 2 * theta * e[t + 1, i]
                for j in np.flatnonzero(x[t]):
                    kap[i, j] += x[t + 1, i]
                    c[i, j] += p(u) - p(u - b[i, j])
    return kap, c


@pytest.mark.parametrize("seed", [0, 1])
def test_branching_counter_brute_force(seed):
    W, m, roles, _ = core.build_network((4, 3, 3, 3), "CLAA", p_same=0.8, p_cross=0.2, seed=1)
    v = core.stimulus(0.3, m, roles)
    S0 = sp.run_heads(W, v, 2, 12, 0.8, np.random.default_rng(seed))
    S, Y, X = sp.run_heads(W, v, 2, 12, 0.8, np.random.default_rng(seed), return_drive=True)
    assert (S == S0).all()
    kap, c = sp.branching_counters(S, Y, X, W, 2)
    kb, cb = brute(S, X, W, 2)
    assert np.abs(kap - kb).max() < 1e-12 and np.abs(c - cb).max() < 1e-12 and c.sum() > 0


def test_checkpoint_round_trip(tmp_path):
    cfg = basic.Config("spiking", heads=4, engine="numpy", rule="emergent", obs_tau=5.0, **{**V4, "beta_kappa": 0.0})
    a, _ = train(cfg, 2)
    snaps = {"W_ep0": a.W.astype(np.float32)}
    ck = tmp_path / "r.ckpt.npz"
    basic.save_checkpoint(ck, a, [1.0, 2.0], [{"episode": 0}], snaps, 0, 3.5)
    b = basic.Agent(cfg)
    returns, evals, sn, resumes, wall = basic.load_checkpoint(ck, b)
    assert (returns, evals, resumes, wall) == ([1.0, 2.0], [{"episode": 0}], 0, 3.5)
    assert (sn["W_ep0"] == snaps["W_ep0"]).all()
    assert (a.W == b.W).all() and (a.u == b.u).all() and a.rng.bit_generator.state == b.rng.bit_generator.state
    assert all((x == y).all() for x, y in zip(a.run_obs, b.run_obs))
    for k in basic.Agent.STATE:
        if hasattr(a, k):
            x, y = getattr(a, k), getattr(b, k)
            assert json.dumps(x, default=lambda o: o.tolist()) == json.dumps(y, default=lambda o: o.tolist()), k
    env_a, env_b = gym.make(a.task.env_id), gym.make(b.task.env_id)
    assert a.episode(env_a) == b.episode(env_b) and (a.W == b.W).all()  # continues identically


def test_run_resume_and_save_model(tmp_path, monkeypatch):
    cfg = basic.Config(rule="emergent", **V4)
    monkeypatch.setattr(basic, "LOG_EVERY", 1)
    monkeypatch.setattr(basic, "SAVE_EVERY", 2)
    full = basic.run(cfg, 4, out=tmp_path / "full.json", save_model_path=tmp_path / "full.npz")
    calls, episode = [0], basic.Agent.episode

    def dies(self, env):
        calls[0] += 1
        if calls[0] == 3:
            raise KeyboardInterrupt
        return episode(self, env)
    monkeypatch.setattr(basic.Agent, "episode", dies)
    with pytest.raises(KeyboardInterrupt):
        basic.run(cfg, 4, out=tmp_path / "cut.json", save_model_path=tmp_path / "cut.npz")
    assert (tmp_path / "cut.ckpt.npz").exists()
    monkeypatch.setattr(basic.Agent, "episode", episode)
    cut = basic.run(cfg, 4, out=tmp_path / "cut.json", save_model_path=tmp_path / "cut.npz")
    assert not (tmp_path / "cut.ckpt.npz").exists() and cut["resumes"] == 1 and full["resumes"] == 0
    assert cut["returns"] == full["returns"] and json.dumps(cut["emergent"]) == json.dumps(full["emergent"])
    z, zc = np.load(tmp_path / "full.npz"), np.load(tmp_path / "cut.npz")
    assert sorted(z.files) == sorted(["W_ep0", "W_ep2", "W_ep4", "mask", "m", "roles", "config", "u"])
    assert z["W_ep4"].dtype == np.float32 and json.loads(str(z["config"]))["rule"] == "emergent"
    assert all((z[k] == zc[k]).all() for k in ("W_ep0", "W_ep2", "W_ep4", "u"))


def test_qif_emergent_frame():
    """engine='qif', rule='emergent' (Eq. 11b): the training raster equals rule v4's for the same rng (the counters
    draw nothing), shapes and signs of (S, kappa_raw, c), c only on synapses, the gain estimate updated by training
    frames only, evaluate's frames leave the training state alone, and the gain sums survive a checkpoint."""
    a = basic.Agent(basic.Config("spiking", engine="qif", heads=2))
    b = basic.Agent(basic.Config("spiking", engine="qif", heads=2, rule="emergent"))
    N, k = len(b.W), b.qp.k
    for f, v in enumerate(b.held[:4]):
        Sa = a.spike_frame(v)
        S, kraw, c = b.counter_frame(v, b.rng)
        assert (S == Sa).all() and S.dtype == bool and S.shape == (k, 2 * N)
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
        assert a.qrng.bit_generator.state == b.qrng.bit_generator.state
        assert kraw.shape == c.shape == (N, N) and (kraw >= 0).all() and (kraw == np.round(kraw)).all()
        assert (c >= 0).all() and (c[b.W == 0] == 0).all() and (f == 0 or c.sum() > 0)
    assert b.qgain.n == 4 and np.isfinite(b.qgain.sxr).all()
    st, n, r = tuple(x.copy() for x in b.qstate), b.qgain.n, b.rng.bit_generator.state
    S, kraw, c = b.counter_frame(b.held[5], np.random.default_rng(1))
    assert all((x == y).all() for x, y in zip(st, b.qstate)) and b.qgain.n == n and b.rng.bit_generator.state == r
    pi, ob, act = b.frame(b.held[6])
    assert np.isclose(pi.sum(), 1) and ob.c.shape == (N, N) and 0 <= act < len(b.A)
    arr, meta = b.state()
    fresh = basic.Agent(basic.Config("spiking", engine="qif", heads=2, rule="emergent"))
    fresh.load_state(arr, meta)
    assert fresh.qgain.n == b.qgain.n and (fresh.qgain.sxr == b.qgain.sxr).all()
    assert (fresh.qgain.rbar == b.qgain.rbar).all()


def test_lif_emergent_frame():
    """engine='lif' (rule 'emergent' only): shapes and signs of (S, kappa_raw, c), c only on synapses, an evaluation
    rng leaves the training state, the training rng and the carried gain alone, the gain moments (regression counter)
    persist across episodes and survive a checkpoint round trip, and the episode continues identically after it."""
    with pytest.raises(AssertionError):
        basic.Agent(basic.Config("spiking", engine="lif", heads=2))
    cfg = basic.Config("spiking", engine="lif", heads=2, rule="emergent", lif={"counter": "regression"})
    a = basic.Agent(cfg)
    N, k = len(a.W), a.lp.k
    for v in a.held[:3]:
        S, kraw, c = a.counter_frame(v, a.rng)
        assert S.dtype == bool and S.shape == (k, 2 * N) and kraw.shape == c.shape == (N, N)
        assert (kraw >= 0).all() and (kraw == np.round(kraw)).all() and (c >= 0).all() and (c[a.W == 0] == 0).all()
    assert c.sum() > 0 and a.lcarry[1]["n"] == 3
    st, carry, r = {x: np.copy(a.lstate[x]) for x in ("V", "ref", "z", "tail")}, a.lcarry[1]["r"].copy(), \
        a.rng.bit_generator.state
    a.counter_frame(a.held[5], np.random.default_rng(1))
    assert all((st[x] == a.lstate[x]).all() for x in st) and a.rng.bit_generator.state == r
    assert (a.lcarry[1]["r"] == carry).all() and a.lcarry[1]["n"] == 3
    env = gym.make(a.task.env_id)
    a.episode(env)
    n1 = a.lcarry[1]["n"]
    assert n1 > 3  # the moments carried into the new episode's state
    arr, meta = a.state()
    b = basic.Agent(cfg)
    b.load_state(arr, meta)
    assert b.lcarry[1]["n"] == n1 and all((b.lcarry[1][x] == a.lcarry[1][x]).all() for x in ("m", "r", "mm", "mr"))
    assert a.episode(env) == b.episode(gym.make(b.task.env_id)) and (a.W == b.W).all()


@pytest.mark.parametrize("study", ["tdstdp-baseline", "tdstdp-baseline-flow"])
def test_report_tdstdp_summaries_unchanged(tmp_path, study):
    for st in ("tdstdp-baseline", "tdstdp-baseline-flow"):
        (tmp_path / st).mkdir()
        (tmp_path / st / "runs").symlink_to(report.REPORTS / st / "runs")
        shutil.copy(report.REPORTS / st / "manifest.json", tmp_path / st)
    report.summarize(study, root=tmp_path)
    for f in ("summary.csv", "summary.md"):
        assert (tmp_path / study / f).read_bytes() == (report.REPORTS / study / f).read_bytes(), f


def test_report_m3_summary(tmp_path):
    from scipy.stats import mannwhitneyu
    studies = {"m3fake": dict(title="fake", runner="m3", reference={}, gates={}, pairs=[("on", "off")],
                              groups={"on": dict(task="CartPole", engine="flow", flags=["--lam", "0.05"], seeds=[0, 1]),
                                      "off": dict(task="CartPole", engine="flow", flags=["--lam", "0"], seeds=[0, 1])})}
    ev = lambda L: [{"episode": 0, "D": 6.0, "L_obs": 6.0}, {"episode": 200, "D": L, "L_obs": L}]  # noqa: E731
    runs = {"on": [([500.0] * 200, ev(5.0)), ([480.0] * 200, ev(5.2))],
            "off": [([100.0] * 200, ev(6.0)), ([20.0] * 100 + [480.0] * 100, ev(6.1))]}
    for g, rs in runs.items():
        (tmp_path / "m3fake" / "runs" / g).mkdir(parents=True)
        for i, (ret, e) in enumerate(rs):
            (tmp_path / "m3fake" / "runs" / g / f"s{i}.json").write_text(
                json.dumps({"returns": ret, "evals": e, "ms_per_frame": 1.0, "resumes": i}))
    report.summarize("m3fake", root=tmp_path, studies=studies)
    rows = {r["group"]: r for r in csv.DictReader(open(tmp_path / "m3fake" / "summary.csv"))}
    on, off = rows["on"], rows["off"]
    assert float(on["auc_mean"]) == 490.0 and float(off["auc_mean"]) == 175.0 and on["thr_n"] == "2"
    assert off["thr_n"] == "1" and float(off["T_m_mean"]) == 199.0 and float(on["T_m_mean"]) == 100.0
    assert float(on["L_obs_last_mean"]) == 5.1 and float(on["L_obs_first"]) == 6.0 and on["resumes"] == "1"
    pr = next(csv.DictReader(open(tmp_path / "m3fake" / "pairs.csv")))
    assert float(pr["auc_p"]) == mannwhitneyu([500.0, 480.0], [100.0, 250.0]).pvalue == float(pr["auc_p_holm"])
    assert float(pr["L_obs_last_diff"]) == pytest.approx(-0.95)
    assert "## Pairs" in (tmp_path / "m3fake" / "summary.md").read_text() and pr["family"] == "primary"
    studies["m3fake"]["pairs"] = [("on", "off"), ("off", "on", "other")]  # one pair per family: Holm = raw p
    report.summarize("m3fake", root=tmp_path, studies=studies)
    prs = list(csv.DictReader(open(tmp_path / "m3fake" / "pairs.csv")))
    assert [r["family"] for r in prs] == ["primary", "other"] and all(r["auc_p"] == r["auc_p_holm"] for r in prs)
    cmd = report.command(studies["m3fake"]["groups"]["on"], 3, "o.json", "m3", "m.npz")
    assert cmd[1:] == ["-m", "models.basic", "--task", "CartPole", "--seed", "3", "--backend", "flow", "--lam",
                       "0.05", "--out", "o.json", "--save-model", "m.npz"]


def test_numba_paths_match_numpy(monkeypatch):
    """The compiled counters equal their numpy references: branching c (Eq. 11a; up to summation order) and the LIF
    pair counts (exact integers)."""
    from mapstdp import lif
    W, m, roles, _ = core.build_network((6, 4, 4, 4), "CLAA", p_same=0.6, p_cross=0.1, seed=2)
    S, Y, X = sp.run_heads(W, core.stimulus(0.6, m, roles), 3, 20, 0.8, np.random.default_rng(0), return_drive=True)
    kap, c = sp.branching_counters(S, Y, X, W, 3)
    monkeypatch.setattr(sp, "_counter_nb", None)
    kap0, c0 = sp.branching_counters(S, Y, X, W, 3)
    assert (kap == kap0).all() and np.abs(c - c0).max() < 1e-12 and c.sum() > 0
    full = np.random.default_rng(1).random((90, 3, 11)) < 0.08
    for Ws, start in ((30, 30), (5, 0), (40, 7)):
        assert (lif.pair_counts(full, Ws, start) == lif._pair_counts_np(full, Ws, start)).all()


def test_lif_kernels_agree():
    """LIF kernels 'nbf' (default, fast noise), 'nb' and 'np' (reference) give the same rates on a v4 network within
    the sampling noise of 40 frames x 16 heads (they use different random streams)."""
    import dataclasses

    from mapstdp import lif
    a = basic.Agent(basic.Config("spiking", engine="lif", heads=16, rule="emergent", **V4 | {"beta_kappa": 0}))
    rates = {}
    for kern in ("np", "nb", "nbf"):
        p, rng = dataclasses.replace(a.lp, kernel=kern), np.random.default_rng(3)
        st, n = lif.init_state(len(a.W), 16, p), 0.0
        for v in a.held[:40]:
            n = n + lif.run_lif(a.W, v, 16, rng, st, p)[1]["n"]
        rates[kern] = n.sum() / (40 * 16 * p.k * p.dt * len(a.W))
    assert 3 < rates["np"] < 20 and all(abs(rates[k] / rates["np"] - 1) < 0.03 for k in ("nb", "nbf")), rates


def test_p_cross_zero_seals_modules():
    """--p-cross 0: no cross-module synapse outside the routing pairs; every other synapse as in the default build."""
    a, b = basic.Agent(basic.Config(p_cc=0.0, p_cross=0.0)), basic.Agent(basic.Config(p_cc=0.0))
    t = a.roles[a.m]
    cross = (a.m[:, None] != a.m[None, :]) & ~((t[None, :] == "C") & np.isin(t[:, None], ["A", "L"]))
    assert not a.mask[cross].any() and b.mask[cross].any() and (a.mask[~cross] == b.mask[~cross]).all()
