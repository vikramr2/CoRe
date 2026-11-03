"""experiments.ablate (iteration 017) and report.summarize_ablate: the conditions' weights, the paired statistics, and
one small end-to-end run per family."""
import json
import types

import numpy as np

from experiments import ablate, report
from models import basic
from models import tdstdp as ex


def _v4_net(seed=0):
    m = np.repeat(np.arange(5), [4, 3, 3, 2, 2])
    roles = np.array(["C", "L", "L", "A", "A"])
    W = np.random.default_rng(seed).uniform(0.1, 1.0, (len(m), len(m)))
    return W, m, roles


def test_v4_weights():
    """Lesions cut the latents' outgoing weights and leave every other neuron's out-degree d; removal also cuts their
    inputs (the controller's outflow renormalises onto the rest); latent_path_only cuts C -> A only; flow keeps a
    1e-12 self-loop on each latent left without outgoing weight."""
    W, m, roles = _v4_net()
    L, L1 = np.flatnonzero(np.isin(m, [1, 2])), np.flatnonzero(m == 1)
    other = np.setdiff1d(np.arange(len(m)), L)
    assert np.array_equal(ablate.v4_weights(W, m, roles, "intact", False), W)
    X = ablate.v4_weights(W, m, roles, "lesion_L", False)
    assert (X[:, L] == 0).all() and np.array_equal(X[:, other], W[:, other])
    X = ablate.v4_weights(W, m, roles, "lesion_L1", False)
    assert (X[:, L1] == 0).all() and np.array_equal(np.delete(X, L1, 1), np.delete(W, L1, 1))
    X = ablate.v4_weights(W, m, roles, "remove_L", False)
    assert (X[:, L] == 0).all() and (X[L, :] == 0).all()
    X = ablate.v4_weights(W, m, roles, "latent_path_only", False)
    A, C = np.isin(m, [3, 4]), m == 0
    assert (X[np.ix_(A, C)] == 0).all() and np.array_equal(X[~A], W[~A])
    X = ablate.v4_weights(W, m, roles, "lesion_L", True)
    assert np.allclose(X[L, L], ablate.EPS_LOOP) and (X.sum(0) > 0).all()
    assert np.isfinite(basic.core.flow(X, np.r_[np.full(4, 0.25), np.zeros(10)], 0.2, 10)).all()


def test_ck_kicks():
    """Lesions keep the intact d; rec_untrained uses R0 with its own d; on sealed R the cross-column lesion is the
    intact kicks."""
    rng = np.random.default_rng(1)
    m = np.repeat([0, 1], 3)
    R = rng.uniform(0.1, 1, (6, 6))
    hr = types.SimpleNamespace(R=R, m=m, chi=0.3)
    d = R.sum(0)
    same = m[:, None] == m[None, :]
    assert ablate.ck_kicks(hr, "intact") is None
    assert (ablate.ck_kicks(hr, "rec_off")() == 0).all()
    assert np.allclose(ablate.ck_kicks(hr, "cross_lesion")(), 0.3 * R * same / d)
    assert np.allclose(ablate.ck_kicks(hr, "within_lesion")(), 0.3 * R * ~same / d)
    R0 = rng.uniform(0.1, 1, (6, 6))
    assert np.allclose(ablate.ck_kicks(hr, "rec_untrained", R0)(), 0.3 * R0 / R0.sum(0))
    hr.R = R * same
    assert np.allclose(ablate.ck_kicks(hr, "cross_lesion")(), 0.3 * hr.R / hr.R.sum(0))


def test_walsh_hl():
    x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0, 6.0, 7.5, 9.0])
    hl, lo, hi = report.walsh_hl(x)
    w = [(x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x))]
    assert hl == np.median(w) and lo <= hl <= hi


def test_summarize_ablate(tmp_path):
    """Paired deltas, Wilcoxon with Holm within a family (checks excluded), and a contrast between groups."""
    s = dict(title="t", runner="ablate", reference={}, gates={}, checks={"g_b": ["cut"]},
             groups={k: dict(task="CartPole", engine="flow", family="f", seeds=range(8), flags=[], models="x")
                     for k in ("g_a", "g_b")}, contrasts=[("g_a", "g_b", "cut", "fc")])
    rng = np.random.default_rng(0)
    for k, shift in (("g_a", -100.0), ("g_b", 0.0)):
        for sd in range(8):
            I0 = 400 + 10 * rng.standard_normal()
            run = {"conditions": {"intact": {"mean": I0}, "cut": {"mean": I0 + shift + rng.standard_normal()}}}
            p = tmp_path / "runs" / k / f"s{sd}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(run))
    rows, cons = report.summarize_ablate("t", s, tmp_path)
    r = {(x["group"], x["condition"]): x for x in rows}
    a = r[("g_a", "cut")]
    assert a["hl"] < -95 and a["p"] < 0.01 and a["norm_loss_median"] > 0.2
    assert r[("g_b", "cut")]["check"] and r[("g_b", "cut")]["p_holm"] is None
    assert r[("g_a", "cut")]["p_holm"] == r[("g_a", "cut")]["p"]  # one test in the family: Holm leaves it
    assert len(cons) == 1 and cons[0]["hl"] < -95 and cons[0]["p"] < 0.01
    assert (tmp_path / "summary.md").exists() and (tmp_path / "contrasts.csv").exists()


def test_run_ck_sealed(tmp_path):
    """A small sealed-column model end to end: every condition runs; the cross-column lesion of sealed columns is the
    intact model (same returns on the same stream)."""
    cfg = ex.preset("CartPole", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, readout="race",
                    race_beta=6.0, rec="cols", chi_rec=0.3, lam=0.0, rec_seal=True, eta_rec=0.1, lam_rec=0.0)
    r = ex.CKRunner(cfg)
    for _ in range(30):
        r.tick()
    p = tmp_path / "m.npz"
    ex.save_ck_model(p, cfg, r, eta_sgd=np.array([0.01, 0.02]))
    out = ablate.run_ck(str(p), 2, 0)
    c = out["conditions"]
    assert list(c) == list(ablate.CONDITIONS["ck"])
    assert c["cross_lesion"]["returns"] == c["intact"]["returns"]
    assert all(len(x["returns"]) == 2 for x in c.values())


def test_run_v4_flow(tmp_path):
    """A small flow model end to end: every condition runs; the intact frozen check reproduces basic.frozen_returns."""
    a = basic.Agent(basic.Config())
    p, run = tmp_path / "models" / "g" / "s0.npz", tmp_path / "runs" / "g" / "s0.json"
    basic.save_model(p, a, {"W_ep0": a.W}, a.cfg)
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"evals": [{"beta": a.beta}], "frozen_returns": basic.frozen_returns(a, 3)}))
    out = ablate.run_v4(str(p), str(run), 2, 0)
    assert list(out["conditions"]) == list(ablate.CONDITIONS["v4"])
    assert out["frozen_check"]["returns"][:3] == json.loads(run.read_text())["frozen_returns"]
    assert all(np.isfinite(x["mean"]) for x in out["conditions"].values())
