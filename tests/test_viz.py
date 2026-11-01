"""experiments.viz (the replay recorder): recording does not change the played episode, every sampled cause is an
earlier spike of the same head or copy inside the engine's window, branching parents are one step earlier, and the
page's schema keys are present. Short episodes (MAX frames) keep it fast."""
import numpy as np
import pytest

from experiments import ablate, viz

MAX = 30
MODELS = {"flow": "models/emergent-v4/CartPole_flow_on/s2004.npz",
          "branching": "models/emergent-v4/CartPole_branching_on/s2005.npz",
          "lif": "models/emergent-v4/CartPole_lif_on/s2017.npz",
          "qif": "models/emergent-v4/CartPole_qif_on/s2004.npz",
          "nlif": "models/race-cartpole/CartPole_cols_on_race_rp/s7010.npz"}
_CACHE = {}


def rec(eng):
    if eng not in _CACHE:
        _CACHE[eng] = viz.record(viz.ROOT / MODELS[eng], tries=1, max_frames=MAX)
    return _CACHE[eng]


@pytest.mark.parametrize("eng", ["flow", "branching"])
def test_v4_recording_keeps_episode(eng):
    model, run = viz.v4_paths(viz.ROOT / MODELS[eng])
    plain = viz.play_v4(ablate.v4_agent(model, run)[0], 0, MAX)
    a, store = ablate.v4_agent(model, run)[0], {}
    with viz.patched(viz.v4_wrappers(store, len(a.W))):
        tapped = viz.play_v4(a, 0, MAX, store)
    assert plain[0] == tapped[0] and len(plain[1]) == len(tapped[1]) == MAX
    for x, y in zip(plain[1], tapped[1]):
        assert x["a"] == y["a"] and np.array_equal(x["obs"], y["obs"])
    assert viz.sp.run_heads.__name__ == "run_heads"  # restored


def test_race_recording_keeps_episode():
    model = viz.ROOT / MODELS["nlif"]
    plain = viz.play_race(viz.race_runner(model), 1, MAX)
    r, store = viz.race_runner(model), {}
    with viz.patched(viz.race_wrappers(r, store)):
        tapped = viz.play_race(r, 1, MAX, store)
    assert plain[0][0] == tapped[0][0]
    for x, y in zip(plain[0][1], tapped[0][1]):
        assert x["a"] == y["a"] and x["live"] == y["live"] and np.array_equal(x["obs"], y["obs"])
    assert "race_block" not in vars(r) and viz.td.draw.__name__ == "draw"


@pytest.mark.parametrize("eng", list(MODELS))
def test_causes_are_earlier_spikes(eng):
    out = rec(eng)
    k, W = out["k"], viz.WINDOW[eng]
    seen = set()
    for f, fr in enumerate(out["frames"]):
        for t, n, _ in fr["s"]:
            seen.add((f * k + t, n))
    n_par = 0
    for f, fr in enumerate(out["frames"]):
        for t, n, c in fr["s"]:
            if c < 0:
                continue
            assert c != n or eng == "flow"
            ta = f * k + t
            if eng == "flow":
                assert t >= 1 and (ta - 1, c) in seen
            elif ta - W >= 0:
                assert any((ta - lag, c) in seen for lag in range(1, W + 1)), (f, t, n, c)
            n_par += 1
    assert n_par > 0


def test_branching_parents_one_step_earlier():
    out = rec("branching")
    for fr in out["frames"]:
        at = {(t, n) for t, n, _ in fr["s"]}
        for t, _, c in fr["s"]:
            assert c < 0 or (t - 1, c) in at


@pytest.mark.parametrize("eng", list(MODELS))
def test_schema(eng):
    out = rec(eng)
    keys = {"id", "task", "family", "engine", "arm", "label", "seed", "model", "dt_ms", "k", "burn", "modules", "m",
            "edges", "actions", "note", "shares", "return", "train", "env", "frames"}
    assert keys <= set(out)
    assert set(out["shares"]) == {"walk", "stim", "none"} and abs(sum(out["shares"].values()) - 1) < 1e-3
    assert all(0 <= n < len(out["m"]) for fr in out["frames"] for _, n, _ in fr["s"])
    assert max(out["m"]) == len(out["modules"]) - 1
    for mod in out["modules"]:
        assert {"name", "role", "action", "grid", "grid_labels"} <= set(mod)
        assert mod["role"] in ("stim", "ctrl", "lat", "act", "critic", "inh")
    assert all(len(e) == 3 and 0 < e[2] <= 1 for e in out["edges"]) and len(out["edges"]) <= viz.MAX_EDGES
    for fr in out["frames"]:
        assert {"obs", "a", "dec", "P", "s", "pose"} <= set(fr)
        assert -1 <= fr["dec"] < out["k"]
        assert all(0 <= t < out["k"] and c >= -2 for t, _, c in fr["s"])
        if out["family"] == "race":  # input neurons: counts only, never in s
            n_in = sum(1 for q in out["m"] if out["modules"][q]["role"] == "stim")
            assert all(0 <= n < n_in and c > 0 for n, c in fr["inp"]) and all(n >= n_in for _, n, _ in fr["s"])
        else:
            assert "inp" not in fr
    assert (out["family"] == "race") == ("inp_max" in out) and out.get("inp_max", 1) > 0
    out_n = {int(j): v for j, v in out["out"].items()}  # each neuron's outgoing synapses, strongest first
    assert all(0 <= j < len(out["m"]) and all(0 <= i < len(out["m"]) and w > 0 for i, w in v) for j, v in out_n.items())
    assert all([w for _, w in v] == sorted((w for _, w in v), reverse=True) for v in out_n.values())
    assert 0 <= out["rec"]["cross"] <= 1 and any(out["m"][e[0]] == out["m"][e[1]] for e in out["edges"])
    if out["family"] == "race":  # one input bundle per post module; recurrent edges only among actor posts
        assert sorted(b[1] for b in out["bundles"]) == list(range(1, len(out["modules"])))
        assert all(b[0] == 0 and 0 < b[2] <= 1 for b in out["bundles"]) and max(b[2] for b in out["bundles"]) == 1
        acts = {n for n, q in enumerate(out["m"]) if out["modules"][q]["role"] == "act"}
        assert out["edges"] and all(e[0] in acts and e[1] in acts for e in out["edges"])
        assert 0 < out["rec"]["cross"] < 1
        n_in = out["m"].count(0)
        for j in range(n_in):  # inputs: at most the strongest OUT_TOP_INPUT targets in each population
            per = np.bincount([out["m"][i] for i, _ in out_n[j]], minlength=len(out["modules"]))
            assert per[0] == 0 and per.max() <= viz.OUT_TOP_INPUT
        assert any(out["m"][e[0]] == out["m"][e[1]] for e in out["edges"])  # within-column recurrence is drawn
