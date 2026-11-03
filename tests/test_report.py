"""experiments.report summarize on a tiny fake study: summary.csv from runs/ only."""
import csv
import json

from scipy.stats import mannwhitneyu

from experiments import report

STUDY = {"fake": dict(title="fake", reference={"CartPole": "ref"}, gates={"ref": 2},
                      groups={"ref": dict(task="CartPole", engine="lif", flags=["--block", "20"], seeds=[0, 1]),
                              "alt": dict(task="CartPole", engine="branching", flags=[], seeds=[0, 1, 2])})}


def fake(Tf, Ts, ret, ms):
    return {"config": {"episodes": 400, "max_eps": 400}, "returns": ret, "T_f": Tf, "T_s": Ts, "ms_per_frame": ms}


def test_summarize_fake_study(tmp_path):
    runs = {"ref": [fake(1, 100, [500.0] * 150, 1.0), fake(3, 102, [10.0, 10.0] + [500.0] * 148, 3.0)],
            "alt": [fake(None, None, [10.0] * 150, 2.0), fake(None, None, [20.0] * 150, 2.0)]}
    for g, rs in runs.items():
        (tmp_path / "fake" / "runs" / g).mkdir(parents=True)
        for i, r in enumerate(rs):
            (tmp_path / "fake" / "runs" / g / f"s{i}.json").write_text(json.dumps(r))
    (tmp_path / "fake" / "runs" / "alt" / "s2.partial").write_text("{}")  # unfinished: ignored
    report.summarize("fake", root=tmp_path, studies=STUDY)
    rows = {r["group"]: r for r in csv.DictReader(open(tmp_path / "fake" / "summary.csv"))}
    ref, alt = rows["ref"], rows["alt"]
    assert (ref["n"], ref["solved"], ref["T_f_n"], ref["T_m_n"], ref["gate_pass"]) == ("2", "2", "2", "2", "True")
    assert float(ref["T_f_mean"]) == 2.0 and float(ref["T_f_sd"]) == 1.0 and float(ref["T_s_median"]) == 101.0
    assert float(ref["T_m_mean"]) == 100.0 and float(ref["ret_mean"]) == 500.0 and float(ref["ms_per_frame"]) == 2.0
    assert ref["mw_ref"] == "" and ref["mw_T_f"] == ""
    assert (alt["n_declared"], alt["n"], alt["solved"], alt["T_f_n"], alt["mw_ref"]) == ("3", "2", "0", "0", "ref")
    assert float(alt["T_s_median"]) == float("inf") and float(alt["ret_mean"]) == 15.0 and float(alt["ret_sd"]) == 5.0
    assert float(alt["mw_T_s"]) == mannwhitneyu([401, 401], [100, 102]).pvalue  # never reached = cap + 1
    assert float(alt["mw_ret"]) == mannwhitneyu([10.0, 20.0], [500.0, 500.0]).pvalue
    assert alt["gate_pass"] == "" and "| `ref` | 2/2 | 2/2 |" in (tmp_path / "fake" / "summary.md").read_text()


def test_summarize_cross_study_reference(tmp_path):
    studies = STUDY | {"comp": dict(title="comp", reference={"CartPole": "fake/ref"}, gates={},
                                    groups={"new": dict(task="CartPole", engine="flow", flags=[], seeds=[0])})}
    for st, g, r in (("fake", "ref", fake(1, 100, [500.0] * 150, 1.0)),
                     ("comp", "new", fake(None, None, [10.0] * 150, 1.0))):
        (tmp_path / st / "runs" / g).mkdir(parents=True)
        (tmp_path / st / "runs" / g / "s0.json").write_text(json.dumps(r))
    report.summarize("comp", root=tmp_path, studies=studies)
    row = next(csv.DictReader(open(tmp_path / "comp" / "summary.csv")))
    assert row["group"] == "new" and row["mw_ref"] == "fake/ref"
    assert float(row["mw_ret"]) == mannwhitneyu([10.0], [500.0]).pvalue


def test_summarize_tdstdp_pairs(tmp_path):
    """Declared pairs of a tdstdp-runner study: pairs.csv with the Hodges-Lehmann shift, Mann-Whitney p and Holm over
    the family; the primary metric is AUC over the first `episodes` episodes on CartPole, last-100 on LunarLander."""
    g = lambda t: dict(task=t, engine="nlif", flags=[], seeds=[0, 1, 2])  # noqa: E731
    studies = {"pr": dict(title="pr", reference={}, gates={}, pairs=[("CP_on", "CP_off"), ("LL_on", "LL_off")],
                          groups={k: g("CartPole" if k[0] == "C" else "LunarLander")
                                  for k in ("CP_on", "CP_off", "LL_on", "LL_off")})}
    run = lambda t, ret: {"config": {"task": t, "episodes": 2, "max_eps": 3}, "returns": ret, "T_f": None,  # noqa: E731
                          "T_s": None, "ms_per_frame": 1.0}
    data = {"CP_on": [run("CartPole", [x, x, 0.0]) for x in (10.0, 20.0, 30.0)],
            "CP_off": [run("CartPole", [x, x, 99.0]) for x in (1.0, 2.0, 3.0)],
            "LL_on": [run("LunarLander", [0.0, x]) for x in (5.0, 6.0, 7.0)],
            "LL_off": [run("LunarLander", [0.0, x]) for x in (5.0, 6.0, 7.0)]}
    for k, rs in data.items():
        (tmp_path / "pr" / "runs" / k).mkdir(parents=True)
        for i, r in enumerate(rs):
            (tmp_path / "pr" / "runs" / k / f"s{i}.json").write_text(json.dumps(r))
    report.summarize("pr", root=tmp_path, studies=studies)
    rows = {r["on"]: r for r in csv.DictReader(open(tmp_path / "pr" / "pairs.csv"))}
    cp, ll = rows["CP_on"], rows["LL_on"]
    assert float(cp["primary_on"]) == 20.0 and float(cp["primary_hl"]) == 18.0  # AUC over 2 episodes, not the 3rd
    assert float(cp["primary_hl_lo"]) <= 18.0 <= float(cp["primary_hl_hi"]) and float(ll["primary_hl"]) == 0.0
    p = mannwhitneyu([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]).pvalue
    assert float(cp["primary_p"]) == p and float(cp["primary_p_holm"]) == min(1.0, 2 * p)
    assert "## Pairs" in (tmp_path / "pr" / "summary.md").read_text()


def test_summarize_cross_study_pairs(tmp_path):
    """A pair member of the form "<study>/<group>" is loaded from that study's runs/ (companion studies, 016 §19)."""
    studies = STUDY | {"comp": dict(title="comp", reference={}, gates={}, pairs=[("fake/ref", "new", "x")],
                                    groups={"new": dict(task="CartPole", engine="flow", flags=[], seeds=[0, 1])})}
    run = lambda ret: {"config": {"task": "CartPole", "episodes": 2, "max_eps": 2}, "returns": ret,  # noqa: E731
                       "T_f": None, "T_s": None, "ms_per_frame": 1.0}
    data = {("fake", "ref"): [run([10.0, 10.0]), run([20.0, 20.0])],
            ("comp", "new"): [run([1.0, 1.0]), run([2.0, 2.0])]}
    for (st, g), rs in data.items():
        (tmp_path / st / "runs" / g).mkdir(parents=True)
        for i, r in enumerate(rs):
            (tmp_path / st / "runs" / g / f"s{i}.json").write_text(json.dumps(r))
    report.summarize("comp", root=tmp_path, studies=studies)
    row = next(csv.DictReader(open(tmp_path / "comp" / "pairs.csv")))
    assert (row["on"], row["off"], row["family"], row["n_on"], row["n_off"]) == ("fake/ref", "new", "x", "2", "2")
    assert float(row["primary_hl"]) == 13.5 and float(row["primary_on"]) == 15.0
    assert [r["group"] for r in csv.DictReader(open(tmp_path / "comp" / "summary.csv"))] == ["new"]
