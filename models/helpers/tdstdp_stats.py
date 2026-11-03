"""Statistics of TD-STDP runs: group_stats (the statistics report.py uses) and the --summary table
against Chung & Kozma's Table 2 and the v4 runs."""
import glob
import json
import pathlib
import re

import numpy as np

from models.helpers.runner import OUT, PRESETS, trailing_mean_time

PAPER = {"CartPole": dict(T_f=(57.0, 23.97), T_s=(169.5, 23.47)),
         "LunarLander": dict(T_f=(383.8, 71.49), T_s=(2575.2, 666.51))}  # CK Table 2 (10 runs)
M_THR = {"CartPole": 475.0, "LunarLander": 200.0, "Acrobot": -100.0, "Taxi": 8.0}  # secondary: trailing-100 mean


def v4_returns(task):
    """Last-100 returns over seeds of the model v4 runs (flow; spiking per heads and variant), from models.basic."""
    pat = re.compile(rf"m3_{task}_(flow|spiking_h\d+)_lb0\.3_tn0\.04_pcc0_bk0\.01m16(_\w+?)?_eta5_s\d+")
    out = {}
    for f in sorted(glob.glob(str(OUT / f"m3_{task}_*_eta5_s*.json"))):
        m = pat.fullmatch(pathlib.Path(f).stem)
        if m and (m[1] != "flow" or not m[2]):
            k = "v4 " + m[1].replace("_", " ") + (" " + m[2][1:] if m[2] else "")
            out.setdefault(k, []).append(np.mean(json.loads(pathlib.Path(f).read_text())["returns"][-100:]))
    return out


def group_stats(task, rs, ref=None):
    """Summary statistics of one group of runs (result dicts) on `task`, shared by --summary and experiments.report:
    T_f / T_s (CK definitions) mean, sd (ddof 0) and count over runs that reach them; median T_s with unsolved =
    infinite; T_m = first episode whose trailing-100 mean reaches M_THR; last-100 return mean, sd and the count >=
    M_THR; mean ms per frame. With ref (another group's runs): two-sided Mann-Whitney p on T_f, T_s (never reached =
    episode cap + 1) and the last-100 return."""
    from scipy.stats import mannwhitneyu
    Tf = [r["T_f"] for r in rs if r["T_f"] is not None]
    Ts = [r["T_s"] for r in rs if r["T_s"] is not None]
    Tm = [t for t in (trailing_mean_time(r["returns"], M_THR[task]) for r in rs) if t is not None]
    R = [float(np.mean(r["returns"][-100:])) for r in rs]
    ms = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (float("nan"),) * 2  # noqa: E731
    out = {"n": len(rs), "solved": len(Ts), "T_f_mean": ms(Tf)[0], "T_f_sd": ms(Tf)[1], "T_f_n": len(Tf),
           "T_s_mean": ms(Ts)[0], "T_s_sd": ms(Ts)[1],
           "T_s_median": float(np.median([r["T_s"] if r["T_s"] is not None else np.inf for r in rs])),
           "T_m_mean": ms(Tm)[0], "T_m_sd": ms(Tm)[1], "T_m_n": len(Tm), "ret_mean": float(np.mean(R)),
           "ret_sd": float(np.std(R)), "ret_ge_M": sum(x >= M_THR[task] for x in R),
           "ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rs]))}
    if ref is not None:
        cap = lambda r: (r["config"].get("max_eps") or r["config"]["episodes"]) + 1  # noqa: E731
        cens = lambda rr, k: [r[k] or cap(r) for r in rr]  # noqa: E731
        R0 = [np.mean(r["returns"][-100:]) for r in ref]
        out |= {"mw_T_f": float(mannwhitneyu(cens(rs, "T_f"), cens(ref, "T_f")).pvalue),
                "mw_T_s": float(mannwhitneyu(cens(rs, "T_s"), cens(ref, "T_s")).pvalue),
                "mw_ret": float(mannwhitneyu(R, R0).pvalue)}
    return out


def summary():
    print("T_f / T_s (CK definitions, pre-registered): mean +- sd over runs that reach it [n]; median T_s counts "
          "unsolved as infinite. Authors'-code groups (ck) count episodes pooled over copies (main.py); legacy "
          "LunarLander groups count per copy (copy-mean returns). Secondary: T_m = first episode whose trailing-100 "
          "mean reaches M_THR (CartPole 475, Gymnasium's threshold; LunarLander 200 = T_s). ret: last-100 return, "
          "mean +- sd over seeds; >=M: runs with ret >= M_THR. MW: Mann-Whitney p against the family's base group "
          "(the task preset), on T_f / T_s (never reached = episode cap + 1) / ret.")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "tdstdp_*.json"))):
        groups.setdefault(pathlib.Path(f).stem.rsplit("_s", 1)[0], []).append(json.loads(pathlib.Path(f).read_text()))
    ms = lambda m, s: f"{m:6.1f} +- {s:5.1f}" if m == m else "   nan          "  # noqa: E731
    for task in PRESETS:
        if not any(g.split("_")[1] == task for g in groups):
            continue
        p = PAPER.get(task)
        print(f"\n{task}: " + (f"CK 2020 T_f {p['T_f'][0]:.1f} +- {p['T_f'][1]:.1f}  "
                                f"T_s {p['T_s'][0]:.1f} +- {p['T_s'][1]:.1f}" if p else "(no CK 2020 result)"))
        for k, R in v4_returns(task).items():
            print(f"  {k:28} n={len(R):2d} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}")
        for ck, base in ((False, f"tdstdp_{task}"), (True, f"tdstdp_{task}_ck")):
            gs = {g: rs for g, rs in groups.items() if g.split("_")[1] == task and (g.split("_")[2:3] == ["ck"]) == ck}
            if not gs:
                continue
            print("  authors' code (default):" if ck else "  pre-audit code (--legacy):")
            for g, rs in sorted(gs.items(), key=lambda kv: (kv[0] != base, kv[0])):
                st = group_stats(task, rs, gs[base] if g != base and base in gs else None)
                line = (f"  {g:28} n={st['n']:2d} solved {st['solved']:2d}/{st['n']:2d}  T_f "
                        f"{ms(st['T_f_mean'], st['T_f_sd'])} [{st['T_f_n']:2d}]  T_s {ms(st['T_s_mean'], st['T_s_sd'])}"
                        f" median {st['T_s_median']:5.0f}  T_m {ms(st['T_m_mean'], st['T_m_sd'])} [{st['T_m_n']:2d}]  "
                        f"ret {st['ret_mean']:6.1f} +- {st['ret_sd']:5.1f} >=M {st['ret_ge_M']:2d}  "
                        f"ms/fr {st['ms_per_frame']:5.2f}")
                if "mw_T_f" in st:
                    line += f"  MW {st['mw_T_f']:.2g} / {st['mw_T_s']:.2g} / {st['mw_ret']:.2g}"
                print(line)
