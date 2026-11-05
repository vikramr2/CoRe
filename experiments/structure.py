"""How the map term changes network structure, from saved models (m3 --save-model; report.py puts them in
reports/<study>/models/<group>/s<seed>.npz). Run from the repo root:
  python -m experiments.structure <study>             -> reports/<study>/structure.csv, structure.md, structure_*.png
  python -m experiments.structure --models DIR [--out DIR] [--pair ON OFF ...]   (dev runs: any directory of groups)
Per model and snapshot (W_ep<k>): weight shares by module block (basic.block_shares: within, C->A, C->L, A<->A, A<->L,
L<->L, ->C), the K x K module weight table WK[b, a] / sum W, and the map equation L (bits, Eq. 3) of W's own walk
T = W/d under uniform teleportation (core.flow, alpha 0.2): an engine-free measure of how modular W is.
Pairs (on, off) are compared at the last snapshot by two-sided Mann-Whitney; figures show the mean
final module table per arm and on - off.
"""
import argparse
import csv
import json
import pathlib
import re

import numpy as np
from scipy.stats import mannwhitneyu

from experiments import report
from mapstdp import core
from models import basic

BLOCKS = ("within", "C->A", "C->L", "A<->A", "A<->L", "L<->L", "->C")
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential blue (dataviz palette, light mode)
DIV = ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c53030"]  # diverging blue <-> gray <-> red
INK, MUTED, SURFACE = "#2b2b2a", "#6b6b68", "#fcfcfb"


def module_labels(roles):
    """C, L1, L2, ..., A1, A2, ... in module order."""
    n = {"C": 0, "L": 0, "A": 0}
    out = []
    for r in roles:
        n[r] += 1
        out.append(r if r == "C" else f"{r}{n[r]}")
    return out


def walk_L(W, m, K, alpha=0.2):
    """Map equation (bits) of W's walk T = W / d with uniform teleportation alpha."""
    pi = core.flow(W, np.full(len(m), 1 / len(m)), alpha)
    p, q, _, _ = core.module_stats(W, pi, m, K)
    return core.map_equation(p, q, pi)


def load(path):
    """Snapshots {episode: W}, m, roles, config of one saved model."""
    z = np.load(path, allow_pickle=False)
    snaps = {int(re.fullmatch(r"W_ep(\d+)", k)[1]): z[k].astype(float) for k in z.files if k.startswith("W_ep")}
    return snaps, z["m"], z["roles"].astype(str), json.loads(str(z["config"]))


def analyse(models):
    """Rows (group, seed, episode, L, block shares) and final module tables {group: [WK, ...]}."""
    rows, final = [], {}
    for gdir in sorted(p for p in models.iterdir() if p.is_dir()):
        for f in sorted(gdir.glob("s*.npz")):
            snaps, m, roles, cfg = load(f)
            K = len(roles)
            H = np.eye(K)[m]
            for ep in sorted(snaps):
                W = snaps[ep]
                rows.append(dict(group=gdir.name, seed=int(f.stem[1:]), episode=ep, task=cfg.get("task", ""),
                                 lam=cfg.get("lam"), L_W=walk_L(W, m, K), **basic.block_shares(W, m, roles)))
            W = snaps[max(snaps)]
            final.setdefault(gdir.name, []).append((H.T @ W @ H / W.sum(), roles))
    return rows, final


def last(rows):
    """{group: [row at each run's last snapshot]}."""
    end = {}
    for r in rows:
        k = (r["group"], r["seed"])
        if k not in end or r["episode"] > end[k]["episode"]:
            end[k] = r
    out = {}
    for (g, _), r in sorted(end.items()):
        out.setdefault(g, []).append(r)
    return out


def compare(end, pairs):
    """Two-sided Mann-Whitney on the final L_W and block shares, per (on, off) pair."""
    res = []
    for on, off, *_ in pairs:
        if on not in end or off not in end:
            continue
        for k in ("L_W", *BLOCKS):
            a, b = [r[k] for r in end[on]], [r[k] for r in end[off]]
            p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
            res.append(dict(on=on, off=off, metric=k, on_mean=float(np.mean(a)), off_mean=float(np.mean(b)),
                            diff=float(np.mean(a) - np.mean(b)), p=float(p)))
    return res


def heat(ax, M, labels, cmap, vmin, vmax, title, fmt):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, color=MUTED, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, color=MUTED, fontsize=8)
    ax.set_xlabel("from module (pre)", color=MUTED, fontsize=8)
    ax.set_ylabel("to module (post)", color=MUTED, fontsize=8)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    for (i, j), x in np.ndenumerate(M):
        ax.text(j, i, fmt(x), ha="center", va="center", fontsize=7,
                color=SURFACE if abs(x - vmin) > 0.6 * (vmax - vmin) else INK)


def figures(final, pairs, out):
    """structure_<on>.png per pair: mean final module weight table of each group of the pair, and on - off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    seq, div = LinearSegmentedColormap.from_list("seq", SEQ), LinearSegmentedColormap.from_list("div", DIV)
    paths = []
    for on, off, *_ in pairs:
        if on not in final or off not in final:
            continue
        A, B = (np.mean([x for x, _ in final[g]], 0) for g in (on, off))
        labels = module_labels(final[on][0][1])
        top, dmax = max(A.max(), B.max()), max(np.abs(A - B).max(), 1e-12)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE)
        pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
        heat(axs[0], A, labels, seq, 0, top, f"{on} (n = {len(final[on])})\n% of total weight", pct)
        heat(axs[1], B, labels, seq, 0, top, f"{off} (n = {len(final[off])})\n% of total weight", pct)
        heat(axs[2], A - B, labels, div, -dmax, dmax, "On − off, percentage points\nred: more weight with the map term",
             lambda x: f"{100 * x:+.1f}")
        fig.tight_layout()
        p = out / f"structure_{on}.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths


def write(rows, cmp, paths, out, title):
    with open(out / "structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["group", "seed", "episode", "task", "lam", "L_W", *BLOCKS])
        w.writeheader()
        w.writerows(rows)
    end = last(rows)
    md = [f"# {title}: network structure", "",
          "Generated by `python -m experiments.structure` from the saved models only. L_W = map equation (bits) of the "
          "trained W's own walk under uniform teleportation (engine-free); block columns = share of the total weight "
          "(within modules; C→A, C→L routing; A↔A, A↔L, L↔L across modules; into the controller). Final snapshot, "
          "mean ± sd over seeds.", "",
          "| Group | n | L_W | " + " | ".join(BLOCKS) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in BLOCKS)
          + " |"]
    for g, rs in end.items():
        cell = lambda k, rs=rs: f"{np.mean([r[k] for r in rs]):.3f} ± {np.std([r[k] for r in rs]):.3f}"  # noqa: E731
        md.append(f"| `{g}` | {len(rs)} | {cell('L_W')} | " + " | ".join(cell(k) for k in BLOCKS) + " |")
    if cmp:
        md += ["", "## Pairs (final snapshot)", "",
               "Two-sided Mann–Whitney p, unadjusted; difference = group − against.", "",
               "| Group | Against | Metric | Group mean | Against mean | Difference | p |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| `{r['on']}` | `{r['off']}` | {r['metric']} | {r['on_mean']:.3f} | {r['off_mean']:.3f} | "
               f"{r['diff']:+.3f} | {r['p']:.2g} |" for r in cmp]
    md += [""] + [f"![{p.stem}]({p.name})" for p in paths] + [""]
    (out / "structure.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", nargs="?")
    ap.add_argument("--models", help="directory of <group>/s<seed>.npz (default reports/<study>/models)")
    ap.add_argument("--out", help="output directory (default reports/<study>, or --models' parent)")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ON", "OFF"), help="(on, off) groups to compare")
    a = ap.parse_args()
    assert a.study or a.models, "give a study or --models"
    models = pathlib.Path(a.models) if a.models else report.REPORTS / a.study / "models"
    out = pathlib.Path(a.out) if a.out else (report.REPORTS / a.study if a.study else models.parent)
    pairs = a.pair or (report.STUDIES[a.study].get("pairs", []) if a.study in report.STUDIES else [])
    rows, final = analyse(models)
    cmp = compare(last(rows), pairs)
    paths = figures(final, pairs, out)
    write(rows, cmp, paths, out, a.study or models.name)
    print(f"{len(rows)} snapshots, {len(last(rows))} groups -> {out}/structure.csv, structure.md, {len(paths)} figures")


if __name__ == "__main__":
    main()
