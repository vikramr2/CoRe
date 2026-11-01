"""Check that the reproduced runs of a race study match the reported ones bit for bit, that every model was saved in
the archive (archive/models/<study>/, gitignored) and that the best-per-arm copies in models/ equal their archived file.
Usage: python models/verify.py <study> [reports_root]   (from the repo root; reports_root defaults to reports/, or e.g.
../map-stdp-racl/reports for a study whose results are still on its own branch)"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run as a script from the repo root

from experiments import report

study = sys.argv[1]
root = pathlib.Path(__file__).resolve().parent.parent
reports = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "reports"
bad, n = 0, 0
for k, g in report.STUDIES[study]["groups"].items():
    for sd in g["seeds"]:
        rep = reports / study / "runs" / k / f"s{sd}.json"
        new = root / "logs" / "models" / study / f"{k}_s{sd}.json"
        npz = root / "archive" / "models" / study / k / f"s{sd}.npz"
        kept = root / "models" / study / k / f"s{sd}.npz"
        n += 1
        if not new.exists() or not npz.exists():
            print(f"MISSING {k} s{sd}: {'json ' if not new.exists() else ''}{'npz' if not npz.exists() else ''}")
            bad += 1
            continue
        a, b = json.load(open(rep))["returns"], json.load(open(new))["returns"]
        z = np.load(npz, allow_pickle=False)
        if a != b or z["W"].shape[0] == 0 or (kept.exists() and kept.read_bytes() != npz.read_bytes()):
            print(f"DIFFERS {k} s{sd}: reported {len(a)} episodes, mean {np.mean(a):.2f}; "
                  f"reproduced {len(b)}, {np.mean(b):.2f}")
            bad += 1
print(f"{study}: {n - bad}/{n} runs reproduced bit for bit with their model saved")
sys.exit(1 if bad else 0)
