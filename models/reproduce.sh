#!/bin/bash
# Reproduce the reported runs of the race studies and save their final models (iteration 016 §20-21).
# Each reported run is deterministic given its seed and code state, so rerunning its exact command with --save-model
# writes the weights of the reported run into archive/models/<study>/ (gitignored; models/ keeps the best run per arm);
# models/verify.py checks the returns against reports/<study>/runs/ bit for bit.
# Usage: models/reproduce.sh <study> [workers]   (study: race-cartpole | race-acrobot | race-lunarlander; from the repo root, env map-stdp)
STUDY=${1:?race-cartpole | race-acrobot | race-lunarlander}; W=${2:-20}; ARMS=${ARMS:-.}  # ARMS: a regex selecting the arms to rerun
# race-cartpole ran at commit 23cc0b9, before the re-probe fit gate (74f3dd1); its runs re-probed without a gate, so
# the reproduction disables it. race-acrobot (74f3dd1) uses no re-probe; no extra flag.
EXTRA=""; [ "$STUDY" = race-cartpole ] && EXTRA="--race-fit-gate inf"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p archive/models/$STUDY logs/models/$STUDY
python - "$STUDY" "$ARMS" "$EXTRA" <<'PY' | sed "s/ *$//" | xargs -P $W -L 1 bash -c 'python -m models.tdstdp "${@:1}" > logs/models/'"$STUDY"'/$0.log 2>&1; echo "$0 rc $?"'
import re, sys
from experiments import report
study, arms, extra = sys.argv[1], sys.argv[2], sys.argv[3].split()
for k, g in report.STUDIES[study]["groups"].items():
    if not re.search(arms, k):
        continue
    for sd in g["seeds"]:
        cmd = report.study_command(study, k, sd)[3:]  # [python, -m, models.tdstdp, --task, ...] -> the flags
        i = cmd.index("--out"); cmd[i + 1] = f"logs/models/{study}/{k}_s{sd}.json"  # the result JSON (and its
        # checkpoint while running) go to logs/, which is gitignored; the reported JSON is reports/<study>/runs/
        print(f"{k}_s{sd}", *cmd, *extra, "--save-model", f"archive/models/{study}/{k}/s{sd}.npz")
PY
echo done
