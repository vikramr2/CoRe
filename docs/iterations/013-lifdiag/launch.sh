#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/vr9/map-stdp
cd /home/vr9/map-stdp
PY=$(conda run -n map-stdp which python)
cat /tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/jobs.txt | xargs -P 30 -L 1 bash -c '$0 /tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/run.py $1 $2 $3 > /tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/logs/$1_$2_$3.log 2>&1' $PY
echo ALLDONE
