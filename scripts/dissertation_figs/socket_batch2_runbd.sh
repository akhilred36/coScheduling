#!/usr/bin/env bash
# Bidirectional full-data fit (crossed CV selection + final random_split), as the artifact-equal local BD-TT run.
# usage: socket_batch2_runbd.sh <code_dir> <out_dir> <data_root|-> [extra pipeline args...]
#   data_root '-' = the code dir's default data/; otherwise <data_root>/{training,evaluation}/
set -euo pipefail
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
CODE=$1; OUT=$2; DATA=$3; shift 3
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
cd "$CODE"
ARGS=()
if [ "$DATA" != "-" ]; then
  ARGS=(--jobs-csv "$DATA/training/jobs.csv" --inhibitors-csv "$DATA/training/inhibitors.csv"
        --job-inh-csv "$DATA/training/job_inh.csv" --pair-csv "$DATA/evaluation/pair.csv")
fi
$PY pipeline.py --output-dir "$OUT" "${ARGS[@]}" "$@"
