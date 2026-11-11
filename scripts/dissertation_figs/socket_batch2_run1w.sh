#!/usr/bin/env bash
# One-way runner, random-split only: plan -> tune -> evaluate.  usage: socket_batch2_run1w.sh <code_dir> <root> <data_dir>
set -euo pipefail
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
CODE=$1; ROOT=$2; DATA=$3
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
cd "$CODE"
[ -d "$ROOT" ] || $PY run_experiments.py plan --root "$ROOT" --profile standard --random-split-only ${PLAN_EXTRA:-} \
  --jobs-csv "$DATA/jobs.csv" --inhibitors-csv "$DATA/inhibitors.csv" --job-inh-csv "$DATA/job_inh.csv" --pair-csv "$DATA/pair.csv"
$PY run_experiments.py tune --root "$ROOT" --resume --progress-interval 120
$PY run_experiments.py evaluate --root "$ROOT" --resume --progress-interval 120
$PY run_experiments.py status --root "$ROOT"
