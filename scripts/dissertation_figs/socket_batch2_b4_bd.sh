#!/usr/bin/env bash
# Socket batch 2, B4: restricted training-size tiers for the bidirectional copies (blind + aware, MPI + TT).
# Plans a standard runner root per run, registers the existing full-data CV run (socket_batch2/<run>) as the root's
# tune_global task (same code, data, config; the runner then freezes its selection.json exactly as it would its own
# tuning output), then runs `evaluate` (random_split + 70 evaluate_training_size tasks).
# The random_split task doubles as a check: it must reproduce <run>/directional_predictions.csv.
# usage: socket_batch2_b4_bd.sh <blind_bd_mpi|aware_bd_mpi|blind_bd_tt|aware_bd_tt> [max_workers]
# DO NOT EDIT WHILE RUNNING (bash re-reads scripts).
set -euo pipefail
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
EXP=$HOME/hpcResearch/experiment_outputs/socket_batch2
UT=$HOME/hpcResearch/repos/coScheduling/notebooks/utils
RUN=$1; WORKERS=${2:-4}
case $RUN in
  *_bd_mpi) CODE=$UT/delta_response_filteredInhibs_biDirectional_socketAware ;;
  *_bd_tt)  CODE=$UT/delta_response_filteredInhibs_biDirectional_totalTime_socketAware ;;
  *) echo "unknown run $RUN"; exit 2 ;;
esac
if [[ $RUN == aware_* ]]; then D=$CODE/data_socket; EXTRA=(--socket-feature); else D=$CODE/data; EXTRA=(); fi
ROOT=$EXP/b4_$RUN
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
if [ ! -f "$ROOT/plan.json" ]; then
  $PY run_experiments.py plan --root "$ROOT" --profile standard \
    --jobs-csv "$D/training/jobs.csv" --inhibitors-csv "$D/training/inhibitors.csv" \
    --job-inh-csv "$D/training/job_inh.csv" --pair-csv "$D/evaluation/pair.csv" "${EXTRA[@]}"
fi
if [ ! -f "$ROOT/registered.ok" ]; then
  rm -rf "$ROOT"/tasks/*
  $PY - "$ROOT" "$EXP/$RUN" <<'EOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, ".")
import run_experiments as rx  # the copy's runner (cwd = code dir), for the resume fingerprint
root, src = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
tasks = json.loads((root / "plan.json").read_text())
tune = [t for t in tasks if t["stage"] == "tune_global"]
assert len(tune) == 1
tid = tune[0]["task_id"]
req = json.loads((src / "requested_run_config.json").read_text())
# the CV run must have been run with the tune task's settings
for k, v in tune[0]["config"].items():
    if k in req and req[k] != v and k not in ("evaluation_method", "training_apps"):
        raise SystemExit(f"tune config mismatch on {k}: plan {v!r} vs CV run {req[k]!r}")
assert json.loads((src / "run_report.json").read_text())["status"] == "complete"
d = root / "tasks" / tid
d.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256((src / "run_report.json").read_bytes()).hexdigest()
(d / "status.json").write_text(json.dumps({"task_id": tid, "status": "succeeded", "attempt": 1,
    "output_dir": str(src), "note": f"registered existing full-data CV run {src}"}, indent=2))
fp = rx.execution_fingerprint(root, tune[0], rx.resolve_task_config(root, tune[0]))
(d / "_SUCCESS.json").write_text(json.dumps({"task_id": tid, "attempt": 1, "run_report_sha256": sha,
    "execution_fingerprint": fp, "note": "registered, not executed by this runner"}, indent=2))
assert rx.successful_output(root, tid) == src
print(f"registered {src} as tune_global {tid}")
EOF
  touch "$ROOT/registered.ok"
fi
# refuse to start if resume would re-run the registered tuning task
$PY - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
import run_experiments as rx
root = Path(sys.argv[1])
tune = [t for t in json.loads((root / "plan.json").read_text()) if t["stage"] == "tune_global"][0]
cfg = rx.resolve_task_config(root, tune)
ok = rx.read_json(rx.task_root(root, tune["task_id"]) / "_SUCCESS.json").get("execution_fingerprint") == \
    rx.execution_fingerprint(root, tune, cfg)
sys.exit(0 if ok else "tune_global would be re-run (fingerprint mismatch); aborting")
EOF
$PY run_experiments.py evaluate --root "$ROOT" --max-workers "$WORKERS" --resume
