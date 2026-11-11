#!/usr/bin/env bash
# Re-tune probe chain (relaunch 2026-11-10 after the empty-ablation crash): stage A (6 remaining CV jobs) and stage B
# (descriptive curve, independent of A) run concurrently on 6 + 2 workers; then build C -> run C. Writes "probe FAILED" on error.
set -u
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
D=/home/akhil/hpcResearch/repos/coScheduling/scripts/dissertation_figs
R=$HOME/hpcResearch/experiment_outputs/retune_probe
cd "$D"
fail() { echo "$(date +%T) probe FAILED: $1"; exit 1; }
echo "$(date +%T) starting stage A (6 workers) + stage B (2 workers)"
PROBE=retune $PY retune_probe_run.py 2 B > "$R/stageB.log" 2>&1 &
BPID=$!
PROBE=retune $PY retune_probe_run.py 6 A || fail "stage A runner"
for v in legacy_delta legacy_generic legacy_absolute joint_profile bidirectional_profile bidirectional_context bidirectional_profile_wall bidirectional_context_forward; do
  [ -f "$R/runs/A_cv_$v/selection.json" ] || fail "A_cv_$v has no selection.json"
done
echo "$(date +%T) stage A done; building C"
PROBE=retune $PY retune_probe_build.py C || fail "build C"
PROBE=retune $PY retune_probe_run.py 6 C || fail "stage C runner"
wait $BPID
cat "$R/stageB.log"
for d in "$R"/runs/B_* "$R"/runs/C_*; do
  grep -q '"complete"' "$d/run_report.json" 2>/dev/null || fail "$(basename $d) incomplete"
done
echo "$(date +%T) probe done"
