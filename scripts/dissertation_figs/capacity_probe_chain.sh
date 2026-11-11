#!/usr/bin/env bash
# After the re-tune probe: score it; if no calibrated BD-MPI variant 'helps', run the capacity probe and score it.
set -u
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
D=/home/akhil/hpcResearch/repos/coScheduling/scripts/dissertation_figs
R=$HOME/hpcResearch/experiment_outputs/retune_probe
C=$HOME/hpcResearch/experiment_outputs/capacity_probe
cd "$D"
until grep -qE "probe (done|FAILED)" "$R/chain.log" 2>/dev/null; do sleep 120; done
grep -q "probe FAILED" "$R/chain.log" && { echo "$(date +%T) retune probe FAILED; capacity chain exits"; exit 1; }
echo "$(date +%T) retune probe done; scoring"
$PY retune_probe_score.py retune > "$R/score.log" 2>&1 || { echo "retune scoring failed"; exit 1; }
echo "$(date +%T) RETUNE SCORED"
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$D/output/retune_probe_trigger.json'))['run capacity probe'] else 1)"; then
  echo "$(date +%T) a calibrated variant helps: capacity probe NOT run"; echo "CAPACITY SKIPPED"; exit 0
fi
echo "$(date +%T) starting capacity stage A"
PROBE=capacity $PY retune_probe_run.py 8 A || exit 1
PROBE=capacity $PY retune_probe_build.py C || exit 1
PROBE=capacity $PY retune_probe_run.py 7 C
PROBE_ROOT=$C $PY retune_probe_score.py capacity > "$C/score.log" 2>&1 || { echo "capacity scoring failed"; exit 1; }
echo "$(date +%T) CAPACITY SCORED"
