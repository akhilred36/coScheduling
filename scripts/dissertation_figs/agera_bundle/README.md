# agera export bundle — plan v2 prediction study

Built by `scripts/dissertation_figs/v2/v2_build.py` on the analysis machine. The scripts read only this bundle;
`manifest.json` holds each file's row count and sha256, and the pipeline verifies them on startup.

## Files to copy to agera

Copy them into the repo checkout at `/nfs/research/compass/aalasandagutti/coScheduling`, keeping the same relative paths:

```
scripts/dissertation_figs/agera_bundle/   (this folder: 6 CSVs + manifest.json + README.md)
scripts/dissertation_figs/v2/agera_pipeline.py
scripts/dissertation_figs/v2/env_check.py
```

## Run on agera

```bash
cd /nfs/research/compass/aalasandagutti/coScheduling
PY=/nfs/student/a/aalasandagutti/hpcResearch/python_virtual_envs/pytorch-gpu/bin/python
export CUDA_VISIBLE_DEVICES=""

# 1. Environment check (versions, deterministic torch/sklearn/xgboost). Paste the output back.
$PY scripts/dissertation_figs/v2/env_check.py

# 2. Full run (resumable: re-running the same command skips finished tasks).
nohup $PY scripts/dissertation_figs/v2/agera_pipeline.py \
    --bundle scripts/dissertation_figs/agera_bundle \
    --out scripts/dissertation_figs/agera_out/v2_full --workers 78 \
    > scripts/dissertation_figs/agera_out/v2_full.log 2>&1 &
```

- `--workers` is capped at 80 in the code. Each worker uses 1 thread, and GPUs are never used.
- Progress lines go to the log. Failed tasks write a traceback to `agera_out/v2_full/errors/` and the exit code is 1;
  re-run the same command to retry only the unfinished tasks.
- Copy `scripts/dissertation_figs/agera_out/v2_full/` (and the log) back to the analysis machine at the same path.
  The scorer `v2/v2_score.py` runs there.

## Runs

- **Main run `v2_full_20261123`:** launched 2026-11-23 05:01 (agera time), 78 workers, 7,280 tasks. It uses the A4 grids,
  extended once after the B6 pilot edge check; see `GRIDS` in `agera_pipeline.py`, where each extension is marked `ext`.
- **Follow-up `v2_xgbdeep_20261123`** (director, 2026-11-22): the xgb tasks only, with the same splits and seeds and the
  full extended xgb grid plus max_depth {6, 8}:

  ```bash
  $PY scripts/dissertation_figs/v2/agera_pipeline.py --bundle scripts/dissertation_figs/agera_bundle \
      --out scripts/dissertation_figs/agera_out/v2_xgbdeep_20261123 --methods xgb --xgb-depths 1,2,4,6,8 --workers 78
  ```

  The scorer (`v2_score.py --xgb-rerun ...`) uses these xgb results. The main run's xgb (max_depth 1, 2, 4) is kept
  as method `xgb_d124`, a sensitivity row.
- **Package installs on agera:** none. env_check passed with the existing venv: py 3.10.12, numpy 2.1.3, pandas 2.2.3,
  sklearn 1.5.2, torch 2.3.0+cu121, xgboost 2.1.3.

## Contents

| File | Rows | What |
|---|---|---|
| `apps.csv` | 10 | Isolated 8-node base profile per app: mpi_time, comm_frac, total_msgs, total_bytes, runtime (canonical `jobs.csv`) |
| `panel_inhibitors.csv` | 9 | Isolated profile of the 9 panel inhibitor configs (canonical `inhibitors.csv`) |
| `panel_app.csv` | 90 | Per (app, panel config): rep-averaged JMs log ratio `y_mpi`, `y_tt` (app = victim); BD features `c_mpi`, `c_tt` |
| `appinh_resolved.csv` | 169 | Per (app, panel config, app socket `s_V`): JMs log ratios. Training rows for the absolute/delta models |
| `appapp_rows.csv` | 176 | Prediction rows: 166 socket-resolved non-self cells (victim, corunner, s_V) + 10 pooled self-pairs (s_V = 0.5) |
| `jms_factors.csv` | 20 | JMs factors per app and timer (d, CI, applied) |

## Definitions

- **Targets:** unclipped `y = ln(T_corun / T_base)`. Nothing is clipped anywhere.
  - `T_base` is the JMs baseline: the June isolated mean × exp(d_app) for socket-1 replicates.
  - d_app is applied only where its 95% t CI excludes 0 (AMG2023 and Quicksilver, both timers).
  - A cell that mixes sockets uses the replicate-weighted baseline.
- **Self-pairs:** which copy ran on which socket isn't recoverable, so each self-pair is one row. It pools both
  copies × 4 reps against a 50/50 baseline, with `s_V = 0.5`.
- **BD features** (inhibitor slowdown caused by the app). The inhibitor runs for a fixed duration, so this is its time
  per unit of work, co-run vs isolated, with no JMs on the inhibitor side:
  - `c_mpi = ln[(T_MPI/bytes)_co / (T_MPI/bytes)_iso]`
  - `c_tt = ln[(T_app/msgs)_co / (T_app/msgs)_iso]`
- **Socket bit:** aware runs use the victim's `s_V`; for the response models the co-runner or inhibitor gets 1 − s_V.
  Blind runs use a constant 0.

Design decisions D1–D10 are in `Dissertation/network_interference_modeling_plan_v2.md`. The pipeline's module docstring
documents the tiers, methods, grids and outputs.
