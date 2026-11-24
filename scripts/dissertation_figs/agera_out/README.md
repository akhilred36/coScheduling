# agera_out: raw outputs of the plan v2 prediction runs

The raw per-task outputs (`preds/`, `hp/`, `done/`; 166 MB in total) are **not in git**. The full copies live in two places:

- agera: `/nfs/research/compass/aalasandagutti/coScheduling/scripts/dissertation_figs/agera_out/`
- Akhil's analysis machine: `/home/akhil/hpcResearch/repos/coScheduling/scripts/dissertation_figs/agera_out/`

Only each run's `<run>.log`, `<run>/run_config.json` and `<run>/status.json` are committed here.

| Run | Tasks | Notes |
|---|---|---|
| `v2_full_20261123` | 7,280 | Main run, 78 workers, 0 failures, 14,303 s |
| `v2_xgbdeep_20261123` | 1,040 | xgb follow-up (max_depth 1, 2, 4, 6, 8); replaces the main run's xgb in scoring |
| `pilot_a`, `pilot_b`, `pilot2_a`, `pilot2_b` | small | B6 pilots: grid-edge and determinism checks |

The scored results that the chapter uses are in `../output/v2_pred_v2_full_20261123*`, and those files are enough to
regenerate every figure. You need the raw outputs only to re-score. See
`.claude/skills/coscheduling-data-analyzer/references/v2_handoff.md`.
