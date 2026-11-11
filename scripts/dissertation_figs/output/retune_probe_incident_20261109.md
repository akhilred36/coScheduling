# Re-tune probe stage A failure (2026-11-09 22:35) and relaunch (2026-11-10 19:18)

## Root cause (deterministic, not concurrency)
- `pipeline.py` (copied socket-aware BD pipeline) line ~1937: after crossed CV it re-reads `ablation_summary.csv`.
- `write_validation_reports` writes that file from `ablation_contrasts()`. These are within-run method pairs, e.g.
  joint_profile_calibrated vs joint_profile_direct.
- The probe runs **one variant per process**. Six variants have no internal contrast:
  legacy_delta/generic/absolute, bidirectional_profile/profile_wall/context_forward.
  - Their CV writes an empty DataFrame, so the file holds 1 byte, `\n`. The read then raises `EmptyDataError`
    after all 200 fits.
  - joint_profile and bidirectional_context each have a calibrated-vs-direct contrast (floor_aware_calibration,
    context_calibration), so they finished.
- The seed-batch restricted scorer (20:38) only reads seed_batch outputs. It did not touch retune_probe files.

## Validity of the two rc=0 runs
- Both runs: status complete, 200 folds, epochs_run = 240 for all folds (early stopping off), single config,
  λ = 1.0 (= blind selection).
- Held-out blocks are identical to blind_bd_mpi.

## Fix
- **Pipeline guard:** `pd.errors.EmptyDataError` → `validation_ablation = None` (copy only). This path was already
  handled for fixed_global runs. Verified with a 2-epoch smoke test of legacy_generic: rc 0, selection.json written.
  Backup at scratchpad `pipeline_bd_sa.bak`.
- **Failed runs:** moved to `retune_probe/runs/_failed_20261109/`. The old chain.log and done.tsv are in
  `retune_probe/_old_20261109/`.
- **Retune chain:** A (6 remaining jobs, 6 workers) and B (2 workers) now run concurrently, then build C, then C.
  It verifies selection.json and run completion, and writes "probe FAILED: <reason>" on error.
- **Capacity chain:** waits for "probe (done|FAILED)" and exits on FAILED.

## Preliminary (the two completed CV runs)
| variant | protocol epochs (median per-fold best) | fold-avg argmin | curve @1 / @7 / @30 / @240 | folds best@1 / best≥200 |
|---|---|---|---|---|
| joint_profile | 37 | 1 | 0.178 / 0.191 / 0.197 / 0.265 | 22% / 11% |
| bidirectional_context | 20 | 1 | 0.182 / 0.195 / 0.205 / 0.251 | 22% / 10% |
