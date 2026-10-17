# Delta-Response Repair Plan

1. Enforce observed slowdown support at load, prediction, exponentiation, and
   metric boundaries. Train latent potentials with censored Huber losses:
   exact losses above the floor and one-sided constraints for observations at
   the floor. Apply `max(0, latent_log_slowdown)` only as the observation map;
   latent potential deltas retain identity, antisymmetry, and path consistency.
   Use only uncensored anchors for latent residual calibration.
2. Add `--skip-holdout`, make it complete after App-Inhibitor selection and
   final fitting, and ensure it never calls the pair loader. Move documented
   smoke runs to synthetic data or this holdout-free mode.
3. Cross-validate low-rank, generic-potential, and absolute-response models
   independently with equal fold, seed, epoch, and batch budgets. Select each
   model's configuration and final epoch count using only crossed
   App-Inhibitor validation. Select generic aggregation independently.
4. Keep fold-local model scalers leakage-safe, but compute every OOD and kernel
   distance under one frozen inhibitor-only physical-profile scaler. Save that
   scaler and calibration provenance.
5. Replace exact-minimum primary aggregation selection with a deterministic
   grouped one-standard-error rule. Compute victim/block fold means after seed
   averaging, admit candidates within one standard error of the best candidate,
   then choose by the prespecified complexity order uniform, median, kernel,
   OOD-kernel and deterministic parameter order.
6. Rename unweighted anchor spread as an uncalibrated residual diagnostic. Add
   weighted residual dispersion for linear rules. For median fallback, mark
   aggregate effective-anchor count undefined and report kernel-component
   effective count/dispersion plus fallback MAD.
7. Generate bootstrap cluster draws once, reuse them across methods, preserve
   both directional records in every sampled pair cluster, and emit paired
   method-difference confidence intervals.
8. Reject duplicate unordered pair keys, including reversed duplicates. Keep
   App-Inhibitor response replicates with explicit replicate IDs.
9. Add focused tests for crossed exclusions, fixed queries, sampler behavior,
   shared low-rank checkpoints, common-coordinate OOD calibration, clipped
   finite predictions, latent invariants, diagnostics, paired bootstrap,
   unordered pair uniqueness, quarantine/replicates, and skip-holdout.
10. Update `schematic.md` and `README.md`, run all tests in `hpcResearch`, run
    deterministic synthetic experiments twice, and run a practical real
    App-Inhibitor-only crossed-validation experiment without opening pair data.

Historical App-App contamination and the prior post-holdout result cannot be
repaired in code. No design choice or validation in this repair will use those
outcomes; a newly collected untouched App-App dataset is required for any new
transfer-performance conclusion.
