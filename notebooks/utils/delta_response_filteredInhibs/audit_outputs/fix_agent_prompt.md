# Task: Repair and Validate the Transitive Delta-Response Pipeline

Work on the transitive delta-response pipeline in
`notebooks/utils/delta_response_1/`. Begin by reading:

1. `audit_outputs/concerns.md`
2. `schematic.md`
3. Every implementation and test file directly under this pipeline directory

Design and implement the fixes, update documentation, add regression tests,
and verify the corrected pipeline end to end.

## Strict Workspace Boundary

- Do not read, search, inspect, modify, or execute files outside
  `notebooks/utils/delta_response_1/`.
- The only exception is read-only access to these training/profile CSVs:
  - `notebooks/utils/few_shot/data/jobs.csv`
  - `notebooks/utils/few_shot/data/inhibitors.csv`
  - `notebooks/utils/few_shot/data/job_inh.csv`
- Do not open or inspect `notebooks/utils/few_shot/data/pair.csv` for any reason.
- Do not inspect earlier implementations elsewhere in the repository.
- Do not use Git history or repository-wide searches.
- Write generated test and experiment artifacts only under
  `notebooks/utils/delta_response_1/audit_outputs/`.
- Implementation, tests, `schematic.md`, and `README.md` in
  `delta_response_1/` may be modified as needed.
- Use the `hpcResearch` Conda environment for every test and experiment.

## Holdout Policy

The existing real App-App dataset is historically contaminated by prior smoke
runs. Treat it as unavailable during this task.

- Do not use real App-App labels for design, tuning, testing, model selection,
  epoch selection, OOD calibration, debugging, or performance assessment.
- Do not read pair-derived files under `outputs/` or
  `audit_outputs/primary_full_default/`.
- Do not use post-holdout performance in `audit_outputs/concerns.md` to choose
  among alternative fixes. Those numbers document the problem but are not a
  tuning set.
- Use synthetic pair data for end-to-end and smoke tests.
- Do not claim that implementation changes improve App-App performance. A new
  untouched App-App dataset will be required for that conclusion.

## Confirmed Target Semantics

Observed slowdown is clipped at 1 when co-scheduling causes a speedup:

```text
observed_slowdown = max(1, latent_slowdown)
```

The operational prediction target in this pipeline is the clipped observed
slowdown. Predictions must therefore satisfy:

```text
predicted_slowdown >= 1
predicted_log_slowdown >= 0
```

The magnitude of latent speedups is not recoverable from the available CSVs.
Document this limitation. Design a principled boundary-aware formulation using
only App-Inhibitor data. At minimum, do not permit predictions outside the
known target support. Prefer a statistically coherent censored or two-part
approach over silently clipping only at report time, but keep the solution
proportionate to this small dataset and explain the chosen design.

## Required Fixes

Address every concern in `audit_outputs/concerns.md` that is fixable in code.
At minimum:

1. Prevent accidental real-holdout use during smoke and development runs.
   Add a clear `--skip-holdout` or training/CV-only mode. Make the documented
   smoke workflow synthetic or holdout-free. The pipeline must produce a useful
   completion report without loading a pair CSV.

2. Implement the clipped-target semantics consistently in training,
   inference, metrics, validation, saved predictions, and documentation.
   Preserve the architectural identity, antisymmetry, and path-consistency
   properties of the delta potential. Be explicit about whether those
   invariants apply to latent deltas before the clipped observation mapping.

3. Give low-rank delta, generic potential, and absolute-response comparators
   independent App-Inhibitor-only epoch selection and any model-specific
   validation they require. Keep computational budgets and seed treatment
   comparable. Do not use App-App outcomes.

4. Repair OOD calibration so distances compared at inference are in a common,
   mathematically valid physical-profile coordinate system. Do not pool raw
   Euclidean distances produced by different fold-local scalers. Preserve the
   requirement that model scalers in each fold exclude the held-out victim and
   held-out inhibitor block.

5. Replace exact-minimum aggregation selection with a prespecified robust rule
   that accounts for grouped validation uncertainty and favors the simpler
   method when differences are negligible. Use only App-Inhibitor folds.
   Document the rule precisely and make it deterministic.

6. Redesign uncertainty diagnostics so their meaning is explicit:
   - Do not present unweighted per-anchor spread as target-specific uncertainty.
   - Provide target-dependent weighted dispersion where mathematically valid.
   - Mark effective-anchor count as undefined for nonlinear median fallback,
     and report useful component diagnostics instead.
   - Do not label any uncalibrated spread as a prediction interval.

7. Make cluster bootstrap draws paired across methods. Sample original pair
   rows as clusters once per replicate, reuse the draw for every method, and
   report confidence intervals for paired method differences. Unit tests must
   prove both directional outcomes remain together.

8. Validate unordered pair uniqueness, including reversed duplicates. If pair
   replicates are supported, represent their cluster identity explicitly
   rather than silently treating duplicates as independent pairs.

9. Add finite-range validation around log predictions and exponentiation.
   Invalid or overflowing predictions must fail clearly rather than silently
   contaminating metrics.

10. Update `schematic.md` and `README.md` so target clipping, holdout policy,
    smoke usage, model-specific selection, OOD calibration, diagnostics, and
    paired inference match the implementation.

The historical holdout contamination and the previous negative App-App result
cannot be repaired in code. State that clearly rather than attempting to
optimize against them.

## Invariants and Leakage Requirements

Preserve and independently test all previously verified behavior:

- The real pair holdout is never required for training, CV, tuning, or smoke
  execution.
- Fold-local model scalers exclude the held-out victim profile and held-out
  inhibitor-block profiles.
- Crossed training excludes both the held-out victim and inhibitor block.
- Validation query identities and rows remain fixed and absent from training
  episodes.
- Victims are sampled uniformly.
- Anchor and query inhibitor IDs are distinct, including when response
  replicates exist.
- Uniform, median, kernel, and OOD-gated comparisons use the same low-rank
  checkpoint.
- Kernel and OOD distances use normalized physical profiles, never learned
  embedding geometry.
- OOD calibration and all epoch/model/aggregation choices use only
  App-Inhibitor validation data.
- Directional expansion and self-pair averaging remain correct.
- Bootstrap resamples original pair rows as clusters.
- Missing profiles are quarantined and reported.
- Genuine response replicates are preserved without accidental leakage or
  inappropriate averaging.
- Identity and antisymmetry remain exact, and path consistency remains exact up
  to floating-point tolerance for the latent potential deltas.

## Required Tests

Add focused tests that would fail if each leakage-sensitive exclusion or
correctness fix were removed. Include at least:

- Synthetic crossed-fold tests that inspect training rows and scaler fit rows.
- A test proving held-out query rows never reach `EpisodeSampler`.
- Statistical or deterministic tests for uniform victim sampling and distinct
  inhibitor IDs.
- Tests proving all low-rank aggregation methods receive the same model object
  or checkpoint.
- A test demonstrating OOD calibration remains valid when fold scalers differ.
- Boundary tests proving every reported clipped-target prediction is finite and
  at least 1.
- Tests for identity, antisymmetry, and path consistency before the observation
  boundary mapping.
- Tests for weighted dispersion and undefined median effective-anchor count.
- Tests proving paired bootstrap methods use identical cluster draws and keep
  both directions together.
- Tests rejecting duplicate and reversed unordered pair rows.
- Tests for missing profiles and genuine response replicates.
- A test proving `--skip-holdout` completes without opening any pair file.

Run all existing tests as well as the new tests with `hpcResearch`.

## End-to-End Verification

1. Run a reduced end-to-end experiment using deterministic synthetic jobs,
   inhibitors, App-Inhibitor responses, and pair data, all under
   `audit_outputs/`.
2. Run the same synthetic experiment twice and compare outputs. Outputs should
   match except for explicitly nondeterministic metadata such as elapsed time.
3. Run the largest practical App-Inhibitor-only crossed-validation experiment
   using `--skip-holdout`. Do not open the real pair CSV.
4. Verify all saved predictions and metrics are finite and obey the clipped
   target support.
5. Record every deviation from the default configuration.

## Implementation Approach

- First analyze the concerns and current design, then write a concise repair
  plan before editing.
- Prefer the smallest correct changes, but do not preserve flawed behavior only
  for backward compatibility.
- Avoid tuning an unnecessarily large model or feature grid. There are only ten
  victim profiles.
- Keep all selection deterministic and record requested and selected choices in
  output reports.
- Do not remove valid diagnostics merely because they exposed poor behavior;
  correct their definitions and labels.
- Do not delete or overwrite historical audit outputs.

## Final Response

Report:

1. Design decisions and how each concern was addressed.
2. Exact files changed and key line references.
3. Tests and experiment commands run, with pass/fail results.
4. Synthetic reproducibility results.
5. App-Inhibitor-only validation results and configuration deviations.
6. Concerns intentionally left unresolved and why, especially historical
   holdout contamination and the need for a new App-App dataset.
7. Any behavior changes or migration considerations.

Do not report new real App-App performance because the real pair holdout must
remain unopened throughout this task.
