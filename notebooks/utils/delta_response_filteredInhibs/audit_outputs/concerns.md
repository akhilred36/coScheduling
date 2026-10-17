# Transitive Delta-Response Audit Concerns

## 1. The App-App holdout is not historically sealed

Severity: critical

The current pipeline opens `pair.csv` only after crossed validation, selection,
and final training (`pipeline.py:748-775`). However, the documented smoke command
uses the real default `pair.csv` (`README.md:22-29`), and
`outputs/smoke/run_report.json:2-5` confirms that an earlier smoke run evaluated
all 55 pair rows and 110 directional outcomes.

Consequences:

- `pair.csv` cannot be considered a pristine one-time confirmatory holdout.
- Prior App-App exposure may have influenced later development, even if the
  present implementation does not directly leak labels during fitting.
- The full audit result is useful as a post-holdout evaluation, but it cannot
  support a clean first-look confirmatory claim.

Recommended action: make smoke runs use synthetic data or add a
`--skip-holdout` mode. Use a newly collected App-App dataset for future
confirmatory evaluation.

## 2. The primary method fails the important baselines

Severity: high

The App-Inhibitor selection rule chose OOD-gated delta aggregation as the
primary method. On the full default App-App run its repeated-self log-MAE was
`0.2233`, compared with:

- `0.1058` for constant slowdown `1`.
- `0.1514` for nearest anchor.
- `0.1865` for absolute-response regression.
- `0.1822` for single-anchor delta.
- `0.2099` for ordinary kernel delta.

Paired bootstrap comparisons over the 55 original pair rows show that the
primary method is credibly worse than constant `1`, nearest anchor,
single-anchor delta, ordinary kernel delta, and the generic potential. Its
difference from absolute regression is not clearly distinguishable from zero.
See `audit_outputs/post_holdout/paired_log_mae_comparisons.csv`.

The result does not support a claim that transitive anchor aggregation improves
over absolute regression or the simple baselines.

## 3. Slowdown is clipped at 1 but modeled as unconstrained regression

Severity: high

Slowdown values equal to `1` are confirmed to be clipped observations where
co-scheduling caused a speedup. Therefore the observed target is
left-censored:

```text
observed_slowdown = max(1, latent_slowdown)
```

The implementation instead applies ordinary Huber regression to log slowdown
(`training.py:108` and `training.py:177`) and permits predictions below `1`.
The full holdout contains 28 clipped outcomes among 110 directions, while the
App-Inhibitor data contains 249 values equal to `1` among 2,339 raw rows. The
primary method generated five predictions below `1`.

This has two distinct implications:

- If the objective is the clipped operational target, predictions should obey
  `predicted_slowdown >= 1`, and a two-part or boundary-aware model is more
  appropriate.
- If the objective is latent interference including speedup magnitude, the
  clipped CSVs have discarded the information needed to estimate that target.

Constant `1` receives zero error on every clipped outcome. This does not fully
explain its advantage: among the 82 outcomes above `1`, its log-MAE was still
`0.142`, versus `0.165` for nearest anchor and `0.205` for the primary method.

Clipping predictions now would be a post-holdout analysis and must not replace
the frozen primary result. A corrected formulation should be selected only on
App-Inhibitor data and evaluated on a new untouched holdout.

## 4. Comparator training is not independently selected

Severity: high

Only the low-rank delta model participates in crossed-validation epoch
selection (`pipeline.py:295-319` and `pipeline.py:405-406`). The resulting
median of 12 epochs is then imposed on low-rank, generic, and absolute models
(`pipeline.py:430-463`). Absolute regression has a different objective and may
require a different training duration.

Absolute-response log-MAE varied from `0.0993` to `0.3436` across five seeds,
with a standard deviation of `0.1025`. This makes the central comparison
against absolute regression unstable and potentially unfair.

Recommended action: independently select epochs and relevant hyperparameters
for each comparator using only crossed App-Inhibitor validation, with the same
computational budget and seed policy.

## 5. OOD calibration mixes incompatible distance scales

Severity: high

Each crossed fold fits a different profile scaler (`pipeline.py:268-273`). Its
query-to-support distances are then pooled into one global reference
distribution (`pipeline.py:291-293` and `pipeline.py:362-369`). Final App-App
distances are computed under another scaler fitted to all profiles
(`pipeline.py:423-426` and `inference.py:109-119`).

Euclidean distances from these different standardized coordinate systems are
not directly comparable. Although all calibration data originates from
App-Inhibitor folds and therefore does not leak App-App labels, the resulting
OOD percentiles and thresholds are not calibrated in a common metric space.

Only 11 of 110 final primary predictions activated the OOD gate. OOD gating was
significantly worse than ordinary kernel aggregation despite its validation
advantage.

Recommended action: calibrate dimensionless within-fold distance ranks or
ratios, or recompute all profile-only reference distances under one frozen
physical-profile scaler.

## 6. OOD selection is based on a negligible validation difference

Severity: medium

The selected OOD method achieved crossed-validation log-MAE `0.16126`, versus
`0.16165` for ordinary kernel aggregation. The difference, `0.00039`, is about
0.24 percent and was treated as decisive without uncertainty estimation or
nested selection (`pipeline.py:382-402`).

App-App performance reversed the ordering, and paired bootstrap analysis found
OOD gating worse than kernel by `0.0134` log-MAE with a 95 percent interval of
`[0.0014, 0.0296]`.

Recommended action: use grouped nested validation or a prespecified
uncertainty/one-standard-error rule that favors the simpler method when the
validation difference is negligible.

## 7. Anchor-spread diagnostics are not target-specific

Severity: medium

For each anchor, the prediction is computed as a common target potential plus
an anchor residual (`inference.py:107-109`). Standard deviation and MAD are
translation-invariant, so the reported unweighted spread depends mainly on the
victim's residual distribution and not on the target aggressor
(`inference.py:127-130`).

Anchor standard deviation had a strong overall correlation with error, but
that association fell substantially after controlling by victim. MAD was
constant within victims. These diagnostics can identify difficult victims but
do not provide meaningful target-specific uncertainty.

Recommended action: report target-dependent weighted dispersion where linear
weights exist and calibrate uncertainty from held-out App-Inhibitor residuals.
Do not interpret raw anchor spread as a prediction interval.

## 8. Effective-anchor count is missing for gated median fallback

Severity: medium

The implementation correctly computes `1 / sum(w^2)` when finite linear
weights exist (`inference.py:124-126`). When OOD gating blends toward the median,
the code records weights as `NaN` (`inference.py:80-84`). Consequently,
effective-anchor count is absent for all 11 predictions on which the OOD gate
actually activated.

The `NaN` is more honest than inventing linear weights for a nonlinear median,
but it means the required diagnostic is unavailable in the cases where it is
most needed. Reports should explicitly label this quantity as undefined and
provide a separate diagnostic for the kernel component or robust fallback.

## 9. Built-in bootstrap intervals are not paired across methods

Severity: medium

`metrics.py:81-90` samples the correct original pair-row clusters, and the full
run reports 55 clusters. However, the random generator advances separately for
each method, so methods do not use identical bootstrap samples.

Marginal method intervals are valid individually, but they do not directly
answer whether one method improves over another. Paired cluster bootstrap
differences were added independently under
`audit_outputs/post_holdout/paired_log_mae_comparisons.csv`.

Recommended action: generate cluster draws once, reuse them for all methods,
and report confidence intervals for paired metric differences.

## 10. Validation is predictive of victim difficulty but optimistic for transfer

Severity: medium

Crossed-validation versus App-App log-MAE was:

| Method | Crossed validation | App-App | Increase |
|---|---:|---:|---:|
| Delta uniform | 0.198 | 0.290 | 0.092 |
| Delta median | 0.198 | 0.293 | 0.095 |
| Delta kernel | 0.162 | 0.210 | 0.048 |
| Delta OOD kernel | 0.161 | 0.223 | 0.062 |

Per-victim primary validation and App-App errors had Spearman correlation
`0.818`, so validation identifies relatively difficult victims. However, it
systematically understates final error and did not correctly choose between
kernel and OOD kernel. Strong inhibitor-fold performance is therefore not
sufficient evidence of synthetic-inhibitor-to-application transfer.

## 11. Pair loading does not reject duplicate unordered pairs

Severity: low

`data.py:179-194` validates columns, known application IDs, and positive finite
slowdowns, but it does not reject duplicate unordered application pairs or
reversed duplicates. The current dataset has the expected 55 distinct rows, so
this did not affect the audit run.

Recommended action: canonicalize unordered pair keys, assert uniqueness, and
represent genuine pair replicates explicitly so bootstrap clustering remains
correct.

## 12. Final exponentiation has no finite-range guard

Severity: low

Final predictions are exponentiated directly (`pipeline.py:641-643` and
`pipeline.py:667`). Extreme OOD log predictions could overflow before metrics
are computed. All predictions in the full audit run were finite, so this is a
latent numerical risk rather than an observed failure.

Recommended action: validate finite log predictions and fail explicitly before
exponentiation.

## Verified Areas

The following areas passed code inspection and independent tests:

- Within the current pipeline invocation, `pair.csv` is loaded only after
  selection and final model fitting.
- Fold-local model scalers exclude the held-out victim and inhibitor block.
- Crossed training rows exclude both held-out dimensions.
- Validation support and query sets are fixed and absent from training.
- Victims are sampled uniformly.
- Anchor and query inhibitor IDs are distinct.
- All low-rank aggregation comparisons use the same checkpoint.
- Kernel distances use standardized physical profiles rather than embedding
  geometry.
- OOD calibration provenance is App-Inhibitor-only, despite the scale problem
  described above.
- Final epoch selection does not use App-App outcomes.
- Directional expansion and self-pair averaging are correct.
- Bootstrap sampling preserves the 55 original pair rows as clusters.
- Missing-profile rows are quarantined; all 12 rejected rows had missing
  inhibitor profiles.
- Duplicate App-Inhibitor responses would be retained as replicates; the
  current data contains no duplicate keys.
- Identity and antisymmetry errors were exactly zero for all five final
  checkpoints. Maximum path-consistency error was `1.49e-8`.
- All four existing unit tests and nine independent audit tests passed in the
  `hpcResearch` environment.
- Two identical synthetic end-to-end runs produced byte-identical artifacts
  except for elapsed wall time.

## Overall Assessment

The implementation has strong architectural invariants and generally correct
crossed-fold exclusions. No direct App-App label leakage was found inside a
single normal pipeline run. The main concerns are scientific-protocol and
statistical issues: the real holdout has already been used by smoke testing,
the clipped target is modeled incorrectly, OOD calibration compares distances
from incompatible scalers, comparator selection is not controlled fairly, and
the selected primary method performs materially worse than simple baselines.
