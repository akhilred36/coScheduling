# Static Random-Split Evaluation Instructions

## Role And Mission

You are the execution agent for the retrospective Delta Response 2 evaluation.
Carry the work through implementation, verification, execution, analysis, and a
final report. Do not stop after proposing a plan.

Evaluate the frozen Delta Response 2 model on exactly the historical static
`random_split` App-App dataset used for the final Delta Response 1 analysis.
The purpose is to measure the new model's transfer behavior under the same pair
membership, direction expansion, metrics, and pair-cluster bootstrap, then test
the design hypotheses that motivated Delta Response 2.

Run only `random_split`. Do not run `one_known`, `zero_shot`, or
training-application-count experiments.

## Required Reading

Read these two files in full before using tools or writing evaluation code:

1. `README.md` explains the environment, pipeline commands, immutable root,
   completed run, artifacts, and replay interface.
2. `schematic.md` explains the architecture, censored objective, correction
   families, selection design, and the exact candidate that was frozen.

Then read:

3. `scientific_spec.md` for the development estimands and selection policy.
4. `experiments/compact_v1/FINAL_REPORT.md` for the completed App-Inhibitor
   result.
5. `experiments/compact_v1/frozen_recipe.json` for the exact final recipe and
   checkpoint roster.
6. `../delta_response_1/experiments/random_split_model_search_v1/final_random_split/run_report.json`
   for the inherited evaluation provenance.

Do not infer the frozen winner from the general architecture alone. The selected
Delta Response 2 recipe is the legacy exact-anchor OOD candidate
`current_delta_ood_h8_e4`; the all-anchor censored model is a prespecified
secondary design diagnostic.

## Evidence Label

The historical `pair.csv` has already been opened by previous research. This
task explicitly redesignates it as a retrospective static evaluation dataset.
It is not an untouched holdout.

Every output, table, plot, and conclusion must use language such as:

```text
retrospective static random_split evaluation
historical diagnostic dataset
development-only transfer evidence
```

Never use:

```text
untouched test set
unbiased generalization estimate
confirmatory App-App result
```

A newly collected immutable App-App holdout remains necessary for a fresh
transfer claim.

## Exact Evaluation Contract

Reproduce this contract exactly:

| Item | Required value |
| --- | --- |
| Evaluation method | `random_split` |
| Training applications | All ten applications |
| Unknown applications | None |
| Valid App-Inhibitor fitting rows | 2,623 |
| Final model seeds | `0, 1, 2, 3, 4` |
| Static unordered pair rows | 55 |
| Self-pair rows | 10 |
| Non-self pair clusters | 45 |
| Directional rows including self | 110 |
| Primary directional rows | 90 non-self directions |
| Primary aggregation | Mean latent prediction across seeds, then clip |
| Primary metric | Non-self log MAE |
| Bootstrap unit | Original unordered `pair_cluster_id` |
| Bootstrap draws | 1,000 |
| Bootstrap seed | 923 |
| Confidence interval | Percentile 2.5% and 97.5% |

`random_split` is a historical name. Do not randomly split the 55 pair rows.
All ten applications are known during model fitting, and every validated static
pair is evaluated only after the model choices are frozen.

## Fixed Paths And Hashes

Run from `notebooks/utils`.

```text
Delta Response 2 package:
  delta_response_2/

Frozen development root:
  delta_response_2/experiments/compact_v1/

Static input directory:
  few_shot/data/

Historical Delta Response 1 result:
  delta_response_1/experiments/random_split_model_search_v1/final_random_split/

New evaluation output root:
  delta_response_2/experiments/static_random_split_evaluation_v1/
```

Expected input hashes:

| Input | SHA-256 |
| --- | --- |
| `jobs.csv` | `356a3136a2d32527e75eada31841807d0ccc13dfe48ad0540664666938c0179c` |
| `inhibitors.csv` | `a8db332145b560f51c2176bc8378a423cb1a318bb9a7bfcc31788b1addf12573` |
| `job_inh.csv` | `cb721f466161e816e3a2711b9e6d807b4c7b72113af506a0eca4d483513d2118` |
| `pair.csv` | `369381f33b9689bc7fbc2a5b3d8d285fc4a604c5480968391c4c84a9b24b20c0` |

Expected archived Delta Response 1 hashes, relative to
`final_random_split/`:

| Archived input | SHA-256 |
| --- | --- |
| `directional_predictions.csv` | `85f0cecfc55e3dee634ecf9ad9598b957f45378e5d4e0865f84c260289fb2725` |
| `evaluation_pair_manifest.csv` | `0eb6822506fdd20f1c5cfaa110fede545498fc81200568dae1949f685d03c8f1` |
| `requested_run_config.json` | `a191f6c85f66b1c39d28f8df22417ae2e5738412beb42f7f43e85e56e5e9d736` |
| `selection.json` | `e3276b0ff0c8fd41baa1ba0f02a927ccbd53fc2ce40e9b1eeeba6a3567272cb2` |
| `run_report.json` | `21371fe7f8a684747b7ea56f4ab7a305430b7a52ac0f5a866c05536a23522a2a` |

Expected frozen-root identifiers:

```text
selected candidate:
  current_delta_ood_h8_e4__de045c14efb9

frozen_recipe.json SHA-256:
  3b95393dba19ddb851bf974e793b1669d769f6cd6b11388ba7818d486e9d8628

plan seal SHA-256:
  1cd46f81278203e1c565c063c8bc95eee69784067935a40433c4ae3095aea340

completion seal digest:
  114c3003053cd89ea7e656dc09ca84202cf09d715e708af8a0fabf72a82e116f
```

Treat any mismatch as a blocker. Do not silently evaluate a different dataset,
recipe, or source tree.

## Non-Negotiable Boundaries

- Do not modify any top-level `delta_response_2/*.py` file before evaluating the
  existing frozen root. Its source hashes are part of the plan seal.
- Do not modify `compact_v1`, including its reports, manifests, checkpoints, or
  output hash roster.
- Do not place evaluation code or output inside `compact_v1`; its exact roster
  intentionally rejects undeclared files.
- Do not read, hash, inspect, or summarize `pair.csv` until the pre-access
  preparation phase below is complete.
- Do not use App-App outcomes to choose candidates, train models, set priors,
  calibrate scale, choose checkpoints, alter OOD thresholds, or define strata.
- Do not rerun a capacity or optimizer search after seeing static-pair outcomes.
- Do not use directional rows as independent bootstrap units.
- Do not pool self pairs into the primary result.
- Do not overwrite the archived Delta Response 1 outputs.
- Do not import Delta Response 1 model or inference runtime into the new model.
  Archived predictions may be read only for post-freeze comparison.

## Phase 0: Verify The Frozen Development Result

Run the redesigned tests:

```bash
conda run -n hpcResearch python -m unittest discover \
  -s delta_response_2/tests -v
```

Verify the completed root using the CPU visibility with which it was sealed:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n hpcResearch \
  python -m delta_response_2.run_experiments status \
  --root delta_response_2/experiments/compact_v1
```

Require all of the following before continuing:

- Status is `complete`.
- `app_app_file_opened` is `false`.
- Candidate count is 53.
- Fold schemes are `profile` and `mechanism`.
- Seeds are `0,1,2,3,4`.
- Selected candidate is `current_delta_ood_h8_e4__de045c14efb9`.
- Scientific acceptance is true.
- Exact output-roster and hash verification succeeds.

Run a profile-only replay smoke check:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n hpcResearch \
  python -m delta_response_2.run_experiments predict-profile \
  --root delta_response_2/experiments/compact_v1 \
  --victim-id amg \
  --aggressor-id beatnik \
  --device cpu
```

The sealed reference prediction has ensemble latent log slowdown approximately
`0.7722588152923945` and slowdown `2.1646502808078`. This is a profile-only
replay checksum, not a comparison with an App-App label.

If the completed root is absent, stop this protocol. Do not recreate it under
the same path and do not substitute a reproduction for the fixed identifiers
above. A reproduction requires a separately versioned evaluation protocol,
output root, and preregistered frozen-root hashes.

## Phase 1: Build And Freeze Evaluation Machinery Without Pair Data

The current model package intentionally has no App-App loader. Implement an
evaluation-only harness under a subdirectory such as:

```text
delta_response_2/static_evaluation/
```

Do not add or edit a top-level package Python file. The harness may import
Delta Response 2 data transforms, model classes, censoring, inference, frozen
recipe verification, and metric primitives. It must not import Delta Response 1
training or model code.

Give the harness three non-interactive stages:

```text
prepare   verify the model, freeze methods, fit any diagnostic checkpoints,
          and seal evaluation code/config without opening pair.csv
evaluate  verify the preparation seal, then hash/open the static pair table,
          predict, score, bootstrap, report, and seal outputs
status    verify every evaluation artifact and print a concise report
```

Before `pair.csv` is accessed, create
`static_random_split_evaluation_v1/evaluation_plan.json` containing:

- The evidence label from this document.
- Absolute input paths and the expected hashes above.
- Frozen model root and frozen recipe hash.
- Exact method panel.
- Exact saved method aliases and candidate IDs.
- Seed ensemble rule.
- Separate seed-row and ensemble-row schemas.
- Pair membership and direction rules.
- Primary and secondary scopes.
- Complete metric list.
- Bootstrap unit, count, seed, and interval definition.
- Expected archived Delta Response 1 paths and hashes.
- A declaration that no App-App outcome has yet been opened.
- Hashes of all evaluation-only source files.
- Hashes of every newly fitted diagnostic checkpoint.

Test pair parsing, direction expansion, metric calculations, and paired
bootstrap on synthetic fixtures before sealing the preparation phase.

## Prespecified Method Panel

### Primary New-Model Method

The only primary Delta Response 2 method is the already frozen recipe:

```text
delta2_selected = current_delta_ood_h8_e4__de045c14efb9
```

Use its five existing checkpoints. Do not retrain it.

For every direction, preserve the ordered seed-specific latent predictions,
average them in latent log space, clip the ensemble at zero, and exponentiate.
Do not average clipped predictions or raw slowdowns.

### Pair-Independent Baselines

Compute these without fitting to App-App outcomes:

```text
delta2_constant_1
delta2_victim_inhibitor_median
delta2_nearest_anchor
```

Use all available *uncensored* App-Inhibitor anchors for the victim, exactly as
Delta Response 1 did. `delta2_victim_inhibitor_median` is the median observed
log slowdown over those anchors. `delta2_nearest_anchor` is the observed log
slowdown of the uncensored anchor nearest to the query application profile.
Define both algorithms before pair access. Physical distance must use the
frozen inhibitor-only base-feature distance scaler.

### Secondary Delta Response 2 Design Diagnostics

These methods are secondary retrospective diagnostics, not alternatives from
which to select after evaluation. Use these exact saved aliases and sealed
candidate IDs:

| Saved method | Sealed candidate ID |
| --- | --- |
| `delta2_profile_only_ood_h32_e16` | `current_delta_ood_h32_e16__726110c4a776` |
| `delta2_censored_zero_h32_e16` | `censored_correction_delta_h32_e16_zero_l10p0__e12c04b3e734` |
| `delta2_query_shrinkage_h32_e16` | `censored_correction_delta_with_query_shrinkage_h32_e16_l10p0_g0p75__2571628c7df2` |
| `delta2_absolute_neural_h32_e16` | `regularized_absolute_neural_h32_e16__a452e916ac46` |
| `delta2_absolute_linear_r0p01` | `regularized_absolute_linear_r0p01__a7f8020fd65b` |

`delta2_profile_only_ood_h32_e16` is also the minimum
`profile_unweighted_log_mae` candidate among the sealed admissible candidates.
This designation is fixed from App-Inhibitor development results, not from the
static outcomes.

Fit missing final diagnostic models during `prepare`, before pair access:

- Use all 2,623 valid App-Inhibitor rows.
- Use seeds `0,1,2,3,4` for neural models.
- Use the sealed base profile transform and final scaler policy.
- Use rank 2, 80 optimizer steps, batch size 128, learning rate `0.003`, and
  weight decay `0.001`.
- Validate saved parameters and profile-only predictions as finite.
- Save and hash every checkpoint.
- Reuse a shared `32/16` potential fit for the fixed legacy and censored
  correction variants; only their prespecified inference correction differs.
- Compute population priors independently for each seed and the fixed query
  gamma where required.
- Fit and replay every new diagnostic on CPU with CUDA hidden. `prepare` must
  reject a non-CPU device and record the device and environment in its state.

The diagnostic checkpoint directory has this exact roster:

```text
diagnostic_checkpoints/
  profile_scaler.json
  distance_profile_scaler.json
  delta_h32_e16_seed0.pt
  delta_h32_e16_seed1.pt
  delta_h32_e16_seed2.pt
  delta_h32_e16_seed3.pt
  delta_h32_e16_seed4.pt
  delta_h32_e16_training_curves.csv
  delta_h32_e16_ood_reference.csv
  absolute_neural_h32_e16_seed0.pt
  absolute_neural_h32_e16_seed1.pt
  absolute_neural_h32_e16_seed2.pt
  absolute_neural_h32_e16_seed3.pt
  absolute_neural_h32_e16_seed4.pt
  absolute_neural_h32_e16_training_curves.csv
  absolute_linear_r0p01.json
  artifact_hashes.json
  preparation_state.json
```

Each shared potential checkpoint must bind its seed, architecture, candidate
IDs, and the population correction prior for query gamma `0.75`.
`preparation_state.json` must bind the exact candidate-to-artifact mapping,
training settings, CPU device, scaler files, and OOD reference.
`artifact_hashes.json` must contain exactly the sixteen model, scaler, curve,
and OOD-reference files above; the preparation seal separately binds both
metadata JSON files. Reject missing, extra, non-regular, or symlink entries.

Do not add another candidate after static outcomes are visible. Report the
primary method first and keep this panel labeled as design diagnosis.

### Archived Delta Response 1 Comparators

After new predictions are frozen, verify all five archived hashes in the fixed
hash table above. Only then read the archived directional predictions from:

```text
delta_response_1/experiments/random_split_model_search_v1/
  final_random_split/directional_predictions.csv
```

Use its `anchor_budget=all` rows to compare against the exact inherited methods:

```text
constant_1
victim_inhibitor_median
nearest_anchor
absolute_response
delta_single_anchor
delta_uniform
delta_median
delta_kernel
delta_ood_kernel
generic_potential
```

Do not retrain Delta Response 1. Merge archived and new predictions by
`pair_row_id`, `pair_cluster_id`, `direction`, `victim_id`, and `aggressor_id`.
Require identical true outcomes after the merge. Preserve the archived name in
an `archive_method` column and save its comparison alias as
`delta1_<archive_method>` so inherited and newly recomputed baselines cannot
collide. Save the filtered, namespaced 110-row-per-method archive frame as
`archived_delta1_comparison.csv`; do not append those rows to
`directional_predictions.csv`. For ensemble metric and bootstrap tables,
concatenate the validated archive frame with the Delta Response 2 directional
frame so every cross-generation comparison uses the same scopes and shared
cluster draws.

## Phase 2: Open And Validate The Static Pair Dataset

Only after `prepare` is complete and sealed may `evaluate` access:

```text
few_shot/data/pair.csv
```

First compute its SHA-256 and require the expected value. Then validate:

- Required columns are `jobA_id`, `jobB_id`, `slowdown_A`, and `slowdown_B`.
- Both endpoint IDs exist in the ten sealed application profiles.
- Both slowdown columns are finite and at least one.
- No unordered pair is duplicated, including a reversed duplicate.
- There are exactly 55 rows.

Assign row membership exactly as Delta Response 1 did:

```text
pair_row_id = original validated row position, 0 through 54
pair_cluster_id = pair_row_id
evaluation_method = random_split
jobA_known = true
jobB_known = true
known_endpoint_count = 2
included = true
```

Write `evaluation_pair_manifest.csv` before prediction. It must match the
archived manifest on row ID and endpoints.

Expand every pair into two directions:

```text
direction A:
  victim_id = jobA_id
  aggressor_id = jobB_id
  true_slowdown = slowdown_A

direction B:
  victim_id = jobB_id
  aggressor_id = jobA_id
  true_slowdown = slowdown_B
```

Both directions retain the same `pair_cluster_id`. Preserve both directions of
self pairs; they have the same profile prediction but may have different
observed values.

## Prediction Contract

`seed_predictions.csv` and `directional_predictions.csv` have different row
units. Do not mix seed rows and ensemble rows.

`seed_predictions.csv` contains one row per seeded Delta Response 2 method,
direction, and seed. Its required columns are:

```text
evaluation_method
pair_row_id
pair_cluster_id
direction
victim_id
aggressor_id
self_pair
method
candidate_id
seed
latent_predicted_log_slowdown
predicted_log_slowdown
predicted_slowdown
true_log_slowdown
true_slowdown
```

The seeded methods are `delta2_selected`,
`delta2_profile_only_ood_h32_e16`, `delta2_censored_zero_h32_e16`,
`delta2_query_shrinkage_h32_e16`, and
`delta2_absolute_neural_h32_e16`. Analytic baselines and the deterministic
linear model do not appear in this file.

`directional_predictions.csv` contains exactly one prediction per Delta
Response 2 method and direction. It has the same columns except that `seed` is
absent and these columns are present:

```text
ensemble_seed_count
model_seed_log_std
```

For a seeded method, set `ensemble_seed_count=5` and compute
`model_seed_log_std` from its five latent predictions with `ddof=0`. For an
analytic or deterministic method, set `ensemble_seed_count=1` and
`model_seed_log_std=0`. Use `candidate_id=not_applicable` for the three analytic
baselines.

Save correction diagnostics where applicable:

```text
anchor_count
exact_anchor_count
floor_anchor_count
nearest_anchor_distance
nearest_exact_anchor_distance
ood_alpha
correction
correction_prior
correction_active_floor_violations
```

Required invariants:

- Every required prediction and target value is finite. An inapplicable
  optional diagnostic is blank; every present diagnostic value is finite.
- `predicted_log_slowdown >= 0`.
- `predicted_slowdown >= 1`.
- Every method in `directional_predictions.csv` covers all 110 directions
  exactly once.
- Every seeded method in `seed_predictions.csv` covers every direction exactly
  once for each seed in `0,1,2,3,4`.
- Each seeded directional value equals the mean of its five latent seed
  predictions followed by one clip. Deterministic methods equal their direct
  prediction.
- The two structural self directions receive identical predictions for a given
  method and, where applicable, seed.
- No true outcome is passed to any prediction function.

Spot-check the external evaluator against `predict-profile` for at least ten
non-self endpoint directions before computing metrics. Require agreement within
floating-point tolerance.

## Metrics And Scopes

Use the same primary metrics as Delta Response 1:

```text
log_mae
log_mse
log_rmse
raw_mae
raw_mape
raw_mse
raw_rmse
median_absolute_log_error
median_multiplicative_error
spearman
spearman_defined
```

Also report these Delta Response 2 diagnostics:

```text
signed_log_bias
calibration_intercept
calibration_slope
calibration_defined
pearson
pearson_defined
mean_true_log_slowdown
mean_predicted_log_slowdown
overprediction_count
underprediction_count
seed mean, standard deviation, best, and worst
```

Use the following exact unweighted estimands. For observation `i`, define
`y_i = log(true_slowdown_i)`, `p_i = predicted_log_slowdown_i`,
`e_i = p_i - y_i`, `s_i = true_slowdown_i`, and
`q_i = predicted_slowdown_i`.

| Metric | Definition |
| --- | --- |
| `log_mae` | `mean(abs(e_i))` |
| `log_mse` | `mean(e_i^2)` |
| `log_rmse` | `sqrt(log_mse)` |
| `raw_mae` | `mean(abs(q_i - s_i))` |
| `raw_mape` | `100 * mean(abs(q_i - s_i) / s_i)` |
| `raw_mse` | `mean((q_i - s_i)^2)` |
| `raw_rmse` | `sqrt(raw_mse)` |
| `median_absolute_log_error` | `median(abs(e_i))` |
| `median_multiplicative_error` | `median(exp(abs(e_i)))` |
| `signed_log_bias` | `mean(e_i)`; positive means overprediction |
| `calibration_intercept`, `calibration_slope` | Ordinary least squares fit `y_i = intercept + slope * p_i`; undefined for constant `p_i` |
| `spearman` | Spearman correlation of `p_i` and `y_i`, using average ranks |
| `pearson` | Pearson correlation of `p_i` and `y_i` |
| `mean_true_log_slowdown` | `mean(y_i)` |
| `mean_predicted_log_slowdown` | `mean(p_i)` |
| `overprediction_count` | Count where `e_i > 0` |
| `underprediction_count` | Count where `e_i < 0`; exact ties are in neither count |

Set `spearman_defined` and `pearson_defined` false when there are fewer than two
observations, the range of either input is at most `1e-12`, or the estimate is
non-finite. Set `calibration_defined` false when there are fewer than two
observations, the range of `p_i` is at most `1e-12`, or either coefficient is
non-finite. Store an undefined estimate as null, not as a numeric result, and
always retain its defined flag.

Compute and save metrics for these scopes separately:

```text
primary:          90 non-self directional rows
secondary:        all 110 directional rows
self diagnostic:  20 self directional rows
self averaged:     10 observations, one per self pair
per victim:        9 non-self rows for each victim_id
macro victim:      equal average of the ten per-victim estimates
per seed:          90 non-self rows for each seeded method and seed
ensemble:          90 non-self latent-space ensemble rows per method
```

For `self averaged`, average the two observed self-direction slowdowns in raw
space, retain the identical model prediction, and then recompute the true log.
For macro-victim rank or calibration metrics, average only defined victim
estimates and save `defined_victim_count`; report null if none are defined.
`metrics_by_seed.csv` contains one row per seeded method and seed.
`seed_stability.csv` contains one row per seeded method and metric, with the
five-seed mean, population standard deviation (`ddof=0`), best, and worst.
For error metrics best/worst mean minimum/maximum. For signed bias and the
calibration intercept they mean closest/farthest from zero; for the calibration
slope they mean closest/farthest from one; for correlations they mean
maximum/minimum.

Do not replace the 90-row primary result with a pooled self-inclusive metric.
If Spearman or a calibration slope is undefined, store an explicit defined flag
instead of presenting a numeric placeholder as an estimate.

## Pair-Cluster Bootstrap

Implement the same paired cluster bootstrap used by Delta Response 1:

1. Start from the 45 non-self `pair_cluster_id` values.
2. With NumPy RNG seed 923, draw 45 cluster indices with replacement.
3. Include both directions every time a cluster is drawn.
4. Reuse one complete draw matrix for every method and metric.
5. Generate exactly 1,000 replicates.
6. Report the point estimate and percentile `[0.025, 0.975]` interval.
7. For method comparisons, compute each replicate's paired
   `new_method - comparison_method` difference.

Negative error-metric differences favor the new method. State the sign
convention in every paired table.

Bootstrap ensemble rows, not seed rows. Produce intervals for the nine error
metrics, signed log bias, calibration intercept/slope, Spearman, Pearson, and
the two mean-log metrics. Counts and defined flags are point diagnostics only.
For a metric that is undefined in a replicate, exclude that replicate from its
quantiles and save `defined_replicate_count`; the interval is null if no
replicate is defined.

Do not use `paired_fold_bootstrap` from Delta Response 2 for this task; that
development utility clusters by victim, whereas this static experiment must
cluster by unordered pair.

## Required Outputs

Write all evaluation products under the separate output root:

```text
evaluation_plan.json
evaluation_preparation_seal.json
input_hashes.json
diagnostic_checkpoints/                exact roster specified above
evaluation_pair_manifest.csv
seed_predictions.csv
directional_predictions.csv
metrics_non_self.csv
metrics_non_self_per_victim.csv
metrics_non_self_macro_victim.csv
metrics_all.csv
metrics_self.csv
metrics_self_averaged.csv
metrics_by_seed.csv
seed_stability.csv
cluster_bootstrap_non_self.csv
paired_method_differences_non_self.csv
archived_delta1_comparison.csv
design_diagnostic_comparison.csv
run_report.json
STATIC_RANDOM_SPLIT_REPORT.md
output_hashes.json
completion_seal.json
```

The final completion seal must bind the preparation seal, pair hash, frozen
recipe hash, all five archived Delta Response 1 hashes, evaluation source
hashes, diagnostic checkpoint hashes, and exact output roster. Reject unknown
files and symlinks. A `status` invocation must verify all artifacts after
completion.

## Required Analysis

`STATIC_RANDOM_SPLIT_REPORT.md` must answer these questions without tuning the
model:

1. Does the primary frozen Delta Response 2 recipe improve non-self log MAE
   relative to the frozen Delta Response 1 OOD delta?
2. Does it beat the no-degradation constant and inherited absolute-response
   model on magnitude error?
3. Does it preserve useful ordering even if magnitude transfer fails?
4. What are its signed bias, calibration intercept/slope, mean prediction, and
   overprediction count?
5. Are results stable across the five final seeds and ten victims?
6. Does the all-anchor censored correction reduce floor-related or global
   overprediction relative to the fixed legacy correction on the same `32/16`
   potential checkpoints?
7. Does fixed query shrinkage improve transfer scale without destroying rank?
8. Does the absolute model again outperform delta correction, suggesting that
   synthetic-to-real residual calibration remains the failure mode?
9. Are apparent differences supported by paired 45-cluster intervals, or are
   they too uncertain for a directional conclusion?
10. Which original design hypotheses are supported, weakened, or rejected?

Discuss at least these prespecified hypotheses:

```text
H1: Positive-only residual anchors caused upward transfer bias; all-anchor
    censor inequalities should reduce it.

H2: The frozen multiview/one-standard-error selection should transfer with lower
    non-self log MAE than the prespecified profile/unweighted log-MAE winner,
    current_delta_ood_h32_e16__726110c4a776.

H3: Under the same fixed-step training regimen, the selected 8/4 network should
    have no worse App-App seed dispersion than the 32/16 network.

H4: If the selected legacy recipe still overpredicts App-App slowdown, then
    App-Inhibitor validation remains an inadequate proxy for transfer scale.

H5: If the constant or absolute model remains best, the central problem is
    conditional domain shift rather than insufficient delta-model capacity.
```

Use the following prespecified decision rules. All pairwise differences are
`first method - second method`, use the shared 45-cluster draws, and use the
90-row ensemble scope unless stated otherwise.

| Hypothesis | Prespecified estimand and interpretation |
| --- | --- |
| H1 | Compare `delta2_censored_zero_h32_e16` with `delta2_profile_only_ood_h32_e16`, which share potential checkpoints. Call the result supportive only if the legacy method has positive signed bias, the censored method has smaller absolute signed bias and fewer overpredictions, and the paired signed-bias interval is entirely below zero. An overlapping interval is weakened evidence; movement away from zero is contrary evidence. |
| H2 | Compare `delta2_selected` with `delta2_profile_only_ood_h32_e16`. A log-MAE difference and interval entirely below zero supports the stated selector contrast; overlap weakens it; an interval entirely above zero rejects it. This contrast does not isolate which selection component caused a difference. |
| H3 | On `metrics_by_seed.csv`, compare the population SD and worst value of the five non-self log-MAE estimates for `delta2_selected` and `delta2_profile_only_ood_h32_e16`. Both selected-model values no larger than the 32/16 values support the capacity-stability statement; a mixed result weakens it; both larger reject it. This is descriptive and cannot identify an effect of fixed-step training without a non-fixed-step control. |
| H4 | Positive `delta2_selected` signed bias with an interval entirely above zero and more over- than underpredictions supports the conditional observation; overlap weakens it; non-positive bias rejects it. It is evidence of scale mismatch, not proof of its cause. |
| H5 | Compare each of `delta2_constant_1`, both Delta Response 2 absolute models, and `delta1_absolute_response` as the first method with `delta2_selected` as the second. A constant or absolute method with lower log MAE and a paired interval entirely below zero is consistent with H5; overlap is weakened evidence. It does not causally identify conditional domain shift. |

Separate observations from explanations. Label causal statements as hypotheses
unless the experiment directly identifies them.

## Historical Delta Response 1 Checksums

Do not use these values as tuning targets. After new predictions and the
evaluation plan are sealed, use them to verify that your metric scopes and
archived comparison are correct.

Historical non-self log MAE on 90 directions:

| Method | Log MAE |
| --- | ---: |
| `absolute_response` | 0.097189 |
| `constant_1` | 0.103012 |
| `nearest_anchor` | 0.193877 |
| `delta_median` | 0.211887 |
| `delta_single_anchor` | 0.229221 |
| `generic_potential` | 0.237422 |
| `delta_uniform` | 0.239282 |
| `delta_ood_kernel` | 0.251164 |
| `delta_kernel` | 0.261504 |
| `victim_inhibitor_median` | 0.380145 |

Additional inherited checks:

```text
mean true non-self App-App log slowdown: 0.103012
Delta Response 1 OOD-delta mean prediction: 0.340791
Delta Response 1 OOD-delta overpredictions: 78 of 90
Delta Response 1 OOD-delta Spearman: 0.418745
Delta Response 1 OOD-delta log-MAE interval: [0.209078, 0.294224]
Delta OOD minus absolute log-MAE difference: +0.153975
paired interval for that difference: [0.106098, 0.203071]
```

If your reproduction of the archived metrics differs materially, stop and fix
pair membership, direction expansion, ensemble construction, or metric scope
before interpreting Delta Response 2.

## Acceptance Criteria

The task is complete only when:

- `README.md` and `schematic.md` were followed as prerequisite specifications.
- The original `compact_v1` status still verifies after evaluation.
- No top-level model source or frozen artifact changed.
- The static pair hash and all three training-input hashes match this document.
- The pair manifest has 55 rows and matches the archived endpoint membership.
- All five archived Delta Response 1 hashes match before archive access.
- The primary table has exactly 90 non-self directions per method.
- Every seeded method has exactly five predictions per direction.
- Seed and directional schemas, method namespaces, and the diagnostic checkpoint
  roster match this document exactly.
- Archived Delta Response 1 checksum metrics reproduce within numeric tolerance.
- Pair-cluster bootstrap uses 45 clusters, 1,000 draws, and seed 923.
- New primary, baseline, design-diagnostic, seed, victim, bias, calibration, and
  paired-comparison tables are present.
- The evaluation root has an exact verified completion seal.
- `STATIC_RANDOM_SPLIT_REPORT.md` clearly states that the result is
  retrospective and that a new untouched holdout is still required.

In the final response, report the selected Delta Response 2 non-self log MAE,
raw MAE, signed bias, Spearman, calibration slope, paired differences against
Delta Response 1 OOD delta, constant one, and absolute response, and the result
of each hypothesis. Include the evaluation-root path and verification status.
