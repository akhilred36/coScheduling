# Transitive Delta-Response Pipeline

This directory contains a boundary-aware pipeline for predicting an MPI
application's communication slowdown when it is co-scheduled with another
application. The model learns from application-inhibitor measurements and
isolated communication profiles. App-App measurements are optional and are
used only for final evaluation after all training and selection are complete.

The implementation provides:

- A low-rank transitive response-potential model.
- Generic-potential and absolute-response neural comparators.
- Censored training for slowdown measurements clipped at `1`.
- Leakage-safe crossed application/inhibitor-block validation.
- Uniform, median, physical-profile kernel, and OOD-gated anchor aggregation.
- Independent model-specific configuration and epoch selection.
- Deterministic grouped one-standard-error aggregation selection.
- Finite-range validation, uncertainty diagnostics, and paired cluster
  bootstrap comparisons.
- Three endpoint-familiarity evaluation tiers: `random_split`, `one_known`,
  and `zero_shot`.

See `schematic.md` for the extended research specification.

## Holdout Policy

The historical real App-App dataset was opened by earlier smoke runs and is not
a pristine holdout. The runner defaults to this standard pair file for
repeatable evaluation, but its outcomes remain excluded from development,
tuning, calibration, and model fitting. Results from it must not be described
as evaluation on a previously untouched holdout.

A newly collected untouched App-App dataset is required for any future claim
about transfer performance. The pipeline deliberately loads a supplied pair
CSV only after crossed validation or fixed-global selection and final model
fitting.

## Data Inputs

The default training files are under `../few_shot/data/`. Paths can be replaced
with command-line options.

### Application Profiles

`jobs.csv` contains one row per application:

```text
job_id,mpi_time,comm_frac,total_msgs,total_bytes
```

### Inhibitor Profiles

`inhibitors.csv` contains one row per synthetic inhibitor. It must include:

```text
inhib_id,mpi_time,comm_frac,total_msgs,total_bytes
```

Additional inhibitor-only control columns are allowed but are not model
features because corresponding values do not exist for real applications.

### App-Inhibitor Responses

`job_inh.csv` contains directional victim responses:

```text
job_id,inhib_id,slowdown
```

Rows with missing profiles, non-finite slowdown, or slowdown below `1` are
quarantined. Repeated `(job_id, inhib_id)` observations are retained and receive
an explicit `replicate_id`; they are not silently averaged.

### Optional App-App Evaluation

A synthetic or newly collected pair CSV has one row per unordered pair:

```text
jobA_id,jobB_id,slowdown_A,slowdown_B
```

Each row expands into two directional outcomes. Both directions retain the
same `pair_cluster_id` for paired inference and bootstrap resampling. Duplicate
unordered pairs, including reversed duplicates, are rejected. Slowdowns must
be finite and at least `1`.

## Three-Tier Evaluation

The tier names are compatible with the neighboring `few_shot` pipeline, but
the protocol is adapted to this model's App-Inhibitor-only training objective:

- `random_split`: every application and App-Inhibitor response may be used for
  model fitting. Evaluation contains all validated App-App interactions. The
  historical name is retained even though App-App outcomes are never split or
  used for fitting in this pipeline.
- `one_known`: only configured training applications influence model fitting
  and scaler statistics. Evaluation contains canonical pairs with exactly one
  training application and one unknown application.
- `zero_shot`: uses the same restricted fit as `one_known`, but evaluation
  contains only pairs where both applications are unknown during fitting.

For `one_known` and `zero_shot`, unknown applications' measured App-Inhibitor
responses remain available as inference anchors. This matches the intended
delta-response inference setting and the response-set inputs used by the
reference `few_shot` model. Unknown application profiles and responses are
excluded from parameter, epoch, aggregation, and model-scaler fitting.

The focused paper runner described below makes one explicit exception for its
cross-condition comparison: it selects one global architecture, optimizer,
epoch count, and aggregation recipe using App-Inhibitor data from all
applications, then freezes that recipe for every endpoint tier and training
size. Model weights and scalers are still refitted using only the applications
allowed by each condition. Consequently, its `one_known` and `zero_shot`
results are transductive with respect to hyperparameter selection, although
App-App outcomes remain completely sealed during tuning.

The restricted tiers accept a comma-separated fitting set through
`--training-apps`. When that option is omitted for the repository's standard
dataset, the pipeline uses `amg`, `beatnik`, `fiesta`, `laghos`, `lammps`,
`minife`, and `minivite`. Nonstandard datasets must provide the fitting set
explicitly. `random_split` always resolves every application as a training
application and rejects a partial fitting set.

Pair eligibility is decided on canonical pair rows before directional
expansion. The two directional outcomes therefore cannot cross tiers. The
pipeline writes `evaluation_pair_manifest.csv` with endpoint-known flags and
the inclusion decision for every pair row.

## Target Semantics

The available slowdown is a clipped observation:

```text
observed_slowdown = max(1, latent_slowdown)
observed_log_slowdown = max(0, latent_log_slowdown)
```

An observed slowdown above `1` identifies the latent value. An observed
slowdown equal to `1` states only that the latent log slowdown is at most zero.
The magnitude of a latent speedup was discarded during clipping and cannot be
recovered from these CSVs.

Models produce latent log-slowdown values. Every reported operational
prediction passes through the observation boundary and guarded exponentiation:

```text
predicted_log_slowdown = max(0, latent_prediction)
predicted_slowdown = exp(predicted_log_slowdown)
```

Non-finite latent values, negative observed log predictions, and values too
large for finite exponentiation fail explicitly. Consequently, every saved
operational prediction is finite and at least `1`.

## Profile Features

The default `base` representation uses measurements shared by applications and
inhibitors:

```text
x(E) = [
    log1p(mpi_time),
    comm_frac,
    log1p(total_msgs),
    log1p(total_bytes),
]
```

The optional `augmented` representation adds transformed estimated runtime,
mean message size, message rate, and byte rate. Each model scaler standardizes
the selected representation.

During crossed validation, a fold-local model scaler is fitted only to training
application profiles and retained inhibitor profiles. It excludes both the
held-out victim and the held-out inhibitor block. After selection, final model
scalers use every inhibitor profile and only the application profiles allowed
to influence the selected tier. Withheld application profiles are transformed
using those frozen statistics at inference and never affect scaler fitting.

Kernel and OOD distances do not use model embeddings or fold-local scaler
coordinates. They use a separate base-feature scaler fitted once to inhibitor
profiles, giving every validation and inference distance a common physical
coordinate system.

## Models

### Low-Rank Response Potential

The primary model learns a scalar latent response potential `R(A, X)` for
victim application `A` and aggressor profile `X`:

```text
R(A, X) = h(E(X)) + dot(u(E(A)), v(E(X)))
```

The default architecture is:

```text
shared encoder E:  input_dim -> 8 -> 4, ReLU activations
global head h:     4 -> 8 -> 1
victim head u:     4 -> rank
aggressor head v:  4 -> rank
default rank:      2
```

The predicted replacement effect between anchor inhibitor `J` and query
aggressor `K` is a potential difference:

```text
delta(A, J, K) = R(A, K) - R(A, J)
```

This construction guarantees exact latent-delta identity and antisymmetry, with
path consistency up to floating-point tolerance:

```text
delta(A, J, J) = 0
delta(A, J, K) = -delta(A, K, J)
delta(A, J, M) + delta(A, M, K) = delta(A, J, K)
```

These invariants apply before the nonlinear clipped-observation mapping.

### Generic Potential Comparator

The generic comparator retains the same potential-difference structure but
uses a higher-capacity response head over encoded victim, aggressor, and
elementwise interaction features:

```text
R_generic(A, X) = MLP([E(A), E(X), E(A) * E(X)])
```

It is trained with the same censored delta objective and receives independent
App-Inhibitor-only model, epoch, and aggregation selection.

### Absolute-Response Comparator

The absolute model uses the low-rank response architecture but predicts
`R(A, X)` directly rather than taking a potential difference. It is trained
against absolute observed log slowdown with its own censored objective and
independent epoch/configuration selection.

## Censored Training Objectives

Potential models sample victims uniformly, then sample two response rows for
distinct inhibitor IDs. This remains true when response replicates exist.

For anchor log response `y_J`, query log response `y_K`, and predicted latent
delta `d`, the loss is:

```text
y_J > 0 and y_K > 0: Huber(d - (y_K - y_J))
y_J > 0 and y_K = 0: one-sided Huber enforcing d <= -y_J
y_J = 0 and y_K > 0: one-sided Huber enforcing d >= y_K
y_J = 0 and y_K = 0: no identified pairwise delta constraint
```

For absolute regression, values above the floor use ordinary Huber residuals.
Floor observations use a one-sided Huber penalty only when the latent
prediction exceeds zero.

## Anchor-Based Inference

For an uncensored measured anchor `(A, J, y_J)`, a single-anchor latent query
prediction is:

```text
latent_y_hat(A, K | J) = y_J + R(A, K) - R(A, J)
```

Equivalently, each anchor supplies a residual correction:

```text
residual(A, J) = y_J - R(A, J)
latent_y_hat(A, K) = R(A, K) + aggregate_J(residual(A, J))
```

Only uncensored anchors are used as exact latent residual calibrators. Floor
anchors still inform model training through one-sided constraints.

The implemented aggregation rules are:

- `single`: residual from the nearest physical-profile anchor.
- `uniform`: arithmetic mean of anchor residuals.
- `median`: robust median of anchor residuals.
- `kernel`: softmax-weighted mean using physical-profile distances.
- `ood_kernel`: kernel correction blended toward uniform or median fallback as
  nearest-anchor distance leaves validated inhibitor support.

All five low-rank aggregation methods use the same low-rank checkpoint. The
aggregation comparison therefore changes only inference, not model fitting.

The evaluation table also includes constant slowdown `1`, per-victim inhibitor
median, nearest-anchor response without a learned delta, absolute response
regression, and the independently selected generic potential.

## Crossed Validation And Selection

Inhibitors are assigned to profile-space blocks by deterministic K-means. For
each held-out victim and held-out inhibitor block, the pipeline:

1. Fits the model scaler without the victim or inhibitor block.
2. Removes all response rows for the victim and block from model training.
3. Uses retained responses for the held-out victim as support anchors.
4. Uses responses in the held-out block as fixed validation queries.
5. Trains low-rank, generic, and absolute models with equal budgets and seeds.
6. Records model-specific best epochs using only App-Inhibitor validation.
7. Evaluates aggregation candidates in the common physical distance system.

Model candidates are selected independently for low-rank, generic, and
absolute models by mean crossed-validation loss. The final epoch count for each
model kind is the median of its selected fold-specific best epochs.

Aggregation uses a deterministic grouped one-standard-error rule. Query errors
are averaged within seed and then across seeds for each victim/block group. A
candidate is eligible when its mean error is no more than one grouped standard
error above the numerical best. Eligible methods are preferred in this
prespecified complexity order:

```text
uniform, median, kernel, OOD-kernel
```

Parameter ties use sorted deterministic order. Low-rank and generic potential
models select aggregation independently. No App-App outcome participates in
model, epoch, aggregation, kernel-temperature, or OOD selection.

## Diagnostics And Metrics

Linear aggregation rules report effective-anchor count and weighted residual
dispersion. Median and OOD mixtures using median fallback are nonlinear, so
aggregate effective-anchor count and weighted dispersion are explicitly
undefined (`NaN`). Kernel-component effective count/dispersion and fallback
residual MAD remain available.

Unweighted residual spread is labeled as uncalibrated and is not presented as
target-specific uncertainty or a prediction interval. This pipeline does not
produce calibrated prediction intervals.

Evaluation metrics include log-space MAE/RMSE, raw-space MAE/RMSE, median
absolute log error, median multiplicative error, Spearman rank correlation,
per-victim metrics, and macro-victim averages. Constant or degenerate groups
store `spearman = 0` with `spearman_defined = false`; the placeholder is not an
estimated correlation.

Pair bootstrap draws sample original pair rows as clusters, preserving both
directional outcomes. One cluster-draw matrix is reused for every method. The
pipeline reports marginal method intervals and paired method-minus-comparison
confidence intervals.

## Pipeline Flow

An invocation runs these stages in order:

1. Load and validate App-Inhibitor inputs without opening a pair CSV.
2. Quarantine invalid responses and report response replicates.
3. Build deterministic inhibitor blocks.
4. In `crossed_cv` mode, run crossed validation for every model candidate,
   model kind, and CV seed, then select configurations, epochs, and
   aggregations.
5. In `fixed_global` mode, validate the supplied frozen recipe, skip crossed
   model selection, and construct only the task-local physical OOD distance
   reference.
6. Fit final low-rank, generic, and absolute models on all valid rows from the
   applications allowed by the selected evaluation tier.
7. Stop and write a complete report when `--skip-holdout` is active.
8. Otherwise load the supplied pair CSV, evaluate fitted models, and write
   predictions, metrics, and paired bootstrap reports. No fitting follows pair
   loading.

## Requirements

Run all commands from this directory using the project Python virtual
environment. This machine does not require or use Conda:

```bash
cd notebooks/utils/delta_response_1
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python --version
```

The environment must provide NumPy, pandas, SciPy, scikit-learn, PyTorch, and
`tqdm`. Device selection defaults to CUDA when available and CPU otherwise.

## Running The Pipeline

### Recommended App-Inhibitor-Only Run

This is the safe mode for real training data. It performs validation, selection,
and final model fitting without opening any pair CSV:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --output-dir audit_outputs/app_inhib_only \
  --skip-holdout
```

The default run is computationally substantial: five CV seeds, five final
seeds, four inhibitor blocks, and up to 80 epochs per fold/model.

### Reduced Holdout-Free Development Run

Use this for a faster integration check on the real App-Inhibitor inputs:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --output-dir audit_outputs/app_inhib_dev \
  --skip-holdout \
  --cv-seeds 0 --final-seeds 0 \
  --inhibitor-blocks 2 \
  --max-epochs 2 --patience 1 \
  --batches-per-epoch 1
```

### Synthetic End-To-End Smoke Run

Generate deterministic synthetic profiles, responses, and pair outcomes:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python audit_outputs/make_synthetic_data.py
```

Run the complete pipeline using only those synthetic files:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --jobs-csv audit_outputs/synthetic_data/jobs.csv \
  --inhibitors-csv audit_outputs/synthetic_data/inhibitors.csv \
  --job-inh-csv audit_outputs/synthetic_data/job_inh.csv \
  --pair-csv audit_outputs/synthetic_data/pair.csv \
  --output-dir audit_outputs/synthetic_smoke \
  --cv-seeds 0 --final-seeds 0 \
  --inhibitor-blocks 2 \
  --max-epochs 2 --patience 1 \
  --batches-per-epoch 1 \
  --bootstrap-samples 20
```

Run the restricted tiers by specifying the fitting applications:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --eval-method one_known \
  --training-apps amg,beatnik,fiesta,laghos,lammps,minife,minivite \
  --pair-csv /path/to/new_untouched_pair.csv \
  --output-dir audit_outputs/one_known

/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --eval-method zero_shot \
  --training-apps amg,beatnik,fiesta,laghos,lammps,minife,minivite \
  --pair-csv /path/to/new_untouched_pair.csv \
  --output-dir audit_outputs/zero_shot
```

Never use the historical real pair data as a smoke input.

### Future Evaluation With A New Holdout

After collecting a genuinely untouched App-App dataset and fixing every choice
without using its outcomes, run:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --pair-csv /path/to/new_untouched_pair.csv \
  --output-dir audit_outputs/new_holdout_evaluation
```

Do not use this mode with the historically contaminated pair dataset for a new
confirmatory claim.

## Configuration

Defaults are defined by `RunConfig` and `ModelConfig` in `pipeline.py`. A JSON
configuration can override them:

```json
{
  "inhibitor_blocks": 4,
  "evaluation_method": "random_split",
  "training_apps": null,
  "clustering_algorithm": "kmeans",
  "clustering_seed": 1701,
  "cv_seeds": [0, 1, 2, 3, 4],
  "final_seeds": [0, 1, 2, 3, 4],
  "max_epochs": 80,
  "patience": 10,
  "batches_per_epoch": 4,
  "batch_size": 128,
  "temperatures": [0.1, 0.3, 1.0, 3.0, 10.0],
  "ood_quantiles": [0.9, 0.95, 0.99],
  "ood_fallbacks": ["uniform", "median"],
  "inference_anchor_counts": ["all"],
  "inference_anchor_seed": 1701,
  "bootstrap_samples": 1000,
  "bootstrap_seed": 923,
  "device": "auto",
  "selection_mode": "crossed_cv",
  "model_candidates": [
    {
      "feature_set": "base",
      "hidden_dim": 8,
      "embedding_dim": 4,
      "rank": 2,
      "learning_rate": 0.001,
      "weight_decay": 0.0001
    }
  ]
}
```

Run with the configuration:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python pipeline.py \
  --config audit_outputs/config.json \
  --output-dir audit_outputs/configured_run \
  --skip-holdout
```

The following command-line options override corresponding JSON values:

```text
--cv-seeds SEED [SEED ...]
--final-seeds SEED [SEED ...]
--inhibitor-blocks COUNT
--max-epochs COUNT
--patience COUNT
--batches-per-epoch COUNT
--bootstrap-samples COUNT
--eval-method {both,random_split,one_known,zero_shot}
--training-apps APP_ID,APP_ID,...
```

Input and execution options are:

```text
--jobs-csv PATH
--inhibitors-csv PATH
--job-inh-csv PATH
--pair-csv PATH
--output-dir PATH
--config PATH
--skip-holdout
```

Requested and selected settings are saved in every run, so reduced budgets or
other deviations remain auditable.

## Paper Experiment Runner

`run_experiments.py` implements the focused paper study around three questions:
accuracy by endpoint-familiarity tier, accuracy by training-application count,
and selection of one global low-rank recipe. It deliberately omits synthetic
stress sweeps, fold-sensitivity sweeps, and anchor-budget sweeps.

The global low-rank search is sequential:

1. Feature set `{base, augmented}` and rank `{1, 2, 4}`.
2. AdamW learning rate `{3e-4, 1e-3, 3e-3}` and weight decay
   `{0, 1e-4, 1e-3}` around the selected architecture.
3. `(hidden_dim, embedding_dim)` in `{(4,2), (8,4), (16,8)}` around the
   selected architecture and optimizer.

Generic-potential and absolute-response models use the fixed default
architecture as baselines. Their epoch and generic-aggregation settings are
fixed from the same final App-Inhibitor tuning lineage. The resulting
architecture, optimizer, epoch counts, and aggregation rules are then reused
without crossed model selection in every App-App evaluation task. Each task
still refits weights and scalers from its allowed training applications.
For the primary low-rank recipe, the runner freezes the aggregation candidate
with minimum grouped CV log MAE rather than the general pipeline's simpler
one-standard-error choice.

The standard evaluation matrix contains:

- One `random_split` fit using all ten applications and all pair rows.
- Training sizes `2` through `8` for restricted fits.
- Ten balanced cyclic rotations of one deterministic application ordering.
- One fit per `(split, size)` with paired `one_known` and `zero_shot`
  evaluation from the same checkpoints.
- All available uncensored anchors at inference; there is no anchor-budget
  sweep.

The standard plan has 74 tasks: three tuning tasks, one `random_split` task,
and 70 restricted tasks. The `smoke` profile uses one seed, two epochs, one
split, and training size two for integration testing.

The standard profile is preregistered for exactly ten applications. Planning
rejects a different application count instead of silently changing the task
matrix.

### Plan And Run

Planning uses the repository's standard `jobs.csv`, `inhibitors.csv`,
`job_inh.csv`, and `pair.csv` paths by default:

```bash
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
ROOT=audit_outputs/experiments/focused_paper_v1

$PY run_experiments.py plan \
  --root "$ROOT" \
  --profile standard

$PY run_experiments.py tune --root "$ROOT" --resume
$PY run_experiments.py evaluate --root "$ROOT" --resume
$PY run_experiments.py status --root "$ROOT"
```

All four paths can be overridden during planning with `--jobs-csv`,
`--inhibitors-csv`, `--job-inh-csv`, and `--pair-csv`. Planning hashes every
input and writes immutable task specifications plus `subset_membership.csv`.
Execution fails if any input changes after planning; a changed dataset requires
a new experiment root.

The runner exposes these commands:

```text
plan        Create plan.json, experiment_spec.json, input hashes, and subsets.
tune        Run tune_architecture, tune_optimizer, and tune_capacity.
evaluate    Run evaluate_random_split and evaluate_training_size.
status      Refresh and summarize experiment_manifest.csv.
consolidate Rebuild paper-facing CSV and JSON outputs without fitting models.
```

`tune` and `evaluate` accept `--max-workers`, `--max-tasks`,
`--progress-interval`, and `--resume`. `evaluate` also accepts
`--[no-]include-predictions`; `tune` always consolidates without predictions.
With `--resume`, bounded batches advance past successful target tasks while
still including required tuning dependencies.

`evaluate` automatically includes incomplete tuning dependencies. Use
`--max-tasks COUNT` for bounded batches and `--max-workers COUNT` only when the
allocated hardware can support concurrent PyTorch subprocesses. One worker is
the safe default for a single GPU.

Each retry receives a new attempt directory. Standard output and error are
retained under that attempt, progress heartbeats report the latest pipeline
line, and `--resume` skips successful tasks only when input, code, environment,
configuration, and dependency-selection fingerprints still match.

### Accuracy Comparisons

The primary paper metric is non-self directional log MAE. Consolidation also
retains log RMSE, raw MAE/RMSE, median multiplicative error, Spearman, per-victim
metrics, paired cluster bootstrap results, and complete predictions.

Two eval-method views are produced:

- `protocol_native`: each tier's metric on its naturally eligible pair rows.
- `matched_random_split`: predictions from the one full-data fit restricted to
  the exact non-self pair manifest of a `one_known` or `zero_shot` condition.

The matched view separates training-restriction effects from changes in pair
difficulty. One-known and zero-shot remain different endpoint-familiarity sets.
Balanced rotations reduce application-composition confounding across the full
learning curve. Their between-split SD is descriptive because the cyclic subsets
overlap; it is not converted to an independence-based standard error. Model
seeds are repeated fits, not independent samples. The size-8 zero-shot point has
only two unknown applications and one non-self pair, so it is marked
`low_support_zero_shot=true`.

### Consolidated Results

Successful commands consolidate automatically. Results can also be rebuilt:

```bash
$PY run_experiments.py consolidate --root "$ROOT" --include-predictions
```

`consolidated/` contains:

- `global_selection.json`: the frozen recipe, provenance, and transductive
  selection disclosure.
- `architecture_search.csv`: every low-rank tuning candidate and mean crossed
  App-Inhibitor validation log MAE.
- `accuracy_by_eval_method.csv`: protocol-native and matched comparisons.
- `accuracy_by_training_size.csv`: split-level native learning-curve metrics.
- `learning_curve_summary.csv`: means and descriptive between-split SD for
  sizes `2..8`.
- `metrics_long.csv`: all metric scopes and bootstrap diagnostics.
- `predictions.csv`: complete directional predictions when enabled.
- `pair_eligibility.csv`: canonical pair inclusion and endpoint familiarity.
- `fold_metrics.csv`, `candidates.csv`, and `selections.csv`: tuning audit data.
- `failures.csv`: failed task metadata; full logs remain in attempt directories.

For example:

```python
from pathlib import Path
import pandas as pd

root = Path("audit_outputs/experiments/focused_paper_v1/consolidated")
accuracy = pd.read_csv(root / "accuracy_by_eval_method.csv")
learning = pd.read_csv(root / "learning_curve_summary.csv")
primary = accuracy.query(
    "is_global_primary_low_rank and metric == 'log_mae'"
)
```

## Outputs

Every run writes:

- `requested_run_config.json`: fully resolved requested configuration.
- `data_validation_summary.json`: row counts, censoring, and anchor counts.
- `rejected_job_inh.csv`: quarantined response rows and reasons.
- `duplicate_job_inh_keys.csv`: preserved response-replicate rows.
- `inhibitor_blocks.csv`: deterministic inhibitor block assignments.
- `validation_distance_reference.csv`: common-coordinate OOD distances.
- `selection.json`: selected model configurations, epochs, and aggregations.
- `checkpoints/*.pt`: final model state, model configuration, scaler, seed, and
  selected epoch count, including the fitting-application list.
- `run_report.json`: completion status, holdout state, calibration provenance,
  target semantics, endpoint-familiarity protocol, fitting/unknown application
  lists, evaluated pair counts, selected choices, and runtime metadata.

Runs using crossed selection additionally write:

- `crossed_validation_folds.csv`: model/fold/seed training selection records.
- `crossed_validation_predictions.csv`: all potential-model CV predictions.
- `crossed_validation_summary.csv`: mean CV errors by candidate and method.

Fixed-global evaluation runs deliberately omit these three files and record a
zero crossed-fold count in `selection.json`.

Runs that evaluate a synthetic or new App-App holdout additionally write:

- `directional_predictions.csv`: ensemble predictions and diagnostics.
- `evaluation_pair_manifest.csv`: canonical pair eligibility, known-endpoint
  flags, and inclusion decisions for the selected tier.
- `seed_predictions.csv`: predictions from each model seed.
- `metrics_repeated_self.csv`: metrics retaining both self-pair measurements.
- `metrics_self_averaged.csv`: metrics after self-pair outcome averaging.
- `metrics_per_victim.csv`: method metrics for each victim.
- `metrics_macro_victim.csv`: equal-weight victim macro averages.
- `metrics_by_seed.csv`: initialization-seed variation.
- `cluster_bootstrap_confidence_intervals.csv`: marginal intervals from the
  shared pair-cluster draws.
- `paired_method_differences.csv`: paired method-minus-comparison intervals.
- `metrics_non_self.csv`: overall metrics excluding self-pairs.
- `metrics_non_self_per_victim.csv`: non-self metrics by victim.
- `metrics_non_self_macro_victim.csv`: equal-weight non-self victim averages.
- `cluster_bootstrap_non_self.csv`: conditional pair-cluster intervals after
  excluding self-pairs.
- `paired_method_differences_non_self.csv`: non-self paired method intervals.

When `--eval-method both` is used, fitting occurs once and tier-specific files
are written below `evaluations/one_known/` and `evaluations/zero_shot/`.

Undefined nonlinear diagnostics remain `NaN` by design. Prediction and metric
fields are finite and validated against the observed target support.

## Experiment Analysis And Plots

After consolidation, generate paper-oriented plots and derived summary tables:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python analyze_experiments.py \
  --root experiments
```

By default, results are written under `experiments/analysis/`. The script
creates PNG and PDF versions of model-selection, endpoint-familiarity,
learning-curve, matched-fit, calibration, per-application, pairwise-error,
diagnostic, paired-bootstrap, and censoring-stratum plots. It also writes the
derived values behind the plots under `analysis/tables/`, a `figure_index.csv`,
and an `analysis_summary.json` containing the experiment completion status and
interpretation caveats.

Use `--comparison-size COUNT` to choose the restricted-fit size shown in the
endpoint comparison. Without it, the script selects the largest size not
flagged as low-support zero-shot. Use `--formats png pdf svg` to change output
formats. Prediction-level plots require consolidation with
`--include-predictions`; aggregate plots are still generated when
`predictions.csv` is absent.

## Tests

Run the implementation and regression suite:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python -m unittest discover -s tests -v
```

Run the independent audit checks:

```bash
/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python audit_outputs/audit_tests.py
```

The tests cover crossed-fold exclusions, fixed validation queries, sampler
behavior with replicates, latent potential invariants, common-coordinate OOD
distances, clipped prediction support, finite-range guards, weighted and
nonlinear diagnostics, shared low-rank checkpoints, paired bootstrap draws,
pair uniqueness, missing profiles, endpoint-tier disjointness, restricted
training/scaler inputs, full inference-anchor availability, pair-load ordering,
holdout-free completion, fixed-global CV bypass, immutable input hashes,
balanced plan topology, bounded resume behavior, and stale consolidation
cleanup.

## Limitations

The model assumes that interference behavior learned from synthetic inhibitor
profiles transfers to real application aggressors with comparable isolated
profiles. Strong held-out-inhibitor validation is necessary but does not prove
that transfer assumption.

The available features do not encode placement, network topology, collective
operation type, synchronization behavior, or temporal burst alignment. OOD and
spread diagnostics expose some extrapolation risk but are not calibrated
prediction intervals.

Historical App-App contamination cannot be repaired in code, and clipped
measurements cannot identify latent speedup magnitude. No run on the existing
real pair dataset can establish a new untouched-holdout result.
