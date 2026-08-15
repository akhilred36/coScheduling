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
a pristine holdout. Do not use it for development, tuning, calibration,
debugging, or new confirmatory claims. Use `--skip-holdout` for real
App-Inhibitor development and use synthetic pair data for end-to-end tests.

A newly collected untouched App-App dataset is required for any future claim
about transfer performance. The pipeline deliberately loads a supplied pair
CSV only after crossed validation, selection, and final model fitting.

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
4. Run crossed validation for every model candidate, model kind, and CV seed.
5. Select model configurations, model-specific epochs, and aggregations.
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

The environment must provide NumPy, pandas, SciPy, scikit-learn, and PyTorch.
Device selection defaults to CUDA when available and CPU otherwise.

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
  "inference_anchor_counts": [4, 16, 64, "all"],
  "inference_anchor_seed": 1701,
  "bootstrap_samples": 1000,
  "bootstrap_seed": 923,
  "device": "auto",
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
--eval-method {random_split,one_known,zero_shot}
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

`run_experiments.py` provides the preregistered staged robustness study. It is
separate from `pipeline.py`: the pipeline remains one scientific fit/evaluation,
while the runner generates immutable task specifications, executes isolated
pipeline subprocesses, resumes completed work, and consolidates outputs.

The runner implements:

- A six-candidate base/augmented feature and rank `{1, 2, 4}` architecture
  screen.
- A staged AdamW learning-rate/weight-decay grid followed by capacity checks.
- Inhibitor-block sensitivity at `(blocks, seed)` values `(3,1701)`,
  `(4,1701)`, `(6,1701)`, `(4,1702)`, and `(4,1703)`.
- One-factor synthetic sweeps over data size, true rank, noise, censoring,
  missing edges, response replicates, outliers, target OOD shift, and transfer
  mismatch, plus predefined combined-adversity conditions.
- Deterministic uncensored response-observation budgets `4`, `16`, `64`, and
  `all`. These calibration subsets are nested and affect inference only; model
  training and crossed validation remain unchanged. A requested budget larger
  than the available uncensored observations means "up to" that budget and is
  marked by `anchor_budget_feasible=false`; requested, actual, available, and
  distinct-inhibitor counts are all saved.
- Training-application learning curves for sizes `2` through `8`, using ten
  balanced nested application orderings and five synthetic dataset seeds in
  the standard profile.
- Paired `one_known` and `zero_shot` evaluation from the same fitted checkpoint
  for each training subset.

Architecture, optimizer, and capacity choices are carried forward separately
for low-rank, generic, and absolute models. Synthetic stress and training-size
tasks deliberately use the fixed preregistered default rather than an adaptive
winner from the real-data tuning stages; this isolates the controlled method
study and avoids making synthetic results conditional on repeated real-fold
selection.

The standard plan is intentionally large. Inspect `experiment_manifest.csv`
and run individual stages before launching the entire study.

### Plan A Study

```bash
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
ROOT=audit_outputs/experiments/paper_v1

$PY run_experiments.py plan \
  --root "$ROOT" \
  --profile standard \
  --suites real,stress,learning_curve

$PY run_experiments.py status --root "$ROOT"
```

Profiles are:

- `smoke`: one model seed, two epochs, one balanced application split, and a
  reduced synthetic scenario set.
- `standard`: five model seeds, 80 epochs, 1,000 bootstrap samples, five
  synthetic seeds, and ten balanced nested application splits.
- `extended`: ten model seeds, 160 epochs, 5,000 bootstrap samples, and an
  additional profile-diverse nested-split sensitivity analysis.

Planning generates synthetic CSVs and `subset_membership.csv`, but does not fit
models. Every generated dataset has a `scenario_summary.json` containing the
requested condition, realized censoring/missingness, row counts, and file
hashes.

### Run Stages

```bash
# Architecture, optimizer, capacity, and fold sensitivity. Dependencies are
# included automatically when a later stage is requested.
$PY run_experiments.py run \
  --root "$ROOT" \
  --stages real_fold_robustness \
  --resume

# Synthetic one-factor and combined stress tests.
$PY run_experiments.py run \
  --root "$ROOT" \
  --stages synthetic_stress \
  --resume

# Application-count learning curves for one-known and zero-shot transfer.
$PY run_experiments.py run \
  --root "$ROOT" \
  --stages training_size_learning_curve \
  --resume
```

Use `--max-tasks COUNT` for bounded batches and `--max-workers COUNT` only when
the allocated hardware can support concurrent PyTorch subprocesses. One worker
is the safe default for a single GPU. Each retry receives a new attempt
directory; prior logs and partial outputs are never overwritten.

Omitting `--stages` selects the entire plan. For the full `standard` plan shown
above, that means 508 tasks. Prefer the stage-specific commands above or use
`--max-tasks` when a bounded run is intended.

### Progress, Logs, And Interruption

The runner announces each task when it starts and prints the task attempt-log
directory. Pipeline subprocesses run with unbuffered output, with stdout and
stderr retained as `stdout.log` and `stderr.log` below that attempt directory.
Crossed validation reports completed model fits, total model fits, percentage,
and elapsed time. The runner also prints a heartbeat every 30 seconds while a
task is active, including its elapsed time and latest pipeline stdout line.
Change the runner heartbeat cadence with `--progress-interval SECONDS`; the
value must be positive.

A standard real-data architecture task performs 3,600 cross-validation model
fits, with an upper bound of 288,000 training epochs. It can therefore run for
hours even when execution is healthy. The runner's main thread waits for each
pipeline worker during this time; an interrupted traceback ending in
`waiter.acquire()` indicates that the runner was waiting for that worker, not
that this lock was the source of a deadlock.

Pressing Ctrl-C interrupts the active pipeline, prints a concise resume message,
and exits with status `130` instead of exposing the executor's internal wait
traceback. The interrupted attempt and its partial outputs remain immutable and
are not treated as successful. Re-running with `--resume` skips completed tasks
whose execution fingerprints still match and starts incomplete or failed tasks
in a new numbered attempt directory.

For example, resume only the architecture stage with a ten-second runner
heartbeat:

```bash
$PY run_experiments.py run \
  --root "$ROOT" \
  --stages real_architecture \
  --progress-interval 10 \
  --resume
```

Real-data tasks always invoke `pipeline.py --skip-holdout` and never pass a pair
path. Pair evaluation in this runner is synthetic. Do not substitute the
historical real `pair.csv`; a future untouched holdout requires a separately
authorized final evaluation after the experiment design is frozen.

### Consolidated Results

Successful runs are consolidated automatically. They can also be rebuilt
without rerunning models:

```bash
$PY run_experiments.py consolidate --root "$ROOT" --include-predictions
```

`consolidated/` contains plain CSV files suitable for pandas:

- `fold_metrics.csv`: crossed victim/block/seed/model records with full task
  provenance and resolved candidate fields.
- `candidates.csv`: every model-kind-specific candidate in every staged run.
- `selections.csv`: selected model configuration, epoch count, aggregation,
  kernel temperature, and OOD rule.
- `metrics_long.csv`: tidy overall, self-averaged, non-self, per-victim,
  macro-victim, seed, bootstrap, and paired-difference metrics.
- `predictions.csv`: complete synthetic directional predictions with scenario,
  training-subset, endpoint-familiarity, and anchor-budget columns.
- `pair_eligibility.csv`: canonical pair inclusion and known-endpoint counts.
- `data_conditions.csv`: requested and realized synthetic conditions.
- `learning_curve_replicates.csv`: split- and generator-level non-self learning
  curve estimates.
- `learning_curve_summary.csv`: split-averaged estimates with between-generator
  standard deviations and standard errors.
- `failures.csv`: failed task metadata; subprocess stdout/stderr remain under
  each task attempt.

Typical notebook loading requires only pandas:

```python
from pathlib import Path
import pandas as pd

root = Path("audit_outputs/experiments/paper_v1/consolidated")
metrics = pd.read_csv(root / "metrics_long.csv")
predictions = pd.read_csv(root / "predictions.csv")
learning_curve = metrics.query(
    "stage == 'training_size_learning_curve' "
    "and scope == 'overall_non_self' "
    "and metric == 'log_mae'"
)
```

For the training-size study, `split_id` is the repeated composition unit and
the size-`k` set is a prefix of the same ordering. Pair-cluster bootstrap files
are conditional diagnostics within one generated dataset and fitted lineage;
they are not population-level uncertainty across applications. Use
`learning_curve_summary.csv` for the primary across-generator uncertainty and
retain split-level variation from `learning_curve_replicates.csv`. Model seeds
are repeated fits rather than independent observations. Size-8 zero-shot
results have only two unknown endpoints and should be accompanied by pair
counts and a sensitivity analysis excluding that point.

## Outputs

Every run writes:

- `requested_run_config.json`: fully resolved requested configuration.
- `data_validation_summary.json`: row counts, censoring, and anchor counts.
- `rejected_job_inh.csv`: quarantined response rows and reasons.
- `duplicate_job_inh_keys.csv`: preserved response-replicate rows.
- `inhibitor_blocks.csv`: deterministic inhibitor block assignments.
- `crossed_validation_folds.csv`: model/fold/seed training selection records.
- `crossed_validation_predictions.csv`: all potential-model CV predictions.
- `crossed_validation_summary.csv`: mean CV errors by candidate and method.
- `validation_distance_reference.csv`: common-coordinate OOD distances.
- `selection.json`: selected model configurations, epochs, and aggregations.
- `checkpoints/*.pt`: final model state, model configuration, scaler, seed, and
  selected epoch count, including the fitting-application list.
- `run_report.json`: completion status, holdout state, calibration provenance,
  target semantics, endpoint-familiarity protocol, fitting/unknown application
  lists, evaluated pair counts, selected choices, and runtime metadata.

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
and holdout-free completion.

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
