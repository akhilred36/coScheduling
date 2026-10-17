# Bidirectional Filtered-Inhibitor Pipeline

This directory implements `plan.md`: a self-contained successor to
`delta_response_filteredInhibs/` that

1. rebuilds isolated profiles and **bidirectional** App-Inhibitor measurements
   from one consistent processed-data snapshot,
2. adds a **joint absolute + delta** response model,
3. adds **floor-aware calibration**,
4. adds **reverse-direction bandwidth supervision** (the inhibitor's rate
   change under the application trains the application's aggressor
   representation), and
5. adds measured **application response fingerprints**,

with leakage-safe crossed validation, checkpointed final fits, and a
controlled ablation comparison against the original models. It runs
independently and imports nothing from sibling directories. The sibling
pipeline is left unchanged for comparison.

`schematic.md` shows the dataflow (Part A) and keeps the inherited
delta-response specification (Part B).

## Quick start

From the repository root:

```bash
PY=/nfs/student/a/aalasandagutti/hpcResearch/python_virtual_envs/pytorch-gpu/bin/python
DIR=notebooks/utils/delta_response_filteredInhibs_biDirectional

# 1. Prepare training data (reads isolated + App-Inhibitor aggregates only).
$PY $DIR/prepare_bidirectional_data.py training
# 2. Prepare App-App evaluation labels (separate step; used only for final scoring).
$PY $DIR/prepare_bidirectional_data.py evaluation

# 3. Reduced holdout-free development run (~20 s).
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY $DIR/pipeline.py --skip-holdout \
  --cv-seeds 0 --final-seeds 0 --inhibitor-blocks 2 --max-epochs 2 --patience 1 \
  --batches-per-epoch 1 --output-dir $DIR/outputs/development

# 4. Full budget, all variants incl. ablations (~30 min on one CPU core), then plots.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY $DIR/pipeline.py --config $DIR/comparison_config.json \
  --variants legacy_delta legacy_generic legacy_absolute joint_profile bidirectional_profile \
             bidirectional_context bidirectional_profile_wall bidirectional_context_forward \
  --output-dir $DIR/outputs/full_comparison
$PY $DIR/plot_prediction_errors.py --run-dir $DIR/outputs/full_comparison

# Tests (53; synthetic data are generated in temporary directories).
$PY -m unittest discover -s $DIR/tests -v
```

Defaults: the prepared files under `data/training/` and `data/evaluation/`
(committed, regenerable with step 1–2), eight nodes, five CV seeds, five final
seeds, four inhibitor blocks, 80 epochs with patience 10, four batches of 128
per epoch, 1,000 paired-cluster bootstrap draws. The default device is `cpu`:
profiling showed host/device transfers dominating these tiny models on GPU.

## Stage 1: consistent bidirectional data

`prepare_bidirectional_data.py training` rebuilds everything from
`processed_data/data_isolated_agg.csv`, `data_inhib_isolated_agg.csv`, and
`data_inhib_coscheduled_agg.csv`. It never opens App-App data (a test enforces
this). `evaluation` reads `data_coscheduled_agg.csv` plus the prepared
`jobs.csv`.

**Provenance finding.** The rebuilt snapshot disagrees with the legacy
`few_shot/data` tables used by the sibling pipeline. All 10 application
profiles match, but 1,047 of 1,064 compared inhibitor-profile fields and
2,429 of 2,623 App-Inhibitor slowdowns differ (e.g. inhibitor
`8_20_5000_0.01`: MPI time 3.649 legacy vs 8.148 rebuilt). See
`data/training/source_consistency_report.csv`. Historical sibling numbers
therefore need not reproduce here.

Rebuilt snapshot: 10 applications, 287 inhibitors, 2,870 App-Inhibitor
observations (443 at the slowdown floor, none rejected), 55 eight-node App-App
pairs (10 self-pairs). Every row is an aggregate configuration
(`observation_unit = aggregate_configuration`,
`rate_estimator = ratio_of_aggregate_means`, `replicate_id = 0`), not a
run-level replicate.

### Inhibitor IDs

`{Num Nodes}_{Inhib Message Size}_{Inhib Wait Time (us)}_{sparsity}`; integers
without `.0`, sparsity decimal-normalized (`1.0 → 1`, `0.002237136`). The same
function formats isolated and co-scheduled rows; duplicate isolated
configurations raise and duplicate co-scheduled configurations are rejected
(never Cartesian-joined). `inhibitor_id_map.csv` maps IDs to configurations.

### Targets

| Column | Definition |
|---|---|
| `slowdown` | `max(1, T_MPI_co / T_MPI_iso)` of the application endpoint |
| `log_slowdown`, `is_censored` | `log(slowdown)`; `raw_mpi_time_ratio <= 1` |
| `raw_mpi_time_ratio` | unclipped ratio, diagnostic only |
| `{app,inhib}_bw_wall_log_degradation` | `log(bytes/runtime)_iso − log(bytes/runtime)_co` |
| `{app,inhib}_bw_mpi_log_degradation` | same with MPI time (MPI-time-normalized throughput) |
| `{app,inhib}_msg_wall_log_degradation` | same with messages per runtime |
| `*_valid` | one mask per auxiliary target |

Positive = degradation, negative = increased achieved rate; auxiliary targets
are never clipped and are computed as log differences. A zero or invalid rate
masks that task only; a valid slowdown row is kept. In App-App data the same
columns are prefixed `A_`/`B_` (App-B byte counts are zero on self-pairs, so
those bandwidth labels are masked).

Note: for applications, byte and message volume do not change under
co-scheduling, so `app_bw_mpi` equals the raw log MPI-time ratio and
`app_bw_wall` equals `app_msg_wall`. The wall-clock-only ablation exists to
measure what the MPI-time proxy adds beyond this redundancy.

Preparation outputs: `jobs.csv`, `inhibitors.csv`, `job_inh.csv`,
`inhibitor_id_map.csv`, `preparation_report.json` (hashes, filters, target
definitions, units, counts), `source_consistency_report.csv`,
`rejected_observations.csv`, `auxiliary_target_availability.csv`,
`observation_audit.csv` (all isolated and co-scheduled quantities).

## Inhibitor panel

Unchanged application-targeted facility-location selection (≤10 inhibitors,
four standardized communication features, tail weight 0.5). Eligibility uses
valid forward slowdown responses only; auxiliary labels never affect
eligibility or the objective, and all variants use the same panel. Restricted
tiers design the panel from their training applications only. The rebuilt
snapshot selects **nine** inhibitors (different IDs from the legacy panel).
`--frozen-panel-csv` supplies a fixed panel; `selection_report.json` records
`panel_source` (`recomputed`/`supplied`) and `panel_design_scope`.

## Variants and inference methods

| Variant | Response objective | Aux tasks | Fingerprints | Inference methods |
|---|---|---|---|---|
| `legacy_delta` | delta only (original) | – | – | `legacy_delta_{uniform,median,kernel,ood_kernel}` |
| `legacy_generic` | delta only (original) | – | – | `legacy_generic_{…}` |
| `legacy_absolute` | absolute only (original) | – | – | `legacy_absolute` |
| `joint_profile` | absolute + delta | – | – | `_direct`, `_calibrated` |
| `bidirectional_profile` | absolute + delta | both endpoints, 3 tasks | – | `_direct`, `_calibrated` |
| `bidirectional_context` | absolute + delta | both endpoints, 3 tasks | yes | `_direct`, `_calibrated` |
| `bidirectional_profile_wall` (ablation) | absolute + delta | `bw_wall`, `msg_wall` | – | `_direct`, `_calibrated` |
| `bidirectional_context_forward` (ablation) | absolute + delta | 3 tasks | forward-only tokens | `_direct`, `_calibrated` |

`constant_1` is always reported. Legacy variants keep their original
architectures and loss reductions. Default `--variants` is the first six.

Joint model: shared encoder `input → 8 → 4` (ReLU),
`R(V,X) = b(e_V) + h(e_X) + u(e_V)ᵀv(e_X)` (rank 2). Deltas are only
differences of responses, so identity, antisymmetry, and path consistency hold
exactly for fixed contexts. Loss
`L = 1.0·L_abs + 1.0·L_Δ + 0.1·L_aux` (`L_aux = 0` for `joint_profile`):
censored Huber on absolute targets, delta Huber averaged over *identified*
constraints only (both-floor pairs contribute nothing), and auxiliary Huber on
residuals divided by `σ_t = max(std_train(d_t), 0.1)`, averaged within each
task and then across available tasks. The bandwidth head consumes
`[e_V, e_X, u(e_V), v(e_X), u(e_V)·v(e_X)]` and is evaluated in both orders, so
reverse-direction labels train the same role projections the slowdown model
uses. No inhibitor slowdown labels are invented. Sampling: applications
uniformly, then distinct inhibitor IDs, then a replicate.

### Fingerprints

A token per (application, inhibitor configuration): the inhibitor's scaled
isolated profile, the application's mean log slowdown and floor fraction, and
training-scaled mean rate changes for the application endpoint (incoming
susceptibility) and the inhibitor endpoint (outgoing aggressiveness, dropped in
the forward-only ablation), each with an availability flag. A masked-mean set
encoder (`token → 8 → 4`) plus `M(z, 1_support, log(1+n))` is added to the
application embedding; empty support adds exactly zero. Inhibitors always have
an empty context. Training episodes draw support `S` and two query inhibitors
from the rest (`S ∩ Q = ∅`, all replicates of a query ID excluded), with
`P(|S| = 0) = 0.25` and `|S| ≤ min(n − 2, 7)`. Raw token tensors are cached
under keys naming the fold, application, support IDs, support observation IDs,
normalization state, and model identity.

`bidirectional_context_direct` is an **absolute-response predictor with
measured support features**, not an isolated-profile-only predictor.

### Floor-aware calibration (joint variants only)

`ĉ = argmin_c Σ_J w_J ℓ_J(c) + (λ/2)c²`, with Huber `ℓ` on `r_J + c − y_J` for
exact anchors and on `max(0, r_J + c)` for floor anchors; prediction
`max(0, R + ĉ)`. Each inhibitor configuration gets equal total weight split
across replicates. The root of the monotone derivative is found by bisection
with bracket expansion, a fixed tolerance, and convergence diagnostics. Empty
support falls back to `c = 0`. All-floor support is valid, so joint models do
not skip folds for it. λ is selected from {0.1, 1, 10} by App-Inhibitor CV
(`--calibration-lambda` with one value fixes λ). Legacy delta models keep
their anchor aggregation, since their additive level is not identified.

Context models use **cross-fitted** calibration: an application's panel is
split deterministically into up to three groups; view *k* builds the
fingerprint from the other groups and calibrates on group *k*, with anchors
and target sharing that fixed context. View latents are averaged before the
floor is applied. With a single inhibitor, the fingerprint is empty and that
inhibitor calibrates. For App-App inference the aggressor's fingerprint uses
its full panel and stays fixed across the victim's views. View dispersion is
not a prediction interval.

## Validation, selection, and endpoint tiers

Crossed validation is inherited: hold out application `A*` and block `B*`;
training excludes every label of `A*` and of `B*` runs, including reverse
bandwidth labels. Profile scaler, auxiliary scales, token scales, and weights
are fold-local. `A*`'s support is its rows outside `B*`; queries are its rows
in `B*`. Early stopping uses the calibrated prediction (λ = 1). Selection uses
grouped slowdown log MAE (query → victim/block/seed → seeds → groups); final
epochs are the median best epoch per variant. Auxiliary errors are
diagnostic only. Legacy potentials skip folds without uncensored support and
record this in `skipped_validation_folds.csv` and
`validation_group_eligibility.csv`. `matched_fold_comparisons.csv` compares
methods on the groups that every method evaluates, so skipped groups cannot
masquerade as accuracy differences.

`--eval-method` keeps `random_split`, `one_known`, `zero_shot`, and `both`.
Unknown applications never fit parameters or scalers. Their forward and
reverse App-Inhibitor panel measurements may serve as inference-time anchors
and fingerprints, recorded per row as
`victim_support_source = unknown_application_inference_support`. Global recipe
selection over all applications (runner) remains transductive, as in the
sibling runner.

## App-App boundary and checkpoints

App-App outcomes are parsed only after selection, final fitting, and writing
every checkpoint. Evaluation then **reloads models from checkpoints** and only
performs inference. A checkpoint stores the schema version, variant spec,
seed, epochs, training applications, panel IDs, model config, profile scaler,
auxiliary and token scales, context and calibration recipes (selected λ),
legacy aggregation, data hashes, and weights. Incompatible schemas are
rejected.

## Outputs (`outputs/<run>/`)

Validation: `crossed_validation_{predictions,folds}.csv`,
`selected_validation_predictions.csv`, `training_history.csv` (component
losses and valid-task counts per epoch), `auxiliary_validation_{predictions,metrics}.csv`
(by endpoint, direction, task, application, block),
`context_support_manifest.csv`, `calibration_diagnostics.csv`,
`metrics_by_censoring.csv`, `matched_fold_comparisons.csv`,
`ablation_summary.csv`, `selection.json`, `variant_registry.json`.

App-App: `directional_predictions.csv` (with `variant`, `inference_method`,
`support_information_regime`, context/calibration counts, offset, convergence,
view count, `predicted_{bw_wall,bw_mpi,msg_wall}_log_degradation`,
`swap_equivariance_residual`), `seed_predictions.csv`,
`auxiliary_{predictions,metrics}.csv`, `pair_predictions.csv` /
`pair_metrics.csv` (severity, imbalance, worst-endpoint error and ranking,
false-benign rate at 1.05, swap residual; non-self and self reported
separately), `metrics_by_censoring.csv` (signed bias, floor/above-floor),
`method_coverage.csv`, legacy metric files, and paired cluster-bootstrap
intervals over unordered pair clusters with one shared draw matrix.
`ablation_summary.csv` lists every contrast in both scopes.

**Method coverage.** Legacy delta/generic aggregation needs an uncensored
anchor, so it cannot predict for a victim whose panel responses are all at the
floor. Headline metrics, bootstrap, and ablations use the pair clusters that
*every* method evaluates. `metrics_non_self_all_available.csv` reports each
method on its own evaluable rows.

## Experiment runner

```bash
ROOT=$HOME/hpcResearch/experiment_outputs/bidirectional_v1
$PY $DIR/run_experiments.py plan --root $ROOT --profile standard   # 72 tasks
$PY $DIR/run_experiments.py tune --root $ROOT --resume
$PY $DIR/run_experiments.py evaluate --root $ROOT --max-workers 16 --resume
$PY $DIR/analyze_experiments.py --root $ROOT
```

`tune_global` runs crossed CV for all eight variants. Evaluation tasks
(`random_split` plus 10 balanced splits × 7 training sizes, tiers `both`)
freeze model-specific recipes: epochs, model configs, legacy aggregations,
and calibration λ. Task identity hashes include variants and context and
calibration recipes. All prepared inputs and both preparation reports are
hashed, and a changed input aborts execution. `consolidated/` holds
`metrics_long.csv`, `accuracy_{direct,calibrated,legacy_anchor_aggregation,baseline}.csv`,
`learning_curve_summary.csv`, `ablation_summary.csv`, pair, censoring,
auxiliary, coverage, and calibration tables with panel-source provenance. The
`smoke` profile exists for quick checks; `plan --inhibitor-blocks` changes the
block count.

## Results on the repository data (`outputs/full_comparison/`)

Full budget, all eight variants, nine-inhibitor panel (90 fitting rows),
1,600 crossed-CV fits (40 victim/block groups × 5 seeds × 8 variants; no
legacy fold was skipped because no application's support is all-floor after
block removal), ~25 minutes on one CPU core. **The App-App data are the
historical, already-inspected dataset: an exploratory comparison, not an
untouched holdout.**

**Selected recipes.** Final epochs: legacy delta 13, legacy generic 14,
legacy absolute 2, joint/bidirectional 1–2. Calibration λ = 1 for all joint
variants except the forward-only context ablation (λ = 10). For every model
with an identified level, the fold-mean validation error is lowest at epoch 1
and rises with training. Training loss and in-sample error do fall: in-sample
log MAE goes 0.178 → ~0.11 by epoch 80. So these models overfit the 90-row
panel as soon as they move away from the floor prediction. The selected joint
models are therefore close to "floor prediction plus a per-victim calibrated
offset", which limits how much the ablations below can say about learned
representations.

App-Inhibitor crossed CV (grouped slowdown log MAE, 40 matched groups;
difference ± grouped SE):

| Contrast | Method | Comparison | Difference |
|---|---|---|---:|
| joint abs+delta vs legacy absolute | 0.1521 | 0.1562 | −0.0042 ± 0.0058 |
| floor-aware calibration (joint) | 0.1481 | 0.1521 | −0.0039 ± 0.0061 |
| reverse supervision, calibrated | 0.1479 | 0.1481 | −0.0002 ± 0.0033 |
| fingerprints, calibrated | 0.1481 | 0.1479 | +0.0002 ± 0.0051 |
| wall-clock-only aux, calibrated | 0.1513 | 0.1479 | +0.0034 ± 0.0018 |
| forward-only tokens, direct | 0.1471 | 0.1556 | −0.0086 ± 0.0050 |
| legacy delta (uniform) vs legacy absolute | 0.1690 | 0.1562 | +0.0128 ± 0.0176 |

App-App, non-self (45 pair clusters, 90 directional outcomes; every method
covers all 55 clusters):

| Predictor | Log MAE | MAPE (%) | Signed log bias | Floor-case log MAE (18) | False-benign rate |
|---|---:|---:|---:|---:|---:|
| `bidirectional_context_calibrated` | 0.0938 | 8.55 | −0.048 | 0.063 | 0.31 |
| `joint_profile_calibrated` | 0.0940 | 8.89 | −0.018 | 0.090 | 0.10 |
| `legacy_absolute` | 0.0942 | 8.25 | −0.078 | 0.027 | 0.72 |
| `bidirectional_profile_calibrated` | 0.0953 | 9.13 | −0.009 | 0.099 | 0.03 |
| `bidirectional_profile_direct` | 0.0966 | 8.38 | −0.083 | 0.023 | 0.76 |
| `constant_1` | 0.1030 | 8.85 | −0.103 | 0.000 | 1.00 |
| `legacy_delta_median` | 0.1190 | 13.18 | +0.097 | 0.236 | 0.00 |
| `legacy_delta_uniform` | 0.1457 | 16.58 | +0.128 | 0.256 | 0.00 |

Paired cluster-bootstrap 95% intervals (method − comparison log MAE):
calibration of the joint model −0.0064 [−0.0248, 0.0143]; reverse
supervision (direct) −0.0038 [−0.0073, −0.0002], (calibrated) +0.0013
[−0.0010, 0.0035]; fingerprints (calibrated) −0.0015 [−0.0102, 0.0071],
(direct) +0.0066 [+0.0006, +0.0120]; wall-clock-only and forward-only token
ablations within ±0.003 of their references; joint direct vs legacy absolute
+0.0062 [+0.0012, +0.0118]; legacy delta vs legacy absolute +0.0515
[+0.0088, +0.0947].

**Interpretation.** On this snapshot, the new calibrated models tie with the
retrained legacy absolute model at about 9% lower log MAE than isolated speed.
None of the contrasts that isolate reverse-direction evidence or fingerprints
shows a reliable gain: the only interval excluding zero in their favour (reverse
supervision, direct, −0.004) is tiny and comes from 2-epoch models. Calibration's
practical effect is a trade-off: it removes most of the
absolute models' underprediction bias and cuts the false-benign rate (pairs
whose worse endpoint truly exceeds 1.05× but are predicted below it) from
0.72–0.76 to 0.03–0.31, at the cost of larger floor-case errors. Legacy delta
aggregation still overpredicts strongly (+0.13 log bias). Answering the
central question needs either more panel/application data or stronger
regularization, so that models trained for more than a couple of epochs
generalize across held-out applications and inhibitor blocks.

Plots: `outputs/full_comparison/plots/` (directional error and bias,
floor-case errors, pair severity/imbalance, auxiliary calibration, ablation
contrasts, context/calibration support counts).
