# Bidirectional Filtered-Inhibitor Pipeline — Total Application Time

This is a copy of `delta_response_filteredInhibs_biDirectional/` whose slowdown
target is **total application time** instead of MPI time:

```text
slowdown = max(1, T_app_co / T_app_iso)        # this directory
slowdown = max(1, T_MPI_co / T_MPI_iso)        # delta_response_filteredInhibs_biDirectional/
```

`T_app` is mpiP `AppTime` averaged over ranks and then over runs
(`App Time Average Mean` in `processed_data/`). The only code change is in
`prepare_bidirectional_data.py` (constant `SLOWDOWN_TIME = "runtime"`), plus
tests. Isolated profiles, auxiliary bandwidth/message-rate targets, the
inhibitor panel, all variants, losses, calibration, validation, and evaluation
are unchanged. The prepared `jobs.csv`, `inhibitors.csv`, and every auxiliary
column are byte-identical to the sibling's; only `slowdown`, `log_slowdown`,
`is_censored`, and the new `raw_app_time_ratio` differ. `raw_mpi_time_ratio`
is retained as a diagnostic, and `max(1, raw_mpi_time_ratio)` reproduces the
sibling's slowdowns exactly. The sibling directory is untouched. The
[results](#results-on-the-repository-data-outputsfull_comparison) below compare
both targets on the same snapshot and machine.

The rest of this README is inherited, with target definitions updated. The
inherited pipeline description follows.

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
DIR=notebooks/utils/delta_response_filteredInhibs_biDirectional_totalTime

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

# Tests (55; synthetic data are generated in temporary directories).
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
observations (611 at the application-time slowdown floor versus 443 for MPI
time, none rejected), 55 eight-node App-App pairs (10 self-pairs; 36 of 90
non-self directional outcomes at the floor versus 18 for MPI time). Every row is an aggregate configuration
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
| `slowdown` | `max(1, T_app_co / T_app_iso)` of the application endpoint (total application time) |
| `log_slowdown`, `is_censored` | `log(slowdown)`; `raw_app_time_ratio <= 1` |
| `raw_app_time_ratio` | unclipped application-time ratio |
| `raw_mpi_time_ratio` | unclipped MPI-time ratio, diagnostic only (the sibling's target) |
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
`app_bw_wall` equals `app_msg_wall`. **With the application-time target the
redundancy moves:** `app_bw_wall` (and `app_msg_wall`) now equal the
*unclipped* log target `log(raw_app_time_ratio)`, so the application-endpoint
wall-clock auxiliary tasks supervise the latent target below the floor. In
contrast, `app_bw_mpi` carries independent MPI-time information. The
inhibitor-endpoint tasks are unaffected.

Application-time caveats: mpiP prints `AppTime` with three significant digits,
i.e. whole seconds for these ~300–400 s runs. Isolated run-to-run standard
deviations are 0.5–5 s (16 s for `laghos`). Inhibitor application time is
a controlled duration (≈360 s isolated; capped to the partner's runtime when
co-scheduled).

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

Full budget, all eight variants, same nine-inhibitor panel as the MPI-time
sibling (selection uses isolated profiles only), 90 fitting rows (27 at the
application-time floor versus 16). In crossed CV, `fiesta` keeps a single
uncensored panel anchor, so legacy potentials skip 10 fold/seed fits
(`fiesta`, block 3). Joint variants evaluate every group. Run time was about
45 minutes on one CPU core. The MPI-time column is the unchanged sibling
pipeline re-run on the same machine and snapshot, and it reproduces the
sibling README exactly. **The App-App data are the historical,
already-inspected dataset: an exploratory comparison, not an untouched
holdout.**

**Selected recipes (application time).** Final epochs: legacy delta 3,
legacy generic 4, legacy absolute 6, joint 5, bidirectional profile 3,
context 1, wall-only 4, forward-only context 2 (MPI time: 13/14/2 and 1–2 for
joint variants). Calibration λ = 1 everywhere except the forward-only context
ablation (λ = 10), as for MPI time.

App-App, non-self (45 pair clusters, 90 directional outcomes; every method
covers all 55 clusters). Δ is method minus `constant_1` log MAE with the
paired cluster-bootstrap 95% interval:

| Predictor | App time: log MAE | Signed log bias | Floor-case log MAE (36) | Δ vs constant 1 | MPI time: log MAE | Δ vs constant 1 |
|---|---:|---:|---:|---:|---:|---:|
| `joint_profile_calibrated` | 0.0139 | −0.013 | 0.0004 | −0.0017 [−0.0033, −0.0004] | 0.0940 | −0.0090 [−0.0292, 0.0140] |
| `bidirectional_profile_wall_calibrated` | 0.0143 | −0.012 | 0.0007 | −0.0013 [−0.0027, −0.0001] | 0.0950 | −0.0080 [−0.0293, 0.0170] |
| `bidirectional_profile_calibrated` | 0.0147 | −0.012 | 0.0008 | −0.0010 [−0.0022, 0.0003] | 0.0953 | −0.0077 [−0.0290, 0.0172] |
| `bidirectional_context_calibrated` | 0.0154 | −0.015 | 0.0000 | −0.0003 [−0.0007, 0.0000] | 0.0938 | −0.0092 [−0.0249, 0.0084] |
| `constant_1` | 0.0156 | −0.016 | 0.0000 | – | 0.1030 | – |
| `bidirectional_profile_direct` | 0.0158 | −0.015 | 0.0005 | +0.0002 [0.0000, 0.0004] | 0.0966 | −0.0064 [−0.0119, −0.0002] |
| `legacy_absolute` | 0.0162 | −0.013 | 0.0022 | +0.0006 [−0.0010, 0.0022] | 0.0942 | −0.0088 [−0.0146, −0.0032] |
| `legacy_delta_uniform` | 0.0199 | +0.015 | 0.0215 | +0.0042 [−0.0028, 0.0106] | 0.1457 | +0.0427 [−0.0024, 0.0900] |

In relative terms, `constant_1` has 1.51% MAPE under application time
(8.85% under MPI time), and `joint_profile_calibrated` has 1.35%.

Ablation contrasts (application time, App-App log MAE, paired 95% interval):
floor-aware calibration of the joint model −0.0017 [−0.0033, −0.0004];
joint direct vs legacy absolute −0.0006 [−0.0022, 0.0010]; reverse supervision
(direct) +0.0002 [+0.0000, +0.0004] and (calibrated) +0.0008 [+0.0002,
+0.0013]; fingerprints (calibrated) +0.0007 [−0.0004, +0.0018]; wall-clock-only
aux (calibrated) −0.0003 [−0.0009, +0.0002]; forward-only tokens (calibrated)
+0.0021 [−0.0001, +0.0042]. The App-Inhibitor CV contrasts in
`ablation_summary.csv` point the same way. Calibration helps (−0.0021 ± 0.0012
grouped SE), joint direct beats legacy absolute (−0.0094 ± 0.0039), and reverse
supervision slightly hurts (+0.0013 ± 0.0008).

**Interpretation.** Measured by total application time, interference on these
8-node runs is around 1–2%, near the whole-second resolution of `AppTime`.
Every *direct* predictor collapses to the floor and is indistinguishable from
`constant_1` (`bidirectional_context_direct` equals it exactly). The only
reliable gain comes from **floor-aware calibration**: `joint_profile_calibrated`
lowers log MAE by about 11% relative to isolated speed, and its interval
excludes zero. Under MPI time, calibrated joint models gain a similar ~9%, but
their intervals include zero; there the reliable gain comes from the direct
absolute models (`legacy_absolute`, −0.0088 [−0.0146, −0.0032]). Reverse-direction supervision and fingerprints still give no
gain; reverse supervision is slightly harmful here. Its application-endpoint
wall-clock labels now duplicate the unclipped target (see Stage 1), so that
contrast mostly measures the inhibitor-endpoint and MPI-throughput tasks. Legacy
delta aggregation still overpredicts (+0.015 log bias, floor-case log MAE
0.022), although less severely than under MPI time.

Plots: `outputs/full_comparison/plots/` (directional error and bias,
floor-case errors, pair severity/imbalance, auxiliary calibration, ablation
contrasts, context/calibration support counts).
