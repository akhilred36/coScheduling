# Implementation plan: bidirectional filtered-inhibitor pipeline

**Target directory:** `notebooks/utils/delta_response_filteredInhibs_biDirectional/`

This plan specifies a staged implementation covering:

1. Consistent reconstruction of isolated profiles and bidirectional App-Inhibitor measurements.
2. Joint absolute/delta response learning.
3. Floor-aware inference calibration.
4. Reverse-direction bandwidth auxiliary supervision.
5. Application response fingerprints.
6. Leakage-safe validation, reproducible experiments, and comparison against the existing pipeline.

Both requested bandwidth proxies are included:

- **Bytes per total runtime:** achieved average traffic rate.
- **Bytes per MPI time:** MPI-time-normalized effective throughput.

An explicit iterative mutual-throttling model is a subsequent research extension, outside these implementation stages.

---

## 1. Goals and scientific contract

### 1.1 Primary prediction task

Given two applications, their isolated profiles, and measurements against a shared filtered inhibitor panel, predict:

\[
\hat s_A(B),\qquad \hat s_B(A)
\]

where \(s_A(B)\) is the clipped MPI-time slowdown of \(A\) when paired with \(B\).

The new pipeline must support:

- Direct absolute-response predictions.
- Anchor-calibrated response predictions.
- Replacement deltas derived from response potentials.
- Predictions and diagnostics for bandwidth changes in both directions.

### 1.2 Training information

Model fitting and selection use:

- Isolated application profiles.
- Isolated inhibitor profiles.
- Application slowdown in App-Inhibitor runs.
- Application and inhibitor bandwidth changes in those same runs.
- Response fingerprints constructed from permitted App-Inhibitor support measurements.

App-App outcomes remain final-evaluation data.

At inference, both applications may use their permitted inhibitor-panel measurements. This retains the existing few-shot inference assumption.

### 1.3 Directional semantics

Use this notation throughout code and documentation:

\[
R(V,X)=\text{latent log slowdown of victim }V\text{ under aggressor }X
\]

\[
D_t(V,X)=\text{signed log degradation of endpoint }V
\text{ under partner }X
\]

where \(t\) identifies a bandwidth or message-rate task.

Do not assume:

\[
R(A,B)=R(B,A)
\]

Do preserve replacement-delta structure:

\[
\Delta_A(J,K)=R(A,K)-R(A,J)
\]

Swapping pair inputs must swap their directional predictions.

### 1.4 Success criteria

Implementation success means the pipeline is correct, reproducible, and evaluable—not that a new model necessarily beats the existing absolute model.

Report separately:

- Improvements from absolute/delta joint learning.
- Improvements from floor-aware calibration.
- Improvements from reverse-direction supervision.
- Improvements from measured response fingerprints.

---

## 2. Repository structure and reuse strategy

Create the following directory during implementation:

```text
notebooks/utils/delta_response_filteredInhibs_biDirectional/
├── .gitignore
├── README.md
├── plan.md
├── schematic.md
├── prepare_bidirectional_data.py
├── data.py
├── contexts.py
├── model.py
├── training.py
├── inference.py
├── metrics.py
├── select_inhibitors.py
├── pipeline.py
├── run_experiments.py
├── analyze_experiments.py
├── plot_prediction_errors.py
├── comparison_config.json
├── data/
├── outputs/
└── tests/
    ├── make_synthetic_data.py
    ├── test_data.py
    ├── test_contexts.py
    ├── test_models.py
    ├── test_losses.py
    ├── test_calibration.py
    ├── test_pipeline.py
    └── test_experiments.py
```

### Reuse

Use `delta_response_filteredInhibs/` as the starting implementation.

Reuse or adapt:

- Input validation and guarded slowdown transformations.
- Application-targeted inhibitor selection.
- Profile scaling.
- Inhibitor-block construction.
- Endpoint-familiarity tiers.
- Model-specific epoch selection.
- Grouped aggregation selection for legacy methods.
- Pair-cluster bootstrap.
- Experiment manifests, immutable input hashes, and resumability.
- Existing output and plotting conventions.

Copy source and relevant tests, not historical outputs, caches, checkpoints, or generated audit results.

The new pipeline should be independently runnable. Avoid runtime imports from its sibling directory that create ambiguous `data`, `model`, or `training` module resolution.

Preserve the original implementation for comparison.

---

## 3. Stage 1: reconstruct consistent bidirectional data

### 3.1 Resolve the source-provenance issue first

A repository inspection found a concrete discrepancy:

- `few_shot/data/inhibitors.csv` gives inhibitor `8_20_5000_0.01` an isolated MPI time of approximately `3.6493`.
- `processed_data/data_inhib_isolated_agg.csv` gives the matching configuration approximately `8.1483`.

Therefore, **do not enrich the legacy response CSV by blindly joining current processed reverse measurements**.

The default new data preparation must rebuild profiles and responses from one consistent processed-data snapshot.

Use:

```text
processed_data/data_isolated_agg.csv
processed_data/data_inhib_isolated_agg.csv
processed_data/data_inhib_coscheduled_agg.csv
```

App-App evaluation preparation uses, separately:

```text
processed_data/data_coscheduled_agg.csv
```

Default to eight nodes, matching the current experiment.

### 3.2 Preparation interfaces

Implement:

```python
prepare_training_data(
    isolated_apps_csv,
    isolated_inhibitors_csv,
    app_inhibitor_csv,
    num_nodes,
    output_dir,
) -> PreparationReport
```

and a separate evaluation function:

```python
prepare_pair_evaluation_data(
    app_app_csv,
    prepared_jobs_csv,
    num_nodes,
    output_dir,
) -> PreparationReport
```

The training preparation function must not read App-App data.

For the initial implementation, use the aggregate source files and explicitly record:

```text
observation_unit = aggregate_configuration
rate_estimator = ratio_of_aggregate_means
```

Do not describe these rows as run-level replicates.

Run-level preparation can be added later using the non-aggregate files and retained run identities.

### 3.3 Canonical inhibitor IDs

Construct inhibitor IDs from:

```text
Num Nodes
Inhib Message Size
Inhib Wait Time (us)
Inhib Comm Sparsity
```

Use a single canonical formatting function:

- Integer fields written without `.0`.
- Sparsity formatted consistently using decimal normalization.
- Reject ambiguous duplicate configurations.
- Use the same function for isolated and co-scheduled rows.

Example:

```text
8_20_5000_0.01
```

Export an ID/configuration mapping.

### 3.4 Isolated profiles

Retain the existing common feature schema:

```text
job_id or inhib_id
mpi_time
comm_frac
total_msgs
total_bytes
```

Add explicit raw runtime and configuration fields:

```text
runtime
num_nodes
```

For inhibitors, retain control fields as metadata:

```text
msg_size
wait_time
comm_sparsity
```

Calculate:

\[
\mathrm{comm\_frac}=\frac{T_{\mathrm{MPI}}}{T_{\mathrm{runtime}}}
\]

Use the matching aggregate-mean columns.

### 3.5 Application slowdown

Preserve existing operational target semantics:

\[
s_A(J)=\max\left(1,
\frac{T_{\mathrm{MPI},A}^{co}(J)}
     {T_{\mathrm{MPI},A}^{iso}}
\right)
\]

\[
y_A(J)=\log s_A(J)
\]

Store:

```text
slowdown
log_slowdown
is_censored
raw_mpi_time_ratio
```

The raw ratio is diagnostic metadata. Do not silently substitute it for the clipped slowdown target.

### 3.6 Signed rate-change targets

For each endpoint \(E\), calculate:

\[
B_E^{wall}=\frac{V_E}{T_E^{runtime}}
\]

\[
B_E^{mpi}=\frac{V_E}{T_E^{MPI}}
\]

Also calculate whole-runtime message rate:

\[
M_E^{wall}=\frac{N_E}{T_E^{runtime}}
\]

For each proxy \(t\):

\[
d_E^t(F)=
\log B_E^{t,iso}-\log B_E^{t,co}(F)
\]

Use the corresponding message-rate expression for \(M^{wall}\).

Interpretation:

- Positive: degradation.
- Zero: unchanged.
- Negative: increased achieved rate.

Compute log differences directly to reduce numerical overflow risk.

Do not clip these targets at zero.

### 3.7 Response schema

Each prepared App-Inhibitor observation should contain:

```text
job_id
inhib_id
observation_id
replicate_id
num_nodes

slowdown
log_slowdown
is_censored
raw_mpi_time_ratio

app_bw_wall_log_degradation
app_bw_mpi_log_degradation
app_msg_wall_log_degradation

inhib_bw_wall_log_degradation
inhib_bw_mpi_log_degradation
inhib_msg_wall_log_degradation

<one validity mask per auxiliary target>
```

Preserve underlying isolated and co-scheduled quantities in an audit table.

For aggregate input, normally there is one observation per configuration and `replicate_id = 0`.

### 3.8 Validation rules

- Require finite, positive denominators.
- Require nonnegative counts and volumes.
- A zero rate makes its log-ratio task unavailable.
- Missing or invalid auxiliary measurements mask that task; they do not automatically remove a valid application-slowdown observation.
- Invalid slowdown measurements retain existing rejection behavior.
- Reject ambiguous joins.
- Distinguish missing joins from invalid measurements.
- Never Cartesian-join observations lacking a reliable replicate identity.

Emit:

```text
preparation_report.json
source_consistency_report.csv
rejected_observations.csv
auxiliary_target_availability.csv
observation_audit.csv
```

Include source hashes, filters, target definitions, units, and row counts.

---

## 4. Inhibitor-panel selection

Retain the current application-targeted facility-location method.

Use the same four isolated communication features:

```text
comm_frac
log1p(message_rate)
log1p(byte_rate)
log1p(mean_message_size)
```

Retain:

- Application-only standardization.
- Valid forward-response availability for all target applications.
- Maximum ten inhibitors.
- Mean/worst squared-distance objective with tail weight `0.5`.
- Nearest-candidate union when it fits the budget.
- Greedy additions and one-for-one swaps otherwise.
- Removal of redundant profiles.

### Important requirements

1. Auxiliary response magnitudes must not affect panel selection.
2. Missing auxiliary labels must not silently change panel eligibility.
3. All models and fingerprints use the same selected panel.
4. Preserve additional response columns when exporting filtered data.
5. Restricted tiers design the panel using only their training applications.
6. Record panel design scope separately from fold-local model fitting.

The rebuilt source snapshot may not select exactly nine inhibitors. Verify and report the result; do not hardcode nine IDs.

Provide an optional frozen-panel input for controlled comparisons. Record whether a panel was recomputed or supplied.

---

## 5. Model and ablation registry

Replace hardcoded model-kind loops with a registry.

Required training variants:

| Variant | Response objective | Auxiliary tasks | Fingerprints |
|---|---|---|---|
| `legacy_delta` | Delta only | No | No |
| `legacy_generic` | Delta only | No | No |
| `legacy_absolute` | Absolute only | No | No |
| `joint_profile` | Absolute + delta | No | No |
| `bidirectional_profile` | Absolute + delta | Both endpoints | No |
| `bidirectional_context` | Absolute + delta | Both endpoints | Yes |

Required inference methods:

```text
constant_1
legacy_absolute
legacy_delta_<aggregation>
legacy_generic_<aggregation>

joint_profile_direct
joint_profile_calibrated

bidirectional_profile_direct
bidirectional_profile_calibrated

bidirectional_context_direct
bidirectional_context_calibrated
```

This separates the benefit of a training objective from the benefit of calibration.

Legacy variants must retain the original architecture and losses for faithful same-source comparisons.

---

## 6. Stage 2: joint absolute/delta learning

### 6.1 Response architecture

Start with the existing shared profile encoder:

```text
input_dim → 8 → 4
ReLU
```

For new variants, use:

\[
R(V,X)=b(e_V)+h(e_X)+u(e_V)^\top v(e_X)
\]

where:

- \(e_E\) is the endpoint embedding.
- \(b\) learns victim-specific response level.
- \(h\) learns aggressor pressure.
- \(u,v\) are rank-two role projections by default.

The additional victim-only term cancels in deltas but receives supervision from absolute losses.

For profile-only variants:

\[
e_E=E(x_E)
\]

### 6.2 Delta invariants

Implement deltas only by subtracting response predictions:

```python
delta(victim, anchor, query, fixed_contexts)
    = response(victim, query, fixed_contexts)
    - response(victim, anchor, fixed_contexts)
```

Do not introduce an independent delta head.

Invariants must hold for fixed endpoint contexts:

- Identity.
- Antisymmetry.
- Path consistency.

### 6.3 Joint loss

Use:

\[
L=
\lambda_{abs}L_{abs}
+\lambda_{\Delta}L_{\Delta}
+\lambda_{aux}L_{aux}
\]

Initial values:

```text
lambda_abs = 1.0
lambda_delta = 1.0
lambda_aux = 0.1
```

For `joint_profile`, set `lambda_aux = 0`.

Reuse the existing censored Huber formulations:

- Exact slowdown targets above the floor.
- One-sided absolute constraint at the floor.
- Exact/one-sided delta constraints depending on censoring.
- No identified delta constraint when both observations are at the floor.

For new joint variants, average delta loss over identified constraints. When none are available, return a differentiable zero for that component.

Keep the legacy loss reduction unchanged in legacy variants.

### 6.4 Balanced sampling

Sample applications uniformly, then configurations within application.

For new variants:

- Sample distinct inhibitor IDs before selecting replicate rows.
- Prevent a configuration with many replicates from dominating training.
- Keep both endpoint auxiliary labels from an observation together.
- Record component losses and valid-task counts per epoch.

---

## 7. Stage 3: reverse-bandwidth auxiliary supervision

### 7.1 Shared directional bandwidth predictor

Implement one directional predictor:

```python
bandwidth_response(victim_embedding, aggressor_embedding)
```

returning:

```text
bw_wall_log_degradation
bw_mpi_log_degradation
msg_wall_log_degradation
```

Evaluate it in both orders for an App-Inhibitor run:

```python
bandwidth_response(app, inhibitor)
bandwidth_response(inhibitor, app)
```

Thus, app and inhibitor bandwidth changes train a common directional mechanism.

### 7.2 Architecture

Use the same endpoint embeddings and role projections as the slowdown model.

A small head can consume:

```text
[e_victim,
 e_aggressor,
 u(e_victim),
 v(e_aggressor),
 u(e_victim) * v(e_aggressor)]
```

Default auxiliary hidden dimension: `8`.

Sharing the role projections ensures reverse-task gradients reach the representation used to describe an application as an aggressor.

Do not invent inhibitor slowdown labels.

### 7.3 Auxiliary normalization and masks

Fit one target scale per task on fold-training measurements only:

\[
\sigma_t=\max(\operatorname{std}(d_t),0.1)
\]

Keep predictions in physical log-ratio units. Normalize residuals for the loss:

\[
L_t=\operatorname{Huber}
\left(
\frac{\hat d_t-d_t}{\sigma_t}
\right)
\]

Average:

1. Within each task over valid observations.
2. Across available tasks.

An entirely unavailable task contributes neither loss nor denominator.

Save target scales in checkpoints.

### 7.4 Evaluation

Report auxiliary errors by:

- Endpoint type.
- Direction.
- Task.
- Application.
- Inhibitor block.

MPI-time-normalized throughput can be algebraically redundant with timing slowdown when byte volume is unchanged. Include a wall-clock-only auxiliary ablation in experiment configurations to assess this.

---

## 8. Stage 4: application response fingerprints

This stage requires explicit support/query separation.

### 8.1 Scope of the first fingerprint implementation

Build fingerprints for applications from their inhibitor-panel measurements.

For inhibitor endpoints, use:

```text
context embedding = zero
context available = false
```

This gives a clear cold-endpoint definition and avoids building inhibitor fingerprints from potentially held-out application responses.

### 8.2 Support tokens

For an application \(A\) and permitted support inhibitor \(J\), construct:

```text
isolated profile of J

application log slowdown
application censoring flag

application wall-bandwidth degradation
application MPI-bandwidth degradation
application wall-message-rate degradation

inhibitor wall-bandwidth degradation
inhibitor MPI-bandwidth degradation
inhibitor wall-message-rate degradation

validity masks
```

These contain:

- Incoming susceptibility evidence.
- Outgoing aggressiveness evidence.

No application IDs or inhibitor IDs become learned categorical features.

### 8.3 Replicate handling

Create one token per inhibitor configuration:

- Mean of available numeric auxiliary labels.
- Mean observed log slowdown.
- Fraction of slowdown observations at the floor.
- Per-task availability indicators.

Retain raw observations for calibration losses and audit output.

This prevents repeated runs from implicitly weighting one inhibitor more heavily in the fingerprint.

### 8.4 Set encoder

Implement a small permutation-invariant encoder:

```text
token features → MLP → 4-dimensional token embedding
masked mean pooling → 4-dimensional context embedding
```

Default token hidden dimension: `8`.

Include context-presence and support-count features in the fusion layer:

\[
e_A=
E(x_A)+
M(z_A,\mathbf{1}_{support},\log(1+n_{support}))
\]

For empty support, define the additive context contribution as exactly zero.

Do not scale token slowdown values using validation outcomes. Use training-only scales for auxiliary token features.

### 8.5 Training episodes

An episode specifies:

```text
victim application A
support inhibitor set S
query inhibitor set Q
```

with:

\[
S\cap Q=\varnothing
\]

For joint-delta training, require at least two distinct query inhibitors when available.

Sample support size from:

```text
0 … min(number_of_available_inhibitors - 2, configured_max_support)
```

Use an explicit empty-support probability, initially `0.25`.

For nonempty episodes, select remaining sizes uniformly.

All predicted query responses and deltas in the episode use the **same application fingerprint constructed from S**.

Inhibitor query contexts are always empty.

### 8.6 Leakage rules

A query observation must not appear in:

- Its application's fingerprint.
- A token-normalization fit.
- Any cached support representation used for that prediction.

Exclude all replicates of a query inhibitor ID from application support.

Cache keys must include:

```text
fold
application
support inhibitor IDs
support observation IDs
normalization state
model/checkpoint identity
```

Cache raw support tensors during training, not stale learned embeddings.

### 8.7 Direct inference

For an App-App direct prediction:

- Build each application fingerprint from its permitted selected-panel support.
- Evaluate the same directional response function in both orders.
- Never consume outcomes from the query App-App run.

Document that `bidirectional_context_direct` is an **absolute-response predictor with measured support features**, rather than an isolated-profile-only predictor.

---

## 9. Floor-aware calibration

### 9.1 Apply to level-identified models

Use floor-aware calibration for:

```text
joint_profile
bidirectional_profile
bidirectional_context
```

Their absolute loss identifies a response level.

Retain existing aggregation for delta-only legacy models. A zero-centered correction prior is not automatically meaningful for a delta-only potential with an unidentified additive gauge.

### 9.2 Calibration objective

For predicted anchor responses \(r_J\), estimate scalar correction \(c_A\):

\[
\hat c_A=
\arg\min_c
\left[
\sum_J w_J\ell_J(c)
+\frac{\lambda_c}{2}c^2
\right]
\]

with:

\[
\ell_J(c)=
\begin{cases}
\rho(r_J+c-y_J), & y_J>0\\
\rho(\max(0,r_J+c)), & y_J=0
\end{cases}
\]

and prediction:

\[
\hat y_A(B)=
\max(0,R(A,B)+\hat c_A)
\]

Normalize weights to sum to one.

Give inhibitor configurations equal total weight; divide that weight among their replicates.

Default:

```text
calibration_weights = uniform
calibration_lambda = 1.0
```

Allow App-Inhibitor CV selection over:

```text
calibration_lambda ∈ {0.1, 1.0, 10.0}
```

### 9.3 Numerical implementation

Implement the convex scalar optimization using derivative-root bisection.

Requirements:

- Positive regularization gives a unique solution.
- Expand the initial bracket until the derivative changes sign.
- Deterministic tolerance and iteration limits.
- Explicit non-finite checks.
- Return convergence diagnostics.

Handle:

- Mixed exact/floor anchors.
- All-floor anchors.
- Single-anchor support.
- Empty support: use `c = 0`, record fallback.
- Missing auxiliary labels: irrelevant to calibration feasibility.

All-floor support must no longer force new joint variants to skip validation.

### 9.4 Cross-fitted calibration for fingerprint models

A label used to build a fingerprint must not also calibrate its own response prediction.

For each application, deterministically partition its available inhibitor IDs into up to three groups:

```text
C1, C2, C3
```

For view \(k\):

- Fingerprint support: \(S_k = panel \setminus C_k\).
- Calibration anchors: \(C_k\).
- Predict both anchors and the target using the fixed fingerprint from \(S_k\).

Average the view-level latent query predictions, then apply the observation floor.

For a single available inhibitor:

- Empty fingerprint.
- That inhibitor is the calibration set.

For App-App inference, the aggressor's fingerprint may use its complete permitted panel because those measurements are independent of the query App-App outcome. Keep it fixed across the victim's calibration views.

Report the number of views, context counts, calibration counts, and exact/floor counts per view.

Do not treat view dispersion as a calibrated prediction interval.

---

## 10. Validation and endpoint-familiarity protocols

### 10.1 Preserve crossed application/inhibitor-block validation

For held-out application \(A^*\) and block \(B^*\):

```text
training applications: all allowed applications except A*
training inhibitors: selected panel excluding B*

training outcomes:
    application != A*
    inhibitor not in B*

support for A*:
    inhibitor not in B*

validation queries:
    application == A*
    inhibitor in B*
```

The exclusion applies to **all labels from a run**, including reverse bandwidth measurements.

### 10.2 Fold-local fitting

Fit only on training inputs:

- Profile scaler.
- Auxiliary target scales.
- Fingerprint token scales.
- Model weights.

Construct validation fingerprints from permitted support after fitting the scalers.

The selected global panel remains a documented preprocessing decision based on the run's allowed training population, consistent with the existing pipeline.

### 10.3 Validation methods

For each new checkpoint, evaluate:

- Direct slowdown predictions.
- Floor-aware calibrated slowdown predictions.
- Both-direction auxiliary predictions where labels are available.

For context calibration, use the cross-fitted procedure from Section 9.

Record predictions for **absolute and new joint models as well as potential models**. The existing pipeline records aggregation predictions mainly for potential models; extend this so all methods can be compared on common query rows.

### 10.4 Selection

- Primary selection metric: grouped application-slowdown log MAE.
- Average query errors within victim/block/seed, then across seeds.
- Auxiliary errors are diagnostic and do not determine the winning slowdown model.
- Choose model-specific epochs as the median selected best epoch.
- Select calibration settings using App-Inhibitor CV only.
- Record eligible/skipped groups for every variant.
- Provide matched-fold comparisons when legacy delta skips groups that new methods can evaluate.

Do not let differences in evaluated fold sets masquerade as improved accuracy.

### 10.5 Restricted endpoint tiers

Preserve:

```text
random_split
one_known
zero_shot
both
```

For restricted tiers:

- Unknown applications do not fit model parameters or scalers.
- Their forward and reverse App-Inhibitor measurements may construct inference support.
- Record that support access explicitly.
- Global recipe selection using all applications remains a transductive selection condition, as in the existing runner.

---

## 11. Pipeline interfaces and checkpoints

### 11.1 Data structures

Extend the current data container with:

```python
@dataclass
class BidirectionalTrainingData:
    jobs: pd.DataFrame
    inhibitors: pd.DataFrame
    responses: pd.DataFrame
    rejected_responses: pd.DataFrame
    duplicate_keys: pd.DataFrame
    preparation_metadata: dict
```

Add explicit structures for:

```text
AuxiliaryScaler
ContextConfig
CalibrationConfig
ModelVariantSpec
FoldContext
PredictionDiagnostics
```

Prefer typed configuration objects over implicit DataFrame-column conventions.

### 11.2 Key APIs

```python
load_bidirectional_training_data(...)
build_support_tokens(...)
build_application_context(...)
sample_joint_episode(...)
train_joint_response(...)
predict_pair_direct(...)
fit_floor_aware_offset(...)
predict_with_floor_aware_calibration(...)
predict_with_crossfit_context_calibration(...)
evaluate_bidirectional_holdout(...)
```

Keep model prediction separate from outcome-table loading.

### 11.3 CLI

Preserve existing pipeline flags and add:

```text
--variants
--frozen-panel-csv
--context-max-support
--context-crossfit-folds
--calibration-lambda
```

Preparation CLI:

```text
prepare_bidirectional_data.py training
    --isolated-apps-csv
    --isolated-inhibitors-csv
    --app-inhibitor-csv
    --num-nodes
    --output-dir

prepare_bidirectional_data.py evaluation
    --app-app-csv
    --jobs-csv
    --num-nodes
    --output-dir
```

Training preparation defaults must not require an App-App path.

### 11.4 Initial configuration

Retain inherited computational defaults:

```text
CV seeds: 0–4
Final seeds: 0–4
Inhibitor blocks: 4
Maximum epochs: 80
Patience: 10
Batches per epoch: 4
Batch size: 128
Bootstrap draws: 1000
```

New defaults:

```text
rank: 2
profile hidden dimension: 8
profile embedding dimension: 4
context embedding dimension: 4
context token hidden dimension: 8

lambda_abs: 1.0
lambda_delta: 1.0
lambda_aux: 0.1

context_empty_probability: 0.25
context_crossfit_folds: 3
context_max_support: 7

auxiliary_tasks:
    bw_wall
    bw_mpi
    msg_wall

calibration_lambda_candidates:
    0.1
    1.0
    10.0
```

Treat these as initial recipes; selection provenance must distinguish fixed settings from searched settings.

### 11.5 Checkpoint contents

Save:

```text
schema version
variant
seed
selected epoch count
training applications
selected inhibitor IDs
model configuration
profile scaler
auxiliary target scales
context token scales
context construction recipe
calibration recipe
source/prepared-data hashes
model state
```

Checkpoints must support inference without refitting any scaler.

Reject incompatible schemas clearly.

---

## 12. Evaluation, outputs, and experiment runner

### 12.1 App-App loading boundary

Load App-App evaluation measurements only after:

- Selection.
- Final fitting.
- Checkpoint writing.

Then perform inference and reporting only.

Hashing evaluation inputs for provenance is permitted; parsing their outcomes during tuning is not.

Prepare rich App-App labels separately so evaluation can include both bandwidth proxies and both endpoint directions.

### 12.2 Required prediction outputs

Retain existing directional prediction fields and add:

```text
variant
inference_method
support_information_regime

context_inhibitor_count
calibration_inhibitor_count
exact_calibration_count
floor_calibration_count
calibration_offset
calibration_converged
calibration_view_count

predicted_bw_wall_log_degradation
predicted_bw_mpi_log_degradation
predicted_msg_wall_log_degradation
```

Keep auxiliary truths and masks in an auxiliary prediction table.

Keep raw App-App outcomes outside context construction.

### 12.3 Metrics

Retain existing slowdown metrics and paired cluster bootstrap.

Add:

**Directional diagnostics**

- Signed log bias.
- Floor-case log MAE.
- Above-floor log MAE.
- Per-application metrics.
- Auxiliary MAE/RMSE by task and direction.

**Pair-level diagnostics**

\[
severity=\frac{y_A+y_B}{2}
\]

\[
imbalance=y_A-y_B
\]

Also report:

- Error in worst-endpoint slowdown.
- Worst-endpoint slowdown ranking.
- False-benign pair rate for a fixed slowdown threshold, initially `1.05`.
- Swap-equivariance residual.

Evaluate non-self pairs as the primary comparison. Keep self-pair results separately using existing handling.

### 12.4 Bootstrap

Resample original unordered pair clusters.

Preserve:

- Both directional outcomes.
- All auxiliary labels.
- Method alignment.

Reuse one draw matrix for paired method comparisons.

Model seeds are repeated fits, not independent pair observations.

### 12.5 Experiment runner

Extend `run_experiments.py` to:

- Use the variant registry.
- Hash all prepared training and evaluation inputs.
- Include context and calibration settings in task identity.
- Freeze model-specific recipes for fixed-global runs.
- Consolidate direct and calibrated methods separately.
- Record panel and support-information regimes.
- Preserve smoke and standard execution profiles.

Required ablations:

1. Legacy models on the rebuilt source data.
2. Joint absolute/delta model.
3. Joint model with floor-aware calibration.
4. Bidirectional auxiliary model.
5. Bidirectional context model.
6. Wall-clock-only versus both-bandwidth auxiliary tasks.
7. Forward-only versus forward-plus-reverse context tokens.

Use the same source snapshot, panel policy, seeds, and evaluation pairs for matched comparisons.

### 12.6 Reporting files

Add:

```text
auxiliary_validation_predictions.csv
auxiliary_validation_metrics.csv
context_support_manifest.csv
calibration_diagnostics.csv
pair_predictions.csv
pair_metrics.csv
metrics_by_censoring.csv
matched_fold_comparisons.csv
ablation_summary.csv
```

Plots should cover:

- Directional error and bias.
- Floor-case errors.
- Pair severity and imbalance.
- Auxiliary prediction calibration.
- Forward/reverse context ablations.
- Context and calibration support counts.

---

## 13. Test plan

### 13.1 Data preparation

Test:

- Correct rate and log-degradation calculations for both endpoints.
- Negative degradation preserved.
- MPI-throughput/timing identity when byte counts are fixed.
- Changing inhibitor work volume.
- Zero-rate masking.
- Canonical inhibitor IDs.
- Ambiguous joins rejected.
- Aggregate metadata retained.
- Source inconsistency detected.
- Training preparation never reads App-App data.

### 13.2 Model and loss behavior

Test:

- Delta identity, antisymmetry, and path consistency.
- Swap-equivariant bandwidth outputs.
- Asymmetric endpoint outcomes remain possible.
- Floor losses enforce the correct direction.
- Both-floor delta pairs contribute no invented equality constraint.
- All-masked auxiliary tasks yield finite differentiable loss.
- Reverse labels produce gradients in shared role representations.

### 13.3 Context leakage

Test:

- Token permutation invariance.
- Query inhibitor excluded across all replicates.
- Held-out block absent from every training context.
- Held-out application labels absent from scaler fitting and parameter fitting.
- Empty-context behavior.
- Cross-fitted calibration/context sets are disjoint.
- Mutating a query label does not change its input context.
- Unknown-application support is used only under the permitted inference regime.

### 13.4 Calibration

Test:

- Mixed exact/floor anchors.
- All-floor support.
- Single-anchor support.
- Empty-support fallback.
- Deterministic convergence.
- Finite outputs.
- Equal configuration weighting with replicates.
- Cross-fitted contexts remain fixed while comparing anchor and target responses.

### 13.5 End-to-end

Build deterministic synthetic data with:

- Two distinct susceptibility/aggressiveness patterns.
- Both bandwidth proxies.
- Some negative auxiliary targets.
- Missing auxiliary labels.
- All-floor support for at least one application.
- App-App pairs with unequal directional effects.

Run:

- Holdout-free fit.
- Full synthetic evaluation.
- Restricted tiers.
- Fixed-global replay.
- Checkpoint reload.
- Experiment resume and consolidation.

Use temporary directories rather than requiring preexisting generated fixtures.

---

## 14. Implementation order and verification gates

### Milestone 1 — data and source consistency

Implement preparation, schemas, masks, and selection export.

**Gate:** deterministic preparation, source hashes, correct joins, and passing data tests.

### Milestone 2 — same-source legacy baseline

Run copied legacy models using the rebuilt prepared data.

**Gate:** independent pipeline execution and documented differences from historical results.

Historical numbers need not reproduce if the source snapshot differs.

### Milestone 3 — joint response and calibration

Implement `joint_profile` and floor-aware inference.

**Gate:** delta invariants, correct losses, all-floor calibration, and complete crossed CV.

### Milestone 4 — bidirectional auxiliary learning

Implement `bidirectional_profile`.

**Gate:** both endpoint labels train correctly, masking is correct, and auxiliary predictions survive checkpoint reload.

### Milestone 5 — response fingerprints

Implement contexts, episodic support/query sampling, and cross-fitted calibration.

**Gate:** leakage tests and permutation-invariance tests pass.

### Milestone 6 — experiments and analysis

Extend the runner, outputs, paired metrics, and plots.

**Gate:** synthetic smoke experiments complete, resume deterministically, and consolidate all required ablations.

### Milestone 7 — repository-data development run

Run a reduced holdout-free configuration:

```bash
PY=/nfs/student/a/aalasandagutti/hpcResearch/python_virtual_envs/pytorch-gpu/bin/python
DIR=notebooks/utils/delta_response_filteredInhibs_biDirectional

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$DIR/pipeline.py" \
  --skip-holdout \
  --cv-seeds 0 \
  --final-seeds 0 \
  --inhibitor-blocks 2 \
  --max-epochs 2 \
  --patience 1 \
  --batches-per-epoch 1 \
  --output-dir "$DIR/outputs/development"
```

Then run the full App-Inhibitor selection and fitting budget.

Evaluate historical App-App data as an exploratory comparison only after recipes are frozen. Preserve the existing disclosure that this dataset is not an untouched holdout.

---

## 15. Definition of done

A fresh agent should consider the implementation complete when:

- The new directory runs independently.
- Profiles and labels come from consistent source data.
- Panel selection is reproducible and fully recorded.
- All six model variants are available.
- Both bandwidth proxies retain signed changes.
- Reverse inhibitor measurements influence model learning.
- Fingerprints contain only permitted support observations.
- Joint models can calibrate from all-floor support.
- Direct and calibrated methods are reported separately.
- Crossed validation excludes every label from held-out runs.
- App-App outcomes remain outside training and selection.
- Checkpoints fully specify inference.
- Tests and synthetic integration runs pass.
- Experiment manifests include the new inputs and recipes.
- README documents commands, schemas, target semantics, support regimes, and result interpretation.

**The central deliverable is a controlled comparison:** determine whether reverse-direction evidence improves real-application aggressor representations, while separating that effect from the benefits of absolute supervision and floor-aware calibration.
