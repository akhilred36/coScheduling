# Transitive Delta-Response Model for MPI Slowdown Prediction

## 1. Objective

Predict the communication slowdown of a victim application `A` when it is
co-scheduled with another application `B`, using:

- The isolated communication profiles of `A` and `B`.
- The isolated profiles of a collection of synthetic inhibitors.
- Measured slowdowns of `A` when co-scheduled with those inhibitors.

The historical App-App measurements are contaminated by earlier smoke runs and
are unavailable for design, tuning, debugging, calibration, or performance
assessment. Development and smoke execution must use `--skip-holdout` or
synthetic pairs. A newly collected untouched App-App dataset is required for a
future confirmatory transfer evaluation.

The central inference problem is:

> Given the known slowdown of `A` with inhibitor `J`, predict how that
> slowdown changes when `J` is replaced by a target entity `K`.

During training, `K` is another inhibitor with a known response. During final
evaluation, `K` is a real application `B`.

### 1.1 Implementation status

This document is sufficient for an agent to build a functional research
pipeline, but it is not intended to prescribe a bit-for-bit reproducible
experiment for every optional ablation. Choices outside the recommended
primary configuration in Section 17, such as the exact inhibitor clustering
algorithm or an expanded hyperparameter grid, must be recorded in the run
configuration and output report rather than selected silently.

One previously ambiguous decision is fixed explicitly in this specification:

> Train one response-potential model with single-anchor delta loss. Apply
> uniform, median, kernel, and OOD-gated aggregation only at inference, using
> the same trained checkpoint for every aggregation comparison.

Do not train a separate potential model for each aggregation method in the
primary experiment. This separation ensures that comparisons measure the
effect of aggregation rather than differences between fitted models.

## 2. Available Data

### 2.1 Isolated application profiles

`jobs.csv` contains one row per application:

```text
job_id, mpi_time, comm_frac, total_msgs, total_bytes
```

### 2.2 Isolated inhibitor profiles

`inhibitors.csv` contains one row per inhibitor configuration:

```text
inhib_id, msg_size, wait_time, comm_sparsity,
mpi_time, comm_frac, total_msgs, total_bytes
```

The inhibitor control variables `msg_size`, `wait_time`, and
`comm_sparsity` are not valid model inputs because corresponding values do
not exist for real applications.

### 2.3 App-Inhibitor responses

`job_inh.csv` contains directional observations in which the application is
the victim and the inhibitor is the aggressor:

```text
job_id, inhib_id, slowdown
```

### 2.4 App-App responses

`pair.csv` contains one row per unordered application pair and two directional
outcomes:

```text
jobA_id, jobB_id, slowdown_A, slowdown_B
```

Each row produces two evaluation records:

- `A` as victim and `B` as aggressor, with target `slowdown_A`.
- `B` as victim and `A` as aggressor, with target `slowdown_B`.

The two directions are not independent data augmentation. They are correlated
outcomes from the same co-scheduling experiment and should remain grouped
when constructing confidence intervals.

## 3. Observed Dataset Properties

The current data has:

- 10 applications.
- 236 inhibitor profiles.
- 2,339 raw App-Inhibitor response rows.
- 55 App-App rows, covering all unordered pairs including 10 self-pairs.
- 110 directional App-App outcomes after directional expansion.

Twelve App-Inhibitor rows reference inhibitor IDs that do not have an isolated
profile in `inhibitors.csv`. These rows cannot be used by a profile-based
model and should be quarantined with a warning, leaving 2,327 usable response
rows. Each application retains between 229 and 235 valid anchors, and 214
inhibitors are observed for every application.

Missing App-Inhibitor combinations should not be imputed and a complete
App-Inhibitor matrix should not be assumed. Training episodes are formed only
from observed, valid edges.

Several application profiles are far from the inhibitor profile distribution,
particularly in message and byte counts. App-App prediction is therefore an
out-of-distribution extrapolation problem, not merely interpolation among
inhibitors. Evaluation and uncertainty reporting must make this limitation
visible.

## 4. Data Validation

Before training:

1. Drop unnamed CSV index columns.
2. Assert that `job_id` and `inhib_id` are unique in their profile tables.
3. Assert that profile values are finite and non-negative.
4. Assert that `comm_frac` lies in `[0, 1]`.
5. Assert that every slowdown is finite and strictly positive before taking
   its logarithm.
6. Inner-join response rows with both profile tables.
7. Save or report rejected rows rather than silently discarding them.
8. Detect duplicate `(job_id, inhib_id)` response keys. If genuine replicate
   runs are later added, preserve them as replicates instead of averaging them
   during loading.

## 5. Feature Representation

### 5.1 Primary features

Use only the four measurements available for both applications and
inhibitors. For an entity `E`, define:

```text
x(E) = [
    log1p(mpi_time),
    comm_frac,
    log1p(total_msgs),
    log1p(total_bytes),
]
```

Standardize each transformed column. In each validation fold, fit the scaler
using only profiles allowed by that fold, then reuse those statistics for all
support and query entities in the fold. Never fit a scaler from `pair.csv` or
its targets.

The primary model should use these four features because there are only ten
distinct victim profiles. A large engineered feature vector would increase
the opportunity for overfitting.

### 5.2 Preregistered feature ablation

The following physically motivated features can be tested as one predefined
ablation:

```text
estimated_runtime = mpi_time / max(comm_frac, epsilon)
mean_message_size = total_bytes / max(total_msgs, epsilon)
message_rate      = total_msgs / estimated_runtime
byte_rate         = total_bytes / estimated_runtime
```

Apply `log1p` to these derived non-negative quantities. Select between the
base and augmented feature sets using only the App-Inhibitor validation
protocol in Section 10.

## 6. Target Representation And Censoring

The operational target is clipped observed slowdown:

```text
observed_slowdown = max(1, latent_slowdown)
observed_log_slowdown = max(0, latent_log_slowdown)
```

Let the available target be

```text
y(A, I) = log(observed_slowdown(A, I)).
```

Values above zero are exact latent log slowdowns. A value of zero states only
that the latent value is at most zero. Its speedup magnitude is not recoverable
from the available CSVs. Models therefore produce a latent log score and apply
the observation map before metrics or saved operational predictions.

## 7. Transitive Response Potential

### 7.1 Structural requirement

Do not learn an unconstrained function of `[A, J, K]` to predict a delta.
Instead, learn one scalar response potential `R(A, X)` and define every delta
as a difference of potentials:

```text
predicted_delta(A, J, K) = R(A, K) - R(A, J).
```

This construction guarantees:

- Identity: `predicted_delta(A, J, J) = 0`.
- Antisymmetry: swapping `J` and `K` negates the delta.
- Path consistency: deltas through an intermediate entity cancel exactly.

These properties are architectural and do not have to be approximated from
data.

### 7.2 Recommended low-capacity potential

Use a small shared profile encoder `E` and a low-rank victim-aggressor
interaction:

```text
R(A, X) = h(E(X)) + dot(u(E(A)), v(E(X))).
```

Here:

- `h(E(X))` models the aggressor's global pressure.
- `u(E(A))` models the victim's susceptibility.
- `v(E(X))` models aggressor pressure dimensions.
- The interaction rank should be small, initially selected from `{1, 2, 4}`.

No application-ID embedding should be used in `R`. An ID embedding could
memorize the ten training applications and would not define behavior for a
new application profile.

A suitable initial architecture is:

```text
Shared encoder E: 4 -> 8 -> 4, with ReLU
Global head h:     4 -> 8 -> 1
Victim head u:     4 -> rank
Aggressor head v:  4 -> rank
```

A generic response MLP over `[z_A, z_X, z_A * z_X]` may be evaluated as an
ablation, but it should not be the default because the training set contains
only ten unique victim inputs.

## 8. Anchor-Based Prediction

Suppose the slowdown of `A` against an anchor inhibitor `J` is known. The
single-anchor prediction against target `K` is:

```text
y_hat(A, K | J)
    = y(A, J) + R(A, K) - R(A, J).
```

At inference, all valid inhibitors measured with victim `A` are available, but
only uncensored responses are exact latent anchors. Let that uncensored set be
`S_A`. Aggregate their predictions in latent log space:

```text
    latent_y_hat(A, K)
    = sum over J in S_A of w_J(K) * y_hat(A, K | J).
```

For uncensored anchors this can be rearranged into a residual-calibration form:

```text
y_hat(A, K)
    = R(A, K)
    + sum over J in S_A of w_J(K) * [y(A, J) - R(A, J)].
```

The potential predicts the relative response surface. The measured anchors
then estimate and correct the model's residual bias for victim `A`.

Only uncensored anchors can supply exact latent residuals. Floor anchors still
contribute one-sided training constraints but are excluded from residual
calibration. The observed raw prediction is:

```text
observed_log_hat(A, K) = max(0, latent_y_hat(A, K))
slowdown_hat(A, K) = exp(observed_log_hat(A, K)) >= 1.
```

## 9. Single-Anchor Training Objective

The primary model is trained only on single-anchor delta episodes. Anchor
aggregation is not part of the optimization objective and introduces no
aggregation-specific model parameters.

For each training episode:

1. Sample victim application `A` uniformly, so applications with slightly
   more observations do not dominate.
2. Sample two different inhibitors `J` and `K` that are both observed with
   `A`.
3. Treat `J` as the known support anchor and `K` as the query.
4. Compute the observed and predicted changes in log slowdown.
5. Apply a censored pairwise Huber loss.

```text
target_delta = y(A, K) - y(A, J)
pred_delta   = R(A, K) - R(A, J)
if y_J > 0 and y_K > 0: Huber(pred_delta - (y_K - y_J))
if y_J > 0 and y_K = 0: one-sided Huber enforcing pred_delta <= -y_J
if y_J = 0 and y_K > 0: one-sided Huber enforcing pred_delta >= y_K
if y_J = 0 and y_K = 0: no identified pairwise delta constraint
```

The potential difference remains latent. Identity, antisymmetry, and path
consistency are exact before the nonlinear observation boundary. They do not
apply to observed clipped differences after that mapping.

For absolute-response regression, exact values use Huber loss and floor values
use a one-sided Huber constraint requiring the latent prediction to be at most
zero. Each comparator receives independent crossed-validation epoch selection.

Equivalently, an uncensored single-anchor query prediction is:

```text
y_hat(A, K | J) = y(A, J) + pred_delta.
```

This is the chosen primary training option. Train one potential model per
cross-validation fold and one final potential model. At validation and final
inference, apply every aggregation method in Section 10 to that same model.
Do not refit the model when changing from uniform to median, kernel, or
OOD-gated aggregation.

Multi-anchor episodic training with support sizes such as `4`, `16`, `64`, or
all available anchors may be investigated later as a clearly labeled
secondary experiment. It must not replace the single-anchor objective in the
primary comparison because doing so would confound potential learning with a
particular aggregation rule.

Sampling inhibitor pairs does not create additional independent
measurements. The effective data remains the 2,327 observed response rows,
and model capacity and reported confidence intervals should reflect that
fact.

## 10. Anchor Aggregation

Evaluate the following predefined aggregation rules using the same trained
response-potential checkpoint and the same anchor set. Aggregation is a
post-training inference operation. No aggregation method receives additional
training or access to App-App labels.

### 10.1 Uniform mean

```text
w_J = 1 / |S_A|.
```

This estimates one global residual correction for victim `A` and is the most
stable default when the target is outside inhibitor support.

### 10.2 Robust median

```text
y_hat(A, K) = median over J of y_hat(A, K | J).
```

This protects against noisy or anomalous anchor measurements.

### 10.3 Similarity-weighted mean

For normalized physical profiles, define:

```text
w_J(K) = softmax_J(-distance(x(J), x(K))^2 / temperature).
```

Tune `temperature` using only held-out App-Inhibitor queries. Use physical
feature distance as the primary metric. A response embedding trained only
through score differences is not guaranteed to have an identifiable or
meaningful Euclidean geometry.

A learned diagonal distance metric can be tested only as a regularized
ablation. Its positive feature weights should be constrained and shrunk
toward equal weighting.

### 10.4 Out-of-distribution safeguard

A softmax always returns weights summing to one, even if every anchor is far
from the target. This can create false confidence and concentrate weight on
an arbitrarily distant nearest inhibitor.

Fit one separate base-feature physical-profile scaler to inhibitor profiles
only. Use this frozen coordinate system for every fold's validation distance
and for future target-to-anchor distances; never pool distances from fold-local
model scalers. Measure the target's nearest uncensored-anchor distance relative
to distances observed in the crossed folds. If the target is outside the
validated support region,
blend the local kernel correction toward the uniform or median correction:

```text
residual_correction
    = alpha(K) * local_kernel_residual
    + (1 - alpha(K)) * global_robust_residual,
```

where `alpha(K)` decreases as the target moves outside inhibitor support.
Tune the distance threshold and blending rule without using App-App labels.

## 11. Leakage-Safe Model Selection

Random row splits are not sufficient. They place the same victims and nearly
identical inhibitor profiles on both sides of the split and can substantially
overstate generalization.

Use the following validation hierarchy without opening any App-App data.

### 11.1 Leave-one-application-out validation

For each application `A`:

1. Train model parameters using responses from the other nine applications.
2. Divide `A`'s inhibitor observations into support anchors and held-out
   queries.
3. Predict each query using only `A`'s support anchors.

This tests transfer to a victim profile not used to fit the response
potential while preserving the intended assumption that some responses for
the victim are known at inference.

### 11.2 Held-out-inhibitor-block validation

Partition inhibitors into profile-space clusters or predefined configuration
blocks. Hold out an entire inhibitor block across all victims:

1. Train the potential without responses involving held-out inhibitors.
2. Use retained inhibitors as anchors.
3. Predict responses for the held-out inhibitor profiles.

This tests generalization to unseen aggressor regions and is more relevant
than a random inhibitor split.

### 11.3 Crossed validation

For the strongest test, simultaneously hold out one victim and one inhibitor
block. Train on the remaining victim-block combinations, use retained
responses of the held-out victim as anchors, and predict its responses to the
held-out inhibitor block.

Use these folds to select independently for low-rank delta, generic potential,
and absolute-response regression:

- Base versus augmented feature representation.
- Interaction rank and hidden dimensions.
- Weight decay and learning rate.
- Kernel temperature.
- OOD threshold and fallback rule.
- Training support-size distribution.

Use fixed validation queries within each fold and evaluate several random
initialization seeds. Do not resample the validation set every epoch.

After selecting each formulation, train final models on all valid
App-Inhibitor responses. Each model's predetermined epoch count is the median
of its own selected crossed-fold best epochs. Budgets and seed treatment are
the same across model kinds.

For aggregation, first average query errors within seed and then average seeds
within each victim/block group. Within each family, and then across uniform,
median, kernel, and OOD-kernel, identify the numerical best and admit candidates
whose mean is no more than one grouped standard error above it. Select the
eligible candidate in deterministic complexity order: uniform, median, kernel,
OOD-kernel, followed by sorted parameter order. This is the prespecified
grouped one-standard-error rule; exact-minimum differences do not decide the
primary method.

## 12. Optional Future App-App Evaluation

The current historical App-App data must remain unused. For a newly collected
untouched dataset, expand each row into two directional records. For each record
with victim `A` and aggressor `B`:

1. Look up `A` and `B` in `jobs.csv`.
2. Gather all valid `(A, J, slowdown)` anchors from `job_inh.csv`.
3. Compute `R(A, B)` and `R(A, J)` for each anchor.
4. Form every per-anchor delta prediction.
5. Apply the aggregation rule selected without App-App labels.
6. Exponentiate the aggregated log prediction.

Explicitly perform the reverse prediction `B -> A`; do not infer it by
symmetry because victim and aggressor roles are different.

### 12.1 Endpoint-familiarity tiers

App-App evaluation supports three disjoint endpoint-familiarity tiers. Pair
eligibility is assigned on canonical pair rows before directional expansion:

- `random_split`: all applications may contribute App-Inhibitor fitting rows;
  evaluate all App-App rows. The compatibility name does not imply that this
  pipeline trains on an App-App split.
- `one_known`: fit only on a configured application subset; evaluate pairs
  with exactly one fitted and one unfitted endpoint.
- `zero_shot`: use the same restricted fit; evaluate pairs with two unfitted
  endpoints.

For restricted fits, unfitted application profiles are excluded from model
scaler fitting and their App-Inhibitor responses are excluded from parameter
and model-selection fitting. Their measured App-Inhibitor responses remain
available as inference anchors because the delta-response task assumes victim
anchor measurements at prediction time. App-App outcomes remain sealed until
all fitting and selection are complete in every tier.

### 12.2 Self-pairs

For a self-pair, victim and aggressor profiles are identical. The two measured
outcomes in one self-pair row may nevertheless differ because they are two
runtime measurements or process placements. A deterministic profile model
must produce the same prediction for both.

Report both:

- Metrics retaining both self-pair outcomes as repeated measurements.
- Metrics after averaging the two self-pair outcomes.

This distinguishes model error from irreducible variation that cannot be
represented by the available profile features.

## 13. Baselines and Ablations

The final comparison should include:

1. Constant prediction `slowdown = 1`.
2. Per-victim median App-Inhibitor slowdown.
3. Nearest-anchor slowdown with no learned delta.
4. Absolute response regression trained on App-Inhibitor rows.
5. Low-rank delta potential with one anchor.
6. Low-rank delta potential with uniform aggregation.
7. Low-rank delta potential with median aggregation.
8. Low-rank delta potential with kernel aggregation.
9. Low-rank delta potential with OOD-gated kernel aggregation.
10. Generic neural response potential as a capacity ablation.

The future scientific comparison is whether anchor-calibrated relative
inference improves over an absolute App-Inhibitor response model on a newly
collected untouched App-App holdout. The historical pair data cannot answer
that confirmatory question.

## 14. Metrics

Report:

- MAE and RMSE in log-slowdown space.
- MAE and RMSE in raw slowdown space.
- Median absolute log error.
- Median multiplicative error, such as `exp(abs(log_pred - log_true))`.
- Spearman rank correlation.
- Metrics per victim application and their macro average.
- Complete directional prediction table.

For a constant or otherwise degenerate group, record finite `spearman = 0` and
`spearman_defined = false`; the placeholder is not an estimated correlation.

The prediction table should contain at least:

```text
victim_id
aggressor_id
true_slowdown
predicted_slowdown
absolute_error
absolute_log_error
number_of_anchors
uncensored_anchor_count
nearest_anchor_distance
effective_anchor_count
weighted_residual_std
kernel_component_effective_anchor_count
kernel_component_weighted_residual_std
unweighted_residual_std
fallback_residual_mad
ood_score
```

Directional outcomes from one co-scheduling row remain one cluster. Generate a
cluster draw once per bootstrap replicate, reuse it for every method, retain
both directions, and report both marginal intervals and paired
method-minus-comparison intervals. Also report variation across model seeds.
Reject duplicate unordered pair keys, including reversed duplicates.

## 15. Uncertainty and Diagnostics

For each App-App prediction, record:

- Unweighted residual spread, explicitly labeled as victim-level and
  uncalibrated rather than target-specific uncertainty.
- Target-dependent weighted residual dispersion for linear aggregation rules.
- Effective anchor count `1 / sum(w_J^2)` for linear aggregation rules.
- For nonlinear median fallback, undefined aggregate effective count and
  weighted dispersion, plus kernel-component count/dispersion and fallback MAD.
- Distance to the nearest inhibitor anchor.
- Percentile of that distance relative to validation queries.
- Variation across independently trained model seeds.

None of these spreads is a prediction interval. If intervals
are required, calibrate residual quantiles using held-out queries from the
crossed App-Inhibitor validation folds, for example with cross-conformal
calibration. Report empirical coverage on App-Inhibitor folds before applying
the intervals to App-App predictions.

## 16. Interpretation and Limitations

This formulation can learn how an application's slowdown changes as the
aggressor communication profile changes. It cannot identify interference
mechanisms absent from the isolated profiles, such as network topology,
placement, collective operation type, synchronization structure, or temporal
burst alignment.

The key untestable transfer assumption is:

> The response surface learned by varying synthetic inhibitor profiles remains
> valid when a real application with the same measured profile acts as the
> aggressor.

The App-App holdout is the only direct test of this assumption. Strong
performance on held-out inhibitors is necessary but not sufficient. OOD
distance and uncertainty diagnostics should therefore accompany every final
prediction.

Clipping discards latent speedup magnitude. The censored loss can enforce the
known side of the boundary but cannot reconstruct that missing magnitude.

## 17. Recommended Primary Configuration

The initial experiment should use:

- Four transformed profile features.
- Shared `4 -> 8 -> 4` encoder.
- Rank-2 low-rank response potential.
- Censored latent log-slowdown formulation with an observed floor at zero.
- Single-anchor censored delta Huber loss.
- Uniform sampling across victim applications.
- One shared checkpoint for uniform, median, physical-distance kernel, and
  OOD-gated kernel aggregation at inference.
- Crossed leave-one-application-out and held-out-inhibitor-block validation.
- Independent model-specific crossed-validation epochs with equal seed budgets.
- Grouped one-standard-error aggregation selection.
- One common inhibitor-only physical distance scaler for kernel/OOD geometry.
- Five model seeds.
- Holdout-free execution by default for development; future evaluation only on
  a newly collected untouched App-App dataset.

This is deliberately smaller than the existing 32-dimensional neural
response model. With only ten distinct victim applications, improved
validation and controlled inductive bias are more valuable than additional
network capacity.
