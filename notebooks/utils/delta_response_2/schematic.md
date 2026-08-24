# Delta Response 2 Model Schematic

## Purpose

Delta Response 2 predicts the directional slowdown of victim application `A`
when an aggressor profile `X` is present. Model development uses application
profiles, synthetic-inhibitor profiles, and App-Inhibitor slowdown responses.
App-App outcomes are not fitting inputs.

The package contains a compact candidate family rather than one unconditional
neural model. The completed App-Inhibitor search froze one candidate for
deployment. This document describes both the shared architecture and the
candidate that was actually selected.

## Data Flow

```text
jobs.csv -------------------- application profiles -----------+
                                                               |
inhibitors.csv ------------- inhibitor profiles --------------+--> feature transform
                                                               |    and sealed scaler
job_inh.csv ---------------- App-Inhibitor responses -----------+
                                                                    |
                                                                    v
                                                          crossed development CV
                                                     (profile and mechanism folds)
                                                                    |
                                                                    v
                                                           candidate selection
                                                                    |
                                                                    v
                                                          five final checkpoints
                                                                    |
            victim application profile + aggressor profile --------+
                                                                    |
            victim's measured inhibitor anchors -------------------+--> latent prediction
                                                                         |
                                                                         v
                                                              max(0, latent log)
                                                                         |
                                                                         v
                                                                  exp(log slowdown)
```

The historical App-App table is deliberately absent from this flow. It may be
used only by the separate retrospective evaluation described in
`experiments.md`, after the recipe is frozen.

## Profile Representation

The completed search uses the four-dimensional `base` profile:

```text
x(E) = [
  log1p(mpi_time),
  comm_frac,
  log1p(total_msgs),
  log1p(total_bytes)
]
```

A `ProfileScaler` standardizes these features. During crossed validation it is
fitted only with parameter-training application profiles and retained
inhibitor profiles. The held-out victim and held-out inhibitor block do not
contribute scaler statistics. Final fitting uses all ten application profiles
and all inhibitor profiles.

Physical nearest-anchor and OOD distances use a separate base-feature scaler
fitted to inhibitor profiles. They do not use a learned embedding.

## Low-Rank Response Potential

The shared encoder is:

```text
profile x
   |
   v
Linear(input_dim, hidden_dim)
   |
 ReLU
   |
Linear(hidden_dim, embedding_dim)
   |
 ReLU
   |
embedding E(x)
```

The response potential is:

```text
R(A, X) = h(E(X)) + <u(E(A)), v(E(X))>
```

where:

```text
h: embedding_dim -> hidden_dim -> ReLU -> 1
u: embedding_dim -> rank
v: embedding_dim -> rank
rank = 2
```

For victim `A`, anchor inhibitor `J`, and query aggressor `K`, the model emits:

```text
delta(A, J, K) = R(A, K) - R(A, J)
```

This construction guarantees the latent invariants:

```text
delta(A, J, J) = 0
delta(A, J, K) = -delta(A, K, J)
delta(A, J, M) + delta(A, M, K) = delta(A, J, K)
```

The candidate capacities are `8/4`, `16/8`, and `32/16`, where the two numbers
are `hidden_dim/embedding_dim`.

## Censored Training

Observed slowdown is clipped at one:

```text
observed_slowdown = max(1, latent_slowdown)
observed_log = max(0, latent_log)
```

An observation at log slowdown zero is therefore a censoring inequality, not a
known latent zero.

Potential training samples one victim and two distinct inhibitors. For anchor
log response `y_J`, query response `y_K`, and predicted delta `d`:

```text
y_J > 0, y_K > 0: Huber(d - (y_K - y_J))
y_J > 0, y_K = 0: one-sided Huber enforcing d <= -y_J
y_J = 0, y_K > 0: one-sided Huber enforcing d >= y_K
y_J = 0, y_K = 0: no identified delta constraint
```

The sampler excludes floor-floor episodes, so every sampled example is
informative. Every neural fit uses exactly 80 AdamW steps, batch size 128,
learning rate `0.003`, and weight decay `0.001`. Query labels may be monitored
in development curves but never select a checkpoint or stop training.

## Inference Corrections

### Legacy Exact-Anchor Correction

For exact anchors only:

```text
residual_j = observed_log(A, J) - R(A, J)
prediction = R(A, K) + aggregate_j(residual_j)
```

Implemented aggregations are uniform, median, and OOD-gated physical kernel.
Kernel weights are:

```text
w_j proportional to exp(-distance_j^2 / temperature)
```

The OOD correction blends the kernel correction toward a fixed fallback:

```text
alpha = 1                                             if d_min <= threshold
alpha = exp(-(d_min - threshold) / width)            otherwise
correction = alpha * kernel + (1 - alpha) * fallback
```

### All-Anchor Censored Correction

The redesigned correction uses exact and floor anchors. It first gauge-centers
the support potentials, then solves the convex one-dimensional problem:

```text
minimize over c:
  sum_exact Huber(p_j + c - y_j)
  + sum_floor Huber(max(0, p_j + c))
  + 0.5 * lambda * (c - prior)^2
```

Thus a floor anchor constrains `p_j + c <= 0` instead of inventing a latent
target. Candidates use no prior, a zero prior, or a population correction
estimated from parameter-training victims. Query-potential shrinkage candidates
also apply a fixed `gamma` of `0.5` or `0.75` to centered potential variation.

## Comparator Families

The common crossed folds also evaluate:

- Constant slowdown one.
- Shrunk global observed median.
- Global and victim-level censored intercepts.
- Nearest and robust-median anchor baselines.
- Censored linear absolute regression.
- A small gradient-boosted absolute tree.
- A compact absolute-response neural model.
- Legacy exact-anchor delta models.
- All-anchor censored delta models.
- All-anchor query-shrinkage delta models.

## Validation And Selection

The completed configuration has:

```text
10 victims
4 held-out inhibitor blocks
2 fold schemes: profile and complete mechanism family
5 seeds: 0, 1, 2, 3, 4
53 candidates
400 fold-seed tasks
```

Each fold separates parameter training, victim support anchors, and victim
queries. Magnitude is evaluated in unweighted and label-free
application-profile-weighted views. Candidate selection applies finite-output,
numerical-fit, seed-collapse, estimand-aligned seed-instability, and positive
macro-rank gates. It then uses Pareto filtering and a grouped one-standard-error
preference for the least complex eligible candidate.

## Frozen App-Inhibitor Winner

The sealed run at `experiments/compact_v1` selected:

```text
method:          current_delta_ood_h8_e4
family:          legacy_delta
hidden_dim:      8
embedding_dim:   4
rank:            2
seeds:           0, 1, 2, 3, 4
aggregation:     OOD-gated physical kernel
temperature:     4.0
OOD quantile:    0.90
fallback:        uniform exact-anchor residual mean
```

Its final OOD state is:

```text
threshold:       1.7235320806503296
width:           1.2320533990859985
reference count: 2623
```

The seed-specific latent predictions are averaged first. The ensemble is then
mapped to the observation boundary:

```text
ensemble_latent = mean(seed_latent_predictions)
predicted_log_slowdown = max(0, ensemble_latent)
predicted_slowdown = exp(predicted_log_slowdown)
```

The all-anchor censored redesign was competitive but was not the frozen winner.
The best all-anchor candidate had worst-view App-Inhibitor log MAE `0.163562`;
the one-standard-error rule selected the smaller legacy candidate at `0.166523`.
This distinction must remain explicit when interpreting the static App-App
evaluation.

## Evaluation Boundary

`predict-profile` accepts two sealed profile IDs and returns predictions without
an App-App outcome input. Static App-App evaluation must remain a separate,
read-only consumer of the frozen recipe:

```text
sealed Delta Response 2 recipe --> predictions
historical pair table ----------> outcomes and pair-cluster membership
predictions + outcomes ----------> retrospective metrics only
```

No result on the historical static pair table is a new untouched-holdout
estimate. See `experiments.md` for the exact comparable evaluation protocol.
