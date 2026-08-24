# Delta Response 2: Research Handoff And Redesign Instructions

## 1. Mission

You are taking over an MPI co-scheduling slowdown-prediction experiment. Your
job is to iterate on, or completely replace, the `delta_response_1`
methodology. The previous work performed a substantial leakage-safe model
search and exposed a serious synthetic-inhibitor-to-real-application transfer
failure. Do not respond to that failure by running another blind neural-network
capacity sweep. Start by understanding the evidence in this document and the
saved artifacts, then redesign the methodology around the actual failure mode.

The desired evaluation condition is `random_split` in the terminology of this
repository:

- All ten applications may contribute profiles and App-Inhibitor responses to
  model fitting.
- App-App labels are not model-fitting inputs.
- All eligible App-App pairs are evaluated after fitting.
- Despite its historical name, `random_split` is not currently a random split
  of App-App rows.

Focus only on model methodology, architecture, training, calibration,
selection, and evaluation for this all-application condition. Do not run the
`one_known`, `zero_shot`, or training-application-count studies unless the user
later asks for them.

## 2. Non-Negotiable Scientific Constraint

The repository's historical `pair.csv` was opened by earlier work and is not a
pristine holdout. It was also opened once after the model search described
below. It may be used for retrospective diagnosis, and it may be explicitly
redesignated as development data, but it must not be used to tune a method and
then be reported as an untouched test set.

Use one of these two valid paths:

### Path A: No new App-App data are available

- Continue selecting models using App-Inhibitor data only.
- Use the historical App-App results only as diagnostic evidence or clearly
  labeled exploratory development results.
- Do not claim that App-Inhibitor validation proves App-App transfer.
- Do not repeatedly inspect historical App-App performance while presenting it
  as a fixed test.
- Finish with a frozen recipe and state that a newly collected App-App holdout
  is still required.

### Path B: New App-App measurements are available

- Create immutable App-App development and test datasets before modeling.
- Split on unordered pair clusters, never on directional rows.
- Prefer an outer held-out-application design when sample size permits.
- Use only the development set for domain calibration, architecture choices,
  early stopping decisions, and thresholds.
- Hash and seal the final test set until every choice is frozen.
- Report whether applications, pair clusters, or both are novel in each split.

If the user's intent about the status of a new pair dataset is unclear, ask one
short question before opening it.

## 3. Environment And Starting Files

Use the Conda environment requested by the user:

```bash
conda run -n hpcResearch python --version
```

The previous experiments ran successfully with PyTorch 2.4.1 and CUDA on an
NVIDIA RTX A1000 Laptop GPU.

Assume `delta_response_1` is available as a sibling directory or that its
useful files have been copied into the new directory. Read these first:

- `README.md`: implemented protocol, CLI, outputs, and contamination warning.
- `schematic.md`: scientific specification and assumptions.
- `data.py`: profile transforms, censoring, data validation, and pair tiers.
- `model.py`: low-rank, generic-potential, and absolute-response models.
- `training.py`: samplers, censored losses, optimization, and early stopping.
- `inference.py`: anchor selection, aggregation, OOD gating, and diagnostics.
- `pipeline.py`: crossed validation, selection, final training, and evaluation.
- `metrics.py`: metrics and paired cluster bootstrap.
- `run_experiments.py`: immutable planning and staged search machinery.
- `tests/`: invariants and leakage regression tests.

The most useful completed experiment artifacts are under:

```text
experiments/random_split_model_search_v1/
```

Key artifacts:

- `EXPERIMENT_SUMMARY.md`: concise summary of the completed search.
- `consolidated/architecture_search.csv`: initial architecture and optimizer
  trajectory.
- `refinement_config.json`: first capacity refinement.
- `final_capacity_config.json`: second capacity refinement.
- `capacity_limit_config.json`: capacity stopping check.
- `capacity_limit/selection.json`: decisive App-Inhibitor selection.
- `capacity_limit/crossed_validation_predictions.csv`: selected-run validation
  predictions for detailed analysis.
- `capacity_limit/validation_distance_reference.csv`: inhibitor-space OOD
  reference distances.
- `final_random_split_config.json`: frozen final configuration.
- `final_random_split/run_report.json`: final provenance.
- `final_random_split/directional_predictions.csv`: complete predictions.
- `final_random_split/seed_predictions.csv`: per-seed predictions.
- `final_random_split/metrics_non_self.csv`: primary non-self metrics.
- `final_random_split/metrics_non_self_per_victim.csv`: per-victim results.
- `final_random_split/cluster_bootstrap_non_self.csv`: marginal intervals.
- `final_random_split/paired_method_differences_non_self.csv`: paired
  method-comparison intervals.

The original four-task experiment manifest still shows its planned evaluation
task as pending because the final refined model was evaluated through a
separate fixed-global invocation. The completed result is
`final_random_split/`; do not mistake the stale plan row for a missing result.

## 4. Problem And Current Methodology

The goal is to predict the directional communication slowdown of victim
application `A` when co-scheduled with aggressor application `B`.

Available fitting data consist of:

- Isolated application profiles.
- Isolated synthetic inhibitor profiles.
- Measured slowdown of each application when paired with inhibitors.
- App-App outcomes that are reserved for transfer evaluation or explicitly
  designated development calibration.

The base entity profile is:

```text
[
  log1p(mpi_time),
  comm_frac,
  log1p(total_msgs),
  log1p(total_bytes)
]
```

The augmented representation adds log-transformed estimated runtime, mean
message size, message rate, and byte rate.

Observed slowdown is clipped at one:

```text
observed_slowdown = max(1, latent_slowdown)
observed_log = max(0, latent_log)
```

The primary model learns a response potential:

```text
R(A, X) = h(E(X)) + dot(u(E(A)), v(E(X)))
```

It predicts a replacement delta through:

```text
delta(A, J, K) = R(A, K) - R(A, J)
```

This gives exact latent identity, antisymmetry, and path consistency. For an
uncensored App-Inhibitor anchor `J`, final prediction is:

```text
latent_y_hat(A, K)
  = R(A, K) + aggregate_J[y(A, J) - R(A, J)]
```

The existing implementation discards all floor anchors during residual
calibration. See `inference.py`, especially `rank_inference_anchors()` and
`predict_with_anchors()`. Floor observations still affect model training
through one-sided losses, but they do not constrain the victim-level residual
correction at inference.

Current crossed validation simultaneously holds out one victim and one
profile-space inhibitor block. Responses of the held-out victim outside the
block are support anchors, and responses inside the block are validation
queries. This is leakage-safe for App-Inhibitor interpolation and
extrapolation. It does not directly validate transfer from synthetic
inhibitors to real application aggressors.

## 5. Data Used In The Completed Search

The actual current dataset differs from some older counts in `schematic.md`.
The completed search used:

| Quantity | Value |
| --- | ---: |
| Applications | 10 |
| Inhibitor profiles | 266 |
| Raw App-Inhibitor rows | 2,653 |
| Valid App-Inhibitor rows | 2,623 |
| Rejected App-Inhibitor rows | 30 |
| Duplicate response keys | 0 |
| App-Inhibitor floor rows | 389, or 14.83% |
| Historical App-App pair rows | 55 |
| Historical directional outcomes | 110 |
| Non-self pair clusters | 45 |
| Non-self directional outcomes | 90 |

Every model-search candidate used the same static design:

- 10 held-out victim applications.
- 4 deterministic inhibitor blocks.
- 5 seeds: 0, 1, 2, 3, and 4.
- 40 victim/block groups and 200 seed-specific folds per candidate.
- K-means block seed 1701.
- At most 80 epochs with patience 10.
- 4 sampled batches per epoch.
- Batch size 128.

The initial search plus capacity refinements performed 8,000 crossed model
fits. Input files were hashed by the experiment planner.

## 6. Search Results

### 6.1 Initial feature and rank search

All candidates in this table used hidden dimension 8, embedding dimension 4,
learning rate 0.001, and weight decay 0.0001.

| Feature set | Rank | Mean crossed App-Inhibitor log MAE |
| --- | ---: | ---: |
| Base | 1 | 0.177316 |
| Base | 2 | 0.167432 |
| Base | 4 | 0.171942 |
| Augmented | 1 | 0.181876 |
| Augmented | 2 | **0.165909** |
| Augmented | 4 | 0.178306 |

The augmented rank-2 model won. Rank 4 was worse for both feature sets.

### 6.2 Optimizer search

The architecture winner above was used for this stage.

| Learning rate | Weight decay | Mean crossed App-Inhibitor log MAE |
| ---: | ---: | ---: |
| 0.0003 | 0 | 0.201345 |
| 0.0003 | 0.0001 | 0.201342 |
| 0.0003 | 0.001 | 0.201351 |
| 0.001 | 0 | 0.165870 |
| 0.001 | 0.0001 | 0.165909 |
| 0.001 | 0.001 | 0.165885 |
| 0.003 | 0 | 0.151268 |
| 0.003 | 0.0001 | 0.151230 |
| 0.003 | 0.001 | **0.150413** |

Learning rate 0.003 and weight decay 0.001 won. The very small learning rate
was clearly underfit under the fixed optimization budget.

### 6.3 Capacity search

The search was extended because each initial capacity winner was on the upper
boundary.

| Hidden / embedding | Grouped uniform CV log MAE |
| ---: | ---: |
| 4 / 2 | 0.186840 |
| 8 / 4 | 0.150413 |
| 16 / 8 | 0.148452 |
| 24 / 8 | 0.144615 |
| 24 / 12 | 0.145253 |
| 32 / 16 | 0.141067 |
| 48 / 24 | 0.139480 |
| 64 / 32 | **0.130895** |
| 128 / 64 | 0.140450 |
| 256 / 128 | 0.134534 |

The stopping check showed that 128 and 256 did not improve the architecture
selection score. Under uniform aggregation, 64/32 also minimized log MSE,
log RMSE, raw MAE, and MAPE. The 256/128 model had a very small raw MSE/RMSE
advantage but worsened the other criteria.

Do not overstate the precision of the capacity winner. If each capacity is
allowed its own best OOD aggregation, 64/32 and 256/128 are effectively tied:

| Capacity | Best grouped OOD CV log MAE | Query-weighted OOD CV log MAE |
| --- | ---: | ---: |
| 64 / 32 | 0.124808 | **0.127104** |
| 128 / 64 | 0.126717 | 0.130465 |
| 256 / 128 | **0.124276** | 0.127311 |

Across the 40 victim/block groups, 64/32 beat 256/128 in only 18 groups. The
mean grouped difference was 0.000532 with descriptive standard error 0.002233.
The official 64/32 choice is reasonable because it is much smaller and wins
the preregistered sequential selection, not because it is decisively more
accurate than 256/128.

### 6.4 Aggregation search

For the 64/32 model:

| Aggregation | Grouped CV log MAE |
| --- | ---: |
| Uniform | 0.130895 |
| Median | 0.133551 |
| Kernel, temperature 1 | 0.133044 |
| OOD kernel, temperature 4, q=0.90, uniform fallback | **0.124808** |

The minimum-error recipe used the OOD-gated kernel. The general pipeline's
one-standard-error rule would have selected uniform aggregation, but the
accuracy-focused runner deliberately froze the numerical minimum.

### 6.5 Selected leakage-safe recipe

The final App-Inhibitor-selected recipe was:

- Model family: low-rank transitive response potential.
- Features: augmented, for 8 model inputs.
- Encoder: `8 -> 64 -> 32`, ReLU after each layer.
- Global aggressor head: `32 -> 64 -> 1`, with ReLU.
- Victim head: `32 -> 2`.
- Aggressor interaction head: `32 -> 2`.
- Rank: 2.
- Optimizer: AdamW.
- Learning rate: 0.003.
- Weight decay: 0.001.
- Training objective: censored single-anchor delta Huber loss.
- Final epoch count: 5, selected as the median best fold epoch.
- Aggregation: OOD-gated physical-profile kernel.
- Kernel temperature: 4.0.
- OOD threshold: 0.90 validation-distance quantile.
- OOD fallback: uniform residual mean.
- Inference anchors: all uncensored anchors.
- Final ensemble: 5 independently seeded fits combined in latent log space.

The selected model's grouped App-Inhibitor validation metrics were:

| Metric | Value |
| --- | ---: |
| Log MAE | 0.124808 |
| Log MSE | 0.076014 |
| Log RMSE | 0.275706 |
| Raw MAE | 0.306859 |
| Raw MAPE | 11.371893% |
| Raw MSE | 2.215318 |
| Raw RMSE | 1.488394 |

For model-family context, the best generic-potential aggregation had
App-Inhibitor log MAE 0.147019, and the fixed absolute-response comparator had
0.212049. App-Inhibitor validation therefore strongly favored the low-rank
delta model.

## 7. Historical App-App Results

The final frozen recipe was evaluated once on the historical pair file. The
primary scope excludes self-pairs and contains 90 directional outcomes from 45
unordered pair clusters.

| Method | Log MAE | Log RMSE | Raw MAE | Raw MAPE | Raw MSE | Raw RMSE | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute response | **0.097189** | **0.157460** | **0.118043** | 8.9715% | **0.046443** | **0.215507** | -0.097050 |
| Constant slowdown 1 | 0.103012 | 0.184117 | 0.123284 | **8.8528%** | 0.058954 | 0.242805 | undefined |
| Nearest anchor | 0.193877 | 0.255436 | 0.267977 | 23.1501% | 0.140855 | 0.375307 | **0.533532** |
| Delta median | 0.211887 | 0.270320 | 0.296802 | 25.3445% | 0.166720 | 0.408313 | 0.435446 |
| Delta single anchor | 0.229221 | 0.284172 | 0.318576 | 27.5902% | 0.176204 | 0.419767 | 0.517253 |
| Generic potential | 0.237422 | 0.308795 | 0.334167 | 29.3834% | 0.208032 | 0.456105 | 0.384307 |
| Delta uniform | 0.239282 | 0.306042 | 0.339900 | 29.3693% | 0.213871 | 0.462462 | 0.382538 |
| Delta OOD kernel | 0.251164 | 0.313502 | 0.354551 | 30.8582% | 0.220267 | 0.469326 | 0.418745 |
| Delta kernel | 0.261504 | 0.326410 | 0.371701 | 32.5249% | 0.242439 | 0.492381 | 0.459899 |
| Victim inhibitor median | 0.380145 | 0.498082 | 0.606647 | 54.6644% | 0.724206 | 0.851003 | 0.320456 |

This is a model-ranking reversal:

- The low-rank delta model was best on App-Inhibitor CV and much worse on
  App-App outcomes.
- Absolute response was worst among the three fitted model families on
  App-Inhibitor CV but had the best App-App error point estimates.
- Constant slowdown 1 was almost as good as absolute response and had the best
  MAPE.
- Nearest anchor retained the best rank correlation among nonconstant methods,
  despite poor magnitude error.
- The selected delta OOD method did not win for any of the ten victims.

Cluster-bootstrap log-MAE intervals were:

| Method | Estimate | 95% cluster-bootstrap interval |
| --- | ---: | ---: |
| Absolute response | 0.097189 | [0.077051, 0.120449] |
| Constant 1 | 0.103012 | [0.072709, 0.135255] |
| Nearest anchor | 0.193877 | [0.160905, 0.230861] |
| Delta median | 0.211887 | [0.177954, 0.248307] |
| Delta OOD kernel | 0.251164 | [0.209078, 0.294224] |

Important paired differences in log MAE, expressed as delta OOD minus the
comparison method, were:

| Comparison | Difference | 95% paired cluster-bootstrap interval |
| --- | ---: | ---: |
| Absolute response | +0.153975 | [0.106098, 0.203071] |
| Constant 1 | +0.148152 | [0.099654, 0.202789] |
| Nearest anchor | +0.057286 | [0.027388, 0.085797] |
| Delta median | +0.039277 | [0.023366, 0.054973] |
| Delta uniform | +0.011882 | [0.004146, 0.019810] |
| Delta kernel | -0.010340 | [-0.023440, 0.002236] |

Constant 1 minus absolute response was +0.005823 with interval
[-0.012676, 0.023639]. Absolute response therefore did not demonstrate a clear
log-MAE improvement over the constant baseline, even though its MSE and RMSE
were better.

## 8. The Central Failure: Target Scale Does Not Transfer

The strongest established result is a severe response-distribution shift.

### 8.1 App-Inhibitor validation targets

For the 2,623 unique App-Inhibitor validation outcomes:

| Statistic | True log slowdown |
| --- | ---: |
| Mean | 0.416437 |
| Standard deviation | 0.454988 |
| 25th percentile | 0.038146 |
| Median | 0.263043 |
| 75th percentile | 0.656094 |
| 90th percentile | 1.078523 |
| Maximum | 3.208242 |
| Floor fraction | 14.83% |

The 2,234 uncensored targets had mean log slowdown 0.488950.

The selected model's five-seed-equivalent App-Inhibitor predictions had mean
0.406795. Its Pearson correlation with targets was 0.812853, and Spearman was
0.873519. On its validation domain, the model learned both magnitude and
ordering reasonably well.

### 8.2 App-App targets

For the 90 non-self App-App outcomes:

| Statistic | True log slowdown |
| --- | ---: |
| Mean | 0.103012 |
| Standard deviation | 0.153458 |
| 25th percentile | 0.007041 |
| Median | 0.049783 |
| 75th percentile | 0.122840 |
| 90th percentile | 0.271009 |
| Maximum | 0.792714 |
| Floor fraction | 20.00% |
| Mean raw slowdown | 1.123284 |
| Maximum raw slowdown | 2.209384 |

The selected delta model predicted mean log slowdown 0.340791. Its average
signed log error was +0.237779, and it overpredicted 78 of 90 directions, or
86.67%.

On the 18 floor outcomes, the selected model's mean prediction and MAE were
both 0.324615. On the 72 positive outcomes, its mean prediction was 0.344836
while the mean target was only 0.128765.

This is not primarily a ranking failure. The selected delta model retained
Pearson correlation 0.674147 and Spearman 0.418745 on non-self App-App outcomes.
It often knew which interactions were relatively larger, but predicted the
wrong absolute scale.

A purely diagnostic post-hoc regression on the already observed App-App labels
gave approximately:

```text
true_log = -0.02528 + 0.37644 * delta_predicted_log
```

Clipping this diagnostic fit at zero would reduce log MAE to approximately
0.077528 on the same data. This number is not a valid performance estimate and
must not be reused as a fixed calibration result. It is evidence that a simple,
properly validated domain-calibration layer could be valuable.

## 9. Aggregation And OOD Findings

The OOD safeguard did what it was designed to do locally, but it did not solve
the domain-transfer problem.

The inhibitor validation nearest-anchor distance distribution was:

| Statistic | Distance |
| --- | ---: |
| Minimum | 0.180524 |
| 25th percentile | 0.565546 |
| Median | 1.232053 |
| 75th percentile | 1.526027 |
| 90th percentile | 1.723532 |
| 95th percentile | 1.815925 |
| 99th percentile | 2.481930 |
| Maximum | 3.518762 |

For non-self App-App queries:

- Mean nearest-anchor distance was 1.181337.
- Median was 0.674223.
- 20/90 queries exceeded the validation P90 threshold.
- 19/90 exceeded P95.
- 11/90 exceeded P99.
- The maximum was 3.858612.

Most App-App aggressor profiles therefore appeared in-support under the current
four-feature physical distance, even though their response scale did not
transfer.

Error was not concentrated in the nominally OOD tail:

| OOD stratum | n | Delta OOD log MAE | Uniform log MAE | Median log MAE | Constant log MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| At or below validation P90 | 70 | 0.275292 | 0.262099 | 0.234593 | 0.116065 |
| Above validation P90 | 20 | 0.166715 | 0.159420 | 0.132416 | 0.057328 |

OOD gating improved over the ungated kernel in both strata, but remained worse
than uniform, median, and constant prediction. Nearest-profile distance had
Spearman correlation -0.279658 with absolute log error. The sign is opposite
the intended risk interpretation because the farther App-App targets happened
to have lower true slowdowns and lower predictions.

Residual-dispersion diagnostics were more informative:

| Diagnostic | Spearman correlation with absolute log error |
| --- | ---: |
| Unweighted residual standard deviation | 0.562186 |
| Kernel-weighted residual standard deviation | 0.493468 |
| Fallback residual MAD | 0.388713 |
| Effective kernel anchor count | 0.346372 |
| Model-seed log standard deviation | 0.029691 |

Do not interpret these as calibrated uncertainties. They may be useful inputs
to a learned or validated shrinkage gate.

Aggregation ranking itself did not transfer:

| Method | Query-weighted App-Inhibitor CV log MAE | App-App log MAE |
| --- | ---: | ---: |
| Delta OOD | **0.127104** | 0.251164 |
| Delta kernel | 0.133178 | 0.261504 |
| Delta uniform | 0.134228 | 0.239282 |
| Delta median | 0.136039 | **0.211887** |

The four-method CV-to-App-App rank correlation was -0.8. There are only four
points, so this is descriptive rather than inferential, but it shows that
minimizing held-out-inhibitor error did not choose the best App-App aggregation.

## 10. Victim-Level And Seed-Level Findings

### 10.1 Victim-level transfer

The selected delta OOD model's per-victim App-Inhibitor CV and non-self App-App
log MAE were:

| Victim | App-Inhibitor CV | App-App | Increase |
| --- | ---: | ---: | ---: |
| amg | 0.168298 | 0.333369 | 0.165070 |
| beatnik | 0.097967 | 0.223533 | 0.125565 |
| fiesta | 0.052493 | 0.113082 | 0.060589 |
| kripke | 0.065824 | 0.091889 | 0.026065 |
| laghos | 0.093384 | 0.419472 | 0.326088 |
| lammps | 0.371046 | 0.530787 | 0.159741 |
| minife | 0.161776 | 0.351491 | 0.189716 |
| minivite | 0.120737 | 0.221300 | 0.100564 |
| quicksilver | 0.030111 | 0.066870 | 0.036759 |
| tricount | 0.108730 | 0.159844 | 0.051114 |

Victim-level CV and App-App errors had Spearman correlation 0.757576 and
Pearson correlation 0.827374. CV identifies relatively difficult victims, but
its absolute error level is optimistic and it does not validate the domain
transfer assumption.

Absolute response had the lowest per-victim log MAE for 6/10 victims, constant
1 won 3/10, and victim inhibitor median won 1/10. Delta OOD won 0/10.

### 10.2 Seed instability

Non-self log MAE by final model seed was:

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed SD | Ensemble |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute response | 0.103012 | 0.180424 | 0.223527 | 0.103012 | 0.234316 | 0.063401 | **0.097189** |
| Delta median | 0.304173 | 0.207861 | 0.146717 | 0.235489 | 0.225023 | 0.056555 | 0.211887 |
| Delta uniform | 0.337240 | 0.237483 | 0.175260 | 0.250681 | 0.249683 | 0.057780 | 0.239282 |
| Delta OOD | 0.330424 | 0.247727 | 0.199875 | 0.262379 | 0.255241 | 0.046770 | 0.251164 |
| Delta kernel | 0.330433 | 0.260740 | 0.225778 | 0.279848 | 0.253847 | 0.038895 | 0.261504 |
| Generic potential | 0.270225 | 0.235743 | 0.198997 | 0.224040 | 0.266370 | 0.029828 | 0.237422 |

The selected delta model's CV seed MAEs were tightly grouped from 0.123617 to
0.130020, a range of only 0.006403. Its App-App seed range was 0.130549.
Initialization differences are therefore amplified by transfer even when
App-Inhibitor validation looks stable.

The final low-rank fit used only 5 epochs x 4 batches x 128 examples, or 2,560
sampled episodes. Absolute response used 4 epochs, or 2,048 sampled examples.
Two absolute-response seeds clipped to the same predictions as the constant
baseline, while the remaining seeds were much worse. Latent-space ensembling
produced a better error than any individual absolute seed. This apparent
success is unstable and mostly reflects conservative scale, not useful rank
prediction: ensemble Spearman was -0.097050.

## 11. What Is Established And What Is Still A Hypothesis

Keep factual findings separate from causal interpretations.

### 11.1 Established findings

- App-Inhibitor and App-App target scales differ dramatically.
- The delta model overpredicts App-App slowdown in 86.67% of non-self cases.
- The model retains useful rank information but has poor magnitude calibration.
- Positive-only anchor methods inherit an inhibitor-scale bias.
- Floor anchors are excluded from residual calibration by implementation.
- Current four-feature OOD distance does not identify the largest transfer
  errors.
- App-Inhibitor model-family and aggregation rankings do not transfer to
  historical App-App rankings.
- Larger capacity improved the App-Inhibitor objective up to 64/32, but did not
  repair transfer.
- Final predictions are substantially more seed-sensitive than CV suggested.
- Absolute response's low App-App error is statistically close to constant 1
  in MAE and has no useful rank correlation.
- Historical App-App results cannot support a fresh untouched-holdout claim.

### 11.2 Ranked failure hypotheses

#### Hypothesis 1: Synthetic-to-real conditional response shift

Confidence: high.

Synthetic inhibitors and real applications with similar isolated aggregate
profiles may not exert equivalent network pressure. Missing variables include
collective-operation mix, synchronization, burst timing, process placement,
network topology, concurrency, and temporal overlap. This explains why most
App-App profiles can look geometrically in-support while their target response
scale is much lower.

#### Hypothesis 2: Positive-anchor selection bias

Confidence: medium to high.

Residual calibration uses only anchors with observed slowdown above one. The
389 floor responses are censored evidence that latent response was at most
zero, but they are dropped from the correction estimate. Exact-anchor residuals
therefore come from a positively selected response subset and can bias the
victim correction upward. The large overprediction on App-App floor outcomes
is consistent with this mechanism, but no counterfactual all-anchor censored
calibration run has yet been performed.

#### Hypothesis 3: Objective and scale mismatch

Confidence: high.

The delta objective is optimized over a wide synthetic-inhibitor response
range. It learns relative ordering well on that domain. Final App-App outcomes
occupy a much narrower range near the observation floor. Optimizing inhibitor
delta accuracy does not identify the domain-specific scale compression needed
for application aggressors.

#### Hypothesis 4: OOD geometry is behaviorally incomplete

Confidence: high.

Euclidean distance over four aggregate communication features detects profile
extremes, not mechanism shift. The current OOD gate blends only the residual
correction. It does not constrain the extrapolated target potential `R(A, B)`.
Even a perfect residual fallback cannot fix a target-potential scale error.

#### Hypothesis 5: Reuse of the same validation groups causes selection optimism

Confidence: medium.

Feature set, rank, optimizer, capacity, epochs, aggregation, OOD threshold, and
temperature were all selected using the same 40 victim/block groups. The
64-versus-256 difference is below grouped noise. A nested outer selection
estimate would probably be less optimistic.

#### Hypothesis 6: Training budget and seed sensitivity are inadequate

Confidence: medium.

The final models receive very few optimizer steps. Some absolute seeds collapse
to floor prediction. More optimization, convergence diagnostics, and
seed-stability criteria may help. However, the narrow CV seed range and strong
systematic App-App overprediction show that optimization alone cannot explain
the transfer failure.

#### Hypothesis 7: Model capacity exploits inhibitor-specific structure

Confidence: medium to low.

There are only ten distinct victim profiles. A 64/32 network may learn
inhibitor-manifold details that do not transfer. Smaller networks were worse on
the existing CV objective, so this hypothesis cannot be confirmed without a
domain-relevant outer criterion.

#### Hypothesis 8: Augmented ratio features amplify extrapolation

Confidence: medium to low.

Estimated runtime, message rate, byte rate, and mean message size are ratios.
They can magnify profile differences and may create unstable extrapolation for
application aggressors. Augmented features improved inhibitor CV, but no
domain-calibrated test established that they improve transfer.

## 12. Required Redesign Strategy

Use the following sequence. Do not skip directly to large neural architectures.

### Stage 1: Reproduce and audit

1. Run the full unit-test suite before editing:

```bash
conda run -n hpcResearch python -m unittest discover -s tests -v
```

2. Recompute the key target-distribution, signed-error, OOD-stratum, and
   seed-stability tables from saved predictions.
3. Confirm the observation boundary and potential invariants still hold.
4. Save hashes of all input CSVs and a static fold manifest.
5. Decide and record whether historical App-App outcomes are diagnostic-only
   or explicitly development data.

### Stage 2: Establish stronger baselines

The next methodology must not compare only against neural delta variants.
Implement and tune these baselines under the same static folds and seeds:

1. Constant slowdown 1.
2. Global mean or median log slowdown with strong shrinkage toward zero.
3. Victim-specific censored intercept with hierarchical shrinkage.
4. Regularized linear or generalized additive absolute-response model.
5. Small gradient-boosted-tree absolute model, with capacity heavily limited.
6. Existing absolute neural response model with a real optimizer and capacity
   search rather than a fixed default.
7. Nearest-anchor and robust median-anchor baselines.
8. Rank-only or pairwise ordering model followed by a separate calibrator.

Report ranking and magnitude separately. A model that only matches the mean
but has negative rank correlation is not a satisfactory scientific solution.

### Stage 3: Fix censored anchor calibration

Replace positive-only residual averaging with a victim-level censored
calibration problem.

For exact anchor `i` with response `y_i > 0`, correction `c_A` should satisfy:

```text
c_A approximately y_i - R(A, I_i)
```

For a floor anchor with `y_i = 0`, latent response is known only to obey:

```text
R(A, I_i) + c_A <= 0
```

Estimate `c_A` using a convex censored Huber objective over all anchors, plus a
regularizer toward zero or a population victim correction. Compare:

- Exact-only mean, the current behavior.
- Exact-only median.
- Censored intercept without shrinkage.
- Censored intercept with fixed shrinkage grid.
- Hierarchical censored intercept sharing information across victims.

Do not invent latent negative values for floor observations. Preserve them as
inequality constraints.

Add tests covering:

- All-exact anchors.
- Mixed exact and censored anchors.
- Mostly censored victims such as `fiesta`.
- Monotonic behavior as additional floor anchors are added.
- Finite predictions and the observed lower bound.

### Stage 4: Build a domain-relevant validation objective

The current held-out-inhibitor average gives equal scientific importance to
inhibitor regions that may be irrelevant to application aggressors. Without
using App-App labels, application profiles can still define a transductive
target covariate distribution.

Implement at least these App-Inhibitor validation views:

1. Existing unweighted crossed victim/block metric.
2. App-profile-nearest inhibitor subset metric.
3. Kernel importance weighting from inhibitor-query profiles toward the
   empirical application-profile distribution.
4. Mechanism-block holdout if inhibitor control metadata permit it.
5. Floor-stratified and target-magnitude-stratified metrics.

Declare profile-based weighting as transductive because all application
profiles influence selection. Do not use App-App outcomes for the weights.

Check whether model rankings are stable across these views. If they are not,
prefer a robust or Pareto selection rather than the minimum of one average.

### Stage 5: Separate ranking from calibration

The current model contains useful ordering information and bad scale. Treat
these as separate tasks.

Candidate formulation:

```text
rank_score(A, B) = delta or potential model output
calibrated_log(A, B) = max(0, intercept + slope * rank_score(A, B))
```

Use a constrained low-capacity calibrator:

- Nonnegative slope.
- Strong shrinkage of slope toward zero or one, selected on development folds.
- Optional global intercept.
- At most a victim-level random intercept unless much more data exist.
- No application-ID embeddings in a model intended for unseen applications.

If App-App development data exist, fit the calibrator in nested pair-cluster or
held-out-application folds. Never report calibration performance on the same
rows used to estimate slope and intercept.

If no App-App development labels exist, do not pretend the App-App scale is
identified. You may investigate unlabeled domain adaptation or inhibitor
reweighting, but the final report must state that transfer calibration remains
unvalidated.

### Stage 6: Constrain extrapolated target potential

The current OOD gate changes only anchor residual aggregation. Add methods that
also shrink the query potential when application aggressor behavior is
uncertain.

Candidates:

- Shrink `R(A, B)` toward an absolute low-slowdown prior as domain distance or
  residual dispersion rises.
- Blend delta and absolute predictions with a gate selected on development
  folds.
- Use conformalized or quantile-based abstention when diagnostics indicate
  unreliable transfer.
- Learn a bounded monotone pressure score and victim susceptibility rather
  than an unconstrained MLP potential.
- Normalize aggressor pressure to inhibitor quantiles before applying victim
  susceptibility.

Any gate must be validated. The existing distance is not sufficient evidence
for a gate because error does not increase with that distance.

### Stage 7: Revisit architecture only after the objective is fixed

Once calibration and validation are credible, compare compact architectures:

1. Linear low-rank potential.
2. Existing nonlinear low-rank potential with 8/4, 16/8, and 32/16 capacity.
3. Monotone or sign-constrained pressure/susceptibility model.
4. Separate victim and aggressor encoders with explicit regularization.
5. Residual model over a simple physical baseline.
6. Generic MLP only as a capacity ablation.

Do not repeat the 64/32-to-256/128 sweep unless a new outer criterion shows
that capacity matters. With ten victim profiles, large networks need strong
justification.

### Stage 8: Improve optimization and stability

Decouple final epoch count from a tiny number of optimizer steps.

- Track optimizer steps, not only epochs.
- Compare fixed example budgets across batch sizes.
- Save train and validation curves.
- Require a minimum training budget before patience can stop.
- Evaluate at least five seeds for finalists.
- Report mean single-seed performance, worst seed, ensemble performance, and
  per-prediction seed spread.
- Penalize or reject methods whose apparent gain depends on collapsed seeds.
- Consider averaging checkpoints or predictions only after validating the
  ensemble rule.

## 13. Data Collection Recommendations

Model changes cannot recover mechanisms absent from the features. If new
measurements can be collected, prioritize these additions.

### 13.1 App-App development and test data

- Collect repeated runs for each unordered pair.
- Preserve both directional outcomes under one pair-cluster ID.
- Record process placement, node allocation, and topology.
- Use multiple placements when possible.
- Reserve a truly untouched test collection performed after methodology freeze.

### 13.2 Better application and inhibitor features

- Message-size distribution, not only mean size.
- Point-to-point versus collective operation fractions.
- Collective type and participant count.
- Communication burst duration and duty cycle.
- Synchronization and idle-time measures.
- Temporal overlap or concurrency statistics.
- Rank count, node count, and ranks per node.
- Network path, topology, and placement descriptors.
- Read/write directionality and injection-rate estimates.

### 13.3 Better synthetic inhibitor coverage

- Design inhibitor families that mimic application burstiness and collectives.
- Match application-profile regions deliberately.
- Include low-pressure inhibitors near the App-App target regime.
- Hold out complete inhibitor mechanism families during validation.
- Record inhibitor control metadata for validation grouping even if controls
  cannot be direct model features.

## 14. Static Dataset And Split Requirements

Create an immutable experiment root before fitting:

```text
experiment_root/
  experiment_spec.json
  environment.json
  input_hashes.json
  fold_manifest.csv
  pair_split_manifest.csv
  configs/
  tasks/
  consolidated/
```

The fold manifest must include:

- Victim ID.
- Inhibitor block or mechanism family.
- Query inhibitor IDs.
- Support inhibitor IDs.
- Seed.
- Feature/scaler fitting membership.
- Censoring stratum.
- App-profile importance weight, if used.

The pair manifest must include:

- Canonical unordered pair ID.
- Both endpoint IDs.
- Self-pair flag.
- Development/test membership.
- Endpoint novelty flags.
- A rule preventing two directional outcomes from crossing splits.

Hash inputs, code, configuration, and dependency selections. Never silently
reuse results after a code or data change.

## 15. Metric And Selection Policy

Do not claim to minimize every metric with one scalar unless the scalar is
defined before evaluation. MAE and MSE can prefer different methods. RMSE has
the same ranking as MSE and should not be counted as an independent vote.

Recommended primary metric:

```text
non-self directional log MAE
```

Required secondary metrics:

- Log MSE and RMSE.
- Raw MAE, MAPE, MSE, and RMSE.
- Median absolute log error.
- Median multiplicative error.
- Spearman correlation.
- Signed calibration bias.
- Calibration intercept and slope.
- Floor-stratum and positive-stratum error.
- Per-victim metrics and macro-victim means.
- Seed mean, standard deviation, and worst-seed result.

Use paired unordered-pair cluster bootstrap for method comparisons. Preserve
both directions in each bootstrap draw.

Recommended model-selection hierarchy:

1. Reject methods that violate finite-range or boundary invariants.
2. Reject methods with unacceptable seed collapse or instability.
3. Identify the Pareto set over log MAE, raw MAE, raw MSE, and calibration bias.
4. Use prespecified complexity preference among statistically indistinguishable
   candidates.
5. Keep MAPE visible because most targets are near one, but do not let it alone
   select a constant model with no ranking ability.

## 16. Acceptance Criteria For Delta Response 2

Do not declare success only because App-Inhibitor CV improves.

Minimum engineering criteria:

- All existing tests pass.
- New censoring, calibration, split, and leakage tests pass.
- Predictions are finite and at least one.
- Pair directions remain clustered.
- Potential identity, antisymmetry, and path consistency remain exact if a
  potential-difference model is retained.
- Static input and fold hashes are saved.

Minimum scientific criteria when App-App development data exist:

- Out-of-fold development predictions beat constant 1 and regularized absolute
  response on primary log MAE.
- Paired confidence intervals are reported, not only point estimates.
- Calibration slope and intercept improve without destroying rank correlation.
- Floor and positive strata both improve or the tradeoff is explicit.
- Gains are not driven by one victim or one seed.
- The complete methodology is frozen before opening a new final test.

If these criteria are not met, report the negative result. A robust conclusion
that available features cannot identify transfer is more useful than another
overfit architecture.

## 17. Recommended First Experiment

The highest-value first implementation is not a new deep model. Build a compact
comparison with the following three changes:

1. Replace positive-only residual averaging with censored, shrinkage-regularized
   victim correction using all App-Inhibitor anchors.
2. Add an app-profile-weighted crossed-validation view alongside the current
   unweighted view.
3. Add a simple calibrated rank model, but fit its scale only on properly
   separated App-App development folds if such data are available.

Compare these methods:

```text
constant_1
regularized_absolute_linear
regularized_absolute_neural
current_delta_median
current_delta_ood
censored_correction_delta
censored_correction_delta_with_query_shrinkage
calibrated_delta_rank_model  # only with App-App development labels
```

Use compact capacities first: 8/4, 16/8, and 32/16. Use five seeds and a fixed
optimizer-step budget. Select only after producing out-of-fold predictions for
every candidate.

## 18. Deliverables Expected From The Next Agent

Produce all of the following:

1. A concise revised scientific specification.
2. Immutable data and split manifests with hashes.
3. A script that runs only the `random_split` methodology and its model search.
4. Complete candidate configurations and selection provenance.
5. Out-of-fold App-Inhibitor predictions for every candidate.
6. Out-of-fold App-App development predictions if development labels exist.
7. MAE, MAPE, MSE, RMSE, calibration, rank, per-victim, and seed-stability
   tables.
8. Paired pair-cluster bootstrap comparisons.
9. Ablations isolating censored correction, domain weighting, query shrinkage,
   and calibration.
10. Tests for every new leakage boundary and mathematical invariant.
11. A final report that clearly separates established findings, hypotheses,
    exploratory development results, and untouched evaluation results.

## 19. Final Guidance

The previous model did not fail because its App-Inhibitor validation error was
insufficiently optimized. It failed because the validation task and final
transfer task were not behaviorally equivalent. The next iteration should
therefore prioritize domain calibration, all-anchor censored inference,
domain-relevant validation, stable low-capacity baselines, and better data.

Preserve the valuable structural properties of the potential model only if
they help under a transfer-relevant criterion. Be willing to replace the delta
method if a simpler regularized model is more accurate, stable, and honest
about what the available features can support.
