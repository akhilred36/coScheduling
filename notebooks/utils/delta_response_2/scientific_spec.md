# Delta Response 2 Scientific Specification

## Scope

This implementation follows Path A: no new App-App measurements are available.
All ten applications and their valid App-Inhibitor responses may contribute to
`random_split` fitting. Historical App-App outcomes are diagnostic-only and are
not opened, hashed, calibrated against, selected on, or reported by this
pipeline. A newly collected untouched App-App holdout remains necessary.

## Target And Folds

The observed log target is `max(0, latent_log_slowdown)`. Values above zero are
exact; zero values are one-sided evidence that the latent value is at most zero.
Each outer fold holds out one victim and one complete inhibitor block. The held
victim's retained responses are support anchors, its held-block responses are
queries, and model parameters use neither the victim nor the block. Scalers omit
the same victim and block. Legacy OOD thresholds are calibrated in nested views
of parameter-training rows only; outer support/query censoring labels cannot
influence them.

Two static block schemes are supported:

- K-means blocks in transformed physical profile space.
- Complete `msg_size` mechanism-family blocks, with no family crossing folds.

The application-profile-nearest and density-ratio views use all application
profiles without labels. They are explicitly transductive covariate weighting,
not evidence that App-Inhibitor and App-App conditional responses match.

## All-Anchor Calibration

For gauge-centered potential values `r_i`, victim correction `c` minimizes:

```text
sum_exact Huber(r_i + c - y_i)
+ sum_floor Huber(max(0, r_i + c))
+ shrinkage / 2 * (c - population_prior)^2
```

Floor values remain inequalities. No latent negative targets are invented.
Gauge-centering over the support inhibitors makes correction shrinkage invariant
to arbitrary additive shifts in the learned potential. The population prior is
estimated only from parameter-training victims in each fold.

Positive shrinkage is evaluated both toward zero and toward the fold-local
population correction. Query shrinkage scales the centered support and query
potential consistently before fitting the correction.

## Candidates

The common OOF comparison includes constant and shrunk intercepts, hierarchical
censored victim intercepts, nearest and median anchors, censored linear absolute
regression, a small bounded tree, compact absolute neural models, legacy
positive-only delta aggregation, all-anchor censored delta correction, and a
query-potential shrinkage ablation. Neural capacities are `8/4`, `16/8`, and
`32/16`; every fit uses the same fixed optimizer-step and example budget.

An App-App rank-score calibrator is deliberately absent. Path A provides no
valid labels from which to identify its App-App intercept or slope.

## Selection And Reporting

Every candidate emits predictions for every query and seed. The deployment unit
is the mean latent prediction across configured seeds followed by clipping.
Candidates first pass finite-boundary, optimizer-convergence, seed-collapse,
estimand-aligned seed-instability, and positive-rank gates. Seed instability is
computed from each seed's worst equal-fold score across the same schemes and
unweighted/profile-weighted views used by primary selection. The pipeline then forms a Pareto set
over equal-weight victim/block log MAE, raw MAE, raw MSE, and mean absolute fold
bias in unweighted and app-profile-weighted views, plus unweighted rank as a
separate scientific axis. At least 80% of folds must have defined rank and mean
defined-fold Spearman must be positive in every scheme. It selects the
least complex Pareto candidate within one equal-fold standard error of the
minimum worst-view log MAE. A failure of the rank gate is reported as a negative
scientific result rather than a satisfactory constant solution.

All reported scores are App-Inhibitor development CV, not untouched estimates.
Magnitude, signed bias, calibration slope/intercept, rank correlation, floor and
positive strata, per-victim metrics, seed mean/SD/worst case, ensembles, and
paired victim-cluster bootstrap differences are retained separately and labeled
descriptive because crossed folds share training observations.
