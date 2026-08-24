# App-Inhibitor Audit Reproduction

The inherited App-Inhibitor target and selected-model summaries were recomputed
from `job_inh.csv` and the saved `capacity_limit/crossed_validation_predictions.csv`.
No App-App input, prediction, or metric artifact was opened.

## Target Distribution

| Statistic | Recomputed value |
| --- | ---: |
| Valid rows | 2,623 |
| Floor rows | 389 (14.8303%) |
| Mean log slowdown | 0.416437 |
| Sample standard deviation | 0.454988 |
| Median | 0.263043 |
| P90 | 1.078523 |
| Maximum | 3.208242 |
| Mean above the floor | 0.488950 |

These values reproduce the handoff table.

## Selected Legacy Recipe

The filter was low-rank configuration 0 with OOD kernel temperature 4,
quantile 0.90, and uniform fallback.

| Estimand | Recomputed value |
| --- | ---: |
| Grouped seed-averaged log MAE | 0.124808 |
| Pooled-seed query-weighted log MAE | 0.127104 |
| Latent-ensemble query-weighted log MAE | 0.120149 |
| Latent-ensemble mean prediction | 0.406795 |
| Pooled-seed Pearson | 0.812853 |
| Pooled-seed Spearman | 0.873519 |
| Latent-ensemble Pearson | 0.820560 |
| Latent-ensemble Spearman | 0.885771 |

The handoff's reported correlations are pooled across seed predictions, while
its reported mean prediction is the latent-space ensemble. The reproduction
records both estimands explicitly to avoid mixing them in Delta Response 2.

Detailed generated tables are under the ignored directory
`audit_outputs/app_inhibitor_reproduction_v2/`.
