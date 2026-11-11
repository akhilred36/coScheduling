# tab:ns_cv_vs_appapp with N=100 App-App values (all 4 pipelines)

**Sources**
- CV values: artifact_data_20261017.json, unchanged. Runs: baseSnap (1W-MPI), totalTime (1W-TT), bidir (BD-MPI), bidirTT (BD-TT).
- App-App values: seed_methods.csv (N=100 mean-latent ensemble, blind, native non-self outcomes).
- Delta aggregation: from each run's selection.json (`primary_method`).

**Aggregation per pipeline**
- 1W-MPI: uniform.
- 1W-TT: ood_kernel.
- BD-MPI: uniform.
- BD-TT: uniform.

**Spearman rank correlation, CV vs N=100 App-App**

| pipeline | rows | N=100 | artifact (5 seeds) |
|---|---|---|---|
| 1W-MPI | 3 | +0.50 | +0.50 |
| 1W-TT | 3 | −1.00 | −0.50 |
| BD-MPI | 8 | +0.81 | +0.36 |
| BD-TT | 8 | +0.62 | +0.60 |

**Per-row values**

| pipeline | row | aggregation / role | CV log MAE | App-App log MAE (N=100) |
|---|---|---|---|---|
| 1W-MPI | Absolute | absolute | 0.1562 | 0.1030 |
| 1W-MPI | Delta (tuned) | delta_uniform | 0.1580 | 0.1471 |
| 1W-MPI | Generic | generic | 0.1618 | 0.1469 |
| 1W-TT | Absolute | absolute | 0.0318 | 0.0156 |
| 1W-TT | Delta (tuned) | delta_ood_kernel | 0.0206 | 0.0167 |
| 1W-TT | Generic | generic | 0.0204 | 0.0205 |
| BD-MPI | Absolute | absolute | 0.1562 | 0.1030 |
| BD-MPI | Delta | delta_uniform | 0.1690 | 0.1460 |
| BD-MPI | Generic | generic | 0.1618 | 0.1469 |
| BD-MPI | Joint cal | joint_cal | 0.1481 | 0.0921 |
| BD-MPI | Bprof cal | bprof_cal | 0.1479 | 0.0925 |
| BD-MPI | Bctx cal | bctx_cal | 0.1481 | 0.0923 |
| BD-MPI | Wall-aux cal | bwall_cal | 0.1513 | 0.0925 |
| BD-MPI | Fwd-only cal | bfwd_cal | 0.1443 | 0.0895 |
| BD-TT | Absolute | absolute | 0.0318 | 0.0156 |
| BD-TT | Delta | delta_uniform | 0.0338 | 0.0199 |
| BD-TT | Generic | generic | 0.0204 | 0.0205 |
| BD-TT | Joint cal | joint_cal | 0.0203 | 0.0103 |
| BD-TT | Bprof cal | bprof_cal | 0.0216 | 0.0102 |
| BD-TT | Bctx cal | bctx_cal | 0.0200 | 0.0137 |
| BD-TT | Wall-aux cal | bwall_cal | 0.0184 | 0.0101 |
| BD-TT | Fwd-only cal | bfwd_cal | 0.0223 | 0.0153 |
