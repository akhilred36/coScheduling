# Batch 2 numbers (prediction results)
Source: artifact DATA (output/artifact_data_20261017.json) unless noted. Δ% = (method − baseline) log MAE / baseline log MAE; CI = 95% paired pair-cluster bootstrap (1,000 draws, seed 923, shared draw matrix).
Truly harmful non-self pairs (worse endpoint ≥ 1.05): MPI 29/45, total time 5/45 — false-benign denominators.

## C1. Pipelines
| Pipeline | code dir | target | snapshot | family | variants scored | root |
|---|---:|---:|---:|---:|---:|---:|
| One-way, MPI time | delta_response_filteredInhibs (current snapshot; artifact 'control') | MPI time | current | one-way | 5 | filteredInhibs_mpiTimeSnapshot_full_20261017 |
| Bidirectional, MPI time | delta_response_filteredInhibs_biDirectional | MPI time | current | bidirectional | 13 | filteredInhibs_biDirectional_full_20261017 |
| One-way, total time | delta_response_filteredInhibs_totalTime (data/app_time) | total time | current | one-way | 5 | filteredInhibs_totalTime_full_20261017 |
| Bidirectional, total time | delta_response_filteredInhibs_biDirectional_totalTime | total time | current | bidirectional | 13 | filteredInhibs_biDirectional_totalTime_full_20261017 |
| One-way, MPI time (legacy snapshot) | delta_response_filteredInhibs (few_shot/data) | MPI time | legacy | one-way | 5 | filteredInhibs_full_20261017 |

## C2. Full-data (all 10 apps in training) metrics, best variant per family per pipeline
Best = lowest App-App log MAE within family (optimistic, as in the artifact); ablation variants excluded. FB = pair-level false-benign rate at 1.05 (README/local run; – = not available for that run).
| pipeline | variant | family | log MAE | Δ% vs baseline | CI lo % | CI hi % | signed bias | floor log MAE | above-floor log MAE | MAPE % | Spearman | FB@1.05 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| One-way, MPI time | Isolated-speed baseline | baseline | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | 1.0000 |
| One-way, MPI time | Absolute response | Absolute | 0.0942 | -8.5291 | -14.3100 | -3.0715 | -0.0776 | 0.0265 | 0.1112 | 8.2452 | 0.0583 | 0.7200 |
| One-way, MPI time | Delta, median agg. | Delta | 0.1235 | 19.8909 | -23.9720 | 61.6850 | 0.0982 | 0.2299 | 0.0969 | 13.5980 | 0.2839 | – |
| Bidirectional, MPI time | Isolated-speed baseline | baseline | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | 1.0000 |
| Bidirectional, MPI time | Absolute response | Absolute | 0.0942 | -8.5291 | -13.9275 | -3.2200 | -0.0776 | 0.0265 | 0.1112 | 8.2452 | 0.0583 | 0.7200 |
| Bidirectional, MPI time | Delta, median agg. | Delta | 0.1190 | 15.5564 | -29.3675 | 61.0123 | 0.0968 | 0.2355 | 0.0899 | 13.1804 | 0.2643 | 0.0000 |
| Bidirectional, MPI time | Bidir. profile | Joint (direct) | 0.0966 | -6.2323 | -11.9879 | -0.3417 | -0.0832 | 0.0231 | 0.1150 | 8.3822 | -0.0364 | 0.7600 |
| Bidirectional, MPI time | Bidir. context, calibrated | Joint (calibrated) | 0.0938 | -8.9640 | -24.8680 | 8.4243 | -0.0478 | 0.0628 | 0.1015 | 8.5527 | 0.2121 | 0.3100 |
| One-way, total time | Isolated-speed baseline | baseline | 0.0156 | – | – | – | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| One-way, total time | Absolute response | Absolute | 0.0162 | 3.6144 | -6.3588 | 13.4852 | -0.0127 | 0.0022 | 0.0255 | 1.5750 | 0.0087 | 1.0000 |
| One-way, total time | Generic potential | Delta | 0.0202 | 29.1965 | -18.7372 | 72.0701 | 0.0158 | 0.0221 | 0.0190 | 2.0451 | 0.5383 | – |
| Bidirectional, total time | Isolated-speed baseline | baseline | 0.0156 | – | – | – | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| Bidirectional, total time | Absolute response | Absolute | 0.0162 | 3.6144 | -6.3524 | 14.4767 | -0.0127 | 0.0022 | 0.0255 | 1.5750 | 0.0087 | 1.0000 |
| Bidirectional, total time | Delta, OOD-kernel agg. | Delta | 0.0170 | 8.8472 | -42.0995 | 57.6894 | 0.0128 | 0.0202 | 0.0149 | 1.7340 | 0.4286 | 0.4000 |
| Bidirectional, total time | Bidir. context | Joint (direct) | 0.0156 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| Bidirectional, total time | Joint, calibrated | Joint (calibrated) | 0.0139 | -10.9839 | -21.3280 | -2.8595 | -0.0128 | 0.0004 | 0.0229 | 1.3455 | 0.3778 | 1.0000 |
| One-way, MPI time (legacy snapshot) | Isolated-speed baseline | baseline | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | – |
| One-way, MPI time (legacy snapshot) | Absolute response | Absolute | 0.0882 | -14.3847 | -26.6299 | -1.3416 | -0.0430 | 0.0612 | 0.0949 | 7.9423 | 0.1186 | – |
| One-way, MPI time (legacy snapshot) | Generic potential | Delta | 0.2294 | 122.6760 | 70.4219 | 180.1450 | 0.2180 | 0.3874 | 0.1899 | 27.5518 | 0.2367 | – |

## C3. Full per-variant full-data metrics (supplementary)
### One-way, MPI time — n = 90 non-self outcomes, 18 at floor
| variant | code | log MAE | Δ% | 95% CI | bias | floor | above | MAPE % | Spearman | FB@1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolated-speed baseline | constant_1 | 0.1030 | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | 1.0000 |
| Absolute response | legacy_absolute / absolute_response | 0.0942 | -8.5291 | [-14.3, -3.1] | -0.0776 | 0.0265 | 0.1112 | 8.2452 | 0.0583 | 0.7200 |
| Delta, uniform agg. | legacy_delta_uniform / delta_uniform | 0.1522 | 47.7847 | [-4.3, +94.1] | 0.1341 | 0.2601 | 0.1253 | 17.3609 | 0.2551 | – |
| Delta, median agg. | legacy_delta_median / delta_median | 0.1235 | 19.8909 | [-24.0, +61.7] | 0.0982 | 0.2299 | 0.0969 | 13.5980 | 0.2839 | – |
| Delta, OOD-kernel agg. | legacy_delta_ood_kernel / delta_ood_kernel | 0.1302 | 26.3571 | [-21.9, +70.9] | 0.1220 | 0.2072 | 0.1109 | 14.7545 | 0.3345 | – |
| Generic potential | legacy_generic_<agg> / generic_potential | 0.1472 | 42.9377 | [-6.0, +88.7] | 0.1231 | 0.2565 | 0.1199 | 16.6233 | 0.2467 | – |

### Bidirectional, MPI time — n = 90 non-self outcomes, 18 at floor
| variant | code | log MAE | Δ% | 95% CI | bias | floor | above | MAPE % | Spearman | FB@1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolated-speed baseline | constant_1 | 0.1030 | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | 1.0000 |
| Absolute response | legacy_absolute / absolute_response | 0.0942 | -8.5291 | [-13.9, -3.2] | -0.0776 | 0.0265 | 0.1112 | 8.2452 | 0.0583 | 0.7200 |
| Delta, uniform agg. | legacy_delta_uniform / delta_uniform | 0.1457 | 41.4437 | [-4.0, +90.9] | 0.1279 | 0.2563 | 0.1181 | 16.5796 | 0.2495 | 0.0000 |
| Delta, median agg. | legacy_delta_median / delta_median | 0.1190 | 15.5564 | [-29.4, +61.0] | 0.0968 | 0.2355 | 0.0899 | 13.1804 | 0.2643 | 0.0000 |
| Delta, OOD-kernel agg. | legacy_delta_ood_kernel / delta_ood_kernel | 0.1258 | 22.1460 | [-24.2, +68.6] | 0.1170 | 0.2066 | 0.1056 | 14.2236 | 0.3174 | – |
| Generic potential | legacy_generic_<agg> / generic_potential | 0.1472 | 42.9377 | [-1.7, +92.8] | 0.1231 | 0.2565 | 0.1199 | 16.6233 | 0.2467 | – |
| Joint | joint_profile_direct | 0.1004 | -2.5356 | [-5.4, +0.3] | -0.0962 | 0.0094 | 0.1231 | 8.6414 | -0.1537 | – |
| Joint, calibrated | joint_profile_calibrated | 0.0940 | -8.7737 | [-29.0, +13.4] | -0.0183 | 0.0902 | 0.0949 | 8.8906 | 0.2076 | 0.1000 |
| Bidir. profile | bidirectional_profile_direct | 0.0966 | -6.2323 | [-12.0, -0.3] | -0.0832 | 0.0231 | 0.1150 | 8.3822 | -0.0364 | 0.7600 |
| Bidir. profile, calibrated | bidirectional_profile_calibrated | 0.0953 | -7.4894 | [-29.5, +16.1] | -0.0086 | 0.0992 | 0.0943 | 9.1275 | 0.2094 | 0.0300 |
| Bidir. context | bidirectional_context_direct | 0.1032 | 0.1408 | [-0.2, +0.6] | -0.1026 | 0.0013 | 0.1286 | 8.8684 | -0.1416 | – |
| Bidir. context, calibrated | bidirectional_context_calibrated | 0.0938 | -8.9640 | [-24.9, +8.4] | -0.0478 | 0.0628 | 0.1015 | 8.5527 | 0.2121 | 0.3100 |
| Wall-clock aux (ablation), cal. | bidirectional_profile_wall_calibrated | 0.0950 | -7.7942 | [-29.6, +15.7] | -0.0087 | 0.0993 | 0.0939 | 9.0993 | 0.2201 | – |
| Forward-only tokens (ablation), cal. | bidirectional_context_forward_calibrated | 0.0958 | -6.9798 | [-13.2, -0.1] | -0.0825 | 0.0244 | 0.1137 | 8.3164 | 0.0132 | – |

### One-way, total time — n = 90 non-self outcomes, 36 at floor
| variant | code | log MAE | Δ% | 95% CI | bias | floor | above | MAPE % | Spearman | FB@1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolated-speed baseline | constant_1 | 0.0156 | – | – | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| Absolute response | legacy_absolute / absolute_response | 0.0162 | 3.6144 | [-6.4, +13.5] | -0.0127 | 0.0022 | 0.0255 | 1.5750 | 0.0087 | 1.0000 |
| Delta, uniform agg. | legacy_delta_uniform / delta_uniform | 0.0228 | 45.9762 | [-2.6, +93.6] | 0.0196 | 0.0279 | 0.0195 | 2.3287 | 0.4394 | – |
| Delta, median agg. | legacy_delta_median / delta_median | 0.0243 | 55.3160 | [+8.0, +101.1] | 0.0207 | 0.0302 | 0.0203 | 2.4743 | 0.3934 | – |
| Delta, OOD-kernel agg. | legacy_delta_ood_kernel / delta_ood_kernel | 0.0206 | 31.8066 | [-19.5, +82.2] | 0.0181 | 0.0259 | 0.0171 | 2.0993 | 0.4644 | – |
| Generic potential | legacy_generic_<agg> / generic_potential | 0.0202 | 29.1965 | [-18.7, +72.1] | 0.0158 | 0.0221 | 0.0190 | 2.0451 | 0.5383 | – |

### Bidirectional, total time — n = 90 non-self outcomes, 36 at floor
| variant | code | log MAE | Δ% | 95% CI | bias | floor | above | MAPE % | Spearman | FB@1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolated-speed baseline | constant_1 | 0.0156 | – | – | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| Absolute response | legacy_absolute / absolute_response | 0.0162 | 3.6144 | [-6.4, +14.5] | -0.0127 | 0.0022 | 0.0255 | 1.5750 | 0.0087 | 1.0000 |
| Delta, uniform agg. | legacy_delta_uniform / delta_uniform | 0.0199 | 27.0791 | [-20.3, +70.9] | 0.0152 | 0.0215 | 0.0188 | 2.0162 | 0.5065 | 0.0000 |
| Delta, median agg. | legacy_delta_median / delta_median | 0.0209 | 33.8216 | [-10.0, +74.2] | 0.0160 | 0.0233 | 0.0193 | 2.1181 | 0.5099 | 0.0000 |
| Delta, OOD-kernel agg. | legacy_delta_ood_kernel / delta_ood_kernel | 0.0170 | 8.8472 | [-42.1, +57.7] | 0.0128 | 0.0202 | 0.0149 | 1.7340 | 0.4286 | 0.4000 |
| Generic potential | legacy_generic_<agg> / generic_potential | 0.0202 | 29.1965 | [-16.2, +70.0] | 0.0158 | 0.0221 | 0.0190 | 2.0451 | 0.5383 | 0.0000 |
| Joint | joint_profile_direct | 0.0156 | 0.0512 | [+0.0, +0.2] | -0.0156 | 0.0000 | 0.0261 | 1.5099 | -0.1139 | 1.0000 |
| Joint, calibrated | joint_profile_calibrated | 0.0139 | -10.9839 | [-21.3, -2.9] | -0.0128 | 0.0004 | 0.0229 | 1.3455 | 0.3778 | 1.0000 |
| Bidir. profile | bidirectional_profile_direct | 0.0158 | 1.2155 | [+0.1, +2.6] | -0.0154 | 0.0005 | 0.0261 | 1.5283 | -0.1330 | 1.0000 |
| Bidir. profile, calibrated | bidirectional_profile_calibrated | 0.0147 | -6.1540 | [-15.2, +1.7] | -0.0121 | 0.0008 | 0.0239 | 1.4204 | 0.3049 | 1.0000 |
| Bidir. context | bidirectional_context_direct | 0.0156 | 0.0000 | [+0.0, +0.0] | -0.0156 | 0.0000 | 0.0261 | 1.5091 | – | 1.0000 |
| Bidir. context, calibrated | bidirectional_context_calibrated | 0.0154 | -1.7912 | [-4.5, +0.0] | -0.0154 | 0.0000 | 0.0256 | 1.4818 | 0.1798 | 1.0000 |
| Wall-clock aux (ablation), cal. | bidirectional_profile_wall_calibrated | 0.0143 | -8.3099 | [-17.6, -0.4] | -0.0123 | 0.0007 | 0.0234 | 1.3871 | 0.3730 | 1.0000 |
| Forward-only tokens (ablation), cal. | bidirectional_context_forward_calibrated | 0.0175 | 11.6428 | [-3.7, +25.5] | -0.0099 | 0.0056 | 0.0253 | 1.6988 | -0.0417 | 1.0000 |

### One-way, MPI time (legacy snapshot) — n = 90 non-self outcomes, 18 at floor
| variant | code | log MAE | Δ% | 95% CI | bias | floor | above | MAPE % | Spearman | FB@1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolated-speed baseline | constant_1 | 0.1030 | – | – | -0.1030 | 0.0000 | 0.1288 | 8.8528 | – | – |
| Absolute response | legacy_absolute / absolute_response | 0.0882 | -14.3847 | [-26.6, -1.3] | -0.0430 | 0.0612 | 0.0949 | 7.9423 | 0.1186 | – |
| Delta, uniform agg. | legacy_delta_uniform / delta_uniform | 0.2420 | 134.9717 | [+83.4, +189.9] | 0.2367 | 0.3878 | 0.2056 | 29.2735 | 0.3075 | – |
| Delta, median agg. | legacy_delta_median / delta_median | 0.2380 | 131.0187 | [+79.2, +186.6] | 0.2333 | 0.3883 | 0.2004 | 28.8929 | 0.2881 | – |
| Delta, OOD-kernel agg. | legacy_delta_ood_kernel / delta_ood_kernel | 0.2429 | 135.8143 | [+85.7, +191.6] | 0.2393 | 0.3763 | 0.2096 | 29.5761 | 0.3364 | – |
| Generic potential | legacy_generic_<agg> / generic_potential | 0.2294 | 122.6760 | [+70.4, +180.1] | 0.2180 | 0.3874 | 0.1899 | 27.5518 | 0.2367 | – |

## C4. Matched cells (20 (split, size) cells completed by all five runs, sizes 4–8)
Cells: s04/8, s05/7, s05/8, s06/6, s06/7, s06/8, s07/4, s07/5, s07/6, s07/7, s07/8, s08/4, s08/5, s08/6, s08/7, s08/8, s09/4, s09/5, s09/6, s09/7
Entry = pooled skill (1 − mean method log MAE / mean baseline log MAE over the 20 cells) and wins/20.
### one_known
|  | One-way, MPI time | Bidirectional, MPI time | One-way, total time | Bidirectional, total time | One-way, MPI time (legacy snapshot) |
|---|---:|---:|---:|---:|---:|
| Absolute response | +9.3% (20) | +9.3% (20) | -20.2% (2) | -20.2% (2) | +14.2% (19) |
| Delta, uniform agg. | -46.7% (2) | -48.6% (1) | -46.3% (6) | -68.8% (1) | -148.5% (0) |
| Delta, median agg. | -27.6% (3) | -29.3% (3) | -53.1% (6) | -70.7% (1) | -142.1% (0) |
| Delta, OOD-kernel agg. | -27.8% (4) | -35.4% (4) | -30.6% (7) | -61.8% (3) | -141.6% (0) |
| Generic potential | -39.8% (2) | -39.8% (2) | -32.9% (4) | -32.9% (4) | -109.5% (0) |
| Joint | – | +1.5% (14) | – | +0.2% (7) | – |
| Joint, calibrated | – | +8.4% (17) | – | +9.6% (18) | – |
| Bidir. profile | – | +4.6% (17) | – | +0.2% (12) | – |
| Bidir. profile, calibrated | – | +8.8% (17) | – | +6.8% (17) | – |
| Bidir. context | – | +0.1% (5) | – | +0.0% (0) | – |
| Bidir. context, calibrated | – | +3.3% (15) | – | +0.9% (10) | – |
| Wall-clock aux (ablation), cal. | – | +9.1% (17) | – | +7.4% (17) | – |
| Forward-only tokens (ablation), cal. | – | +6.0% (18) | – | -12.6% (4) | – |

### zero_shot
|  | One-way, MPI time | Bidirectional, MPI time | One-way, total time | Bidirectional, total time | One-way, MPI time (legacy snapshot) |
|---|---:|---:|---:|---:|---:|
| Absolute response | -4.1% (10) | -4.1% (10) | -109.5% (0) | -109.5% (0) | -12.5% (9) |
| Delta, uniform agg. | -145.2% (1) | -160.8% (1) | -88.0% (3) | -157.0% (1) | -312.5% (1) |
| Delta, median agg. | -113.2% (1) | -133.9% (1) | -104.4% (3) | -164.8% (1) | -307.2% (1) |
| Delta, OOD-kernel agg. | -113.4% (2) | -133.8% (2) | -62.7% (7) | -140.6% (1) | -280.0% (1) |
| Generic potential | -114.3% (1) | -114.3% (1) | -82.1% (3) | -82.1% (3) | -215.6% (1) |
| Joint | – | -0.8% (4) | – | -1.0% (1) | – |
| Joint, calibrated | – | +1.0% (14) | – | -4.8% (3) | – |
| Bidir. profile | – | -2.4% (5) | – | -0.3% (0) | – |
| Bidir. profile, calibrated | – | -2.0% (13) | – | +0.9% (7) | – |
| Bidir. context | – | +0.0% (0) | – | +0.0% (0) | – |
| Bidir. context, calibrated | – | +0.7% (12) | – | +0.0% (0) | – |
| Wall-clock aux (ablation), cal. | – | -2.2% (13) | – | +0.3% (4) | – |
| Forward-only tokens (ablation), cal. | – | +0.0% (10) | – | -13.1% (2) | – |

## Restricted tiers, pooled over all completed cells (sizes 4–8), canonical pipelines
### one_known
| pipeline | variant | pooled skill % | cells | per size (skill/splits) |
|---|---|---:|---:|---:|
| One-way, MPI time | Absolute response | 7.2 | 36 | 4:+8%/6, 5:+8%/6, 6:+8%/7, 7:+7%/8, 8:+6%/9 |
| One-way, MPI time | Delta, uniform agg. | -50.8 | 36 | 4:-42%/6, 5:-26%/6, 6:-43%/7, 7:-61%/8, 8:-73%/9 |
| Bidirectional, MPI time | Absolute response | 6.3 | 47 | 4:+7%/8, 5:+7%/9, 6:+7%/10, 7:+6%/10, 8:+6%/10 |
| Bidirectional, MPI time | Delta, uniform agg. | -53.9 | 47 | 4:-49%/8, 5:-43%/9, 6:-58%/10, 7:-59%/10, 8:-58%/10 |
| Bidirectional, MPI time | Joint, calibrated | 5.5 | 47 | 4:+6%/8, 5:+8%/9, 6:+7%/10, 7:+5%/10, 8:+3%/10 |
| Bidirectional, MPI time | Bidir. profile | 5.1 | 47 | 4:+4%/8, 5:+6%/9, 6:+5%/10, 7:+5%/10, 8:+5%/10 |
| Bidirectional, MPI time | Bidir. profile, calibrated | 5.2 | 47 | 4:+6%/8, 5:+7%/9, 6:+6%/10, 7:+4%/10, 8:+2%/10 |
| Bidirectional, MPI time | Bidir. context, calibrated | 4.0 | 47 | 4:+3%/8, 5:+5%/9, 6:+5%/10, 7:+3%/10, 8:+3%/10 |
| One-way, total time | Absolute response | -20.3 | 21 | 4:-24%/4, 5:-20%/3, 6:-20%/4, 7:-27%/5, 8:-11%/5 |
| One-way, total time | Delta, OOD-kernel agg. | -28.8 | 21 | 4:-36%/4, 5:-66%/3, 6:-23%/4, 7:-19%/5, 8:-16%/5 |
| Bidirectional, total time | Absolute response | -21.2 | 47 | 4:-24%/8, 5:-22%/9, 6:-20%/10, 7:-20%/10, 8:-20%/10 |
| Bidirectional, total time | Delta, uniform agg. | -74.5 | 47 | 4:-75%/8, 5:-66%/9, 6:-78%/10, 7:-82%/10, 8:-71%/10 |
| Bidirectional, total time | Joint, calibrated | 2.7 | 47 | 4:+6%/8, 5:+3%/9, 6:+1%/10, 7:+1%/10, 8:+4%/10 |
| Bidirectional, total time | Bidir. profile | -6.7 | 47 | 4:-5%/8, 5:-6%/9, 6:-10%/10, 7:-8%/10, 8:-5%/10 |
| Bidirectional, total time | Bidir. profile, calibrated | -4.6 | 47 | 4:-1%/8, 5:-5%/9, 6:-7%/10, 7:-6%/10, 8:-3%/10 |
| Bidirectional, total time | Bidir. context, calibrated | -0.3 | 47 | 4:+0%/8, 5:+0%/9, 6:-2%/10, 7:-0%/10, 8:+1%/10 |
| One-way, MPI time (legacy snapshot) | Absolute response | 12.1 | 30 | 4:+14%/4, 5:+16%/5, 6:+15%/6, 7:+12%/7, 8:+6%/8 |
| One-way, MPI time (legacy snapshot) | Delta, uniform agg. | -131.6 | 30 | 4:-125%/4, 5:-130%/5, 6:-127%/6, 7:-133%/7, 8:-139%/8 |

### zero_shot
| pipeline | variant | pooled skill % | cells | per size (skill/splits) |
|---|---|---:|---:|---:|
| One-way, MPI time | Absolute response | -8.5 | 36 | 4:-3%/6, 5:-7%/6, 6:-12%/7, 7:-14%/8, 8:-8%/9 |
| One-way, MPI time | Delta, uniform agg. | -137.8 | 36 | 4:-78%/6, 5:-110%/6, 6:-185%/7, 7:-247%/8, 8:-104%/9 |
| Bidirectional, MPI time | Absolute response | -4.0 | 47 | 4:-2%/8, 5:-4%/9, 6:-5%/10, 7:-6%/10, 8:-3%/10 |
| Bidirectional, MPI time | Delta, uniform agg. | -89.2 | 47 | 4:-96%/8, 5:-108%/9, 6:-98%/10, 7:-98%/10, 8:-56%/10 |
| Bidirectional, MPI time | Joint, calibrated | 3.8 | 47 | 4:+2%/8, 5:+1%/9, 6:+2%/10, 7:+4%/10, 8:+9%/10 |
| Bidirectional, MPI time | Bidir. profile | 8.7 | 47 | 4:+6%/8, 5:+5%/9, 6:+9%/10, 7:+11%/10, 8:+11%/10 |
| Bidirectional, MPI time | Bidir. profile, calibrated | 2.0 | 47 | 4:+2%/8, 5:-1%/9, 6:+0%/10, 7:+1%/10, 8:+7%/10 |
| Bidirectional, MPI time | Bidir. context, calibrated | 4.8 | 47 | 4:-1%/8, 5:-1%/9, 6:+2%/10, 7:+9%/10, 8:+12%/10 |
| One-way, total time | Absolute response | -112.5 | 21 | 4:-96%/4, 5:-104%/3, 6:-145%/4, 7:-141%/5, 8:-91%/5 |
| One-way, total time | Delta, OOD-kernel agg. | -61.5 | 21 | 4:-42%/4, 5:-101%/3, 6:-109%/4, 7:-93%/5, 8:+0%/5 |
| Bidirectional, total time | Absolute response | -70.5 | 47 | 4:-74%/8, 5:-71%/9, 6:-69%/10, 7:-70%/10, 8:-69%/10 |
| Bidirectional, total time | Delta, uniform agg. | -113.0 | 47 | 4:-140%/8, 5:-167%/9, 6:-125%/10, 7:-108%/10, 8:-55%/10 |
| Bidirectional, total time | Joint, calibrated | -4.9 | 47 | 4:-8%/8, 5:-16%/9, 6:-10%/10, 7:-1%/10, 8:+4%/10 |
| Bidirectional, total time | Bidir. profile | -3.1 | 47 | 4:-8%/8, 5:-10%/9, 6:-4%/10, 7:-1%/10, 8:+3%/10 |
| Bidirectional, total time | Bidir. profile, calibrated | -7.5 | 47 | 4:-15%/8, 5:-22%/9, 6:-14%/10, 7:-4%/10, 8:+9%/10 |
| Bidirectional, total time | Bidir. context, calibrated | -0.2 | 47 | 4:-6%/8, 5:-6%/9, 6:-6%/10, 7:+7%/10, 8:+5%/10 |
| One-way, MPI time (legacy snapshot) | Absolute response | -18.0 | 30 | 4:-12%/4, 5:-15%/5, 6:-16%/6, 7:-29%/7, 8:-17%/8 |
| One-way, MPI time (legacy snapshot) | Delta, uniform agg. | -210.6 | 30 | 4:-174%/4, 5:-271%/5, 6:-294%/6, 7:-295%/7, 8:-104%/8 |

## C5. Per-victim error ratio (victim log MAE over its 9 non-self outcomes ÷ baseline's), full data, MPI time
One-way absolute = bidirectional legacy absolute (identical predictions), so it appears once.
|  | baseline log MAE | 1W: Absolute response | 1W: Delta, uniform agg. | 1W: Generic potential | BD: Delta, uniform agg. | BD: Joint, calibrated | BD: Bidir. profile | BD: Bidir. profile, calibrated | BD: Bidir. context, calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMG2023 | 0.154 | 0.884 | 1.866 | 1.860 | 1.835 | 1.128 | 0.903 | 1.152 | 1.020 |
| Beatnik | 0.193 | 0.872 | 0.354 | 0.367 | 0.307 | 0.471 | 0.917 | 0.429 | 0.634 |
| FIESTA | 0.044 | 1.134 | 2.497 | 2.479 | 2.513 | 1.056 | 1.282 | 1.168 | 1.032 |
| Kripke | 0.048 | 0.596 | 4.524 | 4.164 | 4.392 | 1.398 | 0.665 | 1.637 | 0.822 |
| Laghos | 0.262 | 0.968 | 0.958 | 1.001 | 0.932 | 0.857 | 0.973 | 0.844 | 0.884 |
| LAMMPS | 0.034 | 1.303 | 3.910 | 3.561 | 3.706 | 1.185 | 1.010 | 1.337 | 0.981 |
| MiniFE | 0.111 | 0.955 | 2.755 | 2.727 | 2.702 | 1.553 | 1.061 | 1.590 | 1.375 |
| miniVite | 0.099 | 0.776 | 0.706 | 0.669 | 0.628 | 0.721 | 0.855 | 0.704 | 0.861 |
| Quicksilver | 0.011 | 2.030 | 3.452 | 2.339 | 3.148 | 0.992 | 1.198 | 1.187 | 0.958 |
| TriC | 0.074 | 0.751 | 0.557 | 0.377 | 0.350 | 0.567 | 0.762 | 0.498 | 0.808 |

Absolute response beats the baseline for 7/10 victims; worse for: FIESTA, LAMMPS, Quicksilver

## C6. App-Inhibitor crossed-CV log MAE vs App-App log MAE (full data)
| pipeline | model | CV log MAE | App-App log MAE | epochs | λ / agg |
|---|---|---:|---:|---:|---:|
| One-way, MPI time | Low-rank delta (tuned) | 0.1580 | 0.1522 | 5 | uniform |
| One-way, MPI time | Generic potential | 0.1618 | 0.1472 | 14 | uniform |
| One-way, MPI time | Absolute response | 0.1562 | 0.0942 | 2 | None |
| Bidirectional, MPI time | Legacy delta | 0.1690 | 0.1457 | 13 | None |
| Bidirectional, MPI time | Legacy generic | 0.1618 | 0.1472 | 14 | None |
| Bidirectional, MPI time | Legacy absolute | 0.1562 | 0.0942 | 2 | None |
| Bidirectional, MPI time | Joint profile (cal.) | 0.1481 | 0.0940 | 1 | 1.0000 |
| Bidirectional, MPI time | Bidir. profile (cal.) | 0.1479 | 0.0953 | 2 | 1.0000 |
| Bidirectional, MPI time | Bidir. context (cal.) | 0.1481 | 0.0938 | 2 | 1.0000 |
| Bidirectional, MPI time | Wall-clock aux (cal.) | 0.1513 | 0.0950 | 2 | 1.0000 |
| Bidirectional, MPI time | Forward-only tokens (cal.) | 0.1443 | 0.0958 | 1 | 10.0000 |
| One-way, total time | Low-rank delta (tuned) | 0.0206 | 0.0206 | 4 | ood_kernel |
| One-way, total time | Generic potential | 0.0204 | 0.0202 | 4 | uniform |
| One-way, total time | Absolute response | 0.0318 | 0.0162 | 6 | None |
| Bidirectional, total time | Legacy delta | 0.0338 | 0.0199 | 3 | None |
| Bidirectional, total time | Legacy generic | 0.0204 | 0.0202 | 4 | None |
| Bidirectional, total time | Legacy absolute | 0.0318 | 0.0162 | 6 | None |
| Bidirectional, total time | Joint profile (cal.) | 0.0203 | 0.0139 | 5 | 1.0000 |
| Bidirectional, total time | Bidir. profile (cal.) | 0.0216 | 0.0147 | 3 | 1.0000 |
| Bidirectional, total time | Bidir. context (cal.) | 0.0200 | 0.0154 | 1 | 1.0000 |
| Bidirectional, total time | Wall-clock aux (cal.) | 0.0184 | 0.0143 | 4 | 1.0000 |
| Bidirectional, total time | Forward-only tokens (cal.) | 0.0223 | 0.0175 | 2 | 10.0000 |
| One-way, MPI time (legacy snapshot) | Low-rank delta (tuned) | 0.1197 | 0.2420 | 6 | uniform |
| One-way, MPI time (legacy snapshot) | Generic potential | 0.1604 | 0.2294 | 10 | uniform |
| One-way, MPI time (legacy snapshot) | Absolute response | 0.1595 | 0.0882 | 4 | None |

One-way, MPI time: Spearman(CV, App-App) over 3 models = +0.50; CV-best = Absolute response, App-App-best = Absolute response
Bidirectional, MPI time: Spearman(CV, App-App) over 8 models = +0.36; CV-best = Forward-only tokens (cal.), App-App-best = Bidir. context (cal.)
One-way, total time: Spearman(CV, App-App) over 3 models = -0.50; CV-best = Generic potential, App-App-best = Absolute response
Bidirectional, total time: Spearman(CV, App-App) over 8 models = +0.60; CV-best = Wall-clock aux (cal.), App-App-best = Joint profile (cal.)
One-way, MPI time (legacy snapshot): Spearman(CV, App-App) over 3 models = -0.50; CV-best = Low-rank delta (tuned), App-App-best = Absolute response

## C7. Ablation contrasts (App-App non-self log MAE, method − comparison, paired 95% CI)
MPI time: from the bidirectional README (same fit as the artifact). Total time: local outputs/full_comparison/ablation_summary.csv (same fit as the artifact).
| contrast | MPI Δ | MPI CI | MPI Δ % of baseline | total Δ | total CI | total Δ % of baseline |
|---|---:|---:|---:|---:|---:|---:|
| Joint direct vs legacy absolute | +0.0062 | [+0.0012, +0.0118] | +6.0187 | -0.0006 | [-0.0022, +0.0010] | -3.5608 |
| Floor-aware calibration (joint cal − joint direct) | -0.0064 | [-0.0248, +0.0143] | -6.2129 | -0.0017 | [-0.0033, -0.0004] | -11.0345 |
| Reverse supervision, direct (bprof − joint) | -0.0038 | [-0.0073, -0.0002] | -3.6889 | +0.0002 | [+0.0000, +0.0004] | +1.1656 |
| Reverse supervision, calibrated | +0.0013 | [-0.0010, +0.0035] | +1.2620 | +0.0008 | [+0.0002, +0.0013] | +4.8303 |
| Fingerprints, direct (bctx − bprof) | +0.0066 | [+0.0006, +0.0120] | +6.4070 | -0.0002 | [-0.0004, -0.0000] | -1.2168 |
| Fingerprints, calibrated | -0.0015 | [-0.0102, +0.0071] | -1.4561 | +0.0007 | [-0.0004, +0.0018] | +4.3600 |
| Wall-clock-only aux, calibrated (bwall − bprof) | -0.0003 | n/a (README: within ±0.003) | -0.3048 | -0.0003 | [-0.0009, +0.0002] | -2.1561 |
| Forward-only tokens, calibrated (bfwd − bctx) | +0.0020 | n/a (README: within ±0.003) | +1.9842 | +0.0021 | [-0.0001, +0.0042] | +13.4358 |
| Legacy delta (uniform) vs legacy absolute | +0.0515 | [+0.0088, +0.0947] | +49.9942 | +0.0037 | [-0.0027, +0.0098] | +23.4649 |

## C8. Calibration effect: signed log bias and false-benign rate, full data
| pipeline | variant | signed bias | floor log MAE | above-floor log MAE | FB@1.05 |
|---|---|---:|---:|---:|---:|
| Bidirectional, MPI time | Isolated-speed baseline | -0.103 | 0.000 | 0.129 | 1.000 |
| Bidirectional, MPI time | Absolute response | -0.078 | 0.027 | 0.111 | 0.720 |
| Bidirectional, MPI time | Joint | -0.096 | 0.009 | 0.123 | – |
| Bidirectional, MPI time | Joint, calibrated | -0.018 | 0.090 | 0.095 | 0.100 |
| Bidirectional, MPI time | Bidir. profile | -0.083 | 0.023 | 0.115 | 0.760 |
| Bidirectional, MPI time | Bidir. profile, calibrated | -0.009 | 0.099 | 0.094 | 0.030 |
| Bidirectional, MPI time | Bidir. context | -0.103 | 0.001 | 0.129 | – |
| Bidirectional, MPI time | Bidir. context, calibrated | -0.048 | 0.063 | 0.102 | 0.310 |
| Bidirectional, total time | Isolated-speed baseline | -0.016 | 0.000 | 0.026 | 1.000 |
| Bidirectional, total time | Absolute response | -0.013 | 0.002 | 0.026 | 1.000 |
| Bidirectional, total time | Joint | -0.016 | 0.000 | 0.026 | 1.000 |
| Bidirectional, total time | Joint, calibrated | -0.013 | 0.000 | 0.023 | 1.000 |
| Bidirectional, total time | Bidir. profile | -0.015 | 0.000 | 0.026 | 1.000 |
| Bidirectional, total time | Bidir. profile, calibrated | -0.012 | 0.001 | 0.024 | 1.000 |
| Bidirectional, total time | Bidir. context | -0.016 | 0.000 | 0.026 | 1.000 |
| Bidirectional, total time | Bidir. context, calibrated | -0.015 | 0.000 | 0.026 | 1.000 |

## C9. Restricted-tier coverage (tasks completed out of 10 per size; reasons for failures)
| pipeline | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 4–8 total | failure reasons (all sizes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| One-way, MPI time | 0 | 0 | 6 | 6 | 7 | 8 | 9 | 36/50 | no_uncensored_anchor 11; panel_smaller_than_cv_blocks 23 |
| Bidirectional, MPI time | 0 | 0 | 8 | 9 | 10 | 10 | 10 | 47/50 | panel_smaller_than_cv_blocks 23 |
| One-way, total time | 0 | 0 | 4 | 3 | 4 | 5 | 5 | 21/50 | no_uncensored_anchor 26; panel_smaller_than_cv_blocks 23 |
| Bidirectional, total time | 0 | 0 | 8 | 9 | 10 | 10 | 10 | 47/50 | panel_smaller_than_cv_blocks 23 |
| One-way, MPI time (legacy snapshot) | 0 | 0 | 4 | 5 | 6 | 7 | 8 | 30/50 | no_response_anchors 1; no_uncensored_anchor 18; panel_smaller_than_cv_blocks 21 |

## B1 forest_fulldata.pdf: 4 canonical panels (legacy-snapshot one-way MPI omitted; see C3).
## B2 bias_vs_floor.pdf: MPI-time canonical pipelines; hollow = one-way (current snapshot), filled = bidirectional; ablation variants omitted; star = baseline.

## B3 learning_curves.pdf: ratio = method / baseline pooled log MAE per size; hollow marker = fewer than 10 splits completed.
  One-way, MPI time one_known: splits per size {4: np.int64(6), 5: np.int64(6), 6: np.int64(7), 7: np.int64(8), 8: np.int64(9)}
  One-way, MPI time zero_shot: splits per size {4: np.int64(6), 5: np.int64(6), 6: np.int64(7), 7: np.int64(8), 8: np.int64(9)}
  Bidirectional, MPI time one_known: splits per size {4: np.int64(8), 5: np.int64(9), 6: np.int64(10), 7: np.int64(10), 8: np.int64(10)}
  Bidirectional, MPI time zero_shot: splits per size {4: np.int64(8), 5: np.int64(9), 6: np.int64(10), 7: np.int64(10), 8: np.int64(10)}
  Parenthesised numbers under the x ticks = completed splits per size.

## B4 pervictim_heatmap.pdf: rows sorted by mean ratio (best first); colour scale log, saturates at 0.5× and 2×.
Columns: one-way Absolute response; one-way Delta, uniform agg.; one-way Generic potential; bidir Delta, uniform agg.; bidir Joint, calibrated; bidir Bidir. profile; bidir Bidir. profile, calibrated; bidir Bidir. context, calibrated

## B5 cv_vs_appapp.pdf: MPI-time canonical pipelines; ablations omitted; hollow = one-way, filled = bidirectional.
