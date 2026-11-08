## S5. Socket-matched co-run slowdowns
Baselines: **J** = June isolated mean (chapter's current numbers); **N0** = new pinned sock0 for all reps (requested 'before'); **NM** = new pinned baseline of the socket each rep's victim ran on (requested 'after'); **JM** = June mean, times exp(mean paired isolated d) for socket-1 reps (socket matching without the S3 launcher/epoch drift); **JMs** = JM with d set to 0 unless its 95% CI excludes 0 (T_MPI: AMG2023 and Quicksilver only; T_app: AMG2023 and Quicksilver). Slowdown = max(1, mean co / mean matched baseline). SE = delta-method SE of the raw ratio (co reps + baseline reps).
Effect of socket matching alone: compare N0 -> NM (new baselines) or J -> JM / JMs (June baselines).

### App-App, non-self directional outcomes, T_MPI slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 90.0000 | 1.1233 | 1.0510 | 1.3113 | 2.2094 | 0.2000 | 0.5111 | 0.3111 | 0.0423 | 0.1504 |
| N0 | 90.0000 | 1.0997 | 1.0074 | 1.2640 | 2.3674 | 0.4556 | 0.3111 | 0.1889 | 0.0527 | 0.1607 |
| NM | 90.0000 | 1.1366 | 1.0220 | 1.2957 | 2.5325 | 0.3556 | 0.3556 | 0.2667 | 0.0547 | 0.1631 |
| JM | 90.0000 | 1.1983 | 1.0982 | 1.5459 | 2.3245 | 0.1222 | 0.6667 | 0.4889 | 0.0552 | 0.2284 |
| JMs | 90.0000 | 1.1352 | 1.0648 | 1.3113 | 2.2094 | 0.1778 | 0.5556 | 0.3667 | 0.0423 | 0.1504 |

### App-App, non-self directional outcomes, T_app slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 90.0000 | 1.0162 | 1.0014 | 1.0427 | 1.1925 | 0.3667 | 0.0667 | 0.0333 | 0.0067 | 0.0182 |
| N0 | 90.0000 | 1.0090 | 1.0000 | 1.0178 | 1.1934 | 0.6778 | 0.0444 | 0.0333 | 0.0074 | 0.0227 |
| NM | 90.0000 | 1.0116 | 1.0000 | 1.0193 | 1.2050 | 0.5778 | 0.0556 | 0.0444 | 0.0075 | 0.0217 |
| JM | 90.0000 | 1.0228 | 1.0104 | 1.0470 | 1.2037 | 0.2000 | 0.0778 | 0.0444 | 0.0084 | 0.0210 |
| JMs | 90.0000 | 1.0171 | 1.0036 | 1.0427 | 1.1925 | 0.3333 | 0.0667 | 0.0333 | 0.0067 | 0.0182 |

### App-Inhibitor configs, T_MPI slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 2870.0000 | 1.4983 | 1.2036 | 2.6375 | 8.5104 | 0.1544 | 0.7049 | 0.6254 | 0.0733 | 0.3235 |
| N0 | 2870.0000 | 1.4364 | 1.0907 | 2.5226 | 6.0230 | 0.3038 | 0.5606 | 0.4850 | 0.0825 | 0.3505 |
| NM | 2870.0000 | 1.4980 | 1.1143 | 2.7001 | 6.6976 | 0.2185 | 0.6070 | 0.5237 | 0.0851 | 0.3362 |
| JM | 2870.0000 | 1.5914 | 1.2630 | 2.7868 | 13.1154 | 0.0645 | 0.7746 | 0.6916 | 0.0914 | 0.4081 |
| JMs | 2870.0000 | 1.5184 | 1.2067 | 2.6801 | 8.5104 | 0.1226 | 0.7136 | 0.6321 | 0.0737 | 0.3273 |

### App-Inhibitor configs, T_app slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 2870.0000 | 1.0586 | 1.0218 | 1.1852 | 1.5095 | 0.2021 | 0.3150 | 0.1801 | 0.0089 | 0.0335 |
| N0 | 2870.0000 | 1.0480 | 1.0076 | 1.1592 | 1.5106 | 0.4233 | 0.2599 | 0.1474 | 0.0108 | 0.0361 |
| NM | 2870.0000 | 1.0516 | 1.0120 | 1.1635 | 1.5402 | 0.3394 | 0.2749 | 0.1519 | 0.0106 | 0.0345 |
| JM | 2870.0000 | 1.0651 | 1.0333 | 1.1903 | 1.5382 | 0.1226 | 0.3502 | 0.1871 | 0.0106 | 0.0361 |
| JMs | 2870.0000 | 1.0596 | 1.0226 | 1.1852 | 1.5095 | 0.1767 | 0.3195 | 0.1801 | 0.0089 | 0.0335 |

Check: J reproduces chapter B2/A6 (App-App MPI mean 1.123, median 1.051, max 2.209, floor 0.200; App-Inh MPI floor 0.154, n = 2870).

Per-victim median App-App MPI slowdown by baseline:
| victim | J | N0 | NM | JM | JMs | frac reps on sock1 |
|---|---|---|---|---|---|---|
| AMG2023 | 1.049 | 1.087 | 1.193 | 1.151 | 1.151 | 0.417 |
| Beatnik | 1.166 | 1.054 | 1.060 | 1.173 | 1.166 | 0.472 |
| FIESTA | 1.016 | 1.058 | 1.093 | 1.048 | 1.016 | 0.583 |
| Kripke | 1.037 | 1.000 | 1.000 | 1.049 | 1.037 | 0.472 |
| Laghos | 1.177 | 1.261 | 1.338 | 1.238 | 1.177 | 0.500 |
| LAMMPS | 1.004 | 1.000 | 1.000 | 1.419 | 1.004 | 0.639 |
| MiniFE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.472 |
| miniVite | 1.051 | 1.000 | 1.000 | 1.086 | 1.051 | 0.472 |
| Quicksilver | 1.011 | 1.004 | 1.017 | 1.025 | 1.025 | 0.417 |
| TriC | 1.067 | 1.013 | 1.014 | 1.069 | 1.067 | 0.556 |

## S6. Re-scoring frozen predictions against socket-matched App-App outcomes (no re-fit)
Predictions: random_split, non-self outcomes (90 directional, 45 pairs). Only the truth changes; predicted log slowdowns are as saved. Δ% = log MAE vs the isolated-speed baseline (constant 1); CI = paired pair-cluster bootstrap over the 45 non-self pairs (1,000 draws, seed 923, one shared draw matrix; re-implemented here). Pooled skill = 1 − MAE(method)/MAE(baseline). FB = false-benign rate at 1.05 over truly harmful pairs (pair worst endpoint, as in metrics.py).
MPI predictions: one-way MPI pipeline on the canonical snapshot, local `filtered_fit_mpi_time` run. Its constant-1, absolute and generic log MAE equal the artifact's control run (baseSnap) exactly; † its delta models use a different recipe from the artifact (artifact delta median 0.1235 vs local 0.1190), so treat them as indicative. The joint MPI model's per-outcome predictions are not on this machine (bidirectional MPI root missing), so joint is scored on total time only.
Total-time predictions: local bidirectional total-time `full_comparison` run, identical to the artifact's bidirTT run.

### T_MPI target (check: max |saved true slowdown − J truth| = 6.7e-16)

**isolated-speed baseline**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1030 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 29/45 |
| N0 | 0.0796 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 24/45 |
| NM | 0.1043 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 26/45 |
| JM | 0.1607 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 38/45 |
| JMs | 0.1122 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 32/45 |

**absolute**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0942 | -8.5292 | -12.8450 | -3.7480 | 0.0853 | 0.7241 | 29/45 |
| N0 | 0.0814 | 2.2592 | -4.7254 | 13.2799 | -0.0226 | 0.7917 | 24/45 |
| NM | 0.1012 | -2.9390 | -7.4079 | 2.9313 | 0.0294 | 0.8077 | 26/45 |
| JM | 0.1435 | -10.7253 | -13.9485 | -8.1299 | 0.1073 | 0.7895 | 38/45 |
| JMs | 0.1006 | -10.3446 | -14.3124 | -6.3985 | 0.1034 | 0.7500 | 32/45 |

**generic**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1472 | 42.9381 | 0.8944 | 120.9666 | -0.4294 | 0.0000 | 29/45 |
| N0 | 0.1714 | 115.2561 | 48.3343 | 261.9528 | -1.1526 | 0.0000 | 24/45 |
| NM | 0.1703 | 63.3160 | 17.6084 | 155.4109 | -0.6332 | 0.0000 | 26/45 |
| JM | 0.1501 | -6.6192 | -27.8298 | 26.7469 | 0.0662 | 0.0000 | 38/45 |
| JMs | 0.1443 | 28.5905 | -8.0006 | 94.6781 | -0.2859 | 0.0000 | 32/45 |

**delta (median)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1190 | 15.5569 | -19.3099 | 81.7672 | -0.1556 | 0.0000 | 29/45 |
| N0 | 0.1452 | 82.3860 | 25.1266 | 205.6307 | -0.8239 | 0.0000 | 24/45 |
| NM | 0.1451 | 39.1466 | 1.0503 | 116.4093 | -0.3915 | 0.0000 | 26/45 |
| JM | 0.1261 | -21.5456 | -39.1322 | 6.8985 | 0.2155 | 0.0000 | 38/45 |
| JMs | 0.1162 | 3.5125 | -26.3106 | 59.5556 | -0.0351 | 0.0000 | 32/45 |

**delta (OOD kernel)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1258 | 22.1459 | -17.9511 | 89.4610 | -0.2215 | 0.0000 | 29/45 |
| N0 | 0.1514 | 90.1187 | 29.2003 | 226.6318 | -0.9012 | 0.0000 | 24/45 |
| NM | 0.1461 | 40.1021 | -0.2116 | 120.9524 | -0.4010 | 0.0000 | 26/45 |
| JM | 0.1252 | -22.0800 | -42.6156 | 8.4904 | 0.2208 | 0.0000 | 38/45 |
| JMs | 0.1179 | 5.0998 | -29.3538 | 61.4920 | -0.0510 | 0.0000 | 32/45 |

**delta (uniform)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1457 | 41.4438 | -1.9199 | 121.7460 | -0.4144 | 0.0000 | 29/45 |
| N0 | 0.1712 | 115.0411 | 43.3655 | 263.1827 | -1.1504 | 0.0000 | 24/45 |
| NM | 0.1702 | 63.2144 | 17.6815 | 159.4803 | -0.6321 | 0.0000 | 26/45 |
| JM | 0.1460 | -9.1517 | -31.8139 | 25.4375 | 0.0915 | 0.0000 | 38/45 |
| JMs | 0.1409 | 25.5786 | -12.0487 | 92.3199 | -0.2558 | 0.0000 | 32/45 |

### T_app target (check: max |saved true slowdown − J truth| = 2.2e-16)

**isolated-speed baseline**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |
| N0 | 0.0086 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 4/45 |
| NM | 0.0110 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |
| JM | 0.0220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 6/45 |
| JMs | 0.0165 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |

**absolute**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0162 | 3.6121 | -5.2886 | 17.1901 | -0.0361 | 1.0000 | 5/45 |
| N0 | 0.0097 | 13.6838 | -2.7422 | 55.0647 | -0.1368 | 1.0000 | 4/45 |
| NM | 0.0119 | 7.9918 | -3.7536 | 31.9803 | -0.0799 | 1.0000 | 5/45 |
| JM | 0.0213 | -3.0280 | -9.1361 | 4.8392 | 0.0303 | 1.0000 | 6/45 |
| JMs | 0.0166 | 0.6297 | -7.5324 | 11.8500 | -0.0063 | 1.0000 | 5/45 |

**generic**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0202 | 29.1951 | -9.7622 | 111.0049 | -0.2920 | 0.0000 | 5/45 |
| N0 | 0.0259 | 202.7301 | 85.0291 | 597.9282 | -2.0273 | 0.0000 | 4/45 |
| NM | 0.0258 | 133.6550 | 54.7046 | 355.9129 | -1.3365 | 0.0000 | 5/45 |
| JM | 0.0199 | -9.4146 | -31.3215 | 26.5391 | 0.0941 | 0.0000 | 6/45 |
| JMs | 0.0202 | 21.9246 | -13.1625 | 90.8018 | -0.2192 | 0.0000 | 5/45 |

**delta (median)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0209 | 33.8214 | -6.6813 | 117.6378 | -0.3382 | 0.0000 | 5/45 |
| N0 | 0.0266 | 210.5537 | 88.2865 | 630.6457 | -2.1055 | 0.0000 | 4/45 |
| NM | 0.0266 | 140.4561 | 57.2079 | 380.3860 | -1.4046 | 0.0000 | 5/45 |
| JM | 0.0199 | -9.5385 | -31.5528 | 25.2925 | 0.0954 | 0.0000 | 6/45 |
| JMs | 0.0206 | 24.4220 | -11.2515 | 92.0426 | -0.2442 | 0.0000 | 5/45 |

**delta (OOD kernel)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0170 | 8.8472 | -29.3873 | 90.4342 | -0.0885 | 0.4000 | 5/45 |
| N0 | 0.0214 | 150.5487 | 50.6479 | 477.1001 | -1.5055 | 0.2500 | 4/45 |
| NM | 0.0208 | 88.6306 | 16.6318 | 282.8039 | -0.8863 | 0.2000 | 5/45 |
| JM | 0.0180 | -18.2066 | -44.4544 | 28.3345 | 0.1821 | 0.3333 | 6/45 |
| JMs | 0.0171 | 3.5973 | -31.3554 | 72.2660 | -0.0360 | 0.4000 | 5/45 |

**delta (uniform)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0199 | 27.0771 | -12.3762 | 106.9904 | -0.2708 | 0.0000 | 5/45 |
| N0 | 0.0251 | 193.6994 | 74.7401 | 594.0364 | -1.9370 | 0.0000 | 4/45 |
| NM | 0.0251 | 126.9573 | 46.9693 | 355.4844 | -1.2696 | 0.0000 | 5/45 |
| JM | 0.0205 | -6.8527 | -31.2765 | 30.6522 | 0.0685 | 0.0000 | 6/45 |
| JMs | 0.0198 | 19.6442 | -16.2466 | 86.0517 | -0.1964 | 0.0000 | 5/45 |

**joint (direct)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0156 | 0.0513 | 0.0000 | 0.1951 | -0.0005 | 1.0000 | 5/45 |
| N0 | 0.0086 | 0.0937 | 0.0000 | 0.4222 | -0.0009 | 1.0000 | 4/45 |
| NM | 0.0111 | 0.0725 | 0.0000 | 0.2982 | -0.0007 | 1.0000 | 5/45 |
| JM | 0.0220 | -0.0365 | -0.1322 | 0.0000 | 0.0004 | 1.0000 | 6/45 |
| JMs | 0.0165 | 0.0485 | 0.0000 | 0.1820 | -0.0005 | 1.0000 | 5/45 |

**joint (calibrated)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0139 | -10.9833 | -18.0409 | -3.7837 | 0.1098 | 1.0000 | 5/45 |
| N0 | 0.0097 | 13.8744 | 1.5221 | 36.9376 | -0.1387 | 1.0000 | 4/45 |
| NM | 0.0120 | 8.6794 | -1.0744 | 22.5164 | -0.0868 | 1.0000 | 5/45 |
| JM | 0.0199 | -9.3318 | -14.3791 | -4.1978 | 0.0933 | 1.0000 | 6/45 |
| JMs | 0.0148 | -10.3844 | -16.9873 | -3.4845 | 0.1038 | 1.0000 | 5/45 |

**bidir profile-wall (cal.)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0143 | -8.3091 | -15.2613 | -0.6257 | 0.0831 | 1.0000 | 5/45 |
| N0 | 0.0103 | 20.4617 | 6.4685 | 48.1411 | -0.2046 | 1.0000 | 4/45 |
| NM | 0.0124 | 12.2016 | 0.5269 | 27.8769 | -0.1220 | 1.0000 | 5/45 |
| JM | 0.0201 | -8.5531 | -13.8019 | -3.5503 | 0.0855 | 1.0000 | 6/45 |
| JMs | 0.0152 | -8.0623 | -14.4294 | -0.7947 | 0.0806 | 1.0000 | 5/45 |
