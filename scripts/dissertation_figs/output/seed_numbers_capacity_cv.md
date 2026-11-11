## Seed batch, part (a): full data, frozen CV recipes, many final seeds

Runs: BD-MPI blind (30 seeds), BD-MPI E1 (30 seeds), BD-MPI D1 (30 seeds)

- Ensemble = mean of per-seed latents, then max(0, ·) (the method as designed); socket-aware runs per cell, then the rep-weighted slowdown mean.
- CIs: pairs-only = 45 pair clusters, the shared 1000 × seed-923 draws, seeds fixed; two-level = the same pair draws, each with a seed resample (with replacement). Seed SE = delete-one-seed jackknife.
- k-seed draws: 1000 random subsets without replacement (k = 1: every seed).

- 1W-MPI: blind run missing
- 1W-TT: blind run missing
### BD-MPI, blind, native outcomes (30 seeds; baseline log MAE 0.1030)
| method | log MAE | Δ% | Δ% CI pairs | Δ% CI pairs×seeds | seed SE | 5-seed Δ% median | 5-seed Δ% 95% | FB@1.05 | 5-seed FB median | Spearman | floor frac | above log MAE | 1-seed at floor everywhere | 1-seed blown up (Δ% > +25) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Absolute response | 0.1338 | 29.9299 | [-6.8, +64.0] | [-6.5, +63.3] | 0.0042 | 30.6952 | [+11.2, +44.8] | 0.0000 | 0.0000 | 0.1890 | 0.0000 | 0.1199 | 0.0000 | 0.8000 |
| Delta, uniform agg. | 0.1543 | 49.8210 | [+2.3, +93.7] | [-0.1, +94.8] | 0.0039 | 49.6283 | [+34.9, +65.8] | 0.0000 | 0.0000 | 0.2276 | 0.0556 | 0.1284 | 0.0000 | 0.9667 |
| Delta, median agg. | 0.1358 | 31.8550 | [-16.1, +75.8] | [-17.9, +75.9] | 0.0041 | 31.9856 | [+16.7, +49.1] | 0.0000 | 0.0000 | 0.2396 | 0.0222 | 0.1068 | 0.0000 | 0.6333 |
| Delta, OOD-kernel agg. | 0.1382 | 34.1708 | [-11.5, +77.3] | [-11.7, +78.7] | 0.0039 | 33.7955 | [+19.6, +50.3] | 0.0000 | 0.0000 | 0.2906 | 0.0222 | 0.1191 | 0.0000 | 0.6333 |
| Generic potential | 0.1448 | 40.5798 | [-6.8, +84.8] | [-9.3, +85.8] | 0.0028 | 41.3028 | [+29.4, +53.6] | 0.0000 | 0.0000 | 0.2405 | 0.0556 | 0.1176 | 0.0000 | 0.9667 |
| Joint | 0.1136 | 10.2975 | [-22.2, +40.7] | [-23.8, +48.5] | 0.0089 | 12.3301 | [-12.7, +43.5] | 0.0000 | 0.0000 | 0.1528 | 0.0000 | 0.1029 | 0.0000 | 0.6000 |
| Joint, calibrated | 0.1158 | 12.3788 | [-24.9, +47.5] | [-24.2, +50.3] | 0.0053 | 13.6749 | [-5.7, +34.4] | 0.0000 | 0.0000 | 0.2408 | 0.0000 | 0.1007 | 0.0000 | 0.4667 |
| Bidir. profile | 0.1081 | 4.9444 | [-24.0, +32.5] | [-25.7, +37.5] | 0.0079 | 9.3447 | [-15.8, +36.8] | 0.0000 | 0.0000 | 0.1881 | 0.0000 | 0.1001 | 0.0000 | 0.5000 |
| Bidir. profile, calibrated | 0.1141 | 10.7897 | [-24.3, +43.2] | [-26.1, +45.1] | 0.0053 | 12.5749 | [-8.0, +31.3] | 0.0000 | 0.0000 | 0.2646 | 0.0000 | 0.1010 | 0.0000 | 0.3667 |
| Bidir. context | 0.1528 | 48.3719 | [+6.6, +85.6] | [-3.5, +106.7] | 0.0194 | 46.5855 | [-6.6, +159.8] | 0.0000 | 0.0000 | 0.1644 | 0.0000 | 0.1367 | 0.0000 | 0.6667 |
| Bidir. context, calibrated | 0.1276 | 23.9020 | [-15.8, +61.3] | [-21.8, +80.1] | 0.0170 | 21.4186 | [-12.7, +127.1] | 0.0000 | 0.0000 | 0.2100 | 0.0000 | 0.1109 | 0.0000 | 0.4333 |
| Wall-clock aux (ablation), cal. | 0.1142 | 10.9067 | [-25.5, +45.2] | [-25.2, +48.4] | 0.0053 | 12.0051 | [-6.7, +33.4] | 0.0000 | 0.0000 | 0.2429 | 0.0000 | 0.0998 | 0.0000 | 0.3667 |
| Forward-only tokens (ablation), cal. | 0.1199 | 16.4392 | [-16.1, +47.0] | [-25.3, +78.6] | 0.0244 | 18.4048 | [-16.9, +152.8] | 0.0000 | 0.0000 | 0.2334 | 0.0000 | 0.1091 | 0.1000 | 0.5333 |

- BD-TT: blind run missing
### §8 claims (blind, native): full-N verdict and % of k-seed draws in which each holds
Verdict (claims with a deciding contrast): 'sign-unstable' when the deciding quantity's jackknife seed SE at full N exceeds 0.002 AND |estimate| < 2 SE (when |estimate| is within 1.9-2.1 SE the two-level CI decides instead). Otherwise: 'fails' if the claim is false at full N; 'holds' if true and the two-level (pairs×seeds) CI excludes 0; 'n.s.' if true but the CI includes 0. Claims without a deciding contrast: holds/fails at full N. 'jackknife vs CI disagree' = YES when the jackknife rule says sign-unstable but the two-level CI excludes 0; the seed-bootstrap SE (seeds resampled, pairs fixed) is shown next to the jackknife SE.
|  | full N | % of 1-seed draws | % of 5-seed draws | % of 10-seed draws | % of 15-seed draws | deciding quantity | seed SE (jackknife) | seed SE (seed bootstrap) | CI pairs×seeds | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'absolute < baseline') | False | 3.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0308 | 0.0042 | 0.0042 | [-0.0067, +0.0652] | fails |  |  |
| ('BD-MPI', 'absolute at the floor on all outcomes') | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'delta_uniform > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0513 | 0.0039 | 0.0039 | [-0.0001, +0.0977] | n.s. |  |  |
| ('BD-MPI', 'delta_median > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0328 | 0.0041 | 0.0041 | [-0.0184, +0.0782] | n.s. |  |  |
| ('BD-MPI', 'delta_ood_kernel > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0352 | 0.0039 | 0.0040 | [-0.0121, +0.0810] | n.s. |  |  |
| ('BD-MPI', 'generic > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0418 | 0.0028 | 0.0028 | [-0.0096, +0.0884] | n.s. |  |  |
| ('BD-MPI', 'a delta variant has the lowest above-floor log MAE') | False | 26.6667 | 13.5000 | 8.7000 | 5.6000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'absolute Spearman > 0') | True | 83.3333 | 98.9000 | 100.0000 | 100.0000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'joint cal < baseline') | False | 13.3333 | 11.4000 | 3.7000 | 0.2000 | 0.0128 | 0.0053 | 0.0054 | [-0.0249, +0.0518] | fails |  |  |
| ('BD-MPI', 'bprof cal < baseline') | False | 13.3333 | 13.3000 | 4.7000 | 0.7000 | 0.0111 | 0.0053 | 0.0051 | [-0.0269, +0.0464] | fails |  | boundary: CI decides |
| ('BD-MPI', 'joint cal FB < absolute FB') | False | 6.6667 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'joint cal < absolute') | True | 60.0000 | 90.3000 | 98.2000 | 100.0000 | -0.0181 | 0.0058 | 0.0058 | [-0.0339, -0.0021] | holds |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, direct (bctx > bprof)') | True | 63.3333 | 83.5000 | 94.4000 | 98.1000 | 0.0447 | 0.0215 | 0.0219 | [+0.0035, +0.0962] | holds |  | boundary: CI decides |
| ('BD-MPI', 'C7 fingerprints hurt, calibrated') | True | 46.6667 | 60.8000 | 67.3000 | 78.7000 | 0.0135 | 0.0182 | 0.0181 | [-0.0171, +0.0583] | sign-unstable |  |  |
| ('BD-MPI', 'C7 joint direct worse than absolute') | False | 53.3333 | 18.6000 | 5.8000 | 1.6000 | -0.0202 | 0.0088 | 0.0081 | [-0.0375, -0.0025] | fails |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, direct (bprof < joint)') | True | 56.6667 | 56.9000 | 60.4000 | 68.2000 | -0.0055 | 0.0112 | 0.0108 | [-0.0300, +0.0152] | sign-unstable |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, calibrated') | True | 56.6667 | 51.7000 | 54.2000 | 58.7000 | -0.0016 | 0.0072 | 0.0072 | [-0.0171, +0.0126] | sign-unstable |  |  |
| ('BD-MPI', 'calibration helps joint (cal < direct)') | False | 86.6667 | 46.4000 | 35.4000 | 28.0000 | 0.0021 | 0.0039 | 0.0031 | [-0.0096, +0.0145] | sign-unstable |  |  |
| ('BD-MPI', 'calibration lowers FB (joint cal < joint direct)') | False | 16.6667 | 3.4000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |

### C7 ablation contrasts (blind, native; method − comparison)
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | 5-seed median | 5-seed 95% | % 5-seed draws same sign | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint direct vs legacy absolute') | -0.0202 | [-0.0258, -0.0139] | [-0.0375, -0.0025] | +0.0088 | +0.0081 | -0.0174 | [-0.0446, +0.0145] | +81.4000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Floor-aware calibration (joint cal − joint direct)') | +0.0021 | [-0.0077, +0.0129] | [-0.0096, +0.0145] | +0.0039 | +0.0031 | +0.0007 | [-0.0110, +0.0098] | +53.6000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, direct (bprof − joint)') | -0.0055 | [-0.0097, -0.0010] | [-0.0300, +0.0152] | +0.0112 | +0.0108 | -0.0037 | [-0.0508, +0.0316] | +56.9000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, calibrated') | -0.0016 | [-0.0050, +0.0021] | [-0.0171, +0.0126] | +0.0072 | +0.0072 | -0.0005 | [-0.0330, +0.0244] | +51.7000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, direct (bctx − bprof)') | +0.0447 | [+0.0287, +0.0592] | [+0.0035, +0.0962] | +0.0215 | +0.0219 | +0.0372 | [-0.0239, +0.1526] | +83.5000 | CI excludes 0 |  | boundary: CI decides |
| ('BD-MPI', 'Fingerprints, calibrated') | +0.0135 | [+0.0043, +0.0219] | [-0.0171, +0.0583] | +0.0182 | +0.0181 | +0.0090 | [-0.0329, +0.1175] | +60.8000 | sign-unstable |  |  |
| ('BD-MPI', 'Wall-clock-only aux, calibrated (bwall − bprof)') | +0.0001 | [-0.0027, +0.0029] | [-0.0141, +0.0154] | +0.0073 | +0.0074 | -0.0012 | [-0.0270, +0.0322] | +47.4000 | sign-unstable |  |  |
| ('BD-MPI', 'Forward-only tokens, calibrated (bfwd − bctx)') | -0.0077 | [-0.0187, +0.0038] | [-0.0603, +0.0477] | +0.0286 | +0.0275 | -0.0065 | [-0.1079, +0.1385] | +55.4000 | sign-unstable |  |  |
| ('BD-MPI', 'Legacy delta (uniform) vs legacy absolute') | +0.0205 | [+0.0005, +0.0397] | [+0.0002, +0.0429] | +0.0042 | +0.0042 | +0.0209 | [+0.0036, +0.0394] | +99.5000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Joint cal vs absolute') | -0.0181 | [-0.0278, -0.0075] | [-0.0339, -0.0021] | +0.0058 | +0.0058 | -0.0165 | [-0.0398, +0.0057] | +90.3000 | CI excludes 0 |  |  |

### C5 per-victim error ratio (victim log MAE ÷ baseline's), blind, native
#### Absolute response: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.05 (70%) |
| Beatnik | 0.25 (0%) |
| FIESTA | 3.19 (100%) |
| Kripke | 3.02 (100%) |
| Laghos | 0.78 (0%) |
| LAMMPS | 4.57 (100%) |
| MiniFE | 1.40 (100%) |
| miniVite | 1.00 (51%) |
| Quicksilver | 11.89 (100%) |
| TriC | 1.27 (85%) |

#### Joint, calibrated: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.29 (100%) |
| Beatnik | 0.26 (0%) |
| FIESTA | 1.97 (100%) |
| Kripke | 3.18 (100%) |
| Laghos | 0.80 (0%) |
| LAMMPS | 3.17 (100%) |
| MiniFE | 1.74 (100%) |
| miniVite | 0.64 (0%) |
| Quicksilver | 5.51 (100%) |
| TriC | 0.49 (1%) |

#### Delta, median agg.: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.33 (100%) |
| Beatnik | 0.29 (0%) |
| FIESTA | 2.91 (100%) |
| Kripke | 2.84 (100%) |
| Laghos | 0.85 (0%) |
| LAMMPS | 4.08 (100%) |
| MiniFE | 2.58 (100%) |
| miniVite | 0.65 (0%) |
| Quicksilver | 5.83 (100%) |
| TriC | 0.74 (2%) |

### Socket subsection (full data): paired run contrasts, seeds paired by id
#### information effect E1 − D1
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | E1 log MAE | D1 log MAE | E1 FB | D1 FB | E1 floor frac | D1 floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | 0.0020 | [+0.0002, +0.0038] | [-0.0006, +0.0050] | 0.0012 | 0.0011 | n.s. |  |  | 0.0018 | 23.4000 | 0.1334 | 0.1314 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | -0.0044 | [-0.0074, -0.0014] | [-0.0082, -0.0004] | 0.0012 | 0.0012 | CI excludes 0 |  |  | -0.0047 | 93.7000 | 0.1782 | 0.1826 | 0.0000 | 0.0000 | 0.0602 | 0.0482 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | -0.0049 | [-0.0085, -0.0006] | [-0.0099, +0.0001] | 0.0015 | 0.0015 | n.s. |  |  | -0.0054 | 94.3000 | 0.1419 | 0.1468 | 0.0000 | 0.0000 | 0.0301 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | -0.0042 | [-0.0062, -0.0021] | [-0.0067, -0.0016] | 0.0009 | 0.0008 | CI excludes 0 |  |  | -0.0041 | 98.7000 | 0.1715 | 0.1758 | 0.0000 | 0.0000 | 0.0422 | 0.0241 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0004 | [-0.0012, +0.0005] | [-0.0020, +0.0011] | 0.0006 | 0.0006 | n.s. |  |  | -0.0005 | 63.3000 | 0.1181 | 0.1185 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0016 | [-0.0025, -0.0006] | [-0.0034, -0.0001] | 0.0006 | 0.0007 | CI excludes 0 |  |  | -0.0015 | 83.8000 | 0.1142 | 0.1158 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | -0.0040 | [-0.0056, -0.0024] | [-0.0117, +0.0031] | 0.0039 | 0.0037 | sign-unstable |  |  | -0.0019 | 64.8000 | 0.1233 | 0.1273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | 0.0048 | [+0.0027, +0.0065] | [+0.0013, +0.0086] | 0.0017 | 0.0015 | CI excludes 0 |  |  | 0.0041 | 8.5000 | 0.1368 | 0.1320 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | -0.0050 | [-0.0081, -0.0017] | [-0.0092, -0.0008] | 0.0014 | 0.0013 | CI excludes 0 |  |  | -0.0050 | 93.9000 | 0.1853 | 0.1903 | 0.0000 | 0.0000 | 0.0444 | 0.0444 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | -0.0055 | [-0.0093, -0.0013] | [-0.0108, -0.0002] | 0.0016 | 0.0015 | CI excludes 0 |  |  | -0.0056 | 93.1000 | 0.1435 | 0.1490 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | -0.0044 | [-0.0065, -0.0021] | [-0.0073, -0.0014] | 0.0009 | 0.0008 | CI excludes 0 |  |  | -0.0043 | 98.5000 | 0.1788 | 0.1832 | 0.0000 | 0.0000 | 0.0222 | 0.0222 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | 0.0003 | [-0.0006, +0.0014] | [-0.0015, +0.0020] | 0.0007 | 0.0007 | n.s. |  |  | 0.0002 | 44.2000 | 0.1169 | 0.1165 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0006 | [-0.0017, +0.0005] | [-0.0026, +0.0011] | 0.0008 | 0.0008 | n.s. |  |  | -0.0006 | 63.0000 | 0.1123 | 0.1128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | -0.0036 | [-0.0053, -0.0017] | [-0.0129, +0.0051] | 0.0047 | 0.0043 | sign-unstable |  |  | -0.0007 | 55.7000 | 0.1247 | 0.1283 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### init/architecture D1 − blind
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | D1 log MAE | D1 FB | D1 floor frac | blind log MAE | blind FB | blind floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | -0.0016 | [-0.0031, +0.0002] | [-0.0114, +0.0086] | 0.0048 | 0.0047 | sign-unstable |  |  | -0.0013 | 55.4000 | 0.1314 | 0.0000 | 0.0000 | 0.1330 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | 0.0266 | [+0.0189, +0.0351] | [+0.0130, +0.0400] | 0.0052 | 0.0053 | CI excludes 0 |  |  | 0.0271 | 0.5000 | 0.1826 | 0.0000 | 0.0482 | 0.1560 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | 0.0075 | [+0.0032, +0.0118] | [-0.0054, +0.0196] | 0.0054 | 0.0056 | sign-unstable |  |  | 0.0082 | 27.2000 | 0.1468 | 0.0000 | 0.0000 | 0.1393 | 0.0000 | 0.0241 |
| ('BD-MPI', 'resolved', 'Generic potential') | 0.0291 | [+0.0204, +0.0390] | [+0.0176, +0.0416] | 0.0036 | 0.0036 | CI excludes 0 |  |  | 0.0290 | 0.0000 | 0.1758 | 0.0000 | 0.0241 | 0.1467 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0008 | [-0.0038, +0.0023] | [-0.0158, +0.0133] | 0.0070 | 0.0069 | sign-unstable |  |  | -0.0015 | 54.5000 | 0.1185 | 0.0000 | 0.0000 | 0.1193 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0020 | [-0.0047, +0.0007] | [-0.0166, +0.0139] | 0.0073 | 0.0072 | sign-unstable |  |  | -0.0014 | 54.3000 | 0.1158 | 0.0000 | 0.0000 | 0.1178 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | 0.0002 | [-0.0024, +0.0026] | [-0.0474, +0.0459] | 0.0224 | 0.0228 | sign-unstable |  |  | 0.0050 | 45.4000 | 0.1273 | 0.0000 | 0.0000 | 0.1271 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | -0.0018 | [-0.0035, +0.0001] | [-0.0137, +0.0112] | 0.0063 | 0.0062 | sign-unstable |  |  | -0.0010 | 53.2000 | 0.1320 | 0.0000 | 0.0000 | 0.1338 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | 0.0360 | [+0.0264, +0.0457] | [+0.0211, +0.0500] | 0.0056 | 0.0056 | CI excludes 0 |  |  | 0.0366 | 0.0000 | 0.1903 | 0.0000 | 0.0444 | 0.1543 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | 0.0132 | [+0.0069, +0.0190] | [-0.0005, +0.0256] | 0.0059 | 0.0060 | n.s. |  |  | 0.0135 | 15.5000 | 0.1490 | 0.0000 | 0.0000 | 0.1358 | 0.0000 | 0.0222 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | 0.0383 | [+0.0277, +0.0488] | [+0.0253, +0.0512] | 0.0039 | 0.0039 | CI excludes 0 |  |  | 0.0383 | 0.0000 | 0.1832 | 0.0000 | 0.0222 | 0.1448 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | 0.0008 | [-0.0029, +0.0044] | [-0.0177, +0.0174] | 0.0085 | 0.0085 | sign-unstable |  |  | -0.0007 | 51.5000 | 0.1165 | 0.0000 | 0.0000 | 0.1158 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0013 | [-0.0046, +0.0016] | [-0.0205, +0.0179] | 0.0095 | 0.0092 | sign-unstable |  |  | -0.0002 | 50.6000 | 0.1128 | 0.0000 | 0.0000 | 0.1141 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | 0.0006 | [-0.0026, +0.0035] | [-0.0571, +0.0540] | 0.0278 | 0.0275 | sign-unstable |  |  | 0.0061 | 45.2000 | 0.1283 | 0.0000 | 0.0000 | 0.1276 | 0.0000 | 0.0000 |

#### Offset model (blind absolute + fixed socket offset), resolved cells
|  | absolute log MAE | offset log MAE | offset − absolute | offset floor frac | 5-seed median diff | % 5-seed draws offset better |
|---|---|---|---|---|---|---|
| ('BD-MPI', 'offset_jms') | 0.1330 | 0.1280 | -0.0050 | 0.0241 | -0.0049 | 100.0000 |
| ('BD-MPI', 'offset_all') | 0.1330 | 0.1128 | -0.0202 | 0.1145 | -0.0202 | 100.0000 |

### k-curve: Δ% vs baseline over random k-seed sub-ensembles (blind, native)
1000 random subsets without replacement per k (k = 1: every seed); k = N is the full ensemble. SD at k = 50 of N = 100 is shrunk by the finite-population factor (subsets overlap).
|  | mean Δ% | SD Δ% | median Δ% | 2.5% | 97.5% | % draws < 0 | mean floor frac |
|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint, calibrated', 1) | +21.90 | +19.54 | +21.59 | -11.74 | +57.85 | +13.33 | +0.03 |
| ('BD-MPI', 'Joint, calibrated', 5) | +13.90 | +11.18 | +14.18 | -6.97 | +35.33 | +12.20 | +0.00 |
| ('BD-MPI', 'Joint, calibrated', 10) | +12.81 | +7.25 | +12.68 | -0.69 | +27.21 | +3.20 | +0.00 |
| ('BD-MPI', 'Joint, calibrated', 25) | +12.45 | +2.39 | +12.35 | +8.41 | +17.84 | +0.00 | +0.00 |
| ('BD-MPI', 'Joint, calibrated', 30) | +12.38 | +0.00 | +12.38 | +12.38 | +12.38 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 1) | +19.74 | +19.24 | +20.94 | -12.92 | +50.07 | +13.33 | +0.04 |
| ('BD-MPI', 'Bidir. profile, calibrated', 5) | +12.82 | +10.49 | +12.99 | -7.89 | +33.17 | +12.50 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 10) | +11.15 | +7.01 | +11.18 | -1.60 | +24.86 | +5.40 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 25) | +10.83 | +2.32 | +10.59 | +6.67 | +15.66 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 30) | +10.79 | +0.00 | +10.79 | +10.79 | +10.79 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 1) | +34.14 | +19.34 | +36.75 | -2.96 | +62.95 | +3.33 | +0.00 |
| ('BD-MPI', 'Absolute response', 5) | +30.35 | +9.01 | +30.86 | +11.90 | +47.10 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 10) | +30.39 | +5.68 | +30.31 | +18.88 | +41.74 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 25) | +30.01 | +1.77 | +29.85 | +26.76 | +33.97 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 30) | +29.93 | +0.00 | +29.93 | +29.93 | +29.93 | +0.00 | +0.00 |

### Floor decomposition (blind, native): where the error comes from
Truth at floor = true slowdown < 1 clipped to 1 (y = 0, 'benign'); above floor = y > 0. Bias = mean(pred − truth) on log slowdown. Seeds 0–4 = the artifact's 5-seed ensemble.
|  | log MAE | bias | n at floor | log MAE at floor | n above | log MAE above | bias above | pred floor frac | pred floor frac on truth-at-floor | share of total abs error from above-floor |
|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'N = 30', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'N = 30', 'Absolute response') | 0.1338 | 0.0859 | 18.0000 | 0.1897 | 72.0000 | 0.1199 | 0.0599 | 0.0000 | 0.0000 | 0.7165 |
| ('BD-MPI', 'N = 30', 'Joint, calibrated') | 0.1158 | 0.0684 | 18.0000 | 0.1761 | 72.0000 | 0.1007 | 0.0415 | 0.0000 | 0.0000 | 0.6957 |
| ('BD-MPI', 'N = 30', 'Bidir. profile, calibrated') | 0.1141 | 0.0600 | 18.0000 | 0.1668 | 72.0000 | 0.1010 | 0.0333 | 0.0000 | 0.0000 | 0.7077 |
| ('BD-MPI', 'seeds 0–4', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'seeds 0–4', 'Absolute response') | 0.1396 | 0.0990 | 18.0000 | 0.2028 | 72.0000 | 0.1238 | 0.0730 | 0.0000 | 0.0000 | 0.7095 |
| ('BD-MPI', 'seeds 0–4', 'Joint, calibrated') | 0.1238 | 0.0817 | 18.0000 | 0.1901 | 72.0000 | 0.1072 | 0.0546 | 0.0000 | 0.0000 | 0.6928 |
| ('BD-MPI', 'seeds 0–4', 'Bidir. profile, calibrated') | 0.1254 | 0.0796 | 18.0000 | 0.1860 | 72.0000 | 0.1103 | 0.0530 | 0.0000 | 0.0000 | 0.7034 |

