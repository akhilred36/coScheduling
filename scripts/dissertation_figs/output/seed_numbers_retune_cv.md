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
| Absolute response | 0.1116 | 8.2957 | [-26.4, +39.8] | [-24.0, +42.1] | 0.0062 | 10.5798 | [-13.4, +32.1] | 0.0000 | 0.0000 | 0.2417 | 0.0000 | 0.0988 | 0.0000 | 0.4333 |
| Delta, uniform agg. | 0.1374 | 33.3924 | [-11.7, +75.1] | [-12.1, +76.9] | 0.0050 | 35.9218 | [+17.6, +54.5] | 0.0000 | 0.0000 | 0.2574 | 0.0556 | 0.1134 | 0.0000 | 0.8000 |
| Delta, median agg. | 0.1173 | 13.8271 | [-30.8, +54.1] | [-32.0, +55.3] | 0.0052 | 16.4076 | [-2.9, +37.4] | 0.0000 | 0.0000 | 0.2650 | 0.0444 | 0.0911 | 0.0000 | 0.4333 |
| Delta, OOD-kernel agg. | 0.1203 | 16.8008 | [-26.3, +59.2] | [-26.6, +60.7] | 0.0051 | 18.7454 | [+2.2, +38.3] | 0.0000 | 0.0000 | 0.3336 | 0.0222 | 0.1033 | 0.0000 | 0.4333 |
| Generic potential | 0.1509 | 46.5157 | [-1.6, +89.9] | [+0.4, +91.1] | 0.0047 | 47.3938 | [+28.7, +66.5] | 0.0000 | 0.0000 | 0.2442 | 0.0556 | 0.1262 | 0.0000 | 0.8000 |
| Joint | 0.1401 | 35.9567 | [-6.8, +75.0] | [-7.4, +76.3] | 0.0048 | 37.4322 | [+18.0, +58.2] | 0.0000 | 0.0000 | 0.1487 | 0.0000 | 0.1220 | 0.0000 | 0.8000 |
| Joint, calibrated | 0.1347 | 30.7672 | [-13.5, +70.7] | [-14.5, +71.8] | 0.0051 | 31.3649 | [+13.4, +52.7] | 0.0000 | 0.0000 | 0.2252 | 0.0000 | 0.1155 | 0.0000 | 0.7000 |
| Bidir. profile | 0.1397 | 35.6616 | [-5.1, +71.5] | [-4.1, +73.5] | 0.0046 | 36.1291 | [+20.5, +56.7] | 0.0000 | 0.0000 | 0.1368 | 0.0000 | 0.1238 | 0.0000 | 0.7667 |
| Bidir. profile, calibrated | 0.1325 | 28.5897 | [-13.7, +66.2] | [-13.1, +67.4] | 0.0034 | 29.1886 | [+15.8, +45.8] | 0.0000 | 0.0000 | 0.2247 | 0.0000 | 0.1144 | 0.0000 | 0.6667 |
| Bidir. context | 0.1220 | 18.4001 | [-15.4, +49.0] | [-23.7, +76.3] | 0.0238 | 21.6112 | [-15.7, +136.9] | 0.0000 | 0.0000 | 0.1880 | 0.0000 | 0.1113 | 0.0333 | 0.6000 |
| Bidir. context, calibrated | 0.1096 | 6.3890 | [-29.8, +39.5] | [-27.6, +63.9] | 0.0194 | 11.8767 | [-14.9, +121.8] | 0.0000 | 0.0000 | 0.2538 | 0.0111 | 0.0964 | 0.0333 | 0.6000 |
| Wall-clock aux (ablation), cal. | 0.1332 | 29.3026 | [-14.3, +68.6] | [-14.8, +70.2] | 0.0041 | 30.3821 | [+15.1, +50.6] | 0.0000 | 0.0000 | 0.2431 | 0.0000 | 0.1140 | 0.0000 | 0.7000 |
| Forward-only tokens (ablation), cal. | 0.1491 | 44.7097 | [+5.1, +81.0] | [-6.5, +103.6] | 0.0227 | 45.6681 | [-10.3, +162.8] | 0.0000 | 0.0000 | 0.1264 | 0.0000 | 0.1344 | 0.0333 | 0.6000 |

- BD-TT: blind run missing
### §8 claims (blind, native): full-N verdict and % of k-seed draws in which each holds
Verdict (claims with a deciding contrast): 'sign-unstable' when the deciding quantity's jackknife seed SE at full N exceeds 0.002 AND |estimate| < 2 SE (when |estimate| is within 1.9-2.1 SE the two-level CI decides instead). Otherwise: 'fails' if the claim is false at full N; 'holds' if true and the two-level (pairs×seeds) CI excludes 0; 'n.s.' if true but the CI includes 0. Claims without a deciding contrast: holds/fails at full N. 'jackknife vs CI disagree' = YES when the jackknife rule says sign-unstable but the two-level CI excludes 0; the seed-bootstrap SE (seeds resampled, pairs fixed) is shown next to the jackknife SE.
|  | full N | % of 1-seed draws | % of 5-seed draws | % of 10-seed draws | % of 15-seed draws | deciding quantity | seed SE (jackknife) | seed SE (seed bootstrap) | CI pairs×seeds | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'absolute < baseline') | False | 20.0000 | 20.9000 | 14.4000 | 6.4000 | 0.0085 | 0.0062 | 0.0061 | [-0.0248, +0.0433] | sign-unstable |  |  |
| ('BD-MPI', 'absolute at the floor on all outcomes') | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'delta_uniform > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0344 | 0.0050 | 0.0049 | [-0.0124, +0.0792] | n.s. |  |  |
| ('BD-MPI', 'delta_median > baseline') | True | 80.0000 | 94.3000 | 98.9000 | 99.9000 | 0.0142 | 0.0052 | 0.0051 | [-0.0329, +0.0570] | n.s. |  |  |
| ('BD-MPI', 'delta_ood_kernel > baseline') | True | 90.0000 | 99.3000 | 100.0000 | 100.0000 | 0.0173 | 0.0051 | 0.0048 | [-0.0274, +0.0625] | n.s. |  |  |
| ('BD-MPI', 'generic > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0479 | 0.0047 | 0.0046 | [+0.0005, +0.0938] | holds |  |  |
| ('BD-MPI', 'a delta variant has the lowest above-floor log MAE') | True | 50.0000 | 57.5000 | 65.7000 | 69.1000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'absolute Spearman > 0') | True | 90.0000 | 100.0000 | 100.0000 | 100.0000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'joint cal < baseline') | False | 6.6667 | 0.1000 | 0.0000 | 0.0000 | 0.0317 | 0.0051 | 0.0049 | [-0.0150, +0.0739] | fails |  |  |
| ('BD-MPI', 'bprof cal < baseline') | False | 3.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0295 | 0.0034 | 0.0035 | [-0.0135, +0.0694] | fails |  |  |
| ('BD-MPI', 'joint cal FB < absolute FB') | False | 16.6667 | 0.3000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'joint cal < absolute') | False | 40.0000 | 9.9000 | 0.9000 | 0.2000 | 0.0231 | 0.0083 | 0.0082 | [+0.0036, +0.0462] | fails |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, direct (bctx > bprof)') | False | 53.3333 | 38.7000 | 31.2000 | 23.4000 | -0.0178 | 0.0245 | 0.0223 | [-0.0572, +0.0306] | sign-unstable |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, calibrated') | False | 53.3333 | 36.3000 | 25.8000 | 16.6000 | -0.0229 | 0.0200 | 0.0183 | [-0.0490, +0.0257] | sign-unstable |  |  |
| ('BD-MPI', 'C7 joint direct worse than absolute') | True | 70.0000 | 95.4000 | 99.8000 | 100.0000 | 0.0285 | 0.0081 | 0.0081 | [+0.0103, +0.0506] | holds |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, direct (bprof < joint)') | True | 60.0000 | 55.7000 | 57.8000 | 55.6000 | -0.0003 | 0.0034 | 0.0033 | [-0.0105, +0.0086] | sign-unstable |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, calibrated') | True | 60.0000 | 65.6000 | 72.1000 | 76.5000 | -0.0022 | 0.0029 | 0.0028 | [-0.0105, +0.0056] | sign-unstable |  |  |
| ('BD-MPI', 'calibration helps joint (cal < direct)') | True | 80.0000 | 99.2000 | 100.0000 | 100.0000 | -0.0053 | 0.0014 | 0.0012 | [-0.0125, +0.0021] | n.s. |  |  |
| ('BD-MPI', 'calibration lowers FB (joint cal < joint direct)') | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |

### C7 ablation contrasts (blind, native; method − comparison)
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | 5-seed median | 5-seed 95% | % 5-seed draws same sign | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint direct vs legacy absolute') | +0.0285 | [+0.0160, +0.0396] | [+0.0103, +0.0506] | +0.0081 | +0.0081 | +0.0270 | [-0.0035, +0.0634] | +95.4000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Floor-aware calibration (joint cal − joint direct)') | -0.0053 | [-0.0118, +0.0018] | [-0.0125, +0.0021] | +0.0014 | +0.0012 | -0.0055 | [-0.0107, -0.0009] | +99.2000 | n.s. |  |  |
| ('BD-MPI', 'Reverse supervision, direct (bprof − joint)') | -0.0003 | [-0.0065, +0.0063] | [-0.0105, +0.0086] | +0.0034 | +0.0033 | -0.0009 | [-0.0128, +0.0142] | +55.7000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, calibrated') | -0.0022 | [-0.0079, +0.0038] | [-0.0105, +0.0056] | +0.0029 | +0.0028 | -0.0022 | [-0.0131, +0.0084] | +65.6000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, direct (bctx − bprof)') | -0.0178 | [-0.0254, -0.0094] | [-0.0572, +0.0306] | +0.0245 | +0.0223 | -0.0150 | [-0.0624, +0.1065] | +61.3000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, calibrated') | -0.0229 | [-0.0294, -0.0153] | [-0.0490, +0.0257] | +0.0200 | +0.0183 | -0.0166 | [-0.0531, +0.0972] | +63.7000 | sign-unstable |  |  |
| ('BD-MPI', 'Wall-clock-only aux, calibrated (bwall − bprof)') | +0.0007 | [-0.0029, +0.0038] | [-0.0036, +0.0057] | +0.0017 | +0.0017 | +0.0016 | [-0.0068, +0.0085] | +64.9000 | n.s. |  |  |
| ('BD-MPI', 'Forward-only tokens, calibrated (bfwd − bctx)') | +0.0395 | [+0.0312, +0.0477] | [-0.0150, +0.0902] | +0.0280 | +0.0263 | +0.0185 | [-0.0739, +0.1611] | +69.6000 | sign-unstable |  |  |
| ('BD-MPI', 'Legacy delta (uniform) vs legacy absolute') | +0.0259 | [+0.0064, +0.0452] | [+0.0022, +0.0505] | +0.0080 | +0.0080 | +0.0265 | [-0.0060, +0.0558] | +94.1000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Joint cal vs absolute') | +0.0231 | [+0.0088, +0.0369] | [+0.0036, +0.0462] | +0.0083 | +0.0082 | +0.0218 | [-0.0098, +0.0575] | +90.1000 | CI excludes 0 |  |  |

### C5 per-victim error ratio (victim log MAE ÷ baseline's), blind, native
#### Absolute response: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 0.97 (41%) |
| Beatnik | 0.23 (0%) |
| FIESTA | 2.71 (100%) |
| Kripke | 2.37 (99%) |
| Laghos | 0.74 (0%) |
| LAMMPS | 3.62 (100%) |
| MiniFE | 1.24 (96%) |
| miniVite | 0.79 (16%) |
| Quicksilver | 8.97 (100%) |
| TriC | 0.71 (19%) |

#### Joint, calibrated: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.36 (100%) |
| Beatnik | 0.27 (0%) |
| FIESTA | 2.53 (100%) |
| Kripke | 4.19 (100%) |
| Laghos | 0.77 (0%) |
| LAMMPS | 4.58 (100%) |
| MiniFE | 1.97 (100%) |
| miniVite | 0.84 (14%) |
| Quicksilver | 5.82 (100%) |
| TriC | 0.62 (3%) |

#### Delta, median agg.: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.17 (93%) |
| Beatnik | 0.21 (0%) |
| FIESTA | 2.58 (100%) |
| Kripke | 2.41 (100%) |
| Laghos | 0.75 (0%) |
| LAMMPS | 3.66 (100%) |
| MiniFE | 2.34 (100%) |
| miniVite | 0.41 (0%) |
| Quicksilver | 5.14 (100%) |
| TriC | 0.56 (0%) |

### Socket subsection (full data): paired run contrasts, seeds paired by id
#### information effect E1 − D1
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | E1 log MAE | D1 log MAE | E1 FB | D1 FB | E1 floor frac | D1 floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | -0.0025 | [-0.0040, -0.0008] | [-0.0052, +0.0005] | 0.0011 | 0.0011 | n.s. |  |  | -0.0021 | 79.1000 | 0.1206 | 0.1231 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | -0.0074 | [-0.0114, -0.0031] | [-0.0142, -0.0009] | 0.0025 | 0.0024 | CI excludes 0 |  |  | -0.0077 | 91.8000 | 0.1787 | 0.1861 | 0.0000 | 0.0000 | 0.0602 | 0.0602 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | -0.0073 | [-0.0127, -0.0015] | [-0.0155, +0.0000] | 0.0029 | 0.0027 | n.s. |  |  | -0.0075 | 88.5000 | 0.1430 | 0.1504 | 0.0000 | 0.0000 | 0.0482 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | -0.0100 | [-0.0140, -0.0059] | [-0.0160, -0.0043] | 0.0020 | 0.0020 | CI excludes 0 |  |  | -0.0094 | 98.8000 | 0.1821 | 0.1921 | 0.0000 | 0.0000 | 0.0602 | 0.0602 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0015 | [-0.0040, +0.0013] | [-0.0060, +0.0034] | 0.0018 | 0.0018 | n.s. |  |  | -0.0017 | 66.4000 | 0.1251 | 0.1265 | 0.0000 | 0.0000 | 0.0301 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0007 | [-0.0024, +0.0013] | [-0.0042, +0.0032] | 0.0015 | 0.0015 | n.s. |  |  | -0.0002 | 52.6000 | 0.1224 | 0.1231 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | -0.0031 | [-0.0047, -0.0013] | [-0.0108, +0.0054] | 0.0030 | 0.0037 | sign-unstable |  |  | -0.0020 | 68.6000 | 0.1037 | 0.1067 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | -0.0013 | [-0.0030, +0.0002] | [-0.0048, +0.0025] | 0.0017 | 0.0016 | n.s. |  |  | -0.0011 | 64.1000 | 0.1207 | 0.1220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | -0.0085 | [-0.0128, -0.0039] | [-0.0159, -0.0018] | 0.0028 | 0.0027 | CI excludes 0 |  |  | -0.0090 | 93.1000 | 0.1840 | 0.1925 | 0.0000 | 0.0000 | 0.0556 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | -0.0076 | [-0.0130, -0.0013] | [-0.0164, +0.0004] | 0.0031 | 0.0030 | n.s. |  |  | -0.0078 | 87.7000 | 0.1422 | 0.1498 | 0.0000 | 0.0000 | 0.0111 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | -0.0109 | [-0.0153, -0.0062] | [-0.0176, -0.0048] | 0.0021 | 0.0021 | CI excludes 0 |  |  | -0.0105 | 98.8000 | 0.1876 | 0.1985 | 0.0000 | 0.0000 | 0.0556 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | 0.0004 | [-0.0025, +0.0033] | [-0.0048, +0.0056] | 0.0021 | 0.0020 | sign-unstable |  |  | 0.0000 | 50.0000 | 0.1265 | 0.1261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | 0.0016 | [-0.0004, +0.0035] | [-0.0031, +0.0058] | 0.0020 | 0.0020 | sign-unstable |  |  | 0.0015 | 36.0000 | 0.1234 | 0.1218 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | -0.0025 | [-0.0041, -0.0007] | [-0.0120, +0.0083] | 0.0040 | 0.0045 | sign-unstable |  |  | -0.0012 | 61.2000 | 0.0981 | 0.1006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### init/architecture D1 − blind
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | D1 log MAE | D1 FB | D1 floor frac | blind log MAE | blind FB | blind floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | 0.0074 | [+0.0046, +0.0108] | [-0.0089, +0.0249] | 0.0084 | 0.0079 | sign-unstable |  |  | 0.0100 | 31.5000 | 0.1231 | 0.0000 | 0.0000 | 0.1156 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | 0.0451 | [+0.0359, +0.0559] | [+0.0305, +0.0616] | 0.0053 | 0.0050 | CI excludes 0 |  |  | 0.0441 | 0.0000 | 0.1861 | 0.0000 | 0.0602 | 0.1410 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | 0.0285 | [+0.0207, +0.0366] | [+0.0156, +0.0422] | 0.0053 | 0.0051 | CI excludes 0 |  |  | 0.0272 | 1.3000 | 0.1504 | 0.0000 | 0.0000 | 0.1219 | 0.0000 | 0.0482 |
| ('BD-MPI', 'resolved', 'Generic potential') | 0.0380 | [+0.0303, +0.0469] | [+0.0243, +0.0527] | 0.0057 | 0.0056 | CI excludes 0 |  |  | 0.0378 | 0.2000 | 0.1921 | 0.0000 | 0.0602 | 0.1540 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0083 | [-0.0118, -0.0051] | [-0.0211, +0.0037] | 0.0061 | 0.0058 | sign-unstable |  |  | -0.0083 | 72.2000 | 0.1265 | 0.0000 | 0.0000 | 0.1348 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0091 | [-0.0135, -0.0051] | [-0.0219, +0.0014] | 0.0054 | 0.0052 | sign-unstable |  |  | -0.0094 | 77.8000 | 0.1231 | 0.0000 | 0.0000 | 0.1322 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | -0.0077 | [-0.0119, -0.0034] | [-0.0462, +0.0353] | 0.0235 | 0.0199 | sign-unstable |  |  | -0.0029 | 55.5000 | 0.1067 | 0.0000 | 0.0000 | 0.1144 | 0.0000 | 0.0120 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | 0.0104 | [+0.0073, +0.0139] | [-0.0127, +0.0319] | 0.0111 | 0.0105 | sign-unstable |  |  | 0.0127 | 31.7000 | 0.1220 | 0.0000 | 0.0000 | 0.1116 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | 0.0551 | [+0.0434, +0.0668] | [+0.0386, +0.0725] | 0.0057 | 0.0056 | CI excludes 0 |  |  | 0.0540 | 0.0000 | 0.1925 | 0.0000 | 0.0556 | 0.1374 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | 0.0325 | [+0.0226, +0.0426] | [+0.0182, +0.0487] | 0.0059 | 0.0058 | CI excludes 0 |  |  | 0.0317 | 1.1000 | 0.1498 | 0.0000 | 0.0000 | 0.1173 | 0.0000 | 0.0444 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | 0.0476 | [+0.0370, +0.0582] | [+0.0327, +0.0638] | 0.0059 | 0.0058 | CI excludes 0 |  |  | 0.0471 | 0.0000 | 0.1985 | 0.0000 | 0.0556 | 0.1509 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | -0.0086 | [-0.0124, -0.0053] | [-0.0232, +0.0051] | 0.0070 | 0.0067 | sign-unstable |  |  | -0.0084 | 70.5000 | 0.1261 | 0.0000 | 0.0000 | 0.1347 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0107 | [-0.0153, -0.0063] | [-0.0258, +0.0023] | 0.0069 | 0.0066 | sign-unstable |  |  | -0.0104 | 75.6000 | 0.1218 | 0.0000 | 0.0000 | 0.1325 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | -0.0090 | [-0.0139, -0.0044] | [-0.0551, +0.0428] | 0.0306 | 0.0245 | sign-unstable |  |  | -0.0037 | 55.5000 | 0.1006 | 0.0000 | 0.0000 | 0.1096 | 0.0000 | 0.0111 |

#### Offset model (blind absolute + fixed socket offset), resolved cells
|  | absolute log MAE | offset log MAE | offset − absolute | offset floor frac | 5-seed median diff | % 5-seed draws offset better |
|---|---|---|---|---|---|---|
| ('BD-MPI', 'offset_jms') | 0.1156 | 0.1107 | -0.0049 | 0.0120 | -0.0047 | 100.0000 |
| ('BD-MPI', 'offset_all') | 0.1156 | 0.0978 | -0.0178 | 0.1084 | -0.0178 | 100.0000 |

### k-curve: Δ% vs baseline over random k-seed sub-ensembles (blind, native)
1000 random subsets without replacement per k (k = 1: every seed); k = N is the full ensemble. SD at k = 50 of N = 100 is shrunk by the finite-population factor (subsets overlap).
|  | mean Δ% | SD Δ% | median Δ% | 2.5% | 97.5% | % draws < 0 | mean floor frac |
|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint, calibrated', 1) | +38.15 | +23.87 | +33.92 | -1.66 | +82.19 | +6.67 | +0.04 |
| ('BD-MPI', 'Joint, calibrated', 5) | +32.28 | +10.24 | +32.16 | +14.15 | +53.37 | +0.00 | +0.01 |
| ('BD-MPI', 'Joint, calibrated', 10) | +31.36 | +6.63 | +31.28 | +18.86 | +44.57 | +0.00 | +0.01 |
| ('BD-MPI', 'Joint, calibrated', 25) | +30.79 | +2.22 | +30.83 | +26.30 | +34.89 | +0.00 | +0.00 |
| ('BD-MPI', 'Joint, calibrated', 30) | +30.77 | +0.00 | +30.77 | +30.77 | +30.77 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 1) | +34.77 | +19.52 | +28.20 | +6.38 | +72.61 | +3.33 | +0.02 |
| ('BD-MPI', 'Bidir. profile, calibrated', 5) | +29.69 | +7.85 | +29.50 | +14.79 | +45.58 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 10) | +29.17 | +4.82 | +29.00 | +20.39 | +39.09 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 25) | +28.58 | +1.41 | +28.57 | +25.78 | +31.16 | +0.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 30) | +28.59 | +0.00 | +28.59 | +28.59 | +28.59 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 1) | +21.38 | +20.08 | +23.19 | -13.84 | +54.93 | +20.00 | +0.04 |
| ('BD-MPI', 'Absolute response', 5) | +10.62 | +11.89 | +10.95 | -12.41 | +30.88 | +19.20 | +0.00 |
| ('BD-MPI', 'Absolute response', 10) | +9.08 | +8.31 | +9.18 | -7.18 | +24.99 | +14.10 | +0.00 |
| ('BD-MPI', 'Absolute response', 25) | +8.51 | +2.69 | +8.36 | +3.69 | +14.27 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 30) | +8.30 | +0.00 | +8.30 | +8.30 | +8.30 | +0.00 | +0.00 |

### Floor decomposition (blind, native): where the error comes from
Truth at floor = true slowdown < 1 clipped to 1 (y = 0, 'benign'); above floor = y > 0. Bias = mean(pred − truth) on log slowdown. Seeds 0–4 = the artifact's 5-seed ensemble.
|  | log MAE | bias | n at floor | log MAE at floor | n above | log MAE above | bias above | pred floor frac | pred floor frac on truth-at-floor | share of total abs error from above-floor |
|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'N = 30', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'N = 30', 'Absolute response') | 0.1116 | 0.0605 | 18.0000 | 0.1627 | 72.0000 | 0.0988 | 0.0350 | 0.0000 | 0.0000 | 0.7083 |
| ('BD-MPI', 'N = 30', 'Joint, calibrated') | 0.1347 | 0.1056 | 18.0000 | 0.2114 | 72.0000 | 0.1155 | 0.0792 | 0.0000 | 0.0000 | 0.6862 |
| ('BD-MPI', 'N = 30', 'Bidir. profile, calibrated') | 0.1325 | 0.0973 | 18.0000 | 0.2048 | 72.0000 | 0.1144 | 0.0704 | 0.0000 | 0.0000 | 0.6907 |
| ('BD-MPI', 'seeds 0–4', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'seeds 0–4', 'Absolute response') | 0.1216 | 0.0886 | 18.0000 | 0.1926 | 72.0000 | 0.1038 | 0.0626 | 0.0000 | 0.0000 | 0.6831 |
| ('BD-MPI', 'seeds 0–4', 'Joint, calibrated') | 0.1324 | 0.1008 | 18.0000 | 0.2102 | 72.0000 | 0.1129 | 0.0734 | 0.0111 | 0.0000 | 0.6825 |
| ('BD-MPI', 'seeds 0–4', 'Bidir. profile, calibrated') | 0.1304 | 0.0940 | 18.0000 | 0.2040 | 72.0000 | 0.1120 | 0.0665 | 0.0000 | 0.0000 | 0.6871 |

