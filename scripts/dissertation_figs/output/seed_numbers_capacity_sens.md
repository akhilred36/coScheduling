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
| Absolute response | 0.0910 | -11.6608 | [-18.8, -4.4] | [-23.0, -0.4] | 0.0050 | -7.1638 | [-17.7, +2.1] | 0.8621 | 0.6552 | 0.1592 | 0.0000 | 0.1055 | 0.1667 | 0.0667 |
| Delta, uniform agg. | 0.1504 | 45.9587 | [-3.1, +89.9] | [-2.9, +89.5] | 0.0019 | 46.2729 | [+39.6, +54.3] | 0.0000 | 0.0000 | 0.2406 | 0.0444 | 0.1232 | 0.0000 | 1.0000 |
| Delta, median agg. | 0.1126 | 9.3060 | [-28.3, +44.3] | [-28.0, +44.5] | 0.0018 | 9.6011 | [+4.0, +17.5] | 0.0000 | 0.0000 | 0.2513 | 0.0000 | 0.0878 | 0.0000 | 0.1000 |
| Delta, OOD-kernel agg. | 0.1228 | 19.1984 | [-25.6, +62.6] | [-25.9, +61.6] | 0.0018 | 19.3908 | [+13.1, +27.0] | 0.0000 | 0.0000 | 0.2972 | 0.0000 | 0.1024 | 0.0000 | 0.3000 |
| Generic potential | 0.1462 | 41.8968 | [-4.0, +84.2] | [-8.7, +85.9] | 0.0036 | 42.0614 | [+27.7, +59.5] | 0.0000 | 0.0000 | 0.2424 | 0.0556 | 0.1198 | 0.0000 | 0.9333 |
| Joint | 0.0894 | -13.2498 | [-22.2, -4.5] | [-25.0, +0.1] | 0.0048 | -6.7209 | [-19.8, +9.4] | 0.4138 | 0.4138 | 0.1010 | 0.0000 | 0.1013 | 0.2000 | 0.1667 |
| Joint, calibrated | 0.0926 | -10.1426 | [-34.5, +13.8] | [-34.7, +17.5] | 0.0030 | -6.9397 | [-15.2, +13.3] | 0.0000 | 0.0000 | 0.2624 | 0.0000 | 0.0878 | 0.0000 | 0.2000 |
| Bidir. profile | 0.0889 | -13.6741 | [-30.2, +2.8] | [-27.6, +14.0] | 0.0054 | -4.9062 | [-17.4, +36.5] | 0.0000 | 0.0000 | 0.1361 | 0.0000 | 0.0918 | 0.1333 | 0.3333 |
| Bidir. profile, calibrated | 0.0997 | -3.2402 | [-31.7, +23.8] | [-32.3, +29.3] | 0.0062 | -1.2792 | [-12.6, +32.1] | 0.0000 | 0.0000 | 0.2614 | 0.0000 | 0.0915 | 0.0000 | 0.3333 |
| Bidir. context | 0.0964 | -6.3921 | [-11.6, -1.4] | [-20.8, +1.0] | 0.0087 | -1.0732 | [-14.6, +31.0] | 0.8276 | 0.7241 | 0.1314 | 0.1444 | 0.1157 | 0.2000 | 0.2667 |
| Bidir. context, calibrated | 0.0912 | -11.4920 | [-33.7, +10.0] | [-32.5, +12.6] | 0.0020 | -7.1165 | [-14.2, +39.7] | 0.0345 | 0.0690 | 0.2501 | 0.1000 | 0.0911 | 0.0000 | 0.2667 |
| Wall-clock aux (ablation), cal. | 0.0920 | -10.6481 | [-34.4, +13.0] | [-34.7, +16.4] | 0.0026 | -7.4094 | [-15.4, +12.0] | 0.0000 | 0.0000 | 0.2604 | 0.0000 | 0.0878 | 0.0000 | 0.2000 |
| Forward-only tokens (ablation), cal. | 0.1167 | 13.2535 | [-23.9, +46.9] | [-30.8, +72.6] | 0.0216 | 17.8640 | [-18.4, +130.8] | 0.0000 | 0.0000 | 0.2260 | 0.0000 | 0.1020 | 0.0333 | 0.5000 |

- BD-TT: blind run missing
### §8 claims (blind, native): full-N verdict and % of k-seed draws in which each holds
Verdict (claims with a deciding contrast): 'sign-unstable' when the deciding quantity's jackknife seed SE at full N exceeds 0.002 AND |estimate| < 2 SE (when |estimate| is within 1.9-2.1 SE the two-level CI decides instead). Otherwise: 'fails' if the claim is false at full N; 'holds' if true and the two-level (pairs×seeds) CI excludes 0; 'n.s.' if true but the CI includes 0. Claims without a deciding contrast: holds/fails at full N. 'jackknife vs CI disagree' = YES when the jackknife rule says sign-unstable but the two-level CI excludes 0; the seed-bootstrap SE (seeds resampled, pairs fixed) is shown next to the jackknife SE.
|  | full N | % of 1-seed draws | % of 5-seed draws | % of 10-seed draws | % of 15-seed draws | deciding quantity | seed SE (jackknife) | seed SE (seed bootstrap) | CI pairs×seeds | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'absolute < baseline') | True | 36.6667 | 88.7000 | 98.0000 | 99.7000 | -0.0120 | 0.0050 | 0.0043 | [-0.0237, -0.0004] | holds |  |  |
| ('BD-MPI', 'absolute at the floor on all outcomes') | False | 16.6667 | 3.0000 | 0.4000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'delta_uniform > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0473 | 0.0019 | 0.0018 | [-0.0030, +0.0922] | n.s. |  |  |
| ('BD-MPI', 'delta_median > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0096 | 0.0018 | 0.0017 | [-0.0289, +0.0458] | n.s. |  |  |
| ('BD-MPI', 'delta_ood_kernel > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0198 | 0.0018 | 0.0017 | [-0.0267, +0.0634] | n.s. |  |  |
| ('BD-MPI', 'generic > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0432 | 0.0036 | 0.0037 | [-0.0089, +0.0885] | n.s. |  |  |
| ('BD-MPI', 'a delta variant has the lowest above-floor log MAE') | False | 73.3333 | 67.5000 | 67.2000 | 66.4000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'absolute Spearman > 0') | True | 50.0000 | 67.3000 | 76.1000 | 86.3000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'joint cal < baseline') | True | 46.6667 | 80.9000 | 93.8000 | 98.2000 | -0.0104 | 0.0030 | 0.0033 | [-0.0358, +0.0181] | n.s. |  |  |
| ('BD-MPI', 'bprof cal < baseline') | True | 43.3333 | 53.6000 | 60.7000 | 66.7000 | -0.0033 | 0.0062 | 0.0060 | [-0.0333, +0.0302] | sign-unstable |  |  |
| ('BD-MPI', 'joint cal FB < absolute FB') | True | 56.6667 | 78.3000 | 88.1000 | 94.4000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'joint cal < absolute') | False | 46.6667 | 46.7000 | 41.7000 | 39.2000 | 0.0016 | 0.0069 | 0.0062 | [-0.0240, +0.0254] | sign-unstable |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, direct (bctx > bprof)') | True | 46.6667 | 59.9000 | 66.5000 | 72.5000 | 0.0075 | 0.0102 | 0.0072 | [-0.0179, +0.0237] | sign-unstable |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, calibrated') | False | 46.6667 | 34.6000 | 18.3000 | 9.8000 | -0.0085 | 0.0063 | 0.0071 | [-0.0258, +0.0084] | sign-unstable |  |  |
| ('BD-MPI', 'C7 joint direct worse than absolute') | False | 53.3333 | 51.7000 | 51.7000 | 43.8000 | -0.0016 | 0.0058 | 0.0053 | [-0.0153, +0.0113] | sign-unstable |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, direct (bprof < joint)') | True | 43.3333 | 42.6000 | 45.2000 | 47.1000 | -0.0004 | 0.0063 | 0.0066 | [-0.0153, +0.0191] | sign-unstable |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, calibrated') | False | 46.6667 | 30.7000 | 21.0000 | 13.9000 | 0.0071 | 0.0066 | 0.0068 | [-0.0058, +0.0239] | sign-unstable |  |  |
| ('BD-MPI', 'calibration helps joint (cal < direct)') | False | 50.0000 | 40.9000 | 41.9000 | 35.2000 | 0.0032 | 0.0070 | 0.0062 | [-0.0212, +0.0226] | sign-unstable |  |  |
| ('BD-MPI', 'calibration lowers FB (joint cal < joint direct)') | True | 50.0000 | 66.4000 | 73.3000 | 76.5000 | – | – | – | – | holds | – | – |

### C7 ablation contrasts (blind, native; method − comparison)
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | 5-seed median | 5-seed 95% | % 5-seed draws same sign | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint direct vs legacy absolute') | -0.0016 | [-0.0041, +0.0011] | [-0.0153, +0.0113] | +0.0058 | +0.0053 | +0.0003 | [-0.0175, +0.0206] | +47.7000 | sign-unstable |  |  |
| ('BD-MPI', 'Floor-aware calibration (joint cal − joint direct)') | +0.0032 | [-0.0153, +0.0212] | [-0.0212, +0.0226] | +0.0070 | +0.0062 | +0.0030 | [-0.0136, +0.0136] | +59.1000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, direct (bprof − joint)') | -0.0004 | [-0.0090, +0.0079] | [-0.0153, +0.0191] | +0.0063 | +0.0066 | +0.0019 | [-0.0194, +0.0460] | +42.6000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, calibrated') | +0.0071 | [+0.0022, +0.0117] | [-0.0058, +0.0239] | +0.0066 | +0.0068 | +0.0059 | [-0.0224, +0.0416] | +69.3000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, direct (bctx − bprof)') | +0.0075 | [-0.0054, +0.0200] | [-0.0179, +0.0237] | +0.0102 | +0.0072 | +0.0020 | [-0.0417, +0.0402] | +59.9000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, calibrated') | -0.0085 | [-0.0168, -0.0000] | [-0.0258, +0.0084] | +0.0063 | +0.0071 | -0.0038 | [-0.0405, +0.0409] | +65.4000 | sign-unstable |  |  |
| ('BD-MPI', 'Wall-clock-only aux, calibrated (bwall − bprof)') | -0.0076 | [-0.0127, -0.0021] | [-0.0242, +0.0052] | +0.0064 | +0.0067 | -0.0061 | [-0.0415, +0.0207] | +71.4000 | sign-unstable |  |  |
| ('BD-MPI', 'Forward-only tokens, calibrated (bfwd − bctx)') | +0.0255 | [+0.0058, +0.0433] | [-0.0113, +0.0756] | +0.0211 | +0.0208 | +0.0206 | [-0.0224, +0.1362] | +74.1000 | sign-unstable |  |  |
| ('BD-MPI', 'Legacy delta (uniform) vs legacy absolute') | +0.0594 | [+0.0140, +0.1009] | [+0.0127, +0.0992] | +0.0048 | +0.0040 | +0.0554 | [+0.0434, +0.0661] | +100.0000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Joint cal vs absolute') | +0.0016 | [-0.0183, +0.0205] | [-0.0240, +0.0254] | +0.0069 | +0.0062 | +0.0010 | [-0.0156, +0.0260] | +53.3000 | sign-unstable |  |  |

### C5 per-victim error ratio (victim log MAE ÷ baseline's), blind, native
#### Absolute response: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 0.87 (6%) |
| Beatnik | 0.80 (0%) |
| FIESTA | 1.11 (76%) |
| Kripke | 0.60 (8%) |
| Laghos | 0.94 (7%) |
| LAMMPS | 1.24 (84%) |
| MiniFE | 1.03 (77%) |
| miniVite | 0.74 (0%) |
| Quicksilver | 2.38 (84%) |
| TriC | 0.53 (0%) |

#### Joint, calibrated: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.18 (93%) |
| Beatnik | 0.36 (0%) |
| FIESTA | 1.09 (81%) |
| Kripke | 1.79 (90%) |
| Laghos | 0.83 (0%) |
| LAMMPS | 1.39 (92%) |
| MiniFE | 1.60 (100%) |
| miniVite | 0.64 (0%) |
| Quicksilver | 1.26 (84%) |
| TriC | 0.31 (0%) |

#### Delta, median agg.: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.35 (100%) |
| Beatnik | 0.27 (0%) |
| FIESTA | 2.70 (100%) |
| Kripke | 1.01 (50%) |
| Laghos | 0.93 (1%) |
| LAMMPS | 2.46 (100%) |
| MiniFE | 2.27 (100%) |
| miniVite | 0.73 (0%) |
| Quicksilver | 1.91 (100%) |
| TriC | 0.32 (0%) |

### Socket subsection (full data): paired run contrasts, seeds paired by id
#### information effect E1 − D1
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | E1 log MAE | D1 log MAE | E1 FB | D1 FB | E1 floor frac | D1 floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | -0.0015 | [-0.0023, -0.0005] | [-0.0036, +0.0006] | 0.0015 | 0.0010 | n.s. |  |  | -0.0008 | 71.4000 | 0.0989 | 0.1004 | 0.0189 | 0.1132 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | -0.0022 | [-0.0029, -0.0013] | [-0.0042, -0.0003] | 0.0009 | 0.0009 | CI excludes 0 |  |  | -0.0019 | 82.8000 | 0.1703 | 0.1724 | 0.0000 | 0.0000 | 0.0241 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | -0.0023 | [-0.0034, -0.0010] | [-0.0046, -0.0002] | 0.0009 | 0.0009 | CI excludes 0 |  |  | -0.0021 | 84.9000 | 0.1274 | 0.1296 | 0.0000 | 0.0000 | 0.0120 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | -0.0064 | [-0.0094, -0.0032] | [-0.0102, -0.0023] | 0.0013 | 0.0012 | CI excludes 0 |  |  | -0.0060 | 99.3000 | 0.1735 | 0.1799 | 0.0000 | 0.0000 | 0.0602 | 0.0482 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | 0.0006 | [+0.0002, +0.0011] | [-0.0004, +0.0019] | 0.0005 | 0.0005 | n.s. |  |  | 0.0008 | 24.0000 | 0.1038 | 0.1031 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0006 | [-0.0012, -0.0001] | [-0.0023, +0.0005] | 0.0007 | 0.0006 | n.s. |  |  | -0.0007 | 68.4000 | 0.1074 | 0.1080 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | 0.0003 | [-0.0001, +0.0007] | [-0.0012, +0.0021] | 0.0007 | 0.0008 | n.s. |  |  | 0.0003 | 41.8000 | 0.1032 | 0.1029 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | -0.0008 | [-0.0018, +0.0001] | [-0.0035, +0.0014] | 0.0008 | 0.0010 | n.s. |  |  | -0.0005 | 65.2000 | 0.0879 | 0.0887 | 0.0000 | 0.1724 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | -0.0020 | [-0.0027, -0.0012] | [-0.0044, -0.0000] | 0.0010 | 0.0010 | CI excludes 0 |  |  | -0.0019 | 79.1000 | 0.1750 | 0.1770 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | -0.0023 | [-0.0034, -0.0012] | [-0.0047, -0.0001] | 0.0011 | 0.0010 | CI excludes 0 |  |  | -0.0021 | 82.5000 | 0.1235 | 0.1258 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | -0.0064 | [-0.0097, -0.0028] | [-0.0110, -0.0021] | 0.0012 | 0.0012 | CI excludes 0 |  |  | -0.0063 | 99.8000 | 0.1809 | 0.1873 | 0.0000 | 0.0000 | 0.0556 | 0.0444 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | 0.0006 | [+0.0001, +0.0011] | [-0.0005, +0.0018] | 0.0005 | 0.0005 | n.s. |  |  | 0.0005 | 31.2000 | 0.0961 | 0.0955 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0003 | [-0.0008, +0.0003] | [-0.0018, +0.0010] | 0.0007 | 0.0006 | n.s. |  |  | -0.0003 | 60.1000 | 0.1020 | 0.1023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | 0.0003 | [-0.0002, +0.0008] | [-0.0013, +0.0025] | 0.0008 | 0.0008 | n.s. |  |  | 0.0003 | 42.1000 | 0.0944 | 0.0941 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### init/architecture D1 − blind
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | D1 log MAE | D1 FB | D1 floor frac | blind log MAE | blind FB | blind floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | -0.0007 | [-0.0052, +0.0038] | [-0.0117, +0.0107] | 0.0033 | 0.0036 | sign-unstable |  |  | 0.0006 | 46.5000 | 0.1004 | 0.1132 | 0.0000 | 0.1011 | 0.8302 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | 0.0200 | [+0.0109, +0.0304] | [+0.0098, +0.0309] | 0.0019 | 0.0019 | CI excludes 0 |  |  | 0.0199 | 0.0000 | 0.1724 | 0.0000 | 0.0000 | 0.1525 | 0.0000 | 0.0482 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | 0.0129 | [+0.0048, +0.0216] | [+0.0036, +0.0223] | 0.0016 | 0.0016 | CI excludes 0 |  |  | 0.0130 | 0.0000 | 0.1296 | 0.0000 | 0.0000 | 0.1167 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | 0.0309 | [+0.0229, +0.0404] | [+0.0183, +0.0438] | 0.0048 | 0.0048 | CI excludes 0 |  |  | 0.0319 | 0.2000 | 0.1799 | 0.0000 | 0.0482 | 0.1490 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | 0.0019 | [-0.0004, +0.0042] | [-0.0101, +0.0170] | 0.0061 | 0.0060 | sign-unstable |  |  | 0.0008 | 47.2000 | 0.1031 | 0.0000 | 0.0000 | 0.1012 | 0.0189 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | 0.0015 | [-0.0015, +0.0044] | [-0.0140, +0.0161] | 0.0083 | 0.0072 | sign-unstable |  |  | 0.0006 | 48.1000 | 0.1080 | 0.0000 | 0.0000 | 0.1066 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | 0.0031 | [-0.0013, +0.0074] | [-0.0079, +0.0312] | 0.0065 | 0.0084 | sign-unstable |  |  | 0.0040 | 33.5000 | 0.1029 | 0.0000 | 0.0000 | 0.0998 | 0.0377 | 0.1024 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | -0.0023 | [-0.0074, +0.0028] | [-0.0159, +0.0130] | 0.0050 | 0.0055 | sign-unstable |  |  | 0.0006 | 47.0000 | 0.0887 | 0.1724 | 0.0000 | 0.0910 | 0.8621 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | 0.0267 | [+0.0173, +0.0367] | [+0.0158, +0.0384] | 0.0022 | 0.0022 | CI excludes 0 |  |  | 0.0267 | 0.0000 | 0.1770 | 0.0000 | 0.0000 | 0.1504 | 0.0000 | 0.0444 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | 0.0132 | [+0.0044, +0.0221] | [+0.0028, +0.0235] | 0.0023 | 0.0021 | CI excludes 0 |  |  | 0.0131 | 0.0000 | 0.1258 | 0.0000 | 0.0000 | 0.1126 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | 0.0411 | [+0.0308, +0.0514] | [+0.0271, +0.0555] | 0.0051 | 0.0051 | CI excludes 0 |  |  | 0.0419 | 0.0000 | 0.1873 | 0.0000 | 0.0444 | 0.1462 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | 0.0029 | [-0.0002, +0.0063] | [-0.0117, +0.0211] | 0.0082 | 0.0079 | sign-unstable |  |  | 0.0011 | 47.8000 | 0.0955 | 0.0000 | 0.0000 | 0.0926 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | 0.0026 | [-0.0010, +0.0060] | [-0.0169, +0.0225] | 0.0107 | 0.0095 | sign-unstable |  |  | 0.0016 | 47.2000 | 0.1023 | 0.0000 | 0.0000 | 0.0997 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | 0.0029 | [-0.0017, +0.0076] | [-0.0101, +0.0396] | 0.0101 | 0.0111 | sign-unstable |  |  | 0.0047 | 35.1000 | 0.0941 | 0.0000 | 0.0000 | 0.0912 | 0.0345 | 0.1000 |

#### Offset model (blind absolute + fixed socket offset), resolved cells
|  | absolute log MAE | offset log MAE | offset − absolute | offset floor frac | 5-seed median diff | % 5-seed draws offset better |
|---|---|---|---|---|---|---|
| ('BD-MPI', 'offset_jms') | 0.1011 | 0.0990 | -0.0021 | 0.0723 | -0.0018 | 92.2000 |
| ('BD-MPI', 'offset_all') | 0.1011 | 0.0983 | -0.0028 | 0.3614 | -0.0027 | 85.7000 |

### k-curve: Δ% vs baseline over random k-seed sub-ensembles (blind, native)
1000 random subsets without replacement per k (k = 1: every seed); k = N is the full ensemble. SD at k = 50 of N = 100 is shrunk by the finite-population factor (subsets overlap).
|  | mean Δ% | SD Δ% | median Δ% | 2.5% | 97.5% | % draws < 0 | mean floor frac |
|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint, calibrated', 1) | +8.56 | +17.21 | +3.36 | -11.76 | +44.54 | +46.67 | +0.16 |
| ('BD-MPI', 'Joint, calibrated', 5) | -5.04 | +7.65 | -6.95 | -15.32 | +14.92 | +79.40 | +0.07 |
| ('BD-MPI', 'Joint, calibrated', 10) | -7.90 | +4.56 | -8.75 | -14.42 | +4.01 | +93.60 | +0.04 |
| ('BD-MPI', 'Joint, calibrated', 25) | -9.81 | +1.32 | -9.99 | -11.75 | -6.62 | +100.00 | +0.00 |
| ('BD-MPI', 'Joint, calibrated', 30) | -10.14 | +0.00 | -10.14 | -10.14 | -10.14 | +100.00 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 1) | +16.31 | +28.98 | +6.09 | -10.10 | +84.84 | +43.33 | +0.12 |
| ('BD-MPI', 'Bidir. profile, calibrated', 5) | +2.72 | +13.04 | -0.83 | -12.72 | +34.89 | +52.50 | +0.04 |
| ('BD-MPI', 'Bidir. profile, calibrated', 10) | -0.79 | +8.36 | -2.30 | -11.34 | +19.61 | +61.20 | +0.02 |
| ('BD-MPI', 'Bidir. profile, calibrated', 25) | -2.95 | +2.75 | -3.24 | -7.69 | +2.83 | +84.60 | +0.00 |
| ('BD-MPI', 'Bidir. profile, calibrated', 30) | -3.24 | +0.00 | -3.24 | -3.24 | -3.24 | +100.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 1) | +4.09 | +12.70 | +0.00 | -10.55 | +34.95 | +36.67 | +0.36 |
| ('BD-MPI', 'Absolute response', 5) | -7.08 | +6.03 | -7.27 | -17.61 | +3.14 | +88.10 | +0.25 |
| ('BD-MPI', 'Absolute response', 10) | -9.28 | +4.93 | -10.06 | -16.89 | -0.05 | +98.00 | +0.14 |
| ('BD-MPI', 'Absolute response', 25) | -11.20 | +2.22 | -11.42 | -14.85 | -6.58 | +100.00 | +0.01 |
| ('BD-MPI', 'Absolute response', 30) | -11.66 | +0.00 | -11.66 | -11.66 | -11.66 | +100.00 | +0.00 |

### Floor decomposition (blind, native): where the error comes from
Truth at floor = true slowdown < 1 clipped to 1 (y = 0, 'benign'); above floor = y > 0. Bias = mean(pred − truth) on log slowdown. Seeds 0–4 = the artifact's 5-seed ensemble.
|  | log MAE | bias | n at floor | log MAE at floor | n above | log MAE above | bias above | pred floor frac | pred floor frac on truth-at-floor | share of total abs error from above-floor |
|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'N = 30', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'N = 30', 'Absolute response') | 0.0910 | -0.0699 | 18.0000 | 0.0331 | 72.0000 | 0.1055 | -0.0956 | 0.0000 | 0.0000 | 0.9273 |
| ('BD-MPI', 'N = 30', 'Joint, calibrated') | 0.0926 | 0.0050 | 18.0000 | 0.1118 | 72.0000 | 0.0878 | -0.0217 | 0.0000 | 0.0000 | 0.7584 |
| ('BD-MPI', 'N = 30', 'Bidir. profile, calibrated') | 0.0997 | 0.0257 | 18.0000 | 0.1322 | 72.0000 | 0.0915 | -0.0009 | 0.0000 | 0.0000 | 0.7348 |
| ('BD-MPI', 'seeds 0–4', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'seeds 0–4', 'Absolute response') | 0.0871 | -0.0338 | 18.0000 | 0.0670 | 72.0000 | 0.0921 | -0.0589 | 0.0000 | 0.0000 | 0.8461 |
| ('BD-MPI', 'seeds 0–4', 'Joint, calibrated') | 0.0938 | 0.0101 | 18.0000 | 0.1183 | 72.0000 | 0.0877 | -0.0169 | 0.0000 | 0.0000 | 0.7477 |
| ('BD-MPI', 'seeds 0–4', 'Bidir. profile, calibrated') | 0.1120 | 0.0495 | 18.0000 | 0.1569 | 72.0000 | 0.1008 | 0.0226 | 0.0000 | 0.0000 | 0.7199 |

