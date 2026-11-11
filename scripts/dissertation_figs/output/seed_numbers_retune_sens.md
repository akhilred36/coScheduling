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
| Absolute response | 0.1116 | 8.3075 | [-26.6, +39.9] | [-24.5, +42.3] | 0.0061 | 10.5957 | [-12.7, +32.1] | 0.0000 | 0.0000 | 0.2411 | 0.0000 | 0.0985 | 0.0000 | 0.4667 |
| Delta, uniform agg. | 0.1428 | 38.6298 | [-7.6, +82.0] | [-8.1, +84.1] | 0.0027 | 40.0749 | [+29.2, +51.0] | 0.0000 | 0.0000 | 0.2548 | 0.0556 | 0.1160 | 0.0000 | 0.9333 |
| Delta, median agg. | 0.1167 | 13.2854 | [-31.9, +51.1] | [-31.0, +52.7] | 0.0031 | 14.3266 | [+2.1, +28.1] | 0.0000 | 0.0000 | 0.2692 | 0.0000 | 0.0898 | 0.0000 | 0.3667 |
| Delta, OOD-kernel agg. | 0.1226 | 18.9827 | [-24.4, +61.3] | [-23.6, +62.2] | 0.0031 | 19.8928 | [+8.2, +32.4] | 0.0000 | 0.0000 | 0.3428 | 0.0111 | 0.1036 | 0.0000 | 0.5000 |
| Generic potential | 0.1474 | 43.1375 | [-4.9, +87.5] | [-6.5, +88.2] | 0.0032 | 43.3386 | [+32.6, +59.5] | 0.0000 | 0.0000 | 0.2506 | 0.0556 | 0.1201 | 0.0000 | 1.0000 |
| Joint | 0.0968 | -6.0512 | [-9.3, -2.7] | [-19.7, +17.5] | 0.0166 | 0.0000 | [-15.2, +104.7] | 1.0000 | 0.9655 | -0.1487 | 0.0778 | 0.1172 | 0.3000 | 0.3333 |
| Joint, calibrated | 0.0937 | -9.0184 | [-30.7, +12.6] | [-28.6, +23.3] | 0.0033 | -3.7150 | [-13.6, +61.1] | 0.0345 | 0.0345 | 0.1650 | 0.1444 | 0.0937 | 0.0667 | 0.3000 |
| Bidir. profile | 0.0970 | -5.8715 | [-9.1, -2.7] | [-19.6, +17.0] | 0.0165 | 0.0000 | [-15.2, +104.8] | 1.0000 | 0.9655 | -0.1571 | 0.0889 | 0.1175 | 0.3000 | 0.3333 |
| Bidir. profile, calibrated | 0.0937 | -9.0147 | [-30.7, +12.6] | [-28.5, +23.1] | 0.0032 | -3.7326 | [-13.4, +61.4] | 0.0345 | 0.0345 | 0.1621 | 0.1444 | 0.0938 | 0.0667 | 0.3000 |
| Bidir. context | 0.1020 | -0.9826 | [-1.7, -0.3] | [-18.8, +7.3] | 0.0125 | 0.0000 | [-13.8, +105.6] | 1.0000 | 1.0000 | -0.0312 | 0.6889 | 0.1271 | 0.3333 | 0.2333 |
| Bidir. context, calibrated | 0.0921 | -10.6381 | [-31.1, +10.0] | [-29.4, +19.7] | 0.0041 | -3.0742 | [-15.5, +72.3] | 0.1034 | 0.1034 | 0.2010 | 0.2000 | 0.0936 | 0.1000 | 0.3000 |
| Wall-clock aux (ablation), cal. | 0.0937 | -9.0022 | [-30.7, +12.6] | [-28.3, +23.1] | 0.0032 | -3.7386 | [-13.4, +61.2] | 0.0345 | 0.0345 | 0.1622 | 0.1444 | 0.0938 | 0.0667 | 0.3333 |
| Forward-only tokens (ablation), cal. | 0.0899 | -12.7369 | [-21.4, -2.8] | [-24.0, +29.9] | 0.0055 | 0.0000 | [-18.8, +105.4] | 0.4138 | 0.2586 | 0.0405 | 0.0000 | 0.1015 | 0.3000 | 0.4667 |

- BD-TT: blind run missing
### §8 claims (blind, native): full-N verdict and % of k-seed draws in which each holds
Verdict (claims with a deciding contrast): 'sign-unstable' when the deciding quantity's jackknife seed SE at full N exceeds 0.002 AND |estimate| < 2 SE (when |estimate| is within 1.9-2.1 SE the two-level CI decides instead). Otherwise: 'fails' if the claim is false at full N; 'holds' if true and the two-level (pairs×seeds) CI excludes 0; 'n.s.' if true but the CI includes 0. Claims without a deciding contrast: holds/fails at full N. 'jackknife vs CI disagree' = YES when the jackknife rule says sign-unstable but the two-level CI excludes 0; the seed-bootstrap SE (seeds resampled, pairs fixed) is shown next to the jackknife SE.
|  | full N | % of 1-seed draws | % of 5-seed draws | % of 10-seed draws | % of 15-seed draws | deciding quantity | seed SE (jackknife) | seed SE (seed bootstrap) | CI pairs×seeds | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'absolute < baseline') | False | 20.0000 | 20.4000 | 13.5000 | 5.9000 | 0.0086 | 0.0061 | 0.0059 | [-0.0253, +0.0436] | sign-unstable |  |  |
| ('BD-MPI', 'absolute at the floor on all outcomes') | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'delta_uniform > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0398 | 0.0027 | 0.0027 | [-0.0083, +0.0866] | n.s. |  |  |
| ('BD-MPI', 'delta_median > baseline') | True | 90.0000 | 99.4000 | 100.0000 | 100.0000 | 0.0137 | 0.0031 | 0.0031 | [-0.0319, +0.0543] | n.s. |  |  |
| ('BD-MPI', 'delta_ood_kernel > baseline') | True | 96.6667 | 100.0000 | 100.0000 | 100.0000 | 0.0196 | 0.0031 | 0.0030 | [-0.0243, +0.0641] | n.s. |  |  |
| ('BD-MPI', 'generic > baseline') | True | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0444 | 0.0032 | 0.0032 | [-0.0067, +0.0908] | n.s. |  |  |
| ('BD-MPI', 'a delta variant has the lowest above-floor log MAE') | True | 60.0000 | 69.2000 | 77.0000 | 86.6000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'absolute Spearman > 0') | True | 93.3333 | 100.0000 | 100.0000 | 100.0000 | – | – | – | – | holds | – | – |
| ('BD-MPI', 'joint cal < baseline') | True | 43.3333 | 68.5000 | 78.0000 | 86.4000 | -0.0093 | 0.0033 | 0.0067 | [-0.0294, +0.0240] | n.s. |  |  |
| ('BD-MPI', 'bprof cal < baseline') | True | 43.3333 | 68.5000 | 78.1000 | 86.6000 | -0.0093 | 0.0032 | 0.0067 | [-0.0293, +0.0238] | n.s. |  |  |
| ('BD-MPI', 'joint cal FB < absolute FB') | False | 6.6667 | 0.0000 | 0.0000 | 0.0000 | – | – | – | – | fails | – | – |
| ('BD-MPI', 'joint cal < absolute') | True | 56.6667 | 72.9000 | 88.2000 | 96.3000 | -0.0178 | 0.0063 | 0.0079 | [-0.0380, +0.0107] | n.s. |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, direct (bctx > bprof)') | True | 40.0000 | 40.2000 | 39.6000 | 45.6000 | 0.0050 | 0.0121 | 0.0067 | [-0.0151, +0.0166] | sign-unstable |  |  |
| ('BD-MPI', 'C7 fingerprints hurt, calibrated') | False | 53.3333 | 54.4000 | 50.4000 | 38.5000 | -0.0017 | 0.0031 | 0.0055 | [-0.0152, +0.0107] | sign-unstable |  |  |
| ('BD-MPI', 'C7 joint direct worse than absolute') | False | 46.6667 | 32.1000 | 16.9000 | 5.6000 | -0.0148 | 0.0188 | 0.0094 | [-0.0428, +0.0224] | sign-unstable |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, direct (bprof < joint)') | False | 33.3333 | 32.3000 | 33.4000 | 33.0000 | 0.0002 | 0.0003 | 0.0001 | [-0.0003, +0.0003] | fails |  |  |
| ('BD-MPI', 'C7 reverse supervision helps, calibrated') | False | 33.3333 | 50.5000 | 55.3000 | 59.4000 | 0.0000 | 0.0001 | 0.0001 | [-0.0002, +0.0002] | fails |  |  |
| ('BD-MPI', 'calibration helps joint (cal < direct)') | True | 70.0000 | 68.0000 | 63.6000 | 57.6000 | -0.0031 | 0.0193 | 0.0072 | [-0.0244, +0.0193] | sign-unstable |  |  |
| ('BD-MPI', 'calibration lowers FB (joint cal < joint direct)') | True | 46.6667 | 65.9000 | 71.6000 | 76.2000 | – | – | – | – | holds | – | – |

### C7 ablation contrasts (blind, native; method − comparison)
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | 5-seed median | 5-seed 95% | % 5-seed draws same sign | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint direct vs legacy absolute') | -0.0148 | [-0.0446, +0.0193] | [-0.0428, +0.0224] | +0.0188 | +0.0094 | -0.0086 | [-0.0371, +0.0905] | +67.9000 | sign-unstable |  |  |
| ('BD-MPI', 'Floor-aware calibration (joint cal − joint direct)') | -0.0031 | [-0.0231, +0.0170] | [-0.0244, +0.0193] | +0.0193 | +0.0072 | -0.0038 | [-0.0436, +0.0109] | +68.0000 | sign-unstable |  |  |
| ('BD-MPI', 'Reverse supervision, direct (bprof − joint)') | +0.0002 | [+0.0000, +0.0003] | [-0.0003, +0.0003] | +0.0003 | +0.0001 | +0.0000 | [-0.0008, +0.0005] | +35.8000 | n.s. |  |  |
| ('BD-MPI', 'Reverse supervision, calibrated') | +0.0000 | [-0.0001, +0.0001] | [-0.0002, +0.0002] | +0.0001 | +0.0001 | -0.0000 | [-0.0005, +0.0003] | +49.4000 | n.s. |  |  |
| ('BD-MPI', 'Fingerprints, direct (bctx − bprof)') | +0.0050 | [+0.0022, +0.0078] | [-0.0151, +0.0166] | +0.0121 | +0.0067 | +0.0000 | [-0.0548, +0.0467] | +40.2000 | sign-unstable |  |  |
| ('BD-MPI', 'Fingerprints, calibrated') | -0.0017 | [-0.0035, +0.0001] | [-0.0152, +0.0107] | +0.0031 | +0.0055 | +0.0008 | [-0.0359, +0.0388] | +45.5000 | sign-unstable |  |  |
| ('BD-MPI', 'Wall-clock-only aux, calibrated (bwall − bprof)') | +0.0000 | [-0.0001, +0.0001] | [-0.0002, +0.0002] | +0.0001 | +0.0001 | -0.0000 | [-0.0005, +0.0006] | +47.0000 | n.s. |  |  |
| ('BD-MPI', 'Forward-only tokens, calibrated (bfwd − bctx)') | -0.0022 | [-0.0155, +0.0111] | [-0.0197, +0.0280] | +0.0063 | +0.0092 | +0.0026 | [-0.0446, +0.0907] | +35.8000 | sign-unstable |  |  |
| ('BD-MPI', 'Legacy delta (uniform) vs legacy absolute') | +0.0312 | [+0.0115, +0.0512] | [+0.0080, +0.0553] | +0.0063 | +0.0064 | +0.0299 | [+0.0053, +0.0558] | +99.4000 | CI excludes 0 |  |  |
| ('BD-MPI', 'Joint cal vs absolute') | -0.0178 | [-0.0350, +0.0003] | [-0.0380, +0.0107] | +0.0063 | +0.0079 | -0.0109 | [-0.0382, +0.0482] | +72.9000 | n.s. |  |  |

### C5 per-victim error ratio (victim log MAE ÷ baseline's), blind, native
#### Absolute response: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 0.98 (42%) |
| Beatnik | 0.23 (0%) |
| FIESTA | 2.71 (100%) |
| Kripke | 2.40 (99%) |
| Laghos | 0.74 (0%) |
| LAMMPS | 3.66 (100%) |
| MiniFE | 1.25 (96%) |
| miniVite | 0.80 (16%) |
| Quicksilver | 8.76 (100%) |
| TriC | 0.69 (17%) |

#### Joint, calibrated: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.17 (76%) |
| Beatnik | 0.41 (0%) |
| FIESTA | 1.00 (47%) |
| Kripke | 1.60 (70%) |
| Laghos | 0.88 (3%) |
| LAMMPS | 1.24 (76%) |
| MiniFE | 1.61 (100%) |
| miniVite | 0.72 (11%) |
| Quicksilver | 0.71 (46%) |
| TriC | 0.39 (6%) |

#### Delta, median agg.: ratio at full N (and % of 5-seed draws with ratio > 1)
| victim | BD-MPI |
|---|---|
| AMG2023 | 1.21 (100%) |
| Beatnik | 0.27 (0%) |
| FIESTA | 2.53 (100%) |
| Kripke | 1.87 (100%) |
| Laghos | 0.87 (0%) |
| LAMMPS | 3.11 (100%) |
| MiniFE | 2.30 (100%) |
| miniVite | 0.51 (0%) |
| Quicksilver | 4.64 (100%) |
| TriC | 0.47 (0%) |

### Socket subsection (full data): paired run contrasts, seeds paired by id
#### information effect E1 − D1
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | E1 log MAE | D1 log MAE | E1 FB | D1 FB | E1 floor frac | D1 floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | -0.0029 | [-0.0044, -0.0012] | [-0.0056, +0.0001] | 0.0011 | 0.0011 | n.s. |  |  | -0.0023 | 82.4000 | 0.1207 | 0.1236 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | -0.0030 | [-0.0044, -0.0014] | [-0.0061, +0.0010] | 0.0016 | 0.0015 | n.s. |  |  | -0.0038 | 81.4000 | 0.1760 | 0.1790 | 0.0000 | 0.0000 | 0.0482 | 0.0361 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | -0.0033 | [-0.0052, -0.0012] | [-0.0069, +0.0013] | 0.0018 | 0.0017 | n.s. |  |  | -0.0042 | 82.0000 | 0.1372 | 0.1405 | 0.0000 | 0.0000 | 0.0060 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | -0.0014 | [-0.0025, -0.0002] | [-0.0033, +0.0005] | 0.0009 | 0.0008 | n.s. |  |  | -0.0013 | 73.5000 | 0.1779 | 0.1794 | 0.0000 | 0.0000 | 0.0361 | 0.0120 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0001 | [-0.0003, +0.0002] | [-0.0008, +0.0007] | 0.0004 | 0.0003 | n.s. |  |  | -0.0000 | 51.7000 | 0.0996 | 0.0997 | 0.2453 | 0.2453 | 0.2289 | 0.2289 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0000 | [-0.0003, +0.0002] | [-0.0008, +0.0007] | 0.0004 | 0.0003 | n.s. |  |  | -0.0000 | 51.8000 | 0.0997 | 0.0998 | 0.2453 | 0.2453 | 0.2289 | 0.2289 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | 0.0002 | [-0.0000, +0.0005] | [-0.0006, +0.0013] | 0.0005 | 0.0004 | n.s. |  |  | 0.0000 | 37.8000 | 0.1040 | 0.1037 | 0.4528 | 0.4717 | 0.5060 | 0.5120 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | -0.0018 | [-0.0035, -0.0001] | [-0.0053, +0.0019] | 0.0016 | 0.0015 | n.s. |  |  | -0.0014 | 66.5000 | 0.1210 | 0.1228 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | -0.0029 | [-0.0044, -0.0012] | [-0.0064, +0.0014] | 0.0019 | 0.0018 | n.s. |  |  | -0.0041 | 80.4000 | 0.1831 | 0.1860 | 0.0000 | 0.0000 | 0.0333 | 0.0333 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | -0.0025 | [-0.0044, -0.0003] | [-0.0069, +0.0025] | 0.0021 | 0.0020 | sign-unstable |  |  | -0.0039 | 80.3000 | 0.1372 | 0.1397 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | -0.0010 | [-0.0021, +0.0001] | [-0.0034, +0.0011] | 0.0010 | 0.0010 | n.s. |  |  | -0.0010 | 66.0000 | 0.1861 | 0.1872 | 0.0000 | 0.0000 | 0.0222 | 0.0111 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | -0.0003 | [-0.0006, -0.0001] | [-0.0010, +0.0003] | 0.0003 | 0.0002 | n.s. |  |  | -0.0002 | 66.3000 | 0.0890 | 0.0893 | 0.2414 | 0.2414 | 0.2333 | 0.2333 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0003 | [-0.0006, -0.0001] | [-0.0010, +0.0004] | 0.0003 | 0.0002 | n.s. |  |  | -0.0002 | 66.9000 | 0.0891 | 0.0895 | 0.2414 | 0.2414 | 0.2333 | 0.2333 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | -0.0002 | [-0.0005, +0.0002] | [-0.0010, +0.0007] | 0.0004 | 0.0003 | n.s. |  |  | -0.0001 | 55.8000 | 0.0937 | 0.0939 | 0.4483 | 0.4828 | 0.4889 | 0.5000 |

#### init/architecture D1 − blind
|  | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | seed SE (seed bootstrap) | verdict | jackknife vs CI disagree | note | 5-seed median | % 5-seed draws < 0 | D1 log MAE | D1 FB | D1 floor frac | blind log MAE | blind FB | blind floor frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'Absolute response') | 0.0080 | [+0.0051, +0.0115] | [-0.0076, +0.0249] | 0.0082 | 0.0077 | sign-unstable |  |  | 0.0105 | 29.8000 | 0.1236 | 0.0000 | 0.0000 | 0.1156 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Delta, uniform agg.') | 0.0337 | [+0.0248, +0.0438] | [+0.0220, +0.0467] | 0.0040 | 0.0040 | CI excludes 0 |  |  | 0.0333 | 0.0000 | 0.1790 | 0.0000 | 0.0361 | 0.1453 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Delta, median agg.') | 0.0195 | [+0.0134, +0.0259] | [+0.0088, +0.0311] | 0.0045 | 0.0044 | CI excludes 0 |  |  | 0.0189 | 2.0000 | 0.1405 | 0.0000 | 0.0000 | 0.1210 | 0.0000 | 0.0000 |
| ('BD-MPI', 'resolved', 'Generic potential') | 0.0296 | [+0.0212, +0.0392] | [+0.0172, +0.0432] | 0.0044 | 0.0044 | CI excludes 0 |  |  | 0.0294 | 0.1000 | 0.1794 | 0.0000 | 0.0120 | 0.1498 | 0.0000 | 0.0602 |
| ('BD-MPI', 'resolved', 'Joint, calibrated') | -0.0020 | [-0.0136, +0.0086] | [-0.0263, +0.0139] | 0.0059 | 0.0069 | sign-unstable |  |  | -0.0030 | 64.0000 | 0.0997 | 0.2453 | 0.2289 | 0.1017 | 0.0377 | 0.1446 |
| ('BD-MPI', 'resolved', 'Bidir. profile, calibrated') | -0.0019 | [-0.0134, +0.0087] | [-0.0263, +0.0139] | 0.0059 | 0.0069 | sign-unstable |  |  | -0.0029 | 63.5000 | 0.0998 | 0.2453 | 0.2289 | 0.1016 | 0.0377 | 0.1446 |
| ('BD-MPI', 'resolved', 'Bidir. context, calibrated') | 0.0035 | [-0.0099, +0.0155] | [-0.0234, +0.0187] | 0.0062 | 0.0076 | sign-unstable |  |  | -0.0007 | 53.4000 | 0.1037 | 0.4717 | 0.5120 | 0.1002 | 0.0943 | 0.1988 |
| ('BD-MPI', 'rep-avg', 'Absolute response') | 0.0112 | [+0.0079, +0.0148] | [-0.0115, +0.0325] | 0.0108 | 0.0103 | sign-unstable |  |  | 0.0131 | 30.1000 | 0.1228 | 0.0000 | 0.0000 | 0.1116 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Delta, uniform agg.') | 0.0432 | [+0.0329, +0.0539] | [+0.0293, +0.0580] | 0.0046 | 0.0045 | CI excludes 0 |  |  | 0.0430 | 0.0000 | 0.1860 | 0.0000 | 0.0333 | 0.1428 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Delta, median agg.') | 0.0230 | [+0.0156, +0.0303] | [+0.0109, +0.0361] | 0.0052 | 0.0051 | CI excludes 0 |  |  | 0.0230 | 1.7000 | 0.1397 | 0.0000 | 0.0000 | 0.1167 | 0.0000 | 0.0000 |
| ('BD-MPI', 'rep-avg', 'Generic potential') | 0.0397 | [+0.0297, +0.0501] | [+0.0257, +0.0544] | 0.0049 | 0.0050 | CI excludes 0 |  |  | 0.0393 | 0.0000 | 0.1872 | 0.0000 | 0.0111 | 0.1474 | 0.0000 | 0.0556 |
| ('BD-MPI', 'rep-avg', 'Joint, calibrated') | -0.0044 | [-0.0161, +0.0072] | [-0.0328, +0.0147] | 0.0074 | 0.0083 | sign-unstable |  |  | -0.0044 | 65.1000 | 0.0893 | 0.2414 | 0.2333 | 0.0937 | 0.0345 | 0.1444 |
| ('BD-MPI', 'rep-avg', 'Bidir. profile, calibrated') | -0.0043 | [-0.0160, +0.0073] | [-0.0326, +0.0146] | 0.0074 | 0.0083 | sign-unstable |  |  | -0.0043 | 65.0000 | 0.0895 | 0.2414 | 0.2333 | 0.0937 | 0.0345 | 0.1444 |
| ('BD-MPI', 'rep-avg', 'Bidir. context, calibrated') | 0.0018 | [-0.0121, +0.0154] | [-0.0287, +0.0209] | 0.0069 | 0.0092 | sign-unstable |  |  | -0.0014 | 55.7000 | 0.0939 | 0.4828 | 0.5000 | 0.0921 | 0.1034 | 0.2000 |

#### Offset model (blind absolute + fixed socket offset), resolved cells
|  | absolute log MAE | offset log MAE | offset − absolute | offset floor frac | 5-seed median diff | % 5-seed draws offset better |
|---|---|---|---|---|---|---|
| ('BD-MPI', 'offset_jms') | 0.1156 | 0.1107 | -0.0049 | 0.0120 | -0.0048 | 100.0000 |
| ('BD-MPI', 'offset_all') | 0.1156 | 0.0975 | -0.0180 | 0.1084 | -0.0180 | 100.0000 |

### k-curve: Δ% vs baseline over random k-seed sub-ensembles (blind, native)
1000 random subsets without replacement per k (k = 1: every seed); k = N is the full ensemble. SD at k = 50 of N = 100 is shrunk by the finite-population factor (subsets overlap).
|  | mean Δ% | SD Δ% | median Δ% | 2.5% | 97.5% | % draws < 0 | mean floor frac |
|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'Joint, calibrated', 1) | +29.42 | +67.50 | +0.02 | -11.04 | +216.56 | +43.33 | +0.28 |
| ('BD-MPI', 'Joint, calibrated', 5) | +3.69 | +20.32 | -3.53 | -14.34 | +65.50 | +67.90 | +0.17 |
| ('BD-MPI', 'Joint, calibrated', 10) | -3.00 | +10.43 | -6.35 | -12.49 | +26.40 | +79.10 | +0.13 |
| ('BD-MPI', 'Joint, calibrated', 25) | -8.22 | +1.78 | -8.70 | -10.30 | -3.62 | +99.70 | +0.12 |
| ('BD-MPI', 'Joint, calibrated', 30) | -9.02 | +0.00 | -9.02 | -9.02 | -9.02 | +100.00 | +0.14 |
| ('BD-MPI', 'Bidir. profile, calibrated', 1) | +29.47 | +67.62 | +0.01 | -11.02 | +216.69 | +43.33 | +0.28 |
| ('BD-MPI', 'Bidir. profile, calibrated', 5) | +3.36 | +18.79 | -3.60 | -13.38 | +55.75 | +67.80 | +0.17 |
| ('BD-MPI', 'Bidir. profile, calibrated', 10) | -2.45 | +10.64 | -6.12 | -12.57 | +30.65 | +77.10 | +0.14 |
| ('BD-MPI', 'Bidir. profile, calibrated', 25) | -8.26 | +1.72 | -8.78 | -10.21 | -3.73 | +99.70 | +0.12 |
| ('BD-MPI', 'Bidir. profile, calibrated', 30) | -9.01 | +0.00 | -9.01 | -9.01 | -9.01 | +100.00 | +0.14 |
| ('BD-MPI', 'Absolute response', 1) | +20.87 | +20.46 | +22.00 | -14.65 | +54.44 | +20.00 | +0.03 |
| ('BD-MPI', 'Absolute response', 5) | +10.58 | +11.66 | +10.91 | -12.01 | +30.47 | +18.80 | +0.00 |
| ('BD-MPI', 'Absolute response', 10) | +9.09 | +8.06 | +9.23 | -6.61 | +24.83 | +13.30 | +0.00 |
| ('BD-MPI', 'Absolute response', 25) | +8.55 | +2.60 | +8.39 | +3.79 | +14.07 | +0.00 | +0.00 |
| ('BD-MPI', 'Absolute response', 30) | +8.31 | +0.00 | +8.31 | +8.31 | +8.31 | +0.00 | +0.00 |

### Floor decomposition (blind, native): where the error comes from
Truth at floor = true slowdown < 1 clipped to 1 (y = 0, 'benign'); above floor = y > 0. Bias = mean(pred − truth) on log slowdown. Seeds 0–4 = the artifact's 5-seed ensemble.
|  | log MAE | bias | n at floor | log MAE at floor | n above | log MAE above | bias above | pred floor frac | pred floor frac on truth-at-floor | share of total abs error from above-floor |
|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'N = 30', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'N = 30', 'Absolute response') | 0.1116 | 0.0615 | 18.0000 | 0.1638 | 72.0000 | 0.0985 | 0.0359 | 0.0000 | 0.0000 | 0.7064 |
| ('BD-MPI', 'N = 30', 'Joint, calibrated') | 0.0937 | -0.0118 | 18.0000 | 0.0937 | 72.0000 | 0.0937 | -0.0382 | 0.1444 | 0.3333 | 0.8000 |
| ('BD-MPI', 'N = 30', 'Bidir. profile, calibrated') | 0.0937 | -0.0121 | 18.0000 | 0.0935 | 72.0000 | 0.0938 | -0.0385 | 0.1444 | 0.3333 | 0.8004 |
| ('BD-MPI', 'seeds 0–4', 'Isolated-speed baseline') | 0.1030 | -0.1030 | 18.0000 | 0.0000 | 72.0000 | 0.1288 | -0.1288 | 1.0000 | 1.0000 | 1.0000 |
| ('BD-MPI', 'seeds 0–4', 'Absolute response') | 0.1207 | 0.0881 | 18.0000 | 0.1921 | 72.0000 | 0.1028 | 0.0621 | 0.0000 | 0.0000 | 0.6817 |
| ('BD-MPI', 'seeds 0–4', 'Joint, calibrated') | 0.0940 | -0.0183 | 18.0000 | 0.0902 | 72.0000 | 0.0949 | -0.0455 | 0.1556 | 0.2778 | 0.8080 |
| ('BD-MPI', 'seeds 0–4', 'Bidir. profile, calibrated') | 0.0940 | -0.0190 | 18.0000 | 0.0896 | 72.0000 | 0.0951 | -0.0461 | 0.1556 | 0.2778 | 0.8095 |

