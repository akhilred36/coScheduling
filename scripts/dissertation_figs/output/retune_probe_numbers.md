## Retune probe, BD-MPI (blind CV; final fits blind / E1 / D1, 30 seeds)

### 1. App-Inhibitor crossed-CV selection (the only selection data)
Curve = mean per-fit validation error (calibrated inference) over all fits of the selected config, at epochs where every fit is still running. Protocol = median of per-fold best epochs (the selector); argmin = sensitivity.
| variant | original epochs (80/patience 10) | protocol epochs (median per-fold best) | fold-avg argmin | curve epochs with all fits alive | per-fold best: % at epoch 1 | grouped CV log MAE at selection | original grouped CV log MAE | curve still falling at end |
|---|---|---|---|---|---|---|---|---|
| legacy_delta | 13 | 45 | 18 | 240 | 14.5000 | 0.1292 | 0.1690 | False |
| legacy_generic | 14 | 47 | 20 | 240 | 12.5000 | 0.1325 | 0.1617 | False |
| legacy_absolute | 2 | 31 | 32 | 240 | 18.5000 | 0.1114 | 0.1562 | False |
| joint_profile | 1 | 37 | 1 | 240 | 22.0000 | 0.1075 | 0.1481 | False |
| bidirectional_profile | 2 | 27 | 1 | 240 | 27.0000 | 0.1094 | 0.1479 | False |
| bidirectional_context | 2 | 20 | 1 | 240 | 21.5000 | 0.1115 | 0.1481 | False |
| bidirectional_profile_wall | 2 | 35 | 1 | 240 | 21.0000 | 0.1200 | 0.1513 | False |
| bidirectional_context_forward | 1 | 16 | 1 | 240 | 22.5000 | 0.0996 | 0.1445 | False |

| variant | curve@1 | curve@2 | curve@4 | curve@7 | curve@15 | curve@30 | curve@60 | curve@120 | curve@240 |
|---|---|---|---|---|---|---|---|---|---|
| legacy_delta | 0.2130 | 0.2115 | 0.2093 | 0.2065 | 0.1991 | 0.1998 | 0.2146 | 0.2267 | 0.2739 |
| legacy_generic | 0.2093 | 0.2088 | 0.2078 | 0.2061 | 0.1997 | 0.1999 | 0.2125 | 0.2465 | 0.2834 |
| legacy_absolute | 0.1994 | 0.2004 | 0.2024 | 0.2037 | 0.1977 | 0.1976 | 0.1993 | 0.2110 | 0.2452 |
| joint_profile | 0.1781 | 0.1808 | 0.1854 | 0.1911 | 0.1954 | 0.1971 | 0.2207 | 0.2364 | 0.2652 |
| bidirectional_profile | 0.1780 | 0.1804 | 0.1843 | 0.1883 | 0.1939 | 0.1948 | 0.2195 | 0.2435 | 0.2565 |
| bidirectional_context | 0.1824 | 0.1844 | 0.1880 | 0.1945 | 0.2021 | 0.2052 | 0.2233 | 0.2377 | 0.2513 |
| bidirectional_profile_wall | 0.1776 | 0.1797 | 0.1831 | 0.1873 | 0.1932 | 0.1956 | 0.2101 | 0.2191 | 0.2608 |
| bidirectional_context_forward | 0.1777 | 0.1794 | 0.1835 | 0.1886 | 0.1973 | 0.2024 | 0.2051 | 0.1925 | 0.2138 |

### 2. App-App error vs training budget — DESCRIPTIVE ONLY, not used for any selection
Δ% of the N-seed mean-latent ensemble vs the isolated-speed baseline, native non-self outcomes (pipeline metrics).
| run | baseline log MAE | absolute Δ% | joint_cal Δ% | joint_direct Δ% | bprof_cal Δ% | bctx_cal Δ% | delta_uniform Δ% | delta_median Δ% | generic Δ% |
|---|---|---|---|---|---|---|---|---|---|
| current recipe (part a, 100 seeds) | 0.1030 | +0.0 | -10.5 | -1.9 | -10.2 | -10.4 | +41.8 | +13.3 | +42.6 |
| all variants at 7 epochs (30 seeds) | 0.1030 | -8.8 | +3.3 | -3.2 | +3.0 | +5.6 | +42.1 | +7.5 | +44.6 |
| all variants at 15 epochs (30 seeds) | 0.1030 | -12.2 | +20.4 | +22.6 | +19.9 | +11.7 | +39.4 | +12.2 | +43.2 |
| all variants at 30 epochs (30 seeds) | 0.1030 | +8.0 | +28.9 | +36.1 | +29.3 | +12.2 | +38.2 | +16.4 | +45.5 |
| all variants at 60 epochs (30 seeds) | 0.1030 | +14.7 | +33.5 | +35.3 | +37.3 | +34.6 | +31.4 | +12.2 | +44.6 |
| all variants at 120 epochs (30 seeds) | 0.1030 | +17.1 | +24.4 | +25.6 | +24.6 | +48.7 | +26.5 | +9.1 | +39.9 |
| all variants at 240 epochs (30 seeds) | 0.1030 | +16.9 | +27.4 | +28.0 | +9.7 | +25.5 | +29.6 | +15.4 | +37.1 |

### 3a. Final fits, CV-selected budget (protocol selector)

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


### 3b. Final fits, SENSITIVITY: fold-averaged-argmin budget (not the protocol selector)

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


### Reference: single-seed degeneracy under the near-init recipe (part a, BD-MPI blind, 100 seeds)
| role | 1-seed at floor everywhere | 1-seed blown up (Δ% > +25) |
|---|---|---|
| absolute | 0.4500 | 0.3000 |
| delta_uniform | 0.0000 | 0.9800 |
| delta_median | 0.0000 | 0.3500 |
| delta_ood_kernel | 0.0000 | 0.4200 |
| generic | 0.0000 | 1.0000 |
| joint_direct | 0.3800 | 0.3500 |
| joint_cal | 0.0900 | 0.3100 |
| bprof_direct | 0.3300 | 0.3600 |
| bprof_cal | 0.0900 | 0.3200 |
| bctx_direct | 0.3400 | 0.3200 |
| bctx_cal | 0.0800 | 0.3500 |
| bwall_cal | 0.0900 | 0.3300 |
| bfwd_cal | 0.3500 | 0.3800 |

### 4. Capacity-probe trigger
Verdicts: {'cv: joint cal < baseline': 'fails', 'cv: bprof cal < baseline': 'fails', 'sens: joint cal < baseline': 'n.s.', 'sens: bprof cal < baseline': 'n.s.'}. A calibrated variant helps (two-level CI excludes 0, not sign-unstable): False. Capacity probe started.
