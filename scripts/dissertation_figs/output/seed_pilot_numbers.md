## S0. Seed pilot, BD-MPI blind, frozen CV recipe, 30 final seeds (P0_bd_mpi_blind)

- Seeds 0–4 sub-ensemble vs the original 5-seed run (E0 = CV run), max |Δ latent| = 1.5e-16.
- Sub-ensemble = mean of per-seed latents, then max(0, ·), exactly as the pipeline builds its ensemble.
- Disjoint 5-seed groups: 6; random 5-subsets: 300 (without replacement from 30).

### log MAE, native outcomes
| method | 1-seed SD | 5-seed SD (disjoint) | 5-seed SD (random, FPC) | 10-seed SD | old 5 | full 30 | old − full | old pctile in 5-subsets | N for SE ≤ 0.002 |
|---|---|---|---|---|---|---|---|---|---|
| Absolute response | 0.0596 | 0.0042 | 0.0104 | 0.0062 | 0.0942 | 0.1030 | -0.0088 | 14.0000 | 211.0000 |
| Delta, uniform agg. | 0.0113 | 0.0062 | 0.0053 | 0.0039 | 0.1457 | 0.1439 | 0.0018 | 54.6667 | 35.0000 |
| Delta, median agg. | 0.0151 | 0.0094 | 0.0063 | 0.0046 | 0.1190 | 0.1142 | 0.0049 | 75.3333 | 52.0000 |
| Delta, OOD-kernel agg. | 0.0124 | 0.0085 | 0.0062 | 0.0045 | 0.1258 | 0.1214 | 0.0044 | 73.3333 | 48.0000 |
| Generic potential | 0.0122 | 0.0023 | 0.0054 | 0.0036 | 0.1472 | 0.1477 | -0.0005 | 46.6667 | 36.0000 |
| Joint | 0.1370 | 0.0281 | 0.0380 | 0.0166 | 0.1004 | 0.0968 | 0.0036 | 27.0000 | 1676.0000 |
| Joint, calibrated | 0.0695 | 0.0213 | 0.0242 | 0.0134 | 0.0940 | 0.0937 | 0.0003 | 19.0000 | 644.0000 |
| Bidir. profile | 0.1276 | 0.0288 | 0.0374 | 0.0173 | 0.0966 | 0.0924 | 0.0042 | 19.6667 | 1605.0000 |
| Bidir. profile, calibrated | 0.0649 | 0.0215 | 0.0241 | 0.0142 | 0.0953 | 0.0948 | 0.0005 | 24.0000 | 630.0000 |
| Bidir. context | 0.1203 | 0.0533 | 0.0303 | 0.0174 | 0.1032 | 0.0947 | 0.0084 | 69.0000 | 1385.0000 |
| Bidir. context, calibrated | 0.0706 | 0.0331 | 0.0247 | 0.0170 | 0.0938 | 0.0931 | 0.0006 | 20.3333 | 846.0000 |
| Wall-clock aux (ablation), cal. | 0.0648 | 0.0213 | 0.0242 | 0.0142 | 0.0950 | 0.0948 | 0.0002 | 22.0000 | 635.0000 |
| Forward-only tokens (ablation), cal. | 0.0948 | 0.0237 | 0.0364 | 0.0205 | 0.0958 | 0.0899 | 0.0059 | 20.3333 | 1653.0000 |

### log MAE, rep-avg outcomes
| method | 1-seed SD | 5-seed SD (disjoint) | 5-seed SD (random, FPC) | 10-seed SD | old 5 | full 30 | old − full | old pctile in 5-subsets | N for SE ≤ 0.002 |
|---|---|---|---|---|---|---|---|---|---|
| Absolute response | 0.0596 | 0.0042 | 0.0104 | 0.0062 | 0.0942 | 0.1030 | -0.0088 | 14.0000 | 211.0000 |
| Delta, uniform agg. | 0.0113 | 0.0062 | 0.0053 | 0.0039 | 0.1457 | 0.1439 | 0.0018 | 54.6667 | 35.0000 |
| Delta, median agg. | 0.0151 | 0.0094 | 0.0063 | 0.0046 | 0.1190 | 0.1142 | 0.0049 | 75.3333 | 52.0000 |
| Delta, OOD-kernel agg. | 0.0124 | 0.0085 | 0.0062 | 0.0045 | 0.1258 | 0.1214 | 0.0044 | 73.3333 | 48.0000 |
| Generic potential | 0.0122 | 0.0023 | 0.0054 | 0.0036 | 0.1472 | 0.1477 | -0.0005 | 46.6667 | 36.0000 |
| Joint | 0.1370 | 0.0281 | 0.0380 | 0.0166 | 0.1004 | 0.0968 | 0.0036 | 27.0000 | 1676.0000 |
| Joint, calibrated | 0.0695 | 0.0213 | 0.0242 | 0.0134 | 0.0940 | 0.0937 | 0.0003 | 19.0000 | 644.0000 |
| Bidir. profile | 0.1276 | 0.0288 | 0.0374 | 0.0173 | 0.0966 | 0.0924 | 0.0042 | 19.6667 | 1605.0000 |
| Bidir. profile, calibrated | 0.0649 | 0.0215 | 0.0241 | 0.0142 | 0.0953 | 0.0948 | 0.0005 | 24.0000 | 630.0000 |
| Bidir. context | 0.1203 | 0.0533 | 0.0303 | 0.0174 | 0.1032 | 0.0947 | 0.0084 | 69.0000 | 1385.0000 |
| Bidir. context, calibrated | 0.0706 | 0.0331 | 0.0247 | 0.0170 | 0.0938 | 0.0931 | 0.0006 | 20.3333 | 846.0000 |
| Wall-clock aux (ablation), cal. | 0.0648 | 0.0213 | 0.0242 | 0.0142 | 0.0950 | 0.0948 | 0.0002 | 22.0000 | 635.0000 |
| Forward-only tokens (ablation), cal. | 0.0948 | 0.0237 | 0.0364 | 0.0205 | 0.0958 | 0.0899 | 0.0059 | 20.3333 | 1653.0000 |

### log MAE, resolved outcomes
| method | 1-seed SD | 5-seed SD (disjoint) | 5-seed SD (random, FPC) | 10-seed SD | old 5 | full 30 | old − full | old pctile in 5-subsets | N for SE ≤ 0.002 |
|---|---|---|---|---|---|---|---|---|---|
| Absolute response | 0.0542 | 0.0035 | 0.0081 | 0.0042 | 0.1035 | 0.1102 | -0.0067 | 10.3333 | 138.0000 |
| Delta, uniform agg. | 0.0109 | 0.0057 | 0.0047 | 0.0036 | 0.1486 | 0.1467 | 0.0019 | 58.3333 | 29.0000 |
| Delta, median agg. | 0.0148 | 0.0087 | 0.0059 | 0.0043 | 0.1244 | 0.1187 | 0.0056 | 80.0000 | 46.0000 |
| Delta, OOD-kernel agg. | 0.0112 | 0.0072 | 0.0052 | 0.0037 | 0.1300 | 0.1254 | 0.0046 | 79.0000 | 34.0000 |
| Generic potential | 0.0114 | 0.0019 | 0.0049 | 0.0033 | 0.1504 | 0.1502 | 0.0001 | 51.6667 | 30.0000 |
| Joint | 0.1323 | 0.0230 | 0.0333 | 0.0132 | 0.1082 | 0.1067 | 0.0015 | 24.3333 | 1310.0000 |
| Joint, calibrated | 0.0649 | 0.0182 | 0.0208 | 0.0110 | 0.1021 | 0.1017 | 0.0004 | 17.0000 | 480.0000 |
| Bidir. profile | 0.1229 | 0.0237 | 0.0325 | 0.0137 | 0.1053 | 0.1037 | 0.0016 | 15.3333 | 1239.0000 |
| Bidir. profile, calibrated | 0.0605 | 0.0183 | 0.0206 | 0.0116 | 0.1028 | 0.1024 | 0.0003 | 19.3333 | 465.0000 |
| Bidir. context | 0.1152 | 0.0475 | 0.0254 | 0.0137 | 0.1101 | 0.1050 | 0.0051 | 35.0000 | 1040.0000 |
| Bidir. context, calibrated | 0.0658 | 0.0284 | 0.0211 | 0.0140 | 0.1025 | 0.1009 | 0.0017 | 20.0000 | 641.0000 |
| Wall-clock aux (ablation), cal. | 0.0604 | 0.0181 | 0.0207 | 0.0116 | 0.1026 | 0.1024 | 0.0003 | 18.3333 | 470.0000 |
| Forward-only tokens (ablation), cal. | 0.0897 | 0.0185 | 0.0311 | 0.0161 | 0.1057 | 0.1025 | 0.0032 | 17.0000 | 1243.0000 |

### §8 headline quantities (native truth): old 5 seeds vs the distribution of 5-seed sub-ensembles
| quantity | old 5 | full 30 | 5-subset median | 5-subset 2.5% | 5-subset 97.5% | 1-seed min | 1-seed max | old pctile |
|---|---|---|---|---|---|---|---|---|
| Absolute response: Δ% | -8.5292 | 0.0000 | 0.0000 | -15.7629 | 19.9505 | -11.6681 | 227.1356 | 14.0000 |
| Absolute response: FB@1.05 | 0.7241 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 28.6667 |
| Absolute response: Spearman | 0.0583 | – | 0.0418 | -0.3666 | 0.3587 | -0.3612 | 0.3787 | 57.1429 |
| Absolute response: floor frac | 0.0333 | 1.0000 | 0.9389 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 24.6667 |
| Delta, uniform agg.: Δ% | 41.4438 | 39.7309 | 41.0181 | 31.2106 | 50.4804 | 27.2561 | 65.9951 | 54.6667 |
| Delta, uniform agg.: Spearman | 0.2495 | 0.2470 | 0.2498 | 0.2170 | 0.2947 | 0.1553 | 0.3484 | 49.0000 |
| Delta, uniform agg.: floor frac | 0.0556 | 0.0444 | 0.0444 | 0.0222 | 0.0667 | 0.0000 | 0.1778 | 61.0000 |
| Delta, median agg.: Δ% | 15.5569 | 10.8243 | 12.0670 | 1.5586 | 23.1453 | -3.0190 | 56.0395 | 75.3333 |
| Delta, median agg.: Spearman | 0.2643 | 0.2674 | 0.2660 | 0.2165 | 0.3264 | 0.1137 | 0.3924 | 49.0000 |
| Delta, median agg.: floor frac | 0.0111 | 0.0000 | 0.0222 | 0.0000 | 0.0444 | 0.0000 | 0.1778 | 15.3333 |
| Delta, OOD-kernel agg.: Δ% | 22.1459 | 17.8607 | 19.0695 | 7.8798 | 29.2885 | 3.2304 | 50.6926 | 73.3333 |
| Delta, OOD-kernel agg.: Spearman | 0.3174 | 0.3355 | 0.3237 | 0.2832 | 0.3586 | 0.2429 | 0.4150 | 39.6667 |
| Delta, OOD-kernel agg.: floor frac | 0.0111 | 0.0000 | 0.0111 | 0.0000 | 0.0333 | 0.0000 | 0.1111 | 34.3333 |
| Generic potential: Δ% | 42.9381 | 43.3812 | 43.2399 | 35.8642 | 54.2717 | 30.8776 | 84.4208 | 46.6667 |
| Generic potential: Spearman | 0.2467 | 0.2491 | 0.2509 | 0.2188 | 0.2917 | 0.1686 | 0.3205 | 39.3333 |
| Generic potential: floor frac | 0.0444 | 0.0556 | 0.0444 | 0.0111 | 0.0556 | 0.0000 | 0.0778 | 41.6667 |
| Joint: Δ% | -2.5358 | -6.0512 | 0.0000 | -15.9513 | 111.1732 | -12.3697 | 604.5429 | 27.0000 |
| Joint: Spearman | -0.1537 | -0.1487 | -0.0739 | -0.4624 | 0.4326 | -0.4799 | 0.2880 | 37.7358 |
| Joint: floor frac | 0.5889 | 0.0778 | 0.1056 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 58.0000 |
| Joint, calibrated: Δ% | -8.7734 | -9.0184 | -3.7828 | -14.2501 | 64.3751 | -11.5918 | 300.4657 | 19.0000 |
| Joint, calibrated: FB@1.05 | 0.1034 | 0.0345 | 0.0345 | 0.0000 | 0.7078 | 0.0000 | 1.0000 | 56.6667 |
| Joint, calibrated: Spearman | 0.2076 | 0.1650 | 0.1951 | 0.0036 | 0.2969 | -0.1421 | 0.3009 | 57.6667 |
| Joint, calibrated: floor frac | 0.1556 | 0.1444 | 0.1000 | 0.0000 | 0.6392 | 0.0000 | 1.0000 | 59.3333 |
| Bidir. profile: Δ% | -6.2324 | -10.3099 | 0.0000 | -16.2132 | 108.9973 | -15.6299 | 574.3522 | 19.6667 |
| Bidir. profile: Spearman | -0.0364 | -0.1372 | -0.0647 | -0.4439 | 0.4335 | -0.4736 | 0.2705 | 52.1930 |
| Bidir. profile: floor frac | 0.2000 | 0.0000 | 0.0278 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 56.6667 |
| Bidir. profile, calibrated: Δ% | -7.4893 | -7.9310 | -3.3507 | -13.5720 | 63.6289 | -11.5678 | 286.3268 | 24.0000 |
| Bidir. profile, calibrated: FB@1.05 | 0.0345 | 0.0345 | 0.0000 | 0.0000 | 0.6043 | 0.0000 | 1.0000 | 52.3333 |
| Bidir. profile, calibrated: Spearman | 0.2094 | 0.1727 | 0.1995 | 0.0469 | 0.2961 | -0.1110 | 0.2867 | 56.3333 |
| Bidir. profile, calibrated: floor frac | 0.1000 | 0.0444 | 0.0556 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 56.0000 |
| Bidir. context: Δ% | 0.1410 | -8.0211 | 0.0000 | -14.9360 | 97.5359 | -15.5174 | 504.1589 | 69.0000 |
| Bidir. context: Spearman | -0.1416 | -0.0080 | -0.0421 | -0.4007 | 0.4284 | -0.3934 | 0.3596 | 29.5455 |
| Bidir. context: floor frac | 0.9333 | 0.0778 | 0.0556 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 69.3333 |
| Bidir. context, calibrated: Δ% | -8.9637 | -9.5933 | -2.6892 | -14.9752 | 69.0338 | -19.0060 | 298.5104 | 20.3333 |
| Bidir. context, calibrated: FB@1.05 | 0.3103 | 0.0345 | 0.0000 | 0.0000 | 0.8966 | 0.0000 | 1.0000 | 80.0000 |
| Bidir. context, calibrated: Spearman | 0.2121 | 0.2047 | 0.2153 | 0.0315 | 0.3573 | -0.2042 | 0.3393 | 48.3333 |
| Bidir. context, calibrated: floor frac | 0.2778 | 0.1667 | 0.0556 | 0.0000 | 0.6725 | 0.0000 | 1.0000 | 75.3333 |
| Wall-clock aux (ablation), cal.: Δ% | -7.7942 | -7.9591 | -3.5267 | -13.6157 | 64.0733 | -11.2893 | 285.4212 | 22.0000 |
| Wall-clock aux (ablation), cal.: Spearman | 0.2201 | 0.1711 | 0.2019 | 0.0479 | 0.2965 | -0.1110 | 0.2733 | 61.3333 |
| Wall-clock aux (ablation), cal.: floor frac | 0.1000 | 0.0444 | 0.0500 | 0.0000 | 0.5058 | 0.0000 | 1.0000 | 56.0000 |
| Forward-only tokens (ablation), cal.: Δ% | -6.9801 | -12.7369 | 0.0000 | -17.2811 | 107.8663 | -16.1639 | 399.3538 | 20.3333 |
| Forward-only tokens (ablation), cal.: Spearman | 0.0132 | 0.0405 | 0.0414 | -0.3309 | 0.4940 | -0.2987 | 0.3586 | 45.2675 |
| Forward-only tokens (ablation), cal.: floor frac | 0.3222 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 65.3333 |

### Convergence: mean and SD of the k-seed ensemble log MAE (native), by k
| method | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 | k=15 | k=30 |
|---|---|---|---|---|---|---|---|---|
| Absolute response | 0.1342 ± 0.0596 | 0.1158 ± 0.0339 | 0.1074 ± 0.0200 | 0.1021 ± 0.0104 | 0.1003 ± 0.0063 | 0.1002 ± 0.0062 | 0.1012 ± 0.0055 | 0.1030 |
| Delta, uniform agg. | 0.1502 ± 0.0113 | 0.1473 ± 0.0082 | 0.1471 ± 0.0066 | 0.1451 ± 0.0053 | 0.1448 ± 0.0044 | 0.1444 ± 0.0039 | 0.1444 ± 0.0033 | 0.1439 |
| Delta, median agg. | 0.1211 ± 0.0151 | 0.1176 ± 0.0101 | 0.1172 ± 0.0082 | 0.1153 ± 0.0063 | 0.1149 ± 0.0051 | 0.1146 ± 0.0046 | 0.1148 ± 0.0040 | 0.1142 |
| Delta, OOD-kernel agg. | 0.1253 ± 0.0124 | 0.1235 ± 0.0094 | 0.1236 ± 0.0079 | 0.1222 ± 0.0062 | 0.1219 ± 0.0050 | 0.1217 ± 0.0045 | 0.1219 ± 0.0039 | 0.1214 |
| Generic potential | 0.1509 ± 0.0122 | 0.1490 ± 0.0084 | 0.1489 ± 0.0070 | 0.1481 ± 0.0054 | 0.1478 ± 0.0041 | 0.1477 ± 0.0036 | 0.1478 ± 0.0033 | 0.1477 |
| Joint | 0.1599 ± 0.1370 | 0.1348 ± 0.0761 | 0.1194 ± 0.0454 | 0.1146 ± 0.0380 | 0.1043 ± 0.0199 | 0.1031 ± 0.0166 | 0.0991 ± 0.0096 | 0.0968 |
| Joint, calibrated | 0.1333 ± 0.0695 | 0.1202 ± 0.0418 | 0.1115 ± 0.0273 | 0.1084 ± 0.0242 | 0.1015 ± 0.0150 | 0.1012 ± 0.0134 | 0.0974 ± 0.0090 | 0.0937 |
| Bidir. profile | 0.1564 ± 0.1276 | 0.1335 ± 0.0716 | 0.1194 ± 0.0442 | 0.1149 ± 0.0374 | 0.1046 ± 0.0205 | 0.1035 ± 0.0173 | 0.0986 ± 0.0107 | 0.0924 |
| Bidir. profile, calibrated | 0.1323 ± 0.0649 | 0.1200 ± 0.0396 | 0.1119 ± 0.0268 | 0.1091 ± 0.0241 | 0.1023 ± 0.0155 | 0.1020 ± 0.0142 | 0.0984 ± 0.0101 | 0.0948 |
| Bidir. context | 0.1555 ± 0.1203 | 0.1320 ± 0.0745 | 0.1219 ± 0.0468 | 0.1127 ± 0.0303 | 0.1039 ± 0.0148 | 0.1033 ± 0.0174 | 0.0982 ± 0.0098 | 0.0947 |
| Bidir. context, calibrated | 0.1383 ± 0.0706 | 0.1235 ± 0.0469 | 0.1174 ± 0.0353 | 0.1093 ± 0.0247 | 0.1026 ± 0.0169 | 0.1020 ± 0.0170 | 0.0972 ± 0.0113 | 0.0931 |
| Wall-clock aux (ablation), cal. | 0.1324 ± 0.0648 | 0.1199 ± 0.0396 | 0.1119 ± 0.0270 | 0.1091 ± 0.0242 | 0.1023 ± 0.0155 | 0.1020 ± 0.0142 | 0.0984 ± 0.0101 | 0.0948 |
| Forward-only tokens (ablation), cal. | 0.1624 ± 0.0948 | 0.1377 ± 0.0647 | 0.1267 ± 0.0469 | 0.1166 ± 0.0364 | 0.1060 ± 0.0238 | 0.1053 ± 0.0205 | 0.0989 ± 0.0131 | 0.0899 |

### Contrasts: seed SD of paired differences
| contrast (native log MAE) | old 5 | full 30 | 1-seed SD | 5-seed SD | 10-seed SD | sign agrees with full, % of 5-subsets |
|---|---|---|---|---|---|---|
| Absolute response − Isolated-speed baseline | -0.0088 | 0.0000 | 0.0596 | 0.0104 | 0.0062 | 44.0000 |
| Joint, calibrated − Absolute response | -0.0003 | -0.0093 | 0.0544 | 0.0245 | 0.0162 | 56.3333 |
| Bidir. profile, calibrated − Joint, calibrated | 0.0013 | 0.0011 | 0.0061 | 0.0021 | 0.0020 | 59.3333 |
| Joint − Absolute response | 0.0062 | -0.0062 | 0.1075 | 0.0371 | 0.0185 | 35.6667 |
| Bidir. context − Bidir. profile | 0.0066 | 0.0024 | 0.0809 | 0.0298 | 0.0149 | 44.6667 |
| Bidir. context, calibrated − Bidir. profile, calibrated | -0.0015 | -0.0017 | 0.0476 | 0.0210 | 0.0131 | 45.3333 |
| Delta, uniform agg. − Absolute response | 0.0515 | 0.0409 | 0.0613 | 0.0116 | 0.0073 | 99.0000 |
| Joint, calibrated − Joint | -0.0064 | -0.0031 | 0.0681 | 0.0164 | 0.0092 | 68.3333 |

### Seed SE of the full 30-seed ensemble (delete-one jackknife) and SD(k) power-law fit (k = 2..10)
|  | jackknife SE at N=30 | beta | N for SE ≤ 0.002 (power fit) |
|---|---|---|---|
| ('native', 'Absolute response') | 0.0000 | 1.0979 | 25.0000 |
| ('native', 'Delta, uniform agg.') | 0.0024 | 0.4439 | 46.0000 |
| ('native', 'Delta, median agg.') | 0.0028 | 0.4903 | 54.0000 |
| ('native', 'Delta, OOD-kernel agg.') | 0.0027 | 0.4603 | 59.0000 |
| ('native', 'Generic potential') | 0.0022 | 0.5266 | 32.0000 |
| ('native', 'Joint') | 0.0166 | 0.9160 | 104.0000 |
| ('native', 'Joint, calibrated') | 0.0033 | 0.6805 | 165.0000 |
| ('native', 'Bidir. profile') | 0.0124 | 0.8536 | 129.0000 |
| ('native', 'Bidir. profile, calibrated') | 0.0049 | 0.6178 | 237.0000 |
| ('native', 'Bidir. context') | 0.0166 | 0.9860 | 76.0000 |
| ('native', 'Bidir. context, calibrated') | 0.0057 | 0.6695 | 218.0000 |
| ('native', 'Wall-clock aux (ablation), cal.') | 0.0048 | 0.6181 | 237.0000 |
| ('native', 'Forward-only tokens (ablation), cal.') | 0.0055 | 0.7058 | 276.0000 |
| ('resolved', 'Absolute response') | 0.0000 | 1.2547 | 17.0000 |
| ('resolved', 'Delta, uniform agg.') | 0.0021 | 0.4480 | 36.0000 |
| ('resolved', 'Delta, median agg.') | 0.0025 | 0.4897 | 48.0000 |
| ('resolved', 'Delta, OOD-kernel agg.') | 0.0022 | 0.4806 | 37.0000 |
| ('resolved', 'Generic potential') | 0.0021 | 0.5245 | 27.0000 |
| ('resolved', 'Joint') | 0.0102 | 1.0120 | 67.0000 |
| ('resolved', 'Joint, calibrated') | 0.0020 | 0.7323 | 104.0000 |
| ('resolved', 'Bidir. profile') | 0.0082 | 0.9497 | 79.0000 |
| ('resolved', 'Bidir. profile, calibrated') | 0.0037 | 0.6699 | 138.0000 |
| ('resolved', 'Bidir. context') | 0.0103 | 1.1002 | 49.0000 |
| ('resolved', 'Bidir. context, calibrated') | 0.0049 | 0.7316 | 129.0000 |
| ('resolved', 'Wall-clock aux (ablation), cal.') | 0.0036 | 0.6693 | 139.0000 |
| ('resolved', 'Forward-only tokens (ablation), cal.') | 0.0053 | 0.7976 | 141.0000 |

| contrast | full 30 | jackknife SE at N=30 |
|---|---|---|
| Absolute response − Isolated-speed baseline | 0.0000 | 0.0000 |
| Joint, calibrated − Absolute response | -0.0093 | 0.0033 |
| Bidir. profile, calibrated − Joint, calibrated | 0.0011 | 0.0019 |
| Joint − Absolute response | -0.0062 | 0.0166 |
| Bidir. context − Bidir. profile | 0.0024 | 0.0108 |
| Bidir. context, calibrated − Bidir. profile, calibrated | -0.0017 | 0.0048 |
| Delta, uniform agg. − Absolute response | 0.0409 | 0.0024 |
| Joint, calibrated − Joint | -0.0031 | 0.0193 |

### §8 claims: fraction of random k-seed sub-ensembles in which each holds
| claim (native, BD-MPI blind) | old 5 | % of 5-subsets | % of 10-subsets | % of 15-subsets | full 30 |
|---|---|---|---|---|---|
| absolute < baseline | True | 41.7 | 46.7 | 37.7 | False |
| absolute at the floor on all 90 outcomes | False | 44.0 | 49.3 | 59.3 | True |
| joint cal FB < absolute FB | True | 78.0 | 90.0 | 98.7 | True |
| joint cal < baseline | True | 64.7 | 76.3 | 86.0 | True |
| joint cal < absolute | True | 56.3 | 66.7 | 78.7 | True |
| delta uniform > baseline | True | 100.0 | 100.0 | 100.0 | True |
| delta median > baseline | True | 99.0 | 100.0 | 100.0 | True |
| generic > baseline | True | 100.0 | 100.0 | 100.0 | True |
| C7 fingerprints hurt, direct (bctx > bprof) | True | 44.7 | 38.3 | 47.0 | True |
| C7 fingerprints hurt, cal (bctx cal > bprof cal) | False | 54.7 | 48.3 | 35.0 | False |
| C7 joint direct hurts vs absolute | True | 46.7 | 38.3 | 21.0 | False |
| C7 reverse supervision helps, direct (bprof < joint) | True | 40.3 | 35.0 | 55.7 | True |
| C7 reverse supervision helps, cal | False | 40.7 | 37.0 | 27.3 | False |
| calibration helps joint (cal < direct) | True | 68.3 | 62.0 | 56.7 | True |

