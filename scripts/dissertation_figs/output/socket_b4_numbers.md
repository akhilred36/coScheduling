## B4. Restricted training-size tiers, bidirectional, blind vs socket-aware

### BD-MPI
Completed evaluate_training_size tasks per size (of 10 splits):
| size | aware | blind |
|---|---|---|
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 8 | 8 |
| 5 | 9 | 9 |
| 6 | 10 | 10 |
| 7 | 10 | 10 |
| 8 | 10 | 10 |
- blind: sizes 4-8 completed 47/50; sizes 2-3 completed 0/20; other statuses: {'failed': np.int64(23)}
- aware: sizes 4-8 completed 47/50; sizes 2-3 completed 0/20; other statuses: {'failed': np.int64(23)}
- matched cells (sizes 4-8, both variants): 47
#### C4 format, rep-avg outcomes (47 matched cells): pooled skill (wins/47)
|  | one_known blind | one_known aware | zero_shot blind | zero_shot aware |
|---|---|---|---|---|
| Absolute response | +6.3% (44) | +0.0% (3) | -4.0% (23) | +0.0% (3) |
| Delta, uniform agg. | -53.9% (2) | -67.1% (2) | -89.2% (7) | -91.0% (6) |
| Delta, median agg. | -36.6% (7) | -28.0% (9) | -74.1% (7) | -46.3% (11) |
| Generic potential | -49.0% (4) | -68.1% (2) | -68.9% (7) | -97.6% (6) |
| Joint, calibrated | +5.5% (36) | +6.0% (47) | +3.8% (29) | +10.6% (38) |
| Bidir. profile, calibrated | +5.2% (34) | +5.2% (47) | +2.0% (28) | +9.4% (39) |
| Bidir. context, calibrated | +4.0% (35) | +0.0% (0) | +4.8% (28) | +0.0% (4) |

#### C4 format, resolved outcomes (47 matched cells): pooled skill (wins/47)
|  | one_known blind | one_known aware | zero_shot blind | zero_shot aware |
|---|---|---|---|---|
| Absolute response | +5.0% (44) | -0.0% (0) | -3.8% (24) | +0.0% (3) |
| Delta, uniform agg. | -38.0% (6) | -48.7% (3) | -92.7% (5) | -91.7% (4) |
| Delta, median agg. | -24.9% (9) | -20.1% (14) | -76.4% (6) | -50.5% (10) |
| Generic potential | -34.9% (7) | -47.7% (3) | -74.2% (6) | -98.4% (6) |
| Joint, calibrated | +5.9% (36) | +4.7% (47) | +2.7% (27) | +7.3% (37) |
| Bidir. profile, calibrated | +5.9% (35) | +4.1% (47) | +1.0% (24) | +6.3% (37) |
| Bidir. context, calibrated | +4.0% (33) | +0.0% (0) | +4.2% (22) | +0.0% (4) |

#### Paired aware − blind (log MAE over matched cells, pair-cluster 95% CI)
##### rep-avg
| method | aware − blind | CI lo | CI hi | blind log MAE | aware log MAE | blind floor frac | aware floor frac | n rows |
|---|---|---|---|---|---|---|---|---|
| one_known: Absolute response | 0.0067 | -0.0004 | 0.0133 | 0.0985 | 0.1052 | 0.1563 | 0.9981 | 2054.0000 |
| one_known: Delta, uniform agg. | 0.0141 | -0.0027 | 0.0318 | 0.1608 | 0.1749 | 0.0777 | 0.0404 | 2054.0000 |
| one_known: Delta, median agg. | -0.0088 | -0.0263 | 0.0100 | 0.1431 | 0.1343 | 0.0493 | 0.0477 | 2054.0000 |
| one_known: Generic potential | 0.0206 | 0.0095 | 0.0311 | 0.1552 | 0.1758 | 0.0648 | 0.0234 | 2054.0000 |
| one_known: Joint, calibrated | -0.0002 | -0.0184 | 0.0174 | 0.0991 | 0.0989 | 0.1324 | 0.3676 | 2054.0000 |
| one_known: Bidir. profile, calibrated | 0.0002 | -0.0201 | 0.0199 | 0.0995 | 0.0997 | 0.1125 | 0.3841 | 2054.0000 |
| one_known: Bidir. context, calibrated | 0.0043 | -0.0120 | 0.0194 | 0.1009 | 0.1052 | 0.2999 | 1.0000 | 2054.0000 |
| zero_shot: Absolute response | -0.0033 | -0.0141 | 0.0068 | 0.0985 | 0.0952 | 0.2032 | 0.9903 | 620.0000 |
| zero_shot: Delta, uniform agg. | 0.0036 | -0.0251 | 0.0322 | 0.1869 | 0.1906 | 0.0695 | 0.0548 | 620.0000 |
| zero_shot: Delta, median agg. | -0.0316 | -0.0587 | -0.0040 | 0.1742 | 0.1426 | 0.0215 | 0.0742 | 620.0000 |
| zero_shot: Generic potential | 0.0282 | 0.0133 | 0.0451 | 0.1653 | 0.1935 | 0.0646 | 0.0452 | 620.0000 |
| zero_shot: Joint, calibrated | -0.0072 | -0.0283 | 0.0143 | 0.0933 | 0.0860 | 0.1742 | 0.4790 | 620.0000 |
| zero_shot: Bidir. profile, calibrated | -0.0074 | -0.0304 | 0.0155 | 0.0944 | 0.0870 | 0.1210 | 0.4968 | 620.0000 |
| zero_shot: Bidir. context, calibrated | 0.0010 | -0.0176 | 0.0211 | 0.0942 | 0.0952 | 0.4161 | 0.9935 | 620.0000 |

##### resolved
| method | aware − blind | CI lo | CI hi | blind log MAE | aware log MAE | blind floor frac | aware floor frac | n rows |
|---|---|---|---|---|---|---|---|---|
| one_known: Absolute response | 0.0059 | -0.0008 | 0.0123 | 0.1111 | 0.1169 | 0.1494 | 0.9990 | 3850.0000 |
| one_known: Delta, uniform agg. | 0.0127 | -0.0040 | 0.0296 | 0.1599 | 0.1727 | 0.0794 | 0.0548 | 3850.0000 |
| one_known: Delta, median agg. | -0.0054 | -0.0220 | 0.0118 | 0.1452 | 0.1398 | 0.0502 | 0.0603 | 3850.0000 |
| one_known: Generic potential | 0.0155 | 0.0050 | 0.0256 | 0.1558 | 0.1712 | 0.0663 | 0.0434 | 3850.0000 |
| one_known: Joint, calibrated | 0.0018 | -0.0155 | 0.0179 | 0.1097 | 0.1115 | 0.1294 | 0.4953 | 3850.0000 |
| one_known: Bidir. profile, calibrated | 0.0024 | -0.0171 | 0.0205 | 0.1097 | 0.1122 | 0.1101 | 0.5223 | 3850.0000 |
| one_known: Bidir. context, calibrated | 0.0048 | -0.0088 | 0.0181 | 0.1121 | 0.1169 | 0.3003 | 1.0000 | 3850.0000 |
| zero_shot: Absolute response | -0.0029 | -0.0121 | 0.0063 | 0.0985 | 0.0956 | 0.1943 | 0.9938 | 1122.0000 |
| zero_shot: Delta, uniform agg. | 0.0022 | -0.0263 | 0.0314 | 0.1917 | 0.1939 | 0.0649 | 0.0633 | 1122.0000 |
| zero_shot: Delta, median agg. | -0.0293 | -0.0549 | -0.0023 | 0.1777 | 0.1484 | 0.0219 | 0.0820 | 1122.0000 |
| zero_shot: Generic potential | 0.0257 | 0.0103 | 0.0435 | 0.1714 | 0.1970 | 0.0631 | 0.0615 | 1122.0000 |
| zero_shot: Joint, calibrated | -0.0069 | -0.0256 | 0.0141 | 0.0968 | 0.0899 | 0.1729 | 0.5784 | 1122.0000 |
| zero_shot: Bidir. profile, calibrated | -0.0075 | -0.0279 | 0.0149 | 0.0981 | 0.0905 | 0.1185 | 0.6034 | 1122.0000 |
| zero_shot: Bidir. context, calibrated | -0.0011 | -0.0175 | 0.0176 | 0.0967 | 0.0956 | 0.4234 | 0.9964 | 1122.0000 |

#### Training panels per size (mean over completed splits): resolved rows, one-sided (app, config) cells, floor rows
| variant | size | rows | floor rows | cells | one-sided cells |
|---|---|---|---|---|---|
| aware | 4.0 | 30.4 | 7.9 | 16.0 | 1.6 |
| aware | 5.0 | 46.3 | 12.9 | 24.4 | 2.6 |
| aware | 6.0 | 60.9 | 16.7 | 32.4 | 3.9 |
| aware | 7.0 | 84.5 | 24.1 | 44.8 | 5.1 |
| aware | 8.0 | 113.8 | 32.8 | 60.0 | 6.2 |
| blind | 4.0 | 16.0 | 2.9 | – | – |
| blind | 5.0 | 24.4 | 4.6 | – | – |
| blind | 6.0 | 32.4 | 6.3 | – | – |
| blind | 7.0 | 44.8 | 8.9 | – | – |
| blind | 8.0 | 60.0 | 12.2 | – | – |

### BD-TT
Completed evaluate_training_size tasks per size (of 10 splits):
| size | aware | blind |
|---|---|---|
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 8 | 8 |
| 5 | 9 | 9 |
| 6 | 10 | 10 |
| 7 | 10 | 10 |
| 8 | 10 | 10 |
- blind: sizes 4-8 completed 47/50; sizes 2-3 completed 0/20; other statuses: {'failed': np.int64(23)}
- aware: sizes 4-8 completed 47/50; sizes 2-3 completed 0/20; other statuses: {'failed': np.int64(23)}
- matched cells (sizes 4-8, both variants): 47
#### C4 format, rep-avg outcomes (47 matched cells): pooled skill (wins/47)
|  | one_known blind | one_known aware | zero_shot blind | zero_shot aware |
|---|---|---|---|---|
| Absolute response | -21.2% (10) | +0.0% (5) | -70.5% (0) | +0.0% (2) |
| Delta, uniform agg. | -74.5% (1) | -54.9% (6) | -113.0% (5) | -73.9% (11) |
| Delta, median agg. | -72.1% (2) | -25.4% (14) | -112.4% (5) | -36.4% (16) |
| Generic potential | -41.9% (8) | -54.2% (7) | -48.5% (8) | -77.4% (7) |
| Joint, calibrated | +2.7% (31) | +1.4% (33) | -4.9% (13) | +7.7% (16) |
| Bidir. profile, calibrated | -4.5% (27) | +1.3% (31) | -7.5% (15) | +7.4% (16) |
| Bidir. context, calibrated | -0.3% (24) | +0.0% (0) | -0.2% (10) | +0.0% (0) |

#### C4 format, resolved outcomes (47 matched cells): pooled skill (wins/47)
|  | one_known blind | one_known aware | zero_shot blind | zero_shot aware |
|---|---|---|---|---|
| Absolute response | -17.3% (14) | -0.1% (0) | -68.1% (1) | +0.1% (3) |
| Delta, uniform agg. | -70.0% (1) | -53.9% (6) | -119.9% (3) | -79.2% (11) |
| Delta, median agg. | -67.8% (1) | -27.7% (11) | -117.0% (4) | -38.0% (15) |
| Generic potential | -39.3% (8) | -49.7% (7) | -57.5% (6) | -87.4% (5) |
| Joint, calibrated | +3.1% (32) | +1.3% (32) | -6.6% (9) | +6.9% (15) |
| Bidir. profile, calibrated | -3.7% (27) | +1.2% (30) | -9.6% (12) | +6.8% (15) |
| Bidir. context, calibrated | +0.0% (24) | +0.0% (0) | -0.7% (10) | +0.0% (0) |

#### Paired aware − blind (log MAE over matched cells, pair-cluster 95% CI)
##### rep-avg
| method | aware − blind | CI lo | CI hi | blind log MAE | aware log MAE | blind floor frac | aware floor frac | n rows |
|---|---|---|---|---|---|---|---|---|
| one_known: Absolute response | -0.0034 | -0.0064 | -0.0008 | 0.0195 | 0.0161 | 0.7546 | 0.9937 | 2054.0000 |
| one_known: Delta, uniform agg. | -0.0032 | -0.0081 | 0.0017 | 0.0280 | 0.0248 | 0.0973 | 0.1704 | 2054.0000 |
| one_known: Delta, median agg. | -0.0075 | -0.0122 | -0.0029 | 0.0277 | 0.0202 | 0.0804 | 0.1918 | 2054.0000 |
| one_known: Generic potential | 0.0020 | -0.0008 | 0.0044 | 0.0226 | 0.0247 | 0.0814 | 0.1027 | 2054.0000 |
| one_known: Joint, calibrated | 0.0002 | -0.0014 | 0.0020 | 0.0157 | 0.0159 | 0.6879 | 0.9620 | 2054.0000 |
| one_known: Bidir. profile, calibrated | -0.0009 | -0.0028 | 0.0008 | 0.0168 | 0.0159 | 0.6967 | 0.9649 | 2054.0000 |
| one_known: Bidir. context, calibrated | -0.0001 | -0.0006 | 0.0005 | 0.0161 | 0.0161 | 0.9333 | 1.0000 | 2054.0000 |
| zero_shot: Absolute response | -0.0104 | -0.0165 | -0.0047 | 0.0249 | 0.0145 | 0.7065 | 0.9919 | 620.0000 |
| zero_shot: Delta, uniform agg. | -0.0052 | -0.0149 | 0.0043 | 0.0345 | 0.0294 | 0.0818 | 0.2194 | 620.0000 |
| zero_shot: Delta, median agg. | -0.0108 | -0.0193 | -0.0028 | 0.0343 | 0.0235 | 0.0634 | 0.2403 | 620.0000 |
| zero_shot: Generic potential | 0.0052 | 0.0003 | 0.0104 | 0.0234 | 0.0286 | 0.0718 | 0.1484 | 620.0000 |
| zero_shot: Joint, calibrated | -0.0023 | -0.0040 | -0.0008 | 0.0158 | 0.0135 | 0.6645 | 0.9274 | 620.0000 |
| zero_shot: Bidir. profile, calibrated | -0.0030 | -0.0054 | -0.0011 | 0.0166 | 0.0136 | 0.6694 | 0.9290 | 620.0000 |
| zero_shot: Bidir. context, calibrated | -0.0006 | -0.0013 | 0.0001 | 0.0151 | 0.0145 | 0.8726 | 1.0000 | 620.0000 |

##### resolved
| method | aware − blind | CI lo | CI hi | blind log MAE | aware log MAE | blind floor frac | aware floor frac | n rows |
|---|---|---|---|---|---|---|---|---|
| one_known: Absolute response | -0.0029 | -0.0059 | -0.0001 | 0.0196 | 0.0167 | 0.7478 | 0.9958 | 3850.0000 |
| one_known: Delta, uniform agg. | -0.0028 | -0.0081 | 0.0023 | 0.0284 | 0.0256 | 0.0927 | 0.2151 | 3850.0000 |
| one_known: Delta, median agg. | -0.0067 | -0.0118 | -0.0019 | 0.0281 | 0.0214 | 0.0747 | 0.2361 | 3850.0000 |
| one_known: Generic potential | 0.0017 | -0.0012 | 0.0042 | 0.0230 | 0.0248 | 0.0823 | 0.1647 | 3850.0000 |
| one_known: Joint, calibrated | 0.0003 | -0.0015 | 0.0022 | 0.0162 | 0.0165 | 0.6865 | 0.9694 | 3850.0000 |
| one_known: Bidir. profile, calibrated | -0.0008 | -0.0027 | 0.0011 | 0.0173 | 0.0165 | 0.6984 | 0.9719 | 3850.0000 |
| one_known: Bidir. context, calibrated | -0.0000 | -0.0006 | 0.0007 | 0.0167 | 0.0167 | 0.9322 | 1.0000 | 3850.0000 |
| zero_shot: Absolute response | -0.0101 | -0.0160 | -0.0043 | 0.0240 | 0.0140 | 0.7005 | 0.9938 | 1122.0000 |
| zero_shot: Delta, uniform agg. | -0.0055 | -0.0147 | 0.0039 | 0.0348 | 0.0293 | 0.0820 | 0.2513 | 1122.0000 |
| zero_shot: Delta, median agg. | -0.0113 | -0.0193 | -0.0035 | 0.0344 | 0.0232 | 0.0673 | 0.2852 | 1122.0000 |
| zero_shot: Generic potential | 0.0054 | 0.0004 | 0.0105 | 0.0240 | 0.0294 | 0.0747 | 0.1970 | 1122.0000 |
| zero_shot: Joint, calibrated | -0.0022 | -0.0039 | -0.0007 | 0.0155 | 0.0132 | 0.6791 | 0.9465 | 1122.0000 |
| zero_shot: Bidir. profile, calibrated | -0.0030 | -0.0054 | -0.0010 | 0.0163 | 0.0133 | 0.6863 | 0.9501 | 1122.0000 |
| zero_shot: Bidir. context, calibrated | -0.0006 | -0.0014 | 0.0000 | 0.0146 | 0.0140 | 0.8904 | 1.0000 | 1122.0000 |

#### Training panels per size (mean over completed splits): resolved rows, one-sided (app, config) cells, floor rows
| variant | size | rows | floor rows | cells | one-sided cells |
|---|---|---|---|---|---|
| aware | 4.0 | 30.4 | 11.4 | 16.0 | 1.6 |
| aware | 5.0 | 46.3 | 18.4 | 24.4 | 2.6 |
| aware | 6.0 | 60.9 | 24.4 | 32.4 | 3.9 |
| aware | 7.0 | 84.5 | 34.8 | 44.8 | 5.1 |
| aware | 8.0 | 113.8 | 46.4 | 60.0 | 6.2 |
| blind | 4.0 | 16.0 | 4.8 | – | – |
| blind | 5.0 | 24.4 | 7.8 | – | – |
| blind | 6.0 | 32.4 | 10.1 | – | – |
| blind | 7.0 | 44.8 | 15.0 | – | – |
| blind | 8.0 | 60.0 | 19.9 | – | – |

