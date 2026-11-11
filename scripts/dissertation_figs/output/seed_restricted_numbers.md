## Seed batch, parts (b) and (c): restricted tiers at N = 30

Ensemble = mean of per-seed latents, then max(0, ·). Pooled skill = 1 − mean(method cell log MAE)/mean(baseline cell log MAE). CIs: pairs-only (the shared 1000 × seed-923 pair draws, the same draw in every cell) and two-level (each pair draw paired with a seed resample). Verdict = the part (a) label rule on Δ = mean cell log MAE(method) − baseline's (jackknife seed SE > 0.002 and |Δ| < 2 SE → sign-unstable; 1.9–2.1 SE → the two-level CI decides).

### Completion vs the artifact (succeeded tasks per size, sizes 4–8)
| size | ('artifact splits', '1W-MPI') | ('artifact splits', '1W-TT') | ('artifact splits', 'BD-MPI') | ('artifact splits', 'BD-TT') | ('succeeded', '1W-MPI') | ('succeeded', '1W-TT') | ('succeeded', 'BD-MPI') | ('succeeded', 'BD-TT') |
|---|---|---|---|---|---|---|---|---|
| 4 | 6 | 4 | 8 | 8 | 6 | 4 | 8 | 8 |
| 5 | 6 | 3 | 9 | 9 | 6 | 3 | 9 | 9 |
| 6 | 7 | 4 | 10 | 10 | 7 | 4 | 10 | 10 |
| 7 | 8 | 5 | 10 | 10 | 8 | 5 | 10 | 10 |
| 8 | 9 | 5 | 10 | 10 | 9 | 5 | 10 | 10 |
- Totals succeeded (sizes 4–8): {'1W-MPI': 36, '1W-TT': 21, 'BD-MPI': 47, 'BD-TT': 47}; artifact: {'1W-MPI': 36, '1W-TT': 21, 'BD-MPI': 47, 'BD-TT': 47}
- C4 cells succeeded (of 20): {'1W-MPI': 20, '1W-TT': 20, 'BD-MPI': 20, 'BD-TT': 20}

Loaded 302 blind (task, tier) cells.

### C4: the 20 matched cells (artifact's common cells), blind, N = 30
Seeds 0–4 must reproduce the artifact's 5-seed C4 skill (column 'artifact skill %').
|  | cells | skill % (N=30) | skill % CI pairs | skill % CI pairs×seeds | skill % seeds 0–4 | 5-seed median skill % | 5-seed 95% | % 5-seed draws skill > 0 | wins | seed SE (Δ log MAE) | verdict | jackknife vs CI disagree | note | FB@1.05 | 5-seed FB median | harmful units | artifact skill % (5 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('1W-MPI', 'one_known', 'Absolute response') | 20 | 0.05 | [-0.1, +0.2] | [-0.6, +15.2] | 9.30 | 0.00 | [-35.9, +16.1] | 49.60 | 4/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 287 | 9.30 |
| ('1W-MPI', 'one_known', 'Delta, uniform agg.') | 20 | -40.43 | [-121.9, +3.9] | [-122.8, +2.9] | -46.73 | -44.08 | [-59.4, -29.6] | 0.00 | 2/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -46.73 |
| ('1W-MPI', 'one_known', 'Delta, median agg.') | 20 | -21.10 | [-84.5, +13.3] | [-84.0, +13.4] | -27.56 | -24.60 | [-41.3, -11.4] | 0.00 | 5/20 | 0.00 | n.s. |  |  | 0.01 | 0.00 | 287 | -27.56 |
| ('1W-MPI', 'one_known', 'Delta, OOD-kernel agg.') | 20 | -21.98 | [-93.5, +18.2] | [-94.0, +16.7] | -27.84 | -24.30 | [-44.0, -11.2] | 0.00 | 6/20 | 0.00 | n.s. |  |  | 0.01 | 0.01 | 287 | -27.84 |
| ('1W-MPI', 'one_known', 'Generic potential') | 20 | -41.20 | [-124.8, +3.5] | [-126.3, +3.9] | -39.80 | -42.53 | [-51.1, -35.1] | 0.00 | 2/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -39.80 |
| ('1W-MPI', 'zero_shot', 'Absolute response') | 20 | -0.21 | [-1.2, +0.5] | [-14.8, +9.8] | -4.12 | -0.16 | [-89.8, +7.8] | 20.60 | 0/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 86 | -4.12 |
| ('1W-MPI', 'zero_shot', 'Delta, uniform agg.') | 20 | -125.96 | [-346.2, -30.6] | [-348.6, -32.8] | -145.17 | -130.81 | [-181.4, -96.2] | 0.00 | 1/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -145.17 |
| ('1W-MPI', 'zero_shot', 'Delta, median agg.') | 20 | -96.59 | [-288.7, -12.1] | [-296.1, -13.1] | -113.21 | -101.79 | [-154.4, -65.3] | 0.00 | 2/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -113.21 |
| ('1W-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | 20 | -103.74 | [-328.6, -6.6] | [-333.5, -6.3] | -113.43 | -106.27 | [-156.2, -82.3] | 0.00 | 2/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -113.43 |
| ('1W-MPI', 'zero_shot', 'Generic potential') | 20 | -110.72 | [-324.8, -18.2] | [-323.3, -18.3] | -114.31 | -111.50 | [-147.2, -88.2] | 0.00 | 1/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -114.31 |
| ('1W-TT', 'one_known', 'Absolute response') | 20 | 0.16 | [-0.5, +0.7] | [-122.6, +9.8] | -20.24 | -6.63 | [-524.8, +6.7] | 15.30 | 3/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 47 | -20.24 |
| ('1W-TT', 'one_known', 'Delta, uniform agg.') | 20 | -36.05 | [-145.7, +11.1] | [-149.0, +10.7] | -46.26 | -42.46 | [-88.7, -12.7] | 0.00 | 6/20 | 0.00 | n.s. |  |  | 0.13 | 0.15 | 47 | -46.26 |
| ('1W-TT', 'one_known', 'Delta, median agg.') | 20 | -40.44 | [-148.9, +6.0] | [-153.4, +6.0] | -53.10 | -47.12 | [-94.8, -15.4] | 0.00 | 6/20 | 0.00 | n.s. |  |  | 0.13 | 0.13 | 47 | -53.10 |
| ('1W-TT', 'one_known', 'Delta, OOD-kernel agg.') | 20 | -19.43 | [-128.6, +24.8] | [-127.7, +24.0] | -30.62 | -26.18 | [-72.9, +1.6] | 3.40 | 7/20 | 0.00 | n.s. |  |  | 0.15 | 0.13 | 47 | -30.62 |
| ('1W-TT', 'one_known', 'Generic potential') | 20 | -34.42 | [-146.0, +12.2] | [-149.9, +11.4] | -32.94 | -37.74 | [-62.1, -19.9] | 0.00 | 4/20 | 0.00 | n.s. |  |  | 0.26 | 0.19 | 47 | -32.94 |
| ('1W-TT', 'zero_shot', 'Absolute response') | 20 | -1.40 | [-3.0, -0.1] | [-233.5, +2.4] | -109.49 | -22.47 | [-817.6, +6.1] | 6.60 | 1/20 | 0.00 | n.s. |  |  | 1.00 | 1.00 | 5 | -109.49 |
| ('1W-TT', 'zero_shot', 'Delta, uniform agg.') | 20 | -111.99 | [-281.3, -41.3] | [-305.4, -41.7] | -88.02 | -125.42 | [-229.6, -62.1] | 0.00 | 2/20 | 0.00 | CI excludes 0 |  |  | 1.00 | 0.60 | 5 | -88.02 |
| ('1W-TT', 'zero_shot', 'Delta, median agg.') | 20 | -126.38 | [-309.0, -50.6] | [-337.3, -51.3] | -104.37 | -140.73 | [-249.0, -72.0] | 0.00 | 2/20 | 0.00 | CI excludes 0 |  |  | 0.80 | 0.60 | 5 | -104.37 |
| ('1W-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 20 | -72.14 | [-239.1, -7.7] | [-268.7, -7.9] | -62.66 | -85.73 | [-186.8, -30.2] | 0.00 | 4/20 | 0.00 | CI excludes 0 |  |  | 0.60 | 0.60 | 5 | -62.66 |
| ('1W-TT', 'zero_shot', 'Generic potential') | 20 | -79.56 | [-257.8, -9.6] | [-261.2, -7.3] | -82.14 | -83.70 | [-144.7, -38.0] | 0.00 | 4/20 | 0.00 | CI excludes 0 |  |  | 0.60 | 0.60 | 5 | -82.14 |
| ('BD-MPI', 'one_known', 'Absolute response') | 20 | 0.05 | [-0.1, +0.2] | [-0.6, +15.2] | 9.30 | 0.00 | [-35.9, +16.1] | 49.60 | 4/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 287 | 9.30 |
| ('BD-MPI', 'one_known', 'Delta, uniform agg.') | 20 | -38.69 | [-121.5, +5.9] | [-121.8, +5.5] | -48.64 | -41.03 | [-56.1, -29.6] | 0.00 | 3/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -48.64 |
| ('BD-MPI', 'one_known', 'Delta, median agg.') | 20 | -18.08 | [-80.0, +15.9] | [-80.4, +16.2] | -29.31 | -20.28 | [-36.7, -10.2] | 0.00 | 5/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -29.31 |
| ('BD-MPI', 'one_known', 'Delta, OOD-kernel agg.') | 20 | -20.58 | [-92.3, +19.3] | [-92.3, +20.0] | -35.40 | -22.36 | [-43.8, -8.9] | 0.00 | 6/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -35.40 |
| ('BD-MPI', 'one_known', 'Generic potential') | 20 | -41.20 | [-124.8, +3.5] | [-126.3, +3.9] | -39.80 | -42.53 | [-51.1, -35.1] | 0.00 | 2/20 | 0.00 | n.s. |  |  | 0.00 | 0.00 | 287 | -39.80 |
| ('BD-MPI', 'one_known', 'Joint') | 20 | 3.46 | [+0.5, +7.0] | [-17.8, +18.5] | 1.48 | 0.00 | [-111.7, +13.4] | 46.00 | 20/20 | 0.01 | sign-unstable |  |  | 1.00 | 0.90 | 287 | 1.48 |
| ('BD-MPI', 'one_known', 'Joint, calibrated') | 20 | 7.67 | [-14.4, +21.1] | [-22.8, +21.1] | 8.41 | 0.29 | [-59.0, +13.0] | 51.20 | 17/20 | 0.00 | n.s. |  |  | 0.11 | 0.11 | 287 | 8.41 |
| ('BD-MPI', 'one_known', 'Bidir. profile') | 20 | 7.50 | [+2.1, +13.0] | [-24.1, +18.5] | 4.57 | 0.00 | [-108.0, +13.7] | 49.40 | 20/20 | 0.01 | sign-unstable |  |  | 0.90 | 0.71 | 287 | 4.57 |
| ('BD-MPI', 'one_known', 'Bidir. profile, calibrated') | 20 | 7.73 | [-16.3, +22.4] | [-25.9, +21.0] | 8.84 | 0.29 | [-60.6, +13.2] | 51.40 | 17/20 | 0.00 | n.s. |  |  | 0.07 | 0.07 | 287 | 8.84 |
| ('BD-MPI', 'one_known', 'Bidir. context') | 20 | 5.40 | [+1.9, +9.5] | [-15.3, +17.9] | 0.10 | 0.00 | [-97.4, +13.7] | 42.00 | 20/20 | 0.01 | sign-unstable |  |  | 1.00 | 0.69 | 287 | 0.10 |
| ('BD-MPI', 'one_known', 'Bidir. context, calibrated') | 20 | 8.87 | [-14.1, +23.4] | [-26.3, +22.4] | 3.29 | -0.35 | [-62.6, +14.0] | 47.80 | 17/20 | 0.00 | n.s. |  | boundary: CI decides | 0.07 | 0.09 | 287 | 3.29 |
| ('BD-MPI', 'zero_shot', 'Absolute response') | 20 | -0.21 | [-1.2, +0.5] | [-14.8, +9.8] | -4.12 | -0.16 | [-89.8, +7.8] | 20.60 | 0/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 86 | -4.12 |
| ('BD-MPI', 'zero_shot', 'Delta, uniform agg.') | 20 | -116.69 | [-328.7, -23.4] | [-331.8, -25.2] | -160.81 | -120.58 | [-170.4, -87.1] | 0.00 | 1/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -160.81 |
| ('BD-MPI', 'zero_shot', 'Delta, median agg.') | 20 | -86.42 | [-276.0, -4.9] | [-280.2, -3.8] | -133.90 | -90.94 | [-141.8, -58.0] | 0.00 | 2/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -133.90 |
| ('BD-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | 20 | -100.49 | [-319.4, -4.1] | [-329.6, -2.0] | -133.78 | -104.01 | [-148.0, -78.5] | 0.00 | 3/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -133.78 |
| ('BD-MPI', 'zero_shot', 'Generic potential') | 20 | -110.72 | [-324.8, -18.2] | [-323.3, -18.3] | -114.31 | -111.50 | [-147.2, -88.2] | 0.00 | 1/20 | 0.01 | CI excludes 0 |  |  | 0.00 | 0.00 | 86 | -114.31 |
| ('BD-MPI', 'zero_shot', 'Joint') | 20 | -0.15 | [-7.2, +4.1] | [-58.9, +19.0] | -0.77 | -0.44 | [-190.7, +9.5] | 23.60 | 12/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.93 | 86 | -0.77 |
| ('BD-MPI', 'zero_shot', 'Joint, calibrated') | 20 | 2.83 | [-59.0, +32.6] | [-83.3, +32.8] | 0.98 | -5.00 | [-100.1, +8.3] | 23.80 | 12/20 | 0.00 | sign-unstable |  |  | 0.26 | 0.22 | 86 | 0.98 |
| ('BD-MPI', 'zero_shot', 'Bidir. profile') | 20 | -0.14 | [-17.9, +8.9] | [-72.3, +18.9] | -2.39 | -2.04 | [-189.0, +7.7] | 23.30 | 10/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.79 | 86 | -2.39 |
| ('BD-MPI', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.52 | [-69.4, +32.3] | [-89.4, +31.8] | -2.03 | -6.75 | [-101.4, +8.2] | 19.80 | 10/20 | 0.00 | sign-unstable |  |  | 0.17 | 0.15 | 86 | -2.03 |
| ('BD-MPI', 'zero_shot', 'Bidir. context') | 20 | -0.07 | [-10.0, +6.0] | [-70.9, +16.6] | 0.00 | -2.83 | [-172.9, +8.2] | 21.00 | 12/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.74 | 86 | 0.00 |
| ('BD-MPI', 'zero_shot', 'Bidir. context, calibrated') | 20 | 2.90 | [-62.9, +34.2] | [-86.4, +34.1] | 0.70 | -7.35 | [-107.5, +8.6] | 22.30 | 11/20 | 0.01 | sign-unstable |  |  | 0.23 | 0.20 | 86 | 0.70 |
| ('BD-TT', 'one_known', 'Absolute response') | 20 | 0.16 | [-0.5, +0.7] | [-122.6, +9.8] | -20.24 | -6.63 | [-524.8, +6.7] | 15.30 | 3/20 | 0.00 | n.s. |  |  | 1.00 | 0.98 | 47 | -20.24 |
| ('BD-TT', 'one_known', 'Delta, uniform agg.') | 20 | -21.59 | [-118.2, +19.2] | [-146.2, +16.9] | -68.81 | -47.10 | [-123.5, -12.5] | 0.00 | 6/20 | 0.00 | sign-unstable |  |  | 0.26 | 0.16 | 47 | -68.81 |
| ('BD-TT', 'one_known', 'Delta, median agg.') | 20 | -15.38 | [-99.3, +21.2] | [-130.0, +19.2] | -70.73 | -40.26 | [-124.9, -6.7] | 0.30 | 8/20 | 0.00 | sign-unstable |  |  | 0.19 | 0.15 | 47 | -70.73 |
| ('BD-TT', 'one_known', 'Delta, OOD-kernel agg.') | 20 | -11.51 | [-120.2, +32.7] | [-142.2, +30.9] | -61.76 | -31.83 | [-118.1, +4.7] | 6.60 | 8/20 | 0.00 | sign-unstable |  |  | 0.09 | 0.11 | 47 | -61.76 |
| ('BD-TT', 'one_known', 'Generic potential') | 20 | -34.42 | [-146.0, +12.2] | [-149.9, +11.4] | -32.94 | -37.74 | [-62.1, -19.9] | 0.00 | 4/20 | 0.00 | n.s. |  |  | 0.26 | 0.19 | 47 | -32.94 |
| ('BD-TT', 'one_known', 'Joint') | 20 | -47.06 | [-110.1, -22.2] | [-535.2, +0.2] | 0.24 | -46.60 | [-1071.2, +2.9] | 7.00 | 0/20 | 0.01 | sign-unstable |  |  | 1.00 | 1.00 | 47 | 0.24 |
| ('BD-TT', 'one_known', 'Joint, calibrated') | 20 | 11.98 | [-8.9, +24.1] | [-224.5, +27.5] | 9.59 | -14.86 | [-504.1, +24.8] | 26.40 | 17/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.96 | 47 | 9.59 |
| ('BD-TT', 'one_known', 'Bidir. profile') | 20 | -31.51 | [-73.9, -15.1] | [-596.1, +0.6] | 0.17 | -31.81 | [-1216.5, +1.7] | 7.70 | 0/20 | 0.01 | sign-unstable |  |  | 1.00 | 1.00 | 47 | 0.17 |
| ('BD-TT', 'one_known', 'Bidir. profile, calibrated') | 20 | 15.83 | [+4.6, +26.5] | [-234.9, +27.6] | 6.85 | -8.31 | [-567.4, +22.1] | 27.60 | 20/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.98 | 47 | 6.85 |
| ('BD-TT', 'one_known', 'Bidir. context') | 20 | -0.11 | [-0.4, +0.0] | [-531.0, +0.8] | 0.00 | -16.13 | [-1258.8, +0.8] | 6.20 | 0/20 | 0.01 | sign-unstable |  |  | 1.00 | 1.00 | 47 | 0.00 |
| ('BD-TT', 'one_known', 'Bidir. context, calibrated') | 20 | 12.06 | [+6.8, +21.1] | [-226.4, +28.0] | 0.90 | -7.05 | [-644.5, +21.0] | 27.30 | 20/20 | 0.00 | sign-unstable |  |  | 1.00 | 0.98 | 47 | 0.90 |
| ('BD-TT', 'zero_shot', 'Absolute response') | 20 | -1.40 | [-3.0, -0.1] | [-233.5, +2.4] | -109.49 | -22.47 | [-817.6, +6.1] | 6.60 | 1/20 | 0.00 | n.s. |  |  | 1.00 | 1.00 | 5 | -109.49 |
| ('BD-TT', 'zero_shot', 'Delta, uniform agg.') | 20 | -59.38 | [-203.6, -1.1] | [-253.4, -2.5] | -157.02 | -101.98 | [-256.0, -19.0] | 0.00 | 4/20 | 0.00 | CI excludes 0 |  | boundary: CI decides | 0.20 | 0.40 | 5 | -157.02 |
| ('BD-TT', 'zero_shot', 'Delta, median agg.') | 20 | -54.59 | [-198.4, +4.4] | [-254.9, +2.1] | -164.81 | -100.79 | [-266.2, -12.7] | 0.70 | 4/20 | 0.00 | sign-unstable |  |  | 0.20 | 0.20 | 5 | -164.81 |
| ('BD-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 20 | -56.89 | [-195.2, +0.2] | [-242.8, -1.4] | -140.56 | -90.67 | [-231.3, -14.3] | 0.10 | 5/20 | 0.00 | CI excludes 0 |  | boundary: CI decides | 0.20 | 0.20 | 5 | -140.56 |
| ('BD-TT', 'zero_shot', 'Generic potential') | 20 | -79.56 | [-257.8, -9.6] | [-261.2, -7.3] | -82.14 | -83.70 | [-144.7, -38.0] | 0.00 | 4/20 | 0.00 | CI excludes 0 |  |  | 0.60 | 0.60 | 5 | -82.14 |
| ('BD-TT', 'zero_shot', 'Joint') | 20 | -74.04 | [-169.7, -30.5] | [-711.7, +0.1] | -1.03 | -83.72 | [-1526.9, +2.5] | 6.60 | 1/20 | 0.02 | sign-unstable |  |  | 1.00 | 1.00 | 5 | -1.03 |
| ('BD-TT', 'zero_shot', 'Joint, calibrated') | 20 | 26.91 | [+0.3, +37.4] | [-284.4, +40.5] | -4.84 | -30.08 | [-709.3, +35.2] | 26.40 | 16/20 | 0.00 | sign-unstable |  |  | 1.00 | 1.00 | 5 | -4.84 |
| ('BD-TT', 'zero_shot', 'Bidir. profile') | 20 | -37.33 | [-99.4, -10.0] | [-843.6, +1.8] | -0.26 | -56.21 | [-1686.1, +1.5] | 6.40 | 2/20 | 0.02 | sign-unstable |  |  | 1.00 | 1.00 | 5 | -0.26 |
| ('BD-TT', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 30.82 | [+12.9, +38.8] | [-309.9, +47.5] | 0.90 | -16.31 | [-798.3, +33.7] | 29.90 | 17/20 | 0.00 | sign-unstable |  |  | 1.00 | 1.00 | 5 | 0.90 |
| ('BD-TT', 'zero_shot', 'Bidir. context') | 20 | 0.00 | [+0.0, +0.0] | [-781.5, +2.6] | 0.00 | -36.26 | [-1769.2, +1.3] | 4.70 | 0/20 | 0.01 | sign-unstable |  |  | 1.00 | 1.00 | 5 | 0.00 |
| ('BD-TT', 'zero_shot', 'Bidir. context, calibrated') | 20 | 14.84 | [+4.1, +27.2] | [-325.3, +46.1] | 0.00 | -9.58 | [-914.2, +31.0] | 26.40 | 14/20 | 0.00 | sign-unstable |  |  | 1.00 | 1.00 | 5 | 0.00 |

### Learning curves: pooled skill per size over all succeeded splits, and the slope of skill vs size
Slope = OLS of per-size pooled skill (pp) on size (4–8); CI from the same pair/seed draws applied at every size. 'No gain with size' holds when the two-level CI's lower end is ≤ 0.
|  | slope (skill pp / app) | CI pairs | CI pairs×seeds | seed SE (jackknife) | 5-seed median slope | % 5-seed draws slope > 0 | 'no gain with size' (slope CI not > 0) |
|---|---|---|---|---|---|---|---|
| ('1W-MPI', 'one_known', 'Absolute response') | -0.05 | [-0.11, +0.04] | [-1.21, +0.24] | 0.50 | -0.04 | 18.00 | holds |
| ('1W-MPI', 'one_known', 'Delta, uniform agg.') | -8.83 | [-18.16, -2.70] | [-18.02, -2.47] | 0.75 | -8.43 | 0.00 | holds |
| ('1W-MPI', 'one_known', 'Delta, median agg.') | -6.19 | [-13.58, -1.14] | [-13.47, -0.83] | 0.74 | -5.68 | 0.40 | holds |
| ('1W-MPI', 'one_known', 'Delta, OOD-kernel agg.') | -3.58 | [-11.07, +3.50] | [-11.30, +3.32] | 0.65 | -3.39 | 1.70 | holds |
| ('1W-MPI', 'one_known', 'Generic potential') | -9.37 | [-18.91, -3.24] | [-18.81, -3.19] | 0.31 | -9.48 | 0.00 | holds |
| ('1W-MPI', 'zero_shot', 'Absolute response') | 0.11 | [-0.21, +0.48] | [-2.64, +2.84] | 0.23 | 0.00 | 46.80 | holds |
| ('1W-MPI', 'zero_shot', 'Delta, uniform agg.') | -16.11 | [-122.13, +14.88] | [-118.93, +15.66] | 2.14 | -15.17 | 0.40 | holds |
| ('1W-MPI', 'zero_shot', 'Delta, median agg.') | -11.58 | [-97.15, +16.42] | [-97.24, +17.34] | 2.27 | -11.18 | 3.10 | holds |
| ('1W-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | -13.76 | [-104.47, +14.33] | [-106.77, +15.25] | 2.09 | -13.13 | 0.60 | holds |
| ('1W-MPI', 'zero_shot', 'Generic potential') | -16.40 | [-123.78, +14.21] | [-121.47, +14.10] | 0.82 | -16.25 | 0.00 | holds |
| ('1W-TT', 'one_known', 'Absolute response') | -0.18 | [-0.71, +0.45] | [-1.72, +11.39] | 0.70 | 0.98 | 62.30 | holds |
| ('1W-TT', 'one_known', 'Delta, uniform agg.') | 1.47 | [-10.94, +14.86] | [-11.46, +16.84] | 2.40 | 3.12 | 76.30 | holds |
| ('1W-TT', 'one_known', 'Delta, median agg.') | -0.04 | [-13.03, +13.68] | [-13.73, +15.37] | 2.47 | 1.49 | 62.80 | holds |
| ('1W-TT', 'one_known', 'Delta, OOD-kernel agg.') | 3.34 | [-9.09, +16.23] | [-9.64, +16.81] | 2.50 | 5.10 | 90.60 | holds |
| ('1W-TT', 'one_known', 'Generic potential') | -2.34 | [-16.94, +9.83] | [-16.79, +10.28] | 1.08 | -1.53 | 26.00 | holds |
| ('1W-TT', 'zero_shot', 'Absolute response') | 0.98 | [-0.01, +2.23] | [-23.32, +14.99] | 4.30 | 0.00 | 41.80 | holds |
| ('1W-TT', 'zero_shot', 'Delta, uniform agg.') | -4.04 | [-40.89, +28.24] | [-41.76, +34.14] | 5.70 | -2.35 | 42.40 | holds |
| ('1W-TT', 'zero_shot', 'Delta, median agg.') | -7.72 | [-51.11, +25.26] | [-53.12, +32.18] | 6.72 | -6.06 | 29.90 | holds |
| ('1W-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 6.79 | [-22.69, +35.69] | [-24.86, +43.02] | 5.75 | 9.51 | 85.30 | holds |
| ('1W-TT', 'zero_shot', 'Generic potential') | -0.44 | [-43.85, +28.55] | [-43.40, +30.83] | 2.92 | -0.06 | 49.50 | holds |
| ('BD-MPI', 'one_known', 'Absolute response') | -0.03 | [-0.08, +0.03] | [-0.87, +0.14] | 0.37 | -0.02 | 19.40 | holds |
| ('BD-MPI', 'one_known', 'Delta, uniform agg.') | -4.65 | [-11.13, +1.31] | [-11.51, +1.22] | 0.44 | -4.56 | 0.00 | holds |
| ('BD-MPI', 'one_known', 'Delta, median agg.') | -2.42 | [-7.61, +3.10] | [-7.69, +3.03] | 0.39 | -2.26 | 1.30 | holds |
| ('BD-MPI', 'one_known', 'Delta, OOD-kernel agg.') | -0.56 | [-5.78, +5.72] | [-6.01, +5.77] | 0.34 | -0.41 | 28.30 | holds |
| ('BD-MPI', 'one_known', 'Generic potential') | -5.17 | [-11.92, +0.82] | [-11.83, +0.78] | 0.22 | -5.17 | 0.00 | holds |
| ('BD-MPI', 'one_known', 'Joint') | 0.20 | [-0.08, +0.45] | [-2.16, +0.52] | 0.55 | -0.12 | 17.70 | holds |
| ('BD-MPI', 'one_known', 'Joint, calibrated') | -0.77 | [-2.94, +1.15] | [-3.65, +1.08] | 0.59 | -0.82 | 11.50 | holds |
| ('BD-MPI', 'one_known', 'Bidir. profile') | 0.13 | [-0.39, +0.51] | [-2.40, +0.50] | 0.85 | -0.20 | 18.90 | holds |
| ('BD-MPI', 'one_known', 'Bidir. profile, calibrated') | -1.02 | [-3.36, +1.07] | [-4.01, +1.03] | 0.66 | -0.97 | 8.10 | holds |
| ('BD-MPI', 'one_known', 'Bidir. context') | -0.02 | [-0.46, +0.37] | [-2.03, +0.69] | 0.80 | -0.15 | 23.30 | holds |
| ('BD-MPI', 'one_known', 'Bidir. context, calibrated') | -0.57 | [-2.90, +1.55] | [-3.82, +1.62] | 0.69 | -0.47 | 26.40 | holds |
| ('BD-MPI', 'zero_shot', 'Absolute response') | 0.07 | [-0.14, +0.31] | [-1.20, +2.71] | 0.14 | 0.13 | 59.70 | holds |
| ('BD-MPI', 'zero_shot', 'Delta, uniform agg.') | 7.64 | [-56.04, +28.05] | [-51.15, +28.51] | 0.99 | 8.18 | 100.00 | holds |
| ('BD-MPI', 'zero_shot', 'Delta, median agg.') | 9.01 | [-39.47, +27.80] | [-38.25, +28.65] | 1.07 | 9.20 | 100.00 | holds |
| ('BD-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | 7.13 | [-46.39, +26.73] | [-47.50, +26.59] | 0.88 | 7.21 | 100.00 | holds |
| ('BD-MPI', 'zero_shot', 'Generic potential') | 7.27 | [-56.07, +27.38] | [-53.79, +26.79] | 0.59 | 7.49 | 100.00 | holds |
| ('BD-MPI', 'zero_shot', 'Joint') | 0.00 | [-1.37, +1.22] | [-5.16, +5.41] | 0.82 | 0.13 | 61.00 | holds |
| ('BD-MPI', 'zero_shot', 'Joint, calibrated') | 1.09 | [-20.17, +7.58] | [-20.33, +8.93] | 0.47 | 1.89 | 98.80 | holds |
| ('BD-MPI', 'zero_shot', 'Bidir. profile') | -0.03 | [-2.82, +1.60] | [-6.32, +6.01] | 0.74 | 0.29 | 65.20 | holds |
| ('BD-MPI', 'zero_shot', 'Bidir. profile, calibrated') | 1.04 | [-22.37, +8.11] | [-23.08, +9.39] | 0.44 | 1.92 | 98.60 | holds |
| ('BD-MPI', 'zero_shot', 'Bidir. context') | -0.54 | [-2.58, +0.62] | [-6.41, +5.49] | 1.08 | 0.29 | 60.20 | holds |
| ('BD-MPI', 'zero_shot', 'Bidir. context, calibrated') | 1.56 | [-20.70, +8.63] | [-21.96, +9.86] | 0.79 | 2.39 | 98.50 | holds |
| ('BD-TT', 'one_known', 'Absolute response') | -0.15 | [-0.42, +0.23] | [-2.53, +3.99] | 0.69 | 0.00 | 39.20 | holds |
| ('BD-TT', 'one_known', 'Delta, uniform agg.') | -5.89 | [-11.68, -0.08] | [-11.48, +1.51] | 1.06 | -4.66 | 3.00 | holds |
| ('BD-TT', 'one_known', 'Delta, median agg.') | -2.82 | [-7.35, +2.93] | [-7.58, +4.88] | 0.91 | -1.60 | 17.80 | holds |
| ('BD-TT', 'one_known', 'Delta, OOD-kernel agg.') | -2.39 | [-6.94, +4.13] | [-6.88, +5.71] | 1.20 | -0.89 | 31.50 | holds |
| ('BD-TT', 'one_known', 'Generic potential') | -6.49 | [-12.65, -0.17] | [-12.48, +0.23] | 0.45 | -5.89 | 0.00 | holds |
| ('BD-TT', 'one_known', 'Joint') | -3.55 | [-6.74, +0.36] | [-18.21, +4.34] | 5.62 | -2.17 | 20.70 | holds |
| ('BD-TT', 'one_known', 'Joint, calibrated') | -1.90 | [-4.37, +0.70] | [-11.66, +1.83] | 4.38 | -1.92 | 21.00 | holds |
| ('BD-TT', 'one_known', 'Bidir. profile') | -2.51 | [-4.70, -0.18] | [-19.23, +3.72] | 5.10 | -1.24 | 19.70 | holds |
| ('BD-TT', 'one_known', 'Bidir. profile, calibrated') | -0.39 | [-2.06, +1.51] | [-12.49, +1.76] | 4.09 | -1.46 | 22.90 | holds |
| ('BD-TT', 'one_known', 'Bidir. context') | -0.10 | [-0.50, +0.60] | [-16.94, +16.05] | 1.06 | 0.00 | 30.00 | holds |
| ('BD-TT', 'one_known', 'Bidir. context, calibrated') | 0.01 | [-1.25, +1.20] | [-12.80, +4.64] | 2.17 | -0.39 | 28.30 | holds |
| ('BD-TT', 'zero_shot', 'Absolute response') | 0.65 | [+0.02, +1.40] | [-11.60, +14.04] | 2.61 | 1.91 | 69.30 | holds |
| ('BD-TT', 'zero_shot', 'Delta, uniform agg.') | 12.84 | [-16.98, +27.50] | [-22.51, +31.95] | 2.18 | 16.21 | 100.00 | holds |
| ('BD-TT', 'zero_shot', 'Delta, median agg.') | 12.14 | [-9.39, +26.24] | [-16.56, +30.26] | 2.40 | 15.63 | 100.00 | holds |
| ('BD-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 11.02 | [-8.14, +26.43] | [-13.95, +29.74] | 2.43 | 15.06 | 100.00 | holds |
| ('BD-TT', 'zero_shot', 'Generic potential') | 14.17 | [-21.13, +31.68] | [-21.64, +32.41] | 1.33 | 15.35 | 100.00 | holds |
| ('BD-TT', 'zero_shot', 'Joint') | 0.57 | [-38.67, +11.13] | [-51.11, +42.59] | 5.89 | 3.47 | 66.50 | holds |
| ('BD-TT', 'zero_shot', 'Joint, calibrated') | -0.64 | [-20.47, +6.23] | [-27.26, +19.58] | 2.71 | 2.84 | 72.30 | holds |
| ('BD-TT', 'zero_shot', 'Bidir. profile') | 0.27 | [-23.56, +6.60] | [-54.11, +44.43] | 4.29 | 2.26 | 65.10 | holds |
| ('BD-TT', 'zero_shot', 'Bidir. profile, calibrated') | -0.01 | [-12.22, +4.91] | [-26.36, +19.18] | 2.62 | 1.86 | 67.70 | holds |
| ('BD-TT', 'zero_shot', 'Bidir. context') | -0.79 | [-3.58, +0.61] | [-32.45, +46.54] | 5.49 | 1.13 | 57.30 | holds |
| ('BD-TT', 'zero_shot', 'Bidir. context, calibrated') | 0.11 | [-9.00, +4.39] | [-22.52, +19.29] | 2.00 | 1.07 | 58.20 | holds |

#### Per-size pooled skill (%)
|  | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| ('1W-MPI', 'one_known', 'Absolute response') | +0.2 | +0.1 | +0.0 | +0.0 | -0.0 |
| ('1W-MPI', 'one_known', 'Delta, OOD-kernel agg.') | -24.1 | -14.4 | -14.7 | -29.5 | -34.5 |
| ('1W-MPI', 'one_known', 'Delta, median agg.') | -23.6 | -8.9 | -18.3 | -34.7 | -41.6 |
| ('1W-MPI', 'one_known', 'Delta, uniform agg.') | -38.5 | -23.5 | -37.5 | -54.6 | -67.1 |
| ('1W-MPI', 'one_known', 'Generic potential') | -36.9 | -24.2 | -36.8 | -55.5 | -68.1 |
| ('1W-MPI', 'zero_shot', 'Absolute response') | -0.3 | -0.5 | -0.1 | -0.0 | +0.0 |
| ('1W-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | -57.0 | -77.5 | -141.6 | -186.2 | -71.4 |
| ('1W-MPI', 'zero_shot', 'Delta, median agg.') | -60.9 | -80.1 | -132.3 | -181.6 | -68.0 |
| ('1W-MPI', 'zero_shot', 'Delta, uniform agg.') | -72.9 | -99.9 | -166.1 | -222.8 | -92.0 |
| ('1W-MPI', 'zero_shot', 'Generic potential') | -59.7 | -92.2 | -151.0 | -211.2 | -82.2 |
| ('1W-TT', 'one_known', 'Absolute response') | +0.6 | +0.5 | -0.0 | +0.0 | +0.0 |
| ('1W-TT', 'one_known', 'Delta, OOD-kernel agg.') | -13.8 | -42.4 | -15.9 | -15.1 | -10.8 |
| ('1W-TT', 'one_known', 'Delta, median agg.') | -26.4 | -57.4 | -36.1 | -41.8 | -34.3 |
| ('1W-TT', 'one_known', 'Delta, uniform agg.') | -24.5 | -56.0 | -34.7 | -33.7 | -28.3 |
| ('1W-TT', 'one_known', 'Generic potential') | -20.1 | -41.5 | -27.6 | -37.1 | -34.0 |
| ('1W-TT', 'zero_shot', 'Absolute response') | -3.3 | -3.3 | +0.0 | +0.0 | +0.0 |
| ('1W-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | -40.7 | -115.1 | -131.0 | -117.0 | -5.7 |
| ('1W-TT', 'zero_shot', 'Delta, median agg.') | -62.3 | -149.0 | -187.5 | -206.9 | -72.0 |
| ('1W-TT', 'zero_shot', 'Delta, uniform agg.') | -54.2 | -143.0 | -174.4 | -178.9 | -56.4 |
| ('1W-TT', 'zero_shot', 'Generic potential') | -43.3 | -104.9 | -111.9 | -129.5 | -33.2 |
| ('BD-MPI', 'one_known', 'Absolute response') | +0.1 | +0.1 | +0.0 | +0.0 | -0.0 |
| ('BD-MPI', 'one_known', 'Bidir. context') | +6.4 | +7.1 | +7.4 | +6.7 | +6.4 |
| ('BD-MPI', 'one_known', 'Bidir. context, calibrated') | +6.5 | +8.1 | +8.9 | +6.3 | +4.5 |
| ('BD-MPI', 'one_known', 'Bidir. profile') | +7.6 | +8.6 | +9.2 | +8.7 | +8.1 |
| ('BD-MPI', 'one_known', 'Bidir. profile, calibrated') | +6.7 | +7.8 | +6.9 | +4.7 | +3.1 |
| ('BD-MPI', 'one_known', 'Delta, OOD-kernel agg.') | -21.9 | -16.5 | -23.0 | -22.9 | -21.5 |
| ('BD-MPI', 'one_known', 'Delta, median agg.') | -23.9 | -16.2 | -34.7 | -32.1 | -28.1 |
| ('BD-MPI', 'one_known', 'Delta, uniform agg.') | -39.9 | -34.0 | -49.9 | -53.1 | -53.6 |
| ('BD-MPI', 'one_known', 'Generic potential') | -41.2 | -35.4 | -52.1 | -56.0 | -56.7 |
| ('BD-MPI', 'one_known', 'Joint') | +3.7 | +4.7 | +5.0 | +4.8 | +4.6 |
| ('BD-MPI', 'one_known', 'Joint, calibrated') | +6.6 | +8.2 | +7.7 | +5.6 | +4.1 |
| ('BD-MPI', 'zero_shot', 'Absolute response') | -0.2 | -0.2 | -0.0 | -0.0 | +0.0 |
| ('BD-MPI', 'zero_shot', 'Bidir. context') | +5.6 | +4.5 | +4.6 | +4.1 | +3.0 |
| ('BD-MPI', 'zero_shot', 'Bidir. context, calibrated') | +0.4 | -2.2 | +0.5 | +1.8 | +6.2 |
| ('BD-MPI', 'zero_shot', 'Bidir. profile') | +6.4 | +4.7 | +5.2 | +5.6 | +5.8 |
| ('BD-MPI', 'zero_shot', 'Bidir. profile, calibrated') | +2.9 | +1.5 | +2.6 | +2.1 | +7.8 |
| ('BD-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | -53.0 | -56.5 | -58.5 | -56.0 | -17.6 |
| ('BD-MPI', 'zero_shot', 'Delta, median agg.') | -62.6 | -67.8 | -64.7 | -62.1 | -20.4 |
| ('BD-MPI', 'zero_shot', 'Delta, uniform agg.') | -71.2 | -81.9 | -78.3 | -79.9 | -34.0 |
| ('BD-MPI', 'zero_shot', 'Generic potential') | -70.2 | -80.9 | -77.7 | -79.5 | -34.5 |
| ('BD-MPI', 'zero_shot', 'Joint') | +4.5 | +4.0 | +3.8 | +4.4 | +4.3 |
| ('BD-MPI', 'zero_shot', 'Joint, calibrated') | +3.8 | +2.8 | +4.0 | +4.1 | +8.6 |
| ('BD-TT', 'one_known', 'Absolute response') | +0.7 | +0.2 | -0.0 | +0.0 | +0.0 |
| ('BD-TT', 'one_known', 'Bidir. context') | -0.6 | -1.0 | -1.3 | -1.2 | -0.9 |
| ('BD-TT', 'one_known', 'Bidir. context, calibrated') | +13.2 | +14.1 | +15.0 | +14.4 | +13.2 |
| ('BD-TT', 'one_known', 'Bidir. profile') | -22.7 | -24.5 | -26.9 | -28.7 | -33.2 |
| ('BD-TT', 'one_known', 'Bidir. profile, calibrated') | +13.5 | +13.7 | +15.4 | +13.2 | +11.8 |
| ('BD-TT', 'one_known', 'Delta, OOD-kernel agg.') | -1.1 | -4.6 | -14.5 | -20.5 | -5.0 |
| ('BD-TT', 'one_known', 'Delta, median agg.') | -8.5 | -11.1 | -23.9 | -25.0 | -15.7 |
| ('BD-TT', 'one_known', 'Delta, uniform agg.') | -13.6 | -16.4 | -32.7 | -38.1 | -32.2 |
| ('BD-TT', 'one_known', 'Generic potential') | -23.8 | -28.6 | -44.8 | -50.4 | -45.3 |
| ('BD-TT', 'one_known', 'Joint') | -36.4 | -35.3 | -40.0 | -42.9 | -50.3 |
| ('BD-TT', 'one_known', 'Joint, calibrated') | +11.5 | +11.1 | +10.5 | +7.4 | +3.9 |
| ('BD-TT', 'zero_shot', 'Absolute response') | -2.7 | -1.0 | +0.0 | +0.0 | +0.0 |
| ('BD-TT', 'zero_shot', 'Bidir. context') | -0.6 | -0.2 | -0.8 | -2.2 | -3.5 |
| ('BD-TT', 'zero_shot', 'Bidir. context, calibrated') | +4.9 | +7.7 | +4.6 | +11.6 | +3.5 |
| ('BD-TT', 'zero_shot', 'Bidir. profile') | -22.8 | -26.5 | -23.0 | -25.3 | -22.1 |
| ('BD-TT', 'zero_shot', 'Bidir. profile, calibrated') | +12.3 | +8.7 | +10.1 | +7.9 | +12.7 |
| ('BD-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | -29.3 | -53.0 | -43.7 | -28.7 | +13.7 |
| ('BD-TT', 'zero_shot', 'Delta, median agg.') | -27.9 | -50.9 | -33.9 | -32.3 | +23.5 |
| ('BD-TT', 'zero_shot', 'Delta, uniform agg.') | -33.8 | -59.5 | -48.0 | -38.8 | +20.1 |
| ('BD-TT', 'zero_shot', 'Generic potential') | -52.8 | -77.8 | -66.0 | -55.2 | +6.7 |
| ('BD-TT', 'zero_shot', 'Joint') | -47.3 | -50.5 | -47.1 | -50.1 | -44.6 |
| ('BD-TT', 'zero_shot', 'Joint, calibrated') | +6.4 | +2.5 | +3.1 | -1.6 | +5.3 |

### Restricted false-benign rate (FB@1.05), pooled over all succeeded cells, sizes 4–8
|  | cells | harmful units | FB (N=30) | FB seeds 0–4 | 5-seed FB median | 5-seed FB 95% |
|---|---|---|---|---|---|---|
| ('1W-MPI', 'one_known', 'Isolated-speed baseline') | 36 | 510 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('1W-MPI', 'one_known', 'Absolute response') | 36 | 510 | 1.000 | 0.608 | 0.967 | [0.00, 1.00] |
| ('1W-MPI', 'one_known', 'Delta, uniform agg.') | 36 | 510 | 0.002 | 0.002 | 0.002 | [0.00, 0.01] |
| ('1W-MPI', 'one_known', 'Delta, median agg.') | 36 | 510 | 0.004 | 0.002 | 0.002 | [0.00, 0.02] |
| ('1W-MPI', 'one_known', 'Delta, OOD-kernel agg.') | 36 | 510 | 0.006 | 0.004 | 0.006 | [0.00, 0.04] |
| ('1W-MPI', 'one_known', 'Generic potential') | 36 | 510 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] |
| ('1W-MPI', 'zero_shot', 'Isolated-speed baseline') | 36 | 129 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('1W-MPI', 'zero_shot', 'Absolute response') | 36 | 129 | 1.000 | 0.535 | 0.957 | [0.00, 1.00] |
| ('1W-MPI', 'zero_shot', 'Delta, uniform agg.') | 36 | 129 | 0.000 | 0.000 | 0.000 | [0.00, 0.02] |
| ('1W-MPI', 'zero_shot', 'Delta, median agg.') | 36 | 129 | 0.000 | 0.000 | 0.000 | [0.00, 0.02] |
| ('1W-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | 36 | 129 | 0.000 | 0.000 | 0.000 | [0.00, 0.02] |
| ('1W-MPI', 'zero_shot', 'Generic potential') | 36 | 129 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] |
| ('1W-TT', 'one_known', 'Isolated-speed baseline') | 21 | 50 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('1W-TT', 'one_known', 'Absolute response') | 21 | 50 | 1.000 | 0.540 | 0.980 | [0.00, 1.00] |
| ('1W-TT', 'one_known', 'Delta, uniform agg.') | 21 | 50 | 0.140 | 0.200 | 0.140 | [0.10, 0.20] |
| ('1W-TT', 'one_known', 'Delta, median agg.') | 21 | 50 | 0.140 | 0.180 | 0.140 | [0.10, 0.20] |
| ('1W-TT', 'one_known', 'Delta, OOD-kernel agg.') | 21 | 50 | 0.160 | 0.180 | 0.140 | [0.08, 0.20] |
| ('1W-TT', 'one_known', 'Generic potential') | 21 | 50 | 0.240 | 0.120 | 0.180 | [0.10, 0.32] |
| ('1W-TT', 'zero_shot', 'Isolated-speed baseline') | 21 | 6 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('1W-TT', 'zero_shot', 'Absolute response') | 21 | 6 | 1.000 | 0.833 | 1.000 | [0.00, 1.00] |
| ('1W-TT', 'zero_shot', 'Delta, uniform agg.') | 21 | 6 | 0.833 | 0.833 | 0.500 | [0.00, 0.83] |
| ('1W-TT', 'zero_shot', 'Delta, median agg.') | 21 | 6 | 0.667 | 0.833 | 0.500 | [0.00, 0.83] |
| ('1W-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 21 | 6 | 0.500 | 0.833 | 0.500 | [0.00, 0.83] |
| ('1W-TT', 'zero_shot', 'Generic potential') | 21 | 6 | 0.500 | 0.167 | 0.500 | [0.00, 0.83] |
| ('BD-MPI', 'one_known', 'Isolated-speed baseline') | 47 | 688 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('BD-MPI', 'one_known', 'Absolute response') | 47 | 688 | 1.000 | 0.692 | 0.975 | [0.00, 1.00] |
| ('BD-MPI', 'one_known', 'Delta, uniform agg.') | 47 | 681 | 0.000 | 0.000 | 0.000 | [0.00, 0.01] |
| ('BD-MPI', 'one_known', 'Delta, median agg.') | 47 | 681 | 0.000 | 0.000 | 0.001 | [0.00, 0.02] |
| ('BD-MPI', 'one_known', 'Delta, OOD-kernel agg.') | 47 | 681 | 0.001 | 0.001 | 0.003 | [0.00, 0.03] |
| ('BD-MPI', 'one_known', 'Generic potential') | 47 | 681 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] |
| ('BD-MPI', 'one_known', 'Joint') | 47 | 688 | 0.997 | 0.890 | 0.879 | [0.00, 1.00] |
| ('BD-MPI', 'one_known', 'Joint, calibrated') | 47 | 688 | 0.070 | 0.106 | 0.069 | [0.00, 0.81] |
| ('BD-MPI', 'one_known', 'Bidir. profile') | 47 | 688 | 0.887 | 0.751 | 0.706 | [0.00, 1.00] |
| ('BD-MPI', 'one_known', 'Bidir. profile, calibrated') | 47 | 688 | 0.041 | 0.084 | 0.044 | [0.00, 0.74] |
| ('BD-MPI', 'one_known', 'Bidir. context') | 47 | 688 | 0.967 | 0.968 | 0.642 | [0.00, 1.00] |
| ('BD-MPI', 'one_known', 'Bidir. context, calibrated') | 47 | 688 | 0.048 | 0.360 | 0.058 | [0.00, 0.85] |
| ('BD-MPI', 'zero_shot', 'Isolated-speed baseline') | 47 | 186 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('BD-MPI', 'zero_shot', 'Absolute response') | 47 | 186 | 1.000 | 0.667 | 0.962 | [0.00, 1.00] |
| ('BD-MPI', 'zero_shot', 'Delta, uniform agg.') | 47 | 184 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] |
| ('BD-MPI', 'zero_shot', 'Delta, median agg.') | 47 | 184 | 0.000 | 0.000 | 0.000 | [0.00, 0.01] |
| ('BD-MPI', 'zero_shot', 'Delta, OOD-kernel agg.') | 47 | 184 | 0.000 | 0.000 | 0.000 | [0.00, 0.03] |
| ('BD-MPI', 'zero_shot', 'Generic potential') | 47 | 184 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] |
| ('BD-MPI', 'zero_shot', 'Joint') | 47 | 186 | 1.000 | 0.801 | 0.866 | [0.00, 1.00] |
| ('BD-MPI', 'zero_shot', 'Joint, calibrated') | 47 | 186 | 0.145 | 0.194 | 0.137 | [0.00, 0.77] |
| ('BD-MPI', 'zero_shot', 'Bidir. profile') | 47 | 186 | 0.957 | 0.656 | 0.699 | [0.00, 1.00] |
| ('BD-MPI', 'zero_shot', 'Bidir. profile, calibrated') | 47 | 186 | 0.091 | 0.124 | 0.086 | [0.00, 0.70] |
| ('BD-MPI', 'zero_shot', 'Bidir. context') | 47 | 186 | 0.984 | 0.952 | 0.664 | [0.00, 1.00] |
| ('BD-MPI', 'zero_shot', 'Bidir. context, calibrated') | 47 | 186 | 0.129 | 0.495 | 0.113 | [0.00, 0.80] |
| ('BD-TT', 'one_known', 'Isolated-speed baseline') | 47 | 109 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('BD-TT', 'one_known', 'Absolute response') | 47 | 109 | 1.000 | 0.734 | 0.991 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Delta, uniform agg.') | 47 | 109 | 0.138 | 0.018 | 0.110 | [0.02, 0.28] |
| ('BD-TT', 'one_known', 'Delta, median agg.') | 47 | 109 | 0.165 | 0.018 | 0.138 | [0.02, 0.42] |
| ('BD-TT', 'one_known', 'Delta, OOD-kernel agg.') | 47 | 109 | 0.110 | 0.092 | 0.128 | [0.06, 0.23] |
| ('BD-TT', 'one_known', 'Generic potential') | 47 | 109 | 0.147 | 0.101 | 0.119 | [0.06, 0.19] |
| ('BD-TT', 'one_known', 'Joint') | 47 | 109 | 1.000 | 0.972 | 0.972 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Joint, calibrated') | 47 | 109 | 0.917 | 0.817 | 0.835 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Bidir. profile') | 47 | 109 | 1.000 | 0.963 | 1.000 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Bidir. profile, calibrated') | 47 | 109 | 0.963 | 0.752 | 0.876 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Bidir. context') | 47 | 109 | 1.000 | 1.000 | 1.000 | [0.00, 1.00] |
| ('BD-TT', 'one_known', 'Bidir. context, calibrated') | 47 | 109 | 0.936 | 0.853 | 0.853 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Isolated-speed baseline') | 47 | 35 | 1.000 | 1.000 | 1.000 | [1.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Absolute response') | 47 | 35 | 1.000 | 0.829 | 1.000 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Delta, uniform agg.') | 47 | 35 | 0.086 | 0.000 | 0.114 | [0.00, 0.29] |
| ('BD-TT', 'zero_shot', 'Delta, median agg.') | 47 | 35 | 0.086 | 0.000 | 0.143 | [0.00, 0.37] |
| ('BD-TT', 'zero_shot', 'Delta, OOD-kernel agg.') | 47 | 35 | 0.029 | 0.000 | 0.143 | [0.00, 0.46] |
| ('BD-TT', 'zero_shot', 'Generic potential') | 47 | 35 | 0.200 | 0.057 | 0.143 | [0.00, 0.26] |
| ('BD-TT', 'zero_shot', 'Joint') | 47 | 35 | 1.000 | 1.000 | 0.971 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Joint, calibrated') | 47 | 35 | 0.771 | 0.600 | 0.714 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Bidir. profile') | 47 | 35 | 1.000 | 0.857 | 1.000 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Bidir. profile, calibrated') | 47 | 35 | 0.914 | 0.400 | 0.800 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Bidir. context') | 47 | 35 | 1.000 | 1.000 | 1.000 | [0.00, 1.00] |
| ('BD-TT', 'zero_shot', 'Bidir. context, calibrated') | 47 | 35 | 0.829 | 0.657 | 0.800 | [0.00, 1.00] |

### (c) Socket subsection, BD, the 20 C4 cells: E1 − D1 (information) and D1 − blind (architecture/init)
#### information E1 − D1
|  | cells | E1 log MAE | D1 log MAE | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | 5-seed median Δ | % 5-seed draws Δ < 0 | verdict | jackknife vs CI disagree | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'one_known', 'Absolute response') | 20 | 0.1165 | 0.1166 | -0.0001 | [-0.0001, -0.0000] | [-0.0018, +0.0001] | 0.0005 | -0.0000 | 53.5000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Delta, uniform agg.') | 20 | 0.1684 | 0.1701 | -0.0017 | [-0.0028, -0.0006] | [-0.0039, +0.0007] | 0.0011 | -0.0021 | 75.2000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Delta, median agg.') | 20 | 0.1355 | 0.1368 | -0.0013 | [-0.0027, -0.0000] | [-0.0036, +0.0011] | 0.0011 | -0.0016 | 69.4000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Generic potential') | 20 | 0.1691 | 0.1700 | -0.0009 | [-0.0016, -0.0003] | [-0.0020, +0.0001] | 0.0005 | -0.0008 | 79.0000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Joint, calibrated') | 20 | 0.1112 | 0.1112 | -0.0000 | [-0.0002, +0.0002] | [-0.0008, +0.0005] | 0.0004 | -0.0000 | 53.3000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.1088 | 0.1090 | -0.0002 | [-0.0004, -0.0000] | [-0.0011, +0.0004] | 0.0004 | -0.0001 | 60.8000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'one_known', 'Bidir. context, calibrated') | 20 | 0.1115 | 0.1112 | 0.0003 | [+0.0001, +0.0005] | [-0.0007, +0.0012] | 0.0005 | 0.0001 | 44.3000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Absolute response') | 20 | 0.0797 | 0.0799 | -0.0002 | [-0.0003, -0.0001] | [-0.0013, +0.0014] | 0.0004 | 0.0000 | 42.1000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.2072 | 0.2073 | -0.0001 | [-0.0021, +0.0016] | [-0.0040, +0.0041] | 0.0019 | -0.0007 | 56.3000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Delta, median agg.') | 20 | 0.1501 | 0.1492 | 0.0009 | [-0.0011, +0.0026] | [-0.0031, +0.0050] | 0.0018 | 0.0002 | 48.3000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Generic potential') | 20 | 0.2003 | 0.1998 | 0.0005 | [-0.0007, +0.0015] | [-0.0015, +0.0022] | 0.0006 | 0.0005 | 37.3000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Joint, calibrated') | 20 | 0.0802 | 0.0802 | 0.0001 | [-0.0002, +0.0003] | [-0.0010, +0.0008] | 0.0003 | -0.0001 | 56.9000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0786 | 0.0786 | -0.0000 | [-0.0002, +0.0002] | [-0.0011, +0.0008] | 0.0006 | -0.0001 | 56.1000 | n.s. |  |  |
| ('BD-MPI', 'resolved', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0802 | 0.0799 | 0.0003 | [+0.0001, +0.0007] | [-0.0008, +0.0015] | 0.0005 | 0.0000 | 46.2000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Absolute response') | 20 | 0.1027 | 0.1027 | -0.0001 | [-0.0001, +0.0000] | [-0.0018, +0.0001] | 0.0006 | -0.0000 | 54.5000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Delta, uniform agg.') | 20 | 0.1772 | 0.1786 | -0.0014 | [-0.0025, -0.0005] | [-0.0041, +0.0015] | 0.0013 | -0.0023 | 73.9000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Delta, median agg.') | 20 | 0.1339 | 0.1344 | -0.0005 | [-0.0017, +0.0006] | [-0.0032, +0.0023] | 0.0013 | -0.0011 | 64.3000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Generic potential') | 20 | 0.1789 | 0.1796 | -0.0007 | [-0.0014, -0.0000] | [-0.0019, +0.0006] | 0.0006 | -0.0007 | 70.0000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Joint, calibrated') | 20 | 0.0953 | 0.0953 | 0.0000 | [-0.0002, +0.0002] | [-0.0007, +0.0005] | 0.0003 | -0.0000 | 52.1000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0927 | 0.0928 | -0.0001 | [-0.0003, +0.0001] | [-0.0009, +0.0005] | 0.0004 | -0.0001 | 58.7000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0965 | 0.0963 | 0.0002 | [+0.0000, +0.0004] | [-0.0007, +0.0010] | 0.0004 | 0.0000 | 46.5000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Absolute response') | 20 | 0.0727 | 0.0727 | -0.0001 | [-0.0002, +0.0001] | [-0.0013, +0.0012] | 0.0003 | 0.0000 | 42.9000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.2058 | 0.2062 | -0.0004 | [-0.0017, +0.0011] | [-0.0048, +0.0045] | 0.0022 | -0.0013 | 59.2000 | sign-unstable |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Delta, median agg.') | 20 | 0.1378 | 0.1368 | 0.0010 | [-0.0005, +0.0025] | [-0.0035, +0.0061] | 0.0022 | 0.0001 | 48.6000 | sign-unstable |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Generic potential') | 20 | 0.1992 | 0.1989 | 0.0003 | [-0.0002, +0.0011] | [-0.0016, +0.0023] | 0.0010 | 0.0004 | 43.1000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Joint, calibrated') | 20 | 0.0717 | 0.0716 | 0.0000 | [-0.0002, +0.0003] | [-0.0009, +0.0006] | 0.0002 | -0.0001 | 60.0000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0702 | 0.0702 | 0.0000 | [-0.0002, +0.0003] | [-0.0008, +0.0006] | 0.0004 | -0.0001 | 58.8000 | n.s. |  |  |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0716 | 0.0715 | 0.0001 | [-0.0002, +0.0006] | [-0.0009, +0.0012] | 0.0004 | -0.0001 | 53.6000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Absolute response') | 20 | 0.0167 | 0.0167 | -0.0000 | [-0.0000, +0.0000] | [-0.0007, +0.0003] | 0.0001 | 0.0000 | 44.7000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Delta, uniform agg.') | 20 | 0.0293 | 0.0298 | -0.0005 | [-0.0008, -0.0002] | [-0.0016, +0.0009] | 0.0006 | -0.0004 | 62.7000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Delta, median agg.') | 20 | 0.0240 | 0.0242 | -0.0001 | [-0.0006, +0.0003] | [-0.0011, +0.0013] | 0.0006 | 0.0000 | 49.9000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Generic potential') | 20 | 0.0310 | 0.0314 | -0.0003 | [-0.0005, -0.0002] | [-0.0009, +0.0001] | 0.0002 | -0.0004 | 75.0000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Joint, calibrated') | 20 | 0.0166 | 0.0166 | -0.0000 | [-0.0001, +0.0000] | [-0.0005, +0.0002] | 0.0001 | -0.0000 | 53.6000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0167 | 0.0167 | -0.0000 | [-0.0000, +0.0000] | [-0.0004, +0.0002] | 0.0000 | 0.0000 | 40.1000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0167 | 0.0167 | 0.0000 | [+0.0000, +0.0000] | [-0.0002, +0.0004] | 0.0000 | 0.0000 | 28.2000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Absolute response') | 20 | 0.0128 | 0.0128 | -0.0000 | [-0.0001, +0.0000] | [-0.0017, +0.0005] | 0.0004 | 0.0000 | 49.0000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.0335 | 0.0338 | -0.0003 | [-0.0008, +0.0001] | [-0.0021, +0.0013] | 0.0008 | -0.0004 | 58.2000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Delta, median agg.') | 20 | 0.0252 | 0.0251 | 0.0002 | [-0.0005, +0.0007] | [-0.0014, +0.0021] | 0.0008 | 0.0002 | 45.9000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Generic potential') | 20 | 0.0350 | 0.0353 | -0.0003 | [-0.0007, +0.0000] | [-0.0012, +0.0004] | 0.0004 | -0.0003 | 63.4000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Joint, calibrated') | 20 | 0.0127 | 0.0127 | -0.0000 | [-0.0000, +0.0000] | [-0.0004, +0.0004] | 0.0000 | 0.0000 | 43.0000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0128 | 0.0128 | 0.0000 | [+0.0000, +0.0000] | [-0.0003, +0.0003] | 0.0000 | 0.0000 | 37.8000 | n.s. |  |  |
| ('BD-TT', 'resolved', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0128 | 0.0128 | 0.0000 | [+0.0000, +0.0000] | [-0.0003, +0.0005] | 0.0000 | 0.0000 | 28.5000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Absolute response') | 20 | 0.0159 | 0.0159 | -0.0000 | [-0.0000, +0.0000] | [-0.0006, +0.0004] | 0.0000 | 0.0000 | 46.8000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Delta, uniform agg.') | 20 | 0.0292 | 0.0296 | -0.0004 | [-0.0007, -0.0001] | [-0.0016, +0.0010] | 0.0007 | -0.0005 | 61.6000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Delta, median agg.') | 20 | 0.0236 | 0.0236 | -0.0001 | [-0.0005, +0.0004] | [-0.0012, +0.0015] | 0.0007 | 0.0000 | 49.5000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Generic potential') | 20 | 0.0312 | 0.0315 | -0.0003 | [-0.0005, -0.0002] | [-0.0009, +0.0002] | 0.0003 | -0.0003 | 72.4000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Joint, calibrated') | 20 | 0.0157 | 0.0157 | -0.0000 | [-0.0001, +0.0000] | [-0.0005, +0.0002] | 0.0002 | -0.0000 | 55.1000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0159 | 0.0159 | 0.0000 | [+0.0000, +0.0000] | [-0.0004, +0.0002] | 0.0000 | 0.0000 | 41.3000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0159 | 0.0159 | 0.0000 | [+0.0000, +0.0000] | [-0.0001, +0.0004] | 0.0000 | 0.0000 | 29.7000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Absolute response') | 20 | 0.0118 | 0.0118 | 0.0000 | [-0.0000, +0.0000] | [-0.0015, +0.0006] | 0.0003 | 0.0000 | 48.3000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.0337 | 0.0341 | -0.0004 | [-0.0008, +0.0001] | [-0.0022, +0.0015] | 0.0009 | -0.0005 | 60.0000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Delta, median agg.') | 20 | 0.0255 | 0.0254 | 0.0000 | [-0.0006, +0.0006] | [-0.0017, +0.0022] | 0.0010 | 0.0000 | 49.0000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Generic potential') | 20 | 0.0353 | 0.0356 | -0.0002 | [-0.0005, -0.0000] | [-0.0012, +0.0005] | 0.0004 | -0.0002 | 61.8000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Joint, calibrated') | 20 | 0.0117 | 0.0117 | -0.0000 | [-0.0000, +0.0000] | [-0.0005, +0.0004] | 0.0001 | 0.0000 | 46.7000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0118 | 0.0118 | 0.0000 | [+0.0000, +0.0000] | [-0.0004, +0.0003] | 0.0000 | 0.0000 | 40.7000 | n.s. |  |  |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0118 | 0.0118 | 0.0000 | [+0.0000, +0.0000] | [-0.0003, +0.0005] | 0.0000 | 0.0000 | 31.8000 | n.s. |  |  |

#### architecture/init D1 − blind
|  | cells | D1 log MAE | Δ log MAE | CI pairs | CI pairs×seeds | seed SE (jackknife) | 5-seed median Δ | % 5-seed draws Δ < 0 | verdict | jackknife vs CI disagree | note | blind log MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('BD-MPI', 'resolved', 'one_known', 'Absolute response') | 20 | 0.1166 | 0.0001 | [-0.0000, +0.0002] | [-0.0095, +0.0111] | 0.0008 | 0.0000 | 43.3000 | n.s. |  |  | 0.1165 |
| ('BD-MPI', 'resolved', 'one_known', 'Delta, uniform agg.') | 20 | 0.1701 | 0.0273 | [+0.0180, +0.0374] | [+0.0153, +0.0402] | 0.0033 | 0.0273 | 0.0000 | CI excludes 0 |  |  | 0.1428 |
| ('BD-MPI', 'resolved', 'one_known', 'Delta, median agg.') | 20 | 0.1368 | 0.0111 | [+0.0052, +0.0178] | [+0.0020, +0.0210] | 0.0032 | 0.0113 | 5.3000 | CI excludes 0 |  |  | 0.1257 |
| ('BD-MPI', 'resolved', 'one_known', 'Generic potential') | 20 | 0.1700 | 0.0249 | [+0.0156, +0.0351] | [+0.0140, +0.0368] | 0.0025 | 0.0239 | 0.0000 | CI excludes 0 |  |  | 0.1451 |
| ('BD-MPI', 'resolved', 'one_known', 'Joint, calibrated') | 20 | 0.1112 | 0.0020 | [-0.0087, +0.0122] | [-0.0188, +0.0151] | 0.0056 | -0.0029 | 66.4000 | sign-unstable |  |  | 0.1092 |
| ('BD-MPI', 'resolved', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.1090 | 0.0001 | [-0.0106, +0.0102] | [-0.0206, +0.0141] | 0.0056 | -0.0035 | 69.5000 | sign-unstable |  |  | 0.1089 |
| ('BD-MPI', 'resolved', 'one_known', 'Bidir. context, calibrated') | 20 | 0.1112 | 0.0035 | [-0.0083, +0.0151] | [-0.0194, +0.0159] | 0.0076 | -0.0026 | 65.5000 | sign-unstable |  |  | 0.1077 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Absolute response') | 20 | 0.0799 | -0.0001 | [-0.0006, +0.0004] | [-0.0100, +0.0091] | 0.0004 | 0.0000 | 46.1000 | n.s. |  |  | 0.0799 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.2073 | 0.0388 | [+0.0137, +0.0632] | [+0.0067, +0.0692] | 0.0086 | 0.0397 | 3.0000 | CI excludes 0 |  |  | 0.1685 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Delta, median agg.') | 20 | 0.1492 | 0.0027 | [-0.0117, +0.0135] | [-0.0223, +0.0226] | 0.0086 | 0.0040 | 44.4000 | sign-unstable |  |  | 0.1464 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Generic potential') | 20 | 0.1998 | 0.0357 | [+0.0114, +0.0583] | [+0.0073, +0.0620] | 0.0066 | 0.0337 | 0.1000 | CI excludes 0 |  |  | 0.1641 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Joint, calibrated') | 20 | 0.0802 | 0.0008 | [-0.0178, +0.0170] | [-0.0291, +0.0171] | 0.0060 | -0.0033 | 71.8000 | sign-unstable |  |  | 0.0794 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0786 | -0.0028 | [-0.0208, +0.0125] | [-0.0314, +0.0148] | 0.0075 | -0.0041 | 73.6000 | sign-unstable |  |  | 0.0815 |
| ('BD-MPI', 'resolved', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0799 | 0.0007 | [-0.0184, +0.0172] | [-0.0343, +0.0198] | 0.0111 | -0.0039 | 67.7000 | sign-unstable |  |  | 0.0792 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Absolute response') | 20 | 0.1027 | 0.0000 | [-0.0001, +0.0001] | [-0.0135, +0.0137] | 0.0011 | 0.0000 | 47.3000 | n.s. |  |  | 0.1027 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Delta, uniform agg.') | 20 | 0.1786 | 0.0361 | [+0.0252, +0.0471] | [+0.0222, +0.0508] | 0.0042 | 0.0359 | 0.0000 | CI excludes 0 |  |  | 0.1426 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Delta, median agg.') | 20 | 0.1344 | 0.0131 | [+0.0061, +0.0213] | [+0.0014, +0.0251] | 0.0045 | 0.0133 | 7.2000 | CI excludes 0 |  |  | 0.1214 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Generic potential') | 20 | 0.1796 | 0.0344 | [+0.0238, +0.0454] | [+0.0220, +0.0478] | 0.0032 | 0.0330 | 0.0000 | CI excludes 0 |  |  | 0.1451 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Joint, calibrated') | 20 | 0.0953 | 0.0004 | [-0.0112, +0.0114] | [-0.0268, +0.0152] | 0.0071 | -0.0034 | 69.7000 | sign-unstable |  |  | 0.0949 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0928 | -0.0020 | [-0.0134, +0.0088] | [-0.0286, +0.0137] | 0.0074 | -0.0045 | 70.6000 | sign-unstable |  |  | 0.0948 |
| ('BD-MPI', 'rep-avg', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0963 | 0.0026 | [-0.0098, +0.0150] | [-0.0282, +0.0162] | 0.0093 | -0.0039 | 68.1000 | sign-unstable |  |  | 0.0937 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Absolute response') | 20 | 0.0727 | 0.0002 | [-0.0007, +0.0009] | [-0.0116, +0.0096] | 0.0010 | 0.0000 | 47.7000 | n.s. |  |  | 0.0725 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.2062 | 0.0493 | [+0.0207, +0.0705] | [+0.0138, +0.0783] | 0.0093 | 0.0495 | 1.5000 | CI excludes 0 |  |  | 0.1569 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Delta, median agg.') | 20 | 0.1368 | 0.0018 | [-0.0139, +0.0131] | [-0.0246, +0.0233] | 0.0094 | 0.0029 | 45.5000 | sign-unstable |  |  | 0.1350 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Generic potential') | 20 | 0.1989 | 0.0463 | [+0.0187, +0.0656] | [+0.0161, +0.0724] | 0.0073 | 0.0434 | 0.0000 | CI excludes 0 |  |  | 0.1526 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Joint, calibrated') | 20 | 0.0716 | 0.0013 | [-0.0175, +0.0183] | [-0.0281, +0.0193] | 0.0049 | -0.0029 | 69.1000 | sign-unstable |  |  | 0.0703 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0702 | -0.0018 | [-0.0202, +0.0144] | [-0.0303, +0.0163] | 0.0063 | -0.0039 | 71.6000 | sign-unstable |  |  | 0.0720 |
| ('BD-MPI', 'rep-avg', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0715 | 0.0012 | [-0.0183, +0.0188] | [-0.0332, +0.0213] | 0.0097 | -0.0038 | 67.0000 | sign-unstable |  |  | 0.0703 |
| ('BD-TT', 'resolved', 'one_known', 'Absolute response') | 20 | 0.0167 | 0.0000 | [-0.0000, +0.0002] | [-0.0151, +0.0040] | 0.0004 | -0.0001 | 53.3000 | n.s. |  |  | 0.0166 |
| ('BD-TT', 'resolved', 'one_known', 'Delta, uniform agg.') | 20 | 0.0298 | 0.0100 | [+0.0067, +0.0136] | [+0.0033, +0.0151] | 0.0023 | 0.0081 | 3.4000 | CI excludes 0 |  |  | 0.0198 |
| ('BD-TT', 'resolved', 'one_known', 'Delta, median agg.') | 20 | 0.0242 | 0.0054 | [+0.0030, +0.0079] | [-0.0009, +0.0098] | 0.0025 | 0.0038 | 17.4000 | n.s. |  |  | 0.0188 |
| ('BD-TT', 'resolved', 'one_known', 'Generic potential') | 20 | 0.0314 | 0.0099 | [+0.0070, +0.0129] | [+0.0059, +0.0145] | 0.0015 | 0.0094 | 0.0000 | CI excludes 0 |  |  | 0.0214 |
| ('BD-TT', 'resolved', 'one_known', 'Joint, calibrated') | 20 | 0.0166 | 0.0019 | [-0.0008, +0.0046] | [-0.0276, +0.0039] | 0.0041 | -0.0017 | 69.2000 | sign-unstable |  |  | 0.0147 |
| ('BD-TT', 'resolved', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0167 | 0.0025 | [+0.0004, +0.0046] | [-0.0295, +0.0043] | 0.0022 | -0.0009 | 65.8000 | sign-unstable |  |  | 0.0142 |
| ('BD-TT', 'resolved', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0167 | 0.0019 | [+0.0010, +0.0029] | [-0.0325, +0.0042] | 0.0032 | -0.0010 | 63.1000 | sign-unstable |  |  | 0.0148 |
| ('BD-TT', 'resolved', 'zero_shot', 'Absolute response') | 20 | 0.0128 | -0.0002 | [-0.0005, +0.0000] | [-0.0218, +0.0082] | 0.0011 | -0.0001 | 52.5000 | n.s. |  |  | 0.0130 |
| ('BD-TT', 'resolved', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.0338 | 0.0142 | [+0.0061, +0.0224] | [+0.0010, +0.0263] | 0.0043 | 0.0124 | 7.2000 | CI excludes 0 |  |  | 0.0197 |
| ('BD-TT', 'resolved', 'zero_shot', 'Delta, median agg.') | 20 | 0.0251 | 0.0059 | [+0.0004, +0.0120] | [-0.0051, +0.0167] | 0.0044 | 0.0044 | 31.4000 | sign-unstable |  |  | 0.0191 |
| ('BD-TT', 'resolved', 'zero_shot', 'Generic potential') | 20 | 0.0353 | 0.0135 | [+0.0064, +0.0204] | [+0.0054, +0.0232] | 0.0027 | 0.0128 | 0.3000 | CI excludes 0 |  |  | 0.0218 |
| ('BD-TT', 'resolved', 'zero_shot', 'Joint, calibrated') | 20 | 0.0127 | 0.0027 | [-0.0000, +0.0052] | [-0.0267, +0.0043] | 0.0037 | -0.0024 | 65.9000 | sign-unstable |  |  | 0.0099 |
| ('BD-TT', 'resolved', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0128 | 0.0035 | [+0.0009, +0.0057] | [-0.0285, +0.0054] | 0.0032 | -0.0014 | 61.0000 | sign-unstable |  |  | 0.0093 |
| ('BD-TT', 'resolved', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0128 | 0.0017 | [+0.0004, +0.0030] | [-0.0292, +0.0050] | 0.0038 | -0.0014 | 60.9000 | sign-unstable |  |  | 0.0111 |
| ('BD-TT', 'rep-avg', 'one_known', 'Absolute response') | 20 | 0.0159 | 0.0000 | [-0.0001, +0.0001] | [-0.0158, +0.0042] | 0.0003 | -0.0001 | 53.6000 | n.s. |  |  | 0.0158 |
| ('BD-TT', 'rep-avg', 'one_known', 'Delta, uniform agg.') | 20 | 0.0296 | 0.0103 | [+0.0069, +0.0139] | [+0.0035, +0.0159] | 0.0027 | 0.0084 | 3.6000 | CI excludes 0 |  |  | 0.0193 |
| ('BD-TT', 'rep-avg', 'one_known', 'Delta, median agg.') | 20 | 0.0236 | 0.0053 | [+0.0030, +0.0079] | [-0.0012, +0.0100] | 0.0028 | 0.0037 | 19.7000 | sign-unstable |  |  | 0.0183 |
| ('BD-TT', 'rep-avg', 'one_known', 'Generic potential') | 20 | 0.0315 | 0.0102 | [+0.0074, +0.0132] | [+0.0058, +0.0149] | 0.0018 | 0.0096 | 0.0000 | CI excludes 0 |  |  | 0.0213 |
| ('BD-TT', 'rep-avg', 'one_known', 'Joint, calibrated') | 20 | 0.0157 | 0.0018 | [-0.0009, +0.0045] | [-0.0287, +0.0040] | 0.0045 | -0.0019 | 69.4000 | sign-unstable |  |  | 0.0140 |
| ('BD-TT', 'rep-avg', 'one_known', 'Bidir. profile, calibrated') | 20 | 0.0159 | 0.0025 | [+0.0005, +0.0046] | [-0.0313, +0.0043] | 0.0024 | -0.0010 | 66.2000 | sign-unstable |  |  | 0.0134 |
| ('BD-TT', 'rep-avg', 'one_known', 'Bidir. context, calibrated') | 20 | 0.0159 | 0.0019 | [+0.0010, +0.0029] | [-0.0342, +0.0044] | 0.0033 | -0.0011 | 63.4000 | sign-unstable |  |  | 0.0140 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Absolute response') | 20 | 0.0118 | -0.0002 | [-0.0005, -0.0000] | [-0.0220, +0.0088] | 0.0012 | -0.0002 | 52.7000 | n.s. |  |  | 0.0120 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Delta, uniform agg.') | 20 | 0.0341 | 0.0152 | [+0.0073, +0.0234] | [+0.0015, +0.0273] | 0.0044 | 0.0131 | 6.6000 | CI excludes 0 |  |  | 0.0188 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Delta, median agg.') | 20 | 0.0254 | 0.0071 | [+0.0016, +0.0131] | [-0.0048, +0.0180] | 0.0046 | 0.0052 | 29.3000 | sign-unstable |  |  | 0.0183 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Generic potential') | 20 | 0.0356 | 0.0143 | [+0.0076, +0.0211] | [+0.0060, +0.0238] | 0.0029 | 0.0136 | 0.3000 | CI excludes 0 |  |  | 0.0212 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Joint, calibrated') | 20 | 0.0117 | 0.0030 | [-0.0000, +0.0057] | [-0.0283, +0.0049] | 0.0040 | -0.0024 | 65.7000 | sign-unstable |  |  | 0.0086 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Bidir. profile, calibrated') | 20 | 0.0118 | 0.0036 | [+0.0010, +0.0058] | [-0.0295, +0.0061] | 0.0037 | -0.0015 | 61.8000 | sign-unstable |  |  | 0.0082 |
| ('BD-TT', 'rep-avg', 'zero_shot', 'Bidir. context, calibrated') | 20 | 0.0118 | 0.0018 | [+0.0004, +0.0030] | [-0.0308, +0.0054] | 0.0044 | -0.0014 | 60.7000 | sign-unstable |  |  | 0.0101 |

