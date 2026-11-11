# Seed-batch figure numbers (written by figs_seed.py)

## forest_fulldata.pdf

Two-level CI (pairs×seeds), N=100. Axis clipped at 90% (arrowhead = CI continues). Hollow = sign-unstable.

| pipeline | role | Δ% | CI lo | CI hi | excl 0 | sign-unstable |
|---|---|---|---|---|---|---|
| 1W-MPI | absolute | 0 | -12.2 | 0 | False | False |
| 1W-MPI | delta_uniform | 42.78 | -5.4 | 87.7 | False | False |
| 1W-MPI | delta_median | 16.63 | -28.5 | 54.8 | False | False |
| 1W-MPI | delta_ood_kernel | 22.82 | -20.7 | 64.2 | False | False |
| 1W-MPI | generic | 42.59 | -5.2 | 87 | False | False |
| 1W-TT | absolute | 0 | -10.3 | 22.6 | False | False |
| 1W-TT | delta_uniform | 19.01 | -29.3 | 61.1 | False | False |
| 1W-TT | delta_median | 20.39 | -25 | 61.9 | False | False |
| 1W-TT | delta_ood_kernel | 6.685 | -43.3 | 50.8 | False | False |
| 1W-TT | generic | 30.87 | -14.5 | 71.8 | False | False |
| BD-MPI | absolute | 0 | -11.8 | 0 | False | False |
| BD-MPI | delta_uniform | 41.76 | -6.5 | 87.2 | False | False |
| BD-MPI | delta_median | 13.27 | -29 | 50.9 | False | False |
| BD-MPI | delta_ood_kernel | 20.55 | -23.3 | 62.6 | False | False |
| BD-MPI | generic | 42.59 | -6 | 87.4 | False | False |
| BD-MPI | joint_direct | -1.949 | -18.4 | 0 | False | True |
| BD-MPI | joint_cal | -10.55 | -31.6 | 13 | False | False |
| BD-MPI | bprof_direct | -7.996 | -20.8 | 0 | False | True |
| BD-MPI | bprof_cal | -10.23 | -31.8 | 15.1 | False | False |
| BD-MPI | bctx_direct | -7.872 | -21.3 | 0 | False | True |
| BD-MPI | bctx_cal | -10.44 | -30.9 | 13.4 | False | False |
| BD-MPI | bwall_cal | -10.25 | -31.8 | 14.8 | False | False |
| BD-MPI | bfwd_cal | -13.14 | -24.2 | 1.8 | False | False |
| BD-TT | absolute | 0 | -8.5 | 8.8 | False | False |
| BD-TT | delta_uniform | 27.04 | -19.6 | 68.8 | False | False |
| BD-TT | delta_median | 13.96 | -27.5 | 53.2 | False | False |
| BD-TT | delta_ood_kernel | 8.893 | -45.7 | 56.6 | False | False |
| BD-TT | generic | 30.87 | -16.2 | 71.4 | False | False |
| BD-TT | joint_direct | 14.82 | -5 | 203.9 | False | True |
| BD-TT | joint_cal | -33.9 | -48.6 | 70.6 | False | False |
| BD-TT | bprof_direct | 1.95 | -1.8 | 201.6 | False | True |
| BD-TT | bprof_cal | -34.52 | -46.1 | 58.4 | False | False |
| BD-TT | bctx_direct | 0 | -0.8 | 181.8 | False | False |
| BD-TT | bctx_cal | -12.44 | -37.4 | 16 | False | True |
| BD-TT | bwall_cal | -35.45 | -47.6 | 57.7 | False | False |
| BD-TT | bfwd_cal | -2.23 | -10.4 | 236.8 | False | True |

## bias_vs_floor.pdf

N=100 ensemble = max(0, mean latent over 100 seeds); floor-case log MAE = mean |pred - y| over the 18 outcomes with y = 0; bias = mean(pred - y) over 90 outcomes (matches seed_methods.csv to <1e-8).

| pipeline | role | bias | floor-case log MAE |
|---|---|---|---|
| 1W-MPI | absolute | -0.103 | 0 |
| 1W-MPI | delta_uniform | 0.126 | 0.2577 |
| 1W-MPI | delta_median | 0.09013 | 0.227 |
| 1W-MPI | delta_ood_kernel | 0.1162 | 0.2047 |
| 1W-MPI | generic | 0.1217 | 0.2574 |
| BD-MPI | absolute | -0.103 | 0 |
| BD-MPI | delta_uniform | 0.1235 | 0.2572 |
| BD-MPI | delta_median | 0.08311 | 0.2227 |
| BD-MPI | delta_ood_kernel | 0.1144 | 0.2037 |
| BD-MPI | generic | 0.1217 | 0.2574 |
| BD-MPI | joint_direct | -0.09879 | 0.005334 |
| BD-MPI | joint_cal | -0.01583 | 0.08983 |
| BD-MPI | bprof_direct | -0.08433 | 0.01979 |
| BD-MPI | bprof_cal | -0.008571 | 0.09602 |
| BD-MPI | bctx_direct | -0.08616 | 0.01762 |
| BD-MPI | bctx_cal | -0.01425 | 0.09024 |
| baseline | constant_1 | -0.103 | 0 |

## pervictim_heatmap.pdf

Victim log MAE / baseline, N=100. Colour log-scaled, saturating at 0.5 and 2 (true value annotated; >=10 rounded).

| victim | 1W-MPI delta_median | BD-MPI delta_median | BD-MPI joint_cal | 1W-TT delta_median | BD-TT delta_median | BD-TT joint_cal |
|---|---|---|---|---|---|---|
| Beatnik | 0.2876 | 0.2752 | 0.4365 | 0.3977 | 0.4353 | 0.3368 |
| TriC | 0.5844 | 0.4358 | 0.403 | 0.3019 | 0.2857 | 0.3376 |
| miniVite | 0.6114 | 0.6063 | 0.6647 | 0.9203 | 1.159 | 1.027 |
| Laghos | 0.883 | 0.9026 | 0.8774 | 0.8593 | 1.023 | 0.8943 |
| Quicksilver | 4.429 | 3.563 | 0.8475 | 1.943 | 1.093 | 0.9087 |
| AMG2023 | 1.251 | 1.277 | 1.146 | 4.74 | 3.778 | 1.635 |
| MiniFE | 2.285 | 2.278 | 1.568 | 9.091 | 9.076 | 0.901 |
| Kripke | 1.787 | 1.586 | 1.435 | 15.88 | 13.47 | 4.19 |
| FIESTA | 2.732 | 2.621 | 1 | 257 | 224.9 | 1 |
| LAMMPS | 3.167 | 3.027 | 1.147 | 247.2 | 144.7 | 2.897 |

## cv_vs_appapp.pdf

CV values from artifact_data_20261017.json (as before); baseline log MAE 0.1030.

| pipeline | role | CV log MAE | App-App log MAE (N=100) |
|---|---|---|---|
| 1W-MPI | delta_uniform | 0.158 | 0.1471 |
| 1W-MPI | generic | 0.1618 | 0.1469 |
| 1W-MPI | absolute | 0.1562 | 0.103 |
| BD-MPI | delta_uniform | 0.169 | 0.146 |
| BD-MPI | generic | 0.1618 | 0.1469 |
| BD-MPI | absolute | 0.1562 | 0.103 |
| BD-MPI | joint_cal | 0.1481 | 0.09214 |
| BD-MPI | bprof_cal | 0.1479 | 0.09247 |
| BD-MPI | bctx_cal | 0.1481 | 0.09225 |

## seed_kcurve.pdf

Recomputed from seed_predictions.csv (rng 925, 1000 k-subsets; k=1 every seed; k=100 full set). Symlog y, linthresh 25.

| pipeline | method | k | mean Δ% | 5% Δ% | 95% Δ% |
|---|---|---|---|---|---|
| BD-MPI | Joint, calibrated | 1 | 33.1 | -10.73 | 190.4 |
| BD-MPI | Joint, calibrated | 5 | 3.709 | -12.4 | 49.88 |
| BD-MPI | Joint, calibrated | 10 | -1.874 | -12.43 | 29.22 |
| BD-MPI | Joint, calibrated | 25 | -7.26 | -12.1 | 4.573 |
| BD-MPI | Joint, calibrated | 50 | -9.447 | -11.72 | -5.487 |
| BD-MPI | Joint, calibrated | 100 | -10.55 | -10.55 | -10.55 |
| BD-MPI | Absolute response | 1 | 39.56 | -9.126 | 247.9 |
| BD-MPI | Absolute response | 5 | 2.393 | -12.18 | 37.06 |
| BD-MPI | Absolute response | 10 | -2.509 | -13.16 | 0.2248 |
| BD-MPI | Absolute response | 25 | -2.971 | -12.7 | 0 |
| BD-MPI | Absolute response | 50 | -1.736 | -10.23 | 0 |
| BD-MPI | Absolute response | 100 | 0 | 0 | 0 |
| BD-TT | Joint, calibrated | 1 | 234 | -4.41 | 1401 |
| BD-TT | Joint, calibrated | 5 | 98.61 | -21.96 | 496.5 |
| BD-TT | Joint, calibrated | 10 | 52.98 | -26.37 | 295.3 |
| BD-TT | Joint, calibrated | 25 | 7.247 | -33.42 | 127.5 |
| BD-TT | Joint, calibrated | 50 | -13.79 | -36.08 | 41.64 |
| BD-TT | Joint, calibrated | 100 | -33.9 | -33.9 | -33.9 |

## seed_kcurve.pdf verification

| pipeline | method | k | mine | csv |
|---|---|---|---|---|
| BD-MPI | Joint, calibrated | 1 | 33.1 | 33.1 |
| BD-MPI | Joint, calibrated | 5 | 3.709 | 3.709 |
| BD-MPI | Joint, calibrated | 100 | -10.55 | -10.55 |
| BD-MPI | Absolute response | 1 | 39.56 | 39.56 |
| BD-MPI | Absolute response | 5 | 2.393 | 2.131 |
| BD-MPI | Absolute response | 100 | 0 | 0 |
| BD-TT | Joint, calibrated | 1 | 234 | 234 |
| BD-TT | Joint, calibrated | 5 | 98.61 | 98.61 |
| BD-TT | Joint, calibrated | 100 | -33.9 | -33.9 |

## socket_info_effect.pdf

Level = resolved; bars = two-level CI (pairs×seeds). Hollow = verdict sign-unstable.

| pipeline | role | comparison | Δ log MAE | CI lo | CI hi | verdict |
|---|---|---|---|---|---|---|
| 1W-MPI | absolute | E1 − D1 (socket information) | -0.0002822 | -0.0013 | 0.0006 | n.s. |
| 1W-MPI | absolute | D1 − blind (init./architecture) | -0.007013 | -0.0154 | 0.0041 | sign-unstable |
| 1W-MPI | delta_uniform | E1 − D1 (socket information) | -0.002733 | -0.0048 | -0.0006 | CI excludes 0 |
| 1W-MPI | delta_uniform | D1 − blind (init./architecture) | 0.02818 | 0.0189 | 0.0394 | CI excludes 0 |
| 1W-MPI | delta_median | E1 − D1 (socket information) | -0.003673 | -0.0067 | -0.0008 | CI excludes 0 |
| 1W-MPI | delta_median | D1 − blind (init./architecture) | 0.01581 | 0.0083 | 0.0239 | CI excludes 0 |
| 1W-MPI | generic | E1 − D1 (socket information) | -0.001114 | -0.002 | -0.0002 | CI excludes 0 |
| 1W-MPI | generic | D1 − blind (init./architecture) | 0.02454 | 0.0149 | 0.0352 | CI excludes 0 |
| BD-MPI | absolute | E1 − D1 (socket information) | -0.0002822 | -0.0011 | 0.0006 | n.s. |
| BD-MPI | absolute | D1 − blind (init./architecture) | -0.007013 | -0.0144 | 0.0032 | sign-unstable |
| BD-MPI | delta_uniform | E1 − D1 (socket information) | -0.002 | -0.0033 | -0.0007 | CI excludes 0 |
| BD-MPI | delta_uniform | D1 − blind (init./architecture) | 0.02493 | 0.0152 | 0.0362 | CI excludes 0 |
| BD-MPI | delta_median | E1 − D1 (socket information) | -0.002484 | -0.0044 | -0.0006 | CI excludes 0 |
| BD-MPI | delta_median | D1 − blind (init./architecture) | 0.01346 | 0.0054 | 0.022 | CI excludes 0 |
| BD-MPI | generic | E1 − D1 (socket information) | -0.001114 | -0.0019 | -0.0002 | CI excludes 0 |
| BD-MPI | generic | D1 − blind (init./architecture) | 0.02454 | 0.015 | 0.0348 | CI excludes 0 |
| BD-MPI | joint_cal | E1 − D1 (socket information) | 0.0001398 | -0.0004 | 0.0008 | n.s. |
| BD-MPI | joint_cal | D1 − blind (init./architecture) | -0.002463 | -0.0102 | 0.0043 | n.s. |
| BD-MPI | bprof_cal | E1 − D1 (socket information) | 2.75e-05 | -0.0005 | 0.0006 | n.s. |
| BD-MPI | bprof_cal | D1 − blind (init./architecture) | -0.002391 | -0.011 | 0.004 | sign-unstable |
| BD-MPI | bctx_cal | E1 − D1 (socket information) | 0.000236 | -0.0003 | 0.0009 | n.s. |
| BD-MPI | bctx_cal | D1 − blind (init./architecture) | -0.001671 | -0.0113 | 0.0066 | n.s. |

## learning_curves.pdf

ratio = 1 - skill/100; N=30 ensemble (solid) vs seeds 0-4 (dotted, equals the old artifact curves). Hollow = <10 splits.

| pipeline | tier | method | size | splits | ratio N=30 | ratio seeds 0-4 |
|---|---|---|---|---|---|---|
| 1W-MPI | one_known | Absolute response | 4 | 6 | 0.9983 | 0.9152 |
| 1W-MPI | one_known | Absolute response | 5 | 6 | 0.9988 | 0.9239 |
| 1W-MPI | one_known | Absolute response | 6 | 7 | 0.9999 | 0.923 |
| 1W-MPI | one_known | Absolute response | 7 | 8 | 1 | 0.9298 |
| 1W-MPI | one_known | Absolute response | 8 | 9 | 1 | 0.9449 |
| 1W-MPI | one_known | Delta, uniform agg. | 4 | 6 | 1.385 | 1.425 |
| 1W-MPI | one_known | Delta, uniform agg. | 5 | 6 | 1.235 | 1.261 |
| 1W-MPI | one_known | Delta, uniform agg. | 6 | 7 | 1.375 | 1.427 |
| 1W-MPI | one_known | Delta, uniform agg. | 7 | 8 | 1.546 | 1.611 |
| 1W-MPI | one_known | Delta, uniform agg. | 8 | 9 | 1.671 | 1.726 |
| 1W-MPI | zero_shot | Absolute response | 4 | 6 | 1.003 | 1.029 |
| 1W-MPI | zero_shot | Absolute response | 5 | 6 | 1.005 | 1.071 |
| 1W-MPI | zero_shot | Absolute response | 6 | 7 | 1.001 | 1.122 |
| 1W-MPI | zero_shot | Absolute response | 7 | 8 | 1 | 1.142 |
| 1W-MPI | zero_shot | Absolute response | 8 | 9 | 1 | 1.077 |
| 1W-MPI | zero_shot | Delta, uniform agg. | 4 | 6 | 1.729 | 1.779 |
| 1W-MPI | zero_shot | Delta, uniform agg. | 5 | 6 | 1.999 | 2.095 |
| 1W-MPI | zero_shot | Delta, uniform agg. | 6 | 7 | 2.661 | 2.853 |
| 1W-MPI | zero_shot | Delta, uniform agg. | 7 | 8 | 3.228 | 3.466 |
| 1W-MPI | zero_shot | Delta, uniform agg. | 8 | 9 | 1.92 | 2.035 |
| BD-MPI | one_known | Absolute response | 4 | 8 | 0.9987 | 0.9335 |
| BD-MPI | one_known | Absolute response | 5 | 9 | 0.9992 | 0.935 |
| BD-MPI | one_known | Absolute response | 6 | 10 | 0.9999 | 0.9337 |
| BD-MPI | one_known | Absolute response | 7 | 10 | 1 | 0.9379 |
| BD-MPI | one_known | Absolute response | 8 | 10 | 1 | 0.9442 |
| BD-MPI | one_known | Delta, uniform agg. | 4 | 8 | 1.399 | 1.493 |
| BD-MPI | one_known | Delta, uniform agg. | 5 | 9 | 1.34 | 1.426 |
| BD-MPI | one_known | Delta, uniform agg. | 6 | 10 | 1.499 | 1.584 |
| BD-MPI | one_known | Delta, uniform agg. | 7 | 10 | 1.531 | 1.592 |
| BD-MPI | one_known | Delta, uniform agg. | 8 | 10 | 1.536 | 1.582 |
| BD-MPI | one_known | Joint, calibrated | 4 | 8 | 0.9341 | 0.9446 |
| BD-MPI | one_known | Joint, calibrated | 5 | 9 | 0.9176 | 0.9238 |
| BD-MPI | one_known | Joint, calibrated | 6 | 10 | 0.9229 | 0.9338 |
| BD-MPI | one_known | Joint, calibrated | 7 | 10 | 0.9437 | 0.954 |
| BD-MPI | one_known | Joint, calibrated | 8 | 10 | 0.9594 | 0.9656 |
| BD-MPI | one_known | Bidir. profile | 4 | 8 | 0.9245 | 0.9574 |
| BD-MPI | one_known | Bidir. profile | 5 | 9 | 0.9145 | 0.9418 |
| BD-MPI | one_known | Bidir. profile | 6 | 10 | 0.9083 | 0.9479 |
| BD-MPI | one_known | Bidir. profile | 7 | 10 | 0.9132 | 0.9491 |
| BD-MPI | one_known | Bidir. profile | 8 | 10 | 0.9189 | 0.9521 |
| BD-MPI | zero_shot | Absolute response | 4 | 8 | 1.002 | 1.02 |
| BD-MPI | zero_shot | Absolute response | 5 | 9 | 1.002 | 1.036 |
| BD-MPI | zero_shot | Absolute response | 6 | 10 | 1 | 1.048 |
| BD-MPI | zero_shot | Absolute response | 7 | 10 | 1 | 1.057 |
| BD-MPI | zero_shot | Absolute response | 8 | 10 | 1 | 1.034 |
| BD-MPI | zero_shot | Delta, uniform agg. | 4 | 8 | 1.712 | 1.957 |
| BD-MPI | zero_shot | Delta, uniform agg. | 5 | 9 | 1.819 | 2.084 |
| BD-MPI | zero_shot | Delta, uniform agg. | 6 | 10 | 1.783 | 1.979 |
| BD-MPI | zero_shot | Delta, uniform agg. | 7 | 10 | 1.799 | 1.984 |
| BD-MPI | zero_shot | Delta, uniform agg. | 8 | 10 | 1.34 | 1.559 |
| BD-MPI | zero_shot | Joint, calibrated | 4 | 8 | 0.9617 | 0.9815 |
| BD-MPI | zero_shot | Joint, calibrated | 5 | 9 | 0.9721 | 0.9901 |
| BD-MPI | zero_shot | Joint, calibrated | 6 | 10 | 0.9598 | 0.9832 |
| BD-MPI | zero_shot | Joint, calibrated | 7 | 10 | 0.9589 | 0.9624 |
| BD-MPI | zero_shot | Joint, calibrated | 8 | 10 | 0.9138 | 0.9125 |
| BD-MPI | zero_shot | Bidir. profile | 4 | 8 | 0.936 | 0.944 |
| BD-MPI | zero_shot | Bidir. profile | 5 | 9 | 0.9529 | 0.9477 |
| BD-MPI | zero_shot | Bidir. profile | 6 | 10 | 0.9476 | 0.9107 |
| BD-MPI | zero_shot | Bidir. profile | 7 | 10 | 0.944 | 0.8938 |
| BD-MPI | zero_shot | Bidir. profile | 8 | 10 | 0.9422 | 0.8875 |
