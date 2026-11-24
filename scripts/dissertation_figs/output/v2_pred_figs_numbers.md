# v2 prediction figures (v2_full_20261123)

Source: output/v2_pred_v2_full_20261123_*.csv (v2_score.py). Methods: xgb is the max_depth 1-8 rerun; xgb_d124 not plotted.

## r_skill.pdf (6.0 x 5.4 in)
Socket-aware; skill = 1 - MAE/MAE_base, mean ± SD over 10 seeds; † = pair-cluster CI of method - baseline below 0 in >= 9/10 seeds. x clipped at -1.0 (n_clipped seeds drawn as '<').

| target | tier | method | form | skill_mean | skill_sd | seed_min | seed_max | n_clipped | seeds_better | ci_below0 | dagger |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpi | full | ridge | 1w | 0.165 | 0.126 | -0.080 | 0.307 | 0 | 8 | 2 | False |
| mpi | full | ridge | bd | 0.126 | 0.186 | -0.351 | 0.269 | 0 | 9 | 1 | False |
| mpi | full | lasso | 1w | 0.144 | 0.171 | -0.280 | 0.291 | 0 | 9 | 2 | False |
| mpi | full | lasso | bd | 0.157 | 0.192 | -0.348 | 0.302 | 0 | 9 | 2 | False |
| mpi | full | svr | 1w | 0.304 | 0.117 | 0.082 | 0.416 | 0 | 10 | 3 | False |
| mpi | full | svr | bd | 0.357 | 0.129 | 0.074 | 0.553 | 0 | 10 | 4 | False |
| mpi | full | rf | 1w | 0.412 | 0.147 | 0.085 | 0.545 | 0 | 10 | 7 | False |
| mpi | full | rf | bd | 0.404 | 0.152 | 0.063 | 0.524 | 0 | 10 | 7 | False |
| mpi | full | xgb | 1w | 0.351 | 0.203 | -0.038 | 0.520 | 0 | 8 | 6 | False |
| mpi | full | xgb | bd | 0.330 | 0.219 | -0.103 | 0.558 | 0 | 8 | 4 | False |
| mpi | full | mlp | 1w | 0.237 | 0.137 | -0.014 | 0.393 | 0 | 9 | 1 | False |
| mpi | full | mlp | bd | 0.245 | 0.232 | -0.230 | 0.526 | 0 | 8 | 2 | False |
| mpi | full | absolute | none | -0.115 | 0.403 | -1.053 | 0.280 | 1 | 6 | 0 | False |
| mpi | full | delta | none | -0.388 | 0.513 | -1.281 | 0.377 | 1 | 2 | 0 | False |
| mpi | one_known | ridge | 1w | 0.035 | 0.000 | 0.035 | 0.035 | 0 | 10 | 0 | False |
| mpi | one_known | ridge | bd | 0.074 | 0.000 | 0.074 | 0.074 | 0 | 10 | 0 | False |
| mpi | one_known | lasso | 1w | -0.025 | 0.000 | -0.025 | -0.025 | 0 | 0 | 0 | False |
| mpi | one_known | lasso | bd | 0.070 | 0.000 | 0.070 | 0.070 | 0 | 10 | 0 | False |
| mpi | one_known | svr | 1w | 0.118 | 0.000 | 0.118 | 0.118 | 0 | 10 | 10 | True |
| mpi | one_known | svr | bd | 0.112 | 0.000 | 0.112 | 0.112 | 0 | 10 | 0 | False |
| mpi | one_known | rf | 1w | 0.164 | 0.002 | 0.160 | 0.167 | 0 | 10 | 10 | True |
| mpi | one_known | rf | bd | 0.203 | 0.001 | 0.202 | 0.205 | 0 | 10 | 10 | True |
| mpi | one_known | xgb | 1w | 0.107 | 0.003 | 0.102 | 0.112 | 0 | 10 | 0 | False |
| mpi | one_known | xgb | bd | 0.119 | 0.001 | 0.116 | 0.120 | 0 | 10 | 0 | False |
| mpi | one_known | mlp | 1w | 0.010 | 0.033 | -0.044 | 0.062 | 0 | 7 | 0 | False |
| mpi | one_known | mlp | bd | -0.006 | 0.042 | -0.109 | 0.042 | 0 | 5 | 0 | False |
| mpi | one_known | absolute | none | -0.439 | 0.227 | -0.885 | -0.215 | 0 | 0 | 0 | False |
| mpi | one_known | delta | none | -0.160 | 0.101 | -0.348 | -0.046 | 0 | 0 | 0 | False |
| mpi | zero_shot | ridge | 1w | -0.132 | 0.000 | -0.132 | -0.132 | 0 | 0 | 0 | False |
| mpi | zero_shot | ridge | bd | -0.051 | 0.000 | -0.051 | -0.051 | 0 | 0 | 0 | False |
| mpi | zero_shot | lasso | 1w | -0.225 | 0.000 | -0.225 | -0.225 | 0 | 0 | 0 | False |
| mpi | zero_shot | lasso | bd | -0.049 | 0.000 | -0.049 | -0.049 | 0 | 0 | 0 | False |
| mpi | zero_shot | svr | 1w | 0.023 | 0.000 | 0.023 | 0.023 | 0 | 10 | 0 | False |
| mpi | zero_shot | svr | bd | -0.037 | 0.000 | -0.037 | -0.037 | 0 | 0 | 0 | False |
| mpi | zero_shot | rf | 1w | 0.015 | 0.002 | 0.011 | 0.018 | 0 | 10 | 0 | False |
| mpi | zero_shot | rf | bd | 0.056 | 0.002 | 0.053 | 0.060 | 0 | 10 | 0 | False |
| mpi | zero_shot | xgb | 1w | -0.058 | 0.005 | -0.064 | -0.049 | 0 | 0 | 0 | False |
| mpi | zero_shot | xgb | bd | -0.035 | 0.004 | -0.041 | -0.028 | 0 | 0 | 0 | False |
| mpi | zero_shot | mlp | 1w | -0.153 | 0.056 | -0.254 | -0.070 | 0 | 0 | 0 | False |
| mpi | zero_shot | mlp | bd | -0.208 | 0.073 | -0.382 | -0.130 | 0 | 0 | 0 | False |
| mpi | zero_shot | absolute | none | -0.682 | 0.368 | -1.468 | -0.368 | 2 | 0 | 0 | False |
| mpi | zero_shot | delta | none | -0.206 | 0.112 | -0.436 | -0.093 | 0 | 0 | 0 | False |
| tt | full | ridge | 1w | 0.183 | 0.154 | -0.094 | 0.386 | 0 | 9 | 2 | False |
| tt | full | ridge | bd | 0.135 | 0.154 | -0.225 | 0.287 | 0 | 8 | 1 | False |
| tt | full | lasso | 1w | 0.177 | 0.133 | -0.075 | 0.332 | 0 | 8 | 1 | False |
| tt | full | lasso | bd | 0.170 | 0.137 | -0.116 | 0.365 | 0 | 9 | 2 | False |
| tt | full | svr | 1w | 0.339 | 0.121 | 0.067 | 0.465 | 0 | 10 | 5 | False |
| tt | full | svr | bd | 0.338 | 0.139 | 0.081 | 0.503 | 0 | 10 | 6 | False |
| tt | full | rf | 1w | 0.372 | 0.078 | 0.253 | 0.494 | 0 | 10 | 9 | True |
| tt | full | rf | bd | 0.367 | 0.091 | 0.205 | 0.481 | 0 | 10 | 8 | False |
| tt | full | xgb | 1w | 0.265 | 0.134 | 0.089 | 0.447 | 0 | 10 | 5 | False |
| tt | full | xgb | bd | 0.227 | 0.188 | -0.139 | 0.466 | 0 | 9 | 5 | False |
| tt | full | mlp | 1w | 0.221 | 0.190 | -0.152 | 0.422 | 0 | 9 | 1 | False |
| tt | full | mlp | bd | 0.205 | 0.185 | -0.212 | 0.424 | 0 | 9 | 1 | False |
| tt | full | absolute | none | 0.015 | 0.282 | -0.446 | 0.403 | 0 | 5 | 1 | False |
| tt | full | delta | none | 0.130 | 0.285 | -0.350 | 0.584 | 0 | 8 | 1 | False |
| tt | one_known | ridge | 1w | 0.165 | 0.000 | 0.165 | 0.165 | 0 | 10 | 0 | False |
| tt | one_known | ridge | bd | 0.092 | 0.000 | 0.092 | 0.092 | 0 | 10 | 0 | False |
| tt | one_known | lasso | 1w | -0.051 | 0.000 | -0.051 | -0.051 | 0 | 0 | 0 | False |
| tt | one_known | lasso | bd | 0.036 | 0.000 | 0.036 | 0.036 | 0 | 10 | 0 | False |
| tt | one_known | svr | 1w | 0.225 | 0.000 | 0.225 | 0.225 | 0 | 10 | 10 | True |
| tt | one_known | svr | bd | 0.162 | 0.000 | 0.162 | 0.162 | 0 | 10 | 10 | True |
| tt | one_known | rf | 1w | 0.211 | 0.002 | 0.208 | 0.214 | 0 | 10 | 10 | True |
| tt | one_known | rf | bd | 0.217 | 0.002 | 0.214 | 0.220 | 0 | 10 | 10 | True |
| tt | one_known | xgb | 1w | 0.170 | 0.004 | 0.165 | 0.176 | 0 | 10 | 10 | True |
| tt | one_known | xgb | bd | 0.154 | 0.010 | 0.138 | 0.174 | 0 | 10 | 4 | False |
| tt | one_known | mlp | 1w | 0.129 | 0.033 | 0.047 | 0.166 | 0 | 10 | 3 | False |
| tt | one_known | mlp | bd | 0.053 | 0.038 | -0.031 | 0.108 | 0 | 9 | 0 | False |
| tt | one_known | absolute | none | -0.180 | 0.100 | -0.311 | -0.009 | 0 | 0 | 0 | False |
| tt | one_known | delta | none | -0.111 | 0.087 | -0.313 | -0.029 | 0 | 0 | 0 | False |
| tt | zero_shot | ridge | 1w | 0.085 | 0.000 | 0.085 | 0.085 | 0 | 10 | 0 | False |
| tt | zero_shot | ridge | bd | 0.009 | 0.000 | 0.009 | 0.009 | 0 | 10 | 0 | False |
| tt | zero_shot | lasso | 1w | -0.191 | 0.000 | -0.191 | -0.191 | 0 | 0 | 0 | False |
| tt | zero_shot | lasso | bd | -0.085 | 0.000 | -0.085 | -0.085 | 0 | 0 | 0 | False |
| tt | zero_shot | svr | 1w | 0.071 | 0.000 | 0.071 | 0.071 | 0 | 10 | 0 | False |
| tt | zero_shot | svr | bd | 0.003 | 0.000 | 0.003 | 0.003 | 0 | 10 | 0 | False |
| tt | zero_shot | rf | 1w | 0.108 | 0.003 | 0.104 | 0.112 | 0 | 10 | 0 | False |
| tt | zero_shot | rf | bd | 0.109 | 0.002 | 0.106 | 0.112 | 0 | 10 | 0 | False |
| tt | zero_shot | xgb | 1w | -0.031 | 0.003 | -0.038 | -0.025 | 0 | 0 | 0 | False |
| tt | zero_shot | xgb | bd | -0.039 | 0.005 | -0.049 | -0.033 | 0 | 0 | 0 | False |
| tt | zero_shot | mlp | 1w | 0.023 | 0.046 | -0.068 | 0.093 | 0 | 7 | 0 | False |
| tt | zero_shot | mlp | bd | 0.019 | 0.041 | -0.037 | 0.076 | 0 | 6 | 0 | False |
| tt | zero_shot | absolute | none | -0.172 | 0.103 | -0.285 | 0.027 | 0 | 1 | 0 | False |
| tt | zero_shot | delta | none | -0.048 | 0.068 | -0.206 | 0.024 | 0 | 2 | 0 | False |

## r_paired_diffs.pdf (6.0 x 5.0 in)
Mean ± SD over 10 seeds of the paired log-MAE difference (×1e-3; negative = first term better). Aware - blind uses the 1W form (response models: their only form); BD - 1W uses socket-aware. Filled = pair-cluster CI excludes 0 (either side) in >= 9/10 seeds. x range per panel = 1.15 x the largest |mean ± SD| of the App-App learners; response-model means outside it are drawn as arrows at the edge with the value printed.

| contrast | target | tier | method | diff_mean_x1e3 | diff_sd_x1e3 | seeds_neg | ci_below0 | ci_above0 | filled |
|---|---|---|---|---|---|---|---|---|---|
| aware-blind | mpi | full | ridge | 0.157 | 4.862 | 7 | 2 | 0 | False |
| aware-blind | mpi | one_known | ridge | -0.305 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | mpi | zero_shot | ridge | -0.200 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | mpi | full | lasso | -1.140 | 1.143 | 9 | 1 | 0 | False |
| aware-blind | mpi | one_known | lasso | -0.003 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | mpi | zero_shot | lasso | 0.930 | 0.000 | 0 | 0 | 0 | False |
| aware-blind | mpi | full | svr | -6.413 | 11.326 | 8 | 2 | 0 | False |
| aware-blind | mpi | one_known | svr | -2.683 | 0.000 | 10 | 10 | 0 | True |
| aware-blind | mpi | zero_shot | svr | -1.758 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | mpi | full | rf | -9.049 | 6.209 | 10 | 4 | 0 | False |
| aware-blind | mpi | one_known | rf | -3.148 | 0.104 | 10 | 10 | 0 | True |
| aware-blind | mpi | zero_shot | rf | -0.711 | 0.294 | 10 | 0 | 0 | False |
| aware-blind | mpi | full | xgb | -9.736 | 9.497 | 9 | 1 | 0 | False |
| aware-blind | mpi | one_known | xgb | -1.582 | 0.252 | 10 | 1 | 0 | False |
| aware-blind | mpi | zero_shot | xgb | 0.738 | 0.679 | 2 | 0 | 3 | False |
| aware-blind | mpi | full | mlp | -5.954 | 9.656 | 8 | 1 | 0 | False |
| aware-blind | mpi | one_known | mlp | 0.296 | 1.471 | 4 | 2 | 1 | False |
| aware-blind | mpi | zero_shot | mlp | 1.357 | 2.921 | 3 | 0 | 2 | False |
| aware-blind | mpi | full | absolute | -18.073 | 42.768 | 8 | 3 | 1 | False |
| aware-blind | mpi | one_known | absolute | 10.293 | 14.775 | 1 | 1 | 7 | False |
| aware-blind | mpi | zero_shot | absolute | 13.143 | 25.960 | 2 | 0 | 3 | False |
| aware-blind | mpi | full | delta | 42.043 | 50.456 | 2 | 0 | 4 | False |
| aware-blind | mpi | one_known | delta | -11.546 | 17.368 | 7 | 6 | 2 | False |
| aware-blind | mpi | zero_shot | delta | -13.697 | 21.335 | 7 | 3 | 0 | False |
| aware-blind | tt | full | ridge | -0.032 | 0.072 | 8 | 1 | 0 | False |
| aware-blind | tt | one_known | ridge | -0.028 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | zero_shot | ridge | -0.040 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | full | lasso | 0.000 | 0.000 | 0 | 0 | 0 | False |
| aware-blind | tt | one_known | lasso | -0.005 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | zero_shot | lasso | -0.008 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | full | svr | -0.229 | 0.867 | 5 | 1 | 0 | False |
| aware-blind | tt | one_known | svr | -0.090 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | zero_shot | svr | -0.088 | 0.000 | 10 | 0 | 0 | False |
| aware-blind | tt | full | rf | -0.421 | 1.911 | 7 | 1 | 0 | False |
| aware-blind | tt | one_known | rf | -0.143 | 0.043 | 10 | 4 | 0 | False |
| aware-blind | tt | zero_shot | rf | -0.121 | 0.068 | 10 | 1 | 0 | False |
| aware-blind | tt | full | xgb | -0.319 | 1.978 | 9 | 2 | 0 | False |
| aware-blind | tt | one_known | xgb | -0.366 | 0.037 | 10 | 10 | 0 | True |
| aware-blind | tt | zero_shot | xgb | -0.158 | 0.053 | 10 | 0 | 0 | False |
| aware-blind | tt | full | mlp | -0.960 | 2.649 | 3 | 2 | 1 | False |
| aware-blind | tt | one_known | mlp | -0.042 | 0.111 | 7 | 1 | 0 | False |
| aware-blind | tt | zero_shot | mlp | -0.050 | 0.192 | 5 | 0 | 0 | False |
| aware-blind | tt | full | absolute | 0.040 | 0.901 | 3 | 1 | 3 | False |
| aware-blind | tt | one_known | absolute | -0.293 | 0.994 | 5 | 4 | 2 | False |
| aware-blind | tt | zero_shot | absolute | -1.239 | 2.947 | 6 | 3 | 1 | False |
| aware-blind | tt | full | delta | -2.759 | 8.164 | 7 | 1 | 0 | False |
| aware-blind | tt | one_known | delta | 1.050 | 1.507 | 2 | 1 | 5 | False |
| aware-blind | tt | zero_shot | delta | 2.176 | 2.531 | 2 | 0 | 6 | False |
| bd-1w | mpi | full | ridge | 4.363 | 9.359 | 5 | 0 | 1 | False |
| bd-1w | mpi | one_known | ridge | -5.474 | 0.000 | 10 | 0 | 0 | False |
| bd-1w | mpi | zero_shot | ridge | -11.736 | 0.000 | 10 | 0 | 0 | False |
| bd-1w | mpi | full | lasso | -2.383 | 6.440 | 7 | 1 | 0 | False |
| bd-1w | mpi | one_known | lasso | -13.261 | 0.000 | 10 | 10 | 0 | True |
| bd-1w | mpi | zero_shot | lasso | -25.464 | 0.000 | 10 | 10 | 0 | True |
| bd-1w | mpi | full | svr | -8.909 | 14.813 | 7 | 2 | 0 | False |
| bd-1w | mpi | one_known | svr | 0.853 | 0.000 | 0 | 0 | 0 | False |
| bd-1w | mpi | zero_shot | svr | 8.642 | 0.000 | 0 | 0 | 0 | False |
| bd-1w | mpi | full | rf | 1.236 | 3.148 | 4 | 0 | 1 | False |
| bd-1w | mpi | one_known | rf | -5.475 | 0.339 | 10 | 10 | 0 | True |
| bd-1w | mpi | zero_shot | rf | -5.962 | 0.458 | 10 | 0 | 0 | False |
| bd-1w | mpi | full | xgb | 2.840 | 6.376 | 3 | 0 | 1 | False |
| bd-1w | mpi | one_known | xgb | -1.627 | 0.335 | 10 | 6 | 0 | False |
| bd-1w | mpi | zero_shot | xgb | -3.347 | 1.205 | 10 | 6 | 0 | False |
| bd-1w | mpi | full | mlp | -3.824 | 21.442 | 6 | 1 | 1 | False |
| bd-1w | mpi | one_known | mlp | 2.120 | 6.500 | 5 | 0 | 1 | False |
| bd-1w | mpi | zero_shot | mlp | 7.949 | 11.608 | 2 | 0 | 2 | False |
| bd-1w | tt | full | ridge | 1.167 | 1.369 | 1 | 0 | 3 | False |
| bd-1w | tt | one_known | ridge | 1.442 | 0.000 | 0 | 0 | 10 | True |
| bd-1w | tt | zero_shot | ridge | 1.929 | 0.000 | 0 | 0 | 0 | False |
| bd-1w | tt | full | lasso | 0.058 | 1.319 | 5 | 1 | 0 | False |
| bd-1w | tt | one_known | lasso | -1.701 | 0.000 | 10 | 10 | 0 | True |
| bd-1w | tt | zero_shot | lasso | -2.710 | 0.000 | 10 | 10 | 0 | True |
| bd-1w | tt | full | svr | 0.099 | 2.498 | 6 | 0 | 0 | False |
| bd-1w | tt | one_known | svr | 1.222 | 0.000 | 0 | 0 | 10 | True |
| bd-1w | tt | zero_shot | svr | 1.714 | 0.000 | 0 | 0 | 10 | True |
| bd-1w | tt | full | rf | -0.095 | 1.478 | 6 | 1 | 1 | False |
| bd-1w | tt | one_known | rf | -0.124 | 0.043 | 10 | 0 | 0 | False |
| bd-1w | tt | zero_shot | rf | -0.018 | 0.082 | 5 | 0 | 0 | False |
| bd-1w | tt | full | xgb | 0.564 | 1.492 | 3 | 1 | 2 | False |
| bd-1w | tt | one_known | xgb | 0.315 | 0.231 | 1 | 0 | 5 | False |
| bd-1w | tt | zero_shot | xgb | 0.190 | 0.127 | 0 | 0 | 0 | False |
| bd-1w | tt | full | mlp | 0.059 | 2.665 | 5 | 1 | 0 | False |
| bd-1w | tt | one_known | mlp | 1.500 | 0.416 | 0 | 0 | 3 | False |
| bd-1w | tt | zero_shot | mlp | 0.116 | 0.944 | 4 | 0 | 0 | False |

## r_pervictim.pdf (6.0 x 6.6 in)
MPI time, socket-aware, 1W (response models: their only form). Skill = 1 - mean log MAE / mean baseline MAE over 10 seeds, per victim (top) and per co-runner (bottom). Colour clipped at -1; * = out of the inhibitor hull (Quicksilver). base_mae = baseline log MAE for that app.

### one_known, by victim

| victim | base_mae | ridge | lasso | svr | rf | xgb | mlp | absolute | delta |
|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 0.247 | 0.288 | 0.152 | 0.295 | 0.409 | 0.342 | 0.262 | -0.001 | -0.016 |
| Beatnik | 0.204 | 0.468 | 0.384 | 0.411 | 0.478 | 0.436 | 0.427 | 0.436 | 0.559 |
| FIESTA | 0.175 | -0.148 | -0.183 | -0.010 | 0.023 | -0.000 | -0.112 | -0.195 | -0.027 |
| Kripke | 0.050 | -0.666 | -0.698 | -0.166 | -0.182 | -0.566 | -0.652 | -2.329 | -3.232 |
| Laghos | 0.259 | 0.051 | 0.075 | 0.094 | 0.093 | 0.068 | 0.015 | -0.246 | 0.175 |
| LAMMPS | 0.121 | -0.233 | -0.317 | -0.207 | -0.187 | -0.173 | -0.272 | -1.167 | -0.663 |
| MiniFE | 0.172 | -0.138 | -0.135 | -0.002 | 0.031 | -0.064 | -0.198 | -0.584 | -0.324 |
| miniVite | 0.093 | 0.160 | 0.149 | 0.275 | 0.259 | 0.192 | 0.078 | -0.497 | 0.046 |
| Quicksilver | 0.023 | -2.162 | -3.136 | -0.838 | -0.503 | -0.742 | -1.860 | -5.821 | -2.431 |
| TriC | 0.076 | 0.383 | 0.402 | 0.337 | 0.409 | 0.416 | 0.349 | -0.491 | 0.029 |

### one_known, by corunner

| corunner | base_mae | ridge | lasso | svr | rf | xgb | mlp | absolute | delta |
|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 0.097 | 0.001 | 0.039 | 0.043 | 0.064 | -0.003 | -0.071 | -0.135 | -0.069 |
| Beatnik | 0.226 | 0.269 | 0.238 | 0.270 | 0.398 | 0.279 | 0.218 | 0.131 | 0.384 |
| FIESTA | 0.090 | 0.074 | 0.016 | 0.204 | 0.314 | 0.249 | 0.070 | -2.003 | -1.220 |
| Kripke | 0.113 | 0.075 | 0.057 | 0.064 | 0.226 | 0.189 | 0.028 | -0.506 | -0.109 |
| Laghos | 0.116 | -0.067 | -0.171 | 0.005 | 0.098 | 0.094 | -0.057 | -0.229 | -0.094 |
| LAMMPS | 0.099 | -0.317 | -0.312 | -0.204 | -0.131 | -0.204 | -0.316 | -0.818 | -0.384 |
| MiniFE | 0.069 | -0.504 | -0.429 | -0.015 | -0.093 | -0.123 | -0.510 | -0.952 | -0.593 |
| miniVite | 0.123 | 0.182 | 0.180 | 0.232 | 0.300 | 0.249 | 0.177 | -0.379 | -0.150 |
| Quicksilver | 0.127 | -0.200 | -0.600 | 0.026 | -0.469 | -0.314 | -0.225 | -1.858 | -1.556 |
| TriC | 0.330 | 0.167 | 0.126 | 0.200 | 0.345 | 0.226 | 0.143 | 0.200 | 0.382 |

### zero_shot, by victim

| victim | base_mae | ridge | lasso | svr | rf | xgb | mlp | absolute | delta |
|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 0.241 | 0.172 | -0.063 | 0.123 | 0.314 | 0.200 | 0.166 | 0.240 | 0.300 |
| Beatnik | 0.263 | 0.287 | 0.212 | 0.257 | 0.310 | 0.277 | 0.287 | 0.253 | 0.525 |
| FIESTA | 0.145 | -0.230 | -0.352 | -0.114 | -0.194 | -0.158 | -0.251 | -0.691 | -0.174 |
| Kripke | 0.041 | -1.635 | -1.391 | -0.759 | -0.474 | -1.016 | -2.046 | -2.211 | -4.935 |
| Laghos | 0.227 | -0.036 | -0.140 | 0.032 | -0.106 | -0.134 | -0.102 | -0.695 | 0.206 |
| LAMMPS | 0.125 | -0.510 | -0.351 | -0.370 | -0.282 | -0.338 | -0.506 | -1.369 | -0.870 |
| MiniFE | 0.147 | -0.711 | -0.479 | -0.213 | -0.411 | -0.539 | -0.680 | -0.758 | -0.459 |
| miniVite | 0.104 | 0.159 | 0.133 | 0.330 | 0.146 | 0.098 | 0.163 | -0.481 | 0.186 |
| Quicksilver | 0.024 | -3.257 | -5.852 | -1.175 | -0.870 | -1.277 | -3.044 | -13.843 | -6.039 |
| TriC | 0.147 | 0.291 | 0.155 | 0.376 | 0.307 | 0.268 | 0.256 | -0.485 | 0.012 |

### zero_shot, by corunner

| corunner | base_mae | ridge | lasso | svr | rf | xgb | mlp | absolute | delta |
|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 0.124 | 0.130 | 0.075 | -0.003 | 0.265 | 0.064 | 0.064 | 0.087 | 0.277 |
| Beatnik | 0.280 | 0.196 | 0.166 | 0.177 | 0.275 | 0.159 | 0.158 | 0.016 | 0.420 |
| FIESTA | 0.079 | -0.223 | -0.370 | -0.149 | 0.031 | -0.091 | -0.302 | -2.650 | -1.438 |
| Kripke | 0.090 | -0.378 | -0.479 | -0.064 | 0.029 | -0.011 | -0.538 | -1.137 | -0.725 |
| Laghos | 0.115 | -0.369 | -0.457 | -0.082 | -0.204 | -0.219 | -0.300 | -0.561 | -0.161 |
| LAMMPS | 0.107 | -0.713 | -0.523 | -0.375 | -0.433 | -0.511 | -0.695 | -1.063 | -0.561 |
| MiniFE | 0.065 | -1.515 | -1.019 | -0.210 | -0.667 | -0.664 | -1.485 | -1.650 | -0.924 |
| miniVite | 0.128 | 0.188 | 0.129 | 0.234 | 0.328 | 0.237 | 0.223 | -0.216 | 0.061 |
| Quicksilver | 0.106 | -0.496 | -1.683 | -0.064 | -0.739 | -0.669 | -0.496 | -3.321 | -1.790 |
| TriC | 0.350 | 0.130 | 0.105 | 0.131 | 0.185 | 0.119 | 0.109 | -0.036 | 0.177 |

