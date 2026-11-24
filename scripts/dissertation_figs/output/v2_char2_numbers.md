# v2 batch B8: unclipped JMs re-computations of the §4 characterization numbers (v2/v2_char2.py)

Ratio = T_co / T_base (ratio of rep-averaged means), unclipped; JMs baseline unless marked [J] (June, no socket matching). y = ln ratio. 90 non-self directed outcomes unless stated. JMs factors applied to AMG2023 and Quicksilver only (both timers), so [J] differs only for rows with those victims.

## 1. App-App matrices (victim rows x co-runner columns)

### MPI time (JMs; diagonal = pooled self-pair, both copies x 4 reps vs 50/50 baseline)

| victim \ corunner | AMG2023 | Beatnik | FIESTA | Kripke | Laghos | LAMMPS | MiniFE | miniVite | Quicksilver | TriC |
|---|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 1.244 | 1.699 | 1.129 | 1.205 | 1.151 | 1.144 | 0.983 | 1.152 | 1.101 | 2.109 |
| Beatnik | 1.144 | 1.587 | 1.154 | 1.205 | 1.159 | 1.172 | 1.187 | 1.160 | 1.166 | 1.631 |
| FIESTA | 1.016 | 0.884 | 0.961 | 1.240 | 0.975 | 1.103 | 0.927 | 1.051 | 1.019 | 0.935 |
| Kripke | 1.035 | 1.091 | 1.008 | 1.004 | 1.037 | 1.041 | 1.102 | 1.018 | 1.021 | 1.091 |
| Laghos | 1.564 | 1.820 | 1.177 | 1.081 | 0.894 | 0.917 | 0.979 | 1.320 | 0.854 | 2.209 |
| LAMMPS | 1.068 | 1.115 | 1.004 | 0.831 | 0.921 | 0.869 | 1.060 | 1.072 | 1.003 | 1.000 |
| MiniFE | 0.920 | 1.347 | 0.908 | 1.089 | 0.917 | 0.931 | 0.951 | 1.066 | 0.913 | 1.730 |
| miniVite | 1.042 | 1.310 | 1.135 | 1.050 | 1.051 | 1.022 | 1.065 | 1.154 | 1.020 | 1.288 |
| Quicksilver | 1.029 | 1.017 | 1.035 | 1.017 | 1.030 | 1.033 | 1.025 | 1.021 | 1.029 | 1.022 |
| TriC | 1.057 | 1.119 | 1.066 | 1.065 | 1.067 | 1.088 | 1.087 | 1.104 | 1.039 | 1.540 |

Row (victim) and column (co-runner) means over the 9 off-diagonal cells, MPI time; geometric mean = exp(mean y):

| app | victim mean (geo) | victim mean [J] | victim arith mean | co-runner mean (geo) | co-runner mean [J] | co-runner arith mean |
|---|---|---|---|---|---|---|
| Laghos | 1.262 | 1.262 | 1.325 | 1.031 | 1.029 | 1.034 |
| AMG2023 | 1.260 | 1.165 | 1.297 | 1.086 | 1.083 | 1.097 |
| Beatnik | 1.212 | 1.212 | 1.220 | 1.235 | 1.219 | 1.267 |
| miniVite | 1.104 | 1.104 | 1.109 | 1.104 | 1.092 | 1.107 |
| TriC | 1.077 | 1.077 | 1.077 | 1.376 | 1.351 | 1.446 |
| MiniFE | 1.065 | 1.065 | 1.091 | 1.044 | 1.042 | 1.046 |
| Kripke | 1.049 | 1.049 | 1.049 | 1.080 | 1.067 | 1.087 |
| Quicksilver | 1.025 | 1.010 | 1.025 | 1.011 | 1.001 | 1.015 |
| FIESTA | 1.012 | 1.012 | 1.017 | 1.065 | 1.053 | 1.068 |
| LAMMPS | 1.005 | 1.005 | 1.008 | 1.047 | 1.035 | 1.050 |

### app time (JMs; diagonal = pooled self-pair, both copies x 4 reps vs 50/50 baseline)

| victim \ corunner | AMG2023 | Beatnik | FIESTA | Kripke | Laghos | LAMMPS | MiniFE | miniVite | Quicksilver | TriC |
|---|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 1.005 | 1.013 | 1.002 | 1.005 | 1.004 | 1.004 | 0.996 | 1.004 | 1.001 | 1.019 |
| Beatnik | 1.024 | 1.129 | 1.025 | 1.041 | 1.031 | 1.029 | 1.035 | 1.030 | 1.033 | 1.147 |
| FIESTA | 0.992 | 0.984 | 0.989 | 1.001 | 0.987 | 0.997 | 0.984 | 0.991 | 0.990 | 0.990 |
| Kripke | 1.001 | 1.004 | 0.999 | 0.999 | 1.001 | 1.001 | 1.005 | 0.999 | 0.999 | 1.003 |
| Laghos | 1.086 | 1.128 | 1.025 | 1.008 | 0.977 | 0.982 | 0.992 | 1.045 | 0.971 | 1.192 |
| LAMMPS | 0.997 | 1.000 | 0.996 | 0.984 | 0.989 | 0.987 | 1.001 | 0.999 | 0.994 | 0.992 |
| MiniFE | 0.990 | 1.003 | 0.988 | 0.997 | 0.990 | 0.990 | 0.991 | 0.995 | 0.989 | 1.012 |
| miniVite | 1.001 | 1.042 | 1.015 | 1.001 | 1.004 | 1.000 | 1.005 | 1.017 | 0.997 | 1.036 |
| Quicksilver | 1.013 | 1.009 | 1.016 | 1.008 | 1.014 | 1.016 | 1.012 | 1.010 | 1.014 | 1.011 |
| TriC | 1.029 | 1.063 | 1.035 | 1.034 | 1.036 | 1.046 | 1.047 | 1.056 | 1.019 | 1.290 |

Row (victim) and column (co-runner) means over the 9 off-diagonal cells, app time; geometric mean = exp(mean y):

| app | victim mean (geo) | victim mean [J] | victim arith mean | co-runner mean (geo) | co-runner mean [J] | co-runner arith mean |
|---|---|---|---|---|---|---|
| Laghos | 1.046 | 1.046 | 1.048 | 1.006 | 1.005 | 1.006 |
| Beatnik | 1.043 | 1.043 | 1.044 | 1.027 | 1.025 | 1.027 |
| TriC | 1.041 | 1.041 | 1.041 | 1.042 | 1.041 | 1.045 |
| Quicksilver | 1.012 | 1.005 | 1.012 | 0.999 | 0.999 | 0.999 |
| miniVite | 1.011 | 1.011 | 1.011 | 1.014 | 1.014 | 1.014 |
| AMG2023 | 1.005 | 1.002 | 1.005 | 1.014 | 1.014 | 1.015 |
| Kripke | 1.001 | 1.001 | 1.001 | 1.009 | 1.008 | 1.009 |
| MiniFE | 0.995 | 0.995 | 0.995 | 1.008 | 1.007 | 1.009 |
| LAMMPS | 0.995 | 0.995 | 0.995 | 1.007 | 1.006 | 1.007 |
| FIESTA | 0.991 | 0.991 | 0.991 | 1.011 | 1.010 | 1.011 |

### Top 8 off-diagonal MPI cells

| victim | co-runner | MPI ratio | [J] | app ratio | frac sock1 |
|---|---|---|---|---|---|
| Laghos | TriC | 2.209 | 2.209 | 1.192 | 0.250 |
| AMG2023 | TriC | 2.109 | 1.826 | 1.019 | 0.750 |
| Laghos | Beatnik | 1.820 | 1.820 | 1.128 | 1.000 |
| MiniFE | TriC | 1.730 | 1.730 | 1.012 | 0.250 |
| AMG2023 | Beatnik | 1.699 | 1.547 | 1.013 | 0.500 |
| Beatnik | TriC | 1.631 | 1.631 | 1.147 | 0.500 |
| Laghos | AMG2023 | 1.564 | 1.564 | 1.086 | 1.000 |
| MiniFE | Beatnik | 1.347 | 1.347 | 1.003 | 0.250 |

### 10 self-pairs (pooled both copies)

| self-pair | MPI | MPI [J] | app | app [J] |
|---|---|---|---|---|
| AMG2023 | 1.244 | 1.128 | 1.005 | 1.002 |
| Beatnik | 1.587 | 1.587 | 1.129 | 1.129 |
| FIESTA | 0.961 | 0.961 | 0.989 | 0.989 |
| Kripke | 1.004 | 1.004 | 0.999 | 0.999 |
| Laghos | 0.894 | 0.894 | 0.977 | 0.977 |
| LAMMPS | 0.869 | 0.869 | 0.987 | 0.987 |
| MiniFE | 0.951 | 0.951 | 0.991 | 0.991 |
| miniVite | 1.154 | 1.154 | 1.017 | 1.017 |
| Quicksilver | 1.029 | 1.010 | 1.014 | 1.006 |
| TriC | 1.540 | 1.540 | 1.290 | 1.290 |

## 2. Asymmetry: s_A(B) vs s_B(A) over the 45 non-self pairs

Spearman over unordered pairs depends on which app is called A. Primary = the as-run A/B orientation (the canonical pair.csv one, as in the old number); also the APPS-order orientation, and an orientation-free version (both orderings, n = 90 points; its p-value treats them as independent and is optimistic, so only rho is given).

- MPI time JMs: as-run rho = 0.326, p = 0.0287; APPS-order rho = 0.408, p = 0.00535; orientation-free rho = 0.343 (n = 45 pairs)
- MPI time [J]: as-run rho = 0.428, p = 0.0034; APPS-order rho = 0.453, p = 0.00176; orientation-free rho = 0.435 (n = 45 pairs)
- app time JMs: as-run rho = 0.204, p = 0.179; APPS-order rho = 0.232, p = 0.126; orientation-free rho = 0.237 (n = 45 pairs)
- app time [J]: as-run rho = 0.205, p = 0.177; APPS-order rho = 0.248, p = 0.101; orientation-free rho = 0.246 (n = 45 pairs)

Top 5 most asymmetric pairs (MPI, by |y_A - y_B|; A = first named app):

| pair | s_A(B) | s_B(A) | |y_A - y_B| | |.| [J] |
|---|---|---|---|---|
| TriC + Laghos | 1.067 | 2.209 | 0.727 | 0.727 |
| TriC + AMG2023 | 1.057 | 2.109 | 0.691 | 0.547 |
| MiniFE + TriC | 1.730 | 1.087 | 0.464 | 0.464 |
| Beatnik + Laghos | 1.159 | 1.820 | 0.451 | 0.451 |
| Beatnik + AMG2023 | 1.144 | 1.699 | 0.395 | 0.302 |

## 3. Fixed-effects decomposition of y (90 non-self outcomes)

### MPI time, JMs: n = 90, var(y) = 0.02994
R² victim-only = 0.236; co-runner-only = 0.272; additive = 0.523; residual (interaction + noise) = 0.477
Overall mean (grand mean of y, sum-to-zero effects) = 0.0981 (x1.103)
| app | victim factor | co-runner factor |
|---|---|---|
| AMG2023 | 1.142 | 0.999 |
| Laghos | 1.137 | 0.948 |
| Beatnik | 1.115 | 1.133 |
| miniVite | 1.001 | 1.001 |
| TriC | 1.000 | 1.247 |
| MiniFE | 0.959 | 0.942 |
| Kripke | 0.948 | 0.974 |
| Quicksilver | 0.920 | 0.908 |
| FIESTA | 0.913 | 0.956 |
| LAMMPS | 0.904 | 0.938 |
Strongest victim: AMG2023 (x1.142); weakest: LAMMPS (x0.904). Strongest co-runner: TriC (x1.247); weakest: Quicksilver (x0.908).
Co-runner ranking: TriC, Beatnik, miniVite, AMG2023, Kripke, FIESTA, Laghos, MiniFE, LAMMPS, Quicksilver

### MPI time, [J]: n = 90, var(y) = 0.02755
R² victim-only = 0.208; co-runner-only = 0.274; additive = 0.501; residual (interaction + noise) = 0.499
Overall mean (grand mean of y, sum-to-zero effects) = 0.0888 (x1.093)
| app | victim factor | co-runner factor |
|---|---|---|
| Laghos | 1.149 | 0.956 |
| Beatnik | 1.125 | 1.130 |
| AMG2023 | 1.066 | 0.998 |
| miniVite | 1.011 | 1.001 |
| TriC | 1.009 | 1.238 |
| MiniFE | 0.969 | 0.950 |
| Kripke | 0.957 | 0.972 |
| FIESTA | 0.921 | 0.955 |
| Quicksilver | 0.914 | 0.907 |
| LAMMPS | 0.913 | 0.938 |
Strongest victim: Laghos (x1.149); weakest: LAMMPS (x0.913). Strongest co-runner: TriC (x1.238); weakest: Quicksilver (x0.907).
Co-runner ranking: TriC, Beatnik, miniVite, AMG2023, Kripke, Laghos, FIESTA, MiniFE, LAMMPS, Quicksilver

### app time, JMs: n = 90, var(y) = 0.00099
R² victim-only = 0.399; co-runner-only = 0.132; additive = 0.565; residual (interaction + noise) = 0.435
Overall mean (grand mean of y, sum-to-zero effects) = 0.0137 (x1.014)
| app | victim factor | co-runner factor |
|---|---|---|
| Beatnik | 1.031 | 1.016 |
| Laghos | 1.031 | 0.996 |
| TriC | 1.030 | 1.032 |
| miniVite | 0.998 | 1.000 |
| Quicksilver | 0.997 | 0.985 |
| AMG2023 | 0.992 | 1.000 |
| Kripke | 0.987 | 0.994 |
| MiniFE | 0.981 | 0.993 |
| LAMMPS | 0.980 | 0.991 |
| FIESTA | 0.977 | 0.995 |
Strongest victim: Beatnik (x1.031); weakest: FIESTA (x0.977). Strongest co-runner: TriC (x1.032); weakest: Quicksilver (x0.985).
Co-runner ranking: TriC, Beatnik, miniVite, AMG2023, Laghos, FIESTA, Kripke, MiniFE, LAMMPS, Quicksilver

### app time, [J]: n = 90, var(y) = 0.00100
R² victim-only = 0.405; co-runner-only = 0.124; additive = 0.565; residual (interaction + noise) = 0.435
Overall mean (grand mean of y, sum-to-zero effects) = 0.0127 (x1.013)
| app | victim factor | co-runner factor |
|---|---|---|
| Laghos | 1.032 | 0.996 |
| Beatnik | 1.032 | 1.015 |
| TriC | 1.031 | 1.031 |
| miniVite | 0.999 | 1.001 |
| Quicksilver | 0.991 | 0.985 |
| AMG2023 | 0.990 | 1.000 |
| Kripke | 0.988 | 0.994 |
| MiniFE | 0.982 | 0.993 |
| LAMMPS | 0.981 | 0.991 |
| FIESTA | 0.978 | 0.995 |
Strongest victim: Laghos (x1.032); weakest: FIESTA (x0.978). Strongest co-runner: TriC (x1.031); weakest: Quicksilver (x0.985).
Co-runner ranking: TriC, Beatnik, miniVite, AMG2023, Laghos, FIESTA, Kripke, MiniFE, LAMMPS, Quicksilver

## 4. Profile relationship (MPI y vs isolated profile; rates over T_app, as select_inhibitors.py)

- (a) Co-runner byte rate: rho = 0.295, p = 0.00471; [J] rho = 0.306, p = 0.0034
- (b) Co-runner message rate: rho = 0.026, p = 0.806; [J] rho = 0.054, p = 0.61
- (c) Victim comm. fraction: rho = 0.043, p = 0.69; [J] rho = 0.103, p = 0.336

slowdown_vs_profile.pdf regenerated: y = unclipped JMs MPI ratio (log axis, range 0.831-2.209), dotted line at 1; no floor encoding. Correlations on y (rank-identical to the ratio).

## 5. Propagated SE of the App-App MPI ratio (delta method), JMs

SE = ratio x sqrt(relvar(co-run mean) + relvar(June iso mean) + (f_d x SE(d))²), where f_d = d ln(mean baseline)/dd is the replicate-weighted socket-1 share and SE(d) is the paired pinned SE (AMG2023, Quicksilver only; 0 elsewhere). Co-run replicates are socket-adjusted (each scaled by mean baseline / its own baseline) so the known socket shift is not counted as noise; the unadjusted variant is shown for comparison. 4 co-run reps x 4 June iso reps per outcome.

| variant | n | median_se | p90_se | max_se | harm_resolved | n_harm | above1_resolved | speedup_resolved | n_below1 | within_2se |
|---|---|---|---|---|---|---|---|---|---|---|
| JMs | 90.0000 | 0.0459 | 0.1614 | 0.3599 | 33.0000 | 50.0000 | 39.0000 | 5.0000 | 16.0000 | 46.0000 |
| JMs, unadjusted co-run var | 90.0000 | 0.0459 | 0.1674 | 0.3599 | 31.0000 | 50.0000 | 33.0000 | 5.0000 | 16.0000 | 52.0000 |
| [J] | 90.0000 | 0.0459 | 0.1638 | 0.3599 | 30.0000 | 46.0000 | 32.0000 | 5.0000 | 18.0000 | 53.0000 |

Columns: harm_resolved = ratio >= 1.05 and > 1 + 2 SE (of n_harm with ratio >= 1.05); above1_resolved = any ratio > 1 + 2 SE; speedup_resolved = ratio < 1 and < 1 - 2 SE (of n_below1); within_2se = |ratio - 1| <= 2 SE.

Per-victim SE (median over its 9 outcomes); requested apps: Laghos, LAMMPS, FIESTA:

| victim | median_se | max_se | median_seJ | median_ratio |
|---|---|---|---|---|
| AMG2023 | 0.0547 | 0.1386 | 0.0788 | 1.1509 |
| Beatnik | 0.0307 | 0.0462 | 0.0307 | 1.1658 |
| FIESTA | 0.1178 | 0.1833 | 0.1178 | 1.0161 |
| Kripke | 0.0235 | 0.1117 | 0.0235 | 1.0370 |
| Laghos | 0.1667 | 0.3599 | 0.1667 | 1.1769 |
| LAMMPS | 0.1365 | 0.1903 | 0.1365 | 1.0039 |
| MiniFE | 0.0323 | 0.1821 | 0.0323 | 0.9306 |
| miniVite | 0.0574 | 0.1341 | 0.0574 | 1.0508 |
| Quicksilver | 0.0109 | 0.0179 | 0.0146 | 1.0247 |
| TriC | 0.0255 | 0.0300 | 0.0255 | 1.0675 |

d-term share: median SE contribution from SE(d) for AMG2023 0.0097, Quicksilver 0.0056.

Resolved speedups (ratio < 1 - 2 SE): FIESTA<-Beatnik 0.884±0.041; MiniFE<-FIESTA 0.908±0.011; MiniFE<-Quicksilver 0.913±0.014; MiniFE<-Laghos 0.917±0.016; MiniFE<-AMG2023 0.920±0.026

## 6. Inhibitor response (inhib_response.pdf, unclipped JMs MPI ratio)

- AMG2023: (a) slice 1.038-6.383 [J 0.992-5.812], n = 32; (b) slice 1.324-3.450 [J 1.089-3.142]; whole grid (287 configs) 0.895-6.461 [J 0.775-5.812]; frac < 1 over grid 0.073
- miniVite: (a) slice 1.024-1.898 [J 1.024-1.898], n = 32; (b) slice 1.066-1.835 [J 1.066-1.835]; whole grid (287 configs) 0.920-2.208 [J 0.920-2.208]; frac < 1 over grid 0.052
- Quicksilver: (a) slice 1.012-1.095 [J 0.992-1.095], n = 32; (b) slice 1.019-1.095 [J 0.987-1.095]; whole grid (287 configs) 1.004-1.151 [J 0.970-1.141]; frac < 1 over grid 0.000
Quicksilver max over the grid: 1.151 [J 1.141] at config 8_200000_0_0.1.
Same victims and slices as the original C5 (base grid). Dotted line at 1; nothing clipped.

## 7. 1-node vs 8-node, 6 apps (Beatnik, FIESTA, LAMMPS, MiniFE, miniVite, TriC), 30 non-self outcomes each

1-node uses the June 1-node isolated baseline (no 1-node pinned runs, so no socket matching is possible). 8-node is shown with JMs and with the June baseline [J]. None of the 6 apps has a JMs factor applied (only AMG2023 and Quicksilver do), so the 8-node JMs and [J] values coincide here.

| set | n | median | p90 | max | min | frac_lt1 | frac_ge105 |
|---|---|---|---|---|---|---|---|
| 1 node [J] | 30.000 | 1.032 | 1.391 | 1.662 | 0.953 | 0.300 | 0.433 |
| 8 nodes JMs | 30.000 | 1.088 | 1.314 | 1.730 | 0.884 | 0.200 | 0.733 |
| 8 nodes [J] | 30.000 | 1.088 | 1.314 | 1.730 | 0.884 | 0.200 | 0.733 |

KS two-sample (1 node vs 8 nodes JMs): D = 0.300, p = 0.135
Per-pair Spearman (same 30 directed outcomes): rho = -0.442, p = 0.0143
- TriC <- LAMMPS (victim <- co-runner): 1 node 1.010, 8 nodes 1.088
- TriC <- MiniFE (victim <- co-runner): 1 node 1.031, 8 nodes 1.087
- LAMMPS <- TriC (victim <- co-runner): 1 node 1.662, 8 nodes 1.000
- MiniFE <- TriC (victim <- co-runner): 1 node 1.113, 8 nodes 1.730
(The pair named 'TriC->LAMMPS' is ambiguous in direction; both directions are listed. Above, the victim is the first app.)

scale_1node.pdf regenerated: ECDF of the unclipped ratio, log x 0.866-1.782, dotted line at 1.

## 8. noise_cv.pdf

No dependence on clipping or on the baseline: both panels are replicate CVs of raw times (isolated apps T_MPI and T_app; isolated inhibitors, App-Inhibitor and App-App co-run T_MPI), computed from the processed replicates in figs_batch1.py C1. No slowdown ratio enters. No change.

