## A3. Isolated 8-node application profiles
|  | T_app (s) | T_MPI (s) | comm_frac | messages | bytes | msg rate (1/s) | byte rate (B/s) | mean msg size (B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMG2023 | 342 | 12.8 | 0.0373 | 1.06e+08 | 1.01e+12 | 3.11e+05 | 2.94e+09 | 9.47e+03 |
| Beatnik | 398 | 99.9 | 0.251 | 2e+07 | 1.09e+13 | 5.03e+04 | 2.75e+10 | 5.46e+05 |
| FIESTA | 359 | 65.7 | 0.183 | 6.45e+05 | 1.64e+12 | 1.8e+03 | 4.57e+09 | 2.54e+06 |
| Kripke | 354 | 17 | 0.048 | 3.03e+05 | 3.88e+11 | 854 | 1.09e+09 | 1.28e+06 |
| Laghos | 331 | 54.9 | 0.166 | 8.01e+08 | 6.34e+11 | 2.42e+06 | 1.91e+09 | 791 |
| LAMMPS | 355 | 17.4 | 0.049 | 4.57e+05 | 6.7e+11 | 1.29e+03 | 1.89e+09 | 1.47e+06 |
| MiniFE | 388 | 13.6 | 0.035 | 6.79e+06 | 3.82e+11 | 1.75e+04 | 9.85e+08 | 5.63e+04 |
| miniVite | 371 | 51.6 | 0.139 | 2.41e+07 | 1.13e+12 | 6.49e+04 | 3.06e+09 | 4.71e+04 |
| Quicksilver | 295 | 134 | 0.454 | 7.63e+05 | 1.6e+08 | 2.59e+03 | 5.43e+05 | 210 |
| TriC | 378 | 204 | 0.54 | 7.12e+06 | 1.02e+13 | 1.88e+04 | 2.7e+10 | 1.43e+06 |

## A4. Campaign counts (processed_data replicate-level CSVs)
| campaign | configs | min reps | max reps | runs |
|---|---:|---:|---:|---:|
| isolated apps, 8 node | 10 | 4 | 4 | 40 |
| isolated apps, 1 node | 6 | 4 | 4 | 24 |
| isolated inhibitors, 8 node | 287 | 3 | 4 | 1146 |
| App-Inhibitor, 8 node | 2870 | 2 | 4 | 11478 |
| App-App, 8 node | 55 | 4 | 4 | 220 |
| App-App, 1 node | 21 | 4 | 4 | 84 |

## A6. Clipping at 1
| set | n | frac raw<1 | frac raw==1 | frac floor (<=1) | min raw |
|---|---:|---:|---:|---:|---:|
| App-App MPI, non-self | 90 | 0.2 | 0 | 0.2 | 0.8313 |
| App-App MPI, self | 20 | 0.45 | 0 | 0.45 | 0.8452 |
| App-App total, non-self | 90 | 0.3667 | 0.03333 | 0.4 | 0.9714 |
| App-App total, self | 20 | 0.5 | 0.05 | 0.55 | 0.9751 |
| App-Inhibitor MPI | 2870 | 0.1544 | 0 | 0.1544 | 0.6874 |

App-Inhibitor total-time raw ratio: frac<1 = 0.2021, frac<=1 = 0.2129, min = 0.9419, n = 2870

## B1. Replicate noise
|  | CV T_MPI | CV T_app | T_app range (s) | n |
|---|---:|---:|---:|---:|
| AMG2023 | 0.0779 | 0.0029 | 2.0000 | 4.0000 |
| Beatnik | 0.0329 | 0.0122 | 10.9821 | 4.0000 |
| FIESTA | 0.0865 | 0.0103 | 7.7589 | 4.0000 |
| Kripke | 0.0322 | 0.0016 | 1.0000 | 4.0000 |
| Laghos | 0.2745 | 0.0483 | 39.0647 | 4.0000 |
| LAMMPS | 0.1613 | 0.0093 | 7.0000 | 4.0000 |
| MiniFE | 0.0232 | 0.0013 | 1.0000 | 4.0000 |
| miniVite | 0.0521 | 0.0113 | 10.0000 | 4.0000 |
| Quicksilver | 0.0052 | 0.0028 | 2.0000 | 4.0000 |
| TriC | 0.0261 | 0.0140 | 12.0000 | 4.0000 |

Isolated 8-node CV T_MPI: median 0.0425, mean 0.0772; CV T_app: median 0.0098, max 0.0483

| set | n | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| isolated apps T_MPI (per app) | 10.0000 | 0.0425 | 0.0844 | 0.1727 | 0.2745 |
| isolated inhibitors T_MPI (per config) | 287.0000 | 0.0063 | 0.0353 | 0.0965 | 1.0277 |
| App-Inhibitor app T_MPI (per config) | 2870.0000 | 0.0831 | 0.1787 | 0.3106 | 1.7422 |
| App-App T_MPI (per pair endpoint, incl. self) | 110.0000 | 0.0669 | 0.1221 | 0.2257 | 0.3805 |
| isolated apps T_app | 10.0000 | 0.0098 | 0.0120 | 0.0174 | 0.0483 |
| App-Inhibitor app T_app | 2870.0000 | 0.0152 | 0.0322 | 0.0532 | 0.5450 |
| App-App T_app | 110.0000 | 0.0112 | 0.0155 | 0.0233 | 0.0744 |

Isolated inhibitors: fraction of configs with CV>10%: 0.091; >50%: 0.014
Top-5 isolated-inhibitor CV configs (msg, wait us, sparsity -> CV):
  (4000, 100000, 0.1) -> 1.028
  (200, 25000, 0.01) -> 0.986
  (500, 5000, 0.01) -> 0.711
  (500, 5000, 0.05) -> 0.579
  (10000, 200000, 0.002237136) -> 0.406

| App | median CV T_MPI | p90 CV T_MPI |
|---|---:|---:|
| AMG2023 | 0.1562 | 0.5204 |
| Beatnik | 0.0520 | 0.1039 |
| FIESTA | 0.1795 | 0.3796 |
| Kripke | 0.0506 | 0.3026 |
| Laghos | 0.1470 | 0.2955 |
| LAMMPS | 0.1903 | 0.5183 |
| MiniFE | 0.0877 | 0.3971 |
| miniVite | 0.0875 | 0.1805 |
| Quicksilver | 0.0281 | 0.0479 |
| TriC | 0.0500 | 0.0967 |

Propagated SE of raw App-App MPI ratio (delta method, 4 co reps x 4 iso reps), non-self:
  median SE 0.0459, p90 0.1638, max 0.3599
  outcomes with raw-1 < 2 SE: 58 / 90 (of which 18 are at/below the floor); above-floor outcomes within 2 SE of 1: 40 / 72
  outcomes with clipped slowdown >= 1.05 that are > 2 SE above 1: 30 / 46
  check: max |raw(canonical) - raw(replicates)| = 6.66e-16
  per-victim median SE: AMG2023 0.079, Beatnik 0.031, FIESTA 0.118, Kripke 0.023, Laghos 0.167, LAMMPS 0.137, MiniFE 0.032, miniVite 0.057, Quicksilver 0.015, TriC 0.026

## B2. App-App slowdown distribution (8 node)
| set | n | mean | median | p75 | p90 | max | floor | frac>=1.05 | frac>=1.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPI, non-self | 90.000 | 1.123 | 1.051 | 1.131 | 1.311 | 2.209 | 0.200 | 0.511 | 0.311 |
| total, non-self | 90.000 | 1.016 | 1.001 | 1.025 | 1.043 | 1.192 | 0.400 | 0.067 | 0.033 |
| MPI, self (20 copies) | 20.000 | 1.142 | 1.004 | 1.176 | 1.545 | 1.588 | 0.450 | 0.400 | 0.350 |
| total, self (20 copies) | 20.000 | 1.044 | 1.000 | 1.015 | 1.146 | 1.290 | 0.550 | 0.200 | 0.200 |

Self-pairs, MPI and total slowdown of both copies (A, B) and mean:
| jobA_id | MPI A | MPI B | MPI mean | total A | total B |
|---|---:|---:|---:|---:|---:|
| AMG2023 | 1.186 | 1.069 | 1.128 | 1.004 | 1.000 |
| Beatnik | 1.588 | 1.587 | 1.587 | 1.128 | 1.130 |
| FIESTA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Kripke | 1.006 | 1.002 | 1.004 | 1.000 | 1.000 |
| Laghos | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LAMMPS | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MiniFE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| miniVite | 1.173 | 1.135 | 1.154 | 1.020 | 1.014 |
| Quicksilver | 1.021 | 1.000 | 1.010 | 1.010 | 1.001 |
| TriC | 1.540 | 1.541 | 1.540 | 1.290 | 1.290 |

Top-5 most asymmetric non-self pairs (MPI, by |log s_A - log s_B|):
| pair | s_A(B) | s_B(A) | |log ratio| |
|---|---:|---:|---:|
| TriC + Laghos | 1.067 | 2.209 | 0.727 |
| TriC + AMG2023 | 1.057 | 1.826 | 0.547 |
| MiniFE + TriC | 1.730 | 1.087 | 0.464 |
| Beatnik + Laghos | 1.159 | 1.820 | 0.451 |
| Beatnik + TriC | 1.631 | 1.119 | 0.377 |

Top-5 most asymmetric non-self pairs (total time):
| pair | s_A(B) | s_B(A) | |log ratio| |
|---|---:|---:|---:|
| TriC + Laghos | 1.036 | 1.192 | 0.141 |
| Beatnik + Laghos | 1.031 | 1.128 | 0.090 |
| AMG2023 + Laghos | 1.004 | 1.086 | 0.079 |
| Beatnik + TriC | 1.147 | 1.063 | 0.075 |
| TriC + LAMMPS | 1.046 | 1.000 | 0.045 |

## B3. Victim / aggressor decomposition of log slowdown (non-self, 90 outcomes)
### MPI time: n = 90, var(log s) = 0.02329
R2 victim-only = 0.243; aggressor-only = 0.301; additive victim+aggressor = 0.558; residual (interaction + noise) share = 0.442
Partial R2 adding aggressor to victim = 0.315; adding victim to aggressor = 0.257
Grand mean log s = 0.1030 (x1.109)
Effects (sum-to-zero; factor = exp(effect)); sorted by victim effect:
|  | victim effect (log) | victim factor | aggressor effect (log) | aggressor factor |
|---|---:|---:|---:|---:|
| Laghos | 0.155 | 1.168 | -0.036 | 0.965 |
| Beatnik | 0.103 | 1.109 | 0.121 | 1.129 |
| AMG2023 | 0.051 | 1.052 | -0.008 | 0.992 |
| MiniFE | 0.002 | 1.002 | -0.049 | 0.952 |
| miniVite | -0.006 | 0.994 | -0.015 | 0.985 |
| TriC | -0.006 | 0.994 | 0.205 | 1.227 |
| Kripke | -0.058 | 0.944 | -0.024 | 0.976 |
| FIESTA | -0.064 | 0.938 | -0.048 | 0.953 |
| LAMMPS | -0.075 | 0.928 | -0.059 | 0.942 |
| Quicksilver | -0.101 | 0.904 | -0.086 | 0.918 |

Aggressor ranking: TriC, Beatnik, AMG2023, miniVite, Kripke, Laghos, FIESTA, MiniFE, LAMMPS, Quicksilver

### Total time: n = 90, var(log s) = 0.00088
R2 victim-only = 0.418; aggressor-only = 0.122; additive victim+aggressor = 0.575; residual (interaction + noise) share = 0.425
Partial R2 adding aggressor to victim = 0.157; adding victim to aggressor = 0.453
Grand mean log s = 0.0156 (x1.016)
Effects (sum-to-zero; factor = exp(effect)); sorted by victim effect:
|  | victim effect (log) | victim factor | aggressor effect (log) | aggressor factor |
|---|---:|---:|---:|---:|
| Laghos | 0.035 | 1.035 | -0.003 | 0.997 |
| Beatnik | 0.028 | 1.029 | 0.014 | 1.014 |
| TriC | 0.027 | 1.028 | 0.029 | 1.030 |
| miniVite | -0.004 | 0.996 | -0.001 | 0.999 |
| Quicksilver | -0.011 | 0.989 | -0.011 | 0.989 |
| AMG2023 | -0.013 | 0.987 | -0.001 | 0.999 |
| MiniFE | -0.015 | 0.985 | -0.007 | 0.993 |
| Kripke | -0.015 | 0.985 | -0.008 | 0.992 |
| FIESTA | -0.016 | 0.984 | -0.005 | 0.995 |
| LAMMPS | -0.016 | 0.984 | -0.008 | 0.992 |

Aggressor ranking: TriC, Beatnik, miniVite, AMG2023, Laghos, FIESTA, MiniFE, LAMMPS, Kripke, Quicksilver

Sensitivity: MPI time using unclipped raw ratio (log raw):
### MPI time, unclipped: n = 90, var(log s) = 0.02755
R2 victim-only = 0.208; aggressor-only = 0.274; additive victim+aggressor = 0.501; residual (interaction + noise) share = 0.499
Partial R2 adding aggressor to victim = 0.293; adding victim to aggressor = 0.227
Grand mean log s = 0.0888 (x1.093)
Effects (sum-to-zero; factor = exp(effect)); sorted by victim effect:
|  | victim effect (log) | victim factor | aggressor effect (log) | aggressor factor |
|---|---:|---:|---:|---:|
| Laghos | 0.139 | 1.149 | -0.045 | 0.956 |
| Beatnik | 0.117 | 1.125 | 0.122 | 1.130 |
| AMG2023 | 0.064 | 1.066 | -0.002 | 0.998 |
| miniVite | 0.010 | 1.011 | 0.001 | 1.001 |
| TriC | 0.009 | 1.009 | 0.213 | 1.238 |
| MiniFE | -0.032 | 0.969 | -0.052 | 0.950 |
| Kripke | -0.044 | 0.957 | -0.029 | 0.972 |
| FIESTA | -0.082 | 0.921 | -0.046 | 0.955 |
| Quicksilver | -0.090 | 0.914 | -0.098 | 0.907 |
| LAMMPS | -0.091 | 0.913 | -0.065 | 0.938 |

Aggressor ranking: TriC, Beatnik, miniVite, AMG2023, Kripke, Laghos, FIESTA, MiniFE, LAMMPS, Quicksilver

Spearman s_A(B) vs s_B(A) over 45 non-self pairs, MPI: rho = 0.412 (p = 0.00487); on raw ratios rho = 0.428 (p = 0.0034)
Spearman s_A(B) vs s_B(A) over 45 non-self pairs, total: rho = 0.216 (p = 0.154); on raw ratios rho = 0.389 (p = 0.00822)

## B4. 1-node subset vs 8-node on the same 6 apps / 21 pairs
| nodes | time | set | n | median | p75 | p90 | max | floor | min_raw |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MPI | non-self | 30.000 | 1.032 | 1.115 | 1.391 | 1.662 | 0.300 | 0.953 |
| 1 | MPI | self | 12.000 | 1.042 | 1.093 | 1.252 | 1.472 | 0.167 | 0.988 |
| 1 | total | non-self | 30.000 | 1.002 | 1.007 | 1.015 | 1.026 | 0.400 | 0.996 |
| 1 | total | self | 12.000 | 1.003 | 1.009 | 1.011 | 1.012 | 0.417 | 0.999 |
| 8 | MPI | non-self | 30.000 | 1.088 | 1.159 | 1.314 | 1.730 | 0.200 | 0.884 |
| 8 | MPI | self | 12.000 | 1.067 | 1.540 | 1.582 | 1.588 | 0.500 | 0.845 |
| 8 | total | non-self | 30.000 | 1.004 | 1.035 | 1.048 | 1.147 | 0.433 | 0.984 |
| 8 | total | self | 12.000 | 1.007 | 1.128 | 1.274 | 1.290 | 0.500 | 0.986 |

check: max |8-node processed-route MPI slowdown - canonical| = 4.44e-16
Spearman 1-node vs 8-node MPI slowdown, same 30 directional outcomes: rho = -0.437 (p = 0.0158)
| victim | aggressor | s 1-node | s 8-node |
|---|---|---:|---:|
| MiniFE | TriC | 1.113 | 1.730 |
| Beatnik | TriC | 1.009 | 1.631 |
| MiniFE | Beatnik | 1.073 | 1.347 |
| miniVite | Beatnik | 1.000 | 1.310 |
| miniVite | TriC | 1.000 | 1.288 |
| Beatnik | MiniFE | 1.000 | 1.187 |
| Beatnik | LAMMPS | 1.000 | 1.172 |
| Beatnik | miniVite | 1.000 | 1.160 |
| Beatnik | FIESTA | 1.000 | 1.154 |
| miniVite | FIESTA | 1.000 | 1.135 |
| TriC | Beatnik | 1.141 | 1.119 |
| LAMMPS | Beatnik | 1.419 | 1.115 |
| TriC | miniVite | 1.012 | 1.104 |
| FIESTA | LAMMPS | 1.000 | 1.103 |
| TriC | LAMMPS | 1.010 | 1.088 |
| TriC | MiniFE | 1.031 | 1.087 |
| LAMMPS | miniVite | 1.397 | 1.072 |
| MiniFE | miniVite | 1.043 | 1.066 |
| TriC | FIESTA | 1.148 | 1.066 |
| miniVite | MiniFE | 1.112 | 1.065 |
| LAMMPS | MiniFE | 1.390 | 1.060 |
| FIESTA | miniVite | 1.000 | 1.051 |
| miniVite | LAMMPS | 1.021 | 1.022 |
| LAMMPS | FIESTA | 1.243 | 1.004 |
| FIESTA | Beatnik | 1.011 | 1.000 |
| MiniFE | LAMMPS | 1.092 | 1.000 |
| FIESTA | TriC | 1.053 | 1.000 |
| FIESTA | MiniFE | 1.033 | 1.000 |
| LAMMPS | TriC | 1.662 | 1.000 |
| MiniFE | FIESTA | 1.115 | 1.000 |

