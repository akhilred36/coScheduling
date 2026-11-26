# noise_cv.pdf (single panel, request N1)

Replicate CV = sample SD (ddof=1) / mean over the 8-node isolated replicates (processed_data/data_isolated.csv).

| App | n reps | CV T_MPI (%) | CV T_app (%) |
|---|---:|---:|---:|
| Laghos | 4 | 27.45 | 4.83 |
| LAMMPS | 4 | 16.13 | 0.93 |
| FIESTA | 4 | 8.65 | 1.03 |
| AMG2023 | 4 | 7.79 | 0.29 |
| miniVite | 4 | 5.21 | 1.13 |
| Beatnik | 4 | 3.29 | 1.22 |
| Kripke | 4 | 3.22 | 0.16 |
| TriC | 4 | 2.61 | 1.40 |
| MiniFE | 4 | 2.32 | 0.13 |
| Quicksilver | 4 | 0.52 | 0.28 |

Median CV T_MPI 4.25%; median CV T_app 0.98%.
Apps with CV T_app > 2%: Laghos
