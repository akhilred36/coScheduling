# Socket-pinned isolated runs: experiment 20261107_180934

Placement: `srun --whole --mpibind=off --cpu-bind=none` + `numactl --physcpubind --localalloc`, 56 ranks/node on SMT thread 0 (sock0 = PUs 0-55, sock1 = PUs 56-111). Each job ran its sock0 and sock1 steps back to back on the same 8 nodes; even replicates sock0 first, odd replicates sock1 first.

## Run validity
- app sock0: 40 run dirs, valid 40, PLACEMENT_FAILED 0, no .mpiP 0, missing placement_check 0
- app sock1: 40 run dirs, valid 40, PLACEMENT_FAILED 0, no .mpiP 0, missing placement_check 0
- inhib sock0: 36 run dirs, valid 36, PLACEMENT_FAILED 0, no .mpiP 0, missing placement_check 0
- inhib sock1: 36 run dirs, valid 36, PLACEMENT_FAILED 0, no .mpiP 0, missing placement_check 0
- valid app runs: threads seen ['0']; PU range sock0 0-55, sock1 56-111; all 448-448 ranks.

## S1. Isolated socket effect for apps, paired within job: d = log(sock1) - log(sock0)
Negative d means sock1 is faster. CI = t interval over the 4 replicates (n = 4 is too small for a bootstrap). 'sock1<sock0' counts the replicates with sock1 faster. Order effect: o = (mean d | sock0 ran first - mean d | sock1 ran first)/2 = log(second step) - log(first step); the order-balanced socket effect is the average of the two order means, which equals mean d when the design is balanced (2 + 2).

### log T_MPI
| app | n | mean d | median d | CI lo | CI hi | sock1<sock0 | d|sock0 first | d|sock1 first | order o | ratio sock1/sock0 |
|---|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 4 | -0.1971 | -0.2035 | -0.2571 | -0.1370 | 4/4 | -0.2035 | -0.1907 | -0.0064 | 0.8211 |
| Beatnik | 4 | -0.0107 | -0.0186 | -0.0529 | 0.0315 | 3/4 | -0.0205 | -0.0010 | -0.0097 | 0.9893 |
| FIESTA | 4 | -0.0628 | -0.0349 | -0.2149 | 0.0893 | 3/4 | -0.1083 | -0.0172 | -0.0455 | 0.9391 |
| Kripke | 4 | -0.0265 | -0.0125 | -0.1018 | 0.0488 | 3/4 | -0.0405 | -0.0125 | -0.0140 | 0.9738 |
| Laghos | 4 | -0.2207 | -0.2117 | -0.6873 | 0.2458 | 2/4 | 0.0307 | -0.4722 | 0.2514 | 0.8019 |
| LAMMPS | 4 | -0.6314 | -0.7398 | -1.4323 | 0.1695 | 3/4 | -1.0247 | -0.2381 | -0.3933 | 0.5319 |
| MiniFE | 4 | -0.0807 | -0.0502 | -0.3084 | 0.1471 | 3/4 | -0.1633 | 0.0020 | -0.0827 | 0.9225 |
| miniVite | 4 | -0.0453 | -0.1221 | -0.3376 | 0.2470 | 3/4 | -0.1436 | 0.0530 | -0.0983 | 0.9557 |
| Quicksilver | 4 | -0.0357 | -0.0329 | -0.0713 | -0.0001 | 4/4 | -0.0254 | -0.0460 | 0.0103 | 0.9649 |
| TriC | 4 | -0.0020 | -0.0056 | -0.0253 | 0.0213 | 3/4 | 0.0016 | -0.0056 | 0.0036 | 0.9980 |
Pooled over apps (mean of per-app means), app-cluster bootstrap (10,000 draws, seed 923): -0.1313 [-0.2556, -0.0427] (x0.8770); median over all 40 pairs -0.0354; sock1 faster in 31/40 pairs.
Wilcoxon signed-rank on all 40 pairs: p = 0.000123.
Pooled order effect (log second-step - log first-step, socket-balanced): -0.0385 [-0.1380, +0.0507].
Least squares d = socket + order*(+1 if sock0 first else -1): socket -0.1313, order -0.0385.

### log T_app
| app | n | mean d | median d | CI lo | CI hi | sock1<sock0 | d|sock0 first | d|sock1 first | order o | ratio sock1/sock0 |
|---|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | 4 | -0.0071 | -0.0068 | -0.0094 | -0.0047 | 4/4 | -0.0073 | -0.0068 | -0.0002 | 0.9930 |
| Beatnik | 4 | -0.0078 | -0.0105 | -0.0218 | 0.0062 | 3/4 | -0.0105 | -0.0051 | -0.0027 | 0.9922 |
| FIESTA | 4 | -0.0065 | -0.0019 | -0.0214 | 0.0084 | 4/4 | -0.0114 | -0.0016 | -0.0049 | 0.9935 |
| Kripke | 4 | -0.0028 | -0.0014 | -0.0090 | 0.0035 | 2/4 | -0.0042 | -0.0014 | -0.0014 | 0.9972 |
| Laghos | 4 | -0.0380 | -0.0333 | -0.1213 | 0.0452 | 2/4 | 0.0063 | -0.0824 | 0.0444 | 0.9627 |
| LAMMPS | 4 | -0.0566 | -0.0577 | -0.1317 | 0.0185 | 4/4 | -0.0949 | -0.0184 | -0.0382 | 0.9450 |
| MiniFE | 4 | -0.0039 | -0.0026 | -0.0141 | 0.0064 | 3/4 | -0.0077 | -0.0000 | -0.0039 | 0.9962 |
| miniVite | 4 | -0.0061 | -0.0161 | -0.0495 | 0.0373 | 3/4 | -0.0198 | 0.0076 | -0.0137 | 0.9939 |
| Quicksilver | 4 | -0.0165 | -0.0162 | -0.0301 | -0.0030 | 4/4 | -0.0128 | -0.0203 | 0.0037 | 0.9836 |
| TriC | 4 | -0.0026 | -0.0039 | -0.0129 | 0.0076 | 2/4 | -0.0013 | -0.0039 | 0.0013 | 0.9974 |
Pooled over apps (mean of per-app means), app-cluster bootstrap (10,000 draws, seed 923): -0.0148 [-0.0265, -0.0056] (x0.9853); median over all 40 pairs -0.0078; sock1 faster in 31/40 pairs.
Wilcoxon signed-rank on all 40 pairs: p = 6.6e-05.
Pooled order effect (log second-step - log first-step, socket-balanced): -0.0016 [-0.0131, +0.0112].
Least squares d = socket + order*(+1 if sock0 first else -1): socket -0.0148, order -0.0016.

Mean isolated times (s) over 4 replicates:
| app | T_MPI sock0 (s) | T_MPI sock1 (s) | T_app sock0 (s) | T_app sock1 (s) |
|---|---|---|---|---|
| AMG2023 | 12.33 | 10.12 | 344.25 | 341.83 |
| Beatnik | 110.53 | 109.36 | 409.28 | 406.10 |
| FIESTA | 63.08 | 59.00 | 358.52 | 356.19 |
| Kripke | 19.02 | 18.57 | 361.00 | 360.00 |
| Laghos | 51.28 | 39.50 | 331.00 | 318.25 |
| LAMMPS | 36.97 | 17.47 | 374.25 | 353.25 |
| MiniFE | 14.84 | 13.65 | 389.00 | 387.50 |
| miniVite | 58.39 | 56.13 | 378.25 | 376.00 |
| Quicksilver | 134.97 | 130.21 | 297.16 | 292.27 |
| TriC | 215.06 | 214.63 | 389.50 | 388.50 |

## S2. Isolated inhibitors
The inhibitor runs for a fixed duration (`-r 360`), so T_app is ~361 s by construction and is not a socket outcome. The socket outcomes are the work done in that window (log total messages sent; bytes = messages x size, so the same d) and log T_MPI. d = log(sock1) - log(sock0); positive d for messages means the inhibitor injects MORE traffic on sock1.
- valid inhibitor runs: 72/72; threads ['0']; PU range sock0 0-55, sock1 56-111.
- 2 runs report unexpected NUMA sets (sock1/8_inhib_20000_100000_d_0.1_2, sock1/8_inhib_40000_1000000_d_0.7_0). Their probes show one host (dane261) with sub-NUMA clustering OFF (2 NUMA domains: socket 0 = numa 0, socket 1 = numa 1); PUs are on the correct socket, so the runs are kept. The same jobs' sock0 steps are on that host too (numa 0 for all its sock0 ranks), so the pair stays matched.
- T_app range over all 72 runs: 360.0-362.0 s (fixed-duration check).

### log messages sent (work in the 360 s window)
| inhibitor (-m bytes, -w us, -s) | n | mean d | CI lo | CI hi | sock1<sock0 | order o | ratio sock1/sock0 |
|---|---|---|---|---|---|---|---|
| m=200 w=200000 s=0.00223714 | 4 | -0.0028 | -0.0117 | 0.0061 | 1/4 | 0.0028 | 0.9972 |
| m=500 w=5000 s=0.05 | 4 | -0.0009 | -0.0032 | 0.0014 | 3/4 | -0.0008 | 0.9991 |
| m=20000 w=100000 s=0.1 | 4 | 0.0004 | -0.0004 | 0.0012 | 1/4 | 0.0004 | 1.0004 |
| m=40000 w=1000000 s=0.1 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
| m=40000 w=1000000 s=0.7 | 4 | 0.0015 | -0.0012 | 0.0042 | 0/4 | -0.0000 | 1.0015 |
| m=400000 w=1000000 s=0.5 | 4 | -0.0000 | -0.0052 | 0.0052 | 1/4 | -0.0020 | 1.0000 |
| m=2000000 w=1000000 s=0.3 | 4 | -0.0029 | -0.0083 | 0.0025 | 2/4 | -0.0029 | 0.9971 |
| m=2500000 w=200000 s=0.00223714 | 4 | -0.0047 | -0.0209 | 0.0115 | 1/4 | 0.0053 | 0.9953 |
| m=2500000 w=600000 s=0.00223714 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
Pooled over configs (mean of per-config means), config-cluster bootstrap (10,000 draws, seed 923): -0.0011 [-0.0023, +0.0001] (x0.9989); sock1 lower in 9/36 pairs; Wilcoxon p = 0.298.
Pooled order effect (log second-step - log first-step): +0.0003 [-0.0011, +0.0019].

### log T_MPI
| inhibitor (-m bytes, -w us, -s) | n | mean d | CI lo | CI hi | sock1<sock0 | order o | ratio sock1/sock0 |
|---|---|---|---|---|---|---|---|
| m=200 w=200000 s=0.00223714 | 4 | 0.4306 | -1.3344 | 2.1956 | 3/4 | -0.5863 | 1.5382 |
| m=500 w=5000 s=0.05 | 4 | 0.0170 | -0.0280 | 0.0620 | 1/4 | 0.0171 | 1.0171 |
| m=20000 w=100000 s=0.1 | 4 | -0.0067 | -0.0226 | 0.0093 | 3/4 | -0.0062 | 0.9933 |
| m=40000 w=1000000 s=0.1 | 4 | -0.0567 | -0.1011 | -0.0124 | 4/4 | -0.0104 | 0.9448 |
| m=40000 w=1000000 s=0.7 | 4 | -0.0172 | -0.0355 | 0.0012 | 4/4 | 0.0033 | 0.9830 |
| m=400000 w=1000000 s=0.5 | 4 | -0.0002 | -0.0125 | 0.0122 | 2/4 | -0.0003 | 0.9998 |
| m=2000000 w=1000000 s=0.3 | 4 | -0.0013 | -0.0092 | 0.0066 | 2/4 | 0.0039 | 0.9987 |
| m=2500000 w=200000 s=0.00223714 | 4 | 0.0885 | -0.2145 | 0.3916 | 2/4 | -0.0971 | 1.0926 |
| m=2500000 w=600000 s=0.00223714 | 4 | -0.0254 | -0.0464 | -0.0044 | 4/4 | 0.0008 | 0.9749 |
Pooled over configs (mean of per-config means), config-cluster bootstrap (10,000 draws, seed 923): +0.0476 [-0.0187, +0.1515] (x1.0488); sock1 lower in 25/36 pairs; Wilcoxon p = 0.0171.
Pooled order effect (log second-step - log first-step): -0.0750 [-0.2077, +0.0029].

### log T_app (sanity: should be ~0)
| inhibitor (-m bytes, -w us, -s) | n | mean d | CI lo | CI hi | sock1<sock0 | order o | ratio sock1/sock0 |
|---|---|---|---|---|---|---|---|
| m=200 w=200000 s=0.00223714 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
| m=500 w=5000 s=0.05 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
| m=20000 w=100000 s=0.1 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
| m=40000 w=1000000 s=0.1 | 4 | -0.0014 | -0.0039 | 0.0012 | 2/4 | 0.0000 | 0.9986 |
| m=40000 w=1000000 s=0.7 | 4 | 0.0000 | -0.0036 | 0.0036 | 1/4 | 0.0014 | 1.0000 |
| m=400000 w=1000000 s=0.5 | 4 | 0.0007 | -0.0015 | 0.0029 | 0/4 | -0.0007 | 1.0007 |
| m=2000000 w=1000000 s=0.3 | 4 | -0.0007 | -0.0049 | 0.0035 | 2/4 | 0.0007 | 0.9993 |
| m=2500000 w=200000 s=0.00223714 | 4 | 0.0000 | 0.0000 | 0.0000 | 0/4 | 0.0000 | 1.0000 |
| m=2500000 w=600000 s=0.00223714 | 4 | -0.0007 | -0.0029 | 0.0015 | 1/4 | -0.0007 | 0.9993 |
Pooled over configs (mean of per-config means), config-cluster bootstrap (10,000 draws, seed 923): -0.0002 [-0.0006, +0.0002] (x0.9998); sock1 lower in 6/36 pairs; Wilcoxon p = 0.261.
Pooled order effect (log second-step - log first-step): +0.0001 [-0.0003, +0.0005].

Mean isolated inhibitor outcomes over 4 replicates:
| inhibitor | msgs sock0 | msgs sock1 | T_MPI sock0 (s) | T_MPI sock1 (s) |
|---|---|---|---|---|
| m=200 w=200000 s=0.00223714 | 1.609e+06 | 1.605e+06 | 0.5661 | 1.501 |
| m=500 w=5000 s=0.05 | 6.836e+08 | 6.83e+08 | 14.93 | 15.24 |
| m=20000 w=100000 s=0.1 | 7.054e+07 | 7.057e+07 | 16.43 | 16.32 |
| m=40000 w=1000000 s=0.1 | 7.357e+06 | 7.357e+06 | 3.487 | 3.294 |
| m=40000 w=1000000 s=0.7 | 4.737e+07 | 4.744e+07 | 23.21 | 22.82 |
| m=400000 w=1000000 s=0.5 | 2.568e+07 | 2.568e+07 | 105.8 | 105.7 |
| m=2000000 w=1000000 s=0.3 | 1.028e+07 | 1.025e+07 | 191.2 | 191 |
| m=2500000 w=200000 s=0.00223714 | 1.543e+06 | 1.536e+06 | 15.23 | 16.85 |
| m=2500000 w=600000 s=0.00223714 | 5.3e+05 | 5.3e+05 | 5.351 | 5.216 |

## S3. Launcher/epoch drift (NOT a socket effect): new sock0 vs June isolated baselines
June isolated runs: mpibind, 56 ranks/node on PUs 112-167 (SMT thread 1 of socket 0), no `--whole`. New sock0: numactl, PUs 0-55 (thread 0 of the same cores), `--whole`, Nov 2026. Same socket, so the difference mixes launcher, SMT thread index, memory policy (--localalloc), software/system state, and node set.

### T_MPI
| app | June T_MPI (s) | new sock0 T_MPI (s) | ratio MPI | log diff MPI | June CV MPI | new sock0 CV MPI | |log diff|/June CV MPI |
|---|---|---|---|---|---|---|---|
| AMG2023 | 12.7781 | 12.3298 | 0.9649 | -0.0357 | 0.0779 | 0.0426 | 0.4586 |
| Beatnik | 99.9064 | 110.5256 | 1.1063 | 0.1010 | 0.0329 | 0.0258 | 3.0742 |
| FIESTA | 65.6754 | 63.0763 | 0.9604 | -0.0404 | 0.0865 | 0.1166 | 0.4667 |
| Kripke | 17.0066 | 19.0242 | 1.1186 | 0.1121 | 0.0322 | 0.1284 | 3.4835 |
| Laghos | 54.9450 | 51.2785 | 0.9333 | -0.0691 | 0.2745 | 0.3303 | 0.2516 |
| LAMMPS | 17.3812 | 36.9718 | 2.1271 | 0.7548 | 0.1613 | 0.5773 | 4.6781 |
| MiniFE | 13.6014 | 14.8356 | 1.0907 | 0.0869 | 0.0232 | 0.1086 | 3.7467 |
| miniVite | 51.5973 | 58.3850 | 1.1316 | 0.1236 | 0.0521 | 0.0938 | 2.3715 |
| Quicksilver | 133.9955 | 134.9729 | 1.0073 | 0.0073 | 0.0052 | 0.0234 | 1.3985 |
| TriC | 204.0725 | 215.0597 | 1.0538 | 0.0524 | 0.0261 | 0.0846 | 2.0094 |
Mean log diff over apps +0.1093 (median +0.0696); apps with |log diff| > 2 June CVs: Beatnik, Kripke, LAMMPS, MiniFE, miniVite, TriC.

### T_app
| app | June T_app (s) | new sock0 T_app (s) | ratio app | log diff app | June CV app | new sock0 CV app | |log diff|/June CV app |
|---|---|---|---|---|---|---|---|
| AMG2023 | 342.5000 | 344.2500 | 1.0051 | 0.0051 | 0.0029 | 0.0015 | 1.7455 |
| Beatnik | 397.7344 | 409.2801 | 1.0290 | 0.0286 | 0.0122 | 0.0063 | 2.3415 |
| FIESTA | 359.0915 | 358.5201 | 0.9984 | -0.0016 | 0.0103 | 0.0092 | 0.1550 |
| Kripke | 354.5000 | 361.0000 | 1.0183 | 0.0182 | 0.0016 | 0.0099 | 11.1564 |
| Laghos | 331.2377 | 331.0000 | 0.9993 | -0.0007 | 0.0483 | 0.0578 | 0.0149 |
| LAMMPS | 354.7500 | 374.2500 | 1.0550 | 0.0535 | 0.0093 | 0.0567 | 5.7454 |
| MiniFE | 388.2500 | 389.0000 | 1.0019 | 0.0019 | 0.0013 | 0.0051 | 1.4986 |
| miniVite | 371.2500 | 378.2500 | 1.0189 | 0.0187 | 0.0113 | 0.0139 | 1.6538 |
| Quicksilver | 295.0000 | 297.1574 | 1.0073 | 0.0073 | 0.0028 | 0.0100 | 2.6326 |
| TriC | 378.0000 | 389.5000 | 1.0304 | 0.0300 | 0.0140 | 0.0459 | 2.1409 |
Mean log diff over apps +0.0161 (median +0.0127); apps with |log diff| > 2 June CVs: Beatnik, Kripke, LAMMPS, Quicksilver, TriC.

### Inhibitors: new sock0 vs earlier isolated inhibitor runs (`processed_data/data_inhib_isolated.csv`, mpibind, `-r 360`, 8 nodes)
| inhibitor | n old | old msgs | new sock0 msgs | log diff msgs | old CV msgs | old T_MPI | new sock0 T_MPI | log diff T_MPI | old CV T_MPI |
|---|---|---|---|---|---|---|---|---|---|
| m=200 w=200000 s=0.00223714 | 4 | 1.609e+06 | 1.609e+06 | 0 | 0 | 0.3596 | 0.5661 | 0.4539 | 0.0952 |
| m=500 w=5000 s=0.05 | 4 | 6.285e+08 | 6.836e+08 | 0.08392 | 0.07836 | 42.94 | 14.93 | -1.057 | 0.5785 |
| m=20000 w=100000 s=0.1 | 4 | 7.08e+07 | 7.054e+07 | -0.003645 | 0.0006059 | 15.05 | 16.43 | 0.08767 | 0.01318 |
| m=40000 w=1000000 s=0.1 | 4 | 7.357e+06 | 7.357e+06 | 0 | 0 | 3.432 | 3.487 | 0.01586 | 0.02985 |
| m=40000 w=1000000 s=0.7 | 4 | 4.762e+07 | 4.737e+07 | -0.005183 | 0.001706 | 21.73 | 23.21 | 0.06621 | 0.02737 |
| m=400000 w=1000000 s=0.5 | 4 | 2.573e+07 | 2.568e+07 | -0.001961 | 0.03157 | 105.1 | 105.8 | 0.006403 | 0.07605 |
| m=2000000 w=1000000 s=0.3 | 4 | 1.013e+07 | 1.028e+07 | 0.01482 | 0.02413 | 194 | 191.2 | -0.01428 | 0.02343 |
| m=2500000 w=200000 s=0.00223714 | 4 | 1.546e+06 | 1.543e+06 | -0.001595 | 0.001195 | 14.67 | 15.23 | 0.03708 | 0.02827 |
| m=2500000 w=600000 s=0.00223714 | 4 | 5.304e+05 | 5.3e+05 | -0.000845 | 0.001379 | 4.977 | 5.351 | 0.07249 | 0.1005 |
Mean log diff msgs over configs +0.0095 (median -0.0008).
Mean log diff T_MPI over configs -0.0368 (median +0.0371).

## S4. Isolated vs co-run socket effect on log T_MPI (sock1 - sock0)
Co-run columns reproduce figs_batch1 C8 (App-Inh: per config with reps on both sockets; App-App: per non-self directional outcome with reps on both sockets). 'co-run minus isolated' = App-Inh mean minus isolated mean d: the part of the co-run socket gap not explained by the isolated (baseline) socket gap.
| app | iso mean d | iso CI lo | iso CI hi | iso sock1<sock0 | App-Inh n | App-Inh median | App-Inh mean | App-App n | App-App median | App-App mean | App-Inh mean - iso | App-App mean - iso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AMG2023 | -0.1971 | -0.2571 | -0.1370 | 4/4 | 258 | -0.2425 | -0.3152 | 7 | -0.1818 | -0.1883 | -0.1182 | 0.0088 |
| Beatnik | -0.0107 | -0.0529 | 0.0315 | 3/4 | 255 | 0.0048 | 0.0071 | 6 | -0.0044 | 0.0075 | 0.0178 | 0.0182 |
| FIESTA | -0.0628 | -0.2149 | 0.0893 | 3/4 | 258 | -0.2366 | -0.3031 | 7 | -0.3647 | -0.3375 | -0.2403 | -0.2747 |
| Kripke | -0.0265 | -0.1018 | 0.0488 | 3/4 | 249 | -0.0635 | -0.1556 | 8 | -0.0193 | -0.0031 | -0.1291 | 0.0234 |
| Laghos | -0.2207 | -0.6873 | 0.2458 | 2/4 | 248 | -0.1552 | -0.1704 | 7 | -0.0472 | -0.0627 | 0.0504 | 0.1581 |
| LAMMPS | -0.6314 | -1.4323 | 0.1695 | 3/4 | 247 | -0.1650 | -0.1318 | 9 | -0.1584 | -0.0645 | 0.4996 | 0.5669 |
| MiniFE | -0.0807 | -0.3084 | 0.1471 | 3/4 | 240 | -0.1042 | -0.2384 | 8 | -0.0632 | -0.0806 | -0.1577 | 0.0001 |
| miniVite | -0.0453 | -0.3376 | 0.2470 | 3/4 | 240 | -0.0038 | -0.0403 | 7 | -0.0004 | 0.0204 | 0.0050 | 0.0657 |
| Quicksilver | -0.0357 | -0.0713 | -0.0001 | 4/4 | 245 | -0.0516 | -0.0565 | 8 | -0.0507 | -0.0452 | -0.0208 | -0.0095 |
| TriC | -0.0020 | -0.0253 | 0.0213 | 3/4 | 260 | -0.0113 | -0.0065 | 9 | 0.0082 | 0.0067 | -0.0045 | 0.0087 |
Across apps, Spearman(isolated mean d, App-Inh co-run mean d) = 0.65 (p = 0.043).
Pooled: isolated mean of app means -0.1313; co-run App-Inh mean of app means -0.1411; App-App -0.0747.

## S5. Socket-matched co-run slowdowns
Baselines: **J** = June isolated mean (chapter's current numbers); **N0** = new pinned sock0 for all reps (requested 'before'); **NM** = new pinned baseline of the socket each rep's victim ran on (requested 'after'); **JM** = June mean, times exp(mean paired isolated d) for socket-1 reps (socket matching without the S3 launcher/epoch drift); **JMs** = JM with d set to 0 unless its 95% CI excludes 0 (T_MPI: AMG2023 and Quicksilver only; T_app: AMG2023 and Quicksilver). Slowdown = max(1, mean co / mean matched baseline). SE = delta-method SE of the raw ratio (co reps + baseline reps).
Effect of socket matching alone: compare N0 -> NM (new baselines) or J -> JM / JMs (June baselines).

### App-App, non-self directional outcomes, T_MPI slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 90.0000 | 1.1233 | 1.0510 | 1.3113 | 2.2094 | 0.2000 | 0.5111 | 0.3111 | 0.0423 | 0.1504 |
| N0 | 90.0000 | 1.0997 | 1.0074 | 1.2640 | 2.3674 | 0.4556 | 0.3111 | 0.1889 | 0.0527 | 0.1607 |
| NM | 90.0000 | 1.1366 | 1.0220 | 1.2957 | 2.5325 | 0.3556 | 0.3556 | 0.2667 | 0.0547 | 0.1631 |
| JM | 90.0000 | 1.1983 | 1.0982 | 1.5459 | 2.3245 | 0.1222 | 0.6667 | 0.4889 | 0.0552 | 0.2284 |
| JMs | 90.0000 | 1.1352 | 1.0648 | 1.3113 | 2.2094 | 0.1778 | 0.5556 | 0.3667 | 0.0423 | 0.1504 |

### App-App, non-self directional outcomes, T_app slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 90.0000 | 1.0162 | 1.0014 | 1.0427 | 1.1925 | 0.3667 | 0.0667 | 0.0333 | 0.0067 | 0.0182 |
| N0 | 90.0000 | 1.0090 | 1.0000 | 1.0178 | 1.1934 | 0.6778 | 0.0444 | 0.0333 | 0.0074 | 0.0227 |
| NM | 90.0000 | 1.0116 | 1.0000 | 1.0193 | 1.2050 | 0.5778 | 0.0556 | 0.0444 | 0.0075 | 0.0217 |
| JM | 90.0000 | 1.0228 | 1.0104 | 1.0470 | 1.2037 | 0.2000 | 0.0778 | 0.0444 | 0.0084 | 0.0210 |
| JMs | 90.0000 | 1.0171 | 1.0036 | 1.0427 | 1.1925 | 0.3333 | 0.0667 | 0.0333 | 0.0067 | 0.0182 |

### App-Inhibitor configs, T_MPI slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 2870.0000 | 1.4983 | 1.2036 | 2.6375 | 8.5104 | 0.1544 | 0.7049 | 0.6254 | 0.0733 | 0.3235 |
| N0 | 2870.0000 | 1.4364 | 1.0907 | 2.5226 | 6.0230 | 0.3038 | 0.5606 | 0.4850 | 0.0825 | 0.3505 |
| NM | 2870.0000 | 1.4980 | 1.1143 | 2.7001 | 6.6976 | 0.2185 | 0.6070 | 0.5237 | 0.0851 | 0.3362 |
| JM | 2870.0000 | 1.5914 | 1.2630 | 2.7868 | 13.1154 | 0.0645 | 0.7746 | 0.6916 | 0.0914 | 0.4081 |
| JMs | 2870.0000 | 1.5184 | 1.2067 | 2.6801 | 8.5104 | 0.1226 | 0.7136 | 0.6321 | 0.0737 | 0.3273 |

### App-Inhibitor configs, T_app slowdown
| baseline | n | mean | median | p90 | max | frac raw<1 | frac >=1.05 | frac >=1.10 | median SE | p90 SE |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 2870.0000 | 1.0586 | 1.0218 | 1.1852 | 1.5095 | 0.2021 | 0.3150 | 0.1801 | 0.0089 | 0.0335 |
| N0 | 2870.0000 | 1.0480 | 1.0076 | 1.1592 | 1.5106 | 0.4233 | 0.2599 | 0.1474 | 0.0108 | 0.0361 |
| NM | 2870.0000 | 1.0516 | 1.0120 | 1.1635 | 1.5402 | 0.3394 | 0.2749 | 0.1519 | 0.0106 | 0.0345 |
| JM | 2870.0000 | 1.0651 | 1.0333 | 1.1903 | 1.5382 | 0.1226 | 0.3502 | 0.1871 | 0.0106 | 0.0361 |
| JMs | 2870.0000 | 1.0596 | 1.0226 | 1.1852 | 1.5095 | 0.1767 | 0.3195 | 0.1801 | 0.0089 | 0.0335 |

Check: J reproduces chapter B2/A6 (App-App MPI mean 1.123, median 1.051, max 2.209, floor 0.200; App-Inh MPI floor 0.154, n = 2870).

Per-victim median App-App MPI slowdown by baseline:
| victim | J | N0 | NM | JM | JMs | frac reps on sock1 |
|---|---|---|---|---|---|---|
| AMG2023 | 1.049 | 1.087 | 1.193 | 1.151 | 1.151 | 0.417 |
| Beatnik | 1.166 | 1.054 | 1.060 | 1.173 | 1.166 | 0.472 |
| FIESTA | 1.016 | 1.058 | 1.093 | 1.048 | 1.016 | 0.583 |
| Kripke | 1.037 | 1.000 | 1.000 | 1.049 | 1.037 | 0.472 |
| Laghos | 1.177 | 1.261 | 1.338 | 1.238 | 1.177 | 0.500 |
| LAMMPS | 1.004 | 1.000 | 1.000 | 1.419 | 1.004 | 0.639 |
| MiniFE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.472 |
| miniVite | 1.051 | 1.000 | 1.000 | 1.086 | 1.051 | 0.472 |
| Quicksilver | 1.011 | 1.004 | 1.017 | 1.025 | 1.025 | 0.417 |
| TriC | 1.067 | 1.013 | 1.014 | 1.069 | 1.067 | 0.556 |

## S6. Re-scoring frozen predictions against socket-matched App-App outcomes (no re-fit)
Predictions: random_split, non-self outcomes (90 directional, 45 pairs). Only the truth changes; predicted log slowdowns are as saved. Δ% = log MAE vs the isolated-speed baseline (constant 1); CI = paired pair-cluster bootstrap over the 45 non-self pairs (1,000 draws, seed 923, one shared draw matrix; re-implemented here). Pooled skill = 1 − MAE(method)/MAE(baseline). FB = false-benign rate at 1.05 over truly harmful pairs (pair worst endpoint, as in metrics.py).
MPI predictions: one-way MPI pipeline on the canonical snapshot, local `filtered_fit_mpi_time` run. Its constant-1, absolute and generic log MAE equal the artifact's control run (baseSnap) exactly; † its delta models use a different recipe from the artifact (artifact delta median 0.1235 vs local 0.1190), so treat them as indicative. The joint MPI model's per-outcome predictions are not on this machine (bidirectional MPI root missing), so joint is scored on total time only.
Total-time predictions: local bidirectional total-time `full_comparison` run, identical to the artifact's bidirTT run.

### T_MPI target (check: max |saved true slowdown − J truth| = 6.7e-16)

**isolated-speed baseline**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1030 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 29/45 |
| N0 | 0.0796 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 24/45 |
| NM | 0.1043 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 26/45 |
| JM | 0.1607 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 38/45 |
| JMs | 0.1122 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 32/45 |

**absolute**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0942 | -8.5292 | -12.8450 | -3.7480 | 0.0853 | 0.7241 | 29/45 |
| N0 | 0.0814 | 2.2592 | -4.7254 | 13.2799 | -0.0226 | 0.7917 | 24/45 |
| NM | 0.1012 | -2.9390 | -7.4079 | 2.9313 | 0.0294 | 0.8077 | 26/45 |
| JM | 0.1435 | -10.7253 | -13.9485 | -8.1299 | 0.1073 | 0.7895 | 38/45 |
| JMs | 0.1006 | -10.3446 | -14.3124 | -6.3985 | 0.1034 | 0.7500 | 32/45 |

**generic**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1472 | 42.9381 | 0.8944 | 120.9666 | -0.4294 | 0.0000 | 29/45 |
| N0 | 0.1714 | 115.2561 | 48.3343 | 261.9528 | -1.1526 | 0.0000 | 24/45 |
| NM | 0.1703 | 63.3160 | 17.6084 | 155.4109 | -0.6332 | 0.0000 | 26/45 |
| JM | 0.1501 | -6.6192 | -27.8298 | 26.7469 | 0.0662 | 0.0000 | 38/45 |
| JMs | 0.1443 | 28.5905 | -8.0006 | 94.6781 | -0.2859 | 0.0000 | 32/45 |

**delta (median)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1190 | 15.5569 | -19.3099 | 81.7672 | -0.1556 | 0.0000 | 29/45 |
| N0 | 0.1452 | 82.3860 | 25.1266 | 205.6307 | -0.8239 | 0.0000 | 24/45 |
| NM | 0.1451 | 39.1466 | 1.0503 | 116.4093 | -0.3915 | 0.0000 | 26/45 |
| JM | 0.1261 | -21.5456 | -39.1322 | 6.8985 | 0.2155 | 0.0000 | 38/45 |
| JMs | 0.1162 | 3.5125 | -26.3106 | 59.5556 | -0.0351 | 0.0000 | 32/45 |

**delta (OOD kernel)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1258 | 22.1459 | -17.9511 | 89.4610 | -0.2215 | 0.0000 | 29/45 |
| N0 | 0.1514 | 90.1187 | 29.2003 | 226.6318 | -0.9012 | 0.0000 | 24/45 |
| NM | 0.1461 | 40.1021 | -0.2116 | 120.9524 | -0.4010 | 0.0000 | 26/45 |
| JM | 0.1252 | -22.0800 | -42.6156 | 8.4904 | 0.2208 | 0.0000 | 38/45 |
| JMs | 0.1179 | 5.0998 | -29.3538 | 61.4920 | -0.0510 | 0.0000 | 32/45 |

**delta (uniform)†**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.1457 | 41.4438 | -1.9199 | 121.7460 | -0.4144 | 0.0000 | 29/45 |
| N0 | 0.1712 | 115.0411 | 43.3655 | 263.1827 | -1.1504 | 0.0000 | 24/45 |
| NM | 0.1702 | 63.2144 | 17.6815 | 159.4803 | -0.6321 | 0.0000 | 26/45 |
| JM | 0.1460 | -9.1517 | -31.8139 | 25.4375 | 0.0915 | 0.0000 | 38/45 |
| JMs | 0.1409 | 25.5786 | -12.0487 | 92.3199 | -0.2558 | 0.0000 | 32/45 |

### T_app target (check: max |saved true slowdown − J truth| = 2.2e-16)

**isolated-speed baseline**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |
| N0 | 0.0086 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 4/45 |
| NM | 0.0110 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |
| JM | 0.0220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 6/45 |
| JMs | 0.0165 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5/45 |

**absolute**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0162 | 3.6121 | -5.2886 | 17.1901 | -0.0361 | 1.0000 | 5/45 |
| N0 | 0.0097 | 13.6838 | -2.7422 | 55.0647 | -0.1368 | 1.0000 | 4/45 |
| NM | 0.0119 | 7.9918 | -3.7536 | 31.9803 | -0.0799 | 1.0000 | 5/45 |
| JM | 0.0213 | -3.0280 | -9.1361 | 4.8392 | 0.0303 | 1.0000 | 6/45 |
| JMs | 0.0166 | 0.6297 | -7.5324 | 11.8500 | -0.0063 | 1.0000 | 5/45 |

**generic**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0202 | 29.1951 | -9.7622 | 111.0049 | -0.2920 | 0.0000 | 5/45 |
| N0 | 0.0259 | 202.7301 | 85.0291 | 597.9282 | -2.0273 | 0.0000 | 4/45 |
| NM | 0.0258 | 133.6550 | 54.7046 | 355.9129 | -1.3365 | 0.0000 | 5/45 |
| JM | 0.0199 | -9.4146 | -31.3215 | 26.5391 | 0.0941 | 0.0000 | 6/45 |
| JMs | 0.0202 | 21.9246 | -13.1625 | 90.8018 | -0.2192 | 0.0000 | 5/45 |

**delta (median)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0209 | 33.8214 | -6.6813 | 117.6378 | -0.3382 | 0.0000 | 5/45 |
| N0 | 0.0266 | 210.5537 | 88.2865 | 630.6457 | -2.1055 | 0.0000 | 4/45 |
| NM | 0.0266 | 140.4561 | 57.2079 | 380.3860 | -1.4046 | 0.0000 | 5/45 |
| JM | 0.0199 | -9.5385 | -31.5528 | 25.2925 | 0.0954 | 0.0000 | 6/45 |
| JMs | 0.0206 | 24.4220 | -11.2515 | 92.0426 | -0.2442 | 0.0000 | 5/45 |

**delta (OOD kernel)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0170 | 8.8472 | -29.3873 | 90.4342 | -0.0885 | 0.4000 | 5/45 |
| N0 | 0.0214 | 150.5487 | 50.6479 | 477.1001 | -1.5055 | 0.2500 | 4/45 |
| NM | 0.0208 | 88.6306 | 16.6318 | 282.8039 | -0.8863 | 0.2000 | 5/45 |
| JM | 0.0180 | -18.2066 | -44.4544 | 28.3345 | 0.1821 | 0.3333 | 6/45 |
| JMs | 0.0171 | 3.5973 | -31.3554 | 72.2660 | -0.0360 | 0.4000 | 5/45 |

**delta (uniform)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0199 | 27.0771 | -12.3762 | 106.9904 | -0.2708 | 0.0000 | 5/45 |
| N0 | 0.0251 | 193.6994 | 74.7401 | 594.0364 | -1.9370 | 0.0000 | 4/45 |
| NM | 0.0251 | 126.9573 | 46.9693 | 355.4844 | -1.2696 | 0.0000 | 5/45 |
| JM | 0.0205 | -6.8527 | -31.2765 | 30.6522 | 0.0685 | 0.0000 | 6/45 |
| JMs | 0.0198 | 19.6442 | -16.2466 | 86.0517 | -0.1964 | 0.0000 | 5/45 |

**joint (direct)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0156 | 0.0513 | 0.0000 | 0.1951 | -0.0005 | 1.0000 | 5/45 |
| N0 | 0.0086 | 0.0937 | 0.0000 | 0.4222 | -0.0009 | 1.0000 | 4/45 |
| NM | 0.0111 | 0.0725 | 0.0000 | 0.2982 | -0.0007 | 1.0000 | 5/45 |
| JM | 0.0220 | -0.0365 | -0.1322 | 0.0000 | 0.0004 | 1.0000 | 6/45 |
| JMs | 0.0165 | 0.0485 | 0.0000 | 0.1820 | -0.0005 | 1.0000 | 5/45 |

**joint (calibrated)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0139 | -10.9833 | -18.0409 | -3.7837 | 0.1098 | 1.0000 | 5/45 |
| N0 | 0.0097 | 13.8744 | 1.5221 | 36.9376 | -0.1387 | 1.0000 | 4/45 |
| NM | 0.0120 | 8.6794 | -1.0744 | 22.5164 | -0.0868 | 1.0000 | 5/45 |
| JM | 0.0199 | -9.3318 | -14.3791 | -4.1978 | 0.0933 | 1.0000 | 6/45 |
| JMs | 0.0148 | -10.3844 | -16.9873 | -3.4845 | 0.1038 | 1.0000 | 5/45 |

**bidir profile-wall (cal.)**
| baseline | log MAE | Δ% vs isolated-speed | CI lo | CI hi | pooled skill | FB@1.05 | harmful pairs |
|---|---|---|---|---|---|---|---|
| J | 0.0143 | -8.3091 | -15.2613 | -0.6257 | 0.0831 | 1.0000 | 5/45 |
| N0 | 0.0103 | 20.4617 | 6.4685 | 48.1411 | -0.2046 | 1.0000 | 4/45 |
| NM | 0.0124 | 12.2016 | 0.5269 | 27.8769 | -0.1220 | 1.0000 | 5/45 |
| JM | 0.0201 | -8.5531 | -13.8019 | -3.5503 | 0.0855 | 1.0000 | 6/45 |
| JMs | 0.0152 | -8.0623 | -14.4294 | -0.7947 | 0.0806 | 1.0000 | 5/45 |
