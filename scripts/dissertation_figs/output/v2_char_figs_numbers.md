# v2 characterization figures (v2/v2_char_figs.py)

## c_ecdf_unclipped.pdf (6.0 x 2.5 in)
ECDFs of unclipped JMs ratios; solid grey line = 1, dotted = 1.05; log x.
- MPI time | App-App, non-self (90): median 1.065, <1 17.8%, >=1.05 55.6%, min 0.831, max 2.209
- MPI time | App-Inh, full grid (2,870): median 1.207, <1 12.3%, >=1.05 71.4%, min 0.687, max 8.510
- MPI time | App-Inh, panel (90): median 1.087, <1 15.6%, >=1.05 62.2%, min 0.813, max 2.700
- app time | App-App, non-self (90): median 1.004, <1 33.3%, >=1.05 6.7%, min 0.971, max 1.192
- app time | App-Inh, full grid (2,870): median 1.023, <1 17.7%, >=1.05 32.0%, min 0.942, max 1.509
- app time | App-Inh, panel (90): median 1.009, <1 27.8%, >=1.05 15.6%, min 0.965, max 1.232

## c_heatmaps_unclipped.pdf (6.0 x 8.6 in)
Rows = victim, columns = co-runner; rep-averaged JMs ratio, RdBu_r centred at 1.
- MPI time: colour limits ratio [0.4526, 2.2094] (symmetric in log); max 2.209, min 0.831; diagonal = self-pairs (pooled copies, bold)
- app time: colour limits ratio [0.7752, 1.2900] (symmetric in log); max 1.290, min 0.971; diagonal = self-pairs (pooled copies, bold)

## c_socket_isolated.pdf (6.0 x 2.9 in)
Paired pinned isolated runs (n = 4 per app), t 95% CI; negative = socket 1 faster.
- MPI time TriC: -0.0020 [-0.0253, +0.0213]
- MPI time Quicksilver: -0.0357 [-0.0713, -0.0001] *
- MPI time miniVite: -0.0453 [-0.3376, +0.2470]
- MPI time MiniFE: -0.0807 [-0.3084, +0.1471]
- MPI time LAMMPS: -0.6314 [-1.4323, +0.1695]
- MPI time Laghos: -0.2207 [-0.6873, +0.2458]
- MPI time Kripke: -0.0265 [-0.1018, +0.0488]
- MPI time FIESTA: -0.0628 [-0.2149, +0.0893]
- MPI time Beatnik: -0.0107 [-0.0529, +0.0315]
- MPI time AMG2023: -0.1971 [-0.2571, -0.1370] *
- app time TriC: -0.0026 [-0.0129, +0.0076]
- app time Quicksilver: -0.0165 [-0.0301, -0.0030] *
- app time miniVite: -0.0061 [-0.0495, +0.0373]
- app time MiniFE: -0.0039 [-0.0141, +0.0064]
- app time LAMMPS: -0.0566 [-0.1317, +0.0185]
- app time Laghos: -0.0380 [-0.1213, +0.0452]
- app time Kripke: -0.0028 [-0.0090, +0.0035]
- app time FIESTA: -0.0065 [-0.0214, +0.0084]
- app time Beatnik: -0.0078 [-0.0218, +0.0062]
- app time AMG2023: -0.0071 [-0.0094, -0.0047] *

## c_socket_corun.pdf (6.0 x 3.6 in)
Paired within (victim, co-runner/config) socket-1 minus socket-0 log slowdown, 95% CI (from v2_char_B2_corun_socket.csv); solid = JMs baseline, hollow = June baseline.
- MPI time App-App JMs TriC: +0.0067 [-0.0266, +0.0401] (n=9)
- MPI time App-App JMs Quicksilver: -0.0095 [-0.0257, +0.0067] (n=8)
- MPI time App-App JMs miniVite: +0.0200 [-0.0514, +0.0914] (n=7)
- MPI time App-App JMs MiniFE: -0.0859 [-0.1894, +0.0176] (n=8)
- MPI time App-App JMs LAMMPS: -0.0681 [-0.3006, +0.1643] (n=9)
- MPI time App-App JMs Laghos: -0.0659 [-0.1148, -0.0169] (n=7)
- MPI time App-App JMs Kripke: -0.0007 [-0.0564, +0.0550] (n=8)
- MPI time App-App JMs FIESTA: -0.3409 [-0.5274, -0.1545] (n=7)
- MPI time App-App JMs Beatnik: +0.0073 [-0.0296, +0.0443] (n=6)
- MPI time App-App JMs AMG2023: +0.0060 [-0.1105, +0.1226] (n=7)
- MPI time App-App J TriC: +0.0067 [-0.0266, +0.0401] (n=9)
- MPI time App-App J Quicksilver: -0.0452 [-0.0614, -0.0290] (n=8)
- MPI time App-App J miniVite: +0.0200 [-0.0514, +0.0914] (n=7)
- MPI time App-App J MiniFE: -0.0859 [-0.1894, +0.0176] (n=8)
- MPI time App-App J LAMMPS: -0.0681 [-0.3006, +0.1643] (n=9)
- MPI time App-App J Laghos: -0.0659 [-0.1148, -0.0169] (n=7)
- MPI time App-App J Kripke: -0.0007 [-0.0564, +0.0550] (n=8)
- MPI time App-App J FIESTA: -0.3409 [-0.5274, -0.1545] (n=7)
- MPI time App-App J Beatnik: +0.0073 [-0.0296, +0.0443] (n=6)
- MPI time App-App J AMG2023: -0.1910 [-0.3076, -0.0745] (n=7)
- MPI time App-Inh JMs TriC: -0.0067 [-0.0165, +0.0030] (n=260)
- MPI time App-Inh JMs Quicksilver: -0.0209 [-0.0242, -0.0175] (n=245)
- MPI time App-Inh JMs miniVite: -0.0412 [-0.0654, -0.0170] (n=240)
- MPI time App-Inh JMs MiniFE: -0.2377 [-0.2808, -0.1946] (n=240)
- MPI time App-Inh JMs LAMMPS: -0.1147 [-0.1707, -0.0587] (n=247)
- MPI time App-Inh JMs Laghos: -0.1744 [-0.2004, -0.1484] (n=248)
- MPI time App-Inh JMs Kripke: -0.1571 [-0.1895, -0.1248] (n=249)
- MPI time App-Inh JMs FIESTA: -0.3122 [-0.3504, -0.2739] (n=258)
- MPI time App-Inh JMs Beatnik: +0.0066 [-0.0040, +0.0172] (n=255)
- MPI time App-Inh JMs AMG2023: -0.1212 [-0.1742, -0.0682] (n=258)
- MPI time App-Inh J TriC: -0.0067 [-0.0165, +0.0030] (n=260)
- MPI time App-Inh J Quicksilver: -0.0566 [-0.0599, -0.0532] (n=245)
- MPI time App-Inh J miniVite: -0.0412 [-0.0654, -0.0170] (n=240)
- MPI time App-Inh J MiniFE: -0.2377 [-0.2808, -0.1946] (n=240)
- MPI time App-Inh J LAMMPS: -0.1147 [-0.1707, -0.0587] (n=247)
- MPI time App-Inh J Laghos: -0.1744 [-0.2004, -0.1484] (n=248)
- MPI time App-Inh J Kripke: -0.1571 [-0.1895, -0.1248] (n=249)
- MPI time App-Inh J FIESTA: -0.3122 [-0.3504, -0.2739] (n=258)
- MPI time App-Inh J Beatnik: +0.0066 [-0.0040, +0.0172] (n=255)
- MPI time App-Inh J AMG2023: -0.3183 [-0.3713, -0.2653] (n=258)
- app time App-App JMs TriC: +0.0038 [-0.0149, +0.0225] (n=9)
- app time App-App JMs Quicksilver: -0.0047 [-0.0118, +0.0024] (n=8)
- app time App-App JMs miniVite: +0.0025 [-0.0092, +0.0141] (n=7)
- app time App-App JMs MiniFE: -0.0038 [-0.0086, +0.0010] (n=8)
- app time App-App JMs LAMMPS: -0.0055 [-0.0190, +0.0080] (n=9)
- app time App-App JMs Laghos: -0.0146 [-0.0262, -0.0031] (n=7)
- app time App-App JMs Kripke: -0.0011 [-0.0053, +0.0031] (n=8)
- app time App-App JMs FIESTA: -0.0163 [-0.0244, -0.0081] (n=7)
- app time App-App JMs Beatnik: -0.0003 [-0.0133, +0.0126] (n=6)
- app time App-App JMs AMG2023: -0.0008 [-0.0046, +0.0030] (n=7)
- app time App-App J TriC: +0.0038 [-0.0149, +0.0225] (n=9)
- app time App-App J Quicksilver: -0.0213 [-0.0284, -0.0142] (n=8)
- app time App-App J miniVite: +0.0025 [-0.0092, +0.0141] (n=7)
- app time App-App J MiniFE: -0.0038 [-0.0086, +0.0010] (n=8)
- app time App-App J LAMMPS: -0.0055 [-0.0190, +0.0080] (n=9)
- app time App-App J Laghos: -0.0146 [-0.0262, -0.0031] (n=7)
- app time App-App J Kripke: -0.0011 [-0.0053, +0.0031] (n=8)
- app time App-App J FIESTA: -0.0163 [-0.0244, -0.0081] (n=7)
- app time App-App J Beatnik: -0.0003 [-0.0133, +0.0126] (n=6)
- app time App-App J AMG2023: -0.0078 [-0.0116, -0.0040] (n=7)
- app time App-Inh JMs TriC: -0.0076 [-0.0136, -0.0017] (n=260)
- app time App-Inh JMs Quicksilver: -0.0170 [-0.0195, -0.0144] (n=245)
- app time App-Inh JMs miniVite: -0.0216 [-0.0285, -0.0148] (n=240)
- app time App-Inh JMs MiniFE: -0.0238 [-0.0281, -0.0194] (n=240)
- app time App-Inh JMs LAMMPS: -0.0101 [-0.0168, -0.0033] (n=247)
- app time App-Inh JMs Laghos: -0.0493 [-0.0567, -0.0419] (n=248)
- app time App-Inh JMs Kripke: -0.0205 [-0.0249, -0.0161] (n=249)
- app time App-Inh JMs FIESTA: -0.0365 [-0.0449, -0.0281] (n=258)
- app time App-Inh JMs Beatnik: -0.0024 [-0.0067, +0.0020] (n=255)
- app time App-Inh JMs AMG2023: -0.0195 [-0.0254, -0.0137] (n=258)
- app time App-Inh J TriC: -0.0076 [-0.0136, -0.0017] (n=260)
- app time App-Inh J Quicksilver: -0.0335 [-0.0361, -0.0310] (n=245)
- app time App-Inh J miniVite: -0.0216 [-0.0285, -0.0148] (n=240)
- app time App-Inh J MiniFE: -0.0238 [-0.0281, -0.0194] (n=240)
- app time App-Inh J LAMMPS: -0.0101 [-0.0168, -0.0033] (n=247)
- app time App-Inh J Laghos: -0.0493 [-0.0567, -0.0419] (n=248)
- app time App-Inh J Kripke: -0.0205 [-0.0249, -0.0161] (n=249)
- app time App-Inh J FIESTA: -0.0365 [-0.0449, -0.0281] (n=258)
- app time App-Inh J Beatnik: -0.0024 [-0.0067, +0.0020] (n=255)
- app time App-Inh J AMG2023: -0.0266 [-0.0325, -0.0207] (n=258)

## c_coverage.pdf (6.0 x 3.6 in)
PCA of the 4 standardized selection features (comm. fraction, log(1+msg rate), log(1+byte rate), log(1+mean msg size)) on apps + 287 configs; PC1 0.593, PC2 0.336 of variance. Filled app marker = inside the 3-D grid hull (5/10); Quicksilver nearest-config distance 1.348 (selection metric).

## c_panel_vs_appapp.pdf (6.0 x 6.0 in)
Per-app means (JMs ln ratio); left = victim side, right = aggressor side (BD 'caused' = inhibitor time-per-work slowdown). rho/p from v2_char_numbers.md B3(c).
- MPI time victim: rho +0.6485 (recomputed +0.6485), p_perm 0.0485
- MPI time aggr: rho +0.6848 (recomputed +0.6848), p_perm 0.0344
- app time victim: rho +0.8788 (recomputed +0.8788), p_perm 0.0017
- app time aggr: rho +0.7333 (recomputed +0.7333), p_perm 0.0201

## p_panel_objective.pdf (3.0 x 2.5 in)
Objective = 0.5 mean + 0.5 max of nearest squared distance; log y.
- k=1: objective 2.2229, gain nan, random median 7.177
- k=2: objective 1.3635, gain 0.8595, random median 4.549
- k=3: objective 1.2102, gain 0.1533, random median 3.182
- k=4: objective 1.0993, gain 0.1109, random median 2.920
- k=5: objective 1.0630, gain 0.0363, random median 2.878
- k=6: objective 1.0396, gain 0.0234, random median 2.742
- k=7: objective 1.0303, gain 0.0093, random median 2.282
- k=8: objective 1.0269, gain 0.0034, random median 2.119
- k=9: objective 1.0239, gain 0.0030, random median 2.087
- lower bound 1.0239

## p_panel_grid.pdf (6.0 x 3.7 in)
Selected configs on (message size, wait); wait 0 drawn at the '0' tick (symlog); coincident panel configs (40 kB, 1 s: s = 0.1 and 0.7) offset horizontally. Annotation = apps whose nearest panel config it is (selection-metric distance).
- 8_20000_100000_0.1: AMG2023 (0.15)
- 8_400000_1000000_0.5: Beatnik (0.15)
- 8_2500000_200000_0.002237136: FIESTA (0.47)
- 8_2500000_600000_0.002237136: Kripke (0.18), LAMMPS (0.11)
- 8_500_5000_0.05: Laghos (0.22)
- 8_40000_1000000_0.1: MiniFE (0.11)
- 8_40000_1000000_0.7: miniVite (0.29)
- 8_200_200000_0.002237136: Quicksilver (1.35)
- 8_2000000_1000000_0.3: TriC (0.16)

