# Socket batch 2: socket-aware vs socket-blind pipelines

## Part A: facts and proposal (2026-11-08)

Generators: `socket_aware_facts.py` (A1, A2 → `socket_aware_facts.md`, `socket_corun_assign.csv`); S2 in
`analysis_socket_pinned.py` (→ `socket_numbers.md`). Resolved-target counts below come from processed_data joined to
`socket_corun_assign.csv`. The blind rebuild of App-Inh MPI ratios matches the canonical BD snapshot (2870/2870, max |diff| 1.3e-15).

### A1. Complement
Every one of the 56 mpibind task lines per job, on the one node mpibind logs per job:
- App-App: 180/180 runs complementary (s_V, s_X) ∈ {(0,1) 90, (1,0) 90}.
- App-Inh: 11,480/11,480 runs complementary ({(0,1) 6105, (1,0) 5375}).
- No job spans both sockets. All co-run ranks are on SMT thread 1 (PU ≥ 112).

So one bit s_V describes the assignment. Two App-Inh run dirs (Laghos × (500000, 20000, 1-peer) reps 0/1) exist in run_data
but not in processed_data; that was already the case for the snapshot.

### A2. Replicates per socket cell (n0 = reps with victim on sock0, n1 = sock1)
| Set | Cells | (n0,n1) distribution | One-sided cells | Resolved cells | 1-rep cells |
|---|---:|---|---:|---:|---:|
| App-App directed non-self | 90 | (0,4) 7, (1,3) 21, (2,2) 34, (3,1) 21, (4,0) 7 | 14 (7 all-sock0, 7 all-sock1) | 166 | 42 |
| App-Inh panel (10 × 9) | 90 | (0,4) 3, (1,3) 14, (2,2) 32, (3,1) 33, (4,0) 8 | 11 (8 all-sock0, 3 all-sock1) | 169 | 47 |
| App-Inh full grid | 2870 | (0,4) 133, (1,3) 639, (2,2) 1065, (3,1) 796, (4,0) 237 | 370 | 5370 | 1435 |

One-sided App-App outcomes: see `socket_aware_facts.md`. The panel is socket-imbalanced for some victims (share of reps on sock1:
Laghos 0.22, Kripke 0.28, AMG2023 0.33, Quicksilver 0.36).

**Floors with resolved targets** (June socket-0 baseline; clipped):
| Set | Target | Blind floor | Resolved floor | Resolved floor, s_V=0 / s_V=1 |
|---|---|---:|---:|---|
| App-Inh panel | MPI | 16/90 (18%) | 46/169 (27%) | 11/87 / 35/82 |
| App-Inh panel | total | 27/90 (30%) | 67/169 (40%) | 20/87 / 47/82 |
| App-Inh full grid | MPI | 443/2870 | 1133/5370 | 179/2737 / 954/2633 |
| App-Inh full grid | total | 611/2870 | 1471/5370 | 293/2737 / 1178/2633 |
| App-App | MPI | 18/90 | 52/166 | 15/83 / 37/83 (median r 1.084 / 1.008) |
| App-App | total | 36/90 | 72/166 | 22/83 / 50/83 |

### A3. S2
See `socket_numbers.md` §S2/§S3 and findings §7b. 72/72 valid. Inhibitor intensity (messages in the fixed 360 s window) is
socket-invariant: pooled d −0.0011 [−0.0023, +0.0001]. T_MPI d is mostly ~0; pooled +0.048 [−0.019, +0.152] is driven by
2 single-run spikes. CI excludes 0 only for (4e4, 1e6, 0.1) −0.057 and (2.5e6, 6e5, 1-peer) −0.025. T_app is fixed by
design. Drift vs June: messages within ~1% except (500, 5e3, 0.05) +8.4%, whose June reps were noisy (CV 58%).
App-Inh JMs is unchanged: inhibitor d does not enter the app's slowdown.

### Flag: socket-offset baseline
Every app's isolated d̄ (log T_MPI, sock1 − sock0) is negative: AMG −0.197, Beatnik −0.011, FIESTA −0.063, Kripke −0.027,
Laghos −0.221, LAMMPS −0.631, MiniFE −0.081, miniVite −0.045, Quicksilver −0.036, TriC −0.002. So max(1, exp(d̄)) = 1 for
every s_V=1 cell, and the requested "socket-offset baseline" is identical to the isolated-speed baseline.
Proposed replacement: **blind-absolute + socket offset**, i.e. the blind absolute model's latent R̂(V,X) + s_V·d̄_V before
the floor. Use d̄ only where its CI excludes 0 (AMG2023, Quicksilver; the JMs rule), and also report all-app d̄ as a
sensitivity. It answers "is a fixed per-app offset enough?"

### A4. Reproduction and compute (8 cores, 31 GB, CPU only)
- **BD-TT:** already reproduced locally (2026-10-17 `outputs/full_comparison` equals artifact bidirTT). CV took 1600 fits in
  2405 s, about 42 min end to end.
- **BD-MPI:** reproduction running now (original code, `~/hpcResearch/experiment_outputs/socket_batch2/repro_bd_mpi`).
  1200 CV fits at about 4.5 s/fit with 2 threads, so about 90 min. Will report max |Δ log MAE| vs artifact bidir.
- **1W MPI (control) / 1W TT:** the runner tune is 3 sequential CV stages for the low-rank delta (6 + 9 + 3 = 18 candidates)
  plus the absolute and generic models. That is about 4,000 fits, or 1.5–5 h per run depending on threads. Evidence of
  determinism: the local `filtered_fit_mpi_time` absolute = 0.094226 = artifact. Will reproduce via the runner
  (`--random-split-only`) with jobs.csv etc. from the canonical MPI / app_time snapshots.
- **Cost estimate:**
  - (i) Full data, 4 pipelines × {blind, aware} = 8 runs: BD about 1–1.5 h each, 1W about 2–5 h each. Running about 3
    concurrently gives roughly 6–8 h wall.
  - (ii) Restricted tiers: 70 tasks per run with frozen recipes (no CV), so final fits only. My rough estimate is 1–2 min
    per task, about 560 tasks for 8 runs, so about 3–5 h wall. I'll measure on the first tasks before committing.

### A5. Proposed minimal implementation
**Code hygiene:** copy the 4 pipeline dirs to `*_socketAware/` and add one config flag `socket_feature`. Run **both**
blind and aware from the copy; blind must reproduce the original bit-for-bit (checked first). Originals are untouched.

**Data:**
- A prepare script writes the resolved snapshot: one row per (app, config, s_V), ratio of aggregate means over the reps
  with that s_V, June 4-rep isolated baseline, clipped. `replicate_id = s_V`, plus a column `s_V`.
- Aux rate-change targets are recomputed the same way per resolved row (the inhibitor endpoint has socket 1 − s_V).
- `pair_resolved.csv` is built the same way.
- Blind = the existing snapshot (the rebuild is identical).

**Where s enters (all families):** each endpoint's input vector gets **its own** socket bit appended, unscaled (binary):
the victim gets s_V and the aggressor gets s_X = 1 − s_V. input_dim goes from 4 to 5. The encoder is shared, so s reaches
b, h, u, v and the aux head; no model class changes. Predictions are R(V,X | s_V) for both s_V ∈ {0,1}.

| Family | Proposal |
|---|---|
| Absolute | Rows = resolved observations. Weight 1/n_resolved(cell) so each (victim, config) carries the blind weight. |
| Delta / generic (legacy sampler) | Sampler draws config, then one resolved row (as the joint sampler already does); anchor ≠ query config, unchanged. Inference: anchors = **all** uncensored resolved panel rows of the victim, each with its own socket: ρ_J = y_J − R̂(V,J \| s_J), ŝ(V,X \| s) = R̂(V,X \| s) + agg(ρ). **Not** socket-specific anchor subsets: they would halve the anchors, 11/90 panel cells are one-sided, and 35/82 sock1 panel cells are at the floor, so 1W `no_uncensored_anchor` failures would rise. |
| Joint / bidirectional | Sampler is already config-then-replicate, so no reweighting. Aux targets per resolved row. |
| Floor-aware calibration | **One shared offset per victim**, fitted on all resolved anchors with the existing equal-weight-per-config w (split across the config's socket rows). Per-socket offsets would rest on about 4.5 anchors each, and the sock1 side is 43% floor. |
| Context fingerprints | One token per (config, socket) with the socket bit appended (token width +1). Support/query disjointness stays by config. |
| Panel selection, CV blocks, folds, seeds, HP grid | Unchanged (they depend on profiles and configs only), so blind and aware get identical panels and folds. |
| Inhibitor profiles | **No d_inh adjustment.** Intensity is socket-invariant (S2); T_MPI d is ~0 except 2/9 configs at −2.5% and −5.7%; d exists only for the 9 panel configs, while restricted tiers select other inhibitors, so an adjustment would be inconsistent across tiers. The socket bit absorbs the residual. App profiles (as aggressors) stay June, per decision 1. |

**Scoring (B1):**
- Unit = resolved cell (166 App-App). Clusters = the 45 unordered pairs (both directions, both sockets), with a shared
  draw matrix (1,000, seed 923).
- Robustness rows: cells with ≥ 2 reps only (124), and a cell-weight = n_reps variant.
- False-benign at pair level, per socket assignment: FB@1.05 = fraction of HARMFUL outcomes (true slowdown ≥ 1.05;
  count in the `harmful` column) that the method predicts benign (< 1.05). FB = 1.0 means every harmful outcome is
  missed (e.g. a model that predicts the floor everywhere), not that pairs are classified correctly.

**Censoring:** resolved targets have more floor rows (panel MPI 16 → 46 of 169; sock1 43%). 1-rep cells (47 panel, 42
App-App) carry the replicate CV (isolated T_MPI median 4.3%, Laghos 27%) into the targets unaveraged. The censored losses
handle the floors as they are; no change proposed.

### A6. BD runs: verification, selected HPs, near-init diagnosis (2026-11-08)
- Blind copies reproduce the originals exactly: BD-MPI copy vs `repro_bd_mpi` (6 shared variants, 1760 rows) max |Δ| = 0 in every
  column, same epochs and CV MAE; BD-TT copy vs `outputs/full_comparison` (8 variants, 2200 rows) identical. B2 blind = artifact.
- Selected HPs: every BD variant picks config 0 (hidden 8, emb 4, rank 2, lr 1e-3, wd 1e-4) in both modes; only final epochs
  differ (4 batches/epoch). MPI blind→aware: delta 13→10, generic 14→16, absolute 2→1, joint 1→4, bprof 2→4, bctx 2→3,
  bwall 2→4, bfwd 1→4. TT: delta 3→7, generic 4→8, absolute 6→3, joint 5→1, bprof 3→1, bctx 1→1, bwall 4→1, bfwd 2→7.
- Several aware variants predict exactly the floor on every App-App cell (Δ = 0.0000): MPI absolute, context, context-cal;
  TT absolute, joint, bprof, bctx (+cal). Their latents sit below 0 (e.g. TT absolute mean −0.10 vs blind −0.015).
- Diagnosis (scratchpad ctxdiag, subagent): not a socket-code bug. Aware App-App inference reproduces from checkpoints to 4e-8;
  restricting tokens to one socket or clamping the token count moves context latents ≤ 0.04. Models trained 1–7 epochs × 4
  batches are near init: per-seed mean latents at epoch 1 range −0.9 to +0.4, and the ensemble level drifts monotonically with
  epochs (aware context −0.26 @1 → −0.08 @14). Blind-vs-aware differences for absolute/joint/profile/context therefore mix the
  socket feature with the CV-chosen epoch count. The TT forward-only blow-up (+516%) tracks 7 vs 2 epochs plus an aware offset
  that persists at equal epochs.
- Caveat for the aware `directional_predictions.csv`: its latent column is the floored socket average; use the socket CSV.

- A4, 1W-TT: `repro_1w_tt` (original code, blind data) reproduces artifact totalTime random_split log MAE to 6 decimals for
  every method (absolute 0.016197, generic 0.020196, delta uniform/median/OOD-kernel 0.022819/0.024279/0.020604, constant
  0.015632); global selection matches (low-rank emb 8/hid 16/rank 4/lr 3e-3/wd 1e-3, epochs abs 6, generic 4, low-rank 4).
- A4, 1W-MPI: `repro_1w_mpi` (original `delta_response_filteredInhibs` code on totalTime `data/mpi_time`) reproduces artifact
  baseSnap random_split log MAE to 6 decimals (absolute 0.094226, generic 0.147243, delta uniform/median/OOD-kernel
  0.152236/0.123502/0.130162, constant 0.103012); selection identical (low-rank emb 4/hid 8/rank 4/lr 3e-3/wd 0; epochs abs 2,
  generic 14, low-rank 5). CV MAE from global_selection model_selection agrees with artifact cv_log_mae within 2e-6.
  All four reproductions (A4) are now exact.
- A4, 1W-TT blind copy: `blind_1w_tt` == `repro_1w_tt` in every column of 1100 prediction rows, global selection, task IDs.
- B4 cost (measured 2026-11-08): 70 restricted tasks per run (10 splits x sizes 2-8). 1W full-data final evaluation task =
  8-15 min (aware ~8, blind 13-15, 2 threads under load); BD final fits ~1-2 min. 8 runs x 70: 1W ~37-70 CPU-h, BD ~9-14
  CPU-h -> ~6-11 h wall on 8 cores. Exceeds the 5 h gate; sent options (a) full, (b) BD only ~1.5-2 h, (c) BD + 3/10 1W splits,
  (d) skip. Runner resume rejects copied tune tasks (status.json output_dir is absolute); rewrite paths to reuse tuning.
- Ops note: `socket_batch2_run1w.sh` was edited while early 1W runs were executing; bash re-read it mid-run and the
  evaluate step failed ("ss-interval: command not found") after tuning succeeded. Re-running the script resumes (tune
  skipped, evaluate runs). Applies to repro_1w_mpi, aware_1w_mpi, aware_1w_tt.

## Part B: results

Runs: 1W-MPI: blind=yes, aware=yes; 1W-TT: blind=yes, aware=yes; BD-MPI: blind=yes, aware=yes; BD-TT: blind=yes, aware=yes
- 1W-MPI aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 2.78e-16
- 1W-TT aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 5.13e-16
- BD-MPI aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 5.03e-16
- BD-TT aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 4.79e-16

### B1. Socket-resolved App-App outcomes (166 cells)

#### One-way, MPI time: all
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.1102 | – | – | – | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Absolute response [blind] | 166 | 0.1035 | -6.1002 | -11.1364 | -1.2157 | -0.0843 | 0.0251 | 0.1392 | 0.1072 | 0.7170 | 53 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.1021 | -7.3847 | -12.0264 | -2.8415 | -0.0870 | 0.0191 | 0.1399 | 0.2006 | 0.7358 | 53 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.1011 | -8.2448 | -11.5404 | -5.1183 | -0.0949 | 0.0078 | 0.1437 | 0.3658 | 0.8679 | 53 |
| Delta, uniform agg. [blind] | 166 | 0.1546 | 40.2564 | -3.5429 | 82.2636 | 0.1249 | 0.2426 | 0.1144 | 0.2184 | 0.0000 | 53 |
| Delta, median agg. [blind] | 166 | 0.1285 | 16.6310 | -23.3756 | 52.7619 | 0.0890 | 0.2094 | 0.0916 | 0.2317 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.1332 | 20.8624 | -19.9005 | 60.0365 | 0.1133 | 0.2013 | 0.1021 | 0.2937 | 0.0000 | 53 |
| Generic potential [blind] | 166 | 0.1504 | 36.4447 | -5.3995 | 76.6160 | 0.1137 | 0.2360 | 0.1113 | 0.2148 | 0.0000 | 53 |
| Isolated-speed baseline [aware] | 166 | 0.1102 | – | – | – | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Absolute response [aware] | 166 | 0.1102 | 0.0000 | 0.0000 | 0.0000 | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Delta, uniform agg. [aware] | 166 | 0.1679 | 52.3977 | 9.2601 | 94.4947 | 0.1477 | 0.2710 | 0.1210 | 0.2065 | 0.0000 | 53 |
| Delta, median agg. [aware] | 166 | 0.1282 | 16.3296 | -25.4377 | 54.1720 | 0.1001 | 0.2147 | 0.0887 | 0.2804 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.1457 | 32.2107 | -9.5431 | 72.8148 | 0.1286 | 0.2167 | 0.1133 | 0.2812 | 0.0000 | 53 |
| Generic potential [aware] | 166 | 0.1719 | 55.9938 | 8.4058 | 101.0195 | 0.1475 | 0.2852 | 0.1202 | 0.1620 | 0.0000 | 53 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0067 | 0.0013 | 0.0123 | 1.0650 |
| delta_uniform | 0.0134 | 0.0032 | 0.0232 | 1.0866 |
| delta_median | -0.0003 | -0.0089 | 0.0073 | 0.9974 |
| delta_ood_kernel | 0.0125 | 0.0027 | 0.0226 | 1.0939 |
| generic | 0.0215 | 0.0121 | 0.0325 | 1.1433 |
| aware absolute vs offset_jms | 0.0081 | 0.0031 | 0.0133 | 1.0797 |
| aware absolute vs offset_all | 0.0091 | 0.0056 | 0.0127 | 1.0899 |

#### One-way, MPI time: s_V=0
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.1301 | – | – | – | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Absolute response [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.1443 | 10.9794 | -24.2522 | 46.7576 | 0.1019 | 0.2791 | 0.1146 | 0.2750 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.1185 | -8.8777 | -40.4823 | 23.5949 | 0.0657 | 0.2440 | 0.0908 | 0.2939 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.1203 | -7.4600 | -41.9141 | 25.5892 | 0.0905 | 0.2243 | 0.0974 | 0.3480 | – | 0 |
| Generic potential [blind] | 83 | 0.1412 | 8.6045 | -25.3306 | 43.3598 | 0.0913 | 0.2751 | 0.1117 | 0.2695 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.1301 | – | – | – | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Absolute response [aware] | 83 | 0.1301 | 0.0000 | 0.0000 | 0.0000 | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.1648 | 26.7338 | -10.4222 | 63.4962 | 0.1464 | 0.3290 | 0.1286 | 0.3056 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.1294 | -0.5018 | -35.0308 | 34.2544 | 0.0994 | 0.2743 | 0.0974 | 0.3306 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.1441 | 10.8149 | -24.8617 | 45.7971 | 0.1265 | 0.2633 | 0.1178 | 0.3689 | – | 0 |
| Generic potential [aware] | 83 | 0.1626 | 24.9918 | -13.9096 | 63.9973 | 0.1382 | 0.3379 | 0.1239 | 0.2699 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0147 | 0.0091 | 0.0199 | 1.1273 |
| delta_uniform | 0.0205 | 0.0069 | 0.0334 | 1.1420 |
| delta_median | 0.0109 | -0.0002 | 0.0204 | 1.0919 |
| delta_ood_kernel | 0.0238 | 0.0137 | 0.0346 | 1.1975 |
| generic | 0.0213 | 0.0093 | 0.0348 | 1.1509 |
| aware absolute vs offset_jms | 0.0147 | 0.0091 | 0.0199 | 1.1273 |
| aware absolute vs offset_all | 0.0147 | 0.0091 | 0.0199 | 1.1273 |

#### One-way, MPI time: s_V=1
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0903 | – | – | – | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0916 | 1.3706 | -6.2761 | 9.1853 | -0.0645 | 0.0255 | 0.1447 | 0.0316 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0888 | -1.7629 | -8.4255 | 4.5467 | -0.0698 | 0.0171 | 0.1464 | 0.1511 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0869 | -3.8611 | -6.9436 | -1.3745 | -0.0857 | 0.0012 | 0.1558 | 0.3363 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.1648 | 82.3991 | 22.3288 | 136.4792 | 0.1478 | 0.2278 | 0.1141 | 0.2020 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.1385 | 53.3492 | -2.1139 | 102.3757 | 0.1122 | 0.1954 | 0.0928 | 0.2107 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.1460 | 61.6309 | 7.5678 | 114.0849 | 0.1362 | 0.1919 | 0.1091 | 0.2826 | – | 0 |
| Generic potential [blind] | 83 | 0.1595 | 76.5190 | 19.2401 | 130.3372 | 0.1362 | 0.2201 | 0.1107 | 0.2042 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0903 | – | – | – | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0903 | 0.0000 | 0.0000 | 0.0000 | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.1711 | 89.3394 | 31.6318 | 146.4897 | 0.1490 | 0.2474 | 0.1096 | 0.0566 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.1270 | 40.5574 | -14.9272 | 91.4587 | 0.1008 | 0.1905 | 0.0759 | 0.1546 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.1473 | 63.0086 | 8.0729 | 118.7777 | 0.1306 | 0.1978 | 0.1067 | 0.1659 | – | 0 |
| Generic potential [aware] | 83 | 0.1813 | 100.6192 | 37.2209 | 160.7288 | 0.1569 | 0.2639 | 0.1148 | 0.0426 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0012 | -0.0083 | 0.0057 | 0.9865 |
| delta_uniform | 0.0063 | -0.0045 | 0.0167 | 1.0380 |
| delta_median | -0.0116 | -0.0206 | -0.0031 | 0.9166 |
| delta_ood_kernel | 0.0012 | -0.0108 | 0.0133 | 1.0085 |
| generic | 0.0218 | 0.0114 | 0.0335 | 1.1365 |
| aware absolute vs offset_jms | 0.0016 | -0.0041 | 0.0076 | 1.0179 |
| aware absolute vs offset_all | 0.0035 | 0.0012 | 0.0063 | 1.0402 |

#### One-way, MPI time: >=2 reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 124 | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Absolute response [blind] | 124 | 0.0970 | -5.8787 | -11.1176 | -1.1371 | -0.0776 | 0.0257 | 0.1285 | 0.0458 | 0.7568 | 37 |
| Blind absolute + fixed socket offset (JMs) [blind] | 124 | 0.0956 | -7.2184 | -11.8919 | -2.9076 | -0.0807 | 0.0191 | 0.1294 | 0.1363 | 0.7838 | 37 |
| Blind absolute + fixed socket offset (all apps) [blind] | 124 | 0.0927 | -9.9898 | -13.7775 | -6.7308 | -0.0878 | 0.0061 | 0.1310 | 0.4310 | 0.8649 | 37 |
| Delta, uniform agg. [blind] | 124 | 0.1567 | 52.1354 | 4.0629 | 92.9297 | 0.1321 | 0.2550 | 0.1133 | 0.1995 | 0.0000 | 37 |
| Delta, median agg. [blind] | 124 | 0.1270 | 23.2614 | -20.9467 | 59.6252 | 0.0939 | 0.2159 | 0.0877 | 0.2179 | 0.0000 | 37 |
| Delta, OOD-kernel agg. [blind] | 124 | 0.1325 | 28.6272 | -15.8770 | 69.8326 | 0.1182 | 0.2158 | 0.0957 | 0.2808 | 0.0000 | 37 |
| Generic potential [blind] | 124 | 0.1499 | 45.5581 | 1.8066 | 85.7209 | 0.1201 | 0.2457 | 0.1076 | 0.1993 | 0.0000 | 37 |
| Isolated-speed baseline [aware] | 124 | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Absolute response [aware] | 124 | 0.1030 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Delta, uniform agg. [aware] | 124 | 0.1724 | 67.4080 | 21.2359 | 107.5955 | 0.1553 | 0.2857 | 0.1224 | 0.1798 | 0.0000 | 37 |
| Delta, median agg. [aware] | 124 | 0.1295 | 25.7448 | -18.0254 | 62.5336 | 0.1053 | 0.2219 | 0.0887 | 0.2514 | 0.0000 | 37 |
| Delta, OOD-kernel agg. [aware] | 124 | 0.1471 | 42.7831 | -1.7515 | 86.0186 | 0.1335 | 0.2300 | 0.1104 | 0.2484 | 0.0000 | 37 |
| Generic potential [aware] | 124 | 0.1776 | 72.4020 | 24.8225 | 116.0496 | 0.1561 | 0.2994 | 0.1237 | 0.1340 | 0.0000 | 37 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0061 | 0.0012 | 0.0115 | 1.0625 |
| delta_uniform | 0.0157 | 0.0051 | 0.0270 | 1.1004 |
| delta_median | 0.0026 | -0.0063 | 0.0110 | 1.0201 |
| delta_ood_kernel | 0.0146 | 0.0030 | 0.0272 | 1.1101 |
| generic | 0.0277 | 0.0186 | 0.0378 | 1.1844 |
| aware absolute vs offset_jms | 0.0074 | 0.0030 | 0.0122 | 1.0778 |
| aware absolute vs offset_all | 0.0103 | 0.0069 | 0.0142 | 1.1110 |

#### One-way, MPI time: weight=n_reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.1113 | – | – | – | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Absolute response [blind] | 166 | 0.1039 | -6.6397 | -11.3115 | -2.3356 | -0.0859 | 0.0246 | 0.1370 | 0.1072 | 0.7170 | 53 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.1029 | -7.5853 | -12.0026 | -3.5679 | -0.0885 | 0.0189 | 0.1379 | 0.2006 | 0.7358 | 53 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.1011 | -9.1427 | -12.3232 | -6.2493 | -0.0962 | 0.0066 | 0.1406 | 0.3658 | 0.8679 | 53 |
| Delta, uniform agg. [blind] | 166 | 0.1542 | 38.5288 | -5.5921 | 78.1781 | 0.1259 | 0.2483 | 0.1149 | 0.2184 | 0.0000 | 53 |
| Delta, median agg. [blind] | 166 | 0.1271 | 14.2360 | -26.8925 | 49.5068 | 0.0899 | 0.2124 | 0.0916 | 0.2317 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.1314 | 18.0696 | -23.2752 | 57.0782 | 0.1137 | 0.2061 | 0.1002 | 0.2937 | 0.0000 | 53 |
| Generic potential [blind] | 166 | 0.1494 | 34.2306 | -8.3673 | 73.6174 | 0.1149 | 0.2418 | 0.1108 | 0.2148 | 0.0000 | 53 |
| Isolated-speed baseline [aware] | 166 | 0.1113 | – | – | – | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Absolute response [aware] | 166 | 0.1113 | 0.0000 | 0.0000 | 0.0000 | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Delta, uniform agg. [aware] | 166 | 0.1700 | 52.7487 | 9.0961 | 95.1865 | 0.1484 | 0.2803 | 0.1240 | 0.2065 | 0.0000 | 53 |
| Delta, median agg. [aware] | 166 | 0.1293 | 16.1654 | -23.0116 | 52.8937 | 0.1007 | 0.2193 | 0.0917 | 0.2804 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.1468 | 31.9271 | -7.7578 | 73.0521 | 0.1298 | 0.2254 | 0.1140 | 0.2812 | 0.0000 | 53 |
| Generic potential [aware] | 166 | 0.1740 | 56.3086 | 7.4126 | 100.7842 | 0.1486 | 0.2954 | 0.1233 | 0.1620 | 0.0000 | 53 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0074 | 0.0026 | 0.0126 | 1.0650 |
| delta_uniform | 0.0158 | 0.0055 | 0.0266 | 1.0866 |
| delta_median | 0.0021 | -0.0064 | 0.0099 | 0.9974 |
| delta_ood_kernel | 0.0154 | 0.0054 | 0.0262 | 1.0939 |
| generic | 0.0246 | 0.0154 | 0.0348 | 1.1433 |
| aware absolute vs offset_jms | 0.0084 | 0.0040 | 0.0134 | 1.0797 |
| aware absolute vs offset_all | 0.0102 | 0.0070 | 0.0137 | 1.0899 |

#### One-way, total time: all
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.0157 | – | – | – | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Absolute response [blind] | 166 | 0.0161 | 2.0617 | -8.3069 | 12.3551 | -0.0126 | 0.0028 | 0.0262 | 0.0945 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.0159 | 0.8826 | -9.1479 | 11.6556 | -0.0127 | 0.0024 | 0.0262 | 0.1125 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.0157 | 0.0629 | -8.3702 | 8.9535 | -0.0132 | 0.0018 | 0.0264 | 0.1569 | 1.0000 | 11 |
| Delta, uniform agg. [blind] | 166 | 0.0231 | 46.7803 | -3.5614 | 89.8145 | 0.0192 | 0.0259 | 0.0209 | 0.4569 | 0.0909 | 11 |
| Delta, median agg. [blind] | 166 | 0.0244 | 54.8603 | 6.4817 | 95.3311 | 0.0204 | 0.0281 | 0.0215 | 0.4205 | 0.0000 | 11 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.0207 | 31.3845 | -19.8147 | 74.6220 | 0.0176 | 0.0240 | 0.0182 | 0.4907 | 0.2727 | 11 |
| Generic potential [blind] | 166 | 0.0206 | 30.8412 | -15.6109 | 69.3879 | 0.0153 | 0.0227 | 0.0190 | 0.5129 | 0.1818 | 11 |
| Isolated-speed baseline [aware] | 166 | 0.0157 | – | – | – | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Absolute response [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Delta, uniform agg. [aware] | 166 | 0.0231 | 46.5067 | -0.2803 | 84.7108 | 0.0171 | 0.0262 | 0.0206 | 0.4707 | 0.2727 | 11 |
| Delta, median agg. [aware] | 166 | 0.0149 | -5.5597 | -41.6588 | 24.9742 | 0.0067 | 0.0130 | 0.0163 | 0.6293 | 0.3636 | 11 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.0204 | 29.3670 | -30.0467 | 90.8661 | 0.0150 | 0.0246 | 0.0171 | 0.4096 | 0.4545 | 11 |
| Generic potential [aware] | 166 | 0.0243 | 54.7251 | 4.9969 | 94.8086 | 0.0195 | 0.0260 | 0.0231 | 0.5355 | 0.0909 | 11 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0003 | -0.0019 | 0.0013 | 0.9798 |
| delta_uniform | -0.0000 | -0.0037 | 0.0040 | 0.9981 |
| delta_median | -0.0095 | -0.0132 | -0.0055 | 0.6098 |
| delta_ood_kernel | -0.0003 | -0.0055 | 0.0064 | 0.9846 |
| generic | 0.0038 | 0.0019 | 0.0058 | 1.1825 |
| aware absolute vs offset_jms | -0.0001 | -0.0018 | 0.0014 | 0.9913 |
| aware absolute vs offset_all | -0.0000 | -0.0014 | 0.0013 | 0.9994 |

#### One-way, total time: s_V=0
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.0225 | 38.1366 | -7.2152 | 73.6795 | 0.0176 | 0.0262 | 0.0211 | 0.4461 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.0233 | 42.7537 | -1.1122 | 77.4707 | 0.0189 | 0.0276 | 0.0217 | 0.4252 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.0196 | 20.1625 | -27.8346 | 58.2617 | 0.0161 | 0.0243 | 0.0179 | 0.4802 | – | 0 |
| Generic potential [blind] | 83 | 0.0201 | 23.5066 | -15.9905 | 59.2025 | 0.0140 | 0.0252 | 0.0183 | 0.4499 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.0231 | 41.5301 | -1.2623 | 78.5350 | 0.0166 | 0.0311 | 0.0202 | 0.3231 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.0150 | -8.0055 | -40.9827 | 20.0619 | 0.0064 | 0.0158 | 0.0147 | 0.5554 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.0203 | 24.7998 | -31.7162 | 82.2263 | 0.0134 | 0.0288 | 0.0173 | 0.2784 | – | 0 |
| Generic potential [aware] | 83 | 0.0281 | 72.7566 | 25.2880 | 112.0493 | 0.0239 | 0.0371 | 0.0249 | 0.3872 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0001 | -0.0015 | 0.0016 | 1.0062 |
| delta_uniform | 0.0006 | -0.0034 | 0.0044 | 1.0246 |
| delta_median | -0.0083 | -0.0118 | -0.0044 | 0.6444 |
| delta_ood_kernel | 0.0008 | -0.0048 | 0.0075 | 1.0386 |
| generic | 0.0080 | 0.0056 | 0.0105 | 1.3988 |
| aware absolute vs offset_jms | 0.0001 | -0.0015 | 0.0016 | 1.0062 |
| aware absolute vs offset_all | 0.0001 | -0.0015 | 0.0016 | 1.0062 |

#### One-way, total time: s_V=1
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0152 | – | – | – | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0159 | 4.9328 | -6.8264 | 16.6338 | -0.0120 | 0.0027 | 0.0360 | 0.0590 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0156 | 2.4893 | -8.6850 | 13.6796 | -0.0124 | 0.0022 | 0.0358 | 0.0856 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0153 | 0.7904 | -7.0836 | 8.8353 | -0.0134 | 0.0013 | 0.0365 | 0.1241 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.0237 | 56.0502 | -3.1953 | 109.0490 | 0.0208 | 0.0258 | 0.0205 | 0.5066 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.0255 | 67.8441 | 9.3260 | 119.2757 | 0.0219 | 0.0283 | 0.0212 | 0.4574 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.0218 | 43.4195 | -16.7221 | 95.4771 | 0.0192 | 0.0238 | 0.0187 | 0.5445 | – | 0 |
| Generic potential [blind] | 83 | 0.0211 | 38.7073 | -15.0922 | 84.2628 | 0.0166 | 0.0216 | 0.0202 | 0.6112 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0152 | – | – | – | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.0231 | 51.8439 | -0.5507 | 96.7580 | 0.0175 | 0.0241 | 0.0215 | 0.6098 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.0147 | -2.9367 | -43.1703 | 30.6055 | 0.0070 | 0.0118 | 0.0192 | 0.7231 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.0204 | 34.2652 | -33.3493 | 101.2341 | 0.0165 | 0.0227 | 0.0169 | 0.5264 | – | 0 |
| Generic potential [aware] | 83 | 0.0206 | 35.3873 | -18.4106 | 78.7771 | 0.0152 | 0.0211 | 0.0197 | 0.5988 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0007 | -0.0025 | 0.0010 | 0.9530 |
| delta_uniform | -0.0006 | -0.0045 | 0.0038 | 0.9730 |
| delta_median | -0.0107 | -0.0148 | -0.0063 | 0.5783 |
| delta_ood_kernel | -0.0014 | -0.0072 | 0.0060 | 0.9362 |
| generic | -0.0005 | -0.0023 | 0.0013 | 0.9761 |
| aware absolute vs offset_jms | -0.0004 | -0.0021 | 0.0013 | 0.9757 |
| aware absolute vs offset_all | -0.0001 | -0.0013 | 0.0011 | 0.9922 |

#### One-way, total time: >=2 reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 124 | 0.0150 | – | – | – | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Absolute response [blind] | 124 | 0.0155 | 3.0084 | -7.9343 | 13.7750 | -0.0122 | 0.0027 | 0.0247 | 0.0143 | 1.0000 | 7 |
| Blind absolute + fixed socket offset (JMs) [blind] | 124 | 0.0152 | 1.3564 | -9.0315 | 12.0922 | -0.0124 | 0.0022 | 0.0246 | 0.0379 | 1.0000 | 7 |
| Blind absolute + fixed socket offset (all apps) [blind] | 124 | 0.0149 | -0.6244 | -9.8098 | 8.7673 | -0.0128 | 0.0014 | 0.0247 | 0.1730 | 1.0000 | 7 |
| Delta, uniform agg. [blind] | 124 | 0.0228 | 51.5544 | -5.9714 | 99.3795 | 0.0188 | 0.0261 | 0.0204 | 0.4339 | 0.0000 | 7 |
| Delta, median agg. [blind] | 124 | 0.0240 | 59.7367 | 3.9659 | 104.8717 | 0.0201 | 0.0284 | 0.0209 | 0.3975 | 0.0000 | 7 |
| Delta, OOD-kernel agg. [blind] | 124 | 0.0206 | 37.2263 | -21.5668 | 87.9461 | 0.0177 | 0.0248 | 0.0176 | 0.4555 | 0.1429 | 7 |
| Generic potential [blind] | 124 | 0.0198 | 31.4097 | -14.6277 | 67.9079 | 0.0144 | 0.0208 | 0.0190 | 0.5086 | 0.1429 | 7 |
| Isolated-speed baseline [aware] | 124 | 0.0150 | – | – | – | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Absolute response [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Delta, uniform agg. [aware] | 124 | 0.0221 | 46.9657 | 0.9565 | 83.5152 | 0.0166 | 0.0246 | 0.0203 | 0.4865 | 0.1429 | 7 |
| Delta, median agg. [aware] | 124 | 0.0140 | -6.8833 | -43.6366 | 20.8589 | 0.0064 | 0.0116 | 0.0157 | 0.6498 | 0.2857 | 7 |
| Delta, OOD-kernel agg. [aware] | 124 | 0.0203 | 34.9921 | -37.2080 | 100.5528 | 0.0159 | 0.0247 | 0.0171 | 0.4049 | 0.2857 | 7 |
| Generic potential [aware] | 124 | 0.0234 | 55.3466 | 8.5456 | 92.3647 | 0.0185 | 0.0238 | 0.0230 | 0.5422 | 0.0000 | 7 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0005 | -0.0021 | 0.0012 | 0.9708 |
| delta_uniform | -0.0007 | -0.0048 | 0.0036 | 0.9697 |
| delta_median | -0.0100 | -0.0141 | -0.0057 | 0.5829 |
| delta_ood_kernel | -0.0003 | -0.0061 | 0.0062 | 0.9837 |
| generic | 0.0036 | 0.0015 | 0.0058 | 1.1822 |
| aware absolute vs offset_jms | -0.0002 | -0.0018 | 0.0014 | 0.9866 |
| aware absolute vs offset_all | 0.0001 | -0.0013 | 0.0015 | 1.0063 |

#### One-way, total time: weight=n_reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Absolute response [blind] | 166 | 0.0167 | 2.3131 | -6.6501 | 11.1768 | -0.0134 | 0.0026 | 0.0268 | 0.0945 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.0165 | 1.0241 | -7.9032 | 10.4288 | -0.0136 | 0.0022 | 0.0267 | 0.1125 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.0162 | -0.5951 | -8.2076 | 7.0681 | -0.0141 | 0.0013 | 0.0269 | 0.1569 | 1.0000 | 11 |
| Delta, uniform agg. [blind] | 166 | 0.0229 | 40.1350 | -8.7548 | 82.8915 | 0.0189 | 0.0259 | 0.0208 | 0.4569 | 0.0909 | 11 |
| Delta, median agg. [blind] | 166 | 0.0242 | 47.9403 | 0.7769 | 89.7031 | 0.0200 | 0.0282 | 0.0213 | 0.4205 | 0.0000 | 11 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.0206 | 25.9014 | -24.2825 | 68.2022 | 0.0174 | 0.0241 | 0.0181 | 0.4907 | 0.2727 | 11 |
| Generic potential [blind] | 166 | 0.0205 | 25.1116 | -17.6366 | 61.8562 | 0.0150 | 0.0219 | 0.0194 | 0.5129 | 0.1818 | 11 |
| Isolated-speed baseline [aware] | 166 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Absolute response [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Delta, uniform agg. [aware] | 166 | 0.0228 | 39.4544 | -2.9322 | 75.9385 | 0.0169 | 0.0257 | 0.0207 | 0.4707 | 0.2727 | 11 |
| Delta, median agg. [aware] | 166 | 0.0147 | -10.2649 | -42.2984 | 16.1092 | 0.0064 | 0.0124 | 0.0163 | 0.6293 | 0.3636 | 11 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.0205 | 25.5885 | -31.5313 | 85.4055 | 0.0154 | 0.0252 | 0.0172 | 0.4096 | 0.4545 | 11 |
| Generic potential [aware] | 166 | 0.0241 | 47.5961 | 1.0377 | 86.3140 | 0.0192 | 0.0253 | 0.0233 | 0.5355 | 0.0909 | 11 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0004 | -0.0018 | 0.0011 | 0.9798 |
| delta_uniform | -0.0001 | -0.0038 | 0.0039 | 0.9981 |
| delta_median | -0.0095 | -0.0133 | -0.0054 | 0.6098 |
| delta_ood_kernel | -0.0001 | -0.0057 | 0.0069 | 0.9846 |
| generic | 0.0037 | 0.0017 | 0.0058 | 1.1825 |
| aware absolute vs offset_jms | -0.0002 | -0.0017 | 0.0013 | 0.9913 |
| aware absolute vs offset_all | 0.0001 | -0.0012 | 0.0013 | 0.9994 |

#### Bidirectional, MPI time: all
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.1102 | – | – | – | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Absolute response [blind] | 166 | 0.1035 | -6.1002 | -11.1364 | -1.2157 | -0.0843 | 0.0251 | 0.1392 | 0.1072 | 0.7170 | 53 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.1021 | -7.3847 | -12.0264 | -2.8415 | -0.0870 | 0.0191 | 0.1399 | 0.2006 | 0.7358 | 53 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.1011 | -8.2448 | -11.5404 | -5.1183 | -0.0949 | 0.0078 | 0.1437 | 0.3658 | 0.8679 | 53 |
| Delta, uniform agg. [blind] | 166 | 0.1486 | 34.8707 | -7.8206 | 75.8139 | 0.1192 | 0.2346 | 0.1094 | 0.2239 | 0.0000 | 53 |
| Delta, median agg. [blind] | 166 | 0.1244 | 12.8508 | -28.2356 | 49.5728 | 0.0880 | 0.2095 | 0.0855 | 0.2312 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.1300 | 17.9963 | -22.2081 | 57.0030 | 0.1086 | 0.1979 | 0.0991 | 0.2779 | 0.0000 | 53 |
| Generic potential [blind] | 166 | 0.1504 | 36.4447 | -5.3995 | 76.6160 | 0.1137 | 0.2360 | 0.1113 | 0.2148 | 0.0000 | 53 |
| Joint [blind] | 166 | 0.1082 | -1.8258 | -3.7656 | 0.0558 | -0.1035 | 0.0062 | 0.1547 | -0.0940 | 1.0000 | 53 |
| Joint, calibrated [blind] | 166 | 0.1021 | -7.3708 | -25.0356 | 10.4815 | -0.0266 | 0.0834 | 0.1106 | 0.1716 | 0.1132 | 53 |
| Bidir. profile [blind] | 166 | 0.1053 | -4.4333 | -8.8602 | -0.2171 | -0.0903 | 0.0189 | 0.1447 | 0.0457 | 0.7547 | 53 |
| Bidir. profile, calibrated [blind] | 166 | 0.1028 | -6.7477 | -25.6066 | 12.2705 | -0.0168 | 0.0920 | 0.1077 | 0.1767 | 0.0377 | 53 |
| Bidir. context [blind] | 166 | 0.1101 | -0.0666 | -0.2266 | 0.0725 | -0.1098 | 0.0005 | 0.1601 | -0.0517 | 1.0000 | 53 |
| Bidir. context, calibrated [blind] | 166 | 0.1025 | -6.9442 | -20.1608 | 5.5972 | -0.0563 | 0.0572 | 0.1232 | 0.1560 | 0.2830 | 53 |
| Wall-clock aux (ablation), cal. [blind] | 166 | 0.1026 | -6.8659 | -25.9268 | 12.2554 | -0.0169 | 0.0919 | 0.1075 | 0.1841 | 0.0377 | 53 |
| Forward-only tokens (ablation), cal. [blind] | 166 | 0.1057 | -4.0515 | -8.9768 | 0.3981 | -0.0899 | 0.0200 | 0.1449 | 0.0546 | 0.7170 | 53 |
| Isolated-speed baseline [aware] | 166 | 0.1102 | – | – | – | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Absolute response [aware] | 166 | 0.1102 | 0.0000 | 0.0000 | 0.0000 | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Delta, uniform agg. [aware] | 166 | 0.1675 | 51.9660 | 5.7439 | 96.8719 | 0.1365 | 0.2794 | 0.1164 | 0.1483 | 0.0000 | 53 |
| Delta, median agg. [aware] | 166 | 0.1260 | 14.3304 | -24.9540 | 51.8550 | 0.0735 | 0.2178 | 0.0841 | 0.2163 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.1458 | 32.3216 | -11.0663 | 75.5951 | 0.1300 | 0.2308 | 0.1071 | 0.2473 | 0.0000 | 53 |
| Generic potential [aware] | 166 | 0.1719 | 55.9938 | 8.4058 | 101.0195 | 0.1475 | 0.2852 | 0.1202 | 0.1620 | 0.0000 | 53 |
| Joint [aware] | 166 | 0.1083 | -1.7098 | -2.7217 | -0.7319 | -0.1069 | 0.0021 | 0.1568 | 0.2481 | 1.0000 | 53 |
| Joint, calibrated [aware] | 166 | 0.1044 | -5.2245 | -8.0685 | -2.4881 | -0.1001 | 0.0063 | 0.1492 | 0.3905 | 1.0000 | 53 |
| Bidir. profile [aware] | 166 | 0.1088 | -1.2918 | -2.1710 | -0.5314 | -0.1077 | 0.0016 | 0.1577 | 0.2756 | 1.0000 | 53 |
| Bidir. profile, calibrated [aware] | 166 | 0.1049 | -4.8180 | -7.4684 | -2.2438 | -0.1010 | 0.0058 | 0.1501 | 0.3897 | 1.0000 | 53 |
| Bidir. context [aware] | 166 | 0.1102 | 0.0000 | 0.0000 | 0.0000 | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Bidir. context, calibrated [aware] | 166 | 0.1102 | 0.0000 | 0.0000 | 0.0000 | -0.1102 | 0.0000 | 0.1605 | – | 1.0000 | 53 |
| Wall-clock aux (ablation), cal. [aware] | 166 | 0.1060 | -3.7930 | -5.9434 | -1.6523 | -0.1029 | 0.0047 | 0.1522 | 0.3714 | 1.0000 | 53 |
| Forward-only tokens (ablation), cal. [aware] | 166 | 0.1091 | -1.0164 | -20.1448 | 16.7344 | 0.0027 | 0.1042 | 0.1113 | 0.3056 | 0.0000 | 53 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0067 | 0.0013 | 0.0123 | 1.0650 |
| delta_uniform | 0.0188 | 0.0053 | 0.0329 | 1.1268 |
| delta_median | 0.0016 | -0.0118 | 0.0145 | 1.0131 |
| delta_ood_kernel | 0.0158 | 0.0061 | 0.0270 | 1.1214 |
| generic | 0.0215 | 0.0121 | 0.0325 | 1.1433 |
| joint_direct | 0.0001 | -0.0012 | 0.0015 | 1.0012 |
| joint_cal | 0.0024 | -0.0147 | 0.0190 | 1.0232 |
| bprof_direct | 0.0035 | -0.0005 | 0.0076 | 1.0329 |
| bprof_cal | 0.0021 | -0.0166 | 0.0209 | 1.0207 |
| bctx_direct | 0.0001 | -0.0001 | 0.0002 | 1.0007 |
| bctx_cal | 0.0077 | -0.0062 | 0.0222 | 1.0746 |
| bwall_cal | 0.0034 | -0.0160 | 0.0227 | 1.0330 |
| bfwd_cal | 0.0033 | -0.0139 | 0.0196 | 1.0316 |
| aware absolute vs offset_jms | 0.0081 | 0.0031 | 0.0133 | 1.0797 |
| aware absolute vs offset_all | 0.0091 | 0.0056 | 0.0127 | 1.0899 |

#### Bidirectional, MPI time: s_V=0
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.1301 | – | – | – | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Absolute response [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.1154 | -11.2902 | -15.2710 | -7.0059 | -0.1041 | 0.0241 | 0.1355 | 0.1927 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.1407 | 8.2057 | -26.6964 | 43.1507 | 0.0963 | 0.2731 | 0.1115 | 0.2900 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.1151 | -11.5235 | -43.5306 | 21.1492 | 0.0649 | 0.2470 | 0.0860 | 0.3070 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.1185 | -8.8614 | -42.7068 | 23.7547 | 0.0858 | 0.2230 | 0.0955 | 0.3345 | – | 0 |
| Generic potential [blind] | 83 | 0.1412 | 8.6045 | -25.3306 | 43.3598 | 0.0913 | 0.2751 | 0.1117 | 0.2695 | – | 0 |
| Joint [blind] | 83 | 0.1268 | -2.5231 | -4.6664 | -0.4299 | -0.1232 | 0.0083 | 0.1529 | -0.1322 | – | 0 |
| Joint, calibrated [blind] | 83 | 0.1064 | -18.2159 | -33.1152 | -3.2309 | -0.0467 | 0.1007 | 0.1076 | 0.1408 | – | 0 |
| Bidir. profile [blind] | 83 | 0.1201 | -7.6188 | -11.9782 | -3.3456 | -0.1102 | 0.0224 | 0.1417 | 0.0474 | – | 0 |
| Bidir. profile, calibrated [blind] | 83 | 0.1061 | -18.4247 | -34.5088 | -2.5393 | -0.0371 | 0.1111 | 0.1050 | 0.1482 | – | 0 |
| Bidir. context [blind] | 83 | 0.1298 | -0.1604 | -0.5206 | 0.0895 | -0.1296 | 0.0006 | 0.1584 | -0.0324 | – | 0 |
| Bidir. context, calibrated [blind] | 83 | 0.1103 | -15.1870 | -26.8614 | -3.7712 | -0.0758 | 0.0680 | 0.1196 | 0.1344 | – | 0 |
| Wall-clock aux (ablation), cal. [blind] | 83 | 0.1060 | -18.4813 | -34.6320 | -2.4350 | -0.0371 | 0.1111 | 0.1049 | 0.1554 | – | 0 |
| Forward-only tokens (ablation), cal. [blind] | 83 | 0.1202 | -7.5433 | -12.2807 | -3.0207 | -0.1097 | 0.0236 | 0.1416 | 0.0771 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.1301 | – | – | – | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Absolute response [aware] | 83 | 0.1301 | 0.0000 | 0.0000 | 0.0000 | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.1548 | 19.0374 | -19.0771 | 57.4763 | 0.1234 | 0.3261 | 0.1170 | 0.2678 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.1203 | -7.5040 | -39.3688 | 25.6560 | 0.0601 | 0.2700 | 0.0873 | 0.2976 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.1345 | 3.4437 | -32.9428 | 40.3727 | 0.1152 | 0.2630 | 0.1062 | 0.3490 | – | 0 |
| Generic potential [aware] | 83 | 0.1626 | 24.9918 | -13.9095 | 63.9973 | 0.1382 | 0.3379 | 0.1239 | 0.2699 | – | 0 |
| Joint [aware] | 83 | 0.1274 | -2.0750 | -3.3904 | -0.7882 | -0.1248 | 0.0064 | 0.1540 | 0.0756 | – | 0 |
| Joint, calibrated [aware] | 83 | 0.1218 | -6.3299 | -9.7513 | -3.0587 | -0.1153 | 0.0162 | 0.1451 | 0.2017 | – | 0 |
| Bidir. profile [aware] | 83 | 0.1281 | -1.4972 | -2.5559 | -0.5060 | -0.1261 | 0.0052 | 0.1552 | 0.0861 | – | 0 |
| Bidir. profile, calibrated [aware] | 83 | 0.1224 | -5.8483 | -9.0196 | -2.8232 | -0.1165 | 0.0148 | 0.1462 | 0.2090 | – | 0 |
| Bidir. context [aware] | 83 | 0.1301 | 0.0000 | 0.0000 | 0.0000 | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Bidir. context, calibrated [aware] | 83 | 0.1301 | 0.0000 | 0.0000 | 0.0000 | -0.1301 | 0.0000 | 0.1587 | – | – | 0 |
| Wall-clock aux (ablation), cal. [aware] | 83 | 0.1239 | -4.7608 | -7.3658 | -2.2261 | -0.1189 | 0.0127 | 0.1484 | 0.1852 | – | 0 |
| Forward-only tokens (ablation), cal. [aware] | 83 | 0.1084 | -16.6511 | -33.3971 | 0.7291 | -0.0108 | 0.1180 | 0.1063 | 0.0543 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0147 | 0.0091 | 0.0199 | 1.1273 |
| delta_uniform | 0.0141 | -0.0009 | 0.0295 | 1.1001 |
| delta_median | 0.0052 | -0.0079 | 0.0174 | 1.0454 |
| delta_ood_kernel | 0.0160 | 0.0065 | 0.0263 | 1.1350 |
| generic | 0.0213 | 0.0093 | 0.0348 | 1.1509 |
| joint_direct | 0.0006 | -0.0013 | 0.0026 | 1.0046 |
| joint_cal | 0.0155 | -0.0007 | 0.0308 | 1.1453 |
| bprof_direct | 0.0080 | 0.0034 | 0.0125 | 1.0663 |
| bprof_cal | 0.0164 | -0.0014 | 0.0334 | 1.1542 |
| bctx_direct | 0.0002 | -0.0001 | 0.0007 | 1.0016 |
| bctx_cal | 0.0198 | 0.0049 | 0.0349 | 1.1791 |
| bwall_cal | 0.0178 | -0.0005 | 0.0357 | 1.1683 |
| bfwd_cal | -0.0118 | -0.0300 | 0.0065 | 0.9015 |
| aware absolute vs offset_jms | 0.0147 | 0.0091 | 0.0199 | 1.1273 |
| aware absolute vs offset_all | 0.0147 | 0.0091 | 0.0199 | 1.1273 |

#### Bidirectional, MPI time: s_V=1
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0903 | – | – | – | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0916 | 1.3706 | -6.2761 | 9.1853 | -0.0645 | 0.0255 | 0.1447 | 0.0316 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0888 | -1.7629 | -8.4255 | 4.5467 | -0.0698 | 0.0171 | 0.1464 | 0.1511 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0869 | -3.8611 | -6.9436 | -1.3745 | -0.0857 | 0.0012 | 0.1558 | 0.3363 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.1565 | 73.2534 | 13.4468 | 129.1617 | 0.1420 | 0.2190 | 0.1063 | 0.2037 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.1337 | 47.9362 | -7.0524 | 97.0256 | 0.1111 | 0.1943 | 0.0849 | 0.2011 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.1415 | 56.6565 | 3.2134 | 108.2548 | 0.1314 | 0.1878 | 0.1044 | 0.2666 | – | 0 |
| Generic potential [blind] | 83 | 0.1595 | 76.5190 | 19.2401 | 130.3372 | 0.1362 | 0.2201 | 0.1107 | 0.2042 | – | 0 |
| Joint [blind] | 83 | 0.0896 | -0.8221 | -3.7919 | 2.2547 | -0.0838 | 0.0054 | 0.1573 | -0.0567 | – | 0 |
| Joint, calibrated [blind] | 83 | 0.0978 | 8.2401 | -17.2167 | 31.7482 | -0.0064 | 0.0764 | 0.1150 | 0.2269 | – | 0 |
| Bidir. profile [blind] | 83 | 0.0905 | 0.1521 | -6.5317 | 6.8779 | -0.0704 | 0.0175 | 0.1492 | 0.0671 | – | 0 |
| Bidir. profile, calibrated [blind] | 83 | 0.0994 | 10.0606 | -17.5340 | 35.4824 | 0.0035 | 0.0843 | 0.1116 | 0.2328 | – | 0 |
| Bidir. context [blind] | 83 | 0.0904 | 0.0684 | -0.3039 | 0.6005 | -0.0900 | 0.0005 | 0.1627 | -0.0540 | – | 0 |
| Bidir. context, calibrated [blind] | 83 | 0.0948 | 4.9207 | -12.2795 | 22.2012 | -0.0368 | 0.0528 | 0.1286 | 0.1907 | – | 0 |
| Wall-clock aux (ablation), cal. [blind] | 83 | 0.0993 | 9.8539 | -17.7222 | 35.6837 | 0.0033 | 0.0842 | 0.1114 | 0.2410 | – | 0 |
| Forward-only tokens (ablation), cal. [blind] | 83 | 0.0912 | 0.9747 | -6.2683 | 8.1486 | -0.0702 | 0.0185 | 0.1497 | 0.0552 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0903 | – | – | – | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0903 | 0.0000 | 0.0000 | 0.0000 | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.1801 | 99.3649 | 37.9541 | 158.5693 | 0.1495 | 0.2605 | 0.1155 | 0.0379 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.1317 | 45.7598 | -5.7096 | 94.7020 | 0.0868 | 0.1967 | 0.0794 | 0.1293 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.1571 | 73.8897 | 17.5568 | 131.6560 | 0.1448 | 0.2178 | 0.1083 | 0.1568 | – | 0 |
| Generic potential [aware] | 83 | 0.1813 | 100.6192 | 37.2209 | 160.7288 | 0.1569 | 0.2639 | 0.1148 | 0.0426 | – | 0 |
| Joint [aware] | 83 | 0.0893 | -1.1840 | -2.3755 | -0.2614 | -0.0890 | 0.0003 | 0.1608 | 0.2155 | – | 0 |
| Joint, calibrated [aware] | 83 | 0.0871 | -3.6334 | -6.3349 | -1.2327 | -0.0849 | 0.0023 | 0.1552 | 0.4324 | – | 0 |
| Bidir. profile [aware] | 83 | 0.0894 | -0.9961 | -2.0756 | -0.2013 | -0.0893 | 0.0002 | 0.1613 | 0.3202 | – | 0 |
| Bidir. profile, calibrated [aware] | 83 | 0.0873 | -3.3350 | -5.9125 | -1.0458 | -0.0854 | 0.0021 | 0.1559 | 0.4228 | – | 0 |
| Bidir. context [aware] | 83 | 0.0903 | 0.0000 | 0.0000 | 0.0000 | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Bidir. context, calibrated [aware] | 83 | 0.0903 | 0.0000 | 0.0000 | 0.0000 | -0.0903 | 0.0000 | 0.1630 | – | – | 0 |
| Wall-clock aux (ablation), cal. [aware] | 83 | 0.0882 | -2.4000 | -4.4867 | -0.6052 | -0.0869 | 0.0015 | 0.1579 | 0.4050 | – | 0 |
| Forward-only tokens (ablation), cal. [aware] | 83 | 0.1098 | 21.4888 | -4.9899 | 43.9538 | 0.0162 | 0.0985 | 0.1188 | 0.4012 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0012 | -0.0083 | 0.0057 | 0.9865 |
| delta_uniform | 0.0236 | 0.0073 | 0.0392 | 1.1507 |
| delta_median | -0.0020 | -0.0175 | 0.0135 | 0.9853 |
| delta_ood_kernel | 0.0156 | 0.0032 | 0.0296 | 1.1100 |
| generic | 0.0218 | 0.0114 | 0.0335 | 1.1365 |
| joint_direct | -0.0003 | -0.0028 | 0.0020 | 0.9964 |
| joint_cal | -0.0107 | -0.0309 | 0.0101 | 0.8903 |
| bprof_direct | -0.0010 | -0.0067 | 0.0044 | 0.9885 |
| bprof_cal | -0.0121 | -0.0342 | 0.0114 | 0.8783 |
| bctx_direct | -0.0001 | -0.0005 | 0.0003 | 0.9993 |
| bctx_cal | -0.0044 | -0.0201 | 0.0111 | 0.9531 |
| bwall_cal | -0.0111 | -0.0334 | 0.0128 | 0.8885 |
| bfwd_cal | 0.0185 | -0.0012 | 0.0355 | 1.2032 |
| aware absolute vs offset_jms | 0.0016 | -0.0041 | 0.0076 | 1.0179 |
| aware absolute vs offset_all | 0.0035 | 0.0012 | 0.0063 | 1.0402 |

#### Bidirectional, MPI time: >=2 reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 124 | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Absolute response [blind] | 124 | 0.0970 | -5.8787 | -11.1176 | -1.1371 | -0.0776 | 0.0257 | 0.1285 | 0.0458 | 0.7568 | 37 |
| Blind absolute + fixed socket offset (JMs) [blind] | 124 | 0.0956 | -7.2184 | -11.8919 | -2.9076 | -0.0807 | 0.0191 | 0.1294 | 0.1363 | 0.7838 | 37 |
| Blind absolute + fixed socket offset (all apps) [blind] | 124 | 0.0927 | -9.9898 | -13.7775 | -6.7308 | -0.0878 | 0.0061 | 0.1310 | 0.4310 | 0.8649 | 37 |
| Delta, uniform agg. [blind] | 124 | 0.1502 | 45.7712 | -0.0852 | 85.0792 | 0.1247 | 0.2468 | 0.1075 | 0.1982 | 0.0000 | 37 |
| Delta, median agg. [blind] | 124 | 0.1228 | 19.2518 | -24.6242 | 55.5806 | 0.0918 | 0.2165 | 0.0815 | 0.2072 | 0.0000 | 37 |
| Delta, OOD-kernel agg. [blind] | 124 | 0.1284 | 24.6704 | -19.7373 | 64.6857 | 0.1130 | 0.2120 | 0.0915 | 0.2635 | 0.0000 | 37 |
| Generic potential [blind] | 124 | 0.1499 | 45.5581 | 1.8066 | 85.7209 | 0.1201 | 0.2457 | 0.1076 | 0.1993 | 0.0000 | 37 |
| Joint [blind] | 124 | 0.1009 | -2.0877 | -4.1115 | -0.1148 | -0.0966 | 0.0053 | 0.1431 | -0.1009 | 1.0000 | 37 |
| Joint, calibrated [blind] | 124 | 0.0961 | -6.7339 | -25.8276 | 10.5349 | -0.0191 | 0.0851 | 0.1009 | 0.1733 | 0.1081 | 37 |
| Bidir. profile [blind] | 124 | 0.0987 | -4.1706 | -8.5863 | 0.0088 | -0.0835 | 0.0184 | 0.1342 | -0.0101 | 0.7568 | 37 |
| Bidir. profile, calibrated [blind] | 124 | 0.0972 | -5.6285 | -26.0552 | 13.0907 | -0.0094 | 0.0937 | 0.0988 | 0.1754 | 0.0270 | 37 |
| Bidir. context [blind] | 124 | 0.1030 | 0.0139 | -0.2168 | 0.3328 | -0.1027 | 0.0005 | 0.1483 | -0.0693 | 1.0000 | 37 |
| Bidir. context, calibrated [blind] | 124 | 0.0949 | -7.8267 | -21.8185 | 5.2910 | -0.0487 | 0.0582 | 0.1112 | 0.1793 | 0.2432 | 37 |
| Wall-clock aux (ablation), cal. [blind] | 124 | 0.0970 | -5.7970 | -26.2501 | 12.9299 | -0.0097 | 0.0934 | 0.0987 | 0.1869 | 0.0270 | 37 |
| Forward-only tokens (ablation), cal. [blind] | 124 | 0.0986 | -4.3253 | -9.6800 | 0.4309 | -0.0829 | 0.0191 | 0.1337 | 0.0431 | 0.7027 | 37 |
| Isolated-speed baseline [aware] | 124 | 0.1030 | – | – | – | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Absolute response [aware] | 124 | 0.1030 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Delta, uniform agg. [aware] | 124 | 0.1698 | 64.8011 | 19.0984 | 108.3920 | 0.1445 | 0.2900 | 0.1166 | 0.1410 | 0.0000 | 37 |
| Delta, median agg. [aware] | 124 | 0.1213 | 17.7818 | -22.0691 | 53.5873 | 0.0788 | 0.2197 | 0.0778 | 0.2405 | 0.0000 | 37 |
| Delta, OOD-kernel agg. [aware] | 124 | 0.1459 | 41.6377 | -5.2770 | 86.1173 | 0.1350 | 0.2439 | 0.1026 | 0.2300 | 0.0000 | 37 |
| Generic potential [aware] | 124 | 0.1776 | 72.4020 | 24.8225 | 116.0496 | 0.1561 | 0.2994 | 0.1237 | 0.1340 | 0.0000 | 37 |
| Joint [aware] | 124 | 0.1010 | -1.9093 | -3.1103 | -0.8631 | -0.0997 | 0.0019 | 0.1449 | 0.2608 | 1.0000 | 37 |
| Joint, calibrated [aware] | 124 | 0.0969 | -5.9507 | -9.1407 | -2.9778 | -0.0929 | 0.0056 | 0.1372 | 0.4428 | 1.0000 | 37 |
| Bidir. profile [aware] | 124 | 0.1015 | -1.4623 | -2.4638 | -0.5940 | -0.1005 | 0.0015 | 0.1457 | 0.3057 | 1.0000 | 37 |
| Bidir. profile, calibrated [aware] | 124 | 0.0973 | -5.5033 | -8.4770 | -2.7343 | -0.0938 | 0.0052 | 0.1381 | 0.4362 | 1.0000 | 37 |
| Bidir. context [aware] | 124 | 0.1030 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Bidir. context, calibrated [aware] | 124 | 0.1030 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | 0.0000 | 0.1485 | – | 1.0000 | 37 |
| Wall-clock aux (ablation), cal. [aware] | 124 | 0.0984 | -4.4256 | -6.8662 | -2.0946 | -0.0956 | 0.0042 | 0.1401 | 0.4132 | 1.0000 | 37 |
| Forward-only tokens (ablation), cal. [aware] | 124 | 0.1023 | -0.6809 | -21.6319 | 17.5289 | 0.0094 | 0.1019 | 0.1025 | 0.3559 | 0.0000 | 37 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0061 | 0.0012 | 0.0115 | 1.0625 |
| delta_uniform | 0.0196 | 0.0047 | 0.0344 | 1.1305 |
| delta_median | -0.0015 | -0.0151 | 0.0128 | 0.9877 |
| delta_ood_kernel | 0.0175 | 0.0063 | 0.0300 | 1.1361 |
| generic | 0.0277 | 0.0186 | 0.0378 | 1.1844 |
| joint_direct | 0.0002 | -0.0012 | 0.0015 | 1.0018 |
| joint_cal | 0.0008 | -0.0154 | 0.0183 | 1.0084 |
| bprof_direct | 0.0028 | -0.0008 | 0.0066 | 1.0283 |
| bprof_cal | 0.0001 | -0.0172 | 0.0189 | 1.0013 |
| bctx_direct | -0.0000 | -0.0003 | 0.0002 | 0.9999 |
| bctx_cal | 0.0081 | -0.0055 | 0.0225 | 1.0849 |
| bwall_cal | 0.0014 | -0.0166 | 0.0205 | 1.0146 |
| bfwd_cal | 0.0038 | -0.0143 | 0.0199 | 1.0381 |
| aware absolute vs offset_jms | 0.0074 | 0.0030 | 0.0122 | 1.0778 |
| aware absolute vs offset_all | 0.0103 | 0.0069 | 0.0142 | 1.1110 |

#### Bidirectional, MPI time: weight=n_reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.1113 | – | – | – | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Absolute response [blind] | 166 | 0.1039 | -6.6397 | -11.3115 | -2.3356 | -0.0859 | 0.0246 | 0.1370 | 0.1072 | 0.7170 | 53 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.1029 | -7.5853 | -12.0026 | -3.5679 | -0.0885 | 0.0189 | 0.1379 | 0.2006 | 0.7358 | 53 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.1011 | -9.1427 | -12.3232 | -6.2493 | -0.0962 | 0.0066 | 0.1406 | 0.3658 | 0.8679 | 53 |
| Delta, uniform agg. [blind] | 166 | 0.1481 | 33.0674 | -8.8116 | 72.5271 | 0.1197 | 0.2401 | 0.1097 | 0.2239 | 0.0000 | 53 |
| Delta, median agg. [blind] | 166 | 0.1229 | 10.4287 | -30.3858 | 45.6815 | 0.0885 | 0.2126 | 0.0855 | 0.2312 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.1279 | 14.8790 | -25.0169 | 53.6637 | 0.1087 | 0.2027 | 0.0966 | 0.2779 | 0.0000 | 53 |
| Generic potential [blind] | 166 | 0.1494 | 34.2306 | -8.3673 | 73.6174 | 0.1149 | 0.2418 | 0.1108 | 0.2148 | 0.0000 | 53 |
| Joint [blind] | 166 | 0.1091 | -1.9601 | -4.0361 | 0.1484 | -0.1045 | 0.0063 | 0.1520 | -0.0940 | 1.0000 | 53 |
| Joint, calibrated [blind] | 166 | 0.1014 | -8.8761 | -26.6262 | 9.1500 | -0.0266 | 0.0852 | 0.1082 | 0.1716 | 0.1132 | 53 |
| Bidir. profile [blind] | 166 | 0.1063 | -4.4496 | -9.0499 | 0.0426 | -0.0915 | 0.0194 | 0.1426 | 0.0457 | 0.7547 | 53 |
| Bidir. profile, calibrated [blind] | 166 | 0.1022 | -8.1932 | -27.8546 | 10.9845 | -0.0169 | 0.0936 | 0.1058 | 0.1767 | 0.0377 | 53 |
| Bidir. context [blind] | 166 | 0.1113 | -0.0299 | -0.2346 | 0.1993 | -0.1109 | 0.0006 | 0.1574 | -0.0517 | 1.0000 | 53 |
| Bidir. context, calibrated [blind] | 166 | 0.1017 | -8.5928 | -21.9548 | 5.0963 | -0.0561 | 0.0591 | 0.1195 | 0.1560 | 0.2830 | 53 |
| Wall-clock aux (ablation), cal. [blind] | 166 | 0.1021 | -8.2952 | -27.9074 | 10.8333 | -0.0170 | 0.0934 | 0.1057 | 0.1841 | 0.0377 | 53 |
| Forward-only tokens (ablation), cal. [blind] | 166 | 0.1065 | -4.2986 | -9.4308 | 0.8749 | -0.0908 | 0.0205 | 0.1424 | 0.0546 | 0.7170 | 53 |
| Isolated-speed baseline [aware] | 166 | 0.1113 | – | – | – | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Absolute response [aware] | 166 | 0.1113 | 0.0000 | 0.0000 | 0.0000 | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Delta, uniform agg. [aware] | 166 | 0.1679 | 50.8330 | 3.6832 | 93.8319 | 0.1374 | 0.2875 | 0.1180 | 0.1483 | 0.0000 | 53 |
| Delta, median agg. [aware] | 166 | 0.1229 | 10.4083 | -27.6746 | 46.7830 | 0.0740 | 0.2205 | 0.0821 | 0.2163 | 0.0000 | 53 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.1457 | 30.8626 | -11.0588 | 73.8312 | 0.1309 | 0.2388 | 0.1068 | 0.2473 | 0.0000 | 53 |
| Generic potential [aware] | 166 | 0.1740 | 56.3087 | 7.4126 | 100.7842 | 0.1486 | 0.2954 | 0.1233 | 0.1620 | 0.0000 | 53 |
| Joint [aware] | 166 | 0.1093 | -1.8103 | -2.9970 | -0.7750 | -0.1080 | 0.0021 | 0.1540 | 0.2481 | 1.0000 | 53 |
| Joint, calibrated [aware] | 166 | 0.1052 | -5.5152 | -8.7185 | -2.3548 | -0.1006 | 0.0070 | 0.1461 | 0.3905 | 1.0000 | 53 |
| Bidir. profile [aware] | 166 | 0.1098 | -1.3875 | -2.4295 | -0.5085 | -0.1087 | 0.0016 | 0.1549 | 0.2756 | 1.0000 | 53 |
| Bidir. profile, calibrated [aware] | 166 | 0.1056 | -5.0968 | -8.0827 | -2.1231 | -0.1015 | 0.0065 | 0.1470 | 0.3897 | 1.0000 | 53 |
| Bidir. context [aware] | 166 | 0.1113 | 0.0000 | 0.0000 | 0.0000 | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Bidir. context, calibrated [aware] | 166 | 0.1113 | 0.0000 | 0.0000 | 0.0000 | -0.1113 | 0.0000 | 0.1578 | – | 1.0000 | 53 |
| Wall-clock aux (ablation), cal. [aware] | 166 | 0.1068 | -4.0275 | -6.5472 | -1.5702 | -0.1035 | 0.0054 | 0.1492 | 0.3714 | 1.0000 | 53 |
| Forward-only tokens (ablation), cal. [aware] | 166 | 0.1075 | -3.3761 | -22.5320 | 15.3065 | 0.0021 | 0.1043 | 0.1089 | 0.3056 | 0.0000 | 53 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0074 | 0.0026 | 0.0126 | 1.0650 |
| delta_uniform | 0.0198 | 0.0060 | 0.0337 | 1.1268 |
| delta_median | -0.0000 | -0.0131 | 0.0137 | 1.0131 |
| delta_ood_kernel | 0.0178 | 0.0079 | 0.0291 | 1.1214 |
| generic | 0.0246 | 0.0154 | 0.0348 | 1.1433 |
| joint_direct | 0.0002 | -0.0012 | 0.0014 | 1.0012 |
| joint_cal | 0.0037 | -0.0132 | 0.0214 | 1.0232 |
| bprof_direct | 0.0034 | -0.0007 | 0.0077 | 1.0329 |
| bprof_cal | 0.0034 | -0.0153 | 0.0227 | 1.0207 |
| bctx_direct | 0.0000 | -0.0002 | 0.0003 | 1.0007 |
| bctx_cal | 0.0096 | -0.0057 | 0.0244 | 1.0746 |
| bwall_cal | 0.0048 | -0.0149 | 0.0246 | 1.0330 |
| bfwd_cal | 0.0010 | -0.0164 | 0.0174 | 1.0316 |
| aware absolute vs offset_jms | 0.0084 | 0.0040 | 0.0134 | 1.0797 |
| aware absolute vs offset_all | 0.0102 | 0.0070 | 0.0137 | 1.0899 |

#### Bidirectional, total time: all
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.0157 | – | – | – | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Absolute response [blind] | 166 | 0.0161 | 2.0617 | -8.3069 | 12.3551 | -0.0126 | 0.0028 | 0.0262 | 0.0945 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.0159 | 0.8826 | -9.1479 | 11.6556 | -0.0127 | 0.0024 | 0.0262 | 0.1125 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.0157 | 0.0629 | -8.3702 | 8.9535 | -0.0132 | 0.0018 | 0.0264 | 0.1569 | 1.0000 | 11 |
| Delta, uniform agg. [blind] | 166 | 0.0201 | 27.6835 | -18.1534 | 68.2731 | 0.0149 | 0.0220 | 0.0186 | 0.4958 | 0.0909 | 11 |
| Delta, median agg. [blind] | 166 | 0.0211 | 34.0829 | -9.2643 | 71.7883 | 0.0159 | 0.0235 | 0.0193 | 0.5053 | 0.0000 | 11 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.0169 | 7.3311 | -43.6061 | 52.7042 | 0.0121 | 0.0196 | 0.0148 | 0.4453 | 0.4545 | 11 |
| Generic potential [blind] | 166 | 0.0206 | 30.8412 | -15.6109 | 69.3879 | 0.0153 | 0.0227 | 0.0190 | 0.5129 | 0.1818 | 11 |
| Joint [blind] | 166 | 0.0157 | 0.0552 | 0.0000 | 0.1698 | -0.0157 | 0.0000 | 0.0278 | -0.1130 | 1.0000 | 11 |
| Joint, calibrated [blind] | 166 | 0.0142 | -9.4797 | -19.1098 | -1.5071 | -0.0131 | 0.0009 | 0.0245 | 0.3019 | 1.0000 | 11 |
| Bidir. profile [blind] | 166 | 0.0159 | 0.9906 | -0.1905 | 2.5981 | -0.0154 | 0.0005 | 0.0277 | -0.0954 | 1.0000 | 11 |
| Bidir. profile, calibrated [blind] | 166 | 0.0149 | -5.1660 | -13.4070 | 2.1404 | -0.0124 | 0.0015 | 0.0252 | 0.2362 | 1.0000 | 11 |
| Bidir. context [blind] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Bidir. context, calibrated [blind] | 166 | 0.0155 | -1.4197 | -3.7030 | 0.0000 | -0.0155 | 0.0000 | 0.0274 | 0.1767 | 1.0000 | 11 |
| Wall-clock aux (ablation), cal. [blind] | 166 | 0.0146 | -7.3053 | -16.4210 | 0.6275 | -0.0126 | 0.0013 | 0.0247 | 0.2985 | 1.0000 | 11 |
| Forward-only tokens (ablation), cal. [blind] | 166 | 0.0175 | 11.2377 | -2.8058 | 26.0515 | -0.0101 | 0.0060 | 0.0263 | -0.0855 | 1.0000 | 11 |
| Isolated-speed baseline [aware] | 166 | 0.0157 | – | – | – | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Absolute response [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Delta, uniform agg. [aware] | 166 | 0.0233 | 48.0624 | 0.2901 | 87.0182 | 0.0161 | 0.0237 | 0.0230 | 0.4712 | 0.0000 | 11 |
| Delta, median agg. [aware] | 166 | 0.0166 | 5.5497 | -32.5794 | 38.8321 | 0.0086 | 0.0138 | 0.0188 | 0.5898 | 0.1818 | 11 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.0188 | 19.4714 | -37.4801 | 75.2050 | 0.0147 | 0.0235 | 0.0152 | 0.4623 | 0.3636 | 11 |
| Generic potential [aware] | 166 | 0.0243 | 54.7251 | 4.9969 | 94.8086 | 0.0195 | 0.0260 | 0.0231 | 0.5355 | 0.0909 | 11 |
| Joint [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Joint, calibrated [aware] | 166 | 0.0156 | -0.7218 | -1.9018 | 0.0000 | -0.0156 | 0.0000 | 0.0276 | 0.1857 | 1.0000 | 11 |
| Bidir. profile [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Bidir. profile, calibrated [aware] | 166 | 0.0156 | -0.7864 | -2.0070 | 0.0000 | -0.0156 | 0.0000 | 0.0276 | 0.1857 | 1.0000 | 11 |
| Bidir. context [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Bidir. context, calibrated [aware] | 166 | 0.0157 | 0.0000 | 0.0000 | 0.0000 | -0.0157 | 0.0000 | 0.0278 | – | 1.0000 | 11 |
| Wall-clock aux (ablation), cal. [aware] | 166 | 0.0156 | -0.7888 | -2.0015 | 0.0000 | -0.0156 | 0.0000 | 0.0276 | 0.1857 | 1.0000 | 11 |
| Forward-only tokens (ablation), cal. [aware] | 166 | 0.0956 | 507.4747 | 429.2196 | 572.9671 | 0.0950 | 0.0997 | 0.0925 | 0.4767 | 0.0000 | 11 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0003 | -0.0019 | 0.0013 | 0.9798 |
| delta_uniform | 0.0032 | 0.0001 | 0.0065 | 1.1596 |
| delta_median | -0.0045 | -0.0071 | -0.0020 | 0.7872 |
| delta_ood_kernel | 0.0019 | -0.0033 | 0.0081 | 1.1131 |
| generic | 0.0038 | 0.0019 | 0.0058 | 1.1825 |
| joint_direct | -0.0000 | -0.0000 | 0.0000 | 0.9994 |
| joint_cal | 0.0014 | 0.0002 | 0.0028 | 1.0968 |
| bprof_direct | -0.0002 | -0.0004 | 0.0000 | 0.9902 |
| bprof_cal | 0.0007 | -0.0004 | 0.0019 | 1.0462 |
| bctx_direct | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| bctx_cal | 0.0002 | 0.0000 | 0.0006 | 1.0144 |
| bwall_cal | 0.0010 | -0.0002 | 0.0024 | 1.0703 |
| bfwd_cal | 0.0781 | 0.0664 | 0.0874 | 5.4611 |
| aware absolute vs offset_jms | -0.0001 | -0.0018 | 0.0014 | 0.9913 |
| aware absolute vs offset_all | -0.0000 | -0.0014 | 0.0013 | 0.9994 |

#### Bidirectional, total time: s_V=0
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0162 | -0.6155 | -10.1241 | 9.2719 | -0.0131 | 0.0029 | 0.0210 | 0.1447 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.0199 | 22.1636 | -20.5983 | 57.4962 | 0.0136 | 0.0238 | 0.0185 | 0.4564 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.0204 | 25.4043 | -15.3642 | 58.5151 | 0.0147 | 0.0248 | 0.0188 | 0.4675 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.0161 | -1.3107 | -49.9429 | 43.1527 | 0.0105 | 0.0229 | 0.0136 | 0.3614 | – | 0 |
| Generic potential [blind] | 83 | 0.0201 | 23.5066 | -15.9905 | 59.2025 | 0.0140 | 0.0252 | 0.0183 | 0.4499 | – | 0 |
| Joint [blind] | 83 | 0.0163 | 0.0533 | 0.0000 | 0.1640 | -0.0163 | 0.0000 | 0.0222 | -0.1419 | – | 0 |
| Joint, calibrated [blind] | 83 | 0.0145 | -10.7709 | -20.8130 | -2.8270 | -0.0135 | 0.0006 | 0.0195 | 0.2229 | – | 0 |
| Bidir. profile [blind] | 83 | 0.0164 | 0.7432 | -0.5519 | 2.3429 | -0.0160 | 0.0007 | 0.0221 | -0.1557 | – | 0 |
| Bidir. profile, calibrated [blind] | 83 | 0.0154 | -5.4446 | -14.1959 | 2.3271 | -0.0129 | 0.0016 | 0.0204 | 0.1733 | – | 0 |
| Bidir. context [blind] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Bidir. context, calibrated [blind] | 83 | 0.0160 | -1.8660 | -4.9111 | 0.0000 | -0.0160 | 0.0000 | 0.0217 | 0.2149 | – | 0 |
| Wall-clock aux (ablation), cal. [blind] | 83 | 0.0149 | -8.3685 | -18.7169 | 0.1934 | -0.0130 | 0.0013 | 0.0199 | 0.2257 | – | 0 |
| Forward-only tokens (ablation), cal. [blind] | 83 | 0.0176 | 8.2570 | -6.7579 | 23.2334 | -0.0104 | 0.0072 | 0.0214 | -0.1528 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.0260 | 59.3281 | 13.9343 | 99.0536 | 0.0192 | 0.0330 | 0.0234 | 0.3140 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.0191 | 17.3172 | -20.0983 | 49.4981 | 0.0120 | 0.0216 | 0.0182 | 0.4815 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.0201 | 23.7057 | -34.0954 | 80.8304 | 0.0164 | 0.0321 | 0.0158 | 0.2732 | – | 0 |
| Generic potential [aware] | 83 | 0.0281 | 72.7566 | 25.2880 | 112.0493 | 0.0239 | 0.0371 | 0.0249 | 0.3872 | – | 0 |
| Joint [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Joint, calibrated [aware] | 83 | 0.0161 | -0.9512 | -2.6798 | 0.0000 | -0.0161 | 0.0000 | 0.0220 | 0.1521 | – | 0 |
| Bidir. profile [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Bidir. profile, calibrated [aware] | 83 | 0.0161 | -0.9945 | -2.7997 | 0.0000 | -0.0161 | 0.0000 | 0.0219 | 0.1521 | – | 0 |
| Bidir. context [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Bidir. context, calibrated [aware] | 83 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0222 | – | – | 0 |
| Wall-clock aux (ablation), cal. [aware] | 83 | 0.0161 | -0.9635 | -2.7292 | 0.0000 | -0.0161 | 0.0000 | 0.0219 | 0.1521 | – | 0 |
| Forward-only tokens (ablation), cal. [aware] | 83 | 0.1050 | 544.9409 | 467.0710 | 606.5321 | 0.1041 | 0.1177 | 0.1005 | 0.2952 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | 0.0001 | -0.0015 | 0.0016 | 1.0062 |
| delta_uniform | 0.0061 | 0.0029 | 0.0093 | 1.3042 |
| delta_median | -0.0013 | -0.0034 | 0.0008 | 0.9355 |
| delta_ood_kernel | 0.0041 | -0.0010 | 0.0102 | 1.2535 |
| generic | 0.0080 | 0.0056 | 0.0105 | 1.3988 |
| joint_direct | -0.0000 | -0.0000 | 0.0000 | 0.9995 |
| joint_cal | 0.0016 | 0.0003 | 0.0031 | 1.1101 |
| bprof_direct | -0.0001 | -0.0004 | 0.0001 | 0.9926 |
| bprof_cal | 0.0007 | -0.0005 | 0.0022 | 1.0471 |
| bctx_direct | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| bctx_cal | 0.0003 | 0.0000 | 0.0008 | 1.0190 |
| bwall_cal | 0.0012 | -0.0001 | 0.0029 | 1.0808 |
| bfwd_cal | 0.0874 | 0.0755 | 0.0967 | 5.9575 |
| aware absolute vs offset_jms | 0.0001 | -0.0015 | 0.0016 | 1.0062 |
| aware absolute vs offset_all | 0.0001 | -0.0015 | 0.0016 | 1.0062 |

#### Bidirectional, total time: s_V=1
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 83 | 0.0152 | – | – | – | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Absolute response [blind] | 83 | 0.0159 | 4.9328 | -6.8264 | 16.6338 | -0.0120 | 0.0027 | 0.0360 | 0.0590 | – | 0 |
| Blind absolute + fixed socket offset (JMs) [blind] | 83 | 0.0156 | 2.4893 | -8.6850 | 13.6796 | -0.0124 | 0.0022 | 0.0358 | 0.0856 | – | 0 |
| Blind absolute + fixed socket offset (all apps) [blind] | 83 | 0.0153 | 0.7904 | -7.0836 | 8.8353 | -0.0134 | 0.0013 | 0.0365 | 0.1241 | – | 0 |
| Delta, uniform agg. [blind] | 83 | 0.0203 | 33.6033 | -19.5295 | 81.3026 | 0.0162 | 0.0212 | 0.0189 | 0.5685 | – | 0 |
| Delta, median agg. [blind] | 83 | 0.0218 | 43.3902 | -7.7082 | 87.8158 | 0.0171 | 0.0229 | 0.0201 | 0.5823 | – | 0 |
| Delta, OOD-kernel agg. [blind] | 83 | 0.0177 | 16.5990 | -40.0850 | 65.6586 | 0.0137 | 0.0181 | 0.0170 | 0.5420 | – | 0 |
| Generic potential [blind] | 83 | 0.0211 | 38.7073 | -15.0922 | 84.2628 | 0.0166 | 0.0216 | 0.0202 | 0.6112 | – | 0 |
| Joint [blind] | 83 | 0.0152 | 0.0572 | 0.0000 | 0.1759 | -0.0152 | 0.0000 | 0.0382 | -0.0860 | – | 0 |
| Joint, calibrated [blind] | 83 | 0.0140 | -8.0950 | -17.8792 | 0.5675 | -0.0126 | 0.0010 | 0.0336 | 0.3777 | – | 0 |
| Bidir. profile [blind] | 83 | 0.0154 | 1.2559 | -0.1735 | 2.9830 | -0.0149 | 0.0004 | 0.0381 | -0.0485 | – | 0 |
| Bidir. profile, calibrated [blind] | 83 | 0.0144 | -4.8672 | -14.4895 | 4.4222 | -0.0119 | 0.0015 | 0.0341 | 0.3056 | – | 0 |
| Bidir. context [blind] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Bidir. context, calibrated [blind] | 83 | 0.0150 | -0.9411 | -2.8930 | 0.0000 | -0.0150 | 0.0000 | 0.0378 | 0.1356 | – | 0 |
| Wall-clock aux (ablation), cal. [blind] | 83 | 0.0143 | -6.1650 | -15.9365 | 3.1502 | -0.0122 | 0.0014 | 0.0338 | 0.3671 | – | 0 |
| Forward-only tokens (ablation), cal. [blind] | 83 | 0.0174 | 14.4343 | -0.6880 | 29.7796 | -0.0098 | 0.0054 | 0.0355 | -0.0328 | – | 0 |
| Isolated-speed baseline [aware] | 83 | 0.0152 | – | – | – | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Absolute response [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Delta, uniform agg. [aware] | 83 | 0.0207 | 35.9806 | -14.6920 | 79.9278 | 0.0130 | 0.0197 | 0.0221 | 0.5540 | – | 0 |
| Delta, median agg. [aware] | 83 | 0.0141 | -7.0703 | -46.9489 | 29.4660 | 0.0053 | 0.0104 | 0.0197 | 0.6255 | – | 0 |
| Delta, OOD-kernel agg. [aware] | 83 | 0.0175 | 14.9303 | -49.1956 | 78.1508 | 0.0131 | 0.0198 | 0.0140 | 0.5248 | – | 0 |
| Generic potential [aware] | 83 | 0.0206 | 35.3873 | -18.4106 | 78.7771 | 0.0152 | 0.0211 | 0.0197 | 0.5988 | – | 0 |
| Joint [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Joint, calibrated [aware] | 83 | 0.0151 | -0.4757 | -1.2602 | 0.0000 | -0.0151 | 0.0000 | 0.0380 | 0.2306 | – | 0 |
| Bidir. profile [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Bidir. profile, calibrated [aware] | 83 | 0.0151 | -0.5631 | -1.6500 | 0.0000 | -0.0151 | 0.0000 | 0.0380 | 0.2306 | – | 0 |
| Bidir. context [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Bidir. context, calibrated [aware] | 83 | 0.0152 | 0.0000 | 0.0000 | 0.0000 | -0.0152 | 0.0000 | 0.0382 | – | – | 0 |
| Wall-clock aux (ablation), cal. [aware] | 83 | 0.0151 | -0.6014 | -1.8262 | 0.0000 | -0.0151 | 0.0000 | 0.0380 | 0.2306 | – | 0 |
| Forward-only tokens (ablation), cal. [aware] | 83 | 0.0862 | 467.2940 | 382.3113 | 538.4027 | 0.0858 | 0.0917 | 0.0777 | 0.5577 | – | 0 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0007 | -0.0025 | 0.0010 | 0.9530 |
| delta_uniform | 0.0004 | -0.0033 | 0.0039 | 1.0178 |
| delta_median | -0.0077 | -0.0111 | -0.0045 | 0.6481 |
| delta_ood_kernel | -0.0003 | -0.0061 | 0.0061 | 0.9857 |
| generic | -0.0005 | -0.0023 | 0.0013 | 0.9761 |
| joint_direct | -0.0000 | -0.0000 | 0.0000 | 0.9994 |
| joint_cal | 0.0012 | -0.0001 | 0.0026 | 1.0829 |
| bprof_direct | -0.0002 | -0.0005 | 0.0000 | 0.9876 |
| bprof_cal | 0.0007 | -0.0007 | 0.0020 | 1.0452 |
| bctx_direct | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| bctx_cal | 0.0001 | 0.0000 | 0.0004 | 1.0095 |
| bwall_cal | 0.0008 | -0.0005 | 0.0023 | 1.0593 |
| bfwd_cal | 0.0688 | 0.0566 | 0.0789 | 4.9574 |
| aware absolute vs offset_jms | -0.0004 | -0.0021 | 0.0013 | 0.9757 |
| aware absolute vs offset_all | -0.0001 | -0.0013 | 0.0011 | 0.9922 |

#### Bidirectional, total time: >=2 reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 124 | 0.0150 | – | – | – | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Absolute response [blind] | 124 | 0.0155 | 3.0084 | -7.9343 | 13.7750 | -0.0122 | 0.0027 | 0.0247 | 0.0143 | 1.0000 | 7 |
| Blind absolute + fixed socket offset (JMs) [blind] | 124 | 0.0152 | 1.3564 | -9.0315 | 12.0922 | -0.0124 | 0.0022 | 0.0246 | 0.0379 | 1.0000 | 7 |
| Blind absolute + fixed socket offset (all apps) [blind] | 124 | 0.0149 | -0.6244 | -9.8098 | 8.7673 | -0.0128 | 0.0014 | 0.0247 | 0.1730 | 1.0000 | 7 |
| Delta, uniform agg. [blind] | 124 | 0.0194 | 29.3293 | -18.8205 | 68.9678 | 0.0138 | 0.0206 | 0.0186 | 0.4767 | 0.1429 | 7 |
| Delta, median agg. [blind] | 124 | 0.0204 | 35.7680 | -10.0307 | 73.2771 | 0.0149 | 0.0222 | 0.0192 | 0.4919 | 0.0000 | 7 |
| Delta, OOD-kernel agg. [blind] | 124 | 0.0172 | 14.7085 | -47.8559 | 67.2989 | 0.0127 | 0.0208 | 0.0147 | 0.3887 | 0.2857 | 7 |
| Generic potential [blind] | 124 | 0.0198 | 31.4097 | -14.6277 | 67.9079 | 0.0144 | 0.0208 | 0.0190 | 0.5086 | 0.1429 | 7 |
| Joint [blind] | 124 | 0.0150 | 0.0387 | 0.0000 | 0.1240 | -0.0150 | 0.0000 | 0.0259 | -0.0942 | 1.0000 | 7 |
| Joint, calibrated [blind] | 124 | 0.0136 | -9.8687 | -18.2943 | -2.9488 | -0.0127 | 0.0006 | 0.0229 | 0.3204 | 1.0000 | 7 |
| Bidir. profile [blind] | 124 | 0.0152 | 1.0142 | -0.0026 | 2.2769 | -0.0148 | 0.0004 | 0.0259 | -0.1281 | 1.0000 | 7 |
| Bidir. profile, calibrated [blind] | 124 | 0.0143 | -4.5868 | -11.1997 | 2.0304 | -0.0118 | 0.0014 | 0.0237 | 0.2401 | 1.0000 | 7 |
| Bidir. context [blind] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Bidir. context, calibrated [blind] | 124 | 0.0148 | -1.3529 | -3.6589 | 0.0000 | -0.0148 | 0.0000 | 0.0255 | 0.1616 | 1.0000 | 7 |
| Wall-clock aux (ablation), cal. [blind] | 124 | 0.0139 | -7.4640 | -15.1482 | -0.7070 | -0.0122 | 0.0010 | 0.0232 | 0.3157 | 1.0000 | 7 |
| Forward-only tokens (ablation), cal. [blind] | 124 | 0.0168 | 11.4167 | -1.8852 | 25.1287 | -0.0098 | 0.0054 | 0.0250 | -0.0616 | 1.0000 | 7 |
| Isolated-speed baseline [aware] | 124 | 0.0150 | – | – | – | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Absolute response [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Delta, uniform agg. [aware] | 124 | 0.0214 | 42.4350 | -1.0605 | 77.8534 | 0.0148 | 0.0197 | 0.0226 | 0.5307 | 0.0000 | 7 |
| Delta, median agg. [aware] | 124 | 0.0148 | -1.8116 | -36.6437 | 24.9454 | 0.0074 | 0.0100 | 0.0182 | 0.6718 | 0.1429 | 7 |
| Delta, OOD-kernel agg. [aware] | 124 | 0.0183 | 21.9916 | -45.8123 | 84.2181 | 0.0153 | 0.0224 | 0.0154 | 0.4869 | 0.2857 | 7 |
| Generic potential [aware] | 124 | 0.0234 | 55.3466 | 8.5456 | 92.3647 | 0.0185 | 0.0238 | 0.0230 | 0.5422 | 0.0000 | 7 |
| Joint [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Joint, calibrated [aware] | 124 | 0.0149 | -1.0113 | -2.5928 | 0.0000 | -0.0149 | 0.0000 | 0.0256 | 0.2240 | 1.0000 | 7 |
| Bidir. profile [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Bidir. profile, calibrated [aware] | 124 | 0.0149 | -1.1018 | -2.7524 | 0.0000 | -0.0149 | 0.0000 | 0.0256 | 0.2240 | 1.0000 | 7 |
| Bidir. context [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Bidir. context, calibrated [aware] | 124 | 0.0150 | 0.0000 | 0.0000 | 0.0000 | -0.0150 | 0.0000 | 0.0259 | – | 1.0000 | 7 |
| Wall-clock aux (ablation), cal. [aware] | 124 | 0.0149 | -1.1051 | -2.7630 | 0.0000 | -0.0149 | 0.0000 | 0.0256 | 0.2240 | 1.0000 | 7 |
| Forward-only tokens (ablation), cal. [aware] | 124 | 0.0952 | 533.3443 | 446.5853 | 596.0825 | 0.0946 | 0.0967 | 0.0942 | 0.5114 | 0.0000 | 7 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0005 | -0.0021 | 0.0012 | 0.9708 |
| delta_uniform | 0.0020 | -0.0016 | 0.0059 | 1.1013 |
| delta_median | -0.0057 | -0.0082 | -0.0029 | 0.7232 |
| delta_ood_kernel | 0.0011 | -0.0047 | 0.0076 | 1.0635 |
| generic | 0.0036 | 0.0015 | 0.0058 | 1.1822 |
| joint_direct | -0.0000 | -0.0000 | 0.0000 | 0.9996 |
| joint_cal | 0.0013 | 0.0004 | 0.0024 | 1.0983 |
| bprof_direct | -0.0002 | -0.0003 | 0.0000 | 0.9900 |
| bprof_cal | 0.0005 | -0.0004 | 0.0015 | 1.0365 |
| bctx_direct | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| bctx_cal | 0.0002 | 0.0000 | 0.0006 | 1.0137 |
| bwall_cal | 0.0010 | 0.0000 | 0.0019 | 1.0687 |
| bfwd_cal | 0.0785 | 0.0663 | 0.0876 | 5.6845 |
| aware absolute vs offset_jms | -0.0002 | -0.0018 | 0.0014 | 0.9866 |
| aware absolute vs offset_all | 0.0001 | -0.0013 | 0.0015 | 1.0063 |

#### Bidirectional, total time: weight=n_reps
| method | n | log MAE | Δ% | CI lo | CI hi | bias | floor log MAE | above log MAE | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 166 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Absolute response [blind] | 166 | 0.0167 | 2.3131 | -6.6501 | 11.1768 | -0.0134 | 0.0026 | 0.0268 | 0.0945 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (JMs) [blind] | 166 | 0.0165 | 1.0241 | -7.9032 | 10.4288 | -0.0136 | 0.0022 | 0.0267 | 0.1125 | 1.0000 | 11 |
| Blind absolute + fixed socket offset (all apps) [blind] | 166 | 0.0162 | -0.5951 | -8.2076 | 7.0681 | -0.0141 | 0.0013 | 0.0269 | 0.1569 | 1.0000 | 11 |
| Delta, uniform agg. [blind] | 166 | 0.0200 | 22.4908 | -22.5186 | 61.2049 | 0.0145 | 0.0212 | 0.0192 | 0.4958 | 0.0909 | 11 |
| Delta, median agg. [blind] | 166 | 0.0210 | 28.3892 | -15.0035 | 65.0492 | 0.0153 | 0.0227 | 0.0197 | 0.5053 | 0.0000 | 11 |
| Delta, OOD-kernel agg. [blind] | 166 | 0.0172 | 5.1452 | -43.5337 | 47.9638 | 0.0120 | 0.0199 | 0.0152 | 0.4453 | 0.4545 | 11 |
| Generic potential [blind] | 166 | 0.0205 | 25.1116 | -17.6366 | 61.8562 | 0.0150 | 0.0219 | 0.0194 | 0.5129 | 0.1818 | 11 |
| Joint [blind] | 166 | 0.0164 | 0.0490 | 0.0000 | 0.1470 | -0.0163 | 0.0000 | 0.0280 | -0.1130 | 1.0000 | 11 |
| Joint, calibrated [blind] | 166 | 0.0147 | -10.2453 | -19.0911 | -2.5968 | -0.0135 | 0.0008 | 0.0246 | 0.3019 | 1.0000 | 11 |
| Bidir. profile [blind] | 166 | 0.0165 | 1.0230 | -0.0786 | 2.4317 | -0.0161 | 0.0005 | 0.0280 | -0.0954 | 1.0000 | 11 |
| Bidir. profile, calibrated [blind] | 166 | 0.0154 | -5.9143 | -13.6573 | 1.0908 | -0.0128 | 0.0015 | 0.0253 | 0.2362 | 1.0000 | 11 |
| Bidir. context [blind] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Bidir. context, calibrated [blind] | 166 | 0.0161 | -1.7146 | -4.3400 | 0.0000 | -0.0161 | 0.0000 | 0.0275 | 0.1767 | 1.0000 | 11 |
| Wall-clock aux (ablation), cal. [blind] | 166 | 0.0151 | -7.7232 | -16.0579 | -0.1550 | -0.0130 | 0.0013 | 0.0249 | 0.2985 | 1.0000 | 11 |
| Forward-only tokens (ablation), cal. [blind] | 166 | 0.0181 | 10.7116 | -2.7784 | 24.2279 | -0.0106 | 0.0061 | 0.0267 | -0.0855 | 1.0000 | 11 |
| Isolated-speed baseline [aware] | 166 | 0.0163 | – | – | – | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Absolute response [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Delta, uniform agg. [aware] | 166 | 0.0229 | 40.0352 | -2.4852 | 77.6611 | 0.0160 | 0.0222 | 0.0234 | 0.4712 | 0.0000 | 11 |
| Delta, median agg. [aware] | 166 | 0.0161 | -1.5928 | -36.3740 | 29.0395 | 0.0084 | 0.0120 | 0.0190 | 0.5898 | 0.1818 | 11 |
| Delta, OOD-kernel agg. [aware] | 166 | 0.0190 | 16.4623 | -41.7335 | 73.9212 | 0.0153 | 0.0235 | 0.0159 | 0.4623 | 0.3636 | 11 |
| Generic potential [aware] | 166 | 0.0241 | 47.5961 | 1.0377 | 86.3140 | 0.0192 | 0.0253 | 0.0233 | 0.5355 | 0.0909 | 11 |
| Joint [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Joint, calibrated [aware] | 166 | 0.0162 | -0.8210 | -1.9617 | 0.0000 | -0.0162 | 0.0000 | 0.0278 | 0.1857 | 1.0000 | 11 |
| Bidir. profile [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Bidir. profile, calibrated [aware] | 166 | 0.0162 | -0.9317 | -2.2149 | 0.0000 | -0.0162 | 0.0000 | 0.0278 | 0.1857 | 1.0000 | 11 |
| Bidir. context [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Bidir. context, calibrated [aware] | 166 | 0.0163 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0280 | – | 1.0000 | 11 |
| Wall-clock aux (ablation), cal. [aware] | 166 | 0.0162 | -0.9507 | -2.2978 | 0.0000 | -0.0162 | 0.0000 | 0.0278 | 0.1857 | 1.0000 | 11 |
| Forward-only tokens (ablation), cal. [aware] | 166 | 0.0955 | 484.1116 | 412.5612 | 543.4677 | 0.0947 | 0.0987 | 0.0932 | 0.4767 | 0.0000 | 11 |

Paired aware − blind (same role), and aware absolute − offset model:
| role | aware − blind log MAE | CI lo | CI hi | ratio |
|---|---|---|---|---|
| absolute | -0.0004 | -0.0018 | 0.0011 | 0.9798 |
| delta_uniform | 0.0029 | -0.0002 | 0.0062 | 1.1596 |
| delta_median | -0.0049 | -0.0073 | -0.0025 | 0.7872 |
| delta_ood_kernel | 0.0018 | -0.0035 | 0.0076 | 1.1131 |
| generic | 0.0037 | 0.0017 | 0.0058 | 1.1825 |
| joint_direct | -0.0000 | -0.0000 | 0.0000 | 0.9994 |
| joint_cal | 0.0015 | 0.0004 | 0.0029 | 1.0968 |
| bprof_direct | -0.0002 | -0.0004 | 0.0000 | 0.9902 |
| bprof_cal | 0.0008 | -0.0003 | 0.0020 | 1.0462 |
| bctx_direct | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| bctx_cal | 0.0003 | 0.0000 | 0.0007 | 1.0144 |
| bwall_cal | 0.0011 | -0.0001 | 0.0024 | 1.0703 |
| bfwd_cal | 0.0774 | 0.0664 | 0.0861 | 5.4611 |
| aware absolute vs offset_jms | -0.0002 | -0.0017 | 0.0013 | 0.9913 |
| aware absolute vs offset_all | 0.0001 | -0.0012 | 0.0013 | 0.9994 |

### B2. Rep-averaged App-App outcomes (90 directed non-self; tab:ns_fulldata convention)

#### One-way, MPI time
| method | n | log MAE | artifact log MAE | Δ% | CI lo | CI hi | bias | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 90 | 0.1030 | 0.1030 | – | – | – | -0.1030 | – | 1.0000 | 29 |
| Absolute response [blind] | 90 | 0.0942 | 0.0942 | -8.5292 | -13.6802 | -3.0764 | -0.0776 | 0.0583 | 0.7241 | 29 |
| Blind absolute + fixed socket offset (JMs) [blind] | 90 | 0.0941 | – | -8.6717 | -13.3827 | -3.7676 | -0.0802 | 0.0772 | 0.7586 | 29 |
| Blind absolute + fixed socket offset (all apps) [blind] | 90 | 0.0941 | – | -8.6863 | -12.2387 | -5.3921 | -0.0879 | 0.2514 | 0.8621 | 29 |
| Delta, uniform agg. [blind] | 90 | 0.1522 | 0.1522 | 47.7847 | -2.0455 | 92.7974 | 0.1341 | 0.2551 | 0.0000 | 29 |
| Delta, median agg. [blind] | 90 | 0.1235 | 0.1235 | 19.8907 | -26.6768 | 58.1644 | 0.0982 | 0.2839 | 0.0000 | 29 |
| Delta, OOD-kernel agg. [blind] | 90 | 0.1302 | 0.1302 | 26.3567 | -18.5885 | 69.5565 | 0.1220 | 0.3345 | 0.0000 | 29 |
| Generic potential [blind] | 90 | 0.1472 | 0.1472 | 42.9381 | -5.4684 | 86.7255 | 0.1231 | 0.2467 | 0.0000 | 29 |
| Isolated-speed baseline [aware] | 90 | 0.1030 | 0.1030 | – | – | – | -0.1030 | – | 1.0000 | 29 |
| Absolute response [aware] | 90 | 0.1030 | 0.0942 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | – | 1.0000 | 29 |
| Delta, uniform agg. [aware] | 90 | 0.1744 | 0.1522 | 69.3156 | 19.7998 | 116.6100 | 0.1569 | 0.1749 | 0.0000 | 29 |
| Delta, median agg. [aware] | 90 | 0.1289 | 0.1235 | 25.1612 | -19.6258 | 65.9453 | 0.1092 | 0.2568 | 0.0000 | 29 |
| Delta, OOD-kernel agg. [aware] | 90 | 0.1491 | 0.1302 | 44.7177 | -1.1916 | 91.0003 | 0.1382 | 0.2315 | 0.0000 | 29 |
| Generic potential [aware] | 90 | 0.1783 | 0.1472 | 73.0852 | 17.2279 | 122.2356 | 0.1569 | 0.1386 | 0.0000 | 29 |

#### One-way, total time
| method | n | log MAE | artifact log MAE | Δ% | CI lo | CI hi | bias | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 90 | 0.0156 | 0.0156 | – | – | – | -0.0156 | – | 1.0000 | 5 |
| Absolute response [blind] | 90 | 0.0162 | 0.0162 | 3.6121 | -6.4172 | 13.7406 | -0.0127 | 0.0087 | 1.0000 | 5 |
| Blind absolute + fixed socket offset (JMs) [blind] | 90 | 0.0160 | – | 2.2673 | -7.4349 | 12.0055 | -0.0129 | 0.0091 | 1.0000 | 5 |
| Blind absolute + fixed socket offset (all apps) [blind] | 90 | 0.0157 | – | 0.5784 | -7.8012 | 8.6784 | -0.0134 | 0.0444 | 1.0000 | 5 |
| Delta, uniform agg. [blind] | 90 | 0.0228 | 0.0228 | 45.9741 | -5.0162 | 91.4348 | 0.0196 | 0.4393 | 0.0000 | 5 |
| Delta, median agg. [blind] | 90 | 0.0243 | 0.0243 | 55.3135 | 4.8727 | 98.7691 | 0.0207 | 0.3933 | 0.0000 | 5 |
| Delta, OOD-kernel agg. [blind] | 90 | 0.0206 | 0.0206 | 31.8042 | -22.8315 | 77.6957 | 0.0181 | 0.4642 | 0.0000 | 5 |
| Generic potential [blind] | 90 | 0.0202 | 0.0202 | 29.1951 | -16.8396 | 67.8002 | 0.0158 | 0.5385 | 0.0000 | 5 |
| Isolated-speed baseline [aware] | 90 | 0.0156 | 0.0156 | – | – | – | -0.0156 | – | 1.0000 | 5 |
| Absolute response [aware] | 90 | 0.0156 | 0.0162 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Delta, uniform agg. [aware] | 90 | 0.0226 | 0.0228 | 44.6193 | -1.9130 | 84.8550 | 0.0177 | 0.5302 | 0.0000 | 5 |
| Delta, median agg. [aware] | 90 | 0.0140 | 0.0243 | -10.1998 | -46.6073 | 19.8245 | 0.0071 | 0.7018 | 0.0000 | 5 |
| Delta, OOD-kernel agg. [aware] | 90 | 0.0202 | 0.0206 | 29.1940 | -30.7722 | 91.3554 | 0.0161 | 0.4635 | 0.2000 | 5 |
| Generic potential [aware] | 90 | 0.0241 | 0.0202 | 53.8918 | 2.4415 | 96.3649 | 0.0199 | 0.5208 | 0.0000 | 5 |

#### Bidirectional, MPI time
| method | n | log MAE | artifact log MAE | Δ% | CI lo | CI hi | bias | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 90 | 0.1030 | 0.1030 | – | – | – | -0.1030 | – | 1.0000 | 29 |
| Absolute response [blind] | 90 | 0.0942 | 0.0942 | -8.5292 | -13.6802 | -3.0764 | -0.0776 | 0.0583 | 0.7241 | 29 |
| Blind absolute + fixed socket offset (JMs) [blind] | 90 | 0.0941 | – | -8.6717 | -13.3827 | -3.7676 | -0.0802 | 0.0772 | 0.7586 | 29 |
| Blind absolute + fixed socket offset (all apps) [blind] | 90 | 0.0941 | – | -8.6863 | -12.2387 | -5.3921 | -0.0879 | 0.2514 | 0.8621 | 29 |
| Delta, uniform agg. [blind] | 90 | 0.1457 | 0.1457 | 41.4438 | -6.7021 | 85.2566 | 0.1279 | 0.2495 | 0.0000 | 29 |
| Delta, median agg. [blind] | 90 | 0.1190 | 0.1190 | 15.5569 | -31.4392 | 54.1814 | 0.0968 | 0.2643 | 0.0000 | 29 |
| Delta, OOD-kernel agg. [blind] | 90 | 0.1258 | 0.1258 | 22.1459 | -22.1972 | 65.1485 | 0.1170 | 0.3174 | 0.0000 | 29 |
| Generic potential [blind] | 90 | 0.1472 | 0.1472 | 42.9381 | -5.4684 | 86.7255 | 0.1231 | 0.2467 | 0.0000 | 29 |
| Joint [blind] | 90 | 0.1004 | 0.1004 | -2.5358 | -5.3577 | 0.2857 | -0.0962 | -0.1537 | 1.0000 | 29 |
| Joint, calibrated [blind] | 90 | 0.0940 | 0.0940 | -8.7734 | -29.5827 | 11.5341 | -0.0183 | 0.2076 | 0.1034 | 29 |
| Bidir. profile [blind] | 90 | 0.0966 | 0.0966 | -6.2324 | -12.2898 | -0.5661 | -0.0832 | -0.0364 | 0.7586 | 29 |
| Bidir. profile, calibrated [blind] | 90 | 0.0953 | 0.0953 | -7.4893 | -29.4882 | 14.0591 | -0.0086 | 0.2094 | 0.0345 | 29 |
| Bidir. context [blind] | 90 | 0.1032 | 0.1032 | 0.1410 | -0.1779 | 0.5671 | -0.1026 | -0.1416 | 1.0000 | 29 |
| Bidir. context, calibrated [blind] | 90 | 0.0938 | 0.0938 | -8.9637 | -24.7277 | 6.3320 | -0.0478 | 0.2121 | 0.3103 | 29 |
| Wall-clock aux (ablation), cal. [blind] | 90 | 0.0950 | 0.0950 | -7.7942 | -29.9032 | 13.8838 | -0.0087 | 0.2201 | 0.0345 | 29 |
| Forward-only tokens (ablation), cal. [blind] | 90 | 0.0958 | 0.0958 | -6.9801 | -13.4194 | -0.6457 | -0.0825 | 0.0132 | 0.6897 | 29 |
| Isolated-speed baseline [aware] | 90 | 0.1030 | 0.1030 | – | – | – | -0.1030 | – | 1.0000 | 29 |
| Absolute response [aware] | 90 | 0.1030 | 0.0942 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | – | 1.0000 | 29 |
| Delta, uniform agg. [aware] | 90 | 0.1709 | 0.1457 | 65.8766 | 11.0150 | 115.1580 | 0.1458 | 0.1518 | 0.0000 | 29 |
| Delta, median agg. [aware] | 90 | 0.1201 | 0.1190 | 16.5478 | -27.5722 | 57.3177 | 0.0823 | 0.2719 | 0.0000 | 29 |
| Delta, OOD-kernel agg. [aware] | 90 | 0.1477 | 0.1258 | 43.3949 | -5.4601 | 91.4298 | 0.1393 | 0.2150 | 0.0000 | 29 |
| Generic potential [aware] | 90 | 0.1783 | 0.1472 | 73.0852 | 17.2279 | 122.2356 | 0.1569 | 0.1386 | 0.0000 | 29 |
| Joint [aware] | 90 | 0.1006 | 0.1004 | -2.3712 | -3.7289 | -1.1096 | -0.0997 | 0.2503 | 1.0000 | 29 |
| Joint, calibrated [aware] | 90 | 0.0960 | 0.0940 | -6.7730 | -10.4071 | -3.2430 | -0.0923 | 0.3881 | 1.0000 | 29 |
| Bidir. profile [aware] | 90 | 0.1012 | 0.0966 | -1.7968 | -2.9821 | -0.7600 | -0.1004 | 0.2559 | 1.0000 | 29 |
| Bidir. profile, calibrated [aware] | 90 | 0.0967 | 0.0953 | -6.1749 | -9.6519 | -2.9615 | -0.0932 | 0.4042 | 1.0000 | 29 |
| Bidir. context [aware] | 90 | 0.1030 | 0.1032 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | – | 1.0000 | 29 |
| Bidir. context, calibrated [aware] | 90 | 0.1030 | 0.0938 | 0.0000 | 0.0000 | 0.0000 | -0.1030 | – | 1.0000 | 29 |
| Wall-clock aux (ablation), cal. [aware] | 90 | 0.0981 | 0.0950 | -4.8063 | -7.6879 | -2.1122 | -0.0952 | 0.3808 | 1.0000 | 29 |
| Forward-only tokens (ablation), cal. [aware] | 90 | 0.1011 | 0.0958 | -1.8916 | -24.1781 | 19.9018 | 0.0104 | 0.2840 | 0.0000 | 29 |

#### Bidirectional, total time
| method | n | log MAE | artifact log MAE | Δ% | CI lo | CI hi | bias | Spearman | FB@1.05 | harmful |
|---|---|---|---|---|---|---|---|---|---|---|
| Isolated-speed baseline [blind] | 90 | 0.0156 | 0.0156 | – | – | – | -0.0156 | – | 1.0000 | 5 |
| Absolute response [blind] | 90 | 0.0162 | 0.0162 | 3.6121 | -6.4172 | 13.7406 | -0.0127 | 0.0087 | 1.0000 | 5 |
| Blind absolute + fixed socket offset (JMs) [blind] | 90 | 0.0160 | – | 2.2673 | -7.4349 | 12.0055 | -0.0129 | 0.0091 | 1.0000 | 5 |
| Blind absolute + fixed socket offset (all apps) [blind] | 90 | 0.0157 | – | 0.5784 | -7.8012 | 8.6784 | -0.0134 | 0.0444 | 1.0000 | 5 |
| Delta, uniform agg. [blind] | 90 | 0.0199 | 0.0199 | 27.0771 | -21.7902 | 67.0664 | 0.0152 | 0.5067 | 0.0000 | 5 |
| Delta, median agg. [blind] | 90 | 0.0209 | 0.0209 | 33.8214 | -13.2874 | 72.1159 | 0.0160 | 0.5101 | 0.0000 | 5 |
| Delta, OOD-kernel agg. [blind] | 90 | 0.0170 | 0.0170 | 8.8472 | -42.5405 | 54.6793 | 0.0128 | 0.4283 | 0.4000 | 5 |
| Generic potential [blind] | 90 | 0.0202 | 0.0202 | 29.1951 | -16.8396 | 67.8002 | 0.0158 | 0.5385 | 0.0000 | 5 |
| Joint [blind] | 90 | 0.0156 | 0.0156 | 0.0513 | 0.0000 | 0.1538 | -0.0156 | -0.1139 | 1.0000 | 5 |
| Joint, calibrated [blind] | 90 | 0.0139 | 0.0139 | -10.9833 | -20.6748 | -2.5929 | -0.0128 | 0.3780 | 1.0000 | 5 |
| Bidir. profile [blind] | 90 | 0.0158 | 0.0158 | 1.2168 | 0.0409 | 2.7132 | -0.0154 | -0.1322 | 1.0000 | 5 |
| Bidir. profile, calibrated [blind] | 90 | 0.0147 | 0.0147 | -6.1530 | -14.7287 | 1.7569 | -0.0121 | 0.3049 | 1.0000 | 5 |
| Bidir. context [blind] | 90 | 0.0156 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Bidir. context, calibrated [blind] | 90 | 0.0154 | 0.0154 | -1.7930 | -4.5384 | 0.0000 | -0.0154 | 0.1798 | 1.0000 | 5 |
| Wall-clock aux (ablation), cal. [blind] | 90 | 0.0143 | 0.0143 | -8.3091 | -17.4566 | -0.0784 | -0.0123 | 0.3731 | 1.0000 | 5 |
| Forward-only tokens (ablation), cal. [blind] | 90 | 0.0175 | 0.0175 | 11.6428 | -3.0807 | 26.2405 | -0.0099 | -0.0416 | 1.0000 | 5 |
| Isolated-speed baseline [aware] | 90 | 0.0156 | 0.0156 | – | – | – | -0.0156 | – | 1.0000 | 5 |
| Absolute response [aware] | 90 | 0.0156 | 0.0162 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Delta, uniform agg. [aware] | 90 | 0.0226 | 0.0199 | 44.4741 | -3.9860 | 84.9139 | 0.0167 | 0.5400 | 0.2000 | 5 |
| Delta, median agg. [aware] | 90 | 0.0157 | 0.0209 | 0.2901 | -38.1461 | 33.2325 | 0.0091 | 0.6606 | 0.0000 | 5 |
| Delta, OOD-kernel agg. [aware] | 90 | 0.0187 | 0.0170 | 19.8194 | -39.6982 | 80.1626 | 0.0160 | 0.4977 | 0.0000 | 5 |
| Generic potential [aware] | 90 | 0.0241 | 0.0202 | 53.8919 | 2.4415 | 96.3649 | 0.0199 | 0.5208 | 0.0000 | 5 |
| Joint [aware] | 90 | 0.0156 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Joint, calibrated [aware] | 90 | 0.0155 | 0.0139 | -0.8591 | -2.0530 | 0.0000 | -0.0155 | 0.2177 | 1.0000 | 5 |
| Bidir. profile [aware] | 90 | 0.0156 | 0.0158 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Bidir. profile, calibrated [aware] | 90 | 0.0155 | 0.0147 | -0.9749 | -2.3175 | 0.0000 | -0.0155 | 0.2177 | 1.0000 | 5 |
| Bidir. context [aware] | 90 | 0.0156 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Bidir. context, calibrated [aware] | 90 | 0.0156 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | -0.0156 | – | 1.0000 | 5 |
| Wall-clock aux (ablation), cal. [aware] | 90 | 0.0155 | 0.0143 | -0.9948 | -2.4041 | 0.0000 | -0.0155 | 0.2177 | 1.0000 | 5 |
| Forward-only tokens (ablation), cal. [aware] | 90 | 0.0962 | 0.0175 | 515.5570 | 440.7715 | 577.1211 | 0.0955 | 0.5495 | 0.0000 | 5 |
- B3 best calibrated (BD-MPI, lowest aware resolved log MAE): Joint, calibrated

### B3. Per-victim aware/blind log-MAE ratio (socket-resolved cells; < 1 = aware better)

#### Bidirectional, MPI time: Absolute response
| victim | aware log MAE | blind log MAE | ratio | ratio s_V=0 | ratio s_V=1 |
|---|---|---|---|---|---|
| AMG2023 | 0.1867 | 0.1739 | 1.0742 | 1.1129 | 1.0020 |
| Beatnik | 0.2039 | 0.1777 | 1.1476 | 1.1542 | 1.1406 |
| FIESTA | 0.1011 | 0.1119 | 0.9040 | 1.0391 | 0.0000 |
| Kripke | 0.0503 | 0.0321 | 1.5653 | 1.9679 | 1.3007 |
| Laghos | 0.2222 | 0.2147 | 1.0345 | 1.0367 | 1.0332 |
| LAMMPS | 0.0609 | 0.0695 | 0.8762 | 1.0203 | 0.5196 |
| MiniFE | 0.1188 | 0.1181 | 1.0059 | 1.0392 | 0.9612 |
| miniVite | 0.0927 | 0.0777 | 1.1918 | 1.2238 | 1.1569 |
| Quicksilver | 0.0147 | 0.0244 | 0.6011 | 1.8284 | 0.0182 |
| TriC | 0.0762 | 0.0578 | 1.3178 | 1.3375 | 1.3003 |

#### Bidirectional, MPI time: Joint, calibrated
| victim | aware log MAE | blind log MAE | ratio | ratio s_V=0 | ratio s_V=1 |
|---|---|---|---|---|---|
| AMG2023 | 0.1759 | 0.1720 | 1.0227 | 1.4427 | 0.6537 |
| Beatnik | 0.1889 | 0.1038 | 1.8207 | 1.8671 | 1.7758 |
| FIESTA | 0.1007 | 0.1011 | 0.9957 | 1.0257 | 0.1274 |
| Kripke | 0.0382 | 0.0723 | 0.5288 | 0.5456 | 0.5175 |
| Laghos | 0.2133 | 0.2017 | 1.0577 | 1.0523 | 1.0611 |
| LAMMPS | 0.0601 | 0.0731 | 0.8228 | 1.0027 | 0.4292 |
| MiniFE | 0.1177 | 0.1789 | 0.6581 | 0.8449 | 0.5050 |
| miniVite | 0.0875 | 0.0738 | 1.1857 | 1.1701 | 1.2037 |
| Quicksilver | 0.0148 | 0.0147 | 1.0054 | 1.1565 | 0.1893 |
| TriC | 0.0710 | 0.0430 | 1.6528 | 1.6797 | 1.6305 |

#### One-way, MPI time: Absolute response
| victim | aware log MAE | blind log MAE | ratio | ratio s_V=0 | ratio s_V=1 |
|---|---|---|---|---|---|
| AMG2023 | 0.1867 | 0.1739 | 1.0742 | 1.1129 | 1.0020 |
| Beatnik | 0.2039 | 0.1777 | 1.1476 | 1.1542 | 1.1406 |
| FIESTA | 0.1011 | 0.1119 | 0.9040 | 1.0391 | 0.0000 |
| Kripke | 0.0503 | 0.0321 | 1.5653 | 1.9679 | 1.3007 |
| Laghos | 0.2222 | 0.2147 | 1.0345 | 1.0367 | 1.0332 |
| LAMMPS | 0.0609 | 0.0695 | 0.8762 | 1.0203 | 0.5196 |
| MiniFE | 0.1188 | 0.1181 | 1.0059 | 1.0392 | 0.9612 |
| miniVite | 0.0927 | 0.0777 | 1.1918 | 1.2238 | 1.1569 |
| Quicksilver | 0.0147 | 0.0244 | 0.6011 | 1.8284 | 0.0182 |
| TriC | 0.0762 | 0.0578 | 1.3178 | 1.3375 | 1.3003 |
- 1W-MPI aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 2.78e-16
- 1W-TT aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 5.13e-16
- BD-MPI aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 5.03e-16
- BD-TT aware: pipeline's aggregated prediction vs mine (max |Δ log|) = 4.79e-16

### B5. CV (App-Inhibitor crossed CV of the selected recipe) vs App-App error

#### One-way, MPI time [blind]: CV top pick = Absolute response; Spearman(CV, App-App) resolved +0.50, rep-avg +0.50
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Absolute response | 0.1562 | 0.1035 | 0.0942 | 1 | 1 |
| Generic potential | 0.1617 | 0.1504 | 0.1472 | 3 | 2 |
| Delta, uniform agg. | 0.1580 | 0.1546 | 0.1522 | 2 | 3 |

#### One-way, MPI time [aware]: CV top pick = Delta, median agg.; Spearman(CV, App-App) resolved +0.50, rep-avg +0.50
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Absolute response | 0.1891 | 0.1102 | 0.1030 | 2 | 1 |
| Generic potential | 0.1959 | 0.1719 | 0.1783 | 3 | 3 |
| Delta, median agg. | 0.1795 | 0.1282 | 0.1289 | 1 | 2 |

#### One-way, total time [blind]: CV top pick = Delta, OOD-kernel agg.; Spearman(CV, App-App) resolved -1.00, rep-avg -1.00
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Absolute response | 0.0318 | 0.0161 | 0.0162 | 3 | 1 |
| Generic potential | 0.0214 | 0.0206 | 0.0202 | 2 | 2 |
| Delta, OOD-kernel agg. | 0.0206 | 0.0207 | 0.0206 | 1 | 3 |

#### One-way, total time [aware]: CV top pick = Absolute response; Spearman(CV, App-App) resolved -0.50, rep-avg -0.50
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Absolute response | 0.0253 | 0.0157 | 0.0156 | 1 | 2 |
| Generic potential | 0.0267 | 0.0243 | 0.0241 | 2 | 3 |
| Delta, median agg. | 0.0273 | 0.0149 | 0.0140 | 3 | 1 |

#### Bidirectional, MPI time [blind]: CV top pick = Forward-only tokens (ablation), cal.; Spearman(CV, App-App) resolved +0.45, rep-avg +0.36
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Delta, uniform agg. | 0.1690 | 0.1486 | 0.1457 | 8 | 7 |
| Generic potential | 0.1617 | 0.1504 | 0.1472 | 7 | 8 |
| Absolute response | 0.1562 | 0.1035 | 0.0942 | 6 | 5 |
| Joint, calibrated | 0.1481 | 0.1021 | 0.0940 | 4 | 1 |
| Bidir. profile, calibrated | 0.1479 | 0.1028 | 0.0953 | 2 | 4 |
| Bidir. context, calibrated | 0.1481 | 0.1025 | 0.0938 | 3 | 2 |
| Wall-clock aux (ablation), cal. | 0.1513 | 0.1026 | 0.0950 | 5 | 3 |
| Forward-only tokens (ablation), cal. | 0.1445 | 0.1057 | 0.0958 | 1 | 6 |

#### Bidirectional, MPI time [aware]: CV top pick = Forward-only tokens (ablation), cal.; Spearman(CV, App-App) resolved +0.80, rep-avg +0.80
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Delta, uniform agg. | 0.2145 | 0.1675 | 0.1709 | 8 | 7 |
| Generic potential | 0.1959 | 0.1719 | 0.1783 | 7 | 8 |
| Absolute response | 0.1891 | 0.1102 | 0.1030 | 6 | 5 |
| Joint, calibrated | 0.1631 | 0.1044 | 0.0960 | 2 | 1 |
| Bidir. profile, calibrated | 0.1660 | 0.1049 | 0.0967 | 4 | 2 |
| Bidir. context, calibrated | 0.1676 | 0.1102 | 0.1030 | 5 | 5 |
| Wall-clock aux (ablation), cal. | 0.1645 | 0.1060 | 0.0981 | 3 | 3 |
| Forward-only tokens (ablation), cal. | 0.1612 | 0.1091 | 0.1011 | 1 | 4 |

#### Bidirectional, total time [blind]: CV top pick = Wall-clock aux (ablation), cal.; Spearman(CV, App-App) resolved +0.60, rep-avg +0.60
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Delta, uniform agg. | 0.0345 | 0.0201 | 0.0199 | 8 | 7 |
| Generic potential | 0.0214 | 0.0206 | 0.0202 | 4 | 8 |
| Absolute response | 0.0318 | 0.0161 | 0.0162 | 7 | 5 |
| Joint, calibrated | 0.0203 | 0.0142 | 0.0139 | 3 | 1 |
| Bidir. profile, calibrated | 0.0216 | 0.0149 | 0.0147 | 5 | 3 |
| Bidir. context, calibrated | 0.0200 | 0.0155 | 0.0154 | 2 | 4 |
| Wall-clock aux (ablation), cal. | 0.0184 | 0.0146 | 0.0143 | 1 | 2 |
| Forward-only tokens (ablation), cal. | 0.0228 | 0.0175 | 0.0175 | 6 | 6 |

#### Bidirectional, total time [aware]: CV top pick = Forward-only tokens (ablation), cal.; Spearman(CV, App-App) resolved +0.05, rep-avg +0.05
| method | CV log MAE | App-App resolved log MAE | App-App rep-avg log MAE | CV rank | App-App rank (resolved) |
|---|---|---|---|---|---|
| Delta, uniform agg. | 0.0334 | 0.0233 | 0.0226 | 8 | 6 |
| Generic potential | 0.0267 | 0.0243 | 0.0241 | 7 | 7 |
| Absolute response | 0.0253 | 0.0157 | 0.0156 | 6 | 4 |
| Joint, calibrated | 0.0239 | 0.0156 | 0.0155 | 3 | 3 |
| Bidir. profile, calibrated | 0.0248 | 0.0156 | 0.0155 | 5 | 2 |
| Bidir. context, calibrated | 0.0236 | 0.0157 | 0.0156 | 2 | 4 |
| Wall-clock aux (ablation), cal. | 0.0242 | 0.0156 | 0.0155 | 4 | 1 |
| Forward-only tokens (ablation), cal. | 0.0210 | 0.0956 | 0.0962 | 1 | 8 |
