## S1. Seed batch cost plan (2026-11-09)

### Timing model
All jobs run effectively single-threaded (user ≈ real). The machine has 8 cores and 31 GB, so ~7 parallel workers.

| job | measured | model (s, N seeds) |
|---|---|---|
| BD full data, blind | P0: 30 seeds in 441 s alone; 5 seeds ≈ 130 s under load | 60 + 13N |
| BD full data, aware / E1 / D1 / D2 | 5 seeds ≈ 160–180 s under load | 1.25 × (60 + 13N) |
| 1W full data, blind | P1: 10 seeds in 566 s alone; 5 seeds ≈ 460 s under load | 140 + 43N |
| 1W full data, aware / E1 / D1 | not measured separately | 1.2 × (140 + 43N) (assumed) |
| BD restricted task (one_known + zero_shot) | B4: ≈ 215 s at 5 seeds under load | 60 + 31N; 47 succeeding tasks per pipeline (23 fail in seconds) |
| 1W restricted task | **not measured** | assumed 1.6 × 1W full; 36 (MPI) + 21 (TT) succeeding tasks at sizes 4–8 (artifact) |

Training is cheap (BD: 30 seeds × 8 variants in 69 s). Per-seed evaluation (calibration and inference) dominates.

### Component cost (CPU-h; wall time at ~7 workers)

| component | runs / tasks | N = 30 | N = 50 | N = 100 |
|---|---|---|---|---|
| (a) full data: 4 pipelines × {blind, aware-CV, E1, D1} + D2 | 17 runs | 5.0 (~0.8 h) | 8.0 (~1.2 h) | 15.4 (~2.5 h) |
| (b) restricted, blind: BD MPI + TT | 94 tasks | 26 | 42 | 84 |
| (b) restricted, blind: 1W MPI + TT | ~57 tasks | 36 | 58 | 116 |
| (c) restricted, aware + D1: BD, 20 C4-matched cells only | 80 tasks | 25 | 41 | 81 |
| (c) restricted, aware + D1: BD, all 47 cells | 188 tasks | 59 | 97 | 193 |
| (c) restricted, aware + D1: 1W too | ~114 tasks | ~80 | ~130 | ~260 |
| (d) C7 ablations and false-benign | none extra | 0 | 0 | 0 |

C7 contrasts (bwall, bfwd, bctx, bprof, joint) and FB come from the same full-data BD predictions, so they cost nothing extra.

The 1W restricted task cost is unmeasured. The first step after a go is to time one 1W restricted task (~10 min).

Disk: P0 at 30 seeds uses 34 MB; a B4 root at 5 seeds uses 0.9 GB. That's fine (611 GB free).

### Recommendation (≈ 100 CPU-h, ≈ 15 h wall)
1. (a) at N = 100: 15 CPU-h, ~2.5 h. Headline claims live here and the near-init variants need the seeds.
2. (b) at N = 30 (BD + 1W): 62 CPU-h, ~9 h.
3. (c) BD only, 20 C4 cells, N = 30: 25 CPU-h, ~3.6 h.
4. (d) free.
