# Verified findings (as of 2026-11-07)

All numbers below were verified from code, logs or data. Full tables are in
`scripts/dissertation_figs/output/{batch1_numbers,placement_numbers,figs_batch1_numbers,batch2_methods,batch2_numbers,batch2_panels,socket_numbers}.md`.

## 1. Measurement setup (Dane)

- **Hardware.** This comes from external sources, not the logs: the [LLNL reference systems page](https://software.llnl.gov/benchmarks/91_reference_systems/reference_systems.html)
  and [TOP500](https://www.top500.org/system/180228).
  - 2× Intel Xeon Platinum 8480+ (Sapphire Rapids), 56 cores, 2.0 GHz; 112 cores per node.
  - 256 GiB DDR5-4800.
  - Cornelis Omni-Path.

  The logs do show hosts `dane###`, PUs 0–223 (SMT-2) and 0 GPUs.
- **Software:** GCC 10.3.1 and Open MPI 4.1.2 (via modules; it replaces mvapich2 2.3.7), plus a Spack env "beatnik"
  whose MPI override is unverified. mpiP 3.5.0 via `LD_PRELOAD`, timer PMPI_Wtime.
- **Allocation:** one sbatch job per run, `--nodes 8 --ntasks 896 --ntasks-per-node 112 --mem 240G --distribution block:cyclic -p pbatch`,
  so the whole node is allocated.
- **Each job step:** `srun -n 448 --ntasks-per-node 56 --mem 119G --distribution=block:block --mpibind=on,v`.

  | Campaign | Launch |
  |---|---|
  | App-App | Two backgrounded sruns in the same allocation, then `wait`. No `--exclusive`. |
  | App-Inhibitor | Adds `--exclusive --cpu-bind=cores`. The inhibitor is launched first in the background and the app runs in the foreground. Replicates use a job array (0–3). |
  | Isolated app | One srun; the other socket is idle. |
  | Isolated inhibitor | One srun with `-r 360`. |
  | 1-node | `srun --exclusive -n 56`. |

- **Placement:** mpibind binds each job to **one whole socket**, one PU per core. So the jobs share the NIC, fabric,
  UPI link and node memory capacity, but **not** cores or the L3 (and very likely not local memory bandwidth).
- **Measured node architecture.** From the socket pilot on dane[8-15], job 8518293, 2026-11-07, reported by
  `coScheduling-experiment-scripts-dane`:
  - 2× Xeon Platinum 8480+, 56 cores per socket, SMT-2, 224 PUs.
  - **PU numbering:** socket 0 = 0–55 (thread 0) and 112–167 (thread 1); socket 1 = 56–111 and 168–223.
    Socket = 0 if `PU mod 112 < 56`, else 1. This confirms the mapping used in `analysis_placement.py`.
  - **Sub-NUMA clustering is on:** 8 NUMA domains, 4 per socket, each with 14 cores and about 32 GB:

    | NUMA node | PUs | Socket |
    |---|---|---:|
    | node0 | 0–13, 112–125 | 0 |
    | node1 | 14–27, 126–139 | 0 |
    | node2 | 28–41, 140–153 | 0 |
    | node3 | 42–55, 154–167 | 0 |
    | node4 | 56–69, 168–181 | 1 |
    | node5 | 70–83, 182–195 | 1 |
    | node6 | 84–97, 196–209 | 1 |
    | node7 | 98–111, 210–223 | 1 |

    **Never use the NUMA id as the socket id.** A one-socket job spans 4 NUMA domains, so 4 NUMA nodes per run in
    placement checks is expected.
  - Modules: gcc/10.3.1-magic, openmpi/4.1.2, cmake/3.23.1.
- **Launch behavior (measured):**
  - June-style `--mpibind=on,v` with 56 tasks per node puts local rank i on PU 112+i, with `OMP_NUM_THREADS=1` and
    `OMP_PLACES=threads`.
  - A 56-task-per-node srun step **without `--whole`** gets only socket 0's cpuset (0–55, 112–167). So every isolated
    baseline was necessarily on socket 0. That is structural, not chance.
  - In co-runs, the first step to start gets socket 0 and the second gets socket 1. Socket is therefore a proxy for
    **launch order**.
    - Caution: the mpiP start timestamps (1 s resolution) show both jobs starting in the same second, so any launch
      offset is under about 1 s.
    - **Checked against App-Inhibitor mpiP start times (MPI_Init, 1 s resolution, 11,478 runs):**

      | Start offset | Count |
      |---|---:|
      | Same second | 10,140 |
      | App 1 s **earlier** | 536, of which 469 have the app on socket **1** |
      | App 1 s **later** | 675, of which 582 have the app on socket **0** |

      So the job that reaches MPI_Init first tends to be on socket 1, which is the **opposite** of
      "first launched gets socket 0". This may be because the step that wins the allocation race starts MPI_Init
      later, or socket-1 startup may simply be faster.
    - Either way, the two orders are not separable from June/Sep logs alone. Treat "socket effect" and "start-order
      effect" in co-runs as confounded.
    - The new socket-pinned **isolated** runs do **not** resolve this. Their socket-0 and socket-1 steps run
      **sequentially** in one job, each explicitly pinned, so there is no race. They measure (a) the pure socket effect
      on *isolated* baselines and (b) a sequential first/second-step effect through the even/odd alternation (warm
      caches, Lustre, node state). They say nothing about co-run socket assignment.
    - Testing the co-run race needs a new experiment, suggested to Akhil and not built as of 2026-11-07. Per step it
      would record the srun launch epoch, each rank's exec time from a wrapper before MPI_Init, and the resulting
      cpuset, with the launch order deliberately varied.
- **mpiP precision:** the per-task AppTime and MPITime tables use **3 significant digits**. That is 1 s for 100–999 s
  runs (about 0.3%). The CSV values average this over 448 tasks.
- **Inputs:** `run_configs/8_nodes.json`. The 1-node subset is Beatnik, FIESTA, LAMMPS, MiniFE, miniVite and TriC.
  TriC is `apps/tric` (tricount).

## 2. Campaign counts (`processed_data`)

| Campaign | Configs | Replicates | Runs |
|---|---|---|---:|
| Isolated apps | 10 (8 nodes) + 6 (1 node) | ×4 | 64 |
| Isolated inhibitors | 287 | ×4, minus 2 configs with 3 | 1,146 |
| App-Inhibitor | 2,870 | ×4, minus 1 Laghos config with 2 | 11,478 |
| App-App | 55 (8 nodes) + 21 (1 node), self-pairs included | ×4 | 304 |

**Inhibitor grid** (three blocks, not a full factorial):

| Block | Message size × wait × sparsity | Configs |
|---|---|---:|
| base | 8 × 6 × 5 | 240 |
| extended | 4 × 4 × 2 | 32 |
| missing-range | 5 × 3 × 1 | 15 |

That gives 16 message sizes (20 B–4 MB), 12 waits (0–1 s) and 8 sparsities (0.002237136 = 1 peer, up to 1.0).

**Inhibitor loop** (`networkInhib.cpp`):
- Per iteration: Isend to k fixed peers, Irecv, Waitall, sleep(wait), Barrier, then an Allreduce(MAX) stop flag when
  `-r` is set.
- k = round(s·447) is the same for every rank. The peer set is drawn once (LCG seeded 12345 + rank).

## 3. ⚠ Data-quality issues (Akhil: disclose, don't re-fit; socket-pinned isolated re-runs approved 2026-11-07)

1. **Inhibitor runtime bug.** `get_max_runtime()` in `scripts/generate_coscheduled_inhib_scripts_multinode.py`
   returns the first CSV row, which is the 1-node runtime for the six 1-node apps. Effects:
   - `-r` was too short for Beatnik (371 vs 398 s), FIESTA (299/359), LAMMPS (266/355) and TriC (281/378).
   - Median coverage of the app's run by the inhibitor: TriC 0.68, LAMMPS 0.75, FIESTA 0.84, Beatnik 0.86.
   - Laghos also has 0.86 coverage, because its own co-run times are long.
   - Only App-Inhibitor labels are affected.
2. **Socket confound.**
   - All isolated baselines ran on socket 0. This is structural: a step without `--whole` only gets socket 0. PUs
     0–55 and 112–167 are SMT siblings on the same cores (confirmed by measurement).
   - Socket is confounded with launch order (see the measured launch behavior in section 1).
   - In co-runs, which job gets which socket is a launch race.
   - Within the same config, socket 1 has lower T_MPI: −12.2% in App-Inhibitor runs and −6.6% in App-App runs.
   - The effect is concentrated in AMG2023, FIESTA, LAMMPS, Laghos and MiniFE.
   - Socket scans are cached in `output/socket_*.csv`.
   - **Partly resolved by the pinned isolated runs (§7):** socket 1 is also faster *in isolation* (pooled −0.13 log
     T_MPI). That is baseline bias for AMG2023 and Quicksilver, and contention-only for FIESTA. The chapter adopts
     the **JMs** socket-matched sensitivity.
3. **Timing.** Baselines and App-App runs are from June 2026; App-Inhibitor runs are from Aug–Sep 2026. Drift between
   them is not quantified. The Nov 2026 pinned baselines differ from June by +0.109 log T_MPI (§7), so drift is real
   at that scale.

## 4. Characterization (8 nodes; non-self n = 90)

| Target | Median | p75 | p90 | Max | Floor fraction |
|---|---:|---:|---:|---:|---:|
| MPI-time slowdown | 1.051 | 1.131 | 1.311 | 2.209 | 20% |
| Total-time slowdown | 1.001 | 1.025 | 1.043 | 1.192 | 40% |

- Minimum raw ratio: 0.831 for MPI time; 0.687 for App-Inhibitor runs.
- **Replicate CV of isolated T_MPI:** median 4.3%. Laghos 27%, LAMMPS 16%, FIESTA 8.7%.
- **Replicate CV of T_app:** median 0.98%; Laghos 4.8%.
- **Isolated inhibitor CV:** median 0.63%, maximum 103%.
- **Two-way fixed effects on log MPI slowdown:** R² is 0.24 for victim only, 0.30 for aggressor only and 0.56
  additive.
  - Dominant aggressor: TriC (×1.23).
  - Most susceptible victim: **Laghos** (×1.17), then Beatnik (×1.11, also an aggressor at ×1.13).
- **Asymmetry:** Spearman of s_A(B) vs s_B(A) is 0.41.
- **1 node vs 8 nodes:** the magnitudes are similar, but the per-pair Spearman is −0.44, so different pairs degrade.
- **Coverage:** in the 3 independent selection features, 5 of 10 apps lie outside the inhibitor-grid hull. Quicksilver
  is far from every inhibitor. The 4 features are collinear: log byte rate ≈ log message rate + log mean size.

## 5. Modeling (artifact "Filtered-Inhibitor Pipeline Comparison")

- **Artifact:** https://claude.ai/artifact/BX6enpc1oPW4edne4tpdFn. Its embedded DATA object is saved at
  `scripts/dissertation_figs/output/artifact_data_20261017.json`. Fields: `runs` {base, baseSnap, totalTime, bidir,
  bidirTT} → `random_split` (metrics, `diff_vs_c1`, `ci`), `per_victim`, `curve`, `matched`, `cv`, `coverage`; and
  `common_cells`.
- **Run-name trap:** the artifact's `base` run ("filteredInhibs") used the **legacy** snapshot. The canonical one-way
  MPI run is `baseSnap` (the artifact's "control").
- **Reproducibility.** The full-data fits are deterministic:
  - the bidirectional README results tables equal the artifact's `bidir`;
  - `..._biDirectional_totalTime/outputs/full_comparison/` equals `bidirTT`, with local predictions, pair_metrics and
    ablation_summary available.

  For delta models, the local `..._totalTime/outputs/filtered_fit_*` runs do **not** match the artifact, because those
  used different recipes.
- **Panel:**
  - 9 inhibitors, 90 rows; 16 floor rows for MPI time and 27 for total time.
  - Facility location with τ = 0.5 and budget 10. It stops at 9 because Kripke and LAMMPS share a nearest inhibitor.
  - Members (-m, -w µs, -s): (2e6, 1e6, 0.3), (2e4, 1e5, 0.1), (200, 2e5, 0.002237136), (2.5e6, 2e5, 0.002237136),
    (2.5e6, 6e5, 0.002237136), (4e5, 1e6, 0.5), (4e4, 1e6, 0.1), (4e4, 1e6, 0.7), (500, 5e3, 0.05).
- **Headline results on the canonical snapshot:**
  - MPI-time absolute response: −8.5% [−14.3, −3.1].
  - Calibrated joint and context models: about −9%, CIs include 0.
  - Total-time joint calibrated: −11.0% [−21.3, −2.9].
  - One-known matched cells: absolute +9.3%, 20/20 wins.
  - Zero-shot: absolute −4.0% pooled (bidirectional).
  - Coverage at sizes 4–8: 47/50 for both bidirectional runs; 36, 21 and 30 for one-way MPI, one-way total and legacy.
- **Corrections already sent to the plan:**
  - "CV picks the wrong model" holds only on the legacy snapshot.
  - Delta's +27–135% figure is legacy-inclusive; on canonical data it is +9–55%, mostly with CIs including 0.
  - The ablations are not all within ±0.004 (fingerprints direct +0.0066 [+0.0006, +0.0120]).
  - The generic potential is an unfactorized head, not a larger one.
  - The absolute model has no b(e_V) term.
  - Calibration's false-benign gain (0.72–0.76 → 0.03–0.31) applies to MPI time only. Total time has 5 harmful pairs,
    and false-benign is 1.0 for every non-delta method.

## 6. Socket-pinned isolated runs: design and parsing rules

Akhil approved these on 2026-11-07. The session `coScheduling-experiment-scripts-dane` generated them with
`scripts/generate_isolated_socket_pinned_multinode.py` (commits 792b60bc, b75c9aed, 1f80d993, ec8f5fb8). They cover
the 10 apps plus the 9 panel inhibitors (`-r 360`).

- **Data on this machine:** `run_data/experiments_coscheduling_isolated_socket/20261107_180934/`. This is the only
  valid experiment.
  - Ignore the pilots `20261107_150428`, `20261107_174100` and `20261107_175238` if they ever appear.
  - Layout:
    - `data/sock{0,1}/8_<app>_<r>/{mpip_profiles/*.mpiP, placement_probe.log, placement_check.txt, output.log}`
    - `data_inhib/sock{0,1}/8_inhib_<m>_<w>_d_<s>_<r>/`
    - `job_logs/<job>/{steps.txt, mpibind_reference.log, env.txt, *.out, *.err}`
    - `manifest.json`: job → run dirs, plus `socket_order`, where the first element is the socket that ran first.
    - `slurm_scripts{,_inhib}/`
  - The manifest's `"smt_index": 1` is the wrapper's *preferred* thread. The actual placement is thread 0, because the
    `--whole` cpuset is 0–111. Trust `placement_probe.log`, not the manifest.
- **Design:**
  - Each job runs a sock0 step and a sock1 step **sequentially** on the same 8 nodes. Even replicates run sock0
    first and odd replicates run sock1 first.
  - Each step is `srun --whole --mpibind=off --cpu-bind=none` plus `numactl --physcpubind=<PU> --localalloc`, with
    56 ranks/node, `OMP_NUM_THREADS=1` and `OMP_PLACES=threads`.
  - The pinned steps unset `SLURM_CPUS_PER_TASK` and `SLURM_TRES_PER_TASK`; otherwise `--cpus-per-task` makes srun
    ignore `--whole`. Dane runs Slurm 25.11.8.
  - Expected placement: sock0 = PUs 0–55 (NUMA 0–3) and sock1 = PUs 56–111 (NUMA 4–7), all on SMT thread 0.
  - These are the same physical cores June used, but June used thread 1 (PUs 112–167) via mpibind.
- **Because the steps are sequential, there is no socket race.** The design measures the pure isolated socket effect
  and the first-vs-second-step order effect. It says nothing about co-run socket assignment; that needs the co-run
  race experiment in §1, which is not built.
- **Parsing rules, implemented in `analysis_socket_pinned.py::check_run`.** A run is valid only if all of these hold:
  - no `PLACEMENT_FAILED` file;
  - at least one `.mpiP` file;
  - `placement_check.txt` contains `problems=0`;
  - 448 `PLACEMENT` lines;
  - every rank has `thread=0`;
  - the socket taken from `PU mod 112 < 56` matches the `sockN` directory.

  More rules:
  - `placement_check.txt` does not check threads. Check the per-rank `thread=` values yourself.
  - Never use `numa=` as the socket.
  - `parse_isolated` / `parse_inhibitor_isolated` raise on dirs without `.mpiP`. The script parses a temporary
    symlink tree of the valid dirs only.
  - The parser is `CoSchedulingRunsParser` in `notebooks/utils/parse_coscheduling_runs.py`. It uses relative imports,
    so put `notebooks/` on `sys.path` and import `utils.parse_coscheduling_runs`.
  - `steps.txt` has start/end epochs (`date +%s.%N`) per socket step. These are srun launch times, not MPI_Init times.

## 6b. Still pending (as of 2026-11-08)

- Isolated inhibitor data landed 2026-11-08 and S2 is done (§7b). Not yet messaged to the chapter director.
- If the `*_full_20261017` roots become available:
  - regenerate the artifact figures directly from `consolidated/predictions.csv`;
  - re-score joint MPI and the artifact-recipe delta models under JMs (S6 gap).
- Socket batch 2 (socket-aware pipelines, A1–A5 then B/C) requested by the chapter director 2026-11-08.

## 7. Socket-pinned isolated results (experiment 20261107_180934, analyzed 2026-11-07)

Full numbers: `scripts/dissertation_figs/output/socket_numbers.md`.
- **Validity:** all 80 app steps are valid: 448 ranks each, thread=0, problems=0, rc=0. The 72 inhibitor run dirs
  (the 9 panel configs × 4 reps × 2 sockets) were **empty** when analyzed, so S2 and inhibitor drift are still pending.
- **Isolated socket effect** (paired within job, sock1 − sock0):
  - Pooled log T_MPI −0.131 [−0.256, −0.043], app-cluster bootstrap; 31/40 pairs faster on sock1.
  - Pooled log T_app −0.015 [−0.027, −0.006].
  - Clear and consistent for AMG2023 (−0.20, 4/4) and Quicksilver (−0.036, 4/4).
  - About zero for Beatnik, TriC, miniVite and Kripke.
  - LAMMPS and Laghos: intermittent heavy slow runs on sock0 (LAMMPS sock0 T_MPI 17.8–62 s), so unresolved at n=4.
  - Order effect (second vs first step) is not significant.
- **Launcher/epoch drift:** new sock0 vs June isolated is +0.109 log T_MPI on average (+0.016 T_app). Never mix new
  baselines with June/Sep co-runs without saying so.
- **Co-run comparison:** the story is mixed by app.
  - AMG2023 is a baseline effect: its isolated gap matches its App-App gap.
  - FIESTA, and Kripke and MiniFE in App-Inh, show the gap mainly under contention.
- **Socket-matched slowdowns:** the recommended sensitivity is JMs, the June baseline × exp(d) for socket-1 reps,
  applied only where d's CI excludes 0. It moves App-App MPI median 1.051→1.065, mean 1.123→1.135, floor 20%→18%.
- **Re-scoring:** the chapter's model conclusions survive JM and JMs. Absolute MPI stays about −10% and TT joint
  calibrated about −10%, both CIs excluding 0.
- **Re-scoring gap:** joint MPI per-outcome predictions are not local, so it can't be re-scored.
- **Re-scoring prediction sources:**
  - MPI: `delta_response_filteredInhibs_totalTime/outputs/filtered_fit_mpi_time/directional_predictions.csv`.
    Its constant/absolute/generic predictions equal the artifact's baseSnap; its delta predictions do not.
  - TT: `..._biDirectional_totalTime/outputs/full_comparison/directional_predictions.csv`, identical to the
    artifact's bidirTT.
- **Baseline variants** (`analysis_socket_matched.py`): J = June; N0 = new sock0; NM = new, socket-matched;
  JM = June × exp(d̄) for socket-1 reps; JMs = JM with d̄ applied only where its t CI excludes 0 (AMG2023, Quicksilver).
  Raw = mean co / mean of the matched baselines over the victim's reps. SE comes from the delta method.
- **Decisions agreed with the chapter director (2026-11-07):**
  - JMs is the chapter's sensitivity variant.
  - N0/NM are reported only as "baselines must come from the same epoch and launcher". The new baselines push all
    ratios toward 1 through drift, and every model then loses to the isolated-speed baseline.
  - JM is an upper bound, inflated by the LAMMPS/Laghos tails.
- **S6 bootstrap:** re-implemented over the 45 non-self pairs (1,000 draws, seed 923). Point estimates match the
  artifact, but the CIs differ slightly (absolute J [−12.8, −3.7] vs the artifact's [−14.3, −3.1]). Quote the
  artifact CIs for current (J) numbers.
- **Figure:** `Dissertation/figures/node_sharing/socket_isolated.pdf` (full width). Per app it shows the isolated reps
  plus mean and t CI, next to the App-Inh and App-App co-run median and IQR.

## 7b. Socket-pinned isolated inhibitors (S2, analyzed 2026-11-08)

Full numbers: `socket_numbers.md` §S2 and the inhibitor block of §S3; runs in `output/socket_pinned_inhib_{runs,pairs}.csv`.
- **Validity:** 72/72 valid (9 panel configs × 4 reps × 2 sockets), all thread 0, rc 0, T_app 360–362 s.
  - 2 sock1 runs list NUMA {1,4,5,6,7}. That is host **dane261**, which has SNC **off** (2 NUMA domains: socket 0 = numa 0,
    socket 1 = numa 1). PUs are correct, so the runs are kept. Other nodes have SNC on (§1); note that the hardware is not uniform.
- **The inhibitor is fixed-duration (`-r 360`)**, so T_app is not an outcome. The outcomes are messages sent in 360 s
  (= work and intensity) and T_MPI.
- **Messages sent:** no socket effect. Pooled d −0.0011 [−0.0023, +0.0001], every config |d| < 0.5%, order effect ~0.
  Inhibitor **intensity is socket-independent**, so the aggressor strength in App-Inh co-runs doesn't depend on which socket
  the inhibitor got.
- **T_MPI:** pooled mean +0.048 [−0.019, +0.152]. That is driven by 2 single-run spikes (m=200 w=2e5 1-peer rep1 sock1
  4.49 s vs ~0.5; m=2.5e6 w=2e5 1-peer rep3 sock1 21.8 vs ~15), both where sock1 ran first. Sign count 25/36 sock1 lower
  (Wilcoxon p = 0.017). CI excludes 0 only for m=4e4 w=1e6 s=0.1 (−0.057, 4/4) and m=2.5e6 w=6e5 1-peer (−0.025, 4/4).
  So sock1 is slightly faster, by a few %, which is small next to the app effects.
- **Drift vs earlier isolated inhibitors (new sock0 vs `data_inhib_isolated.csv`):** messages agree within ~1% except
  m=500 w=5e3 s=0.05 (+8.4%). Its old reps were noisy (T_MPI 17–70 s, CV 58%) against a new range of 12.7–18.7 s, which
  gives a T_MPI log diff of −1.06. The tiny m=200 config has +0.45 log T_MPI (0.36 → 0.57 s). Other configs are +0.01 to +0.09.
- **App-Inh labels / JMs are unaffected:** the slowdown is the app's T_co/T_iso. Inhibitor d would only matter for
  inhibitor profile features.

## 8. Socket batch 2 (socket-aware pipelines; started 2026-11-08, director-approved A5)

- **Facts:**
  - A1: always complementary sockets.
  - A2: 166 resolved App-App cells (42 one-rep); 169 resolved panel cells.
  - Floors: panel MPI 46/169, App-App MPI 52/166.
  - Details in `output/socket_aware_numbers.md` Part A and `socket_aware_facts.md`.
- **dane261:** in 31 Aug App-Inh co-runs, 4 June/Oct isolated inhibitors and 2 Nov socket jobs. Not in any June
  app baseline or App-App run. The host table is `output/run_hosts.csv` (`scan_hosts.py`).
- **Code:**
  - Copies `notebooks/utils/*_socketAware/` (4 dirs) have a `socket_feature` flag. Originals are untouched.
  - Resolved data is in `*_socketAware/data_socket/`, built by `scripts/dissertation_figs/socket_batch2_prepare.py`.
  - Each endpoint gets its own socket bit as an input.
  - Absolute loss is weighted 1/n_resolved.
  - Anchors: all resolved anchors, no socket subsets. One calibration offset per victim. No d_inh.
- **Runs:**
  - `~/hpcResearch/experiment_outputs/socket_batch2/`: `repro_*` = original code; `blind_*` / `aware_*` = copies.
  - The one-way control is the runner of `delta_response_filteredInhibs` on `_totalTime/data/mpi_time` (the 1W and
    1W-TT code are identical except the default data path).
  - Driver: `socket_batch2_run1w.sh`.
- **Scoring:**
  - `socket_aware_score.py` produces B1 (resolved) and B2 (rep-averaged, aggregated in slowdown space).
  - Offset model = blind absolute latent + s_V·d̄ (JMs: CI excludes 0).
- **Machine:** BD CV costs about 1.6 s per fit at 2 threads.
- **Order agreed with the director:** full-data B1/B2 → B3 → B5 → C1/C2 → B4.
  - B4 runs only if the measured cost is ≤ 5 h; otherwise ask first.
  - Report selected HPs for blind and aware side by side.
  - Blind runs from the copies must be bit-for-bit equal to the originals.
- **Status 2026-11-08 23:59: full data DONE for all 4 pipelines; numbers sent to the director in 3 updates.**
  - Verification:
    - all 4 `repro_*` equal the artifact (6 decimals);
    - all 4 `blind_*` copies equal the repros bit-for-bit;
    - aware App-App aggregation matches the recomputation to 5e-16.
  - Verdict: the learned socket bit does not help.
    - Worse: generic everywhere, absolute and delta uniform on MPI (CIs > 0).
    - Only gain: median delta on total time (1W-TT −0.0095, BD-TT −0.0045).
    - Best socket-aware MPI predictor: the fixed offset model (blind absolute + s_V·d̄, all-app 0.1011 vs 0.1035,
      −8.2% vs baseline). It beats the learned bit by 0.008–0.009.
  - Caveat (a subagent diagnosed it from checkpoints):
    - BD/1W absolute, joint, profile and context models train 1–7 epochs × 4 batches, i.e. near init.
    - The App-App level drifts with epochs and seeds; several aware variants floor everywhere.
    - Not a socket-code bug: token-count clamping and socket filtering move latents ≤ 0.04.
  - The absolute model is shared code in 1W and BD: its ratios are identical, so C2 shows one absolute series.
- **Figures:** `socket_aware_figs.py` → `socket_aware_forest.pdf` (C1), `socket_aware_pervictim.pdf` (C2).
- **B4:** measured cost is about 6–11 h wall (1W final eval 8–15 min per task × 70 tasks per run).
  - Sent options (a) full, (b) BD only ~1.5–2 h, (c) BD + 3/10 1W splits, (d) skip. **Awaiting the director's choice.**
  - Runner `--resume` rejects copied tune tasks, because `status.json` `output_dir` is absolute. Rewrite it to reuse tuning.
- **Ops lesson:** never edit `socket_batch2_run*.sh` while runs use it. Bash re-reads scripts mid-run (exit 127 at the
  evaluate line); fix by re-running the script, which resumes.

**§8 update 2026-11-09: control E (epoch-matched) + B4 launch.**
- E configs: `scripts/dissertation_figs/socket_batch2_epochmatch.py`. Runs are in `socket_batch2/E{0,1,2}_*`; scorer `socket_epochmatch_score.py`
  writes `output/socket_epochmatch_numbers.md`. Both pipelines have `selection_mode: fixed_global`; the BD fixed replay (E0)
  reproduces the CV run exactly (max |Δ| 0.0).
- The aware absolute response predicts the floor on all 166 cells at the blind epochs too, in all 4 pipelines. Blind at the aware
  epochs does not (17.5% floor), so the collapse comes with the socket input (probably the init RNG shift). Untested idea: socket
  bit fed as a constant 0.
- The BD-MPI calibrated FB = 1.0 collapse goes away at the blind recipe (E1: FB 0.25/0.26; λ 10→1 and fewer epochs). The
  delta/generic penalties survive at equal recipe (they come from the socket bit). The TT median-delta win also survives.
- FB@1.05 = fraction of HARMFUL outcomes predicted benign (FB 1.0 = all missed). Don't word it as "classified correctly".
- B4 (BD only, decided by Akhil): `scripts/dissertation_figs/socket_batch2_b4_bd.sh` plans a runner root and registers the existing
  CV run as tune_global. The `_SUCCESS.json` must carry the runner's `execution_fingerprint`, or `evaluate --resume` re-runs the
  1.5 h tuning. Size 2 (and probably 3) fails with "inhibitor_blocks exceeds the number of panel inhibitors", as in the artifact
  (hence 50 = sizes 4–8).
- **D1/D2 (2026-11-09), overturns the B1 reading.** D1 = aware code with the socket bit fed as a constant 0, at the E1 recipe.
  Patched copies are in `socket_batch2/d1_code/`; the generator is `socket_batch2_diag.py`. D1 predictions are identical across
  sockets, so the bit is the only socket pathway. D1 − blind (init from the extra column) reproduces nearly all of the
  penalties (up to +0.03) and the absolute floor collapse. E1 − D1 (information) is small and ≤ 0 wherever it is significant
  (−0.001 to −0.008). The TT median-delta "win" is mostly init too. D2 (aware epochs, λ = 1) shows the calibrated BD-MPI
  FB = 1.0 comes from λ = 10. The CV prefers λ = 10 over 1 by only 0.001–0.005, yet the choice flips FB from 0.09 to 1.0. The
  near-init models are seed/init sensitive at about ±0.03, even averaged over 5 final seeds.
- 2026-11-09 B4 done (`socket_b4_score.py` → `output/socket_b4_numbers.md`). Roots socket_batch2/b4_{blind,aware}_bd_{mpi,tt}.
  Validation: completion 47/50 per root (8/9/10/10/10), sizes 2–3 0/20, the same 23 panel_smaller_than_cv_blocks failures
  as the artifact; each random_split task = its CV run (max |Δ| 0); blind B4 on the artifact's 20 C4 cells reproduces
  the artifact C4 one_known entries exactly (MPI and TT). 47 matched cells.
  BD-MPI: aware absolute and context-cal at the floor (floor frac ≥ 0.99), skill 0 vs blind +4–6%; paired n.s.
  joint/bprof cal paired n.s. (aware floor frac 0.37–0.60 vs 0.11–0.17); generic worse (+0.021*, ZS +0.028*);
  delta median better ZS only (−0.032*). BD-TT: aware better on absolute, delta median, ZS joint/bprof (all ≤ 0.011),
  generic ZS worse; TT is floor-dominated (blind floor frac ≈ 0.7), so the gains come from predicting the floor more.
  Aware training panels: ~1.9× rows, floor-row share 26% vs 18% (MPI), 38–41% vs 30–33% (TT); one-sided cells ≈ 10%.
- 2026-11-09 Seed pilot S0 (`seed_pilot.py` → `output/seed_pilot_numbers.md`, `seed_cost_plan.md`; run socket_batch2/seed_pilot/P0,
  BD-MPI blind, fixed_global, final_seeds 0–29; seeds 0–4 = the old run exactly). Near-init variants (absolute, joint, bprof,
  bctx) are trimodal per seed: ~1/3 blow up (Δ% > +25), many at the floor, median single seed = baseline. The ensemble (mean
  latent → floor) drifts with k: joint cal 0.108 (k=5) → 0.094 (k=30); absolute → the floor on all 90 outcomes at k=30
  (= baseline, FB 1.0). So the "absolute −8.5%" is a 5-seed artifact (holds in 42% of 5-subsets). Delta/generic are stable.
  Jackknife SE at N=30: delta/generic ~0.0025, calibrated 0.003–0.006, direct 0.012–0.017. Costs: BD full 60+13N s,
  1W full 140+43N s, BD restricted task 60+31N s. Sent to the director; waiting on the go and the N choice.

## 9. Seed batch and BD-MPI training probes (2026-11-09 to 11)

### Scripts and outputs
Scripts are in `scripts/dissertation_figs`:
- `seed_score.py`, `seed_restricted_score.py`;
- `retune_probe_{build,run,score}.py`, `retune_probe_chain.sh`, `capacity_probe_chain.sh`;
- `figs_seed.py`, `figs_probe.py`.

Outputs are in `output/`:
- `seed_numbers.md` (part a, N=100);
- `seed_restricted_numbers.md` (b/c, N=30);
- `retune_probe_numbers.md`, `capacity_probe_numbers.md`;
- `cv_vs_appapp_N100.md`;
- `retune_probe_incident_20261109.md`.

Raw runs are in `~/hpcResearch/experiment_outputs/{seed_batch,retune_probe,capacity_probe}`.

### Label rule (director-accepted)
- **sign-unstable:** jackknife seed SE > 0.002 AND |est| < 2·SE. If |est|/SE is between 1.9 and 2.1, the two-level CI decides.
- **Otherwise:** holds / fails / n.s. by the two-level (pairs×seeds) CI.

### Part (a), N=100
- **Absolute:** at the floor in all 4 pipelines, so it equals the baseline. App-App log MAE is 0.1030 (MPI) and 0.0156 (TT). The artifact's 5-seed absolute numbers are seed artifacts.
- **Calibrated variants vs baseline:** no two-level CI excludes 0. BD-MPI joint cal is −10.5%, n.s.
- **CV vs App-App Spearman at N=100:**

  | 1W-MPI | 1W-TT | BD-MPI | BD-TT |
  |---|---|---|---|
  | +0.50 | −1.00 | +0.81 | +0.62 |

- **Delta aggregation:** 1W-TT uses ood_kernel; the other three pipelines use uniform.

### (b)/(c), restricted tiers
- **Completion:** matches the artifact. Seeds 0–4 reproduce the artifact's C4 exactly.
- **Zero-shot delta/generic:** significantly worse than the baseline.
- **"No gain with size":** holds formally in every row. But BD zero-shot delta/generic slopes are > 0 in 100% of 5-seed draws.

### Training probes (BD-MPI)
- **Retune probe** (240 epochs, CV-selected):
  - Learned-profile variants: the fold-averaged CV curve is minimised at epoch 1, but the protocol picks 16–37 epochs.
  - At those budgets, joint cal is +30.8% (n.s.) and generic is significantly worse than the baseline.
  - Mechanism: under-prediction flips to over-prediction on benign pairs.
- **Capacity probe** (4 configs): CV prefers the larger models. At the CV budget, joint cal is +12.4% (n.s.).
  - The only CI-excluding gain anywhere is absolute at the sensitivity budget: 32-dim model, 1 epoch, −11.7% [−23.0, −0.4]. It comes from shrinkage, not skill.
- **At 1 epoch:** joint, bprof and bwall cal are numerically identical, so the C7 ablations are untestable at near-init budgets.
- **Socket bit (E1 − D1):** helps delta/generic by 0.002–0.011 log MAE once they are trained past near-init. It never helps the calibrated variants.

### Pipeline-copy pitfall
- **Symptom:** a single-variant crossed-CV run crashed after all its fits finished.
- **Cause:** the run writes an empty `ablation_summary.csv` (it has no internal contrasts), and re-reading that file raised EmptyDataError.
- **Fix:** an EmptyDataError guard, added to the BD socket-aware copy only.
- **Note:** `pgrep -f <name>` inside a `bash -c` wait loop matches itself, so use a PID or log marker instead.
