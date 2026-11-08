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

## 6b. Still pending (as of 2026-11-07)

- **Isolated inhibitor data (S2 + inhibitor drift).** In `20261107_180934`, all 72 `data_inhib/sock*/8_inhib_*` dirs
  were empty: no `.mpiP`, no placement files, and empty `job_logs/8_inhib_*`. The chapter director is raising this
  with Akhil. The dir names cover exactly the 9 MPI panel inhibitors, with no extras.

  When the data lands, replace the placeholder S2 block in `analysis_socket_pinned.py`, which currently prints
  "Not available":
  1. Parse each valid root with `parse_inhibitor_isolated`. Columns: `Inhib Message Size`, `Inhib Wait Time (us)`,
     `Inhib Comm Sparsity`, `Run Iteration`, `Inhib App Time Average`, `Inhib MPI Time Average`.
  2. Repeat the S1 tables (paired d per config, t CI, sign count, order effect, config-cluster bootstrap).
  3. Repeat the S3 drift check against `processed_data/data_inhib_isolated.csv`, the June/Oct isolated inhibitors,
     which were run with `-r 360`.
  4. Round sparsity to 9 digits when merging.
  5. Check whether inhibitor dirs have placement files at all before applying `check_run`.
  6. Rerun both socket scripts. `analysis_socket_matched.py` rewrites everything from `## S5.` onward in
     `socket_numbers.md`.
  7. Message the chapter director with the results.
- If the `*_full_20261017` roots become available:
  - regenerate the artifact figures directly from `consolidated/predictions.csv`;
  - re-score joint MPI and the artifact-recipe delta models under JMs (S6 gap).

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
