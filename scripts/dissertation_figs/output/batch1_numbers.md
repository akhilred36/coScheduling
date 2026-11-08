# Node-sharing chapter: Batch 1 facts and numbers (A1–A6, B1–B4)

Generated 2026-11-07 by the coScheduling-data-analyzer session. Branch `dane` @ 5eea7198.
Canonical data: the current rebuilt snapshot
(`notebooks/utils/delta_response_filteredInhibs_biDirectional/data/` for MPI time,
`..._biDirectional_totalTime/data/` for total time). Slowdown = max(1, co/iso), ratio of
aggregate means. Directional outcomes are non-self unless stated.

Scripts (run with `~/hpcResearch/python_venvs/ml_analysis/bin/python`):
- `scripts/dissertation_figs/analysis_batch1.py` → `output/batch1_numbers_raw.md` (all tables)
- `scripts/dissertation_figs/analysis_placement.py` → `output/placement_numbers.md` (socket and coverage)

---

## ⚠ Flags (the data disagrees with the plan, or a confound is new)

1. **Inhibitor runtime-limit bug, App-Inhibitor campaign.** `scripts/generate_coscheduled_inhib_scripts_multinode.py:get_max_runtime`
   returns the *first* CSV row for the app. For the six apps that also have 1-node data, that row is the
   **1-node** runtime. The `-r` values actually used (from the job scripts) are below.
   - The inhibitor stops before the app finishes in **100% of runs** for Beatnik, FIESTA, LAMMPS and TriC.
   - Median fraction of the app's run covered by the inhibitor: TriC 0.68, LAMMPS 0.75, FIESTA 0.84, Beatnik 0.86.
   - Even with the right `-r`, the inhibitor stops at roughly the *isolated* runtime, so any app that slows
     down has an uncovered tail. Laghos has `-r` correct but median coverage 0.86.
   - The plan's claim that the inhibitor "overlaps the victim's whole run" is **false**.
   - This affects only App-Inhibitor training labels. App-App co-runs have no runtime limit.
   - Likely effect: susceptibility is understated for those victims in the training data.

   | App | `-r` used (s) | isolated 8-node T_app (s) | median coverage | p10 coverage | frac runs coverage < 0.95 |
   |---|---:|---:|---:|---:|---:|
   | AMG2023 | 343 | 342.5 | 0.980 | 0.934 | 0.148 |
   | Beatnik | 371 | 397.7 | 0.857 | 0.719 | 1.000 |
   | FIESTA | 299 | 359.1 | 0.844 | 0.801 | 1.000 |
   | Kripke | 355 | 354.5 | 0.997 | 0.959 | 0.090 |
   | Laghos | 332 | 331.2 | 0.856 | 0.730 | 0.660 |
   | LAMMPS | 266 | 354.8 | 0.749 | 0.729 | 1.000 |
   | MiniFE | 384 | 388.2 | 0.970 | 0.937 | 0.171 |
   | miniVite | 442 | 371.2 | 1.000 | 1.000 | 0.003 |
   | Quicksilver | 295 | 295.0 | 0.997 | 0.964 | 0.071 |
   | TriC | 281 | 378.0 | 0.681 | 0.619 | 1.000 |

   Coverage = min(1, inhibitor AppTime / app AppTime). The mpiP start timestamps show both jobs start in the same second.

2. **Socket-placement confound.** Each job's 56 ranks per node are bound by mpibind to one PU per core on
   **one socket**.
   - Isolated app baselines (40/40 runs) and isolated inhibitors all ran on **socket 0**, with socket 1 idle.
   - In co-runs, the two jobs get one socket each, and which job gets which socket is a launch race:
     - App-Inhibitor: app on socket 0 in 6,105 runs, socket 1 in 5,375.
     - App-App: 89 / 91.
   - Within the same configuration, a job on socket 1 has **lower** MPI time:

     | Campaign | T_MPI on socket 1 vs socket 0 | T_app on socket 1 vs socket 0 |
     |---|---:|---:|
     | App-Inhibitor | −12.2% (log −0.122) | −2.0% |
     | App-App | −6.6% (log −0.066) | −0.6% |

   - The largest per-app T_MPI effects (log):
     - App-Inhibitor: AMG2023 −0.28, FIESTA −0.26, MiniFE −0.20, Laghos −0.15, Kripke −0.13, LAMMPS −0.11.
     - App-App: FIESTA −0.31, AMG2023 −0.19.
     - Beatnik, TriC, miniVite and Quicksilver are about 0.
   - Because every baseline is on socket 0, co-run ratios are biased **downward** for runs on socket 1. This plausibly
     explains much of the sub-1 raw ratios (15% of App-Inhibitor configs, 20% of App-App outcomes) and part of the
     replicate CV.
   - App-App directions by the number of their 4 replicates with the victim on socket 1:
     0/4 = 7, 1/4 = 21, 2/4 = 34, 3/4 = 21, 4/4 = 7.
   - The cause is not identified. It could be NIC/PCIe locality, OS daemons or interrupt load on socket 0, or similar.
     The socket mapping assumes standard Linux PU numbering on Dane (0–55 s0, 56–111 s1, 112–167 s0 SMT siblings,
     168–223 s1 siblings). This is inferred, not recorded in the logs.
   - The isolated runs used PUs 0–55 (Beatnik, FIESTA, LAMMPS, MiniFE, miniVite, TriC) or 112–167
     (AMG2023, Kripke, Laghos, Quicksilver). Under that numbering, these are the same physical socket-0 cores.

3. **Contention sources (plan §2).** Because each job owns a whole socket, the two jobs do **not** share
   cores or the L3. Very likely they don't share the socket-local memory controllers either. mpibind places memory
   NUMA-locally by default, but that was not verified from logs. What is shared:
   - the NIC and fabric links;
   - the inter-socket (UPI) link;
   - node memory capacity (each job gets `--mem 119G`).

   Rephrase "memory bandwidth, LLC on shared sockets" accordingly.
4. **Most susceptible victim is Laghos, not Beatnik.**
   - Additive victim effects: Laghos ×1.168, Beatnik ×1.109.
   - The largest single outcome is Laghos with TriC as aggressor: 2.21.
   - Laghos is also the noisiest app (isolated T_MPI CV 27%).
   - Beatnik is the only app that is strongly both a victim and an aggressor (aggressor ×1.129).
   - TriC is the dominant aggressor (×1.227), which agrees with the plan.
5. **Smallest isolated T_MPI** is AMG2023 (12.8 s), not MiniFE (13.6 s).
6. **T_app replicate CV** is not "mostly ≤ 1%". The median is 0.98%, 5 of 10 apps are above 1%, and Laghos is 4.8%.
7. **mpiP time resolution.** mpiP prints per-task AppTime and MPITime with **3 significant digits**. That means 1 s for
   100–999 s runs (about 0.3% of a 350 s run) and 0.1 s for MPI times of 10–99 s. The CSV values are means over
   448 tasks. Say "3-significant-digit per-task timers", not "whole-second resolution".
8. **Timing of baselines.** Isolated app baselines were collected in June 2026, App-App runs in June 2026, and
   App-Inhibitor runs in late Aug–Sep 2026. Cross-campaign drift is possible. It is not quantified.

---

## A1. Platform and launch

**Hardware.** This is not recorded in the run data, so it comes from external sources: the LLNL reference-systems
page and TOP500.
- CPUs: 2× Intel Xeon Platinum 8480+ (Sapphire Rapids), 56 cores each, 2.0 GHz. That gives 112 cores per node.
- Memory: 256 GiB DDR5-4800 per node.
- Interconnect: Cornelis Networks Omni-Path.

The logs do confirm some of this: hosts are named `dane###`, PU indices run 0–223 (SMT-2), and mpibind reports 0 GPUs.

**Software.**
- Compiler: GCC 10.3.1 (`module load gcc/10` → `gcc-10.3.1-magic`).
- MPI: Open MPI 4.1.2 (`module load openmpi`, which replaced the default mvapich2 2.3.7; MPI standard 3.1).
  A Spack environment ("beatnik") was activated on top of the modules. I could not confirm whether that environment
  supplies its own MPI.
- Profiler: mpiP 3.5.0 via `LD_PRELOAD`, timer PMPI_Wtime.

**Allocation** (identical for every 8-node campaign; one sbatch job per run):

```
#SBATCH --nodes 8 --ntasks 896 --ntasks-per-node 112 --cpus-per-task 1
#SBATCH --mem 240G --partition pbatch --distribution block:cyclic --time 00:15:00|00:20:00
```

The whole node is always allocated, so no other user's job can land on it.

**Job steps.** Each job (app or inhibitor) runs as:

```
srun -n 448 --nodes 8 --ntasks-per-node 56 --mem 119G --distribution=block:block --mpibind=on,v \
     env LD_PRELOAD=libmpiP.so MPIP="-f <dir>" <exe> <args>
```

| Campaign | How the job steps are launched |
|---|---|
| Isolated app | One srun in the allocation. The other socket is idle. No `--exclusive`, no `--cpu-bind`. |
| Isolated inhibitor | Same as isolated app, with `-r 360`. |
| App-App | Two sruns in the **same allocation**, both backgrounded (`&`), then `wait`. No `--exclusive`, `--overlap` or `--oversubscribe`. |
| App-Inhibitor | Two sruns in the same allocation, each with `--exclusive --cpu-bind=cores` in addition to mpibind. The inhibitor is backgrounded and launched first; the app runs in the foreground; then `wait`. Replicates use a job array (0–3). |
| 1-node | Allocation `--ntasks 112 --mem 120G`; each step is `srun --exclusive -n 56 --mem 119G --distribution=block:block --cpu-bind=cores --mpibind=on,v`. |

**Resulting placement:**
- 56 ranks per node, one per core, all on one socket.
- Isolated runs: always socket 0, with socket 1 idle. This confirms "same half-node placement, other half idle",
  where "half" means **one socket**.
- Co-runs: one job per socket.

## A2. Inputs (`run_configs/8_nodes.json`, 448 ranks)

| App | Command line |
|---|---|
| AMG2023 | `amg -P 14 8 4 -n 90 220 240 -problem 2 -iter 280` |
| Beatnik | `rocketrig -F 0 -n 2400 -w 448 -t 10` |
| FIESTA | `fiesta fiesta_448_1000_120ts.lua` |
| Kripke | `kripke.exe --procs 7,8,8 --zones 210,240,240 --niter 15` |
| Laghos | `laghos -p 1 -dim 2 -nx 185 -ny 185 -nz 185 -rs 3 -tf 0.8 -pa -fs 1500` |
| LAMMPS | `lmp -in in_780_0.9_130.lj` (ExaMiniMD-style LJ input) |
| MiniFE | `miniFE.x --nx 1200 --ny 1200 --nz 1200 --iters 800` (Kokkos build) |
| miniVite | `miniVite -f 60000000_1800000000.bin -i 30` (6e7 vertices, 1.8e9 edges) |
| Quicksilver | `qs --nx 512 --ny 512 --nz 512 --nSteps 280` |
| TriC | `tric -f 60000000_1800000000.bin -i 35` (same graph as miniVite; `apps/tric`, so "TriC" is right) |

The 1-node subset (`1_nodes.json`, 56 ranks) has 6 apps: Beatnik, FIESTA, LAMMPS, MiniFE, miniVite and TriC.

## A3. Isolated 8-node profiles (`data_isolated_agg.csv`; rates over T_app, as in `select_inhibitors.py`)

| App | T_app (s) | T_MPI (s) | comm_frac | messages | bytes | msg rate (1/s) | byte rate (B/s) | mean msg (B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMG2023 | 342.5 | 12.8 | 0.037 | 1.06e8 | 1.01e12 | 3.11e5 | 2.94e9 | 9.47e3 |
| Beatnik | 397.7 | 99.9 | 0.251 | 2.00e7 | 1.09e13 | 5.03e4 | 2.75e10 | 5.46e5 |
| FIESTA | 359.1 | 65.7 | 0.183 | 6.45e5 | 1.64e12 | 1.80e3 | 4.57e9 | 2.54e6 |
| Kripke | 354.5 | 17.0 | 0.048 | 3.03e5 | 3.88e11 | 854 | 1.09e9 | 1.28e6 |
| Laghos | 331.2 | 54.9 | 0.166 | 8.01e8 | 6.34e11 | 2.42e6 | 1.91e9 | 791 |
| LAMMPS | 354.8 | 17.4 | 0.049 | 4.57e5 | 6.70e11 | 1.29e3 | 1.89e9 | 1.47e6 |
| MiniFE | 388.2 | 13.6 | 0.035 | 6.79e6 | 3.82e11 | 1.75e4 | 9.85e8 | 5.63e4 |
| miniVite | 371.2 | 51.6 | 0.139 | 2.41e7 | 1.13e12 | 6.49e4 | 3.06e9 | 4.71e4 |
| Quicksilver | 295.0 | 134 | 0.454 | 7.63e5 | 1.60e8 | 2.59e3 | 5.43e5 | 210 |
| TriC | 378.0 | 204 | 0.540 | 7.12e6 | 1.02e13 | 1.88e4 | 2.70e10 | 1.43e6 |

T_app and T_MPI are means over 4 replicates of the per-rank average. Messages and bytes are totals over all 448
ranks, from the mpiP aggregate-sent table.

## A4. Campaigns

| Campaign | Configs | Replicates | Runs |
|---|---:|---|---:|
| Isolated apps | 10 (8-node) + 6 (1-node) | 4 | 40 + 24 = **64** |
| Isolated inhibitors (8-node) | 287 | 4 (2 configs have 3) | **1,146** |
| App-Inhibitor (8-node) | 2,870 = 10 × 287 | 4 (1 config has 2) | **11,478** |
| App-App | 55 (8-node) + 21 (1-node), self-pairs included | 4 | 220 + 84 = **304** |

The plan's numbers all match. The configs with missing replicates are:
- isolated inhibitors (400000 B, 0 µs, 0.1) and (400000 B, 10000 µs, 1.0), with 3 each;
- App-Inhibitor Laghos × (500000 B, 20000 µs, 0.002237), with 2. The run directories exist (11,480 and 1,148), but
  those runs did not parse.

All 2,870 App-Inhibitor configs survive into the modeling data, with 0 rejected; 443 (15.4%) are at the floor. All 55
App-App pairs survive (10 self). There are no 1-node inhibitor data.

## A5. Inhibitor

**Grid** (3 blocks, not a full factorial; 287 configs at 8 nodes; `-c d` throughout):

| Block | Message sizes (B) | Wait times (µs) | Sparsities | Configs |
|---|---|---|---|---:|
| base | 2000, 4000, 20000, 40000, 200000, 400000, 2000000, 4000000 | 0, 100, 1000, 10000, 100000, 1000000 | 0.1, 0.3, 0.5, 0.7, 1.0 | 8×6×5 = 240 |
| extended | 20, 200, 500, 1000 | 5000, 10000, 25000, 50000 | 0.01, 0.05 | 4×4×2 = 32 |
| missing-range | 200, 10000, 500000, 1500000, 2500000 | 20000, 200000, 600000 | 0.002237136 | 5×3×1 = 15 |

The union is 16 message sizes, 12 wait times and 8 sparsities. 0.002237136 ≈ 1/447, which is exactly one peer.

**Iteration structure** (`networkInhib.cpp`):
1. One `MPI_Barrier` before the loop.
2. Each iteration:
   - `MPI_Isend` of `msgSize` bytes to each of the rank's targets;
   - `MPI_Irecv` from every rank that targets it;
   - `MPI_Waitall` on the sends, then `MPI_Waitall` on the receives;
   - `sleep_for(wait µs)`;
   - `MPI_Barrier`.
3. With `-r`, each iteration also does an `MPI_Allreduce(MAX)` of a local "elapsed ≥ r" flag. This is the collective
   stop.

**Targets:**
- The out-degree is k = round(s·(P−1)) for **every** rank. With P = 448, sparsities 0.0022/0.01/0.05/0.1/0.3/0.5/0.7/1.0
  give k = 1/4/22/45/134/224/313/447.
- In-degree varies by rank (random).
- In `-c d` mode the target set is drawn **once** per rank, with an LCG seeded 12345 + rank, and stays fixed for the
  whole run. Random mode (`-c r`), which redraws every iteration, was not used.

**Ranks.** The inhibitor runs 448 ranks (56 per node) in co-runs and isolated.

**Runtime limit:**
- Isolated inhibitors: `-r 360`.
- App-Inhibitor: `-r` = ceil of the isolated mean T_app, but see Flag 1.

## A6. Clipping

| Set | n | raw < 1 | at floor (≤ 1) | min raw |
|---|---:|---:|---:|---:|
| App-App MPI, non-self | 90 | 20.0% | 20.0% | 0.831 |
| App-App MPI, self (20 copies) | 20 | 45% | 45% | 0.845 |
| App-App total, non-self | 90 | 36.7% (+3.3% exactly 1) | 40.0% | 0.971 |
| App-App total, self | 20 | 50% | 55% | 0.975 |
| App-Inhibitor MPI | 2,870 | 15.4% | 15.4% | 0.687 |
| App-Inhibitor total | 2,870 | 20.2% | 21.3% | 0.942 |

## B1. Replicate noise (CV = sample SD / mean over replicates)

**Isolated 8-node apps:**

| App | CV T_MPI | CV T_app |
|---|---:|---:|
| AMG2023 | 7.8% | 0.29% |
| Beatnik | 3.3% | 1.22% |
| FIESTA | 8.7% | 1.03% |
| Kripke | 3.2% | 0.16% |
| Laghos | **27.4%** | **4.83%** |
| LAMMPS | **16.1%** | 0.93% |
| MiniFE | 2.3% | 0.13% |
| miniVite | 5.2% | 1.13% |
| Quicksilver | 0.5% | 0.28% |
| TriC | 2.6% | 1.40% |
| **median** | **4.3%** | **0.98%** |

**CV distributions:**

| Set (n) | median | p75 | p90 | max |
|---|---:|---:|---:|---:|
| Isolated inhibitors, T_MPI (287 configs) | 0.63% | 3.5% | 9.7% | 103% |
| App-Inhibitor, app T_MPI (2,870) | 8.3% | 17.9% | 31.1% | 174% |
| App-App, T_MPI (110 endpoint outcomes, including self) | 6.7% | 12.2% | 22.6% | 38.1% |
| App-Inhibitor, app T_app | 1.5% | 3.2% | 5.3% | 54.5% |
| App-App, T_app | 1.1% | 1.6% | 2.3% | 7.4% |

Isolated inhibitors with CV > 10%: 9.1%; with CV > 50%: 1.4%. The worst configs are all sparse ones:
(4000 B, 100000 µs, 0.1) at 103%, (200, 25000, 0.01) at 99% and (500, 5000, 0.01) at 71%.

Part of the co-run CV is the socket effect (Flag 2).

**Propagated uncertainty of the App-App MPI ratio.** I used the delta method on the ratio of means, with 4 co-run
and 4 isolated replicates.
- The SE of the raw ratio is median 0.046, p90 0.164, max 0.360.
- 58 of 90 non-self outcomes have raw − 1 < 2 SE, including all 18 floor outcomes. That leaves 40 of the 72
  above-floor outcomes within 2 SE of 1.
- Of the 46 outcomes with slowdown ≥ 1.05, 30 are more than 2 SE above 1.
- Per-victim median SE: Laghos 0.167, LAMMPS 0.137, FIESTA 0.118, AMG2023 0.079, miniVite 0.057, MiniFE 0.032,
  Beatnik 0.031, TriC 0.026, Kripke 0.023, Quicksilver 0.015.

**Resolution.** mpiP per-task times have 3 significant digits (Flag 7).

## B2. App-App slowdown distribution (8 node)

| Set | n | mean | median | p75 | p90 | max | floor | ≥1.05 | ≥1.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPI, non-self | 90 | 1.123 | **1.051** | **1.131** | **1.311** | **2.209** | **20.0%** | 51.1% | 31.1% |
| total, non-self | 90 | 1.016 | **1.001** | 1.025 | **1.043** | **1.192** | **40.0%** | 6.7% | 3.3% |
| MPI, self (both copies) | 20 | 1.142 | 1.004 | 1.176 | 1.545 | 1.588 | 45% | 40% | 35% |
| total, self (both copies) | 20 | 1.044 | 1.000 | 1.015 | 1.146 | 1.290 | 55% | 20% | 20% |

All the plan's numbers match. Self-pairs (copy A / copy B):

| App | MPI A / B | total A / B |
|---|---|---|
| Beatnik | 1.588 / 1.587 | 1.128 / 1.130 |
| TriC | 1.540 / 1.541 | 1.290 / 1.290 |
| miniVite | 1.173 / 1.135 | 1.020 / 1.014 |
| AMG2023 | 1.186 / 1.069 | 1.004 / 1.000 |
| Quicksilver | 1.021 / 1.000 | 1.010 / 1.001 |
| Kripke | 1.006 / 1.002 | 1.000 / 1.000 |
| FIESTA, Laghos, LAMMPS, MiniFE | 1.000 / 1.000 | 1.000 / 1.000 |

The two worst total-time outcomes in the whole App-App set are self-pairs: TriC + TriC (1.29) and Beatnik + Beatnik
(1.13).

**Most asymmetric non-self pairs (MPI), by |log s_A − log s_B|:**

| Pair | s_A(B) | s_B(A) |
|---|---:|---:|
| TriC + Laghos | 1.067 | 2.209 |
| TriC + AMG2023 | 1.057 | 1.826 |
| MiniFE + TriC | 1.730 | 1.087 |
| Beatnik + Laghos | 1.159 | 1.820 |
| Beatnik + TriC | 1.631 | 1.119 |

**Total time:** TriC + Laghos is 1.036 / 1.192, and Beatnik + Laghos is 1.031 / 1.128.

## B3. Victim/aggressor decomposition (y = log slowdown, 90 non-self outcomes, two-way fixed effects)

| Target | R² victim only | R² aggressor only | R² additive | residual (interaction + noise) |
|---|---:|---:|---:|---:|
| MPI time | 0.243 | 0.301 | **0.558** | 0.442 |
| Total time | 0.418 | 0.122 | **0.575** | 0.425 |
| MPI time, unclipped log ratio (sensitivity) | 0.208 | 0.274 | 0.501 | 0.499 |

For MPI time, the aggressor explains slightly more than the victim. For total time, the victim dominates.

**MPI-time effects** (sum-to-zero, shown as multiplicative factors; grand mean ×1.109):

| App | victim factor | aggressor factor |
|---|---:|---:|
| Laghos | **1.168** | 0.965 |
| Beatnik | 1.109 | 1.129 |
| AMG2023 | 1.052 | 0.992 |
| MiniFE | 1.002 | 0.952 |
| miniVite | 0.994 | 0.985 |
| TriC | 0.994 | **1.227** |
| Kripke | 0.944 | 0.976 |
| FIESTA | 0.938 | 0.953 |
| LAMMPS | 0.928 | 0.942 |
| Quicksilver | 0.904 | 0.918 |

- Aggressor ranking (MPI): TriC > Beatnik > AMG2023 > miniVite > Kripke > Laghos > FIESTA > MiniFE > LAMMPS > Quicksilver.
- Total time: grand mean ×1.016.
  - Victim factors: Laghos 1.035, Beatnik 1.029, TriC 1.028; all others 0.984–0.996.
  - Aggressor factors: TriC 1.030, Beatnik 1.014; all others 0.989–0.999.
- Spearman correlation of s_A(B) with s_B(A) over the 45 pairs:
  - MPI: ρ = 0.41 (p = 0.005);
  - total: ρ = 0.22 (p = 0.15) on clipped values, 0.39 (p = 0.008) on raw values.

## B4. 1-node vs 8-node, same 6 apps (15 non-self pairs = 30 directional outcomes; 6 self-pairs = 12 copies)

| Nodes | Target | Set | n | median | p75 | p90 | max | floor | min raw |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MPI | non-self | 30 | 1.032 | 1.115 | 1.391 | 1.662 | 30.0% | 0.953 |
| 8 | MPI | non-self | 30 | 1.088 | 1.159 | 1.314 | 1.730 | 20.0% | 0.884 |
| 1 | total | non-self | 30 | 1.002 | 1.007 | 1.015 | 1.026 | 40.0% | 0.996 |
| 8 | total | non-self | 30 | 1.004 | 1.035 | 1.048 | 1.147 | 43.3% | 0.984 |
| 1 | MPI | self | 12 | 1.042 | 1.093 | 1.252 | 1.472 | 16.7% | 0.988 |
| 8 | MPI | self | 12 | 1.067 | 1.540 | 1.582 | 1.588 | 50.0% | 0.845 |

- The magnitudes are similar, but **which pairs degrade changes completely**.
- Spearman correlation of 1-node vs 8-node MPI slowdown over the same 30 outcomes is ρ = −0.44 (p = 0.016).
- At 1 node, LAMMPS is the most susceptible victim:
  - LAMMPS ← TriC: 1.66 at 1 node, 1.00 at 8 nodes;
  - LAMMPS ← Beatnik: 1.42 at 1 node.
- At 8 nodes, MiniFE ← TriC (1.73) and Beatnik ← TriC (1.63) lead. At 1 node, the same outcomes are 1.11 and 1.01.
- Total-time interference at 1 node is negligible (max 1.026).
- The full 30-row table is in `batch1_numbers_raw.md`.
