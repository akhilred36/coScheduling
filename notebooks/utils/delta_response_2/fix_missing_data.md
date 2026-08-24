# Fix Missing Inhibitor Data

## Purpose

This document is the authoritative handoff for the next inhibitor data
collection. It explains the model failure, the communication-profile gaps in
the current dataset, the exact new inhibitor inputs to run, the existing cells
that are missing, and the collection and validation protocol.

The objective is to improve the fidelity of the synthetic inhibitor benchmark
to real application communication behavior. Configuration selection uses
isolated application communication statistics. It does not use App-App labels
to optimize a predictor.

## Non-negotiable scope: keep the inhibitor unchanged

The user has explicitly decided not to redesign the inhibitor.

The next agent must follow all of these constraints:

- Run the current `networkInhibitor/networkInhib.cpp` executable.
- Do not edit the inhibitor source.
- Do not add flags, communication modes, topology implementations, pacing
  logic, synchronization controls, or telemetry to the inhibitor.
- Use deterministic communication mode `-c d`, as in the existing data.
- Change only existing runtime inputs such as message size, wait time,
  communication sparsity, and runtime.
- It is acceptable to create or fix experiment-generation, orchestration,
  parsing, aggregation, and validation scripts. Those are collection machinery,
  not inhibitor redesign.
- If the current inhibitor cannot reproduce every application statistic in one
  configuration, collect useful bracketing configurations and document the
  residual mismatch. Do not respond by proposing another inhibitor design.

This file supersedes any recommendation elsewhere in this directory to add new
inhibitor mechanisms. Numerical findings in
`INHIBITOR_COMMUNICATION_GAP_ANALYSIS.md` remain useful, but its redesign
sections are not part of the execution plan. The disposition text in
`recommended_inhibitor_configurations.csv` is also superseded by this file.

## Scientific evidence boundary

The new input matrix below was derived from:

- `notebooks/utils/few_shot/data/jobs.csv`
- `notebooks/utils/few_shot/data/inhibitors.csv`
- `notebooks/utils/few_shot/data/job_inh.csv`
- Existing inhibitor controls and source behavior

The historical `pair.csv` was used only for a retrospective diagnostic after
the profile-derived design was fixed. It was not used to choose message sizes,
waits, sparsities, or run priorities. This is not cheating the learning task:
the research goal is to create a synthetic communication benchmark that covers
the behavior of real applications.

## Model and failure summary

Delta Response 2 predicts directional victim slowdown from isolated application
profiles, isolated inhibitor profiles, and App-Inhibitor responses. It does not
fit App-App outcomes.

The frozen App-Inhibitor-selected model performs well enough on its development
domain but fails to transfer its response scale:

| Quantity | Value |
| --- | ---: |
| Frozen App-Inhibitor worst-view log MAE | 0.166523 |
| Historical App-App non-self log MAE | 0.255138 |
| Historical App-App signed log bias | +0.245545 |
| Mean predicted App-App log slowdown | 0.348557 |
| Mean true App-App log slowdown | 0.103012 |
| Historical App-App Spearman | 0.362726 |

The model retains some ordering but predicts inhibitor-scale slowdown for real
application aggressors. This points to a synthetic-domain coverage problem,
not simply insufficient model capacity.

## Current data inventory

| Item | Count | Interpretation |
| --- | ---: | --- |
| Applications | 10 | All have eight-node isolated profiles. |
| Intended inhibitor configurations | 272 | Main grid plus extended grid. |
| Available isolated inhibitor profiles | 266 | Six intended configurations are absent. |
| Raw aggregate App-Inhibitor rows | 2,653 | One aggregate row per available app/configuration cell. |
| Rows currently accepted by Delta Response 2 | 2,623 | Thirty rows reference one of the six missing profiles. |
| Existing missing app/configuration cells | 67 | Relative to a complete 272 x 10 matrix. |
| App-Inhibitor floor rows | 389 (14.83%) | Slowdown recorded as exactly one. |

The existing CSVs contain aggregate means. The raw experiment target was four
repetitions per application/configuration cell.

### Stale NPZ warning

`notebooks/utils/few_shot/data/processed_data.npz` contains only 214 inhibitors.
It excludes every available profile from the extended small-message,
low-fanout sweep. The included subset has median communication fraction
`0.9636`, while the 52 omitted profiles have median `0.0273`.

Delta Response 2 reads the CSV files and is unaffected. Any older `few_shot`
pipeline that reads the NPZ is affected. Do not train a new result from that NPZ
without rebuilding it from a sealed, versioned CSV dataset and verifying its
inhibitor roster.

## Communication statistics

For an isolated profile, define:

```text
runtime_s       = mpi_time / comm_frac
mean_message_B  = total_bytes / total_msgs
message_rate_s  = total_msgs / runtime_s
byte_rate_B_s   = total_bytes / runtime_s
```

The ten application runs and inhibitor runs have similar durations, so totals
and rates are closely related. The important difference is where the empirical
mass lies.

| Statistic | Application median | Inhibitor median | Inhibitor/app ratio |
| --- | ---: | ---: | ---: |
| MPI time | 53.27 s | 338.05 s | 6.35 |
| Communication fraction | 0.152 | 0.939 | 6.16 |
| Total sent messages | 6.95 million | 75.71 million | 10.89 |
| Total sent bytes | 0.839 TB | 23.175 TB | 27.61 |
| Mean message size | 301 KB | 40.0 KB | 0.13 |
| Message rate | 18.2 thousand/s | 210.3 thousand/s | 11.58 |
| Byte rate | 2.43 GB/s | 64.31 GB/s | 26.48 |

Additional coverage facts:

- 179 of 266 inhibitor profiles (67.3%) exceed the maximum application
  communication fraction.
- 155 of 266 (58.3%) exceed the maximum application byte rate.
- 143 exceed both maxima.
- Only 34 of 266 (12.8%) lie jointly within all four application base-feature
  ranges.
- The smallest observed inhibitor message rate is 11,956/s. It is still above
  Fiesta, Kripke, LAMMPS, and QuickSilver.

The high-pressure region also produces systematically larger labels:

| Inhibitor stratum | Profiles | Mean victim log slowdown | Mean floor fraction |
| --- | ---: | ---: | ---: |
| Byte rate at or below application maximum | 111 | 0.184 | 24.8% |
| Byte rate above application maximum | 155 | 0.580 | 7.9% |
| Communication fraction at or below application maximum | 87 | 0.222 | 27.2% |
| Communication fraction above application maximum | 179 | 0.508 | 8.9% |

This makes the existing training target distribution much stronger than the
App-App target distribution.

## Why the existing input grid misses applications

The main input grid uses:

```text
message bytes:  2,000 to 4,000,000
wait:           0 to 1,000,000 us
sparsity:       0.1, 0.3, 0.5, 0.7, 1.0
peers/rank:     45, 134, 224, 313, 447
```

The extended grid uses:

```text
message bytes:  20, 200, 500, 1,000
wait:           5,000 to 50,000 us
sparsity:       0.01, 0.05
peers/rank:     4, 22
```

These are two disjoint Cartesian products. The dataset therefore has no large
messages with one to 22 peers and no small messages with 45 or more peers. It
also has almost no direct coverage between 4 KB and 20 KB. Message size,
fanout, and wait regime are confounded.

### Application targets and closest existing coverage

The mismatch column is the smallest possible worst multiplicative error across
communication fraction, mean message size, message rate, and byte rate among
the existing 266 profiles.

| App | Mean message B | Messages/s | GB/s | MPI fraction | Closest existing input | Worst mismatch |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| AMG | 9,472 | 310,778 | 2.944 | 0.037 | `8_20000_100000_0.1` | 2.11x |
| Beatnik | 546,432 | 50,275 | 27.472 | 0.251 | `8_400000_1000000_0.5` | 1.37x |
| Fiesta | 2,542,163 | 1,797 | 4.567 | 0.183 | `8_2000000_1000000_0.1` | 8.48x |
| Kripke | 1,281,840 | 854 | 1.095 | 0.048 | `8_400000_1000000_0.1` | 21.8x |
| Laghos | 791 | 2,419,143 | 1.913 | 0.166 | `8_2000_10000_0.1` | 2.82x |
| LAMMPS | 1,465,695 | 1,289 | 1.890 | 0.049 | `8_400000_1000000_0.1` | 14.4x |
| MiniFE | 56,312 | 17,484 | 0.985 | 0.035 | `8_40000_1000000_0.1` | 2.52x |
| MiniVite | 47,103 | 64,888 | 3.056 | 0.139 | `8_40000_1000000_0.7` | 2.05x |
| QuickSilver | 210 | 2,585 | 0.000543 | 0.454 | `8_200_25000_0.01` | 63.5x |
| TriCount | 1,432,391 | 18,839 | 26.984 | 0.540 | `8_2000000_1000000_0.3` | 1.99x |

The current executable may not match all four target statistics simultaneously.
That does not change the collection plan. The new inputs deliberately cover the
missing message-size, rate, byte-rate, and fanout regions and include controls
that vary fanout while approximately preserving average traffic.

## Retrospective equivalence diagnostic

This result did not select any input below. After each real aggressor was
matched to its nearest existing inhibitor using isolated profiles only, the
same victims responded as follows across 90 non-self directions:

| Quantity | Value |
| --- | ---: |
| Mean real App-App log slowdown | 0.103012 |
| Mean nearest-inhibitor log slowdown | 0.262619 |
| Mean matched-inhibitor bias | +0.159607 |
| Directions with larger inhibitor response | 63/90 |
| Matched-response Spearman | 0.506 |

The existing inhibitor domain preserves some ordering but remains too strong.
Collecting lower-pressure inputs is therefore justified independently of any
model architecture.

## Fixed inhibitor behavior and input math

The standard experiment launches 448 inhibitor ranks across eight nodes, with
56 inhibitor ranks per node. With `P = 448`:

```text
peers_per_rank = round(comm_sparsity * 447)
sent_messages_per_iteration = 448 * peers_per_rank
```

Every current iteration creates the deterministic target list, posts sends and
receives, waits, sleeps for `-w` microseconds, executes an MPI Barrier, and then
checks `-r`. This behavior must remain unchanged.

The proposed target cycle is:

```text
target_cycle_s = (448 * peers_per_rank) / target_message_rate_s
```

The initial waits below subtract empirical per-cycle overhead estimated from
the existing isolated runs. They must be treated as pilot values because the
large-message, low-fanout region is not currently measured.

The exact sparsities map to peer counts as follows:

| Peers/rank | Sparsity input |
| ---: | ---: |
| 0 | 0.0 |
| 1 | 0.002237136 |
| 6 | 0.013422819 |
| 8 | 0.017897092 |
| 16 | 0.035794183 |
| 22 | 0.049217002 |
| 45 | 0.100671141 |
| 134 | 0.299776286 |

Before launching, verify every value with the same rule used by C++:
`round(sparsity * 447)`. Preserve the full sparsity token in run names and
manifests.

## Collection plan overview

Execute the work in this order:

1. Repair the six missing isolated profiles.
2. Run isolated pilots for the 14 new profile/fanout inputs.
3. Run isolated pilots for the five current-executable Barrier-only inputs.
4. Apply at most one wait correction based only on isolated measurements.
5. Run four 360-second isolated validation repetitions for every finalized new
   input.
6. Co-schedule the first-wave 13 inputs (the eight profile/fanout rows plus the
   five Barrier-only rows) with all ten applications and four repetitions.
7. Co-schedule the six expanded inputs with all ten applications and four
   repetitions.
8. Backfill the 67 missing cells in the original 272 x 10 matrix after the
   higher-value new-input collection, unless rectangular completeness is needed
   first for another analysis.
9. Parse, validate, aggregate, and write a new versioned dataset. Never overwrite
   the only copy of the current CSVs.
10. Run a new Delta Response experiment root. Do not modify or reuse
    `delta_response_2/experiments/compact_v1`.

## Phase 0: repair missing isolated profiles

Run four 360-second isolated repetitions of each current input:

| Inhibitor ID | `-m` | `-w` | `-s` | Peers |
| --- | ---: | ---: | ---: | ---: |
| `8_500_5000_0.01` | 500 | 5,000 | 0.01 | 4 |
| `8_500_5000_0.05` | 500 | 5,000 | 0.05 | 22 |
| `8_2000_0_0.1` | 2,000 | 0 | 0.1 | 45 |
| `8_2000_100_0.1` | 2,000 | 100 | 0.1 | 45 |
| `8_4000_0_0.1` | 4,000 | 0 | 0.1 | 45 |
| `8_4000_100_0.1` | 4,000 | 100 | 0.1 | 45 |

Adding these six profiles makes 30 already-recorded App-Inhibitor rows eligible
for the Delta Response loader. Do not discard those existing response rows.

## Phase 1: new profile and fanout inputs

All configurations use `-c d`. The candidate ID is a human-readable alias; the
actual processed inhibitor ID remains `8_<m>_<w>_<s>`.

### First wave

Run these eight inputs first. After isolated validation, co-schedule every row
with all ten applications and four repetitions, even when the measured MPI
fraction does not exactly match the target. The mismatch must be retained as a
diagnostic column.

| Priority | Alias | `-m` B | Initial `-w` us | Peers | `-s` | Target | Reason |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `AMG-45` | 9,472 | 62,000 | 45 | 0.100671141 | AMG | Fill the 4-20 KB hole at AMG-like rate. |
| 2 | `LAG-6` | 791 | 634 | 6 | 0.013422819 | Laghos | Tiny-message, high-rate application endpoint. |
| 3 | `BEA-22` | 546,432 | 139,472 | 22 | 0.049217002 | Beatnik | Large messages and high byte rate at low fanout. |
| 4 | `QS-1` | 210 | 172,833 | 1 | 0.002237136 | QuickSilver | Lowest application message and byte rate. |
| 5 | `TRI-1` | 1,432,391 | 16,897 | 1 | 0.002237136 | TriCount | High-byte-rate, low-fanout endpoint. |
| 6 | `KRI-1` | 1,281,840 | 518,406 | 1 | 0.002237136 | Kripke | Representative large-message, very-low-rate endpoint. |
| 7 | `LAG-22-C` | 791 | 3,453 | 22 | 0.049217002 | Laghos control | Hold average traffic near `LAG-6` while changing burst fanout. |
| 8 | `AMG-16-C` | 9,472 | 21,798 | 16 | 0.035794183 | AMG control | Hold average traffic near `AMG-45` while changing burst fanout. |

First-wave co-scheduled budget:

```text
8 inputs x 10 applications x 4 repetitions = 320 runs
```

### Expanded wave

After the first wave is parsed successfully, run these six inputs with the same
ten-application, four-repetition design:

| Priority | Alias | `-m` B | Initial `-w` us | Peers | `-s` | Target | Reason |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 9 | `LAM-1` | 1,465,695 | 340,423 | 1 | 0.002237136 | LAMMPS | Exact large-message, low-rate endpoint. |
| 10 | `FIE-1` | 2,542,163 | 237,894 | 1 | 0.002237136 | Fiesta | Largest application mean message size. |
| 11 | `MV-8` | 47,103 | 52,834 | 8 | 0.017897092 | MiniVite | Medium-size, moderate-rate hole. |
| 12 | `MF-8` | 56,312 | 202,304 | 8 | 0.017897092 | MiniFE | Similar size to MiniVite but 3.7x lower rate. |
| 13 | `BEA-45-C` | 546,432 | 286,019 | 45 | 0.100671141 | Beatnik control | Fanout control for `BEA-22`. |
| 14 | `TRI-134-C` | 1,432,391 | 2,198,321 | 134 | 0.299776286 | TriCount control | Dense, infrequent bursts at similar average traffic. |

Expanded co-scheduled budget:

```text
6 inputs x 10 applications x 4 repetitions = 240 runs
```

## Phase 2: current-executable Barrier-only sweep

This is not an inhibitor redesign. With `-s 0.0`, the current executable sends
no point-to-point payload but still performs its existing Barrier every
iteration. Message size is set to one byte to minimize unused buffer allocation.

Run these five fixed inputs without wait calibration:

| Alias | `-m` | `-w` us | `-s` | Peers | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `BAR-0` | 1 | 0 | 0.0 | 0 | Maximum Barrier frequency and no configured payload. |
| `BAR-100` | 1 | 100 | 0.0 | 0 | High-frequency Barrier point. |
| `BAR-500` | 1 | 500 | 0.0 | 0 | Intermediate high-frequency point. |
| `BAR-1000` | 1 | 1,000 | 0.0 | 0 | Intermediate point. |
| `BAR-5000` | 1 | 5,000 | 0.0 | 0 | Lower-frequency, zero-payload point. |

Run three 120-second isolated pilots and four 360-second isolated validation
repetitions for all five. Then co-schedule all five with all ten applications
and four repetitions:

```text
5 inputs x 10 applications x 4 repetitions = 200 runs
```

These rows use only existing input behavior. They provide low configured
network volume at multiple MPI Barrier rates, a region absent from the current
input grid. `total_msgs` and `total_bytes` may parse as zero; that is valid and
must not be replaced with invented positive values.

## Isolated pilot protocol

### Pilot runs

For each of the 14 profile/fanout rows:

1. Run three isolated repetitions for 120 seconds.
2. Parse inhibitor MPI time, elapsed time, total sent messages, and total sent
   bytes from mpiP.
3. Compute communication fraction, mean message size, message rate, and byte
   rate using the same definitions as the existing CSVs.
4. Compare the measured profile with its target application.
5. Keep message size and sparsity fixed.
6. Apply at most one wait correction if message rate differs by more than 15%.
7. Record the initial and final wait in an immutable case manifest.

Use this isolated-only correction:

```text
w_next = w_old + 1e6 * (448 * peers) *
         (1 / target_message_rate - 1 / observed_message_rate)
```

Round to a nonnegative integer number of microseconds. If the result is
negative, use zero. Do not adjust any input using App-App outcomes or
co-scheduled slowdown.

### Validation runs

After the optional correction, run four isolated repetitions for 360 seconds.
Require:

- Correct message-size, wait, mode, and sparsity metadata.
- Correct peer count under `round(s * 447)`.
- 448 ranks and 56 ranks per node.
- Finite nonnegative MPI time, message count, and byte count.
- Runtime close to 360 seconds.
- Mean message rate within 15% of target after the one correction, when the
  target is physically reachable.
- Mean byte rate within 20% of target.
- No obvious failed or truncated repetition.

Communication-fraction mismatch is diagnostic, not a reason to redesign or
silently discard a configuration. Preserve it in the profile table.

If a configuration still misses rate badly after one correction, keep its
measured profile and mark it `target_miss`. Ask the user before spending another
calibration round.

## Co-scheduled experiment contract

### Applications

Use all ten eight-node applications from `run_configs/8_nodes.json`:

```text
amg
beatnik
fiesta
kripke
laghos
lammps
minife
minivite
quicksilver
tricount
```

Do not change application problem sizes or arguments.

### Placement and resources

Match the existing experiment:

```text
allocation nodes:            8
allocation tasks:            896
allocation tasks/node:       112
allocation memory:           240G
inhibitor ranks:             448
inhibitor ranks/node:        56
application ranks:           448
application ranks/node:      56
step memory:                 119G each
step distribution:           block:block
MPI binding:                 --mpibind=on,v
repetitions:                 4
```

Use a batch allocation distribution compatible with the existing scripts and
two exclusive `srun` steps. Do not change placement between isolated and
co-scheduled collections.

The generated batch header should use:

```text
#SBATCH --ntasks 896
#SBATCH --ntasks-per-node 112
#SBATCH --nodes 8
#SBATCH --mem 240G
#SBATCH --time 00:25:00
#SBATCH --partition pbatch
#SBATCH --distribution block:block
```

Retain `--mail-user aalasand1@unm.edu` and `--mail-type FAIL,TIME_LIMIT`.

### Environment

Use the existing Dane environment and absolute paths:

```text
modules:      gcc/10 openmpi cmake
Spack setup:  /g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh
Spack env:    /g/g90/alasandagutt1/spack_envs/beatnik/
mpiP library: /g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so
repository:   /g/g90/alasandagutt1/repos/coScheduling/
inhibitor:    /g/g90/alasandagutt1/repos/coScheduling/build/networkInhib
partition:    pbatch
```

### Inhibitor runtime

The current co-scheduled generator has a correctness bug:
`get_max_runtime()` returns the first CSV row matching an app, without filtering
to eight nodes and without computing a maximum. For Beatnik this selects the
one-node runtime. Do not reuse that logic.

Use these initial clean-exit runtime limits for the new low-pressure inputs:

```text
recommended_inhibitor_runtime = ceil(2.5 * eight_node_isolated_runtime + 60)
```

| App | Eight-node isolated s | Initial inhibitor `-r` s |
| --- | ---: | ---: |
| amg | 342.500 | 917 |
| beatnik | 397.734 | 1,055 |
| fiesta | 359.092 | 958 |
| kripke | 354.500 | 947 |
| laghos | 331.238 | 889 |
| lammps | 354.750 | 947 |
| minife | 388.250 | 1,031 |
| minivite | 371.250 | 989 |
| quicksilver | 295.000 | 798 |
| tricount | 378.000 | 1,005 |

Use at least a 25-minute batch walltime. The inhibitor runs in the background
and the application in the foreground. Record whether the inhibitor process is
still alive when the application exits. A run is invalid if the inhibitor ends
first. If that occurs, rerun that app/configuration cell with a longer `-r` and
walltime; do not average the truncated run.

Allow the inhibitor to exit cleanly through `-r` so mpiP and
`inhib_stats.json` are complete. The inhibitor profile from the co-scheduled run
may include a tail after the application exits; use the separately collected
isolated profile as the model input.

### Required outputs per run

Each data directory must contain:

```text
app_output.log
inhib_output.log
inhib_stats.json
mpip_profiles/<application profile>.mpiP
mpip_profiles/<networkInhib profile>.mpiP
run_status.json
```

`run_status.json` should be written by the wrapper and record at least:

```text
application
candidate alias
all inhibitor arguments
repetition
Slurm job ID
start timestamps
application exit timestamp and status
inhibitor exit timestamp and status
inhibitor_alive_at_application_exit
hostname list
source/config hashes
```

Adding this wrapper metadata does not alter the inhibitor.

## Experiment naming

Keep names compatible with `parse_coscheduling_runs.py`.

Isolated:

```text
8_inhib_<message>_<wait>_d_<sparsity>_<rep>
```

Co-scheduled:

```text
8_<app>_inhib_<message>_<wait>_d_<sparsity>_<rep>
```

Examples:

```text
8_inhib_9472_62000_d_0.100671141_0
8_amg_inhib_9472_62000_d_0.100671141_0
8_quicksilver_inhib_1_0_d_0.0_3
```

The existing parser accepts decimal sparsity strings in deterministic mode.
Do not insert aliases into parse-critical directory names. Store aliases in the
case manifest and status JSON.

## Generator requirements

The existing isolated and co-scheduled generators call `itertools.product` on
argument arrays. Do not encode this plan as independent arrays: it would create
many unintended combinations. The message, wait, and sparsity values are
coupled rows.

The next agent should create new explicit-case experiment generators under
`scripts/`, following the Dane experiment-script workflow:

1. Ask the user for the exact Python generator filename and the exact
   `/p/lustre2/alasandagutt1/` experiment-directory name before writing the
   generator. Those names were not specified in this task.
2. Create the experiment directory only if it does not exist.
3. Create a timestamp subdirectory.
4. Create `data/` and `slurm_scripts/` beneath the timestamp directory.
5. Use absolute paths in every generated Slurm script.
6. Generate one individual Slurm script and one matching data directory per
   explicit application/configuration/repetition case.
7. Preserve the standard email, partition, modules, Spack environment,
   placement, mpiP preload, and output redirection.
8. Write an exact case manifest before submission.
9. Do not edit `networkInhibitor/networkInhib.cpp`.

The generator may read an explicit CSV or a literal list of row dictionaries.
It must assert that the generated case count equals the planned count and that
every generated `(app, message, wait, sparsity, repetition)` key is unique.

## Existing 67-cell backfill

A complete original matrix has 272 configurations x 10 applications. The
current aggregate table is missing the following 67 cells across 32
configurations. This backfill is lower scientific priority than the new
low-pressure inputs, but it is required if a rectangular original matrix is
desired.

Run four repetitions for each listed app/configuration cell. Use the original
input exactly, not a calibrated replacement.

| Inhibitor ID | Missing count | Missing applications |
| --- | ---: | --- |
| `8_20_5000_0.01` | 1 | kripke |
| `8_200_5000_0.01` | 1 | minife |
| `8_1000_5000_0.05` | 1 | minife |
| `8_1000_10000_0.01` | 1 | tricount |
| `8_2000_1000_0.1` | 8 | amg, beatnik, fiesta, kripke, laghos, lammps, minife, minivite |
| `8_2000_100000_0.5` | 1 | minife |
| `8_2000_1000000_0.3` | 1 | minife |
| `8_4000_1000_0.1` | 5 | beatnik, fiesta, laghos, minife, quicksilver |
| `8_4000_1000_0.3` | 1 | quicksilver |
| `8_4000_1000_0.5` | 1 | minivite |
| `8_4000_1000_1.0` | 1 | minivite |
| `8_4000_10000_0.7` | 1 | minivite |
| `8_4000_100000_1.0` | 1 | minivite |
| `8_4000_1000000_0.1` | 1 | fiesta |
| `8_20000_100_0.1` | 1 | tricount |
| `8_20000_100_1.0` | 1 | minife |
| `8_20000_1000_0.1` | 1 | quicksilver |
| `8_40000_100_0.3` | 1 | minivite |
| `8_200000_1000_0.3` | 1 | minivite |
| `8_400000_1000000_1.0` | 1 | minife |
| `8_2000000_0_0.1` | 1 | quicksilver |
| `8_2000000_0_0.5` | 1 | amg |
| `8_2000000_0_0.7` | 1 | amg |
| `8_2000000_0_1.0` | 1 | amg |
| `8_2000000_1000000_0.1` | 1 | amg |
| `8_4000000_100000_0.1` | 1 | minife |
| `8_500_5000_0.01` | 1 | quicksilver |
| `8_500_5000_0.05` | 1 | quicksilver |
| `8_2000_0_0.1` | 8 | amg, beatnik, fiesta, kripke, laghos, lammps, minife, quicksilver |
| `8_2000_100_0.1` | 9 | amg, beatnik, fiesta, kripke, laghos, lammps, minife, quicksilver, tricount |
| `8_4000_0_0.1` | 6 | amg, kripke, laghos, minife, quicksilver, tricount |
| `8_4000_100_0.1` | 5 | fiesta, kripke, lammps, minivite, quicksilver |

Backfill budget:

```text
67 cells x 4 repetitions = 268 co-scheduled runs
```

Some original high-pressure cells may exceed a 25-minute walltime. For
backfill only, use prior successful runtimes for that exact app/configuration
neighborhood or increase `-r` and walltime. Never treat a timed-out application
as a valid slowdown observation.

## Planned run counts

| Work package | Isolated runs | Co-scheduled runs |
| --- | ---: | ---: |
| Six missing profile repairs | 24 | 0 initially |
| Fourteen new profile/fanout inputs, short pilots | 42 | 0 |
| Five Barrier-only inputs, short pilots | 15 | 0 |
| Nineteen finalized new inputs, 360-second validation | 76 | 0 |
| First-wave eight inputs | 0 | 320 |
| Expanded six inputs | 0 | 240 |
| Barrier-only five inputs | 0 | 200 |
| Original-matrix backfill | 0 | 268 |

The high-value new-input collection is 760 co-scheduled runs. Including the
full original-matrix backfill gives 1,028 co-scheduled runs.

## Parsing and validation

Use `notebooks/utils/parse_coscheduling_runs.py` and
`notebooks/utils/mpip_parse.py` as the starting point, but validate their output
against the raw run roster.

Important parser facts:

- `parse_inhibitor_isolated()` expects exactly one mpiP file.
- `parse_inhibitor_coscheduled_allProfiles()` expects exactly two mpiP files and
  is the appropriate full-profile parser.
- `parse_inhibitor_coscheduled()` contains an inconsistent `mpiP_file` versus
  `mpiP_files` key path and should not be trusted without a test.
- The all-profile parser reports failures but does not raise at the end. A new
  collection validator must fail the pipeline when expected runs are absent.
- Barrier-only rows may have no aggregate sent table. Preserve their sent count
  and bytes as zero.

For each expected run, require:

- Exactly one run directory matching the manifest key.
- No symlink substitution.
- Application exit status zero.
- Inhibitor active when the application exited.
- Correct number of mpiP files.
- Correct rank count and command line in mpiP metadata.
- Finite application and MPI times.
- Application runtime greater than zero.
- Inhibitor JSON arguments identical to the case manifest.
- No duplicate `(app, inhibitor ID, repetition)` key.
- All four repetitions before aggregation, or an explicit documented rerun.

Do not silently omit parse failures and then average the remaining repetitions.

## Aggregation and dataset construction

Create a new versioned data directory rather than editing the only current copy
in place. Preserve input hashes, raw run manifests, parser version, and source
hashes.

### Isolated inhibitor profiles

For each finalized new input, aggregate four valid isolated runs using the same
definitions as the current `inhibitors.csv`:

```text
inhib_id
msg_size
wait_time
comm_sparsity
mpi_time
comm_frac
total_msgs
total_bytes
```

Also retain a separate run-level table with means, medians, standard deviations,
and repetition IDs. Do not discard variance after writing the compatibility
CSV.

### App-Inhibitor responses

For each app/configuration cell:

1. Compute application elapsed time from each valid co-scheduled repetition.
2. Divide by the matching eight-node isolated application baseline using the
   established repository aggregation convention.
3. Retain all four run-level slowdown values.
4. Write the compatibility `job_inh.csv` row from the declared aggregate used by
   the current pipeline.
5. Preserve values at or above one according to the existing observation
   policy; do not invent latent values below the floor.

The 19 new inputs should add:

```text
19 isolated inhibitor profiles
19 x 10 = 190 aggregate App-Inhibitor response rows
19 x 10 x 4 = 760 raw co-scheduled repetitions
```

After all repair and backfill work, the original 272 configurations should have
2,720 aggregate cells. Adding 19 new inputs yields 291 profiles and 2,910
aggregate response cells.

### IDs

Use exactly:

```text
inhib_id = 8_<final_message>_<final_wait>_<full_sparsity_token>
```

The isolated and response tables must use identical strings. Do not normalize
`0.100671141` to `0.1`; those inputs have the same current peer count but are
different declared controls and must remain traceable.

## Post-collection coverage analysis

Before training a model, recompute all of the following on the new versioned
dataset:

1. Application and inhibitor quantiles for MPI time, communication fraction,
   total messages, total bytes, mean message size, message rate, and byte rate.
2. Fraction of inhibitors above the application maximum in communication
   fraction and byte rate.
3. Count inside the joint application base-feature envelope.
4. Per-application nearest-inhibitor distance using the exact Delta Response
   physical transform.
5. Per-application worst multiplicative mismatch over communication fraction,
   mean message size, message rate, and byte rate.
6. Response mean, quantiles, floor fraction, and victim-specific distributions
   for old high-pressure, new profile/fanout, and Barrier-only families.
7. Fanout-control contrasts: `LAG-6` versus `LAG-22-C`, `AMG-45` versus
   `AMG-16-C`, `BEA-22` versus `BEA-45-C`, and `TRI-1` versus `TRI-134-C`.
8. Repetition variability and failure rate by family.

The new data improve coverage if they materially increase joint application
envelope coverage, reduce nearest-profile mismatch for the poorly covered
applications, and add lower-response/floor-rich App-Inhibitor examples.

Do not choose or delete configurations based on historical App-App prediction
error.

## Model rerun protocol

After the new CSV version is sealed:

1. Create a new Delta Response experiment root.
2. Record hashes for jobs, inhibitors, responses, source, config, and fold
   manifests.
3. Keep all ten applications in the `random_split` fitting condition.
4. Add an inhibitor-family column distinguishing old main, old extended, new
   profile/fanout, and Barrier-only rows.
5. Hold out complete families in development validation.
6. Report unweighted and application-profile-weighted metrics so the 240-point
   old main grid cannot numerically drown the 19 new inputs.
7. Report floor and positive strata separately.
8. Freeze the model before any retrospective App-App evaluation.
9. Continue to label historical `pair.csv` results diagnostic-only.
10. Use a newly collected immutable App-App holdout for any fresh generalization
    claim.

Do not modify, add files to, or reuse the sealed
`delta_response_2/experiments/compact_v1` root.

## Completion criteria

The data-fix project is complete only when:

- The inhibitor source is unchanged.
- Six missing isolated profiles have four valid repetitions each.
- All 19 new inputs have finalized isolated profiles from four repetitions.
- All 19 new inputs have four valid co-scheduled repetitions for all ten apps.
- Every co-scheduled run confirms the inhibitor remained active for the full
  application execution.
- The 67-cell original backfill is either complete or explicitly deferred with
  a manifest of missing keys.
- Raw and aggregate row counts match the declared design.
- Parser failures are zero or represented by successful reruns.
- Run-level data and aggregate compatibility CSVs both exist.
- Input and output hashes are recorded.
- The stale NPZ is not accidentally used.
- Coverage statistics are recomputed before model fitting.
- A new experiment root is used for the model rerun.
- App-App labels were not used to select inhibitor inputs or calibrate waits.

## Do not do these things

- Do not redesign or edit `networkInhib`.
- Do not add random-mode runs to this collection.
- Do not turn the row-wise input list into a Cartesian product.
- Do not reuse the broken first-match runtime lookup.
- Do not accept a run where the inhibitor stopped before the application.
- Do not silently average fewer than four repetitions.
- Do not overwrite the current raw or aggregate dataset without a versioned
  backup and hashes.
- Do not regenerate `processed_data.npz` by merely deleting it and trusting the
  old preprocessor without verifying the resulting inhibitor roster.
- Do not use historical App-App labels to remove, tune, or reprioritize these
  configurations.
- Do not report the historical pair result as an untouched test.

## Information the next agent must request before writing launch scripts

The scientific matrix and runtime contract are specified here. Two operational
names remain intentionally unset because the user did not request launch-script
implementation in this task:

1. The exact Python generator filename under `scripts/`.
2. The exact experiment directory name under
   `/p/lustre2/alasandagutt1/`.

Ask the user for those two names before creating Dane experiment scripts. No
additional scientific decision is required to start implementing the fixed
current-inhibitor collection.
