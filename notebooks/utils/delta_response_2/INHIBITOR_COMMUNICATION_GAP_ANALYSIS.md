# Inhibitor-to-Application Communication Gap Analysis

> Execution update: the user does not want the inhibitor redesigned. The
> authoritative current-executable-only collection plan is `fix_missing_data.md`.
> Mechanism-extension recommendations later in this document are retained only
> as historical analysis and must not be implemented.

## Scope and evidence boundary

This analysis addresses whether the synthetic network inhibitor represents the
communication behavior of the ten real applications well enough for an
App-Inhibitor model to transfer to App-App pairings.

The configuration recommendations use only:

- `few_shot/data/jobs.csv`: isolated application profiles.
- `few_shot/data/inhibitors.csv`: isolated inhibitor profiles and controls.
- `few_shot/data/job_inh.csv`: coverage and response-distribution diagnostics.
- `networkInhibitor/networkInhib.cpp`: the implemented communication mechanism.
- Existing run configurations and generation scripts.

No App-App slowdown label was used to choose a configuration, message size,
wait, fanout, or promotion criterion. The final retrospective comparison with
`pair.csv` is reported only after the profile-derived design is fixed. This is
benchmark design, not target-label leakage: the objective is to make the
synthetic benchmark reproduce isolated application communication behavior.

## Conclusion

The inhibitor set is too intense for the target application domain, but
intensity is not the only problem.

1. The inhibitor distribution is dominated by high-pressure configurations.
   Of 266 profiles, 179 (67.3%) exceed the largest application communication
   fraction and 155 (58.3%) exceed the largest application byte rate. A total
   of 143 exceed both limits.
2. The existing control grid is structurally confounded. Messages of 2 KB or
   larger use at least 45 destinations per rank, while fanouts below 45 were
   measured only for messages of 1 KB or smaller. This omits the large-message,
   low-rate regime occupied by Fiesta, Kripke, and LAMMPS, and the medium-size,
   sparse regime occupied by MiniFE and MiniVite.
3. Aggregate profile matching is necessary but not sufficient. The inhibitor
   uses a fixed random directed graph, synchronized bursts, two `Waitall`
   calls, a sleep outside MPI, and a global barrier every iteration. Real MPI
   time can instead represent collectives, synchronization, load imbalance,
   or waiting without equivalent network injection.
4. QuickSilver demonstrates an unattainable regime under the current controls:
   45.4% MPI time but only 0.000543 GB/s and 2,585 messages/s. A user-space
   sleep can lower traffic, but it also lowers MPI residency. A synchronization
   or delayed-progress control is required to vary these independently.
5. The highest-value next data are not more points in the existing dense,
   high-pressure grid. They are sparse cross-grid bridges, followed by new
   mechanism families for synchronization, topology, collectives, message-size
   mixtures, and burstiness.

The immediate recommendation is to run the eight-row minimum isolated design
below, promote only the AMG-like and Laghos-like rows that pass profile gates,
and implement MPI-residency control before collecting full App-Inhibitor data
for the remaining unmatched applications.

## Model and transfer failure

Delta Response 2 predicts directional slowdown from isolated profiles and
App-Inhibitor responses without fitting App-App outcomes. Its frozen compact
low-rank potential has a worst primary App-Inhibitor development log MAE of
`0.166523`. On the historical 90-direction non-self App-App diagnostic, its log
MAE is `0.255138`, signed log bias is `+0.245545`, and its mean predicted log
slowdown is `0.348557` versus a true mean of `0.103012`.

The model retains moderate ordering (`Spearman = 0.362726`) but predicts the
wrong scale. This is consistent with a synthetic conditional-response shift:
the model learns which inhibitor profiles are relatively disruptive, then
applies an inhibitor-scale response to real application aggressors.

## Data audit

| Item | Count | Note |
| --- | ---: | --- |
| Applications | 10 | All have isolated profiles. |
| Isolated inhibitor profiles | 266 | Six intended configurations lack a profile. |
| Raw App-Inhibitor rows | 2,653 | Includes rows whose inhibitor profile is absent. |
| Valid Delta Response 2 rows | 2,623 | 30 rows are rejected for missing inhibitor profiles. |
| App-Inhibitor floor rows | 389 (14.83%) | Observed slowdown equals one. |
| Historical non-self App-App directions | 90 | Retrospective diagnostic only. |

### Stale processed archive

`few_shot/data/processed_data.npz` contains only 214 inhibitors. It excludes all
30 available profiles from the small-message, low-fanout extended sweep. The
included subset has median communication fraction `0.9636`; the omitted subset
has median `0.0273`.

This does not affect Delta Response 2, which reads the CSV files directly. It
does affect the older `few_shot` pipeline if that archive is used. Its
preprocessor skips work whenever the NPZ already exists, so the archive will
not update automatically after CSV additions. That pipeline should not be used
for a new result until the archive is regenerated with provenance and its
inhibitor roster is checked.

### Missing isolated profiles

These intended configurations have App-Inhibitor rows but no isolated profile:

```text
8_500_5000_0.01
8_500_5000_0.05
8_2000_0_0.1
8_2000_100_0.1
8_4000_0_0.1
8_4000_100_0.1
```

Rerunning them would recover 30 currently rejected response rows. This is a
data-repair task, not the main scientific coverage improvement.

## Metric definitions

The CSV profiles provide MPI time, communication fraction, total sent messages,
and total sent bytes. The useful derived quantities are:

```text
runtime_s       = mpi_time / comm_frac
mean_message_B  = total_bytes / total_msgs
message_rate_s  = total_msgs / runtime_s
byte_rate_B_s   = total_bytes / runtime_s
```

The profiles are approximately 360-second observations, so totals and rates
carry almost the same information. Mean message size and message rate determine
byte rate exactly. MPI fraction adds a different axis: how much elapsed time is
spent inside MPI, not necessarily how much pressure is placed on the network.

## Distribution comparison

| Statistic | Applications | Inhibitors | Inhibitor/app median ratio |
| --- | ---: | ---: | ---: |
| Median MPI time (s) | 53.27 | 338.05 | 6.35 |
| Median communication fraction | 0.152 | 0.939 | 6.16 |
| Median total messages | 6.95 million | 75.71 million | 10.89 |
| Median total bytes | 0.839 TB | 23.175 TB | 27.61 |
| Median mean message size | 301 KB | 40.0 KB | 0.13 |
| Median message rate | 18.2 thousand/s | 210.3 thousand/s | 11.58 |
| Median byte rate | 2.43 GB/s | 64.31 GB/s | 26.48 |

Marginal range coverage can therefore look acceptable while the empirical mass
is badly placed. Only 34 of 266 inhibitors (12.8%) lie jointly inside the
application ranges for the four base features. Only 34 also lie jointly inside
the application communication-fraction and byte-rate ranges.

The high-pressure points are not harmless extra coverage. Across inhibitor
profiles, the Spearman correlations between mean victim log slowdown and
communication fraction, mean message size, and byte rate are `0.679`, `0.814`,
and `0.801`, respectively.

| Inhibitor stratum | Profiles | Mean victim log slowdown | Mean floor fraction |
| --- | ---: | ---: | ---: |
| Byte rate at or below application maximum | 111 | 0.184 | 24.8% |
| Byte rate above application maximum | 155 | 0.580 | 7.9% |
| Communication fraction at or below application maximum | 87 | 0.222 | 27.2% |
| Communication fraction above application maximum | 179 | 0.508 | 8.9% |

The model is consequently trained mostly on configurations that produce larger
responses and much less censoring than App-App pairings.

## Current grid confounding

The intended main sweep is:

```text
message bytes:  2,000 to 4,000,000
wait:           0 to 1,000,000 us
peer fraction:  0.1, 0.3, 0.5, 0.7, 1.0
peers/rank:     45, 134, 224, 313, 447
```

The intended extended sweep is:

```text
message bytes:  20, 200, 500, 1,000
wait:           5,000 to 50,000 us
peer fraction:  0.01, 0.05
peers/rank:     4, 22
```

These are disjoint blocks, not a factorial bridge. Message size is therefore
confounded with fanout and wait. In particular, the dataset contains no:

- Large messages with one to 22 peers.
- Small messages with 45 or more peers.
- Sparse small messages with waits above 50 ms.
- Medium 4-20 KB messages at intermediate fanout.
- Direct control of one, two, six, eight, or sixteen peers.

At 448 ranks, the smallest measured inhibitor message rate is 11,956/s. This is
still above QuickSilver (2,585/s), Fiesta (1,797/s), Kripke (854/s), and LAMMPS
(1,289/s).

## Application-specific coverage

The table below compares each application with the existing inhibitor that
minimizes the worst multiplicative mismatch over communication fraction, mean
message size, message rate, and byte rate. A value of `8.48x`, for example,
means that at least one of those four dimensions remains wrong by a factor of
8.48 even for the best existing inhibitor.

| Application | Mean message | Messages/s | GB/s | MPI fraction | Best existing inhibitor | Worst mismatch |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| AMG | 9,472 B | 310,778 | 2.944 | 0.037 | `8_20000_100000_0.1` | 2.11x |
| Beatnik | 546,432 B | 50,275 | 27.472 | 0.251 | `8_400000_1000000_0.5` | 1.37x |
| Fiesta | 2,542,163 B | 1,797 | 4.567 | 0.183 | `8_2000000_1000000_0.1` | 8.48x |
| Kripke | 1,281,840 B | 854 | 1.095 | 0.048 | `8_400000_1000000_0.1` | 21.8x |
| Laghos | 791 B | 2,419,143 | 1.913 | 0.166 | `8_2000_10000_0.1` | 2.82x |
| LAMMPS | 1,465,695 B | 1,289 | 1.890 | 0.049 | `8_400000_1000000_0.1` | 14.4x |
| MiniFE | 56,312 B | 17,484 | 0.985 | 0.035 | `8_40000_1000000_0.1` | 2.52x |
| MiniVite | 47,103 B | 64,888 | 3.056 | 0.139 | `8_40000_1000000_0.7` | 2.05x |
| QuickSilver | 210 B | 2,585 | 0.000543 | 0.454 | `8_200_25000_0.01` | 63.5x |
| TriCount | 1,432,391 B | 18,839 | 26.984 | 0.540 | `8_2000000_1000000_0.3` | 1.99x |

The large-message, low-rate hole is the clearest missing region. QuickSilver is
not merely outside the grid; its low-traffic/high-MPI combination is outside
the mechanism currently generated by a sleep outside MPI.

## Retrospective profile-equivalence check

This check uses the historical App-App outcomes only as diagnosis after the
configuration design above. For each real aggressor, its nearest existing
inhibitor was selected using isolated communication features only. The response
of the same victim to that inhibitor was then compared with its response to the
real aggressor across all 90 non-self directions.

| Quantity | Value |
| --- | ---: |
| Mean real App-App log slowdown | 0.103012 |
| Mean nearest-inhibitor log slowdown | 0.262619 |
| Mean nearest-inhibitor minus real bias | +0.159607 |
| Directions with a larger inhibitor response | 63/90 |
| Spearman between matched inhibitor and real responses | 0.506 |

A top-ten nearest-inhibitor average is even higher at `0.300857`. The nearest
profile preserves some ordering but applies too much pressure. This supports
the model result: isolated aggregates contain useful rank information, yet do
not establish conditional response equivalence.

## Configuration math

The current experiment uses 448 inhibitor ranks. For peer count `k`:

```text
k = round(comm_sparsity * 447)
messages_per_iteration = 448 * k
target_cycle_s = (448 * k) / target_application_message_rate
```

The requested wait is less than the target cycle because each iteration also
contains target construction, MPI communication, waits, and a barrier. Initial
wait estimates subtract overhead interpolated from existing isolated profiles.
They are pilot values, not assumed final controls.

The sparsities below use `k / 447` with enough digits to recover the intended
peer count through the current floating-point parser:

| Peers | Sparsity |
| ---: | ---: |
| 1 | 0.002237136 |
| 6 | 0.013422819 |
| 8 | 0.017897092 |
| 16 | 0.035794183 |
| 22 | 0.049217002 |
| 45 | 0.100671141 |
| 134 | 0.299776286 |

The benchmark should eventually expose integer peer count directly so that
fanout is not encoded indirectly through a rounded float.

## Recommended minimum design

All rows use deterministic mode `d`. Run all eight as isolated pilots. Only the
first two should proceed directly to full App-Inhibitor collection after they
pass the profile gates; Beatnik is conditional. The other rows diagnose missing
coverage or mechanism limits before expensive co-scheduled collection.

| Priority | ID | Message B | Wait us | Peers | Sparsity | Purpose | Initial action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `AMG-45` | 9,472 | 62,000 | 45 | 0.100671141 | AMG-like 4-20 KB bridge; predicted MPI fraction 0.038 vs 0.037 target. | Pilot, then full if matched. |
| 2 | `LAG-6` | 791 | 634 | 6 | 0.013422819 | Laghos-like tiny-message/high-rate endpoint; predicted 0.176 vs 0.166. | Pilot, then full if matched. |
| 3 | `BEA-22` | 546,432 | 139,472 | 22 | 0.049217002 | Beatnik-like high byte rate at low fanout; predicted 0.287 vs 0.251. | Pilot; full only if matched. |
| 4 | `QS-1` | 210 | 172,833 | 1 | 0.002237136 | Lowest-rate sentinel; predicted MPI fraction about 0.001 vs 0.454. | Isolated only pending MPI-residency mode. |
| 5 | `TRI-1` | 1,432,391 | 16,897 | 1 | 0.002237136 | TriCount traffic-rate endpoint; predicted 0.277 vs 0.540. | Isolated only pending MPI-residency mode. |
| 6 | `KRI-1` | 1,281,840 | 518,406 | 1 | 0.002237136 | Representative large-message/very-low-rate endpoint for Kripke/LAMMPS. | Isolated only pending MPI-residency mode. |
| 7 | `LAG-22-C` | 791 | 3,453 | 22 | 0.049217002 | Same Laghos average traffic with a different burst fanout. | Isolated fanout control. |
| 8 | `AMG-16-C` | 9,472 | 21,798 | 16 | 0.035794183 | Same AMG average traffic with a different burst fanout. | Isolated fanout control. |

The paired controls are important. They hold message size and long-run message
rate approximately fixed while changing how many destinations are contacted in
each synchronized burst. This tests whether average traffic alone determines
interference.

## Recommended expanded design

Run these after the minimum pilot if resources permit:

| Priority | ID | Message B | Wait us | Peers | Sparsity | Purpose |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 9 | `LAM-1` | 1,465,695 | 340,423 | 1 | 0.002237136 | Exact LAMMPS large-message/low-rate target. |
| 10 | `FIE-1` | 2,542,163 | 237,894 | 1 | 0.002237136 | Exact Fiesta message-size endpoint. |
| 11 | `MV-8` | 47,103 | 52,834 | 8 | 0.017897092 | MiniVite medium-size/moderate-rate hole. |
| 12 | `MF-8` | 56,312 | 202,304 | 8 | 0.017897092 | MiniFE at a similar size but 3.7x lower rate. |
| 13 | `BEA-45-C` | 546,432 | 286,019 | 45 | 0.100671141 | Beatnik fanout control at matched average traffic. |
| 14 | `TRI-134-C` | 1,432,391 | 2,198,321 | 134 | 0.299776286 | Dense/bursty TriCount control at matched average traffic. |

The complete machine-readable list, including repair rows, is in
`recommended_inhibitor_configurations.csv`.

## Measurement budget

| Design | Isolated work | Full App-Inhibitor work |
| --- | --- | --- |
| Data repair | Six configurations x four repetitions | Only missing victim/configuration rows after profiles validate. |
| Minimum pilot | Eight configurations x three 120-second repetitions | None during pilot. |
| Minimum validation | `AMG-45`, `LAG-6`, and `BEA-22` x four 360-second repetitions | First two accepted rows: 10 victims x four repetitions x two configurations = 80 runs. Beatnik adds 40 only if it passes. |
| Expanded | All 14 new rows x four 360-second isolated repetitions | Promote only rows satisfying the same profile gate. |
| Mechanism study | `QS-1`, `TRI-1`, `FIE-1`, and `MV-8` with MPI-residency control | No full sweep until the extended mechanism matches isolated targets. |

The staged budget avoids spending most of the allocation on co-scheduled labels
for configurations already known not to represent their target application.

## Pilot and promotion protocol

### Stage 1: isolated pilots

1. Run three 120-second isolated repetitions for every minimum-design row.
2. Estimate mean message size, message rate, byte rate, MPI fraction, iteration
   period, and between-run variation.
3. Adjust only wait time, using isolated measurements and the fixed target
   application profile.
4. Run four 360-second validation repetitions after the wait is fixed.

An isolated rate correction is:

```text
w_next = w_old + 1e6 * (448 * k) * (1/R_target - 1/R_observed)
```

Clamp negative waits to zero. If a row cannot match at zero wait, its fanout or
mechanism must change rather than silently accepting the mismatch.

### Stage 2: predeclared profile gate

Promote a row to App-Inhibitor collection only if its isolated validation means
satisfy all of:

- Mean message size within 10% of target.
- Aggregate message rate within 15%.
- Aggregate byte rate within 20%.
- MPI-fraction absolute error no greater than `max(0.02, 0.2 * target)`.
- No clear allocation-to-allocation drift.

These gates use no App-App outcome.

### Uncertainty

- `jobs.csv` and `inhibitors.csv` contain aggregate means without run-level
  confidence intervals. The target rates are ratios of those aggregate means.
- `AMG-45` interpolates a well-sampled part of the timing surface. Large-message
  rows below the observed minimum fanout extrapolate timing overhead and may
  require substantial wait correction.
- mpiP message totals and MPI fraction summarize different aspects of a run.
  Matching both summaries does not prove matching peak injection, collectives,
  topology, or temporal overlap.
- The profile gates establish synthetic-profile fidelity, not App-App predictive
  validity. That still requires separately collected and properly split pair
  measurements.

### Stage 3: co-scheduled collection

For each promoted configuration, run all ten victim applications with four
repetitions. The inhibitor must remain active for the full application run.
Keep both the inhibitor's isolated profile and per-run telemetry; do not reduce
the new family immediately to one cross-run mean.

Do not collect a full ten-application sweep for a row known to fail the MPI
fraction gate. That would add more labels from the wrong synthetic mechanism,
not improve benchmark coverage.

## Required mechanism extensions

### 1. Independent MPI-residency control

The current sleep occurs outside MPI. Add a mode that can increase time blocked
inside MPI without increasing payload volume, for example deterministic rotating
rank skew, delayed receive posting, or delayed collective entry. Initial extra
MPI-residency estimates per cycle are:

| Target | Approximate additional MPI time/cycle |
| --- | ---: |
| QuickSilver | 78.5 ms |
| TriCount | 6.3 ms |
| Kripke | 19.2 ms |
| LAMMPS | 10.3 ms |
| Fiesta | 34.4 ms |
| MiniVite | 5.6 ms |
| MiniFE | 4.8 ms |

These are isolated pilot starting points, not final calibrated settings.

### 2. Absolute-rate pacing

Replace sleep-after-completion with a deadline-based target-rate mode. Advance a
monotonic deadline by the requested cycle period and report missed deadlines.
This separates requested traffic rate from communication duration and avoids
systematic drift.

### 3. Communication topology families

Add complete, separately identifiable families:

- One-peer ring and nearest-neighbor stencil.
- Balanced fixed regular graph.
- Rotating peer graph without an extra full-graph `MPI_Bcast` every iteration.
- Hierarchical intra-node/inter-node patterns.
- Collective phases such as Barrier, Allreduce, and Alltoall.

The current fixed graph has equal outdegree but variable indegree. It is not a
generic model of application communication topology.

### 4. Message-size distributions

A fixed message size matches only the mean. Add seeded two-point mixtures first,
then histogram replay if application traces become available. At minimum, test a
small control-message plus large payload mixture around eager/rendezvous protocol
thresholds.

### 5. Burst and overlap controls

Expose messages per burst, barrier cadence, phase jitter, receive-first versus
send-first posting, and `Waitsome`/overlap modes. The present lockstep loop makes
all ranks communicate and synchronize in every iteration.

### 6. Better telemetry

Record actual sent messages and bytes, per-iteration period distribution, time
in each MPI call family, peer count, cross-node edge count, missed pacing
deadlines, and peak as well as mean rate. The current JSON effective-bandwidth
value counts both send and receive volume, while `total_bytes` in the profile is
sent volume; do not compare those values as if they had the same definition.

## Collection prerequisites

Fix these before launching the new co-scheduled runs:

1. `generate_coscheduled_inhib_scripts_multinode.py:get_max_runtime()` returns
   the first row matching an application. It neither filters to eight nodes nor
   computes a maximum. For Beatnik, this can stop the inhibitor before the
   application completes, contaminating the response.
2. Both generators use a Cartesian product of argument lists. The recommended
   design is row-wise and coupled; represent it as explicit cases or one config
   per row to avoid a large unintended cross-product.
3. Include communication mode and preferably integer peer count in every
   configuration ID and parsed table.
4. Preserve run-level rows and failure metadata. Only 47 current inhibitor
   configurations have all ten applications and all four repetitions in the
   retained raw audit.
5. Regenerate any derived archive from sealed CSV inputs rather than relying on
   the `processed_data.npz` existence check.

## Model-development use of the new data

The new configurations should change validation design as well as add rows.

1. Keep the old high-pressure family, the profile-matched family, and each new
   mechanism family identifiable.
2. Hold out complete mechanism families, not random rows, during development.
3. Report unweighted and application-profile-weighted metrics. Do not let 240
   dense-grid points numerically drown a small application-like family.
4. Stratify by floor, communication fraction, byte rate, and mechanism.
5. Test whether the model predicts scale as well as ordering on a newly
   collected App-App development set, then freeze before a separate untouched
   App-App evaluation.
6. Treat a failure of matched synthetic configurations to reproduce real
   interference as a benchmark finding. Do not compensate for it solely with a
   post-hoc model calibrator.

## Expected decision after the pilot

- If `AMG-45` and `LAG-6` match their profiles and produce application-like
  floor rates and response scales, expand the row-wise profile-matched family.
- If the fanout controls produce materially different responses at matched
  aggregate traffic, topology and burst descriptors must become model inputs
  and validation families.
- If MPI-residency variants reproduce QuickSilver-like profiles while ordinary
  variants do not, split active traffic pressure from synchronization pressure
  in both the benchmark and model features.
- If even mechanism-matched synthetic rows remain much more disruptive, collect
  network counters and temporal traces before adding model capacity. The four
  aggregate features are then insufficient to define interference equivalence.
