---
name: coscheduling-data-analyzer
description: Use when analyzing the Dane node-sharing (co-scheduling) data in this repo — processed_data/ CSVs, raw run_data/ logs, the filtered-inhibitor model pipelines, or the "Filtered-Inhibitor Pipeline Comparison" artifact — especially when producing numbers, tables, or figures for Akhil's dissertation chapter (Dissertation/network_interference_modeling.tex) or answering requests from the dissertation session.
---

# coScheduling data analyzer

This skill covers the data analyst role for Akhil's node-sharing dissertation chapter: answering fact checks, computing
statistics, and making publication figures from this repo.

**Peer sessions** (names may change; use `ListAgents`, and reply with `SendMessage` to the incoming `from=` name):
- `coScheduling-chapter-director`: the current main requester. It sends numbered requests (e.g. S1–S6) and
  drafts or reviews chapter sections.
- `dissertation-d8`: the earlier manuscript session. It drafts `network_interference_modeling.tex` from
  `perf_degrad_chapter_plan.md`.
- `coScheduling-experiment-scripts-dane`: writes the Dane job scripts. Its placement and launch facts are recorded
  in findings §1 and §6.

**Start here when taking over:** read `references/findings.md` in full. §6b lists open work. The last state, as of
2026-11-07: all requests answered, and the isolated-inhibitor socket data (S2) is pending because its run dirs were
empty.

Read `references/findings.md` before answering anything numeric. It records what has already been verified, the
confounds, and where each number comes from.

## Environment (this machine)

- **Python:** always use `/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python` (Python 3.14, numpy, pandas 3,
  matplotlib, scipy, torch, sklearn).
  - `tabulate` is **not** installed. Don't call `DataFrame.to_markdown`, and don't install packages without asking.
    The scripts below include a small `md()` writer.
- **Repo:** `/home/akhil/hpcResearch/repos/coScheduling`, branch `dane` (analysis baseline commit 5eea7198).
- **Dissertation repo:** `/home/akhil/hpcResearch/repos/Dissertation`.
  - Chapter plan: `perf_degrad_chapter_plan.md`.
  - Chapter source: `network_interference_modeling.tex`.
  - Figures go to `figures/node_sharing/` as PDF.
- **Not on this machine:** the model experiment roots `~/hpcResearch/experiment_outputs/*_full_20261017`. They were
  produced under `/nfs/MLClass/aalasandagutti/...`, which is not mounted. Use the artifact data (below) instead, or ask
  the user to copy the roots over.
- **No Dane access.** Hardware specs are not in the logs. Cite LLNL and TOP500 for them; see findings.
- Use the scratchpad for temporary files. Render PDFs with `pdftoppm -png -r 110 -singlefile f.pdf out` and look at
  them before reporting.

## Where things are

| What | Path |
|---|---|
| Aggregated data (one row per config; mean/median/std over replicates) | `processed_data/*_agg.csv` |
| Replicate-level data | `processed_data/data_{isolated,inhib_isolated,inhib_coscheduled,coscheduled}.csv` |
| **Canonical** modeling snapshot (MPI time) | `notebooks/utils/delta_response_filteredInhibs_biDirectional/data/{training,evaluation}/` (`jobs.csv`, `inhibitors.csv`, `job_inh.csv`, `pair.csv`) |
| Canonical snapshot (total time) | `notebooks/utils/delta_response_filteredInhibs_biDirectional_totalTime/data/...` (`pair.csv` has `raw_app_time_ratio_*`) |
| Legacy snapshot (footnote only) | `notebooks/utils/few_shot/data/` |
| Socket-pinned isolated re-runs (Nov 2026) | `run_data/experiments_coscheduling_isolated_socket/20261107_180934/` (see findings §6) |
| Raw logs, ~62 GB (mpiP profiles, mpibind output, job scripts) | `run_data/experiments_coscheduling_{isolated,inhib_isolated,inhib_coscheduled,app_coscheduled}/<timestamp>/{data,slurm_scripts}/` |
| Run configs | `run_configs/8_nodes.json`, `1_nodes.json`, `8_nodes_inhib{,_extended,_missingRange}.json` |
| Job-script generators | `scripts/generate_*_multinode.py` |
| Inhibitor source | `networkInhibitor/networkInhib.cpp` |
| Parsers | `notebooks/utils/parse_coscheduling_runs.py`, `mpip_parse.py` |
| Model pipelines | `notebooks/utils/delta_response_filteredInhibs{,_totalTime,_biDirectional,_biDirectional_totalTime}/`. Each README has a design spec plus a results section. |

## Existing analysis code (reuse and extend; don't rewrite)

Everything lives in `scripts/dissertation_figs/`. Numbers go to `output/`.

| File | What it does |
|---|---|
| `common.py` | Paths, `APPS` order, display names (`NAME`: amg→AMG2023, tricount→TriC, …), loaders (`jobs()`, `inhibitors()`, `job_inh()`, `pairs_mpi()`, `pairs_total()`), `directional()`, `directional_from_processed()` (used for 1-node), `selection_features()`, `set_style()`, Okabe–Ito palette `OI`, `FULL_W` = 6.0, `HALF_W` = 3.0, `save()`. |
| `analysis_batch1.py` | Isolated profiles, campaign counts, clipping, replicate noise and delta-method SE, App-App distribution, victim/aggressor two-way fixed effects, 1-node vs 8-node. Writes `output/batch1_numbers{,_raw}.md`. |
| `analysis_placement.py` | Scans `run_data` mpibind logs for socket placement, computes the socket effect and inhibitor runtime coverage. Writes `output/placement_numbers.md` and `output/socket_*.csv` (slow: about 11k log files). |
| `figs_batch1.py` | Figures noise_cv, slowdown_ecdf, appapp_heatmap(_total), slowdown_vs_profile, inhib_response, coverage_pca, scale_1node, socket_effect. Writes `output/figs_batch1_numbers.md`. Run `analysis_placement.py` first. |
| `analysis_batch2_panels.py` | Panel selection (full data plus all 70 restricted cells) and floor rows. Writes `output/batch2_panels.md`. |
| `figs_batch2.py` | Prediction-result figures (forest_fulldata, bias_vs_floor, learning_curves, pervictim_heatmap, cv_vs_appapp) and tables C1–C9, built from `output/artifact_data_20261017.json`. Writes `output/batch2_numbers.md`. |
| `analysis_socket_pinned.py` | Socket-pinned isolated re-runs: validity/placement checks, S1 paired socket effect + order effect, S3 launcher/epoch drift vs June, S4 isolated vs co-run, figure `socket_isolated.pdf`. S2 (inhibitors) is a placeholder until the data exists. Writes `output/socket_numbers.md` (S1–S4), `socket_pinned_{runs,pairs}.csv`. Parses via a symlink tree of valid dirs (parse_isolated raises on empty dirs). |
| `analysis_socket_matched.py` | S5 socket-matched slowdowns (baseline variants J/N0/NM/JM/JMs) and S6 re-scoring of frozen predictions against them. Run after `analysis_socket_pinned.py`; appends S5–S6 to `socket_numbers.md`. |
| `output/batch2_methods.md` | Model, protocol and bootstrap facts with file:line references, plus the variant matrix. |

## Conventions (agreed with the manuscript session)

- **Slowdown** = max(1, T_co / T_iso), computed as a ratio of aggregate means. Name the target every time:
  **MPI-time** or **total-time** slowdown.
- Use **non-self directional outcomes** unless stated otherwise: 45 unordered pairs × 2 = 90 outcomes at 8 nodes.
  Report self-pairs separately: there are 10, each with two copies, and the heatmap diagonal shows the mean of the
  copies.
- **Canonical snapshot** = the current rebuilt one (`prepare_bidirectional_data.py` applied to `processed_data/`).
  The legacy `few_shot/data` snapshot is only a footnote.
- **App-App scoring:** log MAE on log slowdown; Δ% versus the isolated-speed baseline (slowdown = 1); 95% paired
  pair-cluster bootstrap (1,000 draws, seed 923); pooled skill = 1 − mean(method)/mean(baseline).
- **Figures:**
  - matplotlib, serif font with `mathtext.fontset='cm'`, 10 pt labels;
  - 6.0 in full width or 3.0 in half width;
  - **no titles** (captions live in LaTeX);
  - Okabe–Ito colors plus a redundant linestyle or marker;
  - tight bbox, PDF output;
  - always render to PNG and check for label collisions.
- **Display names:** AMG2023, Beatnik, FIESTA, Kripke, Laghos, LAMMPS, MiniFE, miniVite, Quicksilver, TriC.
- **Deliverables:** reply with the numbers in the message (markdown tables are fine) *and* save a `.md` copy under
  `scripts/dissertation_figs/output/`. Flag anything where the data disagrees with `perf_degrad_chapter_plan.md`;
  the data wins.

## Workflow for a new request

1. Read the request and the relevant section of `perf_degrad_chapter_plan.md`.
2. Check `references/findings.md` and the existing `output/*.md` files. Much may already be computed.
3. Verify facts in code and logs, not from memory. Quote file:line. For actual Slurm flags, read the real job scripts
   in `run_data/.../slurm_scripts/`, not the generators; the generators may have changed since the runs.
4. Extend the existing scripts. Keep each script runnable end to end and make it write its numbers file.
5. Render and inspect every figure before replying.
6. Reply to the peer with the numbers, the files, and **flags**. Raise anything that needs a decision from Akhil
   (re-runs, changing pipelines, confounds) with the user directly; don't settle it peer to peer.
7. Commit only when asked.
