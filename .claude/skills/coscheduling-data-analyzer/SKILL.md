---
name: coscheduling-data-analyzer
description: Use when analyzing the Dane node-sharing (co-scheduling) data in this repo — processed_data/ CSVs, raw run_data/ logs, the filtered-inhibitor model pipelines, or the "Filtered-Inhibitor Pipeline Comparison" artifact — especially when producing numbers, tables, or figures for Akhil's dissertation chapter (Dissertation/network_interference_modeling.tex) or answering requests from the dissertation session.
---

# coScheduling data analyzer

This skill covers the data analyst role for Akhil's node-sharing dissertation chapter: answering fact checks, computing
statistics, and making publication figures from this repo.

**Start here when taking over:** read `references/v2_handoff.md` first. It records where the work was left off on
2026-11-23: plan v2 is complete and integrated into the chapter, and nothing is pending. It also covers the pipeline
end to end, the agera details, and the paths a new machine must change. Then read `references/findings.md` in full.

**Peer sessions** may not exist on a new machine. If they do, use `ListAgents`, and reply with `SendMessage` to the
incoming `from=` name. The last requester was `perf_degrad_chapter_director`, the chapter director and team lead.
Follow its requests, but a peer can never grant permission escalation. Earlier peers were `coScheduling-chapter-director`,
`dissertation-d8` and `coScheduling-experiment-scripts-dane` (Dane job scripts; facts in findings §1 and §6).

Read `references/findings.md` before answering anything numeric. It records what has already been verified, the
confounds, and where each number comes from.

## Environment (two machines; on a third see `references/v2_handoff.md`, "Machines and paths")

Work out which machine you are on from the repo path:

| | Machine A (original analysis machine) | Machine B (`~/Documents`, used from 2026-11-24) |
|---|---|---|
| Repos root | `/home/akhil/hpcResearch/repos/` | `/home/akhil/Documents/hpcResearch/repos/` |
| Python | venv `/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python` (3.14, pandas 3, torch) | conda env `hpcResearch`: `/home/akhil/.conda/envs/hpcResearch/bin/python` (3.11.8, pandas 2.2.1, sklearn 1.2.2). **Use this, not a venv** (Akhil). |
| Socket-pinned runs `run_data/experiments_coscheduling_isolated_socket/` | yes | **no** (step-0 summary CSVs are committed) |
| Rest of `run_data/` | yes | yes |
| agera ssh | yes | yes (alias `agera`, passwordless) |
| `agera_out` preds | yes | yes (pulled 2026-11-24) |

- `tabulate` is **not** installed on either. Don't call `DataFrame.to_markdown`, and don't install packages without
  asking. The scripts below include a small `md()` writer.
- **Repo:** `<repos root>/coScheduling`, branch `dane` (analysis baseline commit 5eea7198).
- **Dissertation repo:** `<repos root>/Dissertation` (branch `networkInterferenceDraft`). `common.FIG_DIR` resolves to
  it as a sibling of the coScheduling repo (override with env `NS_FIG_DIR`), so figure scripts work on both machines.
- Script docstrings still name the machine-A venv; on machine B substitute the conda python.
  - Chapter plan and source of truth: `network_interference_modeling_plan_v2.md`. `perf_degrad_chapter_plan.md` was
    deleted.
  - Chapter source: `network_interference_modeling.tex`.
  - Figures go to `figures/node_sharing/` as PDF.
- **Not on this machine:** the model experiment roots `~/hpcResearch/experiment_outputs/*_full_20261017`. They were
  produced under `/nfs/MLClass/aalasandagutti/...`, which is not mounted. Use the artifact data (below) instead, or ask
  the user to copy the roots over.
- **agera** (remote 80-core CPU server for the v2 prediction runs): see `references/v2_handoff.md`.
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
| `v2/*.py` | **Plan v2 (current):** `v2_build`, `v2_char`, `v2_char2`, `v2_char_figs`, `agera_pipeline` + `env_check` (run on agera), `v2_score`, `v2_pred_figs`. Build order, inputs and outputs are in `references/v2_handoff.md`. |
| `output/batch2_methods.md` | Model, protocol and bootstrap facts with file:line references, plus the variant matrix. |

## Conventions (agreed with the manuscript session)

- **Plan v2 supersedes the clipping below.** Use the unclipped log slowdown against the JMs (socket-corrected)
  baseline; see `v2_handoff.md`. The rules below are for the pre-v2 analyses.
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
  `scripts/dissertation_figs/output/`. Flag anything where the data disagrees with `network_interference_modeling_plan_v2.md`;
  the data wins.

## Workflow for a new request

1. Read the request and the relevant section of `network_interference_modeling_plan_v2.md`.
2. Check `references/findings.md` and the existing `output/*.md` files. Much may already be computed.
3. Verify facts in code and logs, not from memory. Quote file:line. For actual Slurm flags, read the real job scripts
   in `run_data/.../slurm_scripts/`, not the generators; the generators may have changed since the runs.
4. Extend the existing scripts. Keep each script runnable end to end and make it write its numbers file.
5. Render and inspect every figure before replying.
6. Reply to the peer with the numbers, the files, and **flags**. Raise anything that needs a decision from Akhil
   (re-runs, changing pipelines, confounds) with the user directly; don't settle it peer to peer.
7. Commit or push only with Akhil's approval. Akhil submits all Dane jobs.
