# Plan v2 handoff: where the work was left off (2026-11-23)

This file is for a new agent, possibly on another machine with no peer sessions. Read it together with `findings.md`
§10 and, in the Dissertation repo, `network_interference_modeling_plan_v2.md`. The plan is the source of truth for
decisions D1–D10 and the verified chapter numbers; `perf_degrad_chapter_plan.md` was deleted.

## Status

**Complete. Nothing is pending from the chapter director.**

- **v2 characterization:** B1–B4 and B8 are done.
- **Prediction runs:** the main run `v2_full_20261123` (7,280 tasks, 0 failures, 3.97 h on 78 workers) and the xgb
  follow-up `v2_xgbdeep_20261123` (1,040 tasks) are both scored.
- **In the chapter:** everything above is integrated (Dissertation branch `networkInterferenceDraft`, commit `fbf3505`).
- **Figures delivered** to `Dissertation/figures/node_sharing/`:
  - c_ecdf_unclipped, c_heatmaps_unclipped, c_socket_isolated, c_socket_corun, c_coverage, c_panel_vs_appapp;
  - p_panel_grid, p_panel_objective;
  - r_skill, r_paired_diffs, r_pervictim;
  - slowdown_vs_profile, inhib_response and scale_1node, regenerated unclipped.
  - noise_cv is still made by the old `figs_batch1.py`.
- **Wording rule (Akhil):** no "June", "JMs", "launcher" or "drift" text may appear in chapter figures. The
  c_socket_corun legend says "socket-corrected baseline" / "uncorrected baseline". Code and notes may still say JMs/J.

## Pipeline end to end

All commands run from the repo root. `PY` is the machine's analysis Python; see "Machines and paths" below. Each script writes its
numbers file to `scripts/dissertation_figs/output/`.

| Step | Script (`scripts/dissertation_figs/v2/`) | Reads | Writes | Feeds |
|---|---|---|---|---|
| 0 (upstream, done) | `../analysis_socket_pinned.py`, `../analysis_socket_matched.py`, `../analysis_placement.py` | `run_data/` raw logs (62 GB, **not in git**, only on Akhil's analysis machine) | `output/socket_pinned_pairs.csv`, `socket_app_inhibitor.csv`, `socket_app_app.csv`, `socket_matched_*.csv` (**committed**) | JMs factors and socket labels |
| 1 | `v2_build.py` | `processed_data/data_*.csv` (committed), canonical snapshots `notebooks/utils/delta_response_filteredInhibs_biDirectional{,_totalTime}/data/` (committed), the step-0 CSVs | `output/v2_appapp_rows.csv`, `v2_appapp_directed.csv`, `v2_appinh_grid.csv`, `v2_panel_app.csv`, `v2_A_facts.md`, plus the bundle `agera_bundle/` (6 CSV + manifest.json with sha256) | everything below |
| 2a | `v2_char.py` | step-1 CSVs, socket CSVs | `output/v2_char_B1.csv`, `_B2_corun_socket.csv`, `_B3_coverage.csv`, `_B3_panel_vs_appapp.csv`, `_B4_curve.csv`, `_appinh_resolved_grid.csv`, `v2_char_numbers.md` | Table ns_slowdown_summary (verify the cell values against plan v2); characterization prose (B1–B4) |
| 2b | `v2_char2.py` (B8) | `v2_appapp_directed.csv` (raw_* = JMs, rawJ_* = June), `v2_appinh_grid.csv`, socket CSVs, canonical snapshot, `processed_data` (1-node) | `output/v2_char2_heat_{mpi,tt}{,_J}.csv`, `v2_char2_se.csv`, `v2_char2_numbers.md`; figures slowdown_vs_profile, inhib_response, scale_1node | Table ns_decomposition, asymmetry, SE |
| 2c | `v2_char_figs.py` | step-1 and 2a CSVs | the c_* and p_* figures, `v2_char_figs_numbers.md` | Characterization and panel figures (the panel members in Table ns_panel come from the panel selection; see p_panel_grid) |
| 3 | `agera_pipeline.py` (on agera) | `agera_bundle/` only (sha256 checked at start) | `agera_out/<run>/{preds/*.csv.gz, hp/*.csv, done/*.json, run_config.json, status.json, errors/}` | — |
| 4 | `v2_score.py` | `agera_out/<run>`, the xgb rerun, the bundle | `output/v2_pred_<tag>.md` and `_metrics`, `_summary`, `_summary_noself`, `_contrasts`, `_contrast_summary`, `_breakdown`, `_hp_selected`, `_hp_edges`, `_hp_distribution`, `_pred_spread` (.csv) | Tables ns_skill, ns_safety (FB@1.05), ns_full_results_mpi/tt; the HP-edge prose (Table ns_hp_grids lists `GRIDS` from agera_pipeline.py) |
| 5 | `v2_pred_figs.py` | only `output/v2_pred_<tag>_*.csv` (no agera_out needed) | r_skill, r_paired_diffs, r_pervictim, `v2_pred_figs_numbers.md` | Results figures |

- `output/v2_pred_v2_full_20261123_digest.md` is a hand-written skill digest made during review, not a script output.
- `v2_pred_pilot*` and `v2_pred_v2_full_mainonly*` are the pilot and pre-xgb-rerun scorings, kept for traceability.

### Exact commands

```bash
PY=/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python      # analysis machine
$PY scripts/dissertation_figs/v2/v2_build.py          # rewrites output/v2_* tables and agera_bundle/
$PY scripts/dissertation_figs/v2/v2_char.py
$PY scripts/dissertation_figs/v2/v2_char2.py
$PY scripts/dissertation_figs/v2/v2_char_figs.py
# agera: copy bundle + agera_pipeline.py + env_check.py, run (see below), rsync agera_out back
$PY scripts/dissertation_figs/v2/v2_score.py --run scripts/dissertation_figs/agera_out/v2_full_20261123 \
    --xgb-rerun scripts/dissertation_figs/agera_out/v2_xgbdeep_20261123 \
    --bundle scripts/dissertation_figs/agera_bundle --tag v2_full_20261123
$PY scripts/dissertation_figs/v2/v2_pred_figs.py --tag v2_full_20261123
```

Re-running `v2_build.py` changes the bundle sha256 only if the inputs changed. If it does change, the agera results no
longer match the bundle, and the scorer would score against new rows.

## Where agera_out lives (raw prediction outputs, 166 MB, NOT in git)

- **On agera:** `/nfs/research/compass/aalasandagutti/coScheduling/scripts/dissertation_figs/agera_out/`
  (v2_full_20261123, v2_xgbdeep_20261123, pilot_a, pilot_b, pilot2_a, pilot2_b, each with a `.log`).
- **On Akhil's analysis machine (A):** `/home/akhil/hpcResearch/repos/coScheduling/scripts/dissertation_figs/agera_out/`,
  which also has `local_pilot_a_partial`.
- **On machine B** (`~/Documents/hpcResearch/repos/coScheduling/...`): all agera runs pulled by rsync on 2026-11-24
  (v2_full 7,280 done, xgbdeep 1,040 done, 0 errors).
- **Committed:** only each agera run's `.log`, `run_config.json` and `status.json` (pilots have no status.json), plus `agera_out/README.md`. The
  `.gitignore` there excludes preds/hp/done/errors.
- **Without agera_out:** the scored CSVs in `output/` are enough to regenerate every r_* figure and table. You need
  agera_out only to re-score (`v2_score.py`), for example with a different metric or bootstrap. If it is gone,
  re-run the pipeline: it is deterministic given the bundle, seeds and package versions.

## Key conventions (verified; see plan v2 for the full list)

- **Target:** unclipped log slowdown y = ln(T_corun / T_base). Nothing is clipped anywhere.
- **JMs baseline** ("socket-corrected" in the chapter): the June 8-node isolated mean × exp(d) for socket-1 replicates.
  - d = the mean paired pinned ln(sock1/sock0), applied only where its 95% t CI excludes 0. That is AMG2023 (MPI
    −0.197, total −0.0071) and Quicksilver (MPI −0.036, total −0.0165).
  - Inhibitor times are not corrected.
  - A cell that mixes sockets uses the replicate-weighted baseline.
- **Rows:** 176 App-App prediction rows. 166 are socket-resolved non-self cells (victim, corunner, s_V); 10 are pooled
  self-pairs (s_V = 0.5, 50/50 baseline). Characterization uses 90 non-self rep-averaged directed outcomes.
- **Features:**
  - 1W (27): A and B panel y (9 + 9), A and B base profiles (4 + 4), and the socket bit.
  - BD (45): 1W plus the slowdown each app causes on the 9 inhibitors (9 + 9). `c_mpi` and `c_tt` are defined in
    `agera_bundle/README.md`.
  - Panel features use the target's timer.
- **Socket modes:**
  - aware uses the victim bit s_V; the response models give the co-runner/inhibitor 1 − s_V;
  - blind sets the bit to 0.
- **Methods:**
  - regressors: ridge, lasso, svr, rf, xgb, mlp;
  - response models: absolute and delta (inhibitor-only, no BD form).
  - The xgb follow-up (max_depth 1–8) replaces the main run's xgb, which is kept as the sensitivity row `xgb_d124`.
- **Tiers:**
  - full: 10 splits, each holding out 11 of the 55 unordered co-runs (both directions); split k uses
    rng(1000 + k), and the init seed is k.
  - one_known and zero_shot: all 120 C(10,3) held-out triples, scored by the same fit, with 10 init seeds.
    Deterministic learners give SD 0 there, by design (the director's decision); the prose says restricted-tier SD
    reflects initialization only.
- **Hyperparameters:** nested CV on each split's training data, init seed 0, then frozen across seeds. The grids are
  `GRIDS` in `agera_pipeline.py`; `ext` marks the post-pilot extensions.
- **Scoring:**
  - pooled log MAE, with skill = 1 − MAE / MAE_baseline (baseline = predict 0, i.e. the isolated speed);
  - paired pair-cluster bootstrap, 1,000 draws, seed 923;
  - "significant" = the CI of method − baseline excludes 0 in ≥ 9/10 seeds;
  - FB@1.05 denominator = outcomes with y ≥ ln 1.05.
- **Chapter tables** round with ROUND_HALF_UP. That is the director's convention, applied when writing the .tex; the
  scripts write full precision.
- **Asymmetry Spearman depends on orientation.** Cite the as-run A/B value; the orientation-free one goes in a
  footnote. Don't use `analysis_batch1.py`'s ρ = 0.39 (a bug; see findings §10).

## agera (where step 3 ran)

- **Host:** ssh alias `agera` (agera.cs.unm.edu). It is passwordless from Akhil's analysis machine; **a new machine
  needs its own SSH key installed first** (ask Akhil).
- **Machine:** a single server with no Slurm. 80 cores (use ≤ 80 workers; the runs used 78), 512 GB RAM, 4×V100.
  **Force CPU:** `CUDA_VISIBLE_DEVICES=""`; the pipeline also sets 1 BLAS thread per worker. Check `uptime` first.
- **Checkout:** `/nfs/research/compass/aalasandagutti/coScheduling` (branch `dane`). **No git operations on agera**
  (no checkout, pull or commit). Copy files over with rsync/scp, and only into `scripts/dissertation_figs/v2/`,
  `scripts/dissertation_figs/agera_bundle/` and `scripts/dissertation_figs/agera_out/`.
- **venv:** `/nfs/student/a/aalasandagutti/hpcResearch/python_virtual_envs/pytorch-gpu/bin/python` (py 3.10.12,
  numpy 2.1.3, pandas 2.2.3, sklearn 1.5.2, torch 2.3.0+cu121, xgboost 2.1.3).
  - pip installs inside this venv are allowed without asking.
  - Prefer changing code over upgrading or downgrading packages.
  - `v2/env_check.py` verifies versions and determinism.
- **Copy over:**

  ```bash
  R=/nfs/research/compass/aalasandagutti/coScheduling
  rsync -av scripts/dissertation_figs/agera_bundle/ agera:$R/scripts/dissertation_figs/agera_bundle/
  rsync -av scripts/dissertation_figs/v2/{agera_pipeline.py,env_check.py} agera:$R/scripts/dissertation_figs/v2/
  ```

- **Launch** (this form survives disconnects and doesn't hang ssh):

  ```bash
  PYA=/nfs/student/a/aalasandagutti/hpcResearch/python_virtual_envs/pytorch-gpu/bin/python
  ssh -f agera "cd $R && export CUDA_VISIBLE_DEVICES='' && setsid nohup $PYA scripts/dissertation_figs/v2/agera_pipeline.py \
      --bundle scripts/dissertation_figs/agera_bundle --out scripts/dissertation_figs/agera_out/<run_id> --workers 78 \
      > scripts/dissertation_figs/agera_out/<run_id>.log 2>&1 < /dev/null &"
  ```

  - The xgb follow-up added `--methods xgb --xgb-depths 1,2,4,6,8`.
  - Pilot: `--splits-full 0,1 --splits-restricted 0,57 --seeds 0,1 --workers 8`.
  - The run is resumable: re-running the same command skips tasks that have a `done/` file.
- **Monitor:** `ssh agera "cat $R/scripts/dissertation_figs/agera_out/<run_id>/status.json; tail -3 $R/.../<run_id>.log"`.
  `status.json` says `"state": "finished"` and gives `failures` when the run ends. Tracebacks go to `errors/`.
- **Pull back:**
  `rsync -av agera:$R/scripts/dissertation_figs/agera_out/<run_id>{,.log} scripts/dissertation_figs/agera_out/`

## Machines and paths

Two machines are set up; the machine table is in SKILL.md ("Environment").

- `scripts/dissertation_figs/common.py`: `FIG_DIR` = `REPO.parent / "Dissertation/figures/node_sharing"` (env
  `NS_FIG_DIR` overrides). It used to be hardcoded to machine A; since 2026-11-24 no change is needed if the
  Dissertation repo sits next to coScheduling. Every figure script saves there.
- **Python:** machine A uses the venv `/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python` (named in
  docstrings); machine B uses conda env `hpcResearch`. Any Python ≥ 3.10 with numpy, pandas, scipy, matplotlib and
  sklearn works for steps 1–2 and 4–5. `tabulate` is not needed.
- **Cross-machine check (2026-11-24):** re-scoring `v2_full_20261123` + xgb rerun on machine B (pandas 2.2.1)
  reproduced every `output/v2_pred_v2_full_20261123_*.csv` to ≤ 5e-15, **except** `_hp_distribution.csv` for rf
  `max_depth`. The rf grid includes `None` (unlimited depth), chosen in 23.8% of fits. The committed file (pandas 3,
  where `astype(str)` keeps NaN missing) drops it, so its depth 1/2/3/6 shares are renormalized without it; pandas 2.2
  gives 2.0/21.8/22.2/30.1/23.8%. `_hp_edges` also ignores `None`, so it treats 6 as rf's top depth. The chapter cites
  neither; Akhil decided (2026-11-24) to leave it as is.
- `REPO`, `OUT_DIR` and the bundle path are relative to the script location, so they need no change.
- The agera paths above are specific to Akhil's NFS account.
- `run_data/` (raw logs) exists on machines A and B, except the socket-pinned runs, which are only on A. Steps 1–5
  don't need it, because the step-0 CSVs are committed.

## Standing rules

- Akhil submits all Dane jobs. No new Dane jobs are planned.
- Never commit or push without Akhil's approval.
- When the data disagree with the plan, the data win: flag the difference and don't silently follow the plan.
- Requests that need Akhil's decision (re-runs, pipeline changes, confounds) go to Akhil, not to peers.
- Render every figure to PNG and inspect it before reporting.
