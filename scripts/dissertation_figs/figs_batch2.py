"""Batch 2: prediction-result figures (B1-B5) and tables (C1-C9) for the node-sharing chapter.

DATA SOURCES (the 17 Oct 2026 experiment roots ~/hpcResearch/experiment_outputs/*_full_20261017
are NOT on this machine):
  * output/artifact_data_20261017.json - the DATA object embedded in the
    "Filtered-Inhibitor Pipeline Comparison" artifact, computed by that artifact's scoring
    code from each run's consolidated/predictions.csv (non-self directional outcomes,
    log MAE on log slowdown, paired pair-cluster bootstrap vs slowdown 1, pooled skill).
    Using it guarantees the numbers match the artifact.
  * False-benign rates and ablation intervals that the artifact did not carry:
      - bidirectional MPI: delta_response_filteredInhibs_biDirectional/README.md
        ("Results on the repository data"); its full-data numbers equal the artifact's
        bidir run to 4 decimals (same deterministic fit).
      - bidirectional total time: .../delta_response_filteredInhibs_biDirectional_totalTime/
        outputs/full_comparison/{pair_metrics,ablation_summary}.csv (also equal to the artifact).
Canonical snapshot = current rebuilt. The one-way MPI pipeline on that snapshot is the run the
artifact calls the "control" (baseSnap); the artifact's "filteredInhibs" run (base) used the legacy
few_shot snapshot and is reported only as a footnote row.

Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/figs_batch2.py
"""

from __future__ import annotations

import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

from common import (APPS, BIDIR, FULL_W, HALF_W, NAME, OI, OUT_DIR, REPO, pairs_mpi, pairs_total,
                    save, set_style)

set_style()
D = json.load(open(OUT_DIR / "artifact_data_20261017.json"))
R = D["runs"]
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, floatfmt=".4f"):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    idx = [n or "" for n in df.index.names]
    lines = ["| " + " | ".join(idx + [str(c) for c in df.columns]) + " |",
             "|" + "|".join(["---"] * len(idx) + ["---:"] * len(df.columns)) + "|"]
    for k, row in df.iterrows():
        ks = list(k) if isinstance(k, tuple) else [k]
        lines.append("| " + " | ".join([str(x) for x in ks] + [f(v) for v in row.values]) + " |")
    p("\n".join(lines))
    p()


# ---------------------------------------------------------------------------- vocabulary
PIPE = {  # display order: canonical four, then the legacy-snapshot footnote run
    "baseSnap": ("One-way, MPI time", "delta_response_filteredInhibs (current snapshot; artifact 'control')"),
    "bidir": ("Bidirectional, MPI time", "delta_response_filteredInhibs_biDirectional"),
    "totalTime": ("One-way, total time", "delta_response_filteredInhibs_totalTime (data/app_time)"),
    "bidirTT": ("Bidirectional, total time", "delta_response_filteredInhibs_biDirectional_totalTime"),
    "base": ("One-way, MPI time (legacy snapshot)", "delta_response_filteredInhibs (few_shot/data)"),
}
CANON = ["baseSnap", "bidir", "totalTime", "bidirTT"]
DELTA_PRIMARY = {"base": "delta_uniform", "baseSnap": "delta_uniform", "totalTime": "delta_ood_kernel",
                 "bidir": "delta_uniform", "bidirTT": "delta_uniform"}  # as in the artifact
LABEL = {"constant_1": "Isolated-speed baseline", "absolute": "Absolute response",
         "delta_uniform": "Delta, uniform agg.", "delta_median": "Delta, median agg.",
         "delta_ood_kernel": "Delta, OOD-kernel agg.", "generic": "Generic potential",
         "joint_direct": "Joint", "joint_cal": "Joint, calibrated",
         "bprof_direct": "Bidir. profile", "bprof_cal": "Bidir. profile, calibrated",
         "bctx_direct": "Bidir. context", "bctx_cal": "Bidir. context, calibrated",
         "bwall_cal": "Wall-clock aux (ablation), cal.", "bfwd_cal": "Forward-only tokens (ablation), cal."}
CODE = {"absolute": "legacy_absolute / absolute_response", "delta_uniform": "legacy_delta_uniform / delta_uniform",
        "delta_median": "legacy_delta_median / delta_median", "delta_ood_kernel": "legacy_delta_ood_kernel / delta_ood_kernel",
        "generic": "legacy_generic_<agg> / generic_potential", "joint_direct": "joint_profile_direct",
        "joint_cal": "joint_profile_calibrated", "bprof_direct": "bidirectional_profile_direct",
        "bprof_cal": "bidirectional_profile_calibrated", "bctx_direct": "bidirectional_context_direct",
        "bctx_cal": "bidirectional_context_calibrated", "bwall_cal": "bidirectional_profile_wall_calibrated",
        "bfwd_cal": "bidirectional_context_forward_calibrated", "constant_1": "constant_1"}
FAMILY = {"absolute": "Absolute", "delta_uniform": "Delta", "delta_median": "Delta", "delta_ood_kernel": "Delta",
          "generic": "Delta", "joint_direct": "Joint (direct)", "bprof_direct": "Joint (direct)",
          "bctx_direct": "Joint (direct)", "joint_cal": "Joint (calibrated)", "bprof_cal": "Joint (calibrated)",
          "bctx_cal": "Joint (calibrated)", "bwall_cal": "Joint (calibrated)", "bfwd_cal": "Joint (calibrated)"}
FAM_COLOR = {"Absolute": OI["blue"], "Delta": OI["vermillion"], "Joint (direct)": OI["green"],
             "Joint (calibrated)": OI["orange"]}
FAM_MARK = {"Absolute": "o", "Delta": "s", "Joint (direct)": "^", "Joint (calibrated)": "D"}
ORDER = ["absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic", "joint_direct", "joint_cal",
         "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal", "bwall_cal", "bfwd_cal"]
ABLATIONS = {"bwall_cal", "bfwd_cal"}


def roles(run):
    return [r for r in ORDER if r in R[run]["random_split"]]


def rs(run, role):
    return R[run]["random_split"][role]


def c1(run):
    return rs(run, "constant_1")["log_mae"]


def pct(run, role):
    m = rs(run, role)
    b = c1(run)
    return 100 * m["diff_vs_c1"] / b, 100 * m["ci"][0] / b, 100 * m["ci"][1] / b


# ---------------------------------------------------------------------------- false-benign sources
# Bidirectional MPI: README.md table (same fit as the artifact's bidir run).
FB_MPI_README = {"bctx_cal": 0.31, "joint_cal": 0.10, "absolute": 0.72, "bprof_cal": 0.03, "bprof_direct": 0.76,
                 "constant_1": 1.00, "delta_median": 0.00, "delta_uniform": 0.00}
tt_dir = REPO / "notebooks/utils/delta_response_filteredInhibs_biDirectional_totalTime/outputs/full_comparison"
pmtt = pd.read_csv(tt_dir / "pair_metrics.csv")
pmtt = pmtt[(pmtt.pair_scope == "non_self") & (pmtt.evaluation_method == "random_split")].set_index("method")
TT_CODE = {"absolute": "legacy_absolute", "delta_uniform": "legacy_delta_uniform", "delta_median": "legacy_delta_median",
           "delta_ood_kernel": "legacy_delta_ood_kernel", "generic": "legacy_generic_uniform",
           "joint_direct": "joint_profile_direct", "joint_cal": "joint_profile_calibrated",
           "bprof_direct": "bidirectional_profile_direct", "bprof_cal": "bidirectional_profile_calibrated",
           "bctx_direct": "bidirectional_context_direct", "bctx_cal": "bidirectional_context_calibrated",
           "bwall_cal": "bidirectional_profile_wall_calibrated", "bfwd_cal": "bidirectional_context_forward_calibrated",
           "constant_1": "constant_1"}
FB_TT = {k: float(pmtt.loc[v, "false_benign_rate"]) for k, v in TT_CODE.items()}


def false_benign(run, role):
    if run == "bidir":
        return FB_MPI_README.get(role, np.nan)
    if run == "bidirTT":
        return FB_TT.get(role, np.nan)
    # one-way absolute/baseline predictions are identical to the bidirectional legacy ones
    if run == "baseSnap" and role in ("absolute", "constant_1"):
        return FB_MPI_README[role]
    if run == "totalTime" and role in ("absolute", "constant_1"):
        return FB_TT[role]
    return np.nan


def harmful_pairs(pairs):
    q = pairs.query("jobA_id != jobB_id")
    return int((np.maximum(q.slowdown_A, q.slowdown_B) >= 1.05).sum())


p("# Batch 2 numbers (prediction results)")
p("Source: artifact DATA (output/artifact_data_20261017.json) unless noted. Δ% = (method − baseline) log MAE / "
  "baseline log MAE; CI = 95% paired pair-cluster bootstrap (1,000 draws, seed 923, shared draw matrix).")
p(f"Truly harmful non-self pairs (worse endpoint ≥ 1.05): MPI {harmful_pairs(pairs_mpi())}/45, "
  f"total time {harmful_pairs(pairs_total())}/45 — false-benign denominators.")
p()

# ============================================================================ C1 pipeline table
p("## C1. Pipelines")
rows = []
for run in [*CANON, "base"]:
    r = R[run]
    rows.append({"Pipeline": PIPE[run][0], "code dir": PIPE[run][1],
                 "target": {"mpi": "MPI time", "app": "total time"}[r["target"]], "snapshot": r["snapshot"],
                 "family": "bidirectional" if run.startswith("bidir") else "one-way",
                 "variants scored": len(roles(run)), "root": r["root"]})
md(pd.DataFrame(rows).set_index("Pipeline"))

# ============================================================================ C2 best per family
p("## C2. Full-data (all 10 apps in training) metrics, best variant per family per pipeline")
p("Best = lowest App-App log MAE within family (optimistic, as in the artifact); ablation variants excluded. "
  "FB = pair-level false-benign rate at 1.05 (README/local run; – = not available for that run).")
rows = []
for run in [*CANON, "base"]:
    fams = {}
    for role in roles(run):
        if role in ABLATIONS:
            continue
        f = FAMILY[role]
        if f not in fams or rs(run, role)["log_mae"] < rs(run, fams[f])["log_mae"]:
            fams[f] = role
    for role in ["constant_1", *[fams[f] for f in FAM_COLOR if f in fams]]:
        m = rs(run, role)
        d, lo, hi = (np.nan, np.nan, np.nan) if role == "constant_1" else pct(run, role)
        rows.append({"pipeline": PIPE[run][0], "variant": LABEL[role],
                     "family": "baseline" if role == "constant_1" else FAMILY[role],
                     "log MAE": m["log_mae"], "Δ% vs baseline": d, "CI lo %": lo, "CI hi %": hi,
                     "signed bias": m["bias"], "floor log MAE": m["floor_log_mae"],
                     "above-floor log MAE": m["above_log_mae"], "MAPE %": m["mape"],
                     "Spearman": np.nan if m["spearman"] is None else m["spearman"],
                     "FB@1.05": false_benign(run, role)})
c2 = pd.DataFrame(rows).set_index(["pipeline", "variant"])
md(c2.round(4))

# ============================================================================ C3 per-variant full tables
p("## C3. Full per-variant full-data metrics (supplementary)")
for run in [*CANON, "base"]:
    p(f"### {PIPE[run][0]} — n = 90 non-self outcomes, {rs(run, 'constant_1')['n_floor']} at floor")
    rows = []
    for role in ["constant_1", *roles(run)]:
        m = rs(run, role)
        d, lo, hi = (np.nan,) * 3 if role == "constant_1" else pct(run, role)
        rows.append({"variant": LABEL[role], "code": CODE[role], "log MAE": m["log_mae"], "Δ%": d,
                     "95% CI": "–" if role == "constant_1" else f"[{lo:+.1f}, {hi:+.1f}]",
                     "bias": m["bias"], "floor": m["floor_log_mae"], "above": m["above_log_mae"],
                     "MAPE %": m["mape"], "Spearman": np.nan if m["spearman"] is None else m["spearman"],
                     "FB@1.05": false_benign(run, role)})
    md(pd.DataFrame(rows).set_index("variant"))

# ============================================================================ C4 matched cells
p("## C4. Matched cells (20 (split, size) cells completed by all five runs, sizes 4–8)")
p("Cells: " + ", ".join(f"{s.replace('balanced_','')}/{n}" for s, n in D["common_cells"]))
p("Entry = pooled skill (1 − mean method log MAE / mean baseline log MAE over the 20 cells) and wins/20.")
for tier in ("one_known", "zero_shot"):
    tab = {}
    for run in [*CANON, "base"]:
        for m in R[run]["matched"]:
            if m["tier"] == tier and m["role"] != "constant_1":
                tab.setdefault(LABEL[m["role"]], {})[PIPE[run][0]] = f"{100 * m['skill']:+.1f}% ({round(m['win'] * m['n'])})"
    t = pd.DataFrame(tab).T.reindex(columns=[PIPE[r][0] for r in [*CANON, "base"]])
    t = t.reindex([LABEL[r] for r in ORDER if LABEL[r] in t.index]).fillna("–")
    p(f"### {tier}")
    md(t)

# ============================================================================ pooled restricted skill (all completed cells)
p("## Restricted tiers, pooled over all completed cells (sizes 4–8), canonical pipelines")
for tier in ("one_known", "zero_shot"):
    rows = []
    for run in [*CANON, "base"]:
        cur = pd.DataFrame(R[run]["curve"])
        cur = cur[cur.tier == tier]
        base = cur[cur.role == "constant_1"].set_index("size")
        for role in ["absolute", DELTA_PRIMARY[run], "joint_cal", "bprof_direct", "bprof_cal", "bctx_cal"]:
            c = cur[cur.role == role].set_index("size")
            if c.empty:
                continue
            w = c.n_splits
            mm = (c.log_mae * w).sum() / w.sum()
            bb = (base.log_mae * w).sum() / w.sum()
            per = ", ".join(f"{s}:{100 * c.loc[s, 'skill']:+.0f}%/{int(c.loc[s, 'n_splits'])}" for s in c.index)
            rows.append({"pipeline": PIPE[run][0], "variant": LABEL[role], "pooled skill %": 100 * (1 - mm / bb),
                         "cells": int(w.sum()), "per size (skill/splits)": per})
    p(f"### {tier}")
    md(pd.DataFrame(rows).set_index(["pipeline", "variant"]), ".1f")

# ============================================================================ C5 per-victim ratio
HEAT_COLS = [("baseSnap", "absolute"), ("baseSnap", "delta_uniform"), ("baseSnap", "generic"),
             ("bidir", "delta_uniform"), ("bidir", "joint_cal"), ("bidir", "bprof_direct"),
             ("bidir", "bprof_cal"), ("bidir", "bctx_cal")]
p("## C5. Per-victim error ratio (victim log MAE over its 9 non-self outcomes ÷ baseline's), full data, MPI time")
p("One-way absolute = bidirectional legacy absolute (identical predictions), so it appears once.")
ratio = pd.DataFrame({f"{'1W' if run == 'baseSnap' else 'BD'}: {LABEL[role]}":
                      {a: R[run]["per_victim"][role][a] / R[run]["per_victim"]["constant_1"][a] for a in APPS}
                      for run, role in HEAT_COLS})
base_err = pd.Series(R["baseSnap"]["per_victim"]["constant_1"])
ratio.insert(0, "baseline log MAE", base_err)
ratio.index = [NAME[a] for a in ratio.index]
md(ratio, ".3f")
nb = (ratio.iloc[:, 1] < 1).sum()
p(f"Absolute response beats the baseline for {nb}/10 victims; worse for: "
  + ", ".join(ratio.index[ratio.iloc[:, 1] > 1]))
p()

# ============================================================================ C6 CV vs App-App
CVMAP = {"baseSnap": [("low_rank", DELTA_PRIMARY["baseSnap"], "Low-rank delta (tuned)"), ("generic", "generic", "Generic potential"),
                      ("absolute", "absolute", "Absolute response")],
         "totalTime": [("low_rank", DELTA_PRIMARY["totalTime"], "Low-rank delta (tuned)"), ("generic", "generic", "Generic potential"),
                       ("absolute", "absolute", "Absolute response")],
         "base": [("low_rank", "delta_uniform", "Low-rank delta (tuned)"), ("generic", "generic", "Generic potential"),
                  ("absolute", "absolute", "Absolute response")]}
for run in ("bidir", "bidirTT"):
    CVMAP[run] = [("legacy_delta", "delta_uniform", "Legacy delta"), ("legacy_generic", "generic", "Legacy generic"),
                  ("legacy_absolute", "absolute", "Legacy absolute"), ("joint_profile", "joint_cal", "Joint profile (cal.)"),
                  ("bidirectional_profile", "bprof_cal", "Bidir. profile (cal.)"),
                  ("bidirectional_context", "bctx_cal", "Bidir. context (cal.)"),
                  ("bidirectional_profile_wall", "bwall_cal", "Wall-clock aux (cal.)"),
                  ("bidirectional_context_forward", "bfwd_cal", "Forward-only tokens (cal.)")]
p("## C6. App-Inhibitor crossed-CV log MAE vs App-App log MAE (full data)")
rows = []
for run in [*CANON, "base"]:
    for cvk, role, lab in CVMAP[run]:
        cv = R[run]["cv"][cvk]
        rows.append({"pipeline": PIPE[run][0], "model": lab, "CV log MAE": cv["cv_log_mae"],
                     "App-App log MAE": rs(run, role)["log_mae"], "epochs": cv.get("epochs"),
                     "λ / agg": cv.get("lambda") if "lambda" in cv else cv.get("aggregation")})
c6 = pd.DataFrame(rows).set_index(["pipeline", "model"])
md(c6)
for run in [*CANON, "base"]:
    sub = c6.loc[PIPE[run][0]]
    rho = sub["CV log MAE"].rank().corr(sub["App-App log MAE"].rank())
    p(f"{PIPE[run][0]}: Spearman(CV, App-App) over {len(sub)} models = {rho:+.2f}; "
      f"CV-best = {sub['CV log MAE'].idxmin()}, App-App-best = {sub['App-App log MAE'].idxmin()}")
p()

# ============================================================================ C7 ablations
p("## C7. Ablation contrasts (App-App non-self log MAE, method − comparison, paired 95% CI)")
p("MPI time: from the bidirectional README (same fit as the artifact). Total time: local "
  "outputs/full_comparison/ablation_summary.csv (same fit as the artifact).")
abl_mpi = [("Joint direct vs legacy absolute", 0.0062, 0.0012, 0.0118),
           ("Floor-aware calibration (joint cal − joint direct)", -0.0064, -0.0248, 0.0143),
           ("Reverse supervision, direct (bprof − joint)", -0.0038, -0.0073, -0.0002),
           ("Reverse supervision, calibrated", 0.0013, -0.0010, 0.0035),
           ("Fingerprints, direct (bctx − bprof)", 0.0066, 0.0006, 0.0120),
           ("Fingerprints, calibrated", -0.0015, -0.0102, 0.0071),
           ("Wall-clock-only aux, calibrated (bwall − bprof)", rs("bidir", "bwall_cal")["log_mae"] - rs("bidir", "bprof_cal")["log_mae"], np.nan, np.nan),
           ("Forward-only tokens, calibrated (bfwd − bctx)", rs("bidir", "bfwd_cal")["log_mae"] - rs("bidir", "bctx_cal")["log_mae"], np.nan, np.nan),
           ("Legacy delta (uniform) vs legacy absolute", 0.0515, 0.0088, 0.0947)]
ab = pd.read_csv(tt_dir / "ablation_summary.csv")
ab = ab[(ab.scope == "app_app_random_split_non_self") & (ab.metric == "log_mae")].set_index("contrast")
tt_map = {"Joint direct vs legacy absolute": "joint_absolute_delta_learning",
          "Floor-aware calibration (joint cal − joint direct)": "floor_aware_calibration",
          "Reverse supervision, direct (bprof − joint)": "reverse_direction_supervision_direct",
          "Reverse supervision, calibrated": "reverse_direction_supervision_calibrated",
          "Fingerprints, direct (bctx − bprof)": "measured_response_fingerprints_direct",
          "Fingerprints, calibrated": "measured_response_fingerprints_calibrated",
          "Wall-clock-only aux, calibrated (bwall − bprof)": "wall_clock_only_auxiliary_calibrated",
          "Forward-only tokens, calibrated (bfwd − bctx)": "forward_only_context_tokens_calibrated",
          "Legacy delta (uniform) vs legacy absolute": "legacy_delta_vs_legacy_absolute"}
rows = []
for name, est, lo, hi in abl_mpi:
    t = ab.loc[tt_map[name]]
    rows.append({"contrast": name,
                 "MPI Δ": est, "MPI CI": "n/a (README: within ±0.003)" if np.isnan(lo) else f"[{lo:+.4f}, {hi:+.4f}]",
                 "MPI Δ % of baseline": 100 * est / c1("bidir"),
                 "total Δ": t.estimate_difference, "total CI": f"[{t.ci_lower:+.4f}, {t.ci_upper:+.4f}]",
                 "total Δ % of baseline": 100 * t.estimate_difference / c1("bidirTT")})
md(pd.DataFrame(rows).set_index("contrast"), "+.4f")

# ============================================================================ C8 calibration
p("## C8. Calibration effect: signed log bias and false-benign rate, full data")
rows = []
for run in ("bidir", "bidirTT"):
    for role in ["constant_1", "absolute", "joint_direct", "joint_cal", "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal"]:
        m = rs(run, role)
        rows.append({"pipeline": PIPE[run][0], "variant": LABEL[role], "signed bias": m["bias"],
                     "floor log MAE": m["floor_log_mae"], "above-floor log MAE": m["above_log_mae"],
                     "FB@1.05": false_benign(run, role)})
md(pd.DataFrame(rows).set_index(["pipeline", "variant"]), ".3f")

# ============================================================================ C9 coverage
p("## C9. Restricted-tier coverage (tasks completed out of 10 per size; reasons for failures)")
rows = []
for run in [*CANON, "base"]:
    c = pd.DataFrame(R[run]["coverage"])
    row = {"pipeline": PIPE[run][0]}
    for size in range(2, 9):
        g = c[c["size"] == size]
        row[str(size)] = int((g.status == "succeeded").sum())
    sub = c[c["size"] >= 4]
    row["4–8 total"] = f"{int((sub.status == 'succeeded').sum())}/50"
    fails = c[c.status != "succeeded"].groupby("reason").size().to_dict()
    row["failure reasons (all sizes)"] = "; ".join(f"{k} {v}" for k, v in fails.items())
    rows.append(row)
md(pd.DataFrame(rows).set_index("pipeline"))

# ============================================================================ B1 forest
p("## B1 forest_fulldata.pdf: 4 canonical panels (legacy-snapshot one-way MPI omitted; see C3).")
fig = plt.figure(figsize=(FULL_W, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[5.5, 13], hspace=0.38, wspace=0.95, top=0.90, bottom=0.08)
panels = [("baseSnap", gs[0, 0]), ("totalTime", gs[0, 1]), ("bidir", gs[1, 0]), ("bidirTT", gs[1, 1])]
XMAX = 60
for k, (run, g) in enumerate(panels):
    ax = fig.add_subplot(g)
    rl = roles(run)
    for i, role in enumerate(rl):
        d, lo, hi = pct(run, role)
        excl = lo > 0 or hi < 0
        col = (OI["green"] if hi < 0 else OI["vermillion"]) if excl else OI["grey"]
        ax.plot([max(lo, -40), min(hi, XMAX)], [i, i], color=col, lw=1.4, solid_capstyle="butt")
        if d > XMAX:
            ax.plot(XMAX, i, marker=">", color=col, ms=5)
            ax.text(XMAX - 2, i - 0.32, f"{d:+.0f}%", fontsize=7, ha="right", va="bottom", color="#333333")
        else:
            ax.plot(d, i, marker=FAM_MARK[FAMILY[role]], color=col, ms=4.5, mec="white", mew=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rl)))
    ax.set_yticklabels([LABEL[r] for r in rl], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-40, XMAX)
    ax.set_xticks([-40, -20, 0, 20, 40, 60])
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_title(PIPE[run][0], fontsize=9, loc="left")  # panel identifier, not a figure title
    if k >= 2:
        ax.set_xlabel(r"$\Delta$ log MAE vs. baseline (%)")
fig.legend(handles=[Line2D([], [], color=OI["green"], lw=1.4, label="CI excludes 0, better"),
                    Line2D([], [], color=OI["vermillion"], lw=1.4, label="CI excludes 0, worse"),
                    Line2D([], [], color=OI["grey"], lw=1.4, label="CI includes 0")],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.995), fontsize=8)
save(fig, "forest_fulldata.pdf")
plt.close(fig)

# ============================================================================ B2 bias vs floor
fig, ax = plt.subplots(figsize=(HALF_W, 2.9))
for run, mk_fill in (("baseSnap", False), ("bidir", True)):
    for role in roles(run):
        if role in ABLATIONS:
            continue
        m = rs(run, role)
        fam = FAMILY[role]
        ax.scatter(m["bias"], m["floor_log_mae"], s=30, marker=FAM_MARK[fam],
                   facecolor=FAM_COLOR[fam] if mk_fill else "white", edgecolor=FAM_COLOR[fam], lw=1.1, zorder=3)
mb = rs("bidir", "constant_1")
ax.scatter(mb["bias"], mb["floor_log_mae"], s=40, marker="*", color="black", zorder=4)
ax.annotate("baseline", (mb["bias"], mb["floor_log_mae"]), xytext=(7, -3), textcoords="offset points", fontsize=7.5)
ax.axvline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("Signed log bias (pred. − true)")
ax.set_ylabel("Floor-case log MAE")
ax.grid(lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f], ls="", ms=5, label=f) for f in FAM_COLOR]
h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=5, label="one-way"),
      Line2D([], [], marker="o", color="black", ls="", ms=5, label="bidirectional")]
ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", ncol=1, handletextpad=0.2, borderaxespad=0.1)
save(fig, "bias_vs_floor.pdf")
plt.close(fig)
p("## B2 bias_vs_floor.pdf: MPI-time canonical pipelines; hollow = one-way (current snapshot), filled = bidirectional; "
  "ablation variants omitted; star = baseline.")
p()

# ============================================================================ B3 learning curves
CURVE_ROLES = {"baseSnap": ["absolute", "delta_uniform"],
               "bidir": ["absolute", "delta_uniform", "joint_cal", "bprof_direct"]}
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.4), sharex=True, sharey="row")
p("## B3 learning_curves.pdf: ratio = method / baseline pooled log MAE per size; hollow marker = fewer than 10 splits completed.")
for c, run in enumerate(["baseSnap", "bidir"]):
    cur = pd.DataFrame(R[run]["curve"])
    for r_, tier in enumerate(["one_known", "zero_shot"]):
        ax = axes[r_, c]
        for role in CURVE_ROLES[run]:
            s = cur[(cur.tier == tier) & (cur.role == role)].sort_values("size")
            fam = FAMILY[role]
            col = FAM_COLOR[fam]
            ls = "--" if role == "bprof_direct" else "-"
            if role == "bprof_direct":
                col = OI["sky"]
            ratio_ = 1 - s.skill
            ax.plot(s["size"], ratio_, color=col, ls=ls, lw=1.3)
            for _, row in s.iterrows():
                full = row.n_splits >= 10
                ax.plot(row["size"], 1 - row.skill, marker=FAM_MARK[fam], ms=4.5, color=col,
                        mfc=col if full else "white", mec=col, mew=1.0)
        ax.axhline(1, color="black", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.grid(lw=0.4, alpha=0.4)
        ns = cur[(cur.tier == tier) & (cur.role == "constant_1")].set_index("size").n_splits
        if r_ == 0:
            ax.set_title(PIPE[run][0], fontsize=9, loc="left")
        if c == 0:
            ax.set_ylabel(("One-known" if tier == "one_known" else "Zero-shot") + "\nerror ratio")
        if r_ == 1:
            ax.set_xlabel("Training apps (completed splits)")
            ax.set_xticks([4, 5, 6, 7, 8])
            ax.set_xticklabels([f"{s}\n({int(ns[s])})" for s in [4, 5, 6, 7, 8]])
        for size in sorted(ns.index):
            pass
        p(f"  {PIPE[run][0]} {tier}: splits per size {dict(ns.astype(int))}")
for ax in axes.ravel():
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in [0.8, 0.9, 1, 1.25, 1.5, 2, 3] if lo <= t <= hi])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
handles = [Line2D([], [], color=FAM_COLOR["Absolute"], marker="o", label="Absolute response"),
           Line2D([], [], color=FAM_COLOR["Delta"], marker="s", label="Delta, uniform agg."),
           Line2D([], [], color=FAM_COLOR["Joint (calibrated)"], marker="D", label="Joint, calibrated"),
           Line2D([], [], color=OI["sky"], marker="^", ls="--", label="Bidir. profile (direct)"),
           Line2D([], [], color="grey", marker="o", mfc="white", ls="", label="< 10 splits completed")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.03),
           columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "learning_curves.pdf")
plt.close(fig)
p("  Parenthesised numbers under the x ticks = completed splits per size.")
p()

# ============================================================================ B4 per-victim heatmap
mat = ratio.iloc[:, 1:]
order_rows = mat.mean(axis=1).sort_values().index
mat = mat.loc[order_rows]
cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#92c5de", "#f2f2f2", "#f4a582", "#b2182b"])
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(mat.to_numpy(), cmap=cmap, norm=LogNorm(vmin=0.5, vmax=2.0), aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.iat[i, j]
        dark = v < 0.62 or v > 1.6
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if dark else "#222222")
short = ["Absolute", "Delta unif.", "Generic", "Delta unif.", "Joint cal.", "Bidir. prof.", "Bidir. prof. cal.", "Bidir. ctx cal."]
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels(mat.index)
ax.axvline(2.5, color="black", lw=1.0)
ax.text(1.0, -0.9, "one-way", ha="center", fontsize=8)
ax.text(5.0, -0.9, "bidirectional", ha="center", fontsize=8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0.5, 0.75, 1, 1.5, 2])
cb.ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2"])
cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.set_label("Victim log MAE ÷ baseline")
cb.outline.set_visible(False)
save(fig, "pervictim_heatmap.pdf")
plt.close(fig)
p("## B4 pervictim_heatmap.pdf: rows sorted by mean ratio (best first); colour scale log, saturates at 0.5× and 2×.")
p("Columns: " + "; ".join(f"{'one-way' if r == 'baseSnap' else 'bidir'} {LABEL[x]}" for r, x in HEAT_COLS))
p()

# ============================================================================ B5 CV vs App-App
fig, ax = plt.subplots(figsize=(HALF_W, 2.8))
for run, filled in (("baseSnap", False), ("bidir", True)):
    for cvk, role, lab in CVMAP[run]:
        if role in ABLATIONS:
            continue
        fam = FAMILY[role]
        x = R[run]["cv"][cvk]["cv_log_mae"]
        y = rs(run, role)["log_mae"]
        ax.scatter(x, y, marker=FAM_MARK[fam], s=30, facecolor=FAM_COLOR[fam] if filled else "white",
                   edgecolor=FAM_COLOR[fam], lw=1.1, zorder=3)
ax.axhline(c1("bidir"), color="black", lw=0.8, ls=":")
ax.text(0.99, c1("bidir") + 0.0015, "baseline", fontsize=7.5, ha="right", transform=ax.get_yaxis_transform())
ax.set_xlabel("App-Inhibitor CV log MAE")
ax.set_ylabel("App-App log MAE")
ax.grid(lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f], ls="", ms=5, label=f) for f in FAM_COLOR if f != "Joint (direct)"]
h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=5, label="one-way"),
      Line2D([], [], marker="o", color="black", ls="", ms=5, label="bidirectional")]
ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", handletextpad=0.2)
save(fig, "cv_vs_appapp.pdf")
plt.close(fig)
p("## B5 cv_vs_appapp.pdf: MPI-time canonical pipelines; ablations omitted; hollow = one-way, filled = bidirectional.")

(OUT_DIR / "batch2_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "batch2_numbers.md")
