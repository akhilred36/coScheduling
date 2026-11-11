"""Dissertation figures from the 100-seed batch (output/seed_*.csv). Writes figures + output/figs_seed_numbers.md.

Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/figs_seed.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import common
from common import APPS, FIG_DIR, FULL_W, HALF_W, NAME, OI, OUT_DIR, save as _save, set_style

set_style()

# ---- vocabulary copied from figs_batch2.py
LABEL = {"constant_1": "Isolated-speed baseline", "absolute": "Absolute response",
         "delta_uniform": "Delta, uniform agg.", "delta_median": "Delta, median agg.",
         "delta_ood_kernel": "Delta, OOD-kernel agg.", "generic": "Generic potential",
         "joint_direct": "Joint", "joint_cal": "Joint, calibrated",
         "bprof_direct": "Bidir. profile", "bprof_cal": "Bidir. profile, calibrated",
         "bctx_direct": "Bidir. context", "bctx_cal": "Bidir. context, calibrated",
         "bwall_cal": "Wall-clock aux (ablation), cal.", "bfwd_cal": "Forward-only tokens (ablation), cal."}
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
PIPES = ["1W-MPI", "1W-TT", "BD-MPI", "BD-TT"]
PIPE_NAME = {"1W-MPI": "One-way, MPI time", "1W-TT": "One-way, total time",
             "BD-MPI": "Bidirectional, MPI time", "BD-TT": "Bidirectional, total time"}

# ---- output bookkeeping
NUM: list[str] = []


def table(title: str, df: pd.DataFrame, note: str = ""):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)
    NUM.append(f"## {title}\n")
    if note:
        NUM.append(note + "\n")
    NUM.append("| " + " | ".join(map(str, df.columns)) + " |")
    NUM.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for row in df.itertuples(index=False):
        NUM.append("| " + " | ".join(f(v) for v in row) + " |")
    NUM.append("")


ARCHIVE = {"forest_fulldata.pdf", "bias_vs_floor.pdf", "pervictim_heatmap.pdf", "cv_vs_appapp.pdf", "learning_curves.pdf"}


def save(fig, name):
    dst = FIG_DIR / name
    arch = FIG_DIR / (dst.stem + "_5seed.pdf")
    if name in ARCHIVE and dst.exists() and not arch.exists():  # only figures that predate the seed batch
        shutil.copy2(dst, arch)
        print("archived", arch)
    _save(fig, name)
    plt.close(fig)


def ci(s):
    a, b = re.findall(r"[-+]?\d+\.?\d*", s)
    return float(a), float(b)


M = pd.read_csv(OUT_DIR / "seed_methods.csv")
M["base"] = M["log MAE"] / (1 + M["Δ%"] / 100)
CL = pd.read_csv(OUT_DIR / "seed_claims.csv")

# ======================================================================= 1 forest
_z = (M["log MAE"] - M["base"]).abs() / M["seed SE"]
# rule from the batch note: borderline (1.9-2.1 SE) is decided by the two-level CI, not by the jackknife rule
M["unstable"] = (M["seed SE"] > 0.002) & (_z < 2) & ~((_z >= 1.9) & (_z <= 2.1))
cl = CL[CL.claim.str.contains(r"(?:<|>) baseline")].copy()
cl["role"] = cl.claim.str.extract(r"^(.*?) [<>] baseline")[0].str.replace(" ", "_")
mism = []
for r in cl.itertuples():
    row = M[(M.pipeline == r.pipeline) & (M.role == r.role)]
    if len(row) and bool(row.unstable.iloc[0]) != (r.verdict == "sign-unstable"):
        mism.append((r.pipeline, r.role, r.verdict, bool(row.unstable.iloc[0])))
print("sign-unstable cross-check vs seed_claims: %d claims checked, mismatches: %s" % (len(cl), mism or "none"))
print("sign-unstable rows:", [(r.pipeline, r.role) for r in M.itertuples() if r.unstable])

XLO, XMAX = -50, 90
fig = plt.figure(figsize=(FULL_W, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[5.5, 13], hspace=0.38, wspace=0.85, top=0.90, bottom=0.08, left=0.19, right=0.92)
panels = [("1W-MPI", gs[0, 0]), ("1W-TT", gs[0, 1]), ("BD-MPI", gs[1, 0]), ("BD-TT", gs[1, 1])]
rows_f = []
for k, (pipe, g) in enumerate(panels):
    ax = fig.add_subplot(g)
    sub = M[M.pipeline == pipe].set_index("role")
    rl = [r for r in ORDER if r in sub.index]
    for i, role in enumerate(rl):
        s = sub.loc[role]
        lo, hi = ci(s["Δ% CI pairs×seeds"])
        d = s["Δ%"]
        excl = lo > 0 or hi < 0
        col = (OI["green"] if hi < 0 else OI["vermillion"]) if excl else OI["grey"]
        mk = FAM_MARK[FAMILY[role]]
        ax.plot([max(lo, XLO), min(hi, XMAX)], [i, i], color=col, lw=1.4, solid_capstyle="butt", snap=False)
        if hi > XMAX:
            ax.plot(XMAX, i, marker=">", color=col, ms=3.5)
        hollow = bool(s["unstable"])
        ax.plot(d, i, marker=mk, color=col, ms=4.8, mfc="white" if hollow else col, mec=col,
                mew=1.0 if hollow else 0.5 if False else 1.0, zorder=3)
        rows_f.append(dict(pipeline=pipe, role=role, **{"Δ%": d, "CI lo": lo, "CI hi": hi,
                                                         "excl 0": excl, "sign-unstable": hollow}))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rl)))
    ax.set_yticklabels([LABEL[r] for r in rl], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(XLO, XMAX + 5)
    ax.set_xticks([-40, 0, 40, 80])
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_title(PIPE_NAME[pipe], fontsize=9, loc="left")
    if k >= 2:
        ax.set_xlabel(r"$\Delta$ log MAE vs. baseline (%)", loc="right")
from matplotlib.legend_handler import HandlerTuple
_hol = tuple(Line2D([], [], color="black", marker=mk, mfc="white", ls="", ms=4.8) for mk in ("^", "D"))
fig.legend(handles=[Line2D([], [], color=OI["grey"], lw=1.4), _hol],
           labels=["two-level 95% CI", "hollow marker = sign-unstable across seeds"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0), fontsize=8)
fig.subplots_adjust(top=0.87)
save(fig, "forest_fulldata.pdf")
table("forest_fulldata.pdf", pd.DataFrame(rows_f),
      f"Two-level CI (pairs×seeds), N=100. Axis clipped at {XMAX}% (arrowhead = CI continues). Hollow = sign-unstable.")

# ======================================================================= 2 bias vs floor
ROOT = Path.home() / "hpcResearch/experiment_outputs/seed_batch/full"
CODE_1W = {"absolute": "absolute_response", "delta_uniform": "delta_uniform", "delta_median": "delta_median",
           "delta_ood_kernel": "delta_ood_kernel", "generic": "generic_potential"}
CODE_BD = {"absolute": "legacy_absolute", "delta_uniform": "legacy_delta_uniform", "delta_median": "legacy_delta_median",
           "delta_ood_kernel": "legacy_delta_ood_kernel", "generic": "legacy_generic_uniform",
           "joint_direct": "joint_profile_direct", "joint_cal": "joint_profile_calibrated",
           "bprof_direct": "bidirectional_profile_direct", "bprof_cal": "bidirectional_profile_calibrated",
           "bctx_direct": "bidirectional_context_direct", "bctx_cal": "bidirectional_context_calibrated",
           "bwall_cal": "bidirectional_profile_wall_calibrated", "bfwd_cal": "bidirectional_context_forward_calibrated"}
_SPcols = ["evaluation_method", "anchor_budget", "seed", "pair_row_id", "direction", "victim_id", "aggressor_id",
           "true_slowdown", "method", "latent_predicted_log_slowdown"]


def floor_stats(pipe_dir, codes, pipe):
    df = pd.read_csv(ROOT / pipe_dir / "seed_predictions.csv", usecols=_SPcols)
    df = df[(df.evaluation_method == "random_split") & (df.anchor_budget == "all") & (df.victim_id != df.aggressor_id)]
    out, maxerr = {}, 0.0
    for role, code in codes.items():
        d = df[df.method == code]
        P = d.pivot_table(index=["pair_row_id", "direction"], columns="seed", values="latent_predicted_log_slowdown")
        y = np.log(np.maximum(1.0, d.drop_duplicates(["pair_row_id", "direction"]).set_index(["pair_row_id", "direction"])
                              .true_slowdown.reindex(P.index).to_numpy()))
        pred = np.maximum(0.0, P.to_numpy().mean(axis=1))
        bias = float((pred - y).mean())
        fl = y == 0
        ref = M[(M.pipeline == pipe) & (M.role == role)].bias.iloc[0]
        maxerr = max(maxerr, abs(bias - ref))
        out[role] = (bias, float(np.abs(pred[fl] - y[fl]).mean()), int(fl.sum()), len(y))
    # baseline: pred = 0
    out["constant_1"] = (float(-y.mean()), 0.0, int(fl.sum()), len(y))
    return out, maxerr


FS = {}
for pipe, dn, codes in (("1W-MPI", "F_blind_1w_mpi", CODE_1W), ("BD-MPI", "F_blind_bd_mpi", CODE_BD)):
    FS[pipe], me = floor_stats(dn, codes, pipe)
    print(f"bias_vs_floor {pipe}: max |bias - seed_methods bias| = {me:.2e} over {len(codes)} methods; "
          f"n at floor {FS[pipe]['absolute'][2]}/{FS[pipe]['absolute'][3]}")
    assert me < 1e-8
fd = pd.read_csv(OUT_DIR / "seed_floor_decomp.csv")
ref_jc = fd[(fd.pipeline == "BD-MPI") & (fd.ensemble == "N = 100") & (fd.method == "Joint, calibrated")]["log MAE at floor"].iloc[0]
print(f"bias_vs_floor check: BD-MPI joint cal floor-case log MAE mine {FS['BD-MPI']['joint_cal'][1]:.4f} vs seed_floor_decomp {ref_jc:.4f}")
assert abs(FS["BD-MPI"]["joint_cal"][1] - ref_jc) < 5e-4

fig, ax = plt.subplots(figsize=(HALF_W, 2.9))
rows_b = []
for pipe, filled in (("1W-MPI", False), ("BD-MPI", True)):
    for role, (bias, fmae, nf, n) in FS[pipe].items():
        if role == "constant_1" or role in ABLATIONS:
            continue
        fam = FAMILY[role]
        ax.scatter(bias, fmae, s=30, marker=FAM_MARK[fam], facecolor=FAM_COLOR[fam] if filled else "white",
                   edgecolor=FAM_COLOR[fam], lw=1.1, zorder=3)
        rows_b.append(dict(pipeline=pipe, role=role, bias=bias, **{"floor-case log MAE": fmae}))
b0 = M[(M.pipeline == "BD-MPI") & (M.role == "absolute")].iloc[0]
mb = FS["BD-MPI"]["constant_1"]
ax.scatter(mb[0], mb[1], s=40, marker="*", color="black", zorder=4)
ax.annotate("baseline", (mb[0], mb[1]), xytext=(7, -3), textcoords="offset points", fontsize=7.5)
ax.axvline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("Signed log bias (pred. − true)")
ax.set_ylabel("Floor-case log MAE")
ax.grid(lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f], ls="", ms=5, label=f) for f in FAM_COLOR]
h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=5, label="one-way"),
      Line2D([], [], marker="o", color="black", ls="", ms=5, label="bidirectional")]
ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", ncol=1, handletextpad=0.2, borderaxespad=0.1)
save(fig, "bias_vs_floor.pdf")
rows_b.append(dict(pipeline="baseline", role="constant_1", bias=mb[0], **{"floor-case log MAE": mb[1]}))
table("bias_vs_floor.pdf", pd.DataFrame(rows_b),
      "N=100 ensemble = max(0, mean latent over 100 seeds); floor-case log MAE = mean |pred - y| over the 18 outcomes with y = 0; bias = mean(pred - y) over 90 outcomes (matches seed_methods.csv to <1e-8).")

# ======================================================================= 3 per-victim heatmap
PV = pd.read_csv(OUT_DIR / "seed_pervictim.csv")
ab = PV[PV.role == "absolute"].ratio
print(f"pervictim: absolute column dropped (ratio min {ab.min():.3f} max {ab.max():.3f}; =1.00 at N=100 for all pipelines).")
COLS = [("1W-MPI", "delta_median", "1W Delta, median"), ("BD-MPI", "delta_median", "BD Delta, median"),
        ("BD-MPI", "joint_cal", "BD Joint, cal."), ("1W-TT", "delta_median", "1W Delta, median"),
        ("BD-TT", "delta_median", "BD Delta, median"), ("BD-TT", "joint_cal", "BD Joint, cal.")]
mat = pd.DataFrame({(p_, r_): PV[(PV.pipeline == p_) & (PV.role == r_)].set_index("victim").ratio.reindex(APPS)
                    for p_, r_, _ in COLS})
mat.index = [NAME[a] for a in mat.index]
mat = mat.loc[np.exp(np.log(mat).mean(axis=1)).sort_values().index]
cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#92c5de", "#f2f2f2", "#f4a582", "#b2182b"])
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(mat.to_numpy(), cmap=cmap, norm=LogNorm(vmin=0.5, vmax=2.0), aspect="auto")  # values outside are clipped
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.iat[i, j]
        dark = v < 0.62 or v > 1.6
        ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if dark else "#222222")
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels([c[2] for c in COLS], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels(mat.index)
ax.axvline(2.5, color="black", lw=1.0)
ax.text(1.0, -0.9, "MPI time", ha="center", fontsize=8)
ax.text(4.0, -0.9, "total time", ha="center", fontsize=8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0.5, 0.75, 1, 1.5, 2], extend="max")
cb.ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2"])
cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.set_label("Victim log MAE ÷ baseline")
cb.outline.set_visible(False)
save(fig, "pervictim_heatmap.pdf")
flat = mat.copy()
flat.columns = [f"{p_} {r_}" for p_, r_, _ in COLS]
flat.insert(0, "victim", flat.index)
table("pervictim_heatmap.pdf", flat, "Victim log MAE / baseline, N=100. Colour log-scaled, saturating at 0.5 and 2 (true value annotated; >=10 rounded).")

# ======================================================================= 4 CV vs App-App
D = json.load(open(OUT_DIR / "artifact_data_20261017.json"))["runs"]
CVMAP = {"1W-MPI": ("baseSnap", [("low_rank", "delta_uniform"), ("generic", "generic"), ("absolute", "absolute")]),
         "BD-MPI": ("bidir", [("legacy_delta", "delta_uniform"), ("legacy_generic", "generic"),
                              ("legacy_absolute", "absolute"), ("joint_profile", "joint_cal"),
                              ("bidirectional_profile", "bprof_cal"), ("bidirectional_context", "bctx_cal")])}
fig, ax = plt.subplots(figsize=(HALF_W, 2.8))
rows_c = []
for pipe, filled in (("1W-MPI", False), ("BD-MPI", True)):
    run, lst = CVMAP[pipe]
    for cvk, role in lst:
        fam = FAMILY[role]
        x = D[run]["cv"][cvk]["cv_log_mae"]
        y = M[(M.pipeline == pipe) & (M.role == role)]["log MAE"].iloc[0]
        ax.scatter(x, y, marker=FAM_MARK[fam], s=30, facecolor=FAM_COLOR[fam] if filled else "white",
                   edgecolor=FAM_COLOR[fam], lw=1.1, zorder=3)
        rows_c.append(dict(pipeline=pipe, role=role, **{"CV log MAE": x, "App-App log MAE (N=100)": y}))
base = float(b0.base)
ax.axhline(base, color="black", lw=0.8, ls=":")
ax.text(0.99, base + 0.0015, "baseline", fontsize=7.5, ha="right", transform=ax.get_yaxis_transform())
ax.set_xlabel("App-Inhibitor CV log MAE")
ax.set_ylabel("App-App log MAE")
ax.grid(lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f], ls="", ms=5, label=f) for f in FAM_COLOR if f != "Joint (direct)"]
h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=5, label="one-way"),
      Line2D([], [], marker="o", color="black", ls="", ms=5, label="bidirectional")]
ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", handletextpad=0.2)
save(fig, "cv_vs_appapp.pdf")
table("cv_vs_appapp.pdf", pd.DataFrame(rows_c), f"CV values from artifact_data_20261017.json (as before); baseline log MAE {base:.4f}.")

# ======================================================================= 5 k-curve
ROOT = Path.home() / "hpcResearch/experiment_outputs/seed_batch/full"
KS = [1, 5, 10, 25, 50, 100]
CURVES = [("BD-MPI", "mpi", "joint_profile_calibrated", "Joint, calibrated", OI["orange"], "D", "-"),
          ("BD-MPI", "mpi", "legacy_absolute", "Absolute response", OI["blue"], "o", "--"),
          ("BD-TT", "tt", "joint_profile_calibrated", "Joint, calibrated", OI["green"], "s", "-.")]
KC = pd.read_csv(OUT_DIR / "seed_kcurve.csv")
cache = {}
rng = np.random.default_rng(925)
res = {}
for pipe, tag, meth, lab, *_ in CURVES:
    if tag not in cache:
        df = pd.read_csv(ROOT / f"F_blind_bd_{tag}/seed_predictions.csv",
                         usecols=["evaluation_method", "anchor_budget", "seed", "pair_row_id", "direction",
                                  "victim_id", "aggressor_id", "true_slowdown", "method",
                                  "latent_predicted_log_slowdown"])
        cache[tag] = df[(df.evaluation_method == "random_split") & (df.anchor_budget == "all")
                        & (df.victim_id != df.aggressor_id)]
    df = cache[tag]
    d = df[df.method == meth]
    P = d.pivot_table(index=["pair_row_id", "direction"], columns="seed", values="latent_predicted_log_slowdown")
    y = np.log(np.maximum(1.0, d.drop_duplicates(["pair_row_id", "direction"]).set_index(["pair_row_id", "direction"])
                          .true_slowdown.reindex(P.index).to_numpy()))
    X = P.to_numpy()
    b = y.mean()
    print(f"kcurve {pipe} {meth}: {X.shape[0]} outcomes x {X.shape[1]} seeds; baseline log MAE {b:.4f}")
    out = []
    for k in KS:
        if k == 1:
            subs = [[s] for s in range(X.shape[1])]
        elif k == X.shape[1]:
            subs = [list(range(k))]
        else:
            subs = [rng.choice(X.shape[1], k, replace=False) for _ in range(1000)]
        dp = np.array([100 * (np.abs(np.maximum(0, X[:, s].mean(1)) - y).mean() / b - 1) for s in subs])
        out.append(dict(k=k, mean=dp.mean(), lo=np.percentile(dp, 5), hi=np.percentile(dp, 95)))
    res[(pipe, lab)] = pd.DataFrame(out)

print("k-curve verification (mine vs seed_kcurve.csv):")
ok = True
vrows = []
for (pipe, lab), r in res.items():
    for k in (1, 5, 100):
        ref = KC[(KC.pipeline == pipe) & (KC.method == lab) & (KC.k == k)]["mean Δ%"].iloc[0]
        mine = r[r.k == k]["mean"].iloc[0]
        vrows.append(dict(pipeline=pipe, method=lab, k=k, mine=mine, csv=ref))
        print(f"  {pipe} {lab} k={k}: mine {mine:+.2f}  csv {ref:+.2f}")
        tol = 0.05 if k == 100 else (6.0 if k == 5 else 1e-6)
        ok &= abs(mine - ref) < tol
print("VERIFICATION", "PASSED" if ok else "FAILED")
if not ok:
    raise SystemExit("k-curve verification failed")

LT = 25
fig, ax = plt.subplots(figsize=(HALF_W, 2.8))
for pipe, tag, meth, lab, col, mk, ls in CURVES:
    r = res[(pipe, lab)]
    ax.fill_between(r.k, r.lo, r.hi, color=col, alpha=0.15, lw=0)
    ax.plot(r.k, r["mean"], color=col, marker=mk, ls=ls, ms=4, lw=1.3, label=f"{pipe}, {lab}")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=LT, linscale=1.0)
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
yt = [-25, 0, 25, 100, 300, 1000]
ax.set_yticks(yt)
ax.set_yticklabels([f"{t:+d}" if t else "0" for t in yt])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Seeds averaged, $k$")
ax.set_ylabel(r"$\Delta$ log MAE vs. baseline (%)")
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), handletextpad=0.3)
save(fig, "seed_kcurve.pdf")
print(f"kcurve: symlog y axis, linthresh {LT} (linear within +-{LT}%, log beyond); band = 5-95% of 1000 random k-subsets.")
kt = pd.concat([r.assign(pipeline=p_, method=m_) for (p_, m_), r in res.items()])[["pipeline", "method", "k", "mean", "lo", "hi"]]
kt.columns = ["pipeline", "method", "k", "mean Δ%", "5% Δ%", "95% Δ%"]
table("seed_kcurve.pdf", kt, "Recomputed from seed_predictions.csv (rng 925, 1000 k-subsets; k=1 every seed; k=100 full set). Symlog y, linthresh 25.")
table("seed_kcurve.pdf verification", pd.DataFrame(vrows))

# ======================================================================= 6 socket information effect
S = pd.read_csv(OUT_DIR / "seed_socket.csv")
S = S[(S.level == "resolved") & S.pipeline.isin(["1W-MPI", "BD-MPI"])
      & S.comparison.isin(["information effect E1 − D1", "init/architecture D1 − blind"])]
SROLES = ["absolute", "delta_uniform", "delta_median", "generic", "joint_cal", "bprof_cal", "bctx_cal"]
CMP = {"information effect E1 − D1": ("E1 − D1 (socket information)", OI["vermillion"], "s"),
       "init/architecture D1 − blind": ("D1 − blind (init./architecture)", OI["blue"], "o")}
rl = {p_: [r for r in SROLES if ((S.pipeline == p_) & (S.role == r)).any()] for p_ in ("1W-MPI", "BD-MPI")}
fig, axes = plt.subplots(2, 1, figsize=(HALF_W, 3.9), sharex=True,
                         gridspec_kw=dict(height_ratios=[len(rl["1W-MPI"]) + 0.8, len(rl["BD-MPI"]) + 0.8], hspace=0.3))
rows_s = []
for ax, pipe in zip(axes, ("1W-MPI", "BD-MPI")):
    ax.axvline(0, color="black", lw=0.8, ls=":")
    for i, role in enumerate(rl[pipe]):
        for cmp_, (lab, col, mk) in CMP.items():
            r = S[(S.pipeline == pipe) & (S.role == role) & (S.comparison == cmp_)]
            if r.empty:
                continue
            r = r.iloc[0]
            lo, hi = ci(r["CI pairs×seeds"])
            dy = -0.17 if cmp_.startswith("info") else 0.17
            hollow = r.verdict == "sign-unstable"
            ax.errorbar(r["Δ log MAE"], i + dy, xerr=[[r["Δ log MAE"] - lo], [hi - r["Δ log MAE"]]], fmt=mk, ms=4.2,
                        color=col, mfc="white" if hollow else col, mec=col, elinewidth=0.9, capsize=1.6, mew=0.9)
            rows_s.append(dict(pipeline=pipe, role=role, comparison=lab, **{"Δ log MAE": r["Δ log MAE"], "CI lo": lo,
                                                                          "CI hi": hi, "verdict": r.verdict}))
    ax.set_yticks(range(len(rl[pipe])))
    ax.set_yticklabels([LABEL[r] for r in rl[pipe]], fontsize=7.5)
    ax.set_ylim(len(rl[pipe]) - 0.4, -0.6)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_title(PIPE_NAME[pipe], fontsize=8, loc="left")
axes[1].set_xlabel(r"$\Delta$ log MAE (negative = better)")
h = [Line2D([], [], marker=mk, color=col, ls="", ms=4.5, label=lab) for lab, col, mk in CMP.values()]
h.append(Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=4.5, label="hollow marker = sign-unstable across seeds"))
fig.legend(handles=h, loc="lower center", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.12))
save(fig, "socket_info_effect.pdf")
table("socket_info_effect.pdf", pd.DataFrame(rows_s), "Level = resolved; bars = two-level CI (pairs×seeds). Hollow = verdict sign-unstable.")

# ======================================================================= 7 learning curves
LC = pd.read_csv(OUT_DIR / "seed_restricted_curves.csv")
LC_ROLES = {"1W-MPI": [("absolute", "Absolute response"), ("delta_uniform", "Delta, uniform agg.")],
            "BD-MPI": [("absolute", "Absolute response"), ("delta_uniform", "Delta, uniform agg."),
                       ("joint_cal", "Joint, calibrated"), ("bprof_direct", "Bidir. profile")]}
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.4), sharex=False, sharey="row")  # per-column tick labels
rows_l = []
for c, pipe in enumerate(["1W-MPI", "BD-MPI"]):
    for r_, tier in enumerate(["one_known", "zero_shot"]):
        ax = axes[r_, c]
        for role, lab in LC_ROLES[pipe]:
            sub = LC[(LC.pipeline == pipe) & (LC.tier == tier) & (LC.method == lab)].sort_values("size")
            fam = FAMILY[role]
            col = OI["sky"] if role == "bprof_direct" else FAM_COLOR[fam]
            ls = "--" if role == "bprof_direct" else "-"
            old = 1 - sub["skill % seeds 0–4"] / 100
            ax.plot(sub["size"], old, color=col, ls=":", lw=0.8, alpha=0.5)
            ratio_ = 1 - sub["skill % (N=30)"] / 100
            ax.plot(sub["size"], ratio_, color=col, ls=ls, lw=1.3)
            for (_, row), rv in zip(sub.iterrows(), ratio_):
                full = row.splits >= 10
                ax.plot(row["size"], rv, marker=FAM_MARK[fam], ms=4.5, color=col, mfc=col if full else "white",
                        mec=col, mew=1.0)
                rows_l.append(dict(pipeline=pipe, tier=tier, method=lab, size=row["size"], splits=row.splits,
                                   **{"ratio N=30": rv, "ratio seeds 0-4": 1 - row["skill % seeds 0–4"] / 100}))
        ax.axhline(1, color="black", lw=0.8, ls=":")
        ax.set_xlim(3.7, 8.3)
        ax.set_yscale("log")
        ax.grid(lw=0.4, alpha=0.4)
        if r_ == 0:
            ax.set_xticks([4, 5, 6, 7, 8])
            ax.tick_params(labelbottom=False)
            ax.set_title(PIPE_NAME[pipe], fontsize=9, loc="left")
        if c == 0:
            ax.set_ylabel(("One-known" if tier == "one_known" else "Zero-shot") + "\nerror ratio")
        if r_ == 1:
            ns = LC[(LC.pipeline == pipe) & (LC.tier == tier) & (LC.method == "Absolute response")].set_index("size").splits
            ax.set_xlabel("Training apps (completed splits)")
            ax.set_xticks([4, 5, 6, 7, 8])
            ax.set_xticklabels([f"{s_}\n({int(ns[s_])})" for s_ in [4, 5, 6, 7, 8]], fontsize=8)
            print(f"learning_curves ticks {pipe} {tier}: {[int(ns[s_]) for s_ in [4,5,6,7,8]]}")
for ax in axes.ravel():
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in [0.8, 0.9, 1, 1.25, 1.5, 2, 3] if lo <= t <= hi])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
handles = [Line2D([], [], color=FAM_COLOR["Absolute"], marker="o", label="Absolute response"),
           Line2D([], [], color=FAM_COLOR["Delta"], marker="s", label="Delta, uniform agg."),
           Line2D([], [], color=FAM_COLOR["Joint (calibrated)"], marker="D", label="Joint, calibrated"),
           Line2D([], [], color=OI["sky"], marker="^", ls="--", label="Bidir. profile (direct)"),
           Line2D([], [], color="grey", marker="o", mfc="white", ls="", label="< 10 splits completed"),
           Line2D([], [], color="grey", ls=":", lw=0.8, alpha=0.6, label="5 seeds (artifact)")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.07),
           columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "learning_curves.pdf")
mm = max(abs(r["ratio N=30"] - r["ratio seeds 0-4"]) for r in rows_l)
print("learning_curves: skill is % in the csv; ratio = 1 - skill/100, N=30 ensemble; dotted = seeds 0-4 column.")
old_art = pd.DataFrame(json.load(open(OUT_DIR / "artifact_data_20261017.json"))["runs"]["bidir"]["curve"])
chk = []
for tier in ("one_known", "zero_shot"):
    for role, lab in LC_ROLES["BD-MPI"]:
        a = old_art[(old_art.tier == tier) & (old_art.role == role)].sort_values("size")
        b = LC[(LC.pipeline == "BD-MPI") & (LC.tier == tier) & (LC.method == lab)].sort_values("size")
        chk.append(np.abs(a.skill.to_numpy() * 100 - b["skill % seeds 0–4"].to_numpy()).max())
        chk.append(np.abs(a.n_splits.to_numpy() - b.splits.to_numpy()).max())
print(f"learning_curves check: max |artifact BD-MPI skill*100 - 'skill % seeds 0-4'| and split-count diff = {max(chk):.2e}")
table("learning_curves.pdf", pd.DataFrame(rows_l),
      "ratio = 1 - skill/100; N=30 ensemble (solid) vs seeds 0-4 (dotted, equals the old artifact curves). Hollow = <10 splits.")

(OUT_DIR / "figs_seed_numbers.md").write_text("# Seed-batch figure numbers (written by figs_seed.py)\n\n" + "\n".join(NUM))
print("wrote", OUT_DIR / "figs_seed_numbers.md")
