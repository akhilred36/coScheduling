"""Plan v2 characterization figures (C): reads the tables written by v2_build.py and v2_char.py.

Writes PDFs to Dissertation/figures/node_sharing/ and output/v2_char_figs_numbers.md.
Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/v2/v2_char_figs.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (APPS, FULL_W, HALF_W, NAME, OI, OUT_DIR, SEL_LABELS, inhibitors, jobs,  # noqa: E402
                    save, selection_features, set_style)
from v2_build import PANEL, jms_factors  # noqa: E402

set_style()
log = io.StringIO()
NAME_TO_ID = {v: k for k, v in NAME.items()}
DISP = [NAME[a] for a in APPS]


def p(*a):
    print(*a)
    print(*a, file=log)


def fig_note(name, size, text):
    p(f"## {name}.pdf ({size[0]:.1f} x {size[1]:.1f} in)\n{text}\n")


def place_labels(ax, xs, ys, labels, fontsize=6.5, obstacles=(), force=None):
    """Greedy label placement: try 8 offsets per point, keep the one overlapping least with points/labels."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    placed = [o.get_window_extent(r).expanded(1.05, 1.1) for o in obstacles]
    cands = [(4, 3, "left", "bottom"), (4, -3, "left", "top"), (-4, 3, "right", "bottom"), (-4, -3, "right", "top"),
             (0, 6, "center", "bottom"), (0, -6, "center", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center")]
    axbox = ax.get_window_extent(r)
    force = force or {}
    for k, lab in enumerate(labels):
        if lab in force:
            dx, dy, ha, va = force[lab]
            t = ax.annotate(lab, (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            ha=ha, va=va)
            placed.append(t.get_window_extent(r).expanded(1.05, 1.1))
    for k in np.argsort(-pts[:, 1]):
        if labels[k] in force:
            continue
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(labels[k], (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha=ha, va=va)
            bb = t.get_window_extent(r).expanded(1.05, 1.1)
            cost = sum(max(0, min(bb.x1, o.x1) - max(bb.x0, o.x0)) * max(0, min(bb.y1, o.y1) - max(bb.y0, o.y0))
                       for o in placed)
            cost += sum(4.0 for q in pts if bb.x0 - 3 < q[0] < bb.x1 + 3 and bb.y0 - 3 < q[1] < bb.y1 + 3)
            if bb.x0 < axbox.x0 or bb.x1 > axbox.x1 or bb.y0 < axbox.y0 or bb.y1 > axbox.y1:
                cost += 1e4
            t.remove()
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, va, bb)
        _, dx, dy, ha, va, bb = best
        ax.annotate(labels[k], (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    ha=ha, va=va)
        placed.append(bb)


TIMERS = [("mpi", "MPI time"), ("tt", "app time")]

# ------------------------------------------------------------------ data
dirm = pd.read_csv(OUT_DIR / "v2_appapp_directed.csv")
grid = pd.read_csv(OUT_DIR / "v2_appinh_grid.csv")
panel = pd.read_csv(OUT_DIR / "v2_panel_app.csv")
B2 = pd.read_csv(OUT_DIR / "v2_char_B2_corun_socket.csv")
cov = pd.read_csv(OUT_DIR / "v2_char_B3_coverage.csv")
pva = pd.read_csv(OUT_DIR / "v2_char_B3_panel_vs_appapp.csv")
curve = pd.read_csv(OUT_DIR / "v2_char_B4_curve.csv")


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float))
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


# ================================================================== 1. c_ecdf_unclipped
size = (FULL_W, 2.5)
fig, axes = plt.subplots(1, 2, figsize=size)
ns = dirm[~dirm.is_self]
series = [("App-App, non-self (90)", lambda tk: ns[f"raw_{tk}"], OI["blue"], "-"),
          ("App-Inh, full grid (2,870)", lambda tk: grid[f"raw_{tk}"], OI["vermillion"], "--"),
          ("App-Inh, panel (90)", lambda tk: panel[f"raw_{tk}"], OI["green"], "-.")]
txt = []
for ax, (tk, lab) in zip(axes, TIMERS):
    for name, get, c, ls in series:
        ecdf(ax, get(tk), color=c, ls=ls, label=name)
        v = get(tk)
        txt.append(f"{lab} | {name}: median {np.median(v):.3f}, <1 {np.mean(v < 1):.1%}, >=1.05 {np.mean(v >= 1.05):.1%}, "
                   f"min {v.min():.3f}, max {v.max():.3f}")
    ax.axvline(1, color=OI["grey"], lw=0.8)
    ax.axvline(1.05, color=OI["black"], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(f"{lab} slowdown $T_{{\\mathrm{{co}}}}/T_{{\\mathrm{{iso}}}}$")
    ax.set_ylim(0, 1.01)
    ax.grid(lw=0.4, alpha=0.4)
axes[0].set_xlim(0.65, 9)
axes[1].set_xlim(0.9, 1.55)
for ax in axes:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_xticks([0.7, 1, 1.5, 2, 3, 5, 8])
axes[1].set_xticks([0.9, 1, 1.1, 1.2, 1.3, 1.5])
axes[0].set_ylabel("Fraction of outcomes")
axes[1].legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(w_pad=1.5)
save(fig, "c_ecdf_unclipped.pdf")
plt.close(fig)
fig_note("c_ecdf_unclipped", size, "ECDFs of unclipped JMs ratios; solid grey line = 1, dotted = 1.05; log x.\n" +
         "\n".join(f"- {t}" for t in txt))

# ================================================================== 2. c_heatmaps_unclipped
size = (FULL_W, 8.6)
fig, axes = plt.subplots(2, 1, figsize=size)
txt = []
for ax, (tk, lab) in zip(axes, TIMERS):
    M = dirm.pivot(index="victim", columns="corunner", values=f"raw_{tk}").loc[APPS, APPS]
    L = np.log(M.to_numpy())
    lim = np.abs(L).max()
    im = ax.imshow(L, cmap="RdBu_r", norm=Normalize(-lim, lim), aspect="equal")
    for i in range(10):
        for j in range(10):
            v = M.to_numpy()[i, j]
            dark = abs(L[i, j]) > 0.55 * lim
            ax.text(j, i, f"{v:.3f}" if tk == "tt" else f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if dark else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10), DISP, rotation=40, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(10), DISP, fontsize=8)
    ax.set_xlabel("Co-runner")
    ax.set_ylabel("Victim")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ticks = np.array([1 / np.exp(lim), 1 / np.exp(lim / 2), 1, np.exp(lim / 2), np.exp(lim)])
    cb.set_ticks(np.log(ticks))
    cb.set_ticklabels([f"{t:.3f}" if tk == "tt" else f"{t:.2f}" for t in ticks])
    cb.set_label(f"{lab} slowdown (log scale)")
    cb.outline.set_linewidth(0.5)
    txt.append(f"{lab}: colour limits ratio [{np.exp(-lim):.4f}, {np.exp(lim):.4f}] (symmetric in log); "
               f"max {M.to_numpy().max():.3f}, min {M.to_numpy().min():.3f}; diagonal = self-pairs (pooled copies, bold)")
fig.tight_layout(h_pad=1.5)
save(fig, "c_heatmaps_unclipped.pdf")
plt.close(fig)
fig_note("c_heatmaps_unclipped", size, "Rows = victim, columns = co-runner; rep-averaged JMs ratio, RdBu_r centred at 1.\n" +
         "\n".join(f"- {t}" for t in txt))

# ================================================================== 3. c_socket_isolated
size = (FULL_W, 2.9)
f = jms_factors()
fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
order = sorted(APPS, key=lambda a: NAME[a].lower())[::-1]
ypos = {a: i for i, a in enumerate(order)}
txt = []
for ax, (tk, lab) in zip(axes, TIMERS):
    t = f[f.timer == tk].set_index("app")
    for a in order:
        r = t.loc[a]
        sig = bool(r.applied)
        ax.errorbar(r.d, ypos[a], xerr=[[r.d - r.ci_lo], [r.ci_hi - r.d]], fmt="o" if sig else "s",
                    ms=5 if sig else 4, color=OI["vermillion"] if sig else OI["blue"],
                    mfc=OI["vermillion"] if sig else "white", mew=1.0, elinewidth=1.0, capsize=2)
        txt.append(f"{lab} {NAME[a]}: {r.d:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]{' *' if sig else ''}")
    ax.axvline(0, color=OI["grey"], lw=0.8)
    ax.set_xlabel(f"Isolated {lab}: ln($T_{{s1}}/T_{{s0}}$)")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
axes[0].set_yticks(range(10), [NAME[a] for a in order])
axes[0].set_xlim(-1.55, 0.35)
axes[1].set_xlim(-0.14, 0.06)
h = [Line2D([], [], marker="o", color=OI["vermillion"], ls="", label="95% CI excludes 0"),
     Line2D([], [], marker="s", color=OI["blue"], mfc="white", ls="", label="CI includes 0")]
axes[1].legend(handles=h, frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout(w_pad=1.0)
save(fig, "c_socket_isolated.pdf")
plt.close(fig)
fig_note("c_socket_isolated", size, "Paired pinned isolated runs (n = 4 per app), t 95% CI; negative = socket 1 faster.\n" +
         "\n".join(f"- {t}" for t in txt))

# ================================================================== 4. c_socket_corun
size = (FULL_W, 3.6)
fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
camp = {"App-App": (OI["blue"], "o", -0.0), "App-Inh": (OI["vermillion"], "s", 0.0)}
offs = {("App-App", "JMs"): -0.27, ("App-App", "J"): -0.09, ("App-Inh", "JMs"): 0.09, ("App-Inh", "J"): 0.27}
txt = []
for ax, (tk, lab) in zip(axes, TIMERS):
    pre = "(iv) MPI time" if tk == "mpi" else "(iii) app time"
    sub = B2[B2.target == pre]
    for (cname, base), off in offs.items():
        s = sub[sub.campaign.str.startswith(cname) & (sub.baseline == base)].set_index("app")
        c, mk, _ = camp[cname]
        for a in order:
            if NAME[a] not in s.index:
                continue
            r = s.loc[NAME[a]]
            if not np.isfinite(r.paired_d):
                continue
            lo, hi = r.ci_lo, r.ci_hi
            ax.errorbar(r.paired_d, ypos[a] + off, xerr=[[r.paired_d - lo], [hi - r.paired_d]], fmt=mk, ms=4,
                        color=c, mfc=c if base == "JMs" else "white", mew=1.0, elinewidth=0.9, capsize=1.5)
            txt.append(f"{lab} {cname} {base} {NAME[a]}: {r.paired_d:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={int(r.n_paired)})")
    ax.axvline(0, color=OI["grey"], lw=0.8)
    ax.set_xlabel(f"Co-run {lab}: ln($T_{{s1}}/T_{{s0}}$)")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
axes[0].set_yticks(range(10), [NAME[a] for a in order])
axes[1].set_xlim(-0.065, 0.03)
h = [Line2D([], [], marker="o", color=OI["blue"], ls="", label="App-App, socket-corrected baseline"),
     Line2D([], [], marker="o", color=OI["blue"], mfc="white", ls="", label="App-App, uncorrected baseline"),
     Line2D([], [], marker="s", color=OI["vermillion"], ls="", label="App-Inh grid, socket-corrected baseline"),
     Line2D([], [], marker="s", color=OI["vermillion"], mfc="white", ls="", label="App-Inh grid, uncorrected baseline")]
fig.legend(handles=h, frameon=False, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92))
save(fig, "c_socket_corun.pdf")
plt.close(fig)
fig_note("c_socket_corun", size, "Paired within (victim, co-runner/config) socket-1 minus socket-0 log slowdown, 95% CI "
         "(from v2_char_B2_corun_socket.csv); solid = JMs baseline, hollow = June baseline.\n" +
         "\n".join(f"- {t}" for t in txt))

# ================================================================== 5. c_coverage
size = (FULL_W, 3.6)
j = jobs()
inh_all = inhibitors().sort_values("inhib_id").reset_index(drop=True)
Xa = selection_features(j.reset_index().rename(columns={"index": "job_id"}))
Xi = selection_features(inh_all)
is_panel = inh_all.inhib_id.isin(PANEL).to_numpy()
pool = np.vstack([Xa, Xi])
mu, sd = pool.mean(0), pool.std(0)
Za, Zi = (Xa - mu) / sd, (Xi - mu) / sd
_, S, Vt = np.linalg.svd(np.vstack([Za, Zi]), full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
Pa, Pi = Za @ Vt[:2].T, Zi @ Vt[:2].T
inhull = cov.set_index("app").in_hull_grid
fig, ax = plt.subplots(figsize=size)
ax.scatter(Pi[~is_panel, 0], Pi[~is_panel, 1], s=7, color="#b0b0b0", lw=0, label="Inhibitor grid (287)")
ax.scatter(Pi[is_panel, 0], Pi[is_panel, 1], s=46, marker="D", facecolor="none", ec=OI["orange"], lw=1.4,
           label="Selected panel (9)")
for k, a in enumerate(j.index):
    inside = bool(inhull[NAME[a]])
    c = OI["vermillion"] if a == "quicksilver" else OI["blue"]
    ax.scatter(Pa[k, 0], Pa[k, 1], s=40, marker="o", color=c if inside else "white", ec=c, lw=1.2, zorder=5)
offs = {"amg": (6, -10), "beatnik": (6, -9), "fiesta": (6, -3), "kripke": (6, 2), "laghos": (6, -8),
        "lammps": (-6, 2), "minife": (6, 3), "minivite": (-6, 3), "quicksilver": (6, 3), "tricount": (6, 3)}
for k, a in enumerate(j.index):
    dx, dy = offs.get(a, (5, 4))
    ax.annotate(NAME[a], Pa[k], xytext=(dx, dy), textcoords="offset points", fontsize=8,
                ha="right" if dx < 0 else "left", zorder=6,
                color=OI["vermillion"] if a == "quicksilver" else "black")
ax.set_xlabel(f"PC1 ({ev[0] * 100:.0f}% of variance)")
ax.set_ylabel(f"PC2 ({ev[1] * 100:.0f}% of variance)")
ax.grid(lw=0.4, alpha=0.4)
h = [Line2D([], [], marker="o", color="#b0b0b0", ls="", ms=3, label="Inhibitor grid (287)"),
     Line2D([], [], marker="D", color=OI["orange"], mfc="none", ls="", label="Selected panel (9)"),
     Line2D([], [], marker="o", color=OI["blue"], ls="", label="App inside grid hull"),
     Line2D([], [], marker="o", color=OI["blue"], mfc="white", ls="", label="App outside grid hull"),
     Line2D([], [], marker="o", color=OI["vermillion"], mfc="white", ls="", label="Quicksilver (far outside)")]
ax.legend(handles=h, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
fig.tight_layout()
save(fig, "c_coverage.pdf")
plt.close(fig)
qs = cov.set_index("app").loc["Quicksilver"]
fig_note("c_coverage", size,
         f"PCA of the 4 standardized selection features ({', '.join(SEL_LABELS)}) on apps + 287 configs; "
         f"PC1 {ev[0]:.3f}, PC2 {ev[1]:.3f} of variance. Filled app marker = inside the 3-D grid hull "
         f"({int(cov.in_hull_grid.sum())}/10); Quicksilver nearest-config distance {qs.nearest_grid:.3f} "
         f"(selection metric).")

# ================================================================== 6. c_panel_vs_appapp
RHO = {("MPI time", "victim"): (0.6485, 0.0485), ("MPI time", "aggr"): (0.6848, 0.0344),
       ("app time", "victim"): (0.8788, 0.0017), ("app time", "aggr"): (0.7333, 0.0201)}
# recompute rho as a check against v2_char_numbers.md
from scipy.stats import spearmanr  # noqa: E402

size = (FULL_W, 6.0)
fig, axes = plt.subplots(2, 2, figsize=size)
txt = []
for row, (tk, lab) in enumerate(TIMERS):
    d = pva[pva.timer == lab].set_index("app")
    for col, (xc, yc, kind, xl, yl, c, mk) in enumerate([
            ("panel_victim", "AA_victim", "victim", "Mean panel ln slowdown (victim)",
             "Mean App-App ln slowdown\n(victim)", OI["blue"], "o"),
            ("panel_caused", "AA_aggressor", "aggr", "Mean inhibitor slowdown caused",
             "Mean App-App ln slowdown\ncaused (co-runner)", OI["vermillion"], "s")]):
        ax = axes[row, col]
        ax.scatter(d[xc], d[yc], color=c, marker=mk, s=26, zorder=3)
        rho_chk = spearmanr(d[xc], d[yc]).statistic
        rho, pv = RHO[(lab, kind)]
        assert abs(rho_chk - rho) < 1e-3, (lab, kind, rho_chk, rho)
        tx, ty, tha, tva = (0.97, 0.03, "right", "bottom") if col == 0 else (0.03, 0.97, "left", "top")
        rt = ax.text(tx, ty, f"{lab[0].upper() + lab[1:]}\n$\\rho$ = {rho:+.2f}, $p$ = {pv:.3f}",
                     transform=ax.transAxes, va=tva, ha=tha, fontsize=8)
        xr = d[xc].max() - d[xc].min()
        yr = d[yc].max() - d[yc].min()
        ax.set_xlim(d[xc].min() - 0.06 * xr, d[xc].max() + 0.25 * xr)
        ax.set_ylim(d[yc].min() - 0.08 * yr, d[yc].max() + 0.12 * yr)
        place_labels(ax, d[xc].to_numpy(), d[yc].to_numpy(), list(d.index), fontsize=6.5, obstacles=[rt],
                     force={"MiniFE": (5, -3, "left", "top"), "FIESTA": (-4, 3, "right", "bottom")} if (tk, col) == ("mpi", 1) else None)
        ax.grid(lw=0.4, alpha=0.4)
        ax.axhline(0, color=OI["grey"], lw=0.6)
        ax.axvline(0, color=OI["grey"], lw=0.6)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        txt.append(f"{lab} {kind}: rho {rho:+.4f} (recomputed {rho_chk:+.4f}), p_perm {pv:.4f}")
fig.tight_layout(h_pad=1.2, w_pad=1.5)
save(fig, "c_panel_vs_appapp.pdf")
plt.close(fig)
fig_note("c_panel_vs_appapp", size, "Per-app means (JMs ln ratio); left = victim side, right = aggressor side "
         "(BD 'caused' = inhibitor time-per-work slowdown). rho/p from v2_char_numbers.md B3(c).\n" +
         "\n".join(f"- {t}" for t in txt))

# ================================================================== 7. p_panel_objective
size = (HALF_W, 2.5)
lb = float(curve.objective.iloc[-1] - curve.excess_over_lb.iloc[-1])
fig, ax = plt.subplots(figsize=size)
ax.plot(curve.k, curve.random_median, color=OI["grey"], ls="--", marker="^", ms=3.5, label="Random panel (median)")
ax.plot(curve.k, curve.objective, color=OI["blue"], ls="-", marker="o", ms=3.5, label="Selected (greedy + swap)")
ax.axhline(lb, color=OI["black"], ls=":", lw=0.9, label=f"Lower bound, all 287 ({lb:.3f})")
ax.plot([9], [curve.objective.iloc[-1]], marker="*", ms=11, color=OI["vermillion"], ls="", zorder=5,
        label="Chosen panel ($k$ = 9)")
ax.set_yscale("log")
ax.set_xlabel("Panel size $k$")
ax.set_ylabel("Coverage objective")
ax.set_xticks(range(1, 10))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (2, 3, 5) else ""))
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_ylim(0.9, 9)
fig.tight_layout()
save(fig, "p_panel_objective.pdf")
plt.close(fig)
fig_note("p_panel_objective", size, "Objective = 0.5 mean + 0.5 max of nearest squared distance; log y.\n" +
         "\n".join(f"- k={int(r.k)}: objective {r.objective:.4f}, gain {r.marginal_gain:.4f}, random median "
                   f"{r.random_median:.3f}" for r in curve.itertuples()) + f"\n- lower bound {lb:.4f}")

# ================================================================== 8. p_panel_grid
size = (FULL_W, 3.7)
inh_all = inhibitors()
covers = {}
for r in cov.itertuples():
    covers.setdefault(r.nearest_panel_id, []).append(f"{r.app} ({r.nearest_panel:.2f})")
SP = sorted(inh_all.comm_sparsity.unique())
SPC = {0.002237136: (OI["vermillion"], "o"), 0.01: (OI["grey"], "v"), 0.05: (OI["orange"], "s"),
       0.1: (OI["blue"], "D"), 0.3: (OI["green"], "^"), 0.5: (OI["purple"], "P"), 0.7: (OI["sky"], "X"),
       1.0: (OI["black"], "*")}


def ypos_w(w):
    return np.where(np.asarray(w) == 0, 10.0, np.asarray(w, float))


fig, ax = plt.subplots(figsize=size)
ax.scatter(inh_all.msg_size, ypos_w(inh_all.wait_time), s=14, color="#c8c8c8", lw=0, marker="o",
           label="Grid config (287; overlapping sparsities)")
pan = inh_all[inh_all.inhib_id.isin(PANEL)].copy()
dup = pan.groupby(["msg_size", "wait_time"]).cumcount()
nd = pan.groupby(["msg_size", "wait_time"]).inhib_id.transform("size")
pan["xp"] = pan.msg_size * np.exp(0.16 * (dup - (nd - 1) / 2) * 2)
ann = {"8_2000000_1000000_0.3": (4, 26, "center"), "8_20000_100000_0.1": (6, -12, "left"),
       "8_200_200000_0.002237136": (6, 6, "left"), "8_2500000_200000_0.002237136": (-8, -12, "right"),
       "8_2500000_600000_0.002237136": (-8, -2, "right"), "8_400000_1000000_0.5": (0, 13, "center"),
       "8_40000_1000000_0.1": (-8, 9, "right"), "8_40000_1000000_0.7": (-2, 26, "center"),
       "8_500_5000_0.05": (6, 6, "left")}
seen = set()
for r in pan.itertuples():
    c, mk = SPC[r.comm_sparsity]
    peers = int(round(r.comm_sparsity * 447))
    lab = f"$s$ = {r.comm_sparsity:.3g} ({peers} peer{'s' if peers > 1 else ''})"
    ax.scatter(r.xp, ypos_w(r.wait_time), s=60, color=c, marker=mk, ec="black", lw=0.5, zorder=5,
               label=None if r.comm_sparsity in seen else lab)
    seen.add(r.comm_sparsity)
    dx, dy, ha = ann[r.inhib_id]
    ax.annotate("\n".join(covers.get(r.inhib_id, [])), (r.xp, ypos_w(r.wait_time)), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.8, ha=ha, va="bottom" if dy > 0 else "top", zorder=6)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=50, linscale=0.4)
ax.set_yticks([10, 100, 1000, 1e4, 1e5, 1e6], ["0", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
ax.set_ylim(5, 1.2e7)
ax.set_xlim(10, 1.2e7)
ax.set_xlabel("Inhibitor message size (bytes)")
ax.set_ylabel("Inhibitor wait (µs)")
ax.grid(lw=0.4, alpha=0.3)
hs, ls_ = ax.get_legend_handles_labels()
order_l = [0] + sorted(range(1, len(ls_)), key=lambda i: float(ls_[i].split("= ")[1].split(" ")[0]))
ax.legend([hs[i] for i in order_l], [ls_[i] for i in order_l], frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
save(fig, "p_panel_grid.pdf")
plt.close(fig)
fig_note("p_panel_grid", size, "Selected configs on (message size, wait); wait 0 drawn at the '0' tick (symlog); "
         "coincident panel configs (40 kB, 1 s: s = 0.1 and 0.7) offset horizontally. Annotation = apps whose nearest "
         "panel config it is (selection-metric distance).\n" +
         "\n".join(f"- {k}: {', '.join(v)}" for k, v in covers.items()))

(OUT_DIR / "v2_char_figs_numbers.md").write_text("# v2 characterization figures (v2/v2_char_figs.py)\n\n" + log.getvalue())
print("wrote", OUT_DIR / "v2_char_figs_numbers.md")
