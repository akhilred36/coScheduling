"""Batch 1 characterization figures C1-C7 for the node-sharing chapter.

Writes PDFs to /home/akhil/hpcResearch/repos/Dissertation/figures/node_sharing/ and
figure-side numbers to output/figs_batch1_numbers.md.
Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/figs_batch1.py
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.optimize import linprog

from common import (APPS, FULL_W, HALF_W, NAME, OI, ONE_NODE_APPS, OUT_DIR, PROCESSED,
                    SEL_LABELS, directional, directional_from_processed, inhibitors, job_inh,
                    jobs, pairs_mpi, pairs_total, read, save, selection_features, set_style)

set_style()
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


C = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"]]
LS = ["-", "--", "-.", ":"]
MK = ["o", "s", "^", "D"]


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.concatenate([[x[0]], x]), np.concatenate([[0], y]), where="post", **kw)


def cv(x):
    return np.std(x, ddof=1) / np.mean(x)


def label_panel(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# ============================================================================ C1 noise_cv
iso = read(PROCESSED / "data_isolated.csv")
i8 = iso[iso["Num Nodes"] == 8]
g = i8.groupby("App")
cv_mpi = g["MPI Time Average"].apply(cv).loc[APPS]
cv_app = g["App Time Average"].apply(cv).loc[APPS]
order = cv_mpi.sort_values(ascending=False).index
ii = read(PROCESSED / "data_inhib_isolated.csv")
K = ["Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]
cv_ii = ii.groupby(K)["Inhib MPI Time Average"].apply(cv)
ai = read(PROCESSED / "data_inhib_coscheduled.csv")
cv_ai = ai.groupby(["App", *K])["MPI Time Average"].apply(cv).dropna()
aa = read(PROCESSED / "data_coscheduled.csv")
ga = aa[aa["Num Nodes"] == 8].groupby(["App A", "App B"])
cv_aa = pd.concat([ga["App A MPI Time Average"].apply(cv), ga["App B MPI Time Average"].apply(cv)])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
xpos = np.arange(len(order))
w = 0.38
a1.bar(xpos - w / 2, cv_mpi[order] * 100, w, color=C[0], label=r"$T_\mathrm{MPI}$", edgecolor="white", linewidth=0.5)
a1.bar(xpos + w / 2, cv_app[order] * 100, w, color=C[1], label=r"$T_\mathrm{app}$", edgecolor="white", linewidth=0.5,
       hatch="////")
a1.set_yscale("log")
a1.set_ylim(0.1, 50)
a1.set_yticks([0.1, 0.3, 1, 3, 10, 30])
a1.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
a1.set_ylabel("Replicate CV (%)")
a1.set_xticks(xpos)
a1.set_xticklabels([NAME[a] for a in order], rotation=45, ha="right")
a1.legend(frameon=False, loc="upper right")
a1.grid(axis="y", lw=0.4, alpha=0.4)
label_panel(a1, "(a)", x=-0.14)
for k, (vals, lab) in enumerate([(cv_ii, "Isolated inhibitors"), (cv_ai, "App-Inhibitor co-runs"),
                                 (cv_aa, "App-App co-runs")]):
    ecdf(a2, np.maximum(vals * 100, 0.01), color=C[k], ls=LS[k], label=lab)
a2.set_xscale("log")
a2.set_xlim(0.05, 200)
a2.set_xticks([0.1, 1, 10, 100])
a2.set_xticklabels(["0.1", "1", "10", "100"])
a2.set_xlabel(r"Replicate CV of $T_\mathrm{MPI}$ (%)")
a2.set_ylabel("Cumulative fraction")
a2.set_ylim(0, 1.01)
a2.grid(lw=0.4, alpha=0.4)
a2.legend(frameon=False, loc="upper left", fontsize=8)
label_panel(a2, "(b)", x=-0.2)
fig.tight_layout(w_pad=1.5)
save(fig, "noise_cv.pdf")
plt.close(fig)
p("## C1 noise_cv.pdf")
p("(a) app order (by CV T_MPI, desc):", ", ".join(NAME[a] for a in order))
p(f"(b) n: isolated inhibitor configs {len(cv_ii)}, App-Inhibitor configs {len(cv_ai)}, App-App endpoint outcomes {len(cv_aa)} (incl. self)")
p()

# ============================================================================ C2 slowdown_ecdf
pm = directional(pairs_mpi(), "raw_mpi_time_ratio")
pt = directional(pairs_total(), "raw_app_time_ratio")
nm, nt = pm[~pm.self], pt[~pt.self]
fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
ecdf(ax, nm.slowdown, color=C[0], ls="-", label="MPI time")
ecdf(ax, nt.slowdown, color=C[1], ls="--", label="Total time")
ax.axvline(1.05, color=OI["grey"], ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlim(0.99, 2.35)
ticks = [1, 1.1, 1.25, 1.5, 2]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Slowdown (clipped at 1)")
ax.set_ylabel("Cumulative fraction")
ax.set_ylim(0, 1.01)
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, loc="lower right")
save(fig, "slowdown_ecdf.pdf")
plt.close(fig)
p("## C2 slowdown_ecdf.pdf: log x, 90 non-self outcomes each; dotted line at 1.05")
p(f"ECDF at 1.05: MPI {(nm.slowdown < 1.05).mean():.3f}, total {(nt.slowdown < 1.05).mean():.3f}")
p()


# ============================================================================ C3 heatmaps
def heat_matrix(pairs, col_a="slowdown_A", col_b="slowdown_B"):
    M = pd.DataFrame(np.nan, index=APPS, columns=APPS)
    for r in pairs.itertuples(index=False):
        d = r._asdict()
        a, b = d["jobA_id"], d["jobB_id"]
        if a == b:
            # Self-pair: two copies of the same app co-run. The diagonal shows the MEAN
            # of the two copies' slowdowns; draw_heat outlines these cells.
            M.loc[a, a] = (d[col_a] + d[col_b]) / 2
        else:
            M.loc[a, b] = d[col_a]  # victim a, aggressor b
            M.loc[b, a] = d[col_b]
    return M


def draw_heat(M, fname, vmax, label, size):
    off = M.where(~np.eye(len(M), dtype=bool))
    rows = off.mean(axis=1).sort_values(ascending=False).index
    cols = off.mean(axis=0).sort_values(ascending=False).index
    M = M.loc[rows, cols]
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=1.0, vmax=vmax, aspect="equal")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = M.loc[r, c]
            frac = (min(v, vmax) - 1) / (vmax - 1)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if frac > 0.55 else "#222222")
            if r == c:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.2))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NAME[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME[r] for r in rows])
    ax.set_xlabel("Aggressor (co-runner)")
    ax.set_ylabel("Victim")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    cb.outline.set_visible(False)
    save(fig, fname)
    plt.close(fig)
    return rows, cols


rows, cols = draw_heat(heat_matrix(pairs_mpi()), "appapp_heatmap.pdf", 2.25, "MPI-time slowdown", (4.6, 4.2))
p("## C3 appapp_heatmap.pdf: rows sorted by victim mean (off-diagonal, desc), cols by aggressor mean (desc); "
  "diagonal = mean of the two self-pair copies, boxed")
p("rows:", ", ".join(NAME[r] for r in rows))
p("cols:", ", ".join(NAME[c] for c in cols))
rows, cols = draw_heat(heat_matrix(pairs_total()), "appapp_heatmap_total.pdf", 1.30, "Total-time slowdown", (4.6, 4.2))
p("appapp_heatmap_total.pdf rows:", ", ".join(NAME[r] for r in rows), "| cols:", ", ".join(NAME[c] for c in cols))
p()

# ============================================================================ C4 slowdown_vs_profile
j = jobs()
# Same definitions as select_inhibitors.communication_features: rates over total runtime T_app
# (runtime = mpi_time / comm_frac).
byte_rate = j.total_bytes / (j.mpi_time / j.comm_frac)
msg_rate = j.total_msgs / (j.mpi_time / j.comm_frac)
d = nm.copy()
d["agg_byte"] = d.aggressor.map(byte_rate)
d["agg_msg"] = d.aggressor.map(msg_rate)
d["vic_frac"] = d.victim.map(j.comm_frac)
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), sharey=True)
specs = [("agg_byte", "Aggressor byte rate\n(B/s)", "(a)"), ("agg_msg", "Aggressor message rate\n(msgs/s)", "(b)"),
         ("vic_frac", "Victim comm. fraction\n" + r"($T_\mathrm{MPI}/T_\mathrm{app}$)", "(c)")]
p("## C4 slowdown_vs_profile.pdf (90 non-self MPI outcomes; Spearman on clipped slowdown)")
for ax, (col, xl, tag) in zip(axes, specs):
    floor = d.slowdown == 1
    ax.scatter(d.loc[~floor, col], d.loc[~floor, "slowdown"], s=14, color=C[0], alpha=0.8, lw=0.4, ec="white")
    ax.scatter(d.loc[floor, col], d.loc[floor, "slowdown"], s=14, facecolor="none", ec=OI["grey"], lw=0.8)
    rho, pv = stats.spearmanr(d[col], d.slowdown)
    ax.text(0.04, 0.95, rf"$\rho = {rho:.2f}$", transform=ax.transAxes, va="top", fontsize=9)
    p(f"{tag} {xl}: rho = {rho:.3f}, p = {pv:.3g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.grid(lw=0.4, alpha=0.4)
    label_panel(ax, tag, x=0.0)
axes[0].set_ylabel("MPI-time slowdown")
yt = [1, 1.25, 1.5, 2]
axes[0].set_yticks(yt)
axes[0].set_yticklabels([f"{t:g}" for t in yt])
axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(w_pad=0.8)
save(fig, "slowdown_vs_profile.pdf")
plt.close(fig)
p("Hollow grey markers = floor outcomes (slowdown = 1). Victim comm. fraction panel uses log x as requested.")
p()

# ============================================================================ C5 inhib_response
VICTIMS = ["amg", "minivite", "quicksilver"]  # most susceptible / median / insensitive by mean log s
ji = job_inh()
inh = inhibitors().set_index("inhib_id")
ji = ji.join(inh[["msg_size", "wait_time", "comm_sparsity"]], on="inhib_id")
rank = ji.groupby("job_id").log_slowdown.mean().sort_values()
WAITS = [0, 1000, 100000, 1000000]
SIZES = [2000, 40000, 400000, 4000000]
fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 6.0), sharex="col", sharey="row")
YT = [1, 1.25, 1.5, 2, 3, 4, 6]
for i, v in enumerate(VICTIMS):
    sub = ji[ji.job_id == v]
    ax = axes[i, 0]
    for k, wt in enumerate(WAITS):
        s = sub[(sub.comm_sparsity == 1.0) & (sub.wait_time == wt)].sort_values("msg_size")
        lab = "0" if wt == 0 else (rf"$10^{{{int(np.log10(wt))}}}$" if wt else "0")
        ax.plot(s.msg_size, s.slowdown, color=C[k], ls=LS[k], marker=MK[k], ms=3.5, label=f"wait {lab} µs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"{NAME[v]}\nMPI-time slowdown")
    ax.grid(lw=0.4, alpha=0.4)
    ax = axes[i, 1]
    for k, ms in enumerate(SIZES):
        s = sub[(sub.wait_time == 0) & (sub.msg_size == ms) & (sub.comm_sparsity >= 0.1)].sort_values("comm_sparsity")
        lab = {2000: "2 KB", 40000: "40 KB", 400000: "400 KB", 4000000: "4 MB"}[ms]
        ax.plot(s.comm_sparsity, s.slowdown, color=C[k], ls=LS[k], marker=MK[k], ms=3.5, label=lab)
    ax.set_yscale("log")
    ax.grid(lw=0.4, alpha=0.4)
    sub_hi = sub.slowdown.max()
    a_ = axes[i, 0]
    hi = max(1.2, sub[(sub.comm_sparsity == 1.0) | (sub.wait_time == 0)].slowdown.max() * 1.08)
    a_.set_ylim(0.95, hi)
    a_.set_yticks([1, 1.1, 1.2, 1.3] if hi < 1.4 else [t for t in YT if t <= hi])
    a_.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    a_.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for b_ in axes[i]:
        b_.axhline(1, color=OI["grey"], lw=0.7, ls=":")
axes[2, 0].set_xlabel("Inhibitor message size (B)")
axes[2, 1].set_xlabel("Inhibitor sparsity")
axes[2, 1].set_xticks([0.1, 0.3, 0.5, 0.7, 1.0])
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[0, 1].get_legend_handles_labels()
axes[0, 0].legend(h0, l0, frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.04),
                  title="(a) sparsity 1.0; lines = wait", title_fontsize=8, alignment="left")
axes[0, 1].legend(h1, l1, frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.04),
                  title="(b) wait 0; lines = message size", title_fontsize=8, alignment="left")
fig.tight_layout(h_pad=0.6, w_pad=1.0)
fig.align_ylabels(axes[:, 0])
save(fig, "inhib_response.pdf")
plt.close(fig)
p("## C5 inhib_response.pdf")
p("Victims picked by mean log MPI slowdown over all 287 inhibitors (rank 1 = least):")
p("  " + ", ".join(f"{NAME[k]} {v:.3f}" for k, v in rank.items()))
p("  chosen: AMG2023 (most susceptible), miniVite (6th of 10, median-ish), Quicksilver (least)")
p("(a) slices: sparsity = 1.0, wait in {0, 1e3, 1e5, 1e6} us, msg size = base grid {2e3, 4e3, 2e4, 4e4, 2e5, 4e5, 2e6, 4e6} B")
p("(b) slices: wait = 0, msg size in {2 KB, 40 KB, 400 KB, 4 MB}, sparsity in {0.1, 0.3, 0.5, 0.7, 1.0} (base grid)")
p("Both from the base block only (fully populated). y = clipped MPI slowdown (log). Note the App-Inhibitor caveats: "
  "inhibitor runtime coverage (AMG2023 0.98, miniVite 1.00, Quicksilver 1.00 median) and socket effect (AMG2023 -0.28 log).")
for v in VICTIMS:
    sub = ji[ji.job_id == v]
    s1 = sub[sub.comm_sparsity == 1.0]
    p(f"  {NAME[v]}: slowdown range in (a) slice {s1.slowdown.min():.2f}-{s1.slowdown.max():.2f}; "
      f"in (b) slice {sub[(sub.wait_time == 0) & (sub.comm_sparsity >= 0.1)].slowdown.min():.2f}-"
      f"{sub[(sub.wait_time == 0) & (sub.comm_sparsity >= 0.1)].slowdown.max():.2f}")
p()

# ============================================================================ C6 coverage_pca
inh_all = inhibitors().sort_values("inhib_id").reset_index(drop=True)
Xa = selection_features(j.reset_index().rename(columns={"index": "job_id"}))
Xi = selection_features(inh_all)
PANEL = ['8_2000000_1000000_0.3', '8_20000_100000_0.1', '8_200_200000_0.002237136',
         '8_2500000_200000_0.002237136', '8_2500000_600000_0.002237136', '8_400000_1000000_0.5',
         '8_40000_1000000_0.1', '8_40000_1000000_0.7', '8_500_5000_0.05']
is_panel = inh_all.inhib_id.isin(PANEL).to_numpy()
assert is_panel.sum() == 9
pool = np.vstack([Xa, Xi])
mu, sd = pool.mean(0), pool.std(0)
Za, Zi = (Xa - mu) / sd, (Xi - mu) / sd
U, S, Vt = np.linalg.svd(np.vstack([Za, Zi]), full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
Pa, Pi = Za @ Vt[:2].T, Zi @ Vt[:2].T
fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.scatter(Pi[~is_panel, 0], Pi[~is_panel, 1], s=7, color="#b0b0b0", lw=0, label="Inhibitor grid (287)")
ax.scatter(Pi[is_panel, 0], Pi[is_panel, 1], s=42, marker="D", facecolor="none", ec=C[1], lw=1.3,
           label="Selected panel (9)")
ax.scatter(Pa[:, 0], Pa[:, 1], s=36, marker="o", color=C[0], ec="white", lw=0.6, label="Applications (10)", zorder=5)
offs = {"amg": (6, -10), "beatnik": (6, -9), "fiesta": (6, -3), "kripke": (6, 2), "laghos": (6, -8),
        "lammps": (-6, 2), "minife": (6, 3), "minivite": (-6, 3), "quicksilver": (6, 3), "tricount": (6, 3)}
for k, a in enumerate(j.index):
    dx, dy = offs.get(a, (5, 4))
    ax.annotate(NAME[a], Pa[k], xytext=(dx, dy), textcoords="offset points", fontsize=8,
                ha="right" if dx < 0 else "left", zorder=6)
ax.set_xlabel(f"PC1 ({ev[0] * 100:.0f}% of variance)")
ax.set_ylabel(f"PC2 ({ev[1] * 100:.0f}% of variance)")
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, loc="best", fontsize=8)
save(fig, "coverage_pca.pdf")
plt.close(fig)

p("## C6 coverage_pca.pdf")
p(f"PCA fit on apps + all 287 inhibitors, features standardized by the pooled mean/SD: {', '.join(SEL_LABELS)}")
p(f"Explained variance: PC1 {ev[0]:.3f}, PC2 {ev[1]:.3f}, PC3 {ev[2]:.3f}, PC4 {ev[3]:.3f} (PC1+PC2 {ev[0] + ev[1]:.3f})")
p("Loadings (PC1, PC2): " + "; ".join(f"{l}: {Vt[0, i]:+.2f}, {Vt[1, i]:+.2f}" for i, l in enumerate(SEL_LABELS)))
# distances in the selection metric (scaler = apps only, RMS over 4 features), as select_inhibitors.py
ma, sa = Xa.mean(0), Xa.std(0)
D = np.sqrt((((Xa[:, None, :] - Xi[None, :, :]) / sa) ** 2).mean(axis=2))
IND3 = [0, 1, 3]
# convex hull membership via LP feasibility
def in_hull(pt, cloud):
    n = len(cloud)
    res = linprog(np.zeros(n), A_eq=np.vstack([cloud.T, np.ones(n)]), b_eq=np.concatenate([pt, [1]]),
                  bounds=[(0, None)] * n, method="highs")
    return res.status == 0
rows = []
for k, a in enumerate(j.index):
    outside = [SEL_LABELS[f] for f in range(4) if Xa[k, f] < Xi[:, f].min() or Xa[k, f] > Xi[:, f].max()]
    nn = inh_all.inhib_id[D[k].argmin()]
    npn = inh_all.inhib_id[np.where(is_panel)[0][D[k, is_panel].argmin()]]
    rows.append({"App": NAME[a], "nearest inhibitor dist": D[k].min(), "nearest inhibitor": nn,
                 "nearest panel dist": D[k, is_panel].min(), "nearest panel inhibitor": npn,
                 "inside 3-D hull of grid": in_hull(Xa[k][IND3], Xi[:, IND3]),
                 "inside 3-D hull of panel": in_hull(Xa[k][IND3], Xi[is_panel][:, IND3]),
                 "outside grid range on": "; ".join(outside) or "-"})
t = pd.DataFrame(rows).set_index("App")
p("Note: the 4 features are nearly collinear (log byte rate ~ log msg rate + log mean msg size; PC4 = 0.000), "
  "so a 4-D hull is degenerate. Hull membership is tested in the 3 independent features "
  "(comm. fraction, log msg rate, log mean msg size) by LP feasibility.")
p("Distances: RMS standardized distance with the apps-only scaler (the select_inhibitors.py metric; "
  "1.0 = one app SD per feature on average).")
p(t.to_string(float_format=lambda x: f"{x:.3f}"))
p()
p("Panel inhibitor knob settings (msg size B, wait us, sparsity):")
for pid in PANEL:
    r = inh.loc[pid]
    p(f"  {pid}: {int(r.msg_size)} B, {int(r.wait_time)} us, {r.comm_sparsity:g}")
p()

# ============================================================================ C7 scale_1node
d1 = directional_from_processed(1, "MPI")
d8 = directional_from_processed(8, "MPI")
sel = lambda d: d[(~d.self) & d.victim.isin(ONE_NODE_APPS) & d.aggressor.isin(ONE_NODE_APPS)]
fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
ecdf(ax, sel(d1).slowdown, color=C[2], ls="-", label="1 node")
ecdf(ax, sel(d8).slowdown, color=C[0], ls="--", label="8 nodes")
ax.set_xscale("log")
ax.set_xlim(0.99, 1.8)
ticks = [1, 1.1, 1.25, 1.5]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("MPI-time slowdown (clipped at 1)")
ax.set_ylabel("Cumulative fraction")
ax.set_ylim(0, 1.01)
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, loc="lower right")
save(fig, "scale_1node.pdf")
plt.close(fig)
p("## C7 scale_1node.pdf: 30 non-self directional outcomes each, same 6 apps (Beatnik, FIESTA, LAMMPS, MiniFE, miniVite, TriC)")
ks = stats.ks_2samp(sel(d1).slowdown, sel(d8).slowdown)
p(f"KS two-sample: D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")

p()

# ============================================================================ C8 socket_effect
# Per configuration (App-Inhibitor) or directional outcome (App-App) with replicates on BOTH
# sockets: mean log T_MPI on socket 1 minus mean log T_MPI on socket 0.
# Socket assignment from analysis_placement.py (mpibind rank-0 PU; standard Linux numbering assumed).
sa = pd.read_csv(OUT_DIR / "socket_app_inhibitor.csv")
ai2 = ai.rename(columns={"Inhib Message Size": "m", "Inhib Wait Time (us)": "w", "Inhib Comm Sparsity": "sp",
                         "Run Iteration": "rep"})
ai2["sp"] = ai2.sp.round(9)
ai2 = ai2.merge(sa, on=["App", "m", "w", "sp", "rep"])
ai2["ly"] = np.log(ai2["MPI Time Average"])
gi = ai2.groupby(["App", "m", "w", "sp", "app_sock"]).ly.mean().unstack().dropna()
diff_ai = (gi[1] - gi[0]).rename("d").reset_index().rename(columns={"App": "victim"})
sb = pd.read_csv(OUT_DIR / "socket_app_app.csv")
aa8 = aa[aa["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"}).merge(sb, on=["A", "B", "rep"])
long = pd.concat([pd.DataFrame({"victim": aa8.A, "other": aa8.B, "sock": aa8.sock_A, "ly": np.log(aa8["App A MPI Time Average"])}),
                  pd.DataFrame({"victim": aa8.B, "other": aa8.A, "sock": aa8.sock_B, "ly": np.log(aa8["App B MPI Time Average"])})])
gd = long.groupby(["victim", "other", "sock"]).ly.mean().unstack().dropna()
diff_aa = (gd[1] - gd[0]).rename("d").reset_index()
order8 = diff_ai.groupby("victim").d.median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(HALF_W, 3.4))
ypos = np.arange(len(order8))
bp = ax.boxplot([diff_ai[diff_ai.victim == a].d for a in order8], positions=ypos + 0.17, vert=False, widths=0.28,
                whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
                medianprops=dict(color=C[0], lw=1.4), boxprops=dict(facecolor="#dbe9f6", edgecolor=C[0], lw=0.8),
                whiskerprops=dict(color=C[0], lw=0.8), capprops=dict(color=C[0], lw=0.8))
rng = np.random.default_rng(0)
for k, a in enumerate(order8):
    v = diff_aa[diff_aa.victim == a].d.to_numpy()
    ax.scatter(v, np.full(len(v), k - 0.2) + rng.uniform(-0.06, 0.06, len(v)), s=10, marker="o", color=C[1],
               lw=0, alpha=0.85)
    if len(v):
        ax.plot([np.median(v)] * 2, [k - 0.34, k - 0.06], color=C[1], lw=1.4)
ax.axvline(0, color=OI["grey"], lw=0.8, ls=":")
ax.set_yticks(ypos)
ax.set_yticklabels([NAME[a] for a in order8])
ax.invert_yaxis()
ax.set_xlabel(r"$\Delta \log T_\mathrm{MPI}$ (socket 1 $-$ socket 0)")
ax.grid(axis="x", lw=0.4, alpha=0.4)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#dbe9f6", edgecolor=C[0], label="App-Inhibitor configs"),
                   Line2D([], [], color=C[1], marker="o", ls="", ms=3.5, label="App-App outcomes")],
          frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3,
          columnspacing=0.8)
save(fig, "socket_effect.pdf")
plt.close(fig)
p("## C8 socket_effect.pdf")
p("Per config (App-Inhibitor) / directional outcome (App-App) having replicates on both sockets: "
  "mean log T_MPI(socket 1) - mean log T_MPI(socket 0). Box = IQR, whiskers = 5-95%, line = median; "
  "orange dots = App-App directional outcomes, orange tick = their median. Apps ordered by App-Inhibitor median.")
t = pd.DataFrame({"App-Inh n": diff_ai.groupby("victim").d.size(), "App-Inh median": diff_ai.groupby("victim").d.median(),
                  "App-Inh mean": diff_ai.groupby("victim").d.mean(),
                  "App-Inh frac<0": diff_ai.groupby("victim").d.apply(lambda x: (x < 0).mean()),
                  "App-App n": diff_aa.groupby("victim").d.size(), "App-App median": diff_aa.groupby("victim").d.median()}).loc[order8]
t.index = [NAME[a] for a in t.index]
p(t.round(3).to_string())
w = stats.wilcoxon(diff_ai.d)
p(f"All App-Inhibitor configs: n = {len(diff_ai)}, median {diff_ai.d.median():.3f}, mean {diff_ai.d.mean():.3f}, "
  f"frac<0 {(diff_ai.d < 0).mean():.3f}, Wilcoxon p = {w.pvalue:.2g}")
w2 = stats.wilcoxon(diff_aa.d)
p(f"All App-App directions: n = {len(diff_aa)}, median {diff_aa.d.median():.3f}, mean {diff_aa.d.mean():.3f}, "
  f"frac<0 {(diff_aa.d < 0).mean():.3f}, Wilcoxon p = {w2.pvalue:.2g}")

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "figs_batch1_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "figs_batch1_numbers.md")
