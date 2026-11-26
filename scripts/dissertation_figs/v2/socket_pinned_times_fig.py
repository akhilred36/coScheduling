"""socket_pinned_times.pdf (request N2): isolated times on socket 0 vs socket 1 per app, socket-pinned runs.

Source: output/socket_pinned_runs.csv (written by analysis_socket_pinned.py from experiment
run_data/experiments_coscheduling_isolated_socket/20261107_180934, all 80 valid app steps). This is the same
source as socket_numbers.md S1; the June baselines are not used.

Two panels, (a) T_MPI and (b) T_app. Bars = mean over the 4 replicates, whiskers = replicate min-max.
App order = noise_cv.pdf order (isolated June CV of T_MPI, descending), so the two figures in the section line up.
App labels at 45 deg: 30 deg collides with 10 apps in a half-width panel.

Outputs (Dissertation/figures/node_sharing/):
  socket_pinned_times.pdf       absolute seconds; both panels linear y from 0
  socket_pinned_times_norm.pdf  alternative: both panels divided by the socket-0 mean (linear y from 0)
and output/socket_pinned_times_numbers.md.
Run: <analysis python> scripts/dissertation_figs/v2/socket_pinned_times_fig.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APPS, FULL_W, NAME, OI, OUT_DIR, PROCESSED, read, save, set_style  # noqa: E402

set_style()
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def label_panel(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


runs = pd.read_csv(OUT_DIR / "socket_pinned_runs.csv")
assert len(runs) == 80 and runs.groupby(["App", "Socket"]).size().eq(4).all(), "expected 10 apps x 2 sockets x 4"
COLS = {"mpi": "MPI Time Average", "app": "App Time Average"}
st = runs.groupby(["App", "Socket"])[list(COLS.values())].agg(["mean", "min", "max"])

# noise_cv order: June 8-node isolated CV of T_MPI, descending (same as v2/noise_cv_fig.py)
iso = read(PROCESSED / "data_isolated.csv")
i8 = iso[iso["Num Nodes"] == 8]
cv_mpi = i8.groupby("App")["MPI Time Average"].agg(lambda x: np.std(x, ddof=1) / np.mean(x)).loc[APPS]
order = list(cv_mpi.sort_values(ascending=False).index)

# per-pair log differences (S1 cross-check)
pairs = pd.read_csv(OUT_DIR / "socket_pinned_pairs.csv")
dm = pairs.groupby("App")[["dMPI", "dApp"]].mean()

SOCK = [(0, OI["blue"], None, "Socket 0"), (1, OI["orange"], "////", "Socket 1")]
assert OI["blue"].lower() == "#0072b2" and OI["orange"].lower() == "#e69f00"


def draw(norm: bool, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    xpos = np.arange(len(order))
    w = 0.38
    for ax, (key, col), lab in zip(axes, COLS.items(), ["(a)", "(b)"]):
        ref = st.loc[(order, 0), (col, "mean")].to_numpy() if norm else 1.0
        for k, (s, c, h, name) in enumerate(SOCK):
            m = st.loc[(order, s), (col, "mean")].to_numpy() / ref
            lo = st.loc[(order, s), (col, "min")].to_numpy() / ref
            hi = st.loc[(order, s), (col, "max")].to_numpy() / ref
            x = xpos + (k - 0.5) * w
            ax.bar(x, m, w, color=c, hatch=h, edgecolor="white", linewidth=0.5, label=name)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5,
                        capthick=0.6)
        sym = r"T_\mathrm{MPI}" if key == "mpi" else r"T_\mathrm{app}"
        if norm:
            ax.set_ylim(0, 1.75)
            ax.axhline(1, color=OI["grey"], lw=0.6, ls=":")
            ax.set_ylabel(rf"Relative ${sym}$")  # = T / socket-0 mean
        elif key == "mpi":
            # linear from 0 (N2 revision, Akhil): tallest whisker is TriC sock0 max 241.96 s
            ax.set_ylim(0, 260)
            ax.set_yticks([0, 50, 100, 150, 200, 250])
            ax.set_ylabel(rf"${sym}$ (s)")
        else:
            ax.set_ylim(0, 520)
            ax.set_ylabel(rf"${sym}$ (s)")
        ax.set_xticks(xpos)
        ax.set_xticklabels([NAME[a] for a in order], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        label_panel(ax, lab, x=-0.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.92))
    save(fig, fname)
    plt.close(fig)


draw(False, "socket_pinned_times.pdf")
draw(True, "socket_pinned_times_norm.pdf")

p("# socket_pinned_times.pdf (request N2)\n")
p("Source: output/socket_pinned_runs.csv (experiment 20261107_180934, 80 valid app steps, 4 reps per app x socket).")
p("Bars = mean of 4 replicates; whiskers = replicate min-max. App order = noise_cv.pdf order.\n")
p("ratio = mean(sock1)/mean(sock0) (ratio of means). exp(d) = exp(mean paired log(sock1/sock0)), the S1 statistic.\n")
for key, col in COLS.items():
    p(f"## {'T_MPI' if key == 'mpi' else 'T_app'} (s)\n")
    p("| App | sock0 mean | sock0 min-max | sock1 mean | sock1 min-max | ratio of means | exp(mean d) (S1) |")
    p("|---|---:|---|---:|---|---:|---:|")
    dcol = "dMPI" if key == "mpi" else "dApp"
    for a in order:
        r0, r1 = st.loc[(a, 0), col], st.loc[(a, 1), col]
        p(f"| {NAME[a]} | {r0['mean']:.2f} | {r0['min']:.2f}-{r0['max']:.2f} | {r1['mean']:.2f} | "
          f"{r1['min']:.2f}-{r1['max']:.2f} | {r1['mean'] / r0['mean']:.4f} | {np.exp(dm.loc[a, dcol]):.4f} |")
    p(f"\nPooled mean d over apps: {dm[dcol].mean():+.4f}\n")
(OUT_DIR / "socket_pinned_times_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "socket_pinned_times_numbers.md")
