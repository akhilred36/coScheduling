"""c_heatmap_mpi.pdf and c_heatmap_tt.pdf (request N3): the two panels of c_heatmaps_unclipped.pdf as separate
full-width figures with larger cell annotations.

Same content and encoding as v2_char_figs.py section 2 (c_heatmaps_unclipped.pdf, left unchanged): rows = victim,
columns = co-runner, APPS order, rep-averaged JMs ratio from output/v2_appapp_directed.csv, RdBu_r on log ratio,
symmetric limits = max |log ratio|, diagonal = self-pairs (bold), 2 decimals for MPI and 3 for app time.

Writes Dissertation/figures/node_sharing/c_heatmap_{mpi,tt}.pdf and output/c_heatmap_split_numbers.md.
Run: <analysis python> scripts/dissertation_figs/v2/c_heatmap_split_fig.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APPS, FULL_W, NAME, OUT_DIR, save, set_style  # noqa: E402

set_style()
log = io.StringIO()
DISP = [NAME[a] for a in APPS]
TIMERS = [("mpi", "MPI-time"), ("tt", "Total-time")]  # chapter terminology (N3 follow-up)
SIZE = (FULL_W, 5.0)
ANNOT_PT = {"mpi": 9.0, "tt": 7.8}  # 3-decimal values need the smaller size to clear neighbours
TICK_PT = 9.5


def p(*a):
    print(*a)
    print(*a, file=log)


dirm = pd.read_csv(OUT_DIR / "v2_appapp_directed.csv")
p("# c_heatmap_mpi.pdf / c_heatmap_tt.pdf (request N3)\n")
p(f"Size {SIZE[0]:.1f} x {SIZE[1]:.1f} in each; cell annotations MPI {ANNOT_PT['mpi']} pt / app time {ANNOT_PT['tt']} pt, tick labels {TICK_PT} pt.\n")
for tk, lab in TIMERS:
    M = dirm.pivot(index="victim", columns="corunner", values=f"raw_{tk}").loc[APPS, APPS]
    L = np.log(M.to_numpy())
    lim = np.abs(L).max()
    fig, ax = plt.subplots(figsize=SIZE)
    im = ax.imshow(L, cmap="RdBu_r", norm=Normalize(-lim, lim), aspect="equal")
    for i in range(10):
        for j in range(10):
            v = M.to_numpy()[i, j]
            dark = abs(L[i, j]) > 0.55 * lim
            ax.text(j, i, f"{v:.3f}" if tk == "tt" else f"{v:.2f}", ha="center", va="center", fontsize=ANNOT_PT[tk],
                    color="white" if dark else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10), DISP, rotation=45, ha="right", rotation_mode="anchor", fontsize=TICK_PT)
    ax.set_yticks(range(10), DISP, fontsize=TICK_PT)
    ax.tick_params(length=0, pad=3)
    ax.set_xlabel("Co-runner")
    ax.set_ylabel("Victim")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks = np.array([1 / np.exp(lim), 1 / np.exp(lim / 2), 1, np.exp(lim / 2), np.exp(lim)])
    cb.set_ticks(np.log(ticks))
    cb.set_ticklabels([f"{t:.3f}" if tk == "tt" else f"{t:.2f}" for t in ticks])
    cb.ax.tick_params(labelsize=TICK_PT)
    cb.set_label(f"{lab} slowdown (log scale)")
    cb.outline.set_linewidth(0.5)
    fig.tight_layout()
    save(fig, f"c_heatmap_{tk}.pdf")
    plt.close(fig)

    p(f"## c_heatmap_{tk}.pdf ({lab})\n")
    p(f"Colour limits [{np.exp(-lim):.4f}, {np.exp(lim):.4f}]; max {M.to_numpy().max():.4f}, min {M.to_numpy().min():.4f}.\n")
    p("| victim \\ co-runner | " + " | ".join(DISP) + " |")
    p("|---|" + "---:|" * 10)
    for a in APPS:
        p(f"| {NAME[a]} | " + " | ".join(f"{M.loc[a, b]:.4f}" for b in APPS) + " |")
    p()
(OUT_DIR / "c_heatmap_split_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "c_heatmap_split_numbers.md")
