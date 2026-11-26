"""Single-panel noise_cv.pdf (fig:ns_noise): replicate CV of isolated T_MPI and T_app per application.

Request N1 (2026-11-25): panel (b) (ECDF of T_MPI CV for isolated inhibitors, App-Inh, App-App) was dropped
by Akhil. This reproduces panel (a) of figs_batch1.py C1 unchanged in content; figs_batch1.py now writes the
old two-panel version to noise_cv_2panel.pdf.

Writes Dissertation/figures/node_sharing/noise_cv.pdf and output/noise_cv_numbers.md.
Run: <analysis python> scripts/dissertation_figs/v2/noise_cv_fig.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APPS, FULL_W, NAME, OI, OUT_DIR, PROCESSED, read, save, set_style  # noqa: E402

set_style()
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def cv(x):
    return np.std(x, ddof=1) / np.mean(x)  # same definition as figs_batch1.py:47 (eq:ns_cv)


iso = read(PROCESSED / "data_isolated.csv")
i8 = iso[iso["Num Nodes"] == 8]
g = i8.groupby("App")
n_rep = g.size().loc[APPS]
cv_mpi = g["MPI Time Average"].apply(cv).loc[APPS]
cv_app = g["App Time Average"].apply(cv).loc[APPS]
order = cv_mpi.sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
xpos = np.arange(len(order))
w = 0.38
ax.bar(xpos - w / 2, cv_mpi[order] * 100, w, color=OI["blue"], label=r"$T_\mathrm{MPI}$",
       edgecolor="white", linewidth=0.5)
ax.bar(xpos + w / 2, cv_app[order] * 100, w, color=OI["vermillion"], label=r"$T_\mathrm{app}$",
       edgecolor="white", linewidth=0.5, hatch="////")
ax.set_yscale("log")
ax.set_ylim(0.1, 50)
ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel("Replicate CV (%)")
ax.set_xticks(xpos)
ax.set_xticklabels([NAME[a] for a in order], rotation=30, ha="right")
ax.set_xlim(-0.6, len(order) - 0.4)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", lw=0.4, alpha=0.4)
fig.tight_layout()
save(fig, "noise_cv.pdf")
plt.close(fig)

p("# noise_cv.pdf (single panel, request N1)\n")
p("Replicate CV = sample SD (ddof=1) / mean over the 8-node isolated replicates (processed_data/data_isolated.csv).\n")
p("| App | n reps | CV T_MPI (%) | CV T_app (%) |")
p("|---|---:|---:|---:|")
for a in order:
    p(f"| {NAME[a]} | {n_rep[a]} | {cv_mpi[a] * 100:.2f} | {cv_app[a] * 100:.2f} |")
p()
p(f"Median CV T_MPI {np.median(cv_mpi) * 100:.2f}%; median CV T_app {np.median(cv_app) * 100:.2f}%.")
p("Apps with CV T_app > 2%: " + ", ".join(NAME[a] for a in order if cv_app[a] > 0.02))
(OUT_DIR / "noise_cv_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "noise_cv_numbers.md")
