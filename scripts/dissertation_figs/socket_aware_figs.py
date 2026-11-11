"""Socket batch 2 figures C1 (forest, blind vs socket-aware) and C2 (per-victim aware/blind ratio).

Reads output/socket_aware_scores_B1.csv and output/socket_aware_pervictim_B3.csv (written by socket_aware_score.py).
C1  socket_aware_forest.pdf     6.0 in; rows one-way / bidirectional, columns MPI / total time; socket-resolved App-App
                                outcomes (166 cells); x = log-MAE change vs the isolated-speed baseline (%), 95% pair-cluster CI.
C2  socket_aware_pervictim.pdf  3.0 in; per-victim aware/blind log-MAE ratio on socket-resolved cells, log x centred on 1.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import APPS, NAME, OI, FULL_W, HALF_W, OUT_DIR, save, set_style

set_style()
b1 = pd.read_csv(OUT_DIR / "socket_aware_scores_B1.csv")
b1 = b1[b1.subset == "all"].rename(columns={"Δ%": "dpct", "CI lo": "lo", "CI hi": "hi"})

ROWS = {  # role -> label; ablations (wall-clock aux, forward-only) are left to the table
    "absolute": "Absolute response",
    "offset_jms": "Absolute + socket offset",
    "delta_uniform": "Delta, uniform agg.",
    "delta_median": "Delta, median agg.",
    "generic": "Generic potential",
    "joint_cal": "Joint, calibrated",
    "bprof_cal": "Bidir. profile, calibrated",
    "bctx_cal": "Bidir. context, calibrated",
}
PANELS = [("1W-MPI", 0, 0), ("1W-TT", 0, 1), ("BD-MPI", 1, 0), ("BD-TT", 1, 1)]
STYLE = {"blind": dict(marker="o", mfc="white", mec=OI["blue"], color=OI["blue"]),
         "aware": dict(marker="s", mfc=OI["vermillion"], mec=OI["vermillion"], color=OI["vermillion"]),
         "offset": dict(marker="D", mfc="white", mec=OI["green"], color=OI["green"])}

fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.2), sharey="row", sharex="col",
                         gridspec_kw=dict(height_ratios=[5, 8], hspace=0.28, wspace=0.08))
for pipe, r, c in PANELS:
    ax = axes[r, c]
    ax.tick_params(labelbottom=(r == 1))
    d = b1[(b1.pipeline == pipe) & b1.role.isin(ROWS)]
    roles = [k for k in ROWS if k in set(d.role)] or [k for k in ROWS if (r == 1 or not k.endswith("_cal"))]
    y = {k: i for i, k in enumerate(reversed(roles))}
    ax.axvline(0, color=OI["grey"], lw=0.8, ls="--", zorder=0)
    if d.empty:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", color=OI["grey"])
    for row in d.itertuples():
        key = "offset" if row.role.startswith("offset") else row.variant
        off = 0.0 if key == "offset" else (0.17 if row.variant == "blind" else -0.17)
        st = STYLE[key]
        if row.role == "constant_1":
            continue
        xe = np.array([[row.dpct - row.lo], [row.hi - row.dpct]])
        ax.errorbar(row.dpct, y[row.role] + off, xerr=xe, fmt=st["marker"], ms=4.2, mfc=st["mfc"], mec=st["mec"],
                    ecolor=st["color"], elinewidth=0.9, capsize=1.8, mew=0.9)
    ax.set_yticks(list(y.values()), [ROWS[k] for k in y])
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    if r == 0:
        ax.text(0.5, 1.04, "MPI time" if c == 0 else "Total time", transform=ax.transAxes, ha="center", va="bottom")
    if c == 1:
        ax.text(1.03, 0.5, "One-way" if r == 0 else "Bidirectional", transform=ax.transAxes, rotation=270,
                ha="left", va="center")
handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=OI["blue"], label="Socket-blind"),
           Line2D([], [], ls="", marker="s", mfc=OI["vermillion"], mec=OI["vermillion"], label="Socket-aware"),
           Line2D([], [], ls="", marker="D", mfc="white", mec=OI["green"], label="Blind + fixed socket offset")]
fig.supxlabel(r"$\Delta$ log MAE vs isolated-speed baseline (%)", x=0.6, y=0.0, fontsize=10)
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.075), frameon=False)
save(fig, "socket_aware_forest.pdf")
fig.savefig(OUT_DIR / "socket_aware_forest.png", dpi=110)

# ------------------------------------------------------------------ C2
b3 = pd.read_csv(OUT_DIR / "socket_aware_pervictim_B3.csv")
# the absolute model is shared code (same data, seeds, epochs) in both pipelines, so one series covers both
SERIES = [("BD-MPI", "absolute", "Absolute response", dict(marker="o", mfc="white", mec=OI["blue"])),
          ("BD-MPI", "joint_cal", "Joint, calibrated (bidir.)", dict(marker="s", mfc=OI["vermillion"], mec=OI["vermillion"]))]
victims = [NAME[a] for a in APPS]
fig, ax = plt.subplots(figsize=(HALF_W, 3.0))
ax.axvline(1, color=OI["grey"], lw=0.8, ls="--", zorder=0)
present = [s for s in SERIES if ((b3.pipeline == s[0]) & (b3.role == s[1])).any()]
for i, (pipe, role, label, st) in enumerate(present):
    d = b3[(b3.pipeline == pipe) & (b3.role == role)].set_index("victim").reindex(victims)
    off = (i - (len(present) - 1) / 2) * 0.22
    ax.plot(d.ratio, np.arange(len(victims))[::-1] + off, ls="", ms=4.2, mew=0.9, label=label, **st)
ax.set_xscale("log")
ax.set_xlim(0.4, 2.5)
ax.set_xticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
ax.minorticks_off()
ax.set_yticks(np.arange(len(victims))[::-1], victims)
ax.set_xlabel("Aware / blind log MAE (MPI time)")
ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, frameon=False)
save(fig, "socket_aware_pervictim.pdf")
fig.savefig(OUT_DIR / "socket_aware_pervictim.png", dpi=110)
