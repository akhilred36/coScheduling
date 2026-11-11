"""probe_epochs.pdf (half width): training-budget probe, BD-MPI blind.

Top: fold-averaged App-Inhibitor crossed-CV validation log MAE vs epochs (the selection data), joint (calibrated
inference) and absolute; marked = protocol-selected epochs (median of per-fold best), x = original near-init selection.
Bottom: App-App Δ% vs the isolated-speed baseline (DESCRIPTIVE only; 30-seed ensembles at fixed epochs; the near-init
recipe point is the part (a) 100-seed ensemble; marked = final fits at the CV-protocol budget).
Writes output/figs_probe_numbers.md. Reads output/retune_probe_{cv_curves,appapp_curve}.csv and the probe run dirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import HALF_W, OI, OUT_DIR, save, set_style

ROOT = Path.home() / "hpcResearch/experiment_outputs/retune_probe"
SB2 = Path.home() / "hpcResearch/experiment_outputs/socket_batch2"
set_style()
cv = pd.read_csv(OUT_DIR / "retune_probe_cv_curves.csv", index_col="epoch")
aa = pd.read_csv(OUT_DIR / "retune_probe_appapp_curve.csv")
sel = json.loads((ROOT / "selected_epochs.json").read_text())
orig = json.loads((SB2 / "blind_bd_mpi/selection.json").read_text())["variants"]

V = [  # variant, App-App column, pipeline method code, label, colour, linestyle, marker
    ("joint_profile", "joint_cal Δ%", "joint_profile_calibrated", "Joint, calibrated", OI["blue"], "-", "o"),
    ("legacy_absolute", "absolute Δ%", "legacy_absolute", "Absolute response", OI["vermillion"], "--", "s"),
]
fixed = aa[aa.run.str.startswith("all variants")].copy()
fixed["epochs"] = fixed.run.str.extract(r"at (\d+) epochs")[0].astype(int)
m = pd.read_csv(ROOT / "runs/C_cv_blind/metrics_non_self.csv")
m = m[m.evaluation_method == "random_split"].set_index("method")
base = m.loc["constant_1", "log_mae"]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(HALF_W, 3.9), sharex=True)
num = ["# probe_epochs.pdf numbers", "", "| variant | original epochs | protocol epochs | CV argmin | CV @orig | CV @protocol | "
       "CV @240 | App-App Δ% near-init recipe | App-App Δ% at protocol (30 seeds) |", "|---|---|---|---|---|---|---|---|---|"]
for v, col, code, lab, c, ls, mk in V:
    e0, ep = int(orig[v]["final_epochs"]), int(sel[v]["protocol"])
    y = cv[v]
    a1.plot(y.index, y.values, color=c, ls=ls, lw=1.2, label=lab)
    a1.plot([ep], [y.loc[ep]], mk, color=c, ms=5, mfc="white", mew=1.2, zorder=3)
    a1.plot([e0], [y.loc[e0]], "x", color=c, ms=5, mew=1.2, zorder=3)
    rec = float(aa.loc[aa.run.str.startswith("current recipe"), col].iloc[0])
    xs = [e0] + fixed.epochs.tolist()
    ys = [rec] + fixed[col].tolist()
    a2.plot(xs, ys, color=c, ls=ls, lw=1.2, marker=mk, ms=3.5, label=lab)
    a2.plot([e0], [rec], "x", color=c, ms=6, mew=1.2, zorder=3)
    dcv = 100 * (m.loc[code, "log_mae"] / base - 1)
    a2.plot([ep], [dcv], mk, color=c, ms=6, mfc="white", mew=1.2, zorder=3)
    num.append(f"| {lab} | {e0} | {ep} | {sel[v]['fold_avg_argmin']} | {y.loc[e0]:.4f} | {y.loc[ep]:.4f} | "
               f"{y.loc[240]:.4f} | {rec:+.1f} | {dcv:+.1f} |")
a1.set_ylabel("CV log MAE\n(App-Inhibitor)")
a2.axhline(0, color=OI["grey"], lw=0.8, ls=":")
a2.set_ylabel("App-App $\\Delta$% vs\nisolated-speed baseline")
a2.set_xlabel("training epochs")
a2.set_xscale("log")
a2.set_xticks([1, 2, 7, 15, 30, 60, 240])
a2.set_xticklabels(["1", "2", "7", "15", "30", "60", "240"])
a2.minorticks_off()
a2.set_xlim(0.8, 300)
from matplotlib.lines import Line2D
a1.add_artist(a1.legend(frameon=False, fontsize=8, loc="upper left"))
a2.legend(handles=[Line2D([], [], color="k", marker="x", ls="none", ms=5, label="original"),
                   Line2D([], [], color="k", marker="o", mfc="white", ls="none", ms=5, label="CV selection")],
          frameon=False, fontsize=7, loc="upper left")
fig.align_ylabels()
fig.tight_layout(h_pad=0.4)
save(fig, "probe_epochs.pdf")
(OUT_DIR / "figs_probe_numbers.md").write_text("\n".join(num) + "\n")
print("\n".join(num))
