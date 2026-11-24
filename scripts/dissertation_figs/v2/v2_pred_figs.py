"""Plan v2 prediction-result figures (R): reads the CSVs written by v2_score.py for one tag.

  python scripts/dissertation_figs/v2/v2_pred_figs.py [--tag v2_full_20261123]

Writes r_skill.pdf, r_paired_diffs.pdf, r_pervictim.pdf to Dissertation/figures/node_sharing/ and the plotted values
to output/v2_pred_figs_numbers.md. xgb = follow-up rerun (max_depth 1-8); the xgb_d124 sensitivity row is not plotted.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APPS, FULL_W, NAME, OI, OUT_DIR, save, set_style  # noqa: E402

set_style()
log = io.StringIO()
LEARN = ["ridge", "lasso", "svr", "rf", "xgb", "mlp"]
RESP = ["absolute", "delta"]
METHODS = LEARN + RESP
MLAB = {"ridge": "Ridge", "lasso": "Lasso", "svr": "SVR", "rf": "RF", "xgb": "XGBoost", "mlp": "MLP",
        "absolute": "Absolute", "delta": "Delta"}
TIERS = ["full", "one_known", "zero_shot"]
TLAB = {"full": "Full", "one_known": "One-known", "zero_shot": "Zero-shot"}
TGT = {"mpi": "MPI time", "tt": "Application time"}
OUT_OF_HULL = {"quicksilver"}
XMIN = -1.0  # skill axis clip (r_skill); clipped seeds drawn as '<' at the edge


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def ypos():
    """Methods top to bottom with a gap between the App-App learners and the response models."""
    y, pos = 0.0, {}
    for m in METHODS:
        if m == RESP[0]:
            y += 0.6
        pos[m] = y
        y += 1.0
    return pos, y - 1.0


def form_of(m, form):
    return "none" if m in RESP else form


# --------------------------------------------------------------------------------------------------------- r_skill
def fig_skill(M, S):
    pos, ymax = ypos()
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 5.4), sharey=True)
    styles = {"1w": dict(marker="o", mfc=OI["blue"], mec=OI["blue"], c=OI["blue"], dy=-0.17),
              "bd": dict(marker="D", mfc="white", mec=OI["orange"], c=OI["orange"], dy=0.17),
              "none": dict(marker="s", mfc=OI["vermillion"], mec=OI["vermillion"], c=OI["vermillion"], dy=0.0)}
    rows = []
    for i, tgt in enumerate(("mpi", "tt")):
        for j, tier in enumerate(TIERS):
            ax = axes[i, j]
            ax.axvline(0, color="0.3", lw=0.7, zorder=0)
            ax.axhline(pos[RESP[0]] - 0.8, color="0.6", lw=0.6, ls="--")
            for m in METHODS:
                for form in (("none",) if m in RESP else ("1w", "bd")):
                    st = styles[form]
                    sel = dict(test_tier=tier, target=tgt, socket="aware", method=m, form=form)
                    seeds = M.loc[np.logical_and.reduce([M[k] == v for k, v in sel.items()]), "skill"].to_numpy()
                    s = S.loc[np.logical_and.reduce([S[k] == v for k, v in sel.items()])].iloc[0]
                    y = pos[m] + st["dy"]
                    clip = seeds < XMIN
                    ax.scatter(seeds[~clip], np.full((~clip).sum(), y), s=5, color=st["c"], alpha=0.25, lw=0,
                               zorder=1)
                    if clip.any():
                        ax.scatter([XMIN + 0.02] * clip.sum(), [y] * clip.sum(), marker="<", s=9, color=st["c"],
                                   alpha=0.4, lw=0, zorder=1)
                    mean, sd = s.skill_mean, s.skill_sd
                    lo, hi = max(mean - sd, XMIN), mean + sd
                    ax.plot([lo, hi], [y, y], color=st["c"], lw=1.0, zorder=2)
                    mx = max(mean, XMIN + 0.02)
                    ax.plot([mx], [y], marker=st["marker"] if mean >= XMIN else "<", mfc=st["mfc"], mec=st["mec"],
                            ms=4.2, mew=0.9, ls="none", zorder=3)
                    sig = s.seeds_ci_below0 >= 9
                    if sig:
                        ax.text(hi + 0.03, y, r"$\dagger$", va="center", ha="left", fontsize=8)
                    rows.append(dict(target=tgt, tier=tier, method=m, form=form, skill_mean=mean, skill_sd=sd,
                                     seed_min=seeds.min(), seed_max=seeds.max(), n_clipped=int(clip.sum()),
                                     seeds_better=int(s.seeds_better_than_base), ci_below0=int(s.seeds_ci_below0),
                                     dagger=sig))
            ax.set_xlim(XMIN, 0.72)
            ax.set_xticks([-1, -0.5, 0, 0.5])
            ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
            if i == 0:
                ax.set_title(TLAB[tier], fontsize=10)  # column header (panel label, not a figure title)
            if j == 0:
                ax.set_ylabel(TGT[tgt])
    axes[0, 0].set_yticks([pos[m] for m in METHODS], [MLAB[m] for m in METHODS])
    axes[0, 0].set_ylim(ymax + 0.6, -0.6)
    handles = [Line2D([], [], marker="o", color=OI["blue"], mfc=OI["blue"], ms=4.5, lw=1, label="1W"),
               Line2D([], [], marker="D", color=OI["orange"], mfc="white", ms=4.5, lw=1, label="BD"),
               Line2D([], [], marker="s", color=OI["vermillion"], mfc=OI["vermillion"], ms=4.5, lw=1,
                      label="Response model (no BD form)"),
               Line2D([], [], marker="o", color="0.4", alpha=0.35, ms=3, lw=0, label="Per seed"),
               Line2D([], [], lw=0, label=r"$\dagger$ CI $<0$ in $\geq$9/10 seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.0)
    fig.supxlabel("Skill vs isolated baseline", fontsize=10, y=0.045)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.4)
    save(fig, "r_skill.pdf")
    plt.close(fig)
    R = pd.DataFrame(rows)
    p("## r_skill.pdf (6.0 x 5.4 in)\nSocket-aware; skill = 1 - MAE/MAE_base, mean ± SD over 10 seeds; † = pair-cluster "
      f"CI of method - baseline below 0 in >= 9/10 seeds. x clipped at {XMIN} (n_clipped seeds drawn as '<').\n")
    p(md(R) + "\n")


# --------------------------------------------------------------------------------------------------- r_paired_diffs
def fig_diffs(C, CS):
    pos, ymax = ypos()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.0), sharey=True)
    tst = {"full": dict(c=OI["blue"], marker="o", dy=-0.22), "one_known": dict(c=OI["green"], marker="s", dy=0.0),
           "zero_shot": dict(c=OI["purple"], marker="^", dy=0.22)}
    panels = [("aware-blind", "mpi"), ("aware-blind", "tt"), ("bd-1w", "mpi"), ("bd-1w", "tt")]
    rows = []
    lim = {}
    for con, tgt in panels:  # x range from the App-App learners' mean ± SD; response rows are clipped to it
        q = CS[(CS.contrast == con) & (CS.target == tgt) & CS.method.isin(LEARN)]
        ext = float(np.max(np.abs(np.r_[q.diff_mean - q.diff_sd, q.diff_mean + q.diff_sd]))) * 1e3 * 1.15
        lim[(con, tgt)] = ext
    for ax, (con, tgt) in zip(axes.flat, panels):
        L = lim[(con, tgt)]
        ax.axvline(0, color="0.3", lw=0.7, zorder=0)
        ax.axhline(pos[RESP[0]] - 0.8, color="0.6", lw=0.6, ls="--")
        ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
        for m in METHODS:
            if con == "bd-1w" and m in RESP:
                ax.text(0, pos[m], "no BD form", ha="center", va="center", fontsize=7, color="0.4")
                continue
            for tier in TIERS:
                st = tst[tier]
                sel = dict(test_tier=tier, target=tgt, method=m, contrast=con)
                if con == "aware-blind":
                    sel["form"] = form_of(m, "1w")
                else:
                    sel["socket"] = "aware"
                c = C.loc[np.logical_and.reduce([C[k] == v for k, v in sel.items()])]
                s = CS.loc[np.logical_and.reduce([CS[k] == v for k, v in sel.items()])].iloc[0]
                d = c["diff"].to_numpy() * 1e3
                y = pos[m] + st["dy"]
                dd = d[np.abs(d) <= L]
                ax.scatter(dd, np.full(len(dd), y), s=4, color=st["c"], alpha=0.25, lw=0, zorder=1)
                mean, sd = s.diff_mean * 1e3, s.diff_sd * 1e3
                sig = (s.seeds_ci_below0 >= 9) or (s.seeds_ci_above0 >= 9)
                ax.plot([max(mean - sd, -L), min(mean + sd, L)], [y, y], color=st["c"], lw=1.0, zorder=2)
                if abs(mean) <= L:
                    ax.plot([mean], [y], marker=st["marker"], color=st["c"], mfc=st["c"] if sig else "white", ms=4,
                            mew=0.9, ls="none", zorder=3)
                else:
                    side = np.sign(mean)
                    ax.plot([side * L * 0.97], [y], marker=">" if side > 0 else "<", color=st["c"],
                            mfc=st["c"] if sig else "white", ms=4, mew=0.9, ls="none", zorder=3)
                    ax.text(side * L * 0.88, y, f"{mean:.0f}".replace("-", "\u2212"), ha="right" if side > 0 else
                            "left", va="center", fontsize=6, color=st["c"])
                rows.append(dict(contrast=con, target=tgt, tier=tier, method=m, diff_mean_x1e3=mean, diff_sd_x1e3=sd,
                                 seeds_neg=int(s.seeds_diff_neg), ci_below0=int(s.seeds_ci_below0),
                                 ci_above0=int(s.seeds_ci_above0), filled=sig))
        ax.set_xlim(-L, L)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
        if con == "aware-blind":
            ax.set_title(TGT[tgt], fontsize=10)
        if tgt == "mpi":
            ax.set_ylabel("Aware $-$ blind" if con == "aware-blind" else "BD $-$ 1W")
    axes[0, 0].set_yticks([pos[m] for m in METHODS], [MLAB[m] for m in METHODS])
    axes[0, 0].set_ylim(ymax + 0.6, -0.6)
    handles = [Line2D([], [], marker=tst[t]["marker"], color=tst[t]["c"], mfc=tst[t]["c"], ms=4.5, lw=1,
                      label=TLAB[t]) for t in TIERS]
    handles += [Line2D([], [], marker="o", color="0.3", mfc="0.3", ms=4.5, lw=0, label=r"CI excludes 0 in $\geq$9/10"),
                Line2D([], [], marker="o", color="0.3", mfc="white", ms=4.5, lw=0, label="otherwise")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.0)
    fig.supxlabel(r"Paired difference in log MAE ($\times 10^{-3}$; negative = first term better)", fontsize=9,
                  y=0.045)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8, w_pad=0.6)
    save(fig, "r_paired_diffs.pdf")
    plt.close(fig)
    R = pd.DataFrame(rows)
    p("## r_paired_diffs.pdf (6.0 x 5.0 in)\nMean ± SD over 10 seeds of the paired log-MAE difference (×1e-3; negative "
      "= first term better). Aware - blind uses the 1W form (response models: their only form); BD - 1W uses "
      "socket-aware. Filled = pair-cluster CI excludes 0 (either side) in >= 9/10 seeds. x range per panel = "
      "1.15 x the largest |mean ± SD| of the App-App learners; response-model means outside it are drawn as arrows "
      "at the edge with the value printed.\n")
    p(md(R, fmt="{:.3f}") + "\n")


# ----------------------------------------------------------------------------------------------------- r_pervictim
def fig_pervictim(D):
    d = D[(D.target == "mpi") & (D.socket == "aware") & D.method.isin(METHODS)
          & (((D.form == "1w") & D.method.isin(LEARN)) | ((D.form == "none") & D.method.isin(RESP)))].copy()
    d["skill"] = 1 - d.log_mae / d.base_mae
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 6.6), sharex=True)
    cmap = plt.get_cmap("RdBu").copy()
    tables = []
    for j, tier in enumerate(("one_known", "zero_shot")):
        for i, by in enumerate(("victim", "corunner")):
            ax = axes[i, j]
            q = d[(d.test_tier == tier) & (d.by == by)]
            T = q.pivot_table(index="level", columns="method", values="skill").reindex(index=APPS, columns=METHODS)
            base = q[q.method == "rf"].set_index("level").base_mae.reindex(APPS)
            ax.imshow(T.clip(lower=-1, upper=1).to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            for r in range(T.shape[0]):
                for c in range(T.shape[1]):
                    v = T.iat[r, c]
                    txt = f"{v:.2f}" if v > -10 else f"{v:.0f}"
                    ax.text(c, r, txt.replace("-", "−"), ha="center", va="center", fontsize=6,
                            color="white" if abs(max(v, -1)) > 0.6 else "black")
            ylabs = [NAME[a] + ("$^{*}$" if a in OUT_OF_HULL else "") for a in APPS]
            ax.set_yticks(range(len(APPS)), ylabs, fontsize=7.5)
            ax.set_xticks(range(len(METHODS)), [MLAB[m] for m in METHODS], rotation=45, ha="right", fontsize=7.5)
            ax.axvline(len(LEARN) - 0.5, color="k", lw=1.0)
            if i == 0:
                ax.set_title(TLAB[tier], fontsize=10)
            if j == 0:
                ax.set_ylabel("By victim" if by == "victim" else "By co-runner")
            T2 = T.copy()
            T2.insert(0, "base_mae", base)
            T2.index = [NAME[a] for a in T2.index]
            tables.append((tier, by, T2))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.5, w_pad=0.8)
    cax = fig.add_axes([0.30, 0.025, 0.40, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Skill vs isolated baseline (MPI time; colour clipped at $-1$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    save(fig, "r_pervictim.pdf")
    plt.close(fig)
    p("## r_pervictim.pdf (6.0 x 6.6 in)\nMPI time, socket-aware, 1W (response models: their only form). Skill = 1 - "
      "mean log MAE / mean baseline MAE over 10 seeds, per victim (top) and per co-runner (bottom). Colour clipped at "
      "-1; * = out of the inhibitor hull (Quicksilver). base_mae = baseline log MAE for that app.\n")
    for tier, by, T2 in tables:
        p(f"### {tier}, by {by}\n")
        p(md(T2.reset_index().rename(columns={"index": by}), fmt="{:.3f}") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2_full_20261123")
    tag = ap.parse_args().tag
    rd = lambda n: pd.read_csv(OUT_DIR / f"v2_pred_{tag}_{n}.csv")  # noqa: E731
    M, S, C, CS, D = rd("metrics"), rd("summary"), rd("contrasts"), rd("contrast_summary"), rd("breakdown")
    p(f"# v2 prediction figures ({tag})\n\nSource: output/v2_pred_{tag}_*.csv (v2_score.py). Methods: xgb is the "
      "max_depth 1-8 rerun; xgb_d124 not plotted.\n")
    fig_skill(M, S)
    fig_diffs(C, CS)
    fig_pervictim(D)
    (OUT_DIR / "v2_pred_figs_numbers.md").write_text(log.getvalue())
    print("wrote", OUT_DIR / "v2_pred_figs_numbers.md")


if __name__ == "__main__":
    main()
