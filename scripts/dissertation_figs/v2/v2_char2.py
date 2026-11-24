"""Plan v2 batch B8: unclipped JMs re-computations of the old §4 characterization numbers + 3 figure regenerations.

Conventions (same as B1 / v2_char.py): unclipped ratio T_co / T_base (ratio of aggregate means), JMs baseline
(v2_build.py), the 90 non-self rep-averaged directed App-App outcomes unless stated; [J] = June baseline (no socket
matching) given alongside. Log ratio y = ln(ratio).

Writes output/v2_char2_numbers.md, output/v2_char2_heat_{mpi,tt}.csv, output/v2_char2_se.csv and regenerates
slowdown_vs_profile.pdf, inhib_response.pdf, scale_1node.pdf (unclipped) in Dissertation/figures/node_sharing/.
Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/v2/v2_char2.py
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (APPS, FULL_W, HALF_W, NAME, OI, ONE_NODE_APPS, OUT_DIR, PROCESSED,  # noqa: E402
                    directional_from_processed, jobs, read, save, set_style)
from v2_build import jms_factors  # noqa: E402

set_style()
log = io.StringIO()
C = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"]]
LS = ["-", "--", "-.", ":"]
MK = ["o", "s", "^", "D"]
TCOL = {"mpi": "MPI Time Average", "tt": "App Time Average"}
TNAME = {"mpi": "MPI time", "tt": "app time"}
L105 = np.log(1.05)


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                                 for v in r]) + " |")
    return "\n".join(out)


def label_panel(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.concatenate([[x[0]], x]), np.concatenate([[0], y]), where="post", **kw)


D = pd.read_csv(OUT_DIR / "v2_appapp_directed.csv")
for tk in ("mpi", "tt"):
    D[f"yJ_{tk}"] = np.log(D[f"rawJ_{tk}"])
NS = D[~D.is_self].copy()
assert len(NS) == 90
p("# v2 batch B8: unclipped JMs re-computations of the §4 characterization numbers (v2/v2_char2.py)\n")
p("Ratio = T_co / T_base (ratio of rep-averaged means), unclipped; JMs baseline unless marked [J] (June, no socket "
  "matching). y = ln ratio. 90 non-self directed outcomes unless stated. JMs factors applied to AMG2023 and "
  "Quicksilver only (both timers), so [J] differs only for rows with those victims.\n")

# ============================================================================ 1. heatmap values
p("## 1. App-App matrices (victim rows x co-runner columns)\n")
for tk in ("mpi", "tt"):
    M = D.pivot(index="victim", columns="corunner", values=f"raw_{tk}").loc[APPS, APPS]
    MJ = D.pivot(index="victim", columns="corunner", values=f"rawJ_{tk}").loc[APPS, APPS]
    out = M.rename(index=NAME, columns=NAME)
    out.index.name = "victim \\ corunner"
    out.to_csv(OUT_DIR / f"v2_char2_heat_{tk}.csv")
    MJ.rename(index=NAME, columns=NAME).to_csv(OUT_DIR / f"v2_char2_heat_{tk}_J.csv")
    p(f"### {TNAME[tk]} (JMs; diagonal = pooled self-pair, both copies x 4 reps vs 50/50 baseline)\n")
    p(md(out, "{:.3f}") + "\n")
    off = np.log(M.where(~np.eye(10, dtype=bool)))
    offJ = np.log(MJ.where(~np.eye(10, dtype=bool)))
    mm = pd.DataFrame({"victim mean (geo)": np.exp(off.mean(axis=1)), "victim mean [J]": np.exp(offJ.mean(axis=1)),
                       "victim arith mean": M.where(~np.eye(10, dtype=bool)).mean(axis=1),
                       "co-runner mean (geo)": np.exp(off.mean(axis=0)),
                       "co-runner mean [J]": np.exp(offJ.mean(axis=0)),
                       "co-runner arith mean": M.where(~np.eye(10, dtype=bool)).mean(axis=0)}).rename(index=NAME)
    mm.index.name = "app"
    p(f"Row (victim) and column (co-runner) means over the 9 off-diagonal cells, {TNAME[tk]}; geometric mean = "
      "exp(mean y):\n")
    p(md(mm.sort_values("victim mean (geo)", ascending=False), "{:.3f}") + "\n")
q = NS.sort_values("raw_mpi", ascending=False).head(8)
p("### Top 8 off-diagonal MPI cells\n")
t = pd.DataFrame({"victim": q.victim.map(NAME).values, "co-runner": q.corunner.map(NAME).values,
                  "MPI ratio": q.raw_mpi.values, "[J]": q.rawJ_mpi.values, "app ratio": q.raw_tt.values,
                  "frac sock1": q.frac_sock1.values}).set_index("victim")
p(md(t, "{:.3f}") + "\n")
s = D[D.is_self].set_index("victim").loc[APPS]
t = pd.DataFrame({"MPI": s.raw_mpi, "MPI [J]": s.rawJ_mpi, "app": s.raw_tt, "app [J]": s.rawJ_tt}).rename(index=NAME)
t.index.name = "self-pair"
p("### 10 self-pairs (pooled both copies)\n")
p(md(t, "{:.3f}") + "\n")

# ============================================================================ 2. asymmetry
p("## 2. Asymmetry: s_A(B) vs s_B(A) over the 45 non-self pairs\n")
key = NS.set_index(["victim", "corunner"])
cco = read(PROCESSED / "data_coscheduled.csv")
cco = cco[(cco["Num Nodes"] == 8) & (cco["App A"] != cco["App B"])]
run_pairs = sorted(set(zip(cco["App A"], cco["App B"])))  # as-run A/B orientation (= canonical pair.csv)
assert len(run_pairs) == 45
alpha_pairs = [(a, b) for i, a in enumerate(APPS) for b in APPS[i + 1:]]
pairs = run_pairs
p("Spearman over unordered pairs depends on which app is called A. Primary = the as-run A/B orientation (the "
  "canonical pair.csv one, as in the old number); also the APPS-order orientation, and an orientation-free version "
  "(both orderings, n = 90 points; its p-value treats them as independent and is optimistic, so only rho is given).\n")
for tk in ("mpi", "tt"):
    for lab, col in (("JMs", f"y_{tk}"), ("[J]", f"yJ_{tk}")):
        res = []
        for pl in (run_pairs, alpha_pairs):
            x = [key.loc[(a, b), col] for a, b in pl]
            y = [key.loc[(b, a), col] for a, b in pl]
            res.append(stats.spearmanr(x, y))
        xs = [key.loc[(a, b), col] for a, b in run_pairs] + [key.loc[(b, a), col] for a, b in run_pairs]
        ys = [key.loc[(b, a), col] for a, b in run_pairs] + [key.loc[(a, b), col] for a, b in run_pairs]
        rs = stats.spearmanr(xs, ys).statistic
        p(f"- {TNAME[tk]} {lab}: as-run rho = {res[0].statistic:.3f}, p = {res[0].pvalue:.3g}; APPS-order rho = "
          f"{res[1].statistic:.3f}, p = {res[1].pvalue:.3g}; orientation-free rho = {rs:.3f} (n = 45 pairs)")
asym = pd.DataFrame([dict(pair=f"{NAME[a]} + {NAME[b]}", **{"s_A(B)": key.loc[(a, b), "raw_mpi"],
                                                             "s_B(A)": key.loc[(b, a), "raw_mpi"]},
                          **{"|y_A - y_B|": abs(key.loc[(a, b), "y_mpi"] - key.loc[(b, a), "y_mpi"]),
                             "|.| [J]": abs(key.loc[(a, b), "yJ_mpi"] - key.loc[(b, a), "yJ_mpi"])})
                     for a, b in pairs]).set_index("pair").sort_values("|y_A - y_B|", ascending=False)
p("\nTop 5 most asymmetric pairs (MPI, by |y_A - y_B|; A = first named app):\n")
p(md(asym.head(5), "{:.3f}") + "\n")


# ============================================================================ 3. fixed effects
def decompose(d: pd.DataFrame, ycol: str, label: str):
    y = d[ycol].to_numpy(float)
    vs, xs = APPS, APPS
    V = pd.get_dummies(d.victim).reindex(columns=vs).to_numpy(float)
    X = pd.get_dummies(d.corunner).reindex(columns=xs).to_numpy(float)
    tss = ((y - y.mean()) ** 2).sum()

    def r2(Mx):
        beta, *_ = np.linalg.lstsq(Mx, y, rcond=None)
        return 1 - ((y - Mx @ beta) ** 2).sum() / tss

    r2v, r2x = r2(V), r2(X)
    A = np.column_stack([np.ones(len(y)), V, X])
    Cn = np.zeros((2, A.shape[1]))
    Cn[0, 1:11], Cn[1, 11:] = 1, 1
    KKT = np.block([[A.T @ A, Cn.T], [Cn, np.zeros((2, 2))]])
    sol = np.linalg.solve(KKT, np.concatenate([A.T @ y, np.zeros(2)]))[:A.shape[1]]
    r2vx = 1 - ((y - A @ sol) ** 2).sum() / tss
    mu, av, bx = sol[0], pd.Series(sol[1:11], vs), pd.Series(sol[11:], xs)
    p(f"### {label}: n = {len(y)}, var(y) = {y.var():.5f}")
    p(f"R² victim-only = {r2v:.3f}; co-runner-only = {r2x:.3f}; additive = {r2vx:.3f}; residual "
      f"(interaction + noise) = {1 - r2vx:.3f}")
    p(f"Overall mean (grand mean of y, sum-to-zero effects) = {mu:.4f} (x{np.exp(mu):.3f})")
    eff = pd.DataFrame({"victim factor": np.exp(av), "co-runner factor": np.exp(bx)}).rename(index=NAME)
    eff.index.name = "app"
    p(md(eff.sort_values("victim factor", ascending=False), "{:.3f}"))
    p(f"Strongest victim: {eff['victim factor'].idxmax()} (x{eff['victim factor'].max():.3f}); weakest: "
      f"{eff['victim factor'].idxmin()} (x{eff['victim factor'].min():.3f}). Strongest co-runner: "
      f"{eff['co-runner factor'].idxmax()} (x{eff['co-runner factor'].max():.3f}); weakest: "
      f"{eff['co-runner factor'].idxmin()} (x{eff['co-runner factor'].min():.3f}).")
    p("Co-runner ranking: " + ", ".join(eff.sort_values("co-runner factor", ascending=False).index) + "\n")
    return dict(r2v=r2v, r2x=r2x, r2vx=r2vx, mu=mu)


p("## 3. Fixed-effects decomposition of y (90 non-self outcomes)\n")
FE = {}
for tk in ("mpi", "tt"):
    FE[tk] = decompose(NS, f"y_{tk}", f"{TNAME[tk]}, JMs")
    FE[tk + "J"] = decompose(NS, f"yJ_{tk}", f"{TNAME[tk]}, [J]")

# ============================================================================ 4. profile relationship
j = jobs()
runtime = j.mpi_time / j.comm_frac
NS["co_byte"] = NS.corunner.map(j.total_bytes / runtime)
NS["co_msg"] = NS.corunner.map(j.total_msgs / runtime)
NS["vic_frac"] = NS.victim.map(j.comm_frac)
p("## 4. Profile relationship (MPI y vs isolated profile; rates over T_app, as select_inhibitors.py)\n")
specs = [("co_byte", "Co-runner byte rate\n(B/s)", "(a)"), ("co_msg", "Co-runner message rate\n(msgs/s)", "(b)"),
         ("vic_frac", "Victim comm. fraction\n" + r"($T_\mathrm{MPI}/T_\mathrm{app}$)", "(c)")]
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), sharey=True)
for ax, (col, xl, tag) in zip(axes, specs):
    r = stats.spearmanr(NS[col], NS.y_mpi)
    rJ = stats.spearmanr(NS[col], NS.yJ_mpi)
    p(f"- {tag} {xl.splitlines()[0]}: rho = {r.statistic:.3f}, p = {r.pvalue:.3g}; [J] rho = {rJ.statistic:.3f}, "
      f"p = {rJ.pvalue:.3g}")
    ax.axhline(1, color=OI["grey"], lw=0.8, ls=":", zorder=0)
    ax.scatter(NS[col], NS.raw_mpi, s=14, color=C[0], alpha=0.8, lw=0.4, ec="white", zorder=2)
    ax.text(0.04, 0.95, rf"$\rho = {r.statistic:.2f}$", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.grid(lw=0.4, alpha=0.4)
    label_panel(ax, tag, x=0.0)
axes[0].set_ylabel("MPI-time ratio")
yt = [0.8, 1, 1.25, 1.5, 2]
axes[0].set_yticks(yt)
axes[0].set_yticklabels([f"{t:g}" for t in yt])
axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(w_pad=0.8)
save(fig, "slowdown_vs_profile.pdf")
plt.close(fig)
p(f"\nslowdown_vs_profile.pdf regenerated: y = unclipped JMs MPI ratio (log axis, range {NS.raw_mpi.min():.3f}-"
  f"{NS.raw_mpi.max():.3f}), dotted line at 1; no floor encoding. Correlations on y (rank-identical to the ratio).\n")

# ============================================================================ 5. noise / SE
p("## 5. Propagated SE of the App-App MPI ratio (delta method), JMs\n")
f = jms_factors()
pp = pd.read_csv(OUT_DIR / "socket_pinned_pairs.csv")
iso = read(PROCESSED / "data_isolated.csv")
i8 = iso[iso["Num Nodes"] == 8]
cc = read(PROCESSED / "data_coscheduled.csv")
cc = cc[cc["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"})
cc = cc.merge(pd.read_csv(OUT_DIR / "socket_app_app.csv"), on=["A", "B", "rep"], how="left", validate="one_to_one")
recs = []
for r in cc.to_dict("records"):
    if r["A"] == r["B"]:
        continue
    for v, x in (("A", "B"), ("B", "A")):
        recs.append(dict(victim=r[v], corunner=r[x], s=int(r[f"sock_{v}"]), T=r[f"App {v} MPI Time Average"]))
R = pd.DataFrame(recs)
fx = f[f.timer == "mpi"].set_index("app")
dcol = "dMPI"
se_rows = []
for (v, x), g in R.groupby(["victim", "corunner"]):
    Tj = i8.loc[i8.App == v, TCOL["mpi"]].to_numpy(float)
    mJ, vJ = Tj.mean(), Tj.var(ddof=1) / len(Tj) / Tj.mean() ** 2  # rel var of June mean
    dd = pp.loc[pp.App == v, dcol].to_numpy(float)
    applied, d = bool(fx.loc[v, "applied"]), float(fx.loc[v, "d_used"])
    se_d = dd.std(ddof=1) / np.sqrt(len(dd)) if applied else 0.0
    s = g.s.to_numpy(float)
    T = g["T"].to_numpy(float)
    b = mJ * np.exp(d * s)  # per-replicate matched baseline
    raw = T.mean() / b.mean()
    # co-run mean: socket-adjusted replicates (each rep scaled to the mixture baseline) remove the known socket shift
    Ta = T * b.mean() / b
    v_co = Ta.var(ddof=1) / len(Ta) / Ta.mean() ** 2
    v_co_plain = T.var(ddof=1) / len(T) / T.mean() ** 2
    fr = (s * np.exp(d * s)).mean() / np.exp(d * s).mean()  # d ln(mean b)/dd
    rel = np.sqrt(v_co + vJ + (fr * se_d) ** 2)
    relJ = np.sqrt(v_co_plain + vJ)
    rawJ = T.mean() / mJ
    se_rows.append(dict(victim=v, corunner=x, raw=raw, se=raw * rel, se_plain_co=raw * np.sqrt(v_co_plain + vJ +
                                                                                               (fr * se_d) ** 2),
                        rawJ=rawJ, seJ=rawJ * relJ, d_term=raw * fr * se_d))
SE = pd.DataFrame(se_rows)
chk = SE.merge(NS[["victim", "corunner", "raw_mpi", "rawJ_mpi"]], on=["victim", "corunner"])
assert np.abs(chk.raw - chk.raw_mpi).max() < 1e-12 and np.abs(chk.rawJ - chk.rawJ_mpi).max() < 1e-12
SE.to_csv(OUT_DIR / "v2_char2_se.csv", index=False)


def counts(raw, se):
    return dict(n=len(raw), median_se=np.median(se), p90_se=np.percentile(se, 90), max_se=se.max(),
                harm_resolved=int(((raw >= 1.05) & (raw - 1 > 2 * se)).sum()), n_harm=int((raw >= 1.05).sum()),
                above1_resolved=int((raw - 1 > 2 * se).sum()),
                speedup_resolved=int(((raw < 1) & (1 - raw > 2 * se)).sum()), n_below1=int((raw < 1).sum()),
                within_2se=int((np.abs(raw - 1) <= 2 * se).sum()))


t = pd.DataFrame({"JMs": counts(SE.raw.to_numpy(), SE.se.to_numpy()),
                  "JMs, unadjusted co-run var": counts(SE.raw.to_numpy(), SE.se_plain_co.to_numpy()),
                  "[J]": counts(SE.rawJ.to_numpy(), SE.seJ.to_numpy())}).T
t.index.name = "variant"
p("SE = ratio x sqrt(relvar(co-run mean) + relvar(June iso mean) + (f_d x SE(d))²), where f_d = d ln(mean "
  "baseline)/dd is the replicate-weighted socket-1 share and SE(d) is the paired pinned SE (AMG2023, Quicksilver "
  "only; 0 elsewhere). Co-run replicates are socket-adjusted (each scaled by mean baseline / its own baseline) so "
  "the known socket shift is not counted as noise; the unadjusted variant is shown for comparison. 4 co-run reps x "
  "4 June iso reps per outcome.\n")
p(md(t, "{:.4f}") + "\n")
p("Columns: harm_resolved = ratio >= 1.05 and > 1 + 2 SE (of n_harm with ratio >= 1.05); above1_resolved = any ratio "
  "> 1 + 2 SE; speedup_resolved = ratio < 1 and < 1 - 2 SE (of n_below1); within_2se = |ratio - 1| <= 2 SE.\n")
pv = SE.groupby("victim").agg(median_se=("se", "median"), max_se=("se", "max"), median_seJ=("seJ", "median"),
                              median_ratio=("raw", "median")).loc[APPS].rename(index=NAME)
pv.index.name = "victim"
p("Per-victim SE (median over its 9 outcomes); requested apps: Laghos, LAMMPS, FIESTA:\n")
p(md(pv, "{:.4f}") + "\n")
p(f"d-term share: median SE contribution from SE(d) for AMG2023 {SE[SE.victim == 'amg'].d_term.median():.4f}, "
  f"Quicksilver {SE[SE.victim == 'quicksilver'].d_term.median():.4f}.\n")
spd = SE[(SE.raw < 1) & (1 - SE.raw > 2 * SE.se)].sort_values("raw")
p("Resolved speedups (ratio < 1 - 2 SE): " + "; ".join(f"{NAME[a]}<-{NAME[b]} {r:.3f}±{s_:.3f}" for a, b, r, s_ in
                                                      zip(spd.victim, spd.corunner, spd.raw, spd.se)) + "\n")

# ============================================================================ 6. inhibitor response
p("## 6. Inhibitor response (inhib_response.pdf, unclipped JMs MPI ratio)\n")
G = pd.read_csv(OUT_DIR / "v2_appinh_grid.csv")
VICTIMS = ["amg", "minivite", "quicksilver"]
WAITS = [0, 1000, 100000, 1000000]
SIZES = [2000, 40000, 400000, 4000000]
fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 6.0), sharex="col", sharey="row")
YT = [0.6, 0.8, 1, 1.25, 1.5, 2, 3, 4, 6]
for i, v in enumerate(VICTIMS):
    sub = G[G.app == v]
    ax = axes[i, 0]
    for k, wt in enumerate(WAITS):
        s = sub[(sub.sp == 1.0) & (sub.w == wt)].sort_values("m")
        lab = "0" if wt == 0 else rf"$10^{{{int(np.log10(wt))}}}$"
        ax.plot(s.m, s.raw_mpi, color=C[k], ls=LS[k], marker=MK[k], ms=3.5, label=f"wait {lab} µs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"{NAME[v]}\nMPI-time ratio")
    ax.grid(lw=0.4, alpha=0.4)
    ax = axes[i, 1]
    for k, ms in enumerate(SIZES):
        s = sub[(sub.w == 0) & (sub.m == ms) & (sub.sp >= 0.1)].sort_values("sp")
        lab = {2000: "2 KB", 40000: "40 KB", 400000: "400 KB", 4000000: "4 MB"}[ms]
        ax.plot(s.sp, s.raw_mpi, color=C[k], ls=LS[k], marker=MK[k], ms=3.5, label=lab)
    ax.set_yscale("log")
    ax.grid(lw=0.4, alpha=0.4)
    sl = sub[((sub.sp == 1.0) & sub.w.isin(WAITS)) | ((sub.w == 0) & sub.m.isin(SIZES) & (sub.sp >= 0.1))]
    lo, hi = min(0.95, sl.raw_mpi.min() * 0.95), max(1.2, sl.raw_mpi.max() * 1.08)
    a_ = axes[i, 0]
    a_.set_ylim(lo, hi)
    tk_ = [t for t in YT if lo <= t <= hi]
    if len(tk_) < 3:
        tk_ = [t for t in np.round(np.arange(0.9, 1.3, 0.05), 2) if lo <= t <= hi]
    a_.set_yticks(tk_)
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
for v in VICTIMS:
    sub = G[G.app == v]
    a_s = sub[(sub.sp == 1.0) & sub.w.isin(WAITS)]
    b_s = sub[(sub.w == 0) & sub.m.isin(SIZES) & (sub.sp >= 0.1)]
    p(f"- {NAME[v]}: (a) slice {a_s.raw_mpi.min():.3f}-{a_s.raw_mpi.max():.3f} [J {a_s.rawJ_mpi.min():.3f}-"
      f"{a_s.rawJ_mpi.max():.3f}], n = {len(a_s)}; (b) slice {b_s.raw_mpi.min():.3f}-{b_s.raw_mpi.max():.3f} "
      f"[J {b_s.rawJ_mpi.min():.3f}-{b_s.rawJ_mpi.max():.3f}]; whole grid (287 configs) {sub.raw_mpi.min():.3f}-"
      f"{sub.raw_mpi.max():.3f} [J {sub.rawJ_mpi.min():.3f}-{sub.rawJ_mpi.max():.3f}]; frac < 1 over grid "
      f"{(sub.raw_mpi < 1).mean():.3f}")
qs = G[G.app == "quicksilver"].sort_values("raw_mpi", ascending=False).iloc[0]
p(f"Quicksilver max over the grid: {qs.raw_mpi:.3f} [J {qs.rawJ_mpi:.3f}] at config {qs.config}.")
p("Same victims and slices as the original C5 (base grid). Dotted line at 1; nothing clipped.\n")

# ============================================================================ 7. scale
p("## 7. 1-node vs 8-node, 6 apps (" + ", ".join(NAME[a] for a in ONE_NODE_APPS) + "), 30 non-self outcomes each\n")
p("1-node uses the June 1-node isolated baseline (no 1-node pinned runs, so no socket matching is possible). 8-node "
  "is shown with JMs and with the June baseline [J]. None of the 6 apps has a JMs factor applied (only AMG2023 and "
  "Quicksilver do), so the 8-node JMs and [J] values coincide here.\n")
d1 = directional_from_processed(1, "MPI")
sel = lambda d: d[(~d.self) & d.victim.isin(ONE_NODE_APPS) & d.aggressor.isin(ONE_NODE_APPS)]  # noqa: E731
d1 = sel(d1).rename(columns={"aggressor": "corunner", "raw": "r1"})[["victim", "corunner", "r1"]]
d8 = NS[NS.victim.isin(ONE_NODE_APPS) & NS.corunner.isin(ONE_NODE_APPS)][["victim", "corunner", "raw_mpi",
                                                                            "rawJ_mpi"]]
S = d1.merge(d8, on=["victim", "corunner"], validate="one_to_one")
assert len(S) == 30
rows = []
for lab, col in (("1 node [J]", "r1"), ("8 nodes JMs", "raw_mpi"), ("8 nodes [J]", "rawJ_mpi")):
    x = S[col]
    rows.append(dict(set=lab, n=len(x), median=x.median(), p90=x.quantile(0.9), max=x.max(), min=x.min(),
                     frac_lt1=(x < 1).mean(), frac_ge105=(x >= 1.05).mean()))
t = pd.DataFrame(rows).set_index("set")
p(md(t, "{:.3f}") + "\n")
ks = stats.ks_2samp(S.r1, S.raw_mpi)
rho = stats.spearmanr(S.r1, S.raw_mpi)
p(f"KS two-sample (1 node vs 8 nodes JMs): D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")
p(f"Per-pair Spearman (same 30 directed outcomes): rho = {rho.statistic:.3f}, p = {rho.pvalue:.3g}")
for v, x in (("tricount", "lammps"), ("tricount", "minife"), ("lammps", "tricount"), ("minife", "tricount")):
    r = S[(S.victim == v) & (S.corunner == x)].iloc[0]
    p(f"- {NAME[v]} <- {NAME[x]} (victim <- co-runner): 1 node {r.r1:.3f}, 8 nodes {r.raw_mpi:.3f}")
p("(The pair named 'TriC->LAMMPS' is ambiguous in direction; both directions are listed. Above, the victim is the "
  "first app.)\n")
fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
ecdf(ax, S.r1, color=C[2], ls="-", label="1 node")
ecdf(ax, S.raw_mpi, color=C[0], ls="--", label="8 nodes")
ax.axvline(1, color=OI["grey"], lw=0.8, ls=":")
ax.set_xscale("log")
lo = min(S.r1.min(), S.raw_mpi.min()) * 0.98
hi = max(S.r1.max(), S.raw_mpi.max()) * 1.03
ax.set_xlim(lo, hi)
ticks = [t for t in (0.8, 0.9, 1, 1.1, 1.25, 1.5, 2) if lo <= t <= hi]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("MPI-time ratio (unclipped)")
ax.set_ylabel("Cumulative fraction")
ax.set_ylim(0, 1.01)
ax.grid(lw=0.4, alpha=0.4)
ax.legend(frameon=False, loc="lower right")
save(fig, "scale_1node.pdf")
plt.close(fig)
p(f"scale_1node.pdf regenerated: ECDF of the unclipped ratio, log x {lo:.3f}-{hi:.3f}, dotted line at 1.\n")

# ============================================================================ 8. noise_cv
p("## 8. noise_cv.pdf\n")
p("No dependence on clipping or on the baseline: both panels are replicate CVs of raw times (isolated apps T_MPI and "
  "T_app; isolated inhibitors, App-Inhibitor and App-App co-run T_MPI), computed from the processed replicates in "
  "figs_batch1.py C1. No slowdown ratio enters. No change.\n")

(OUT_DIR / "v2_char2_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "v2_char2_numbers.md")
