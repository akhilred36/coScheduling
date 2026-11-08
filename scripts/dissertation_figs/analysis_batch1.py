"""Batch 1 numbers (A3, A4, A6, B1-B4) for the dissertation node-sharing chapter.

Writes scripts/dissertation_figs/output/batch1_numbers_raw.md (machine-generated tables).
Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/analysis_batch1.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from common import (APPS, NAME, ONE_NODE_APPS, OUT_DIR, PROCESSED, directional,
                    directional_from_processed, inhibitors, job_inh, jobs, pairs_mpi,
                    pairs_total, read)

out = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=out)


def md(df: pd.DataFrame, floatfmt=".3g"):
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    idx_names = [n if n else "" for n in (df.index.names or [""])]
    head = idx_names + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(idx_names) + ["---:"] * len(df.columns)) + "|"]
    for key, row in df.iterrows():
        keys = list(key) if isinstance(key, tuple) else [key]
        lines.append("| " + " | ".join([str(k) for k in keys] + [f(v) for v in row.values]) + " |")
    p("\n".join(lines))
    p()


def cv(x):
    x = np.asarray(x, float)
    return np.std(x, ddof=1) / np.mean(x) if len(x) > 1 else np.nan


def dist(x, name):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(set=name, n=len(x), median=np.median(x), p75=np.percentile(x, 75),
                p90=np.percentile(x, 90), max=x.max())


# ============================================================================ A3
p("## A3. Isolated 8-node application profiles")
j = jobs()
t = pd.DataFrame({
    "T_app (s)": j.runtime, "T_MPI (s)": j.mpi_time, "comm_frac": j.comm_frac,
    "messages": j.total_msgs, "bytes": j.total_bytes,
    "msg rate (1/s)": j.total_msgs / j.runtime, "byte rate (B/s)": j.total_bytes / j.runtime,
    "mean msg size (B)": j.total_bytes / j.total_msgs,
})
t.index = [NAME[a] for a in t.index]
md(t)

# ============================================================================ A4
p("## A4. Campaign counts (processed_data replicate-level CSVs)")
iso = read(PROCESSED / "data_isolated.csv")
ii = read(PROCESSED / "data_inhib_isolated.csv")
ai = read(PROCESSED / "data_inhib_coscheduled.csv")
aa = read(PROCESSED / "data_coscheduled.csv")
K = ["Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]
rows = []
for nn in (8, 1):
    g = iso[iso["Num Nodes"] == nn].groupby("App").size()
    rows.append((f"isolated apps, {nn} node", len(g), g.min(), g.max(), g.sum()))
g = ii.groupby(K).size()
rows.append(("isolated inhibitors, 8 node", len(g), g.min(), g.max(), g.sum()))
g = ai.groupby(["App", *K]).size()
rows.append(("App-Inhibitor, 8 node", len(g), g.min(), g.max(), g.sum()))
for nn in (8, 1):
    g = aa[aa["Num Nodes"] == nn].groupby(["App A", "App B"]).size()
    rows.append((f"App-App, {nn} node", len(g), g.min(), g.max(), g.sum()))
md(pd.DataFrame(rows, columns=["campaign", "configs", "min reps", "max reps", "runs"]).set_index("campaign"))

# ============================================================================ A6
p("## A6. Clipping at 1")
pm = directional(pairs_mpi(), "raw_mpi_time_ratio")
pt = directional(pairs_total(), "raw_app_time_ratio")
ji = job_inh()
rows = []
for name, d in [("App-App MPI, non-self", pm[~pm.self]), ("App-App MPI, self", pm[pm.self]),
                ("App-App total, non-self", pt[~pt.self]), ("App-App total, self", pt[pt.self])]:
    rows.append((name, len(d), (d.raw < 1).mean(), (d.raw == 1).mean(), (d.raw <= 1).mean(), d.raw.min()))
rows.append(("App-Inhibitor MPI", len(ji), (ji.raw_mpi_time_ratio < 1).mean(),
             (ji.raw_mpi_time_ratio == 1).mean(), (ji.raw_mpi_time_ratio <= 1).mean(),
             ji.raw_mpi_time_ratio.min()))
md(pd.DataFrame(rows, columns=["set", "n", "frac raw<1", "frac raw==1", "frac floor (<=1)", "min raw"]).set_index("set"), ".4g")

# App-Inhibitor total-time clipping (from processed aggregates)
aia = read(PROCESSED / "data_inhib_coscheduled_agg.csv")
isoa = read(PROCESSED / "data_isolated_agg.csv")
isoa = isoa[isoa["Num Nodes"] == 8].set_index("App")
r_tt = aia["App Time Average Mean"] / aia.App.map(isoa["App Time Average Mean"])
p(f"App-Inhibitor total-time raw ratio: frac<1 = {(r_tt<1).mean():.4f}, frac<=1 = {(r_tt<=1).mean():.4f}, min = {r_tt.min():.4f}, n = {len(r_tt)}")
p()

# ============================================================================ B1
p("## B1. Replicate noise")
i8 = iso[iso["Num Nodes"] == 8]
g = i8.groupby("App")
t = pd.DataFrame({"CV T_MPI": g["MPI Time Average"].apply(cv),
                  "CV T_app": g["App Time Average"].apply(cv),
                  "T_app range (s)": g["App Time Average"].agg(lambda x: x.max() - x.min()),
                  "n": g.size()}).loc[APPS]
t.index = [NAME[a] for a in t.index]
md(t, ".4f")
p(f"Isolated 8-node CV T_MPI: median {t['CV T_MPI'].median():.4f}, mean {t['CV T_MPI'].mean():.4f}; "
  f"CV T_app: median {t['CV T_app'].median():.4f}, max {t['CV T_app'].max():.4f}")
p()

cv_ii = ii.groupby(K)["Inhib MPI Time Average"].apply(cv)
cv_ai = ai.groupby(["App", *K])["MPI Time Average"].apply(cv)
cv_ai_app = ai.groupby(["App", *K])["App Time Average"].apply(cv)
aa8 = aa[aa["Num Nodes"] == 8]
ga = aa8.groupby(["App A", "App B"])
cv_aa = pd.concat([ga["App A MPI Time Average"].apply(cv), ga["App B MPI Time Average"].apply(cv)])
cv_aa_app = pd.concat([ga["App A Time Average"].apply(cv), ga["App B Time Average"].apply(cv)])
rows = [dist(t["CV T_MPI"], "isolated apps T_MPI (per app)"),
        dist(cv_ii, "isolated inhibitors T_MPI (per config)"),
        dist(cv_ai, "App-Inhibitor app T_MPI (per config)"),
        dist(cv_aa, "App-App T_MPI (per pair endpoint, incl. self)"),
        dist(t["CV T_app"], "isolated apps T_app"),
        dist(cv_ai_app, "App-Inhibitor app T_app"),
        dist(cv_aa_app, "App-App T_app")]
md(pd.DataFrame(rows).set_index("set"), ".4f")
p(f"Isolated inhibitors: fraction of configs with CV>10%: {(cv_ii>0.10).mean():.3f}; >50%: {(cv_ii>0.5).mean():.3f}")
worst = cv_ii.sort_values(ascending=False).head(5)
p("Top-5 isolated-inhibitor CV configs (msg, wait us, sparsity -> CV):")
for k, v in worst.items():
    p(f"  {k} -> {v:.3f}")
p()
# App-Inhibitor CV per app
md(pd.DataFrame({"median CV T_MPI": cv_ai.groupby(level=0).median(),
                 "p90 CV T_MPI": cv_ai.groupby(level=0).quantile(0.9)}).rename(index=NAME), ".4f")

# Propagated uncertainty of App-App MPI slowdown (delta method on ratio of means)
iso_stats = i8.groupby("App")["MPI Time Average"].agg(["mean", "std", "size"])
recs = []
for (a, b), grp in ga:
    for v, col in (("A", "App A MPI Time Average"), ("B", "App B MPI Time Average")):
        victim = a if v == "A" else b
        x = grp[col].to_numpy(float)
        mi, si, ni = iso_stats.loc[victim]
        r = x.mean() / mi
        rel = np.sqrt((x.std(ddof=1) ** 2 / len(x)) / x.mean() ** 2 + (si ** 2 / ni) / mi ** 2)
        recs.append(dict(victim=victim, aggressor=b if v == "A" else a, raw=r, se=r * rel,
                         self=a == b))
se = pd.DataFrame(recs)
ns = se[~se.self]
above = ns[ns.raw > 1]
p("Propagated SE of raw App-App MPI ratio (delta method, 4 co reps x 4 iso reps), non-self:")
p(f"  median SE {ns.se.median():.4f}, p90 {ns.se.quantile(.9):.4f}, max {ns.se.max():.4f}")
p(f"  outcomes with raw-1 < 2 SE: {(ns.raw - 1 < 2 * ns.se).sum()} / {len(ns)} "
  f"(of which {(ns.raw <= 1).sum()} are at/below the floor); "
  f"above-floor outcomes within 2 SE of 1: {(above.raw - 1 < 2 * above.se).sum()} / {len(above)}")
p(f"  outcomes with clipped slowdown >= 1.05 that are > 2 SE above 1: "
  f"{((ns.raw >= 1.05) & (ns.raw - 1 > 2 * ns.se)).sum()} / {(ns.raw >= 1.05).sum()}")
chk = pm[~pm.self].merge(ns, on=["victim", "aggressor"])
p(f"  check: max |raw(canonical) - raw(replicates)| = {np.abs(chk.raw_x - chk.raw_y).max():.2e}")
p("  per-victim median SE:", ", ".join(f"{NAME[k]} {v:.3f}" for k, v in ns.groupby('victim').se.median().items()))
p()

# ============================================================================ B2
p("## B2. App-App slowdown distribution (8 node)")
rows = []
for name, d in [("MPI, non-self", pm[~pm.self]), ("total, non-self", pt[~pt.self]),
                ("MPI, self (20 copies)", pm[pm.self]), ("total, self (20 copies)", pt[pt.self])]:
    s = d.slowdown.to_numpy()
    rows.append(dict(set=name, n=len(s), mean=s.mean(), median=np.median(s), p75=np.percentile(s, 75),
                     p90=np.percentile(s, 90), max=s.max(), floor=(s == 1).mean(),
                     **{"frac>=1.05": (s >= 1.05).mean(), "frac>=1.10": (s >= 1.10).mean()}))
md(pd.DataFrame(rows).set_index("set"), ".3f")

p("Self-pairs, MPI and total slowdown of both copies (A, B) and mean:")
sp = pairs_mpi().query("jobA_id == jobB_id").set_index("jobA_id")
st = pairs_total().query("jobA_id == jobB_id").set_index("jobA_id")
t = pd.DataFrame({"MPI A": sp.slowdown_A, "MPI B": sp.slowdown_B,
                  "MPI mean": (sp.slowdown_A + sp.slowdown_B) / 2,
                  "total A": st.slowdown_A, "total B": st.slowdown_B}).loc[APPS].rename(index=NAME)
md(t, ".3f")

pr = pairs_mpi().query("jobA_id != jobB_id").copy()
pr["log_ratio"] = np.abs(pr.log_slowdown_A - pr.log_slowdown_B)
pr = pr.sort_values("log_ratio", ascending=False)
p("Top-5 most asymmetric non-self pairs (MPI, by |log s_A - log s_B|):")
md(pd.DataFrame({"pair": [f"{NAME[a]} + {NAME[b]}" for a, b in zip(pr.jobA_id, pr.jobB_id)],
                 "s_A(B)": pr.slowdown_A, "s_B(A)": pr.slowdown_B,
                 "|log ratio|": pr.log_ratio}).head(5).set_index("pair"), ".3f")
prt = pairs_total().query("jobA_id != jobB_id").copy()
prt["log_ratio"] = np.abs(prt.log_slowdown_A - prt.log_slowdown_B)
prt = prt.sort_values("log_ratio", ascending=False)
p("Top-5 most asymmetric non-self pairs (total time):")
md(pd.DataFrame({"pair": [f"{NAME[a]} + {NAME[b]}" for a, b in zip(prt.jobA_id, prt.jobB_id)],
                 "s_A(B)": prt.slowdown_A, "s_B(A)": prt.slowdown_B,
                 "|log ratio|": prt.log_ratio}).head(5).set_index("pair"), ".3f")


# ============================================================================ B3
def decompose(d: pd.DataFrame, label: str):
    y = np.log(d.slowdown.to_numpy())
    V = pd.get_dummies(d.victim).to_numpy(float)
    X = pd.get_dummies(d.aggressor).to_numpy(float)
    tss = ((y - y.mean()) ** 2).sum()

    def r2(M):
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        return 1 - ((y - M @ beta) ** 2).sum() / tss, beta

    r2v, _ = r2(V)
    r2x, _ = r2(X)
    M = np.column_stack([np.ones(len(y)), V[:, 1:], X[:, 1:]])
    r2vx, beta = r2(M)
    # Recover sum-to-zero effects from the fitted values.
    fitted = M @ beta
    f = pd.DataFrame({"v": d.victim.values, "x": d.aggressor.values, "f": fitted})
    # Additive fitted surface: f = mu + a_v + b_x ; with near-balanced design, use
    # least squares with sum-to-zero constraints via explicit construction.
    vs, xs = sorted(d.victim.unique()), sorted(d.aggressor.unique())
    A = np.zeros((len(y), 1 + len(vs) + len(xs)))
    A[:, 0] = 1
    for i, (v, x) in enumerate(zip(d.victim, d.aggressor)):
        A[i, 1 + vs.index(v)] = 1
        A[i, 1 + len(vs) + xs.index(x)] = 1
    C = np.zeros((2, A.shape[1]))
    C[0, 1:1 + len(vs)] = 1
    C[1, 1 + len(vs):] = 1
    KKT = np.block([[A.T @ A, C.T], [C, np.zeros((2, 2))]])
    sol = np.linalg.solve(KKT, np.concatenate([A.T @ y, np.zeros(2)]))
    mu, av, bx = sol[0], sol[1:1 + len(vs)], sol[1 + len(vs):1 + len(vs) + len(xs)]
    assert np.allclose(A @ sol[:A.shape[1]], fitted)
    # Two-way ANOVA style sequential and partial shares
    p(f"### {label}: n = {len(y)}, var(log s) = {y.var(ddof=0):.5f}")
    p(f"R2 victim-only = {r2v:.3f}; aggressor-only = {r2x:.3f}; additive victim+aggressor = {r2vx:.3f}; "
      f"residual (interaction + noise) share = {1 - r2vx:.3f}")
    p(f"Partial R2 adding aggressor to victim = {r2vx - r2v:.3f}; adding victim to aggressor = {r2vx - r2x:.3f}")
    p(f"Grand mean log s = {mu:.4f} (x{np.exp(mu):.3f})")
    eff = pd.DataFrame({"victim effect (log)": pd.Series(av, vs), "victim factor": pd.Series(np.exp(av), vs),
                        "aggressor effect (log)": pd.Series(bx, xs), "aggressor factor": pd.Series(np.exp(bx), xs)})
    eff.index = [NAME[a] for a in eff.index]
    p("Effects (sum-to-zero; factor = exp(effect)); sorted by victim effect:")
    md(eff.sort_values("victim effect (log)", ascending=False), ".3f")
    p("Aggressor ranking:", ", ".join(eff.sort_values("aggressor effect (log)", ascending=False).index))
    p()


p("## B3. Victim / aggressor decomposition of log slowdown (non-self, 90 outcomes)")
decompose(pm[~pm.self], "MPI time")
decompose(pt[~pt.self], "Total time")
dm = pm[~pm.self].copy()
dm["slowdown"] = np.maximum(dm.raw, 1e-9)
p("Sensitivity: MPI time using unclipped raw ratio (log raw):")
decompose(dm.assign(slowdown=dm.raw), "MPI time, unclipped")

for name, pr_ in (("MPI", pairs_mpi()), ("total", pairs_total())):
    q = pr_.query("jobA_id != jobB_id")
    rho, pv = stats.spearmanr(q.slowdown_A, q.slowdown_B)
    rho_raw, pv_raw = stats.spearmanr(q[[c for c in q.columns if c.startswith('raw_') and c.endswith('_A')][0]],
                                      q[[c for c in q.columns if c.startswith('raw_') and c.endswith('_B')][-1]])
    p(f"Spearman s_A(B) vs s_B(A) over 45 non-self pairs, {name}: rho = {rho:.3f} (p = {pv:.3g}); "
      f"on raw ratios rho = {rho_raw:.3f} (p = {pv_raw:.3g})")
p()

# ============================================================================ B4
p("## B4. 1-node subset vs 8-node on the same 6 apps / 21 pairs")
rows = []
for nn in (1, 8):
    for time in ("MPI", "total"):
        d = directional_from_processed(nn, time)
        d = d[d.victim.isin(ONE_NODE_APPS) & d.aggressor.isin(ONE_NODE_APPS)]
        for selfset, dd in (("non-self", d[~d.self]), ("self", d[d.self])):
            s = dd.slowdown.to_numpy()
            rows.append(dict(nodes=nn, time=time, set=selfset, n=len(s), median=np.median(s),
                             p75=np.percentile(s, 75), p90=np.percentile(s, 90), max=s.max(),
                             floor=(s == 1).mean(), min_raw=dd.raw.min()))
md(pd.DataFrame(rows).set_index(["nodes", "time", "set"]), ".3f")
# consistency check of processed-route 8-node vs canonical
chk = directional_from_processed(8, "MPI")
chk = chk[~chk.self].merge(pm[~pm.self], on=["victim", "aggressor"])
p(f"check: max |8-node processed-route MPI slowdown - canonical| = {np.abs(chk.slowdown_x - chk.slowdown_y).max():.2e}")
d1 = directional_from_processed(1, "MPI")
d8 = directional_from_processed(8, "MPI")
m = d1[~d1.self].merge(d8[~d8.self], on=["victim", "aggressor"], suffixes=("_1", "_8"))
rho, pv = stats.spearmanr(m.slowdown_1, m.slowdown_8)
p(f"Spearman 1-node vs 8-node MPI slowdown, same 30 directional outcomes: rho = {rho:.3f} (p = {pv:.3g})")
md(pd.DataFrame({"victim": m.victim.map(NAME), "aggressor": m.aggressor.map(NAME),
                 "s 1-node": m.slowdown_1, "s 8-node": m.slowdown_8}).sort_values("s 8-node", ascending=False).set_index(["victim", "aggressor"]), ".3f")

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "batch1_numbers_raw.md").write_text(out.getvalue())
print("wrote", OUT_DIR / "batch1_numbers_raw.md")
