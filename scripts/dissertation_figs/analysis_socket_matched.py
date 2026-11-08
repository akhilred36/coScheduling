"""S5/S6: socket-matched co-run slowdowns and re-scoring of frozen predictions (no re-fit).

Baseline variants for the victim's isolated time (per app, per target T_MPI / T_app):
  J    June isolated mean (all socket 0, mpibind thread 1)    -> the chapter's current numbers
  N0   new pinned sock0 mean for every rep                    -> "before" as requested
  NM   new pinned baseline of the socket the victim ran on    -> "after" as requested
  JM   June mean x exp(d_app) for socket-1 reps               -> drift-free socket matching (d = mean paired
                                                                  isolated log(sock1/sock0), S1)
  JMs  as JM but d set to 0 unless its 95% t CI excludes 0    -> conservative (AMG2023, Quicksilver for T_MPI)
For each co-run outcome with reps r on sockets s_r: raw = mean_r(T_co) / mean_r(base[s_r]); slowdown = max(1, raw).
Replicate SE: delta method on the ratio of means (co reps; baseline variance from its own reps).

Writes output/socket_matched_numbers.md (appended to socket_numbers.md as well).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from common import APPS, NAME, OUT_DIR, PROCESSED, REPO, read

log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        cells = [str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


VARIANTS = ["J", "N0", "NM", "JM", "JMs"]
TARGETS = {"MPI": ("MPI Time Average", "dMPI"), "app": ("App Time Average", "dApp")}

# ------------------------------------------------------------------ baselines
iso = read(PROCESSED / "data_isolated.csv")
j8 = iso[iso["Num Nodes"] == 8]
new = pd.read_csv(OUT_DIR / "socket_pinned_runs.csv")
pairs = pd.read_csv(OUT_DIR / "socket_pinned_pairs.csv")

# base[target][variant][app] -> (mean_sock0, var_mean_sock0, mean_sock1, var_mean_sock1)
base = {}
for tk, (col, dcol) in TARGETS.items():
    base[tk] = {}
    J = j8.groupby("App")[col].agg(["mean", "var", "size"])
    N = new.groupby(["App", "Socket"])[col].agg(["mean", "var", "size"])
    dd = pairs.groupby("App")[dcol].agg(["mean", "var", "size"])
    for v in VARIANTS:
        base[tk][v] = {}
        for a in APPS:
            mj, vj = J.loc[a, "mean"], J.loc[a, "var"] / J.loc[a, "size"]
            n0, n1 = N.loc[(a, 0)], N.loc[(a, 1)]
            dm, dv = dd.loc[a, "mean"], dd.loc[a, "var"] / dd.loc[a, "size"]
            sig = abs(dm) > stats.t.ppf(0.975, dd.loc[a, "size"] - 1) * np.sqrt(dv)
            if v == "J":
                b = (mj, vj, mj, vj)
            elif v == "N0":
                b = (n0["mean"], n0["var"] / n0["size"], n0["mean"], n0["var"] / n0["size"])
            elif v == "NM":
                b = (n0["mean"], n0["var"] / n0["size"], n1["mean"], n1["var"] / n1["size"])
            else:
                use = dm if (v == "JM" or sig) else 0.0
                m1 = mj * np.exp(use)
                # var of J*exp(d): (exp(d))^2 var(J) + (J exp(d))^2 var(d)
                v1 = np.exp(use) ** 2 * vj + (m1 ** 2) * (dv if use != 0 else 0.0)
                b = (mj, vj, m1, v1)
            base[tk][v][a] = b


def matched(victim, socks, co_vals, tk, v):
    b = base[tk][v][victim]
    socks = np.asarray(socks)
    w1 = socks.mean()
    mb = (1 - w1) * b[0] + w1 * b[2]
    n0, n1 = (socks == 0).sum(), (socks == 1).sum()
    # variance of the mixture mean: weights are the fraction of reps per socket
    vb = (1 - w1) ** 2 * b[1] + w1 ** 2 * b[3]
    x = np.asarray(co_vals, float)
    raw = x.mean() / mb
    rel = np.sqrt((x.var(ddof=1) / len(x)) / x.mean() ** 2 + vb / mb ** 2)
    return raw, raw * rel


# ------------------------------------------------------------------ App-App outcomes (8 node)
aa = read(PROCESSED / "data_coscheduled.csv")
aa8 = aa[aa["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"})
sb = pd.read_csv(OUT_DIR / "socket_app_app.csv")
aa8 = aa8.merge(sb, on=["A", "B", "rep"], how="left")
assert aa8[aa8.A != aa8.B].sock_A.notna().all()
recs = []
for (a, b), g in aa8[aa8.A != aa8.B].groupby(["A", "B"]):
    for end, victim, aggr in (("A", a, b), ("B", b, a)):
        socks = g[f"sock_{end}"].astype(int).to_numpy()
        for tk in TARGETS:
            colname = f"App {end} MPI Time Average" if tk == "MPI" else f"App {end} Time Average"
            for v in VARIANTS:
                raw, se = matched(victim, socks, g[colname], tk, v)
                recs.append(dict(victim=victim, aggressor=aggr, target=tk, variant=v, raw=raw, se=se,
                                 frac_sock1=socks.mean()))
AA = pd.DataFrame(recs)
AA["slowdown"] = AA.raw.clip(lower=1)
AA.to_csv(OUT_DIR / "socket_matched_appapp.csv", index=False)

# ------------------------------------------------------------------ App-Inhibitor configs (8 node)
ai = read(PROCESSED / "data_inhib_coscheduled.csv")
ai = ai[ai["Num Nodes"] == 8].rename(columns={"Inhib Message Size": "m", "Inhib Wait Time (us)": "w",
                                              "Inhib Comm Sparsity": "sp", "Run Iteration": "rep"})
ai["sp"] = ai.sp.round(9)
sa = pd.read_csv(OUT_DIR / "socket_app_inhibitor.csv")
ai = ai.merge(sa, on=["App", "m", "w", "sp", "rep"], how="inner")
recs = []
for (a, m, w, sp), g in ai.groupby(["App", "m", "w", "sp"]):
    socks = g.app_sock.astype(int).to_numpy()
    for tk, (col, _) in TARGETS.items():
        for v in VARIANTS:
            raw, se = matched(a, socks, g[col], tk, v)
            recs.append(dict(App=a, m=m, w=w, sp=sp, target=tk, variant=v, raw=raw, se=se, nrep=len(g)))
AI = pd.DataFrame(recs)
AI["slowdown"] = AI.raw.clip(lower=1)
AI.to_csv(OUT_DIR / "socket_matched_appinhib.csv", index=False)

# ------------------------------------------------------------------ S5 summary
p("## S5. Socket-matched co-run slowdowns")
p("Baselines: **J** = June isolated mean (chapter's current numbers); **N0** = new pinned sock0 for all reps "
  "(requested 'before'); **NM** = new pinned baseline of the socket each rep's victim ran on (requested 'after'); "
  "**JM** = June mean, times exp(mean paired isolated d) for socket-1 reps (socket matching without the S3 "
  "launcher/epoch drift); **JMs** = JM with d set to 0 unless its 95% CI excludes 0 (T_MPI: AMG2023 and "
  "Quicksilver only; T_app: AMG2023 and Quicksilver). Slowdown = max(1, mean co / mean matched baseline). "
  "SE = delta-method SE of the raw ratio (co reps + baseline reps).")
p("Effect of socket matching alone: compare N0 -> NM (new baselines) or J -> JM / JMs (June baselines).")


def summ(d):
    s = d.slowdown
    return {"n": len(d), "mean": s.mean(), "median": s.median(), "p90": s.quantile(0.9), "max": s.max(),
            "frac raw<1": (d.raw < 1).mean(), "frac >=1.05": (s >= 1.05).mean(), "frac >=1.10": (s >= 1.10).mean(),
            "median SE": d.se.median(), "p90 SE": d.se.quantile(0.9)}


for setname, D in (("App-App, non-self directional outcomes", AA), ("App-Inhibitor configs", AI)):
    for tk in TARGETS:
        t = pd.DataFrame({v: summ(D[(D.target == tk) & (D.variant == v)]) for v in VARIANTS}).T
        t.index.name = "baseline"
        p()
        p(f"### {setname}, T_{tk} slowdown")
        p(md(t))
p()
# check vs chapter
chk = AA[(AA.target == "MPI") & (AA.variant == "J")]
p(f"Check: J reproduces chapter B2/A6 (App-App MPI mean {chk.slowdown.mean():.3f}, median {chk.slowdown.median():.3f}, "
  f"max {chk.slowdown.max():.3f}, floor {(chk.raw <= 1).mean():.3f}; App-Inh MPI floor "
  f"{(AI[(AI.target == 'MPI') & (AI.variant == 'J')].raw <= 1).mean():.3f}, n = "
  f"{len(AI[(AI.target == 'MPI') & (AI.variant == 'J')])}).")
p()

# per-victim view for App-App MPI
t = AA[AA.target == "MPI"].pivot_table(index="victim", columns="variant", values="slowdown", aggfunc="median")[VARIANTS]
t["frac reps on sock1"] = AA[(AA.target == "MPI") & (AA.variant == "J")].groupby("victim").frac_sock1.mean()
t.index = [NAME[a] for a in t.index]
t.index.name = "victim"
p("Per-victim median App-App MPI slowdown by baseline:")
p(md(t, "{:.3f}"))
p()

# ------------------------------------------------------------------ S6 re-scoring
p("## S6. Re-scoring frozen predictions against socket-matched App-App outcomes (no re-fit)")
p("Predictions: random_split, non-self outcomes (90 directional, 45 pairs). Only the truth changes; predicted log "
  "slowdowns are as saved. Δ% = log MAE vs the isolated-speed baseline (constant 1); CI = paired pair-cluster "
  "bootstrap over the 45 non-self pairs (1,000 draws, seed 923, one shared draw matrix; re-implemented here). "
  "Pooled skill = 1 − MAE(method)/MAE(baseline). FB = false-benign rate at 1.05 over truly harmful pairs (pair "
  "worst endpoint, as in metrics.py).")

SOURCES = {
    "MPI": (REPO / "notebooks/utils/delta_response_filteredInhibs_totalTime/outputs/filtered_fit_mpi_time/"
            "directional_predictions.csv",
            {"constant_1": "isolated-speed baseline", "absolute_response": "absolute",
             "generic_potential": "generic", "delta_median": "delta (median)†",
             "delta_ood_kernel": "delta (OOD kernel)†", "delta_uniform": "delta (uniform)†"}),
    "app": (REPO / "notebooks/utils/delta_response_filteredInhibs_biDirectional_totalTime/outputs/full_comparison/"
            "directional_predictions.csv",
            {"constant_1": "isolated-speed baseline", "legacy_absolute": "absolute",
             "legacy_generic_uniform": "generic", "legacy_delta_median": "delta (median)",
             "legacy_delta_ood_kernel": "delta (OOD kernel)", "legacy_delta_uniform": "delta (uniform)",
             "joint_profile_direct": "joint (direct)", "joint_profile_calibrated": "joint (calibrated)",
             "bidirectional_profile_wall_calibrated": "bidir profile-wall (cal.)"}),
}
p("MPI predictions: one-way MPI pipeline on the canonical snapshot, local `filtered_fit_mpi_time` run. Its "
  "constant-1, absolute and generic log MAE equal the artifact's control run (baseSnap) exactly; † its delta "
  "models use a different recipe from the artifact (artifact delta median 0.1235 vs local 0.1190), so treat "
  "them as indicative. The joint MPI model's per-outcome predictions are not on this machine (bidirectional MPI "
  "root missing), so joint is scored on total time only.")
p("Total-time predictions: local bidirectional total-time `full_comparison` run, identical to the artifact's "
  "bidirTT run.")

rng = np.random.default_rng(923)
for tk, (path, methods) in SOURCES.items():
    pr = pd.read_csv(path)
    pr = pr[(pr.evaluation_method == "random_split") & (pr.victim_id != pr.aggressor_id) & pr.method.isin(methods)]
    truth = AA[AA.target == tk].pivot_table(index=["victim", "aggressor"], columns="variant", values="slowdown")
    pr = pr.join(truth, on=["victim_id", "aggressor_id"])
    dev = np.abs(pr.true_slowdown - pr["J"]).max()
    pair_ids = np.sort(pr.pair_row_id.unique())
    draws = np.random.default_rng(923).choice(pair_ids, size=(1000, len(pair_ids)), replace=True)
    p()
    p(f"### T_{tk} target (check: max |saved true slowdown − J truth| = {dev:.1e})")
    rows = []
    for v in VARIANTS:
        lt = np.log(pr[v])
        pr["err"] = np.abs(pr.predicted_log_slowdown - lt)
        per_pair = pr.pivot_table(index="pair_row_id", columns="method", values="err", aggfunc="sum")
        base_mae = pr[pr.method == "constant_1"].err.mean()
        cnt = pr[pr.method == "constant_1"].groupby("pair_row_id").size().reindex(pair_ids).to_numpy()
        for mth, lab in methods.items():
            g = pr[pr.method == mth]
            mae = g.err.mean()
            # bootstrap Δ%
            em = per_pair[mth].reindex(pair_ids).to_numpy()
            eb = per_pair["constant_1"].reindex(pair_ids).to_numpy()
            idx = np.searchsorted(pair_ids, draws)
            bm = em[idx].sum(1) / cnt[idx].sum(1)
            bb = eb[idx].sum(1) / cnt[idx].sum(1)
            dpct = 100 * (bm / bb - 1)
            # false benign
            gg = g.assign(lt=lt[g.index]).pivot_table(index="pair_row_id", columns="direction",
                                                      values=["lt", "predicted_log_slowdown"])
            tw = gg["lt"].max(axis=1)
            pw = gg["predicted_log_slowdown"].max(axis=1)
            harm = tw >= np.log(1.05)
            fb = ((pw < np.log(1.05)) & harm).sum() / harm.sum() if harm.sum() else np.nan
            rows.append({"baseline": v, "method": lab, "log MAE": mae,
                         "Δ% vs isolated-speed": 100 * (mae / base_mae - 1),
                         "CI lo": np.percentile(dpct, 2.5), "CI hi": np.percentile(dpct, 97.5),
                         "pooled skill": 1 - mae / base_mae, "FB@1.05": fb,
                         "harmful pairs": f"{int(harm.sum())}/45"})
    R = pd.DataFrame(rows)
    for mth in methods.values():
        t = R[R.method == mth].set_index("baseline").drop(columns="method")
        p()
        p(f"**{mth}**")
        p(md(t))
    R.to_csv(OUT_DIR / f"socket_matched_rescore_{tk}.csv", index=False)

(OUT_DIR / "socket_matched_numbers.md").write_text(log.getvalue())
# merge into socket_numbers.md (written by analysis_socket_pinned.py; run that first)
sn = OUT_DIR / "socket_numbers.md"
head = sn.read_text().split("## S5.")[0].rstrip() + "\n\n"
sn.write_text(head + log.getvalue())
print("wrote", OUT_DIR / "socket_matched_numbers.md")
