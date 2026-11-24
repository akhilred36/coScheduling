"""Plan v2: build the socket-resolved, unclipped prediction tables, write the export bundle, and report facts A1-A3.

All targets are UNCLIPPED log ratios ln(T_co / T_base), where T_base is the JMs baseline: the June 8-node isolated mean
x exp(d_app) for socket-1 replicates, with d_app (the mean paired pinned log(sock1/sock0), output/socket_pinned_pairs.csv)
applied only where its 95% t CI excludes 0, separately for each timer (same rule as analysis_socket_matched.py). A cell
mixing sockets uses the replicate-weighted mixture of the two baselines (ratio of means).

Tables written to the bundle (default scripts/dissertation_figs/agera_bundle/):
  apps.csv         isolated base profile per app (mpi_time, comm_frac, total_msgs, total_bytes) from canonical jobs.csv
  panel_inhibitors.csv  isolated profile of the 9 panel configs (canonical inhibitors.csv)
  panel_app.csv    per (app, panel config): rep-averaged JMs log ratio, y_mpi / y_tt (victim = app), and the
                   inhibitor's slowdown caused by the app, c_mpi / c_tt (BD features; see below)
  appinh_resolved.csv  per (app, panel config, app socket s_V): JMs log ratios (abs/delta training rows)
  appapp_rows.csv  per (victim, corunner, s_V): JMs log ratios. Non-self cells are socket-resolved. Self-pairs
                   (socket of each copy is not recoverable) are one row with s_V = 0.5, pooling both copies x 4 reps,
                   baseline = 50/50 mixture.
  jms_factors.csv  d_app, CI, applied flag per timer
  manifest.json    row counts + sha256 per file

BD "inhibitor slowdown caused by app X" (the inhibitor runs a fixed -r duration, so its time is not an outcome; the
slowdown is its time per unit of work):
  c_mpi = ln[(T_MPI/bytes)_co / (T_MPI/bytes)_iso]   (= inhib_bw_mpi_log_degradation of the canonical BD snapshot)
  c_tt  = ln[(T_app/msgs)_co / (T_app/msgs)_iso]     (= inhib_msg_wall_log_degradation)
Both are recomputed here from the processed replicates (ratio of aggregate means) and checked against the BD snapshot.

Run: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python scripts/dissertation_figs/v2/v2_build.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APPS, BIDIR, BIDIR_TT, NAME, OUT_DIR, PROCESSED, read  # noqa: E402

PANEL = ["8_2000000_1000000_0.3", "8_20000_100000_0.1", "8_200_200000_0.002237136",
         "8_2500000_200000_0.002237136", "8_2500000_600000_0.002237136", "8_400000_1000000_0.5",
         "8_40000_1000000_0.1", "8_40000_1000000_0.7", "8_500_5000_0.05"]
TIMERS = {"mpi": ("MPI Time Average", "dMPI"), "tt": ("App Time Average", "dApp")}

log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.4f}", index=True) -> str:
    df = df if index else df.reset_index(drop=True)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        cells = ([str(i)] if index else []) + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                               for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cfg_id(m, w, sp) -> str:
    sp = float(sp)
    s = "1" if sp == 1.0 else f"{sp:.9f}".rstrip("0").rstrip(".")
    return f"8_{int(m)}_{int(w)}_{s}"


# ------------------------------------------------------------------ JMs baselines
def jms_factors() -> pd.DataFrame:
    iso = read(PROCESSED / "data_isolated.csv")
    j8 = iso[iso["Num Nodes"] == 8]
    pairs = pd.read_csv(OUT_DIR / "socket_pinned_pairs.csv")
    rows = []
    for a in APPS:
        for tk, (col, dcol) in TIMERS.items():
            d = pairs.loc[pairs.App == a, dcol].to_numpy(float)
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            h = stats.t.ppf(0.975, len(d) - 1) * se
            rows.append(dict(app=a, timer=tk, june_mean=j8.loc[j8.App == a, col].mean(),
                             n_june=int((j8.App == a).sum()), d=m, ci_lo=m - h, ci_hi=m + h, n_pairs=len(d),
                             applied=bool(abs(m) > h)))
    f = pd.DataFrame(rows)
    f["d_used"] = np.where(f.applied, f.d, 0.0)
    return f


def base_of(f: pd.DataFrame):
    t = f.set_index(["app", "timer"])
    return lambda app, tk, s: t.loc[(app, tk), "june_mean"] * np.exp(t.loc[(app, tk), "d_used"] * s)


def ratio_jms(B, app, tk, socks, co_vals) -> float:
    """mean(T_co) / mean of the per-replicate matched baselines (s may be 0, 1, or 0.5 for pooled self copies)."""
    socks = np.asarray(socks, float)
    return float(np.mean(co_vals) / np.mean([B(app, tk, s) for s in socks]))


def build():
    f = jms_factors()
    B = base_of(f)

    # ---------------- App-Inhibitor replicates with app socket (panel configs only)
    ai = read(PROCESSED / "data_inhib_coscheduled.csv")
    ai = ai[ai["Num Nodes"] == 8].rename(columns={"Inhib Message Size": "m", "Inhib Wait Time (us)": "w",
                                                  "Inhib Comm Sparsity": "sp", "Run Iteration": "rep"})
    ai["sp"] = ai.sp.round(9)
    ai["config"] = [cfg_id(*t) for t in zip(ai.m, ai.w, ai.sp)]
    sa = pd.read_csv(OUT_DIR / "socket_app_inhibitor.csv")
    sa["sp"] = sa.sp.round(9)
    ai = ai.merge(sa, on=["App", "m", "w", "sp", "rep"], how="left", validate="one_to_one")
    assert ai.app_sock.notna().all()
    full_ai = ai.copy()
    ai = ai[ai.config.isin(PANEL)]
    assert ai.groupby(["App", "config"]).ngroups == 90, ai.groupby(["App", "config"]).ngroups

    # isolated inhibitor aggregates (BD features)
    ii = read(PROCESSED / "data_inhib_isolated.csv")
    ii = ii[ii["Num Nodes"] == 8].rename(columns={"Inhib Message Size": "m", "Inhib Wait Time (us)": "w",
                                                  "Inhib Comm Sparsity": "sp"})
    ii["config"] = [cfg_id(*t) for t in zip(ii.m, ii.w, ii.sp.round(9))]
    ii.columns = [c.replace("Inhib ", "") if c.startswith("Inhib ") and c not in ("Inhib Message Size",) else c for c in ii.columns]
    iiso = ii.groupby("config")[["MPI Time Average", "App Time Average", "Total Messages Sent",
                                 "Total Bytes Sent"]].mean()

    panel, resolved = [], []
    for (a, c), g in ai.groupby(["App", "config"]):
        r = dict(app=a, config=c, n_reps=len(g), frac_sock1=g.app_sock.mean())
        for tk, (col, _) in TIMERS.items():
            raw = ratio_jms(B, a, tk, g.app_sock, g[col])
            r[f"raw_{tk}"], r[f"y_{tk}"] = raw, np.log(raw)
            r[f"rawJ_{tk}"] = g[col].mean() / B(a, tk, 0)
        iso_c = iiso.loc[c]
        r["c_mpi"] = np.log((g["Inhib MPI Time Average"].mean() / g["Inhib Total Bytes Sent"].mean())
                            / (iso_c["MPI Time Average"] / iso_c["Total Bytes Sent"]))
        r["c_tt"] = np.log((g["Inhib App Time Average"].mean() / g["Inhib Total Messages Sent"].mean())
                           / (iso_c["App Time Average"] / iso_c["Total Messages Sent"]))
        panel.append(r)
        for s, gs in g.groupby("app_sock"):
            rr = dict(app=a, config=c, s_V=int(s), n_reps=len(gs))
            for tk, (col, _) in TIMERS.items():
                raw = ratio_jms(B, a, tk, gs.app_sock, gs[col])
                rr[f"raw_{tk}"], rr[f"y_{tk}"] = raw, np.log(raw)
            resolved.append(rr)
    panel, resolved = pd.DataFrame(panel), pd.DataFrame(resolved)

    # ---------------- App-App replicates
    cc = read(PROCESSED / "data_coscheduled.csv")
    cc = cc[cc["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"})
    sb = pd.read_csv(OUT_DIR / "socket_app_app.csv")
    cc = cc.merge(sb, on=["A", "B", "rep"], how="left", validate="one_to_one")
    recs = []
    for r in cc.to_dict("records"):
        for v, x in (("A", "B"), ("B", "A")):
            self_ = r["A"] == r["B"]
            recs.append(dict(victim=r[v], corunner=r[x], rep=r["rep"], is_self=self_,
                             s_V=0.5 if self_ else int(r[f"sock_{v}"]),
                             T_mpi=r[f"App {v} MPI Time Average"], T_tt=r[f"App {v} Time Average"]))
    d = pd.DataFrame(recs)
    assert d[~d.is_self].s_V.isin([0, 1]).all()
    rows = []
    for (v, x, s), g in d.groupby(["victim", "corunner", "s_V"]):
        r = dict(victim=v, corunner=x, s_V=s, is_self=bool(g.is_self.iloc[0]), n_obs=len(g))
        for tk in TIMERS:
            raw = ratio_jms(B, v, tk, g.s_V, g[f"T_{tk}"])
            r[f"raw_{tk}"], r[f"y_{tk}"] = raw, np.log(raw)
        rows.append(r)
    rows = pd.DataFrame(rows)
    rows["pair"] = ["|".join(sorted(t)) for t in zip(rows.victim, rows.corunner)]

    # rep-averaged directed outcomes (for characterization and B2-style aggregation)
    dirm = []
    for (v, x), g in d.groupby(["victim", "corunner"]):
        r = dict(victim=v, corunner=x, is_self=bool(g.is_self.iloc[0]), n_obs=len(g), frac_sock1=g.s_V.mean())
        for tk in TIMERS:
            raw = ratio_jms(B, v, tk, g.s_V, g[f"T_{tk}"])
            r[f"raw_{tk}"], r[f"y_{tk}"] = raw, np.log(raw)
            r[f"rawJ_{tk}"] = g[f"T_{tk}"].mean() / B(v, tk, 0)
        dirm.append(r)
    dirm = pd.DataFrame(dirm)

    # full-grid App-Inh rep-averaged (characterization only)
    grid = []
    for (a, c), g in full_ai.groupby(["App", "config"]):
        r = dict(app=a, config=c, m=g.m.iloc[0], w=g.w.iloc[0], sp=g.sp.iloc[0], n_reps=len(g),
                 in_panel=c in PANEL, frac_sock1=g.app_sock.mean())
        for tk, (col, _) in TIMERS.items():
            raw = ratio_jms(B, a, tk, g.app_sock, g[col])
            r[f"raw_{tk}"], r[f"y_{tk}"] = raw, np.log(raw)
            r[f"rawJ_{tk}"] = g[col].mean() / B(a, tk, 0)
        grid.append(r)
    grid = pd.DataFrame(grid)

    apps = read(BIDIR / "data/training/jobs.csv").set_index("job_id").loc[APPS].reset_index()
    apps = apps.rename(columns={"job_id": "app"})[["app", "mpi_time", "comm_frac", "total_msgs", "total_bytes",
                                                   "runtime"]]
    inh = read(BIDIR / "data/training/inhibitors.csv").set_index("inhib_id").loc[PANEL].reset_index()
    inh = inh.rename(columns={"inhib_id": "config"})[["config", "mpi_time", "comm_frac", "total_msgs",
                                                      "total_bytes", "runtime", "msg_size", "wait_time",
                                                      "comm_sparsity"]]
    return f, panel, resolved, rows, dirm, grid, apps, inh


def facts(f, panel, resolved, rows, dirm, grid):
    p("# v2 facts A1-A3 (plan v2, generated by v2/v2_build.py)\n")
    # ---------------- A2: JMs factors + raw ratio availability
    p("## A2. JMs baseline factors (d = mean paired pinned ln(T_sock1/T_sock0); applied iff 95% t CI excludes 0)\n")
    t = f.copy()
    t["app"] = t.app.map(NAME)
    p(md(t[["app", "timer", "june_mean", "d", "ci_lo", "ci_hi", "n_pairs", "applied"]], index=False))
    p(f"\nApplied: MPI -> {[NAME[a] for a in f[(f.timer == 'mpi') & f.applied].app]}; "
      f"app time -> {[NAME[a] for a in f[(f.timer == 'tt') & f.applied].app]}\n")

    # cross-check vs analysis_socket_matched (JMs, rep-averaged directed outcomes)
    sm = pd.read_csv(OUT_DIR / "socket_matched_appapp.csv")
    sm = sm[sm.variant == "JMs"].pivot_table(index=["victim", "aggressor"], columns="target", values="raw")
    chk = dirm[~dirm.is_self].set_index(["victim", "corunner"])
    e1 = np.abs(chk.raw_mpi.values - sm.loc[chk.index, "MPI"].values).max()
    e2 = np.abs(chk.raw_tt.values - sm.loc[chk.index, "app"].values).max()
    sa = pd.read_csv(OUT_DIR / "socket_matched_appinhib.csv")
    sa = sa[sa.variant == "JMs"]
    sa["config"] = [cfg_id(*x) for x in zip(sa.m, sa.w, sa.sp.round(9))]
    sa = sa.pivot_table(index=["App", "config"], columns="target", values="raw")
    pc = panel.set_index(["app", "config"])
    e3 = max(np.abs(pc.raw_mpi.values - sa.loc[pc.index, "MPI"].values).max(),
             np.abs(pc.raw_tt.values - sa.loc[pc.index, "app"].values).max())
    p(f"Cross-check vs output/socket_matched_*.csv (JMs): max |diff| App-App MPI {e1:.1e}, app time {e2:.1e}; "
      f"panel {e3:.1e}.")
    # vs canonical unclipped ratios (J baseline)
    pm = read(BIDIR / "data/evaluation/pair.csv")
    pt = read(BIDIR_TT / "data/evaluation/pair.csv")
    can = {}
    for frame, col, tk in ((pm, "raw_mpi_time_ratio", "mpi"), (pt, "raw_app_time_ratio", "tt")):
        for r in frame.to_dict("records"):
            if r["jobA_id"] != r["jobB_id"]:
                can[(r["jobA_id"], r["jobB_id"], tk)] = r[f"{col}_A"]
                can[(r["jobB_id"], r["jobA_id"], tk)] = r[f"{col}_B"]
    e4 = max(abs(r[f"rawJ_{tk}"] - can[(r["victim"], r["corunner"], tk)])
             for r in dirm[~dirm.is_self].to_dict("records") for tk in TIMERS)
    ji = read(BIDIR_TT / "data/training/job_inh.csv").set_index(["job_id", "inhib_id"])
    e5 = max(np.abs(pc.rawJ_mpi.values - ji.loc[pc.index, "raw_mpi_time_ratio"].values).max(),
             np.abs(pc.rawJ_tt.values - ji.loc[pc.index, "raw_app_time_ratio"].values).max())
    p(f"Cross-check of the June-baseline (J) ratios vs canonical raw_*_ratio: App-App max |diff| {e4:.1e}; "
      f"panel {e5:.1e}.\n")

    p("Unclipped ratio availability (finite, > 0):\n")
    av = pd.DataFrame({
        "table": ["App-App resolved rows", "App-App directed (rep-avg)", "panel app x config", "panel resolved",
                  "full-grid App-Inh"],
        "rows": [len(rows), len(dirm), len(panel), len(resolved), len(grid)],
        "MPI ok": [int(np.isfinite(x.y_mpi).sum()) for x in (rows, dirm, panel, resolved, grid)],
        "app ok": [int(np.isfinite(x.y_tt).sum()) for x in (rows, dirm, panel, resolved, grid)],
        "MPI ratio < 1": [int((x.raw_mpi < 1).sum()) for x in (rows, dirm, panel, resolved, grid)],
        "app ratio < 1": [int((x.raw_tt < 1).sum()) for x in (rows, dirm, panel, resolved, grid)]})
    p(md(av, index=False))

    # ---------------- A3: BD columns
    p("\n## A3. BD panel columns (inhibitor slowdown caused by each app)\n")
    p("**BD feature definitions (confirmed by the chapter director 2026-11-22).** The inhibitor runs for a fixed "
      "duration (-r), so its slowdown is its time per unit of work, co-run vs isolated, ratio of aggregate means, "
      "no JMs correction on the inhibitor side (S2: inhibitor socket effect ~0):\n")
    p("- MPI timer: c_mpi = ln[(T_MPI / bytes sent)_co / (T_MPI / bytes sent)_iso]  (MPI time per byte)")
    p("- app timer: c_tt = ln[(T_app / messages sent)_co / (T_app / messages sent)_iso]  (app time per message)\n")
    bd = read(BIDIR / "data/training/job_inh.csv").set_index(["job_id", "inhib_id"])
    bdt = read(BIDIR_TT / "data/training/job_inh.csv").set_index(["job_id", "inhib_id"])
    v1 = bd.loc[pc.index]
    v2 = bdt.loc[pc.index]
    p(f"c_mpi = ln[(T_MPI/bytes)_co/(T_MPI/bytes)_iso] (inhibitor side): finite for {int(np.isfinite(pc.c_mpi).sum())}/90; "
      f"BD snapshot inhib_bw_mpi_valid {int(v1.inhib_bw_mpi_valid.sum())}/90; max |mine - BD| "
      f"{np.nanmax(np.abs(pc.c_mpi.values - v1.inhib_bw_mpi_log_degradation.values)):.1e}")
    p(f"c_tt = ln[(T_app/msgs)_co/(T_app/msgs)_iso]: finite for {int(np.isfinite(pc.c_tt).sum())}/90; "
      f"BD-TT snapshot inhib_msg_wall_valid {int(v2.inhib_msg_wall_valid.sum())}/90; max |mine - BD| "
      f"{np.nanmax(np.abs(pc.c_tt.values - v2.inhib_msg_wall_log_degradation.values)):.1e}")
    q = panel[["c_mpi", "c_tt"]].describe(percentiles=[0.5]).T[["min", "50%", "mean", "max"]]
    p("\n" + md(q))

    # ---------------- A1: counts
    p("\n## A1. Row counts\n")
    ns = rows[~rows.is_self]
    p(f"App-App: 55 unordered co-runs (45 non-self + 10 self) = 100 directed outcomes (90 non-self + 10 self).")
    p(f"Socket-resolved rows: {len(ns)} non-self cells (s_V=0: {int((ns.s_V == 0).sum())}, s_V=1: "
      f"{int((ns.s_V == 1).sum())}; one-rep cells {int((ns.n_obs == 1).sum())}) + {int(rows.is_self.sum())} self rows "
      f"(s_V = 0.5, pooled copies) = {len(rows)} rows.")
    cps = ns.groupby("pair").size()
    p(f"Resolved cells per non-self unordered pair: {cps.value_counts().sort_index().to_dict()} "
      f"(2 = all 4 reps had the same assignment).")
    p(f"Panel: 90 app x config cells; resolved by app socket {len(resolved)} (one-rep cells "
      f"{int((resolved.n_reps == 1).sum())}).\n")

    # full data: 80/20 of 55 unordered co-runs
    n_test = round(0.2 * 55)
    p(f"**Full data** (per seed): 55 unordered co-runs -> {55 - n_test} train / {n_test} test (pair-level, both "
      f"directions and all socket cells together). Resolved rows per split depend on the draw; over 10,000 random draws:")
    rng = np.random.default_rng(0)
    per_pair = rows.groupby("pair").size()
    pairs = per_pair.index.to_numpy()
    te = [per_pair.loc[rng.choice(pairs, n_test, replace=False)].sum() for _ in range(10000)]
    p(f"test rows mean {np.mean(te):.1f}, min {min(te)}, max {max(te)}; train rows = {len(rows)} - test.\n")

    recs = []
    for held in combinations(APPS, 3):
        h = set(held)
        nk = rows.victim.isin(h).astype(int) + rows.corunner.isin(h).astype(int)
        nk = np.where(rows.is_self, rows.victim.isin(h) * 2, nk)  # self row: both endpoints held out or not
        tr, ok, zs = rows[nk == 0], rows[nk == 1], rows[nk == 2]
        recs.append(dict(train=len(tr), one_known=len(ok), zero_shot=len(zs),
                         train_dir=len(tr.groupby(["victim", "corunner"])),
                         ok_dir=len(ok.groupby(["victim", "corunner"])),
                         zs_dir=len(zs.groupby(["victim", "corunner"])),
                         abs_train=int(resolved.app.isin(set(APPS) - h).sum())))
    R = pd.DataFrame(recs)
    p("**Restricted tiers** (120 = C(10,3) held-out triples; train = rows among the 7 known apps):\n")
    s = R.describe().T[["mean", "min", "max"]]
    s.index = ["train rows (resolved)", "one-known test rows (resolved)", "zero-shot test rows (resolved)",
               "train directed outcomes", "one-known directed", "zero-shot directed",
               "abs/delta training rows (resolved panel cells of 7 apps)"]
    p(md(s, "{:.1f}"))
    p(f"\nDirected counts are constant: train {R.train_dir.iloc[0]} (42 non-self + 7 self), one-known "
      f"{R.ok_dir.iloc[0]} (= 3 x 7 x 2), zero-shot {R.zs_dir.iloc[0]} (6 non-self + 3 self).")
    return R


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1] / "agera_bundle")
    args = ap.parse_args()
    f, panel, resolved, rows, dirm, grid, apps, inh = build()
    facts(f, panel, resolved, rows, dirm, grid)

    # characterization tables (local only)
    rows.to_csv(OUT_DIR / "v2_appapp_rows.csv", index=False)
    dirm.to_csv(OUT_DIR / "v2_appapp_directed.csv", index=False)
    grid.to_csv(OUT_DIR / "v2_appinh_grid.csv", index=False)
    panel.to_csv(OUT_DIR / "v2_panel_app.csv", index=False)
    (OUT_DIR / "v2_A_facts.md").write_text(log.getvalue())

    b = args.bundle
    b.mkdir(parents=True, exist_ok=True)
    files = {"apps.csv": apps, "panel_inhibitors.csv": inh,
             "panel_app.csv": panel.drop(columns=["rawJ_mpi", "rawJ_tt"]),
             "appinh_resolved.csv": resolved, "appapp_rows.csv": rows, "jms_factors.csv": f}
    man = {"created_by": "scripts/dissertation_figs/v2/v2_build.py", "panel": PANEL, "apps": APPS, "files": {}}
    for name, frame in files.items():
        frame.to_csv(b / name, index=False, float_format="%.12g")
        man["files"][name] = {"rows": len(frame), "sha256": sha(b / name)}
    (b / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"bundle -> {b}")


if __name__ == "__main__":
    main()
