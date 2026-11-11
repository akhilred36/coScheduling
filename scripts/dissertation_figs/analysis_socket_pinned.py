"""Socket-pinned isolated re-runs (experiment 20261107_180934): socket sensitivity in isolation.

S1  paired within-job socket difference d = log(sock1) - log(sock0) per app, plus order effect
S2  same for the isolated inhibitor runs (only if their data is present)
S3  launcher/epoch drift: new sock0 vs June isolated baselines (NOT a socket effect)
S4  isolated socket difference next to the co-run socket difference (from figs_batch1 C8 logic)
F-S1 socket_isolated.pdf

Writes output/socket_numbers.md and output/socket_pinned_runs.csv.
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import APPS, FULL_W, NAME, OI, OUT_DIR, PROCESSED, REPO, read, save, set_style

sys.path.insert(0, str(REPO / "notebooks"))
from utils.parse_coscheduling_runs import CoSchedulingRunsParser  # noqa: E402

EXP = REPO / "run_data/experiments_coscheduling_isolated_socket/20261107_180934"
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


# ------------------------------------------------------------------ placement + validity
def check_run(d: Path) -> dict:
    out = dict(run=d.name, socket_dir=d.parent.name)
    out["failed"] = (d / "PLACEMENT_FAILED").exists()
    out["n_mpip"] = len(list((d / "mpip_profiles").glob("*.mpiP"))) if (d / "mpip_profiles").exists() else 0
    chk = (d / "placement_check.txt")
    out["check"] = chk.read_text().strip() if chk.exists() else ""
    out["problems0"] = "problems=0" in out["check"]
    probe = d / "placement_probe.log"
    if probe.exists():
        lines = [l for l in probe.read_text().splitlines() if l.startswith("PLACEMENT ")]
        kv = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in lines]
        pus = np.array([int(k["pu"]) for k in kv])
        out["n_ranks"] = len(kv)
        out["threads"] = ",".join(sorted({k["thread"] for k in kv}))
        out["sockets_from_pu"] = ",".join(map(str, sorted(set((pus % 112 >= 56).astype(int)))))
        out["pu_min"], out["pu_max"] = int(pus.min()), int(pus.max())
    else:
        out["n_ranks"] = 0
    exp_sock = d.parent.name[-1]
    out["valid"] = (not out["failed"] and out["n_mpip"] > 0 and out["problems0"] and out.get("n_ranks") == 448
                    and out.get("threads") == "0" and out.get("sockets_from_pu") == exp_sock)
    return out


def parse_root(root: Path, valid_runs: set[str], kind: str) -> pd.DataFrame:
    """parse_isolated raises on dirs without .mpiP, so parse a symlink tree of valid dirs only."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in valid_runs:
            (Path(tmp) / name).symlink_to(root / name)
        parser = CoSchedulingRunsParser()
        if kind == "app":
            return parser.parse_isolated(tmp, progress=False)
        return parser.parse_inhibitor_isolated(tmp, progress=False)


manifest = json.loads((EXP / "manifest.json").read_text())
order = {j["run"]: j["socket_order"][0] for j in manifest["jobs"]}  # socket that ran first

p("# Socket-pinned isolated runs: experiment 20261107_180934")
p()
p("Placement: `srun --whole --mpibind=off --cpu-bind=none` + `numactl --physcpubind --localalloc`, 56 ranks/node "
  "on SMT thread 0 (sock0 = PUs 0-55, sock1 = PUs 56-111). Each job ran its sock0 and sock1 steps back to back "
  "on the same 8 nodes; even replicates sock0 first, odd replicates sock1 first.")
p()

checks = []
for sub in ("data", "data_inhib"):
    for s in ("sock0", "sock1"):
        root = EXP / sub / s
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir():
                    checks.append(check_run(d) | dict(kind="app" if sub == "data" else "inhib"))
chk = pd.DataFrame(checks)
p("## Run validity")
for (kind, s), g in chk.groupby(["kind", "socket_dir"]):
    p(f"- {kind} {s}: {len(g)} run dirs, valid {int(g.valid.sum())}, PLACEMENT_FAILED {int(g.failed.sum())}, "
      f"no .mpiP {int((g.n_mpip == 0).sum())}, missing placement_check {int((g.check == '').sum())}")
va = chk[(chk.kind == "app") & chk.valid]
if len(va):
    p(f"- valid app runs: threads seen {sorted(set(va.threads))}; PU range sock0 "
      f"{va[va.socket_dir == 'sock0'].pu_min.min()}-{va[va.socket_dir == 'sock0'].pu_max.max()}, sock1 "
      f"{va[va.socket_dir == 'sock1'].pu_min.min()}-{va[va.socket_dir == 'sock1'].pu_max.max()}; "
      f"all {int(va.n_ranks.min())}-{int(va.n_ranks.max())} ranks.")
p()

rows = []
for s in (0, 1):
    v = set(chk[(chk.kind == "app") & (chk.socket_dir == f"sock{s}") & chk.valid].run)
    if v:
        df = parse_root(EXP / "data" / f"sock{s}", v, "app")
        df["Socket"] = s
        rows.append(df)
new = pd.concat(rows, ignore_index=True)
new["run"] = "8_" + new.App + "_" + new["Run Iteration"].astype(str)
new["first_socket"] = new.run.map(order)
new.to_csv(OUT_DIR / "socket_pinned_runs.csv", index=False)

wide = new.pivot_table(index=["App", "Run Iteration"], columns="Socket",
                       values=["MPI Time Average", "App Time Average"])
wide = wide.dropna()
pairs = pd.DataFrame({
    "dMPI": np.log(wide[("MPI Time Average", 1)]) - np.log(wide[("MPI Time Average", 0)]),
    "dApp": np.log(wide[("App Time Average", 1)]) - np.log(wide[("App Time Average", 0)]),
    "mpi0": wide[("MPI Time Average", 0)], "mpi1": wide[("MPI Time Average", 1)],
    "app0": wide[("App Time Average", 0)], "app1": wide[("App Time Average", 1)],
}).reset_index()
pairs["first"] = ("8_" + pairs.App + "_" + pairs["Run Iteration"].astype(str)).map(order)
pairs.to_csv(OUT_DIR / "socket_pinned_pairs.csv", index=False)


# ------------------------------------------------------------------ S1
def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def cluster_boot(df, col, B=10000, seed=923, by="App"):
    """Cluster bootstrap (clusters = apps, or inhibitor configs) of the mean of per-cluster means."""
    per = df.groupby(by)[col].mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), (B, len(per)))
    bm = per.to_numpy()[idx].mean(1)
    return per.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


p("## S1. Isolated socket effect for apps, paired within job: d = log(sock1) - log(sock0)")
p("Negative d means sock1 is faster. CI = t interval over the 4 replicates (n = 4 is too small for a bootstrap). "
  "'sock1<sock0' counts the replicates with sock1 faster. Order effect: o = (mean d | sock0 ran first - "
  "mean d | sock1 ran first)/2 = log(second step) - log(first step); the order-balanced socket "
  "effect is the average of the two order means, which equals mean d when the design is balanced (2 + 2).")
S1 = {}
for col, lab in (("dMPI", "log T_MPI"), ("dApp", "log T_app")):
    recs = {}
    for a in APPS:
        x = pairs[pairs.App == a]
        if x.empty:
            continue
        lo, hi = t_ci(x[col])
        f0, f1 = x[x["first"] == 0][col].mean(), x[x["first"] == 1][col].mean()
        recs[NAME[a]] = {"n": len(x), "mean d": x[col].mean(), "median d": x[col].median(), "CI lo": lo, "CI hi": hi,
                         "sock1<sock0": f"{int((x[col] < 0).sum())}/{len(x)}", "d|sock0 first": f0,
                         "d|sock1 first": f1, "order o": (f0 - f1) / 2,
                         "ratio sock1/sock0": float(np.exp(x[col].mean()))}
    t = pd.DataFrame(recs).T
    t.index.name = "app"
    S1[col] = t
    p()
    p(f"### {lab}")
    p(md(t))
    m, lo, hi = cluster_boot(pairs, col)
    p(f"Pooled over apps (mean of per-app means), app-cluster bootstrap (10,000 draws, seed 923): "
      f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (x{np.exp(m):.4f}); median over all {len(pairs)} pairs "
      f"{pairs[col].median():+.4f}; sock1 faster in {int((pairs[col] < 0).sum())}/{len(pairs)} pairs.")
    w = stats.wilcoxon(pairs[col])
    p(f"Wilcoxon signed-rank on all {len(pairs)} pairs: p = {w.pvalue:.3g}.")
    # order effect pooled
    pairs["_o"] = np.where(pairs["first"] == 0, pairs[col], -pairs[col])
    mo, olo, ohi = cluster_boot(pairs, "_o")
    p(f"Pooled order effect (log second-step - log first-step, socket-balanced): {mo:+.4f} [{olo:+.4f}, {ohi:+.4f}].")
    # two-way: d = s + o*(+1 if sock0 first else -1)
    X = np.column_stack([np.ones(len(pairs)), np.where(pairs["first"] == 0, 1.0, -1.0)])
    beta, *_ = np.linalg.lstsq(X, pairs[col].to_numpy(), rcond=None)
    p(f"Least squares d = socket + order*(+1 if sock0 first else -1): socket {beta[0]:+.4f}, order {beta[1]:+.4f}.")
    pairs.drop(columns="_o", inplace=True)
p()

# raw times for scale
raw = pairs.groupby("App")[["mpi0", "mpi1", "app0", "app1"]].mean()
raw.index = [NAME[a] for a in raw.index]
raw.index.name = "app"
raw.columns = ["T_MPI sock0 (s)", "T_MPI sock1 (s)", "T_app sock0 (s)", "T_app sock1 (s)"]
p("Mean isolated times (s) over 4 replicates:")
p(md(raw, "{:.2f}"))
p()

# ------------------------------------------------------------------ S2
p("## S2. Isolated inhibitors")
ci = chk[chk.kind == "inhib"]
INH_RE = re.compile(r"8_inhib_(\d+)_(\d+)_d_([\d.]+)_(\d+)$")
KEY = ["Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity", "Run Iteration"]
inh_new = None
inh_pairs = None
if len(ci) and ci.valid.any():
    p("The inhibitor runs for a fixed duration (`-r 360`), so T_app is ~361 s by construction and is not a "
      "socket outcome. The socket outcomes are the work done in that window (log total messages sent; bytes = "
      "messages x size, so the same d) and log T_MPI. d = log(sock1) - log(sock0); positive d for messages means "
      "the inhibitor injects MORE traffic on sock1.")
    ok = ci[ci.valid]
    p(f"- valid inhibitor runs: {len(ok)}/{len(ci)}; threads {sorted(set(ok.threads))}; PU range sock0 "
      f"{ok[ok.socket_dir == 'sock0'].pu_min.min()}-{ok[ok.socket_dir == 'sock0'].pu_max.max()}, sock1 "
      f"{ok[ok.socket_dir == 'sock1'].pu_min.min()}-{ok[ok.socket_dir == 'sock1'].pu_max.max()}.")
    snc = ok[~ok.check.str.contains(r"numa_nodes=\[ 0 1 2 3 \]|numa_nodes=\[ 4 5 6 7 \]")]
    if len(snc):
        p(f"- {len(snc)} runs report unexpected NUMA sets ({', '.join(snc.socket_dir + '/' + snc.run)}). "
          "Their probes show one host (dane261) with sub-NUMA clustering OFF (2 NUMA domains: socket 0 = numa 0, "
          "socket 1 = numa 1); PUs are on the correct socket, so the runs are kept. The same jobs' sock0 steps "
          "are on that host too (numa 0 for all its sock0 ranks), so the pair stays matched.")
    rows = []
    for s in (0, 1):
        v = set(ok[ok.socket_dir == f"sock{s}"].run)
        df = parse_root(EXP / "data_inhib" / f"sock{s}", v, "inhib")
        df["Socket"] = s
        rows.append(df)
    inh_new = pd.concat(rows, ignore_index=True)
    inh_new["Inhib Comm Sparsity"] = inh_new["Inhib Comm Sparsity"].round(9)
    run_of = {}
    for r in ci.run:
        m_, w_, s_, i_ = INH_RE.match(r).groups()
        run_of[(int(m_), int(w_), round(float(s_), 9), int(i_))] = r
    inh_new["run"] = [run_of[tuple(k)] for k in inh_new[KEY].astype(object).itertuples(index=False)]
    inh_new["first_socket"] = inh_new.run.map(order)
    inh_new["cfg"] = ("m=" + inh_new["Inhib Message Size"].astype(str) + " w=" + inh_new["Inhib Wait Time (us)"]
                      .astype(str) + " s=" + inh_new["Inhib Comm Sparsity"].map(lambda x: f"{x:g}"))
    inh_new.to_csv(OUT_DIR / "socket_pinned_inhib_runs.csv", index=False)
    p(f"- T_app range over all {len(inh_new)} runs: {inh_new['Inhib App Time Average'].min():.1f}-"
      f"{inh_new['Inhib App Time Average'].max():.1f} s (fixed-duration check).")

    iw = inh_new.pivot_table(index=["cfg", "Run Iteration"], columns="Socket",
                             values=["Inhib MPI Time Average", "Inhib Total Messages Sent",
                                     "Inhib App Time Average"]).dropna()
    inh_pairs = pd.DataFrame({
        "dMsg": np.log(iw[("Inhib Total Messages Sent", 1)]) - np.log(iw[("Inhib Total Messages Sent", 0)]),
        "dMPI": np.log(iw[("Inhib MPI Time Average", 1)]) - np.log(iw[("Inhib MPI Time Average", 0)]),
        "dApp": np.log(iw[("Inhib App Time Average", 1)]) - np.log(iw[("Inhib App Time Average", 0)]),
        "msg0": iw[("Inhib Total Messages Sent", 0)], "msg1": iw[("Inhib Total Messages Sent", 1)],
        "mpi0": iw[("Inhib MPI Time Average", 0)], "mpi1": iw[("Inhib MPI Time Average", 1)],
    }).reset_index()
    first_of = inh_new.drop_duplicates("run").set_index(["cfg", "Run Iteration"]).first_socket
    inh_pairs["first"] = [first_of[(c, i)] for c, i in zip(inh_pairs.cfg, inh_pairs["Run Iteration"])]
    inh_pairs.to_csv(OUT_DIR / "socket_pinned_inhib_pairs.csv", index=False)
    cfg_order = (inh_new.drop_duplicates("cfg").sort_values(KEY[:3]).cfg.tolist())

    for col, lab in (("dMsg", "log messages sent (work in the 360 s window)"), ("dMPI", "log T_MPI"),
                     ("dApp", "log T_app (sanity: should be ~0)")):
        recs = {}
        for c in cfg_order:
            x = inh_pairs[inh_pairs.cfg == c]
            lo, hi = t_ci(x[col])
            f0, f1 = x[x["first"] == 0][col].mean(), x[x["first"] == 1][col].mean()
            recs[c] = {"n": len(x), "mean d": x[col].mean(), "CI lo": lo, "CI hi": hi,
                       "sock1<sock0": f"{int((x[col] < 0).sum())}/{len(x)}", "order o": (f0 - f1) / 2,
                       "ratio sock1/sock0": float(np.exp(x[col].mean()))}
        t = pd.DataFrame(recs).T
        t.index.name = "inhibitor (-m bytes, -w us, -s)"
        p()
        p(f"### {lab}")
        p(md(t))
        m, lo, hi = cluster_boot(inh_pairs, col, by="cfg")
        w = stats.wilcoxon(inh_pairs[col]) if inh_pairs[col].abs().sum() > 0 else None
        p(f"Pooled over configs (mean of per-config means), config-cluster bootstrap (10,000 draws, seed 923): "
          f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (x{np.exp(m):.4f}); sock1 lower in "
          f"{int((inh_pairs[col] < 0).sum())}/{len(inh_pairs)} pairs"
          + (f"; Wilcoxon p = {w.pvalue:.3g}." if w is not None else "."))
        inh_pairs["_o"] = np.where(inh_pairs["first"] == 0, inh_pairs[col], -inh_pairs[col])
        mo, olo, ohi = cluster_boot(inh_pairs, "_o", by="cfg")
        p(f"Pooled order effect (log second-step - log first-step): {mo:+.4f} [{olo:+.4f}, {ohi:+.4f}].")
        inh_pairs.drop(columns="_o", inplace=True)
    p()
    rawi = inh_pairs.groupby("cfg")[["msg0", "msg1", "mpi0", "mpi1"]].mean().loc[cfg_order]
    rawi.columns = ["msgs sock0", "msgs sock1", "T_MPI sock0 (s)", "T_MPI sock1 (s)"]
    rawi.index.name = "inhibitor"
    p("Mean isolated inhibitor outcomes over 4 replicates:")
    p(md(rawi, "{:.4g}"))
else:
    cfgs = sorted({re.sub(r"_\d+$", "", r) for r in ci.run})
    p(f"**Not available.** data_inhib/sock0 and sock1 contain {len(ci)} run dirs ({len(cfgs)} configs x 4 reps), "
      f"but all are empty: no .mpiP, no placement_probe.log, no placement_check.txt, and job_logs/8_inhib_* are empty "
      f"(no steps.txt, no Slurm output). The inhibitor jobs have not run or their output was not copied.")
    p("Configs covered by the directory names (exactly the 9 full-data MPI panel inhibitors, no extras): "
      + ", ".join(cfgs))
p()

# ------------------------------------------------------------------ S3 drift
p("## S3. Launcher/epoch drift (NOT a socket effect): new sock0 vs June isolated baselines")
p("June isolated runs: mpibind, 56 ranks/node on PUs 112-167 (SMT thread 1 of socket 0), no `--whole`. "
  "New sock0: numactl, PUs 0-55 (thread 0 of the same cores), `--whole`, Nov 2026. Same socket, so the "
  "difference mixes launcher, SMT thread index, memory policy (--localalloc), software/system state, and node set.")
iso = read(PROCESSED / "data_isolated.csv")
j8 = iso[iso["Num Nodes"] == 8]
drift = {}
for a in APPS:
    j = j8[j8.App == a]
    n0 = new[(new.App == a) & (new.Socket == 0)]
    n1 = new[(new.App == a) & (new.Socket == 1)]
    rec = {}
    for col, lab in (("MPI Time Average", "MPI"), ("App Time Average", "app")):
        jm, nm = j[col].mean(), n0[col].mean()
        rec[f"June T_{lab} (s)"] = jm
        rec[f"new sock0 T_{lab} (s)"] = nm
        rec[f"ratio {lab}"] = nm / jm
        rec[f"log diff {lab}"] = np.log(nm) - np.log(jm)
        rec[f"June CV {lab}"] = j[col].std(ddof=1) / jm
        rec[f"new sock0 CV {lab}"] = n0[col].std(ddof=1) / nm
        rec[f"|log diff|/June CV {lab}"] = abs(np.log(nm) - np.log(jm)) / (j[col].std(ddof=1) / jm)
    drift[NAME[a]] = rec
drift = pd.DataFrame(drift).T
drift.index.name = "app"
for lab in ("MPI", "app"):
    cols = [c for c in drift.columns if c.endswith(lab) or f"T_{lab}" in c]
    p()
    p(f"### T_{lab}")
    p(md(drift[cols]))
    ld = drift[f"log diff {lab}"]
    p(f"Mean log diff over apps {ld.mean():+.4f} (median {ld.median():+.4f}); apps with |log diff| > 2 June CVs: "
      f"{', '.join(drift.index[drift[f'|log diff|/June CV {lab}'] > 2])}.")
p()
if inh_new is None:
    p("Inhibitor drift (new sock0 vs June/Oct isolated inhibitor baselines): not available, inhibitor data missing (S2).")
else:
    p("### Inhibitors: new sock0 vs earlier isolated inhibitor runs (`processed_data/data_inhib_isolated.csv`, "
      "mpibind, `-r 360`, 8 nodes)")
    old = read(PROCESSED / "data_inhib_isolated.csv")
    old = old[old["Num Nodes"] == 8].copy()
    old["Inhib Comm Sparsity"] = old["Inhib Comm Sparsity"].round(9)
    n0i = inh_new[inh_new.Socket == 0]
    recs = {}
    for c in cfg_order:
        g = n0i[n0i.cfg == c]
        o = old.merge(g[KEY[:3]].drop_duplicates(), on=KEY[:3])
        rec = {"n old": len(o)}
        for col, lab in (("Inhib Total Messages Sent", "msgs"), ("Inhib MPI Time Average", "T_MPI")):
            om, nm = o[col].mean(), g[col].mean()
            rec[f"old {lab}"] = om
            rec[f"new sock0 {lab}"] = nm
            rec[f"log diff {lab}"] = np.log(nm) - np.log(om)
            rec[f"old CV {lab}"] = o[col].std(ddof=1) / om
        recs[c] = rec
    idr = pd.DataFrame(recs).T
    idr.index.name = "inhibitor"
    p(md(idr, "{:.4g}"))
    for lab in ("msgs", "T_MPI"):
        p(f"Mean log diff {lab} over configs {idr[f'log diff {lab}'].mean():+.4f} "
          f"(median {idr[f'log diff {lab}'].median():+.4f}).")
p()

# ------------------------------------------------------------------ S4 co-run comparison
ai = read(PROCESSED / "data_inhib_coscheduled.csv")
aa = read(PROCESSED / "data_coscheduled.csv")
sa = pd.read_csv(OUT_DIR / "socket_app_inhibitor.csv")
ai2 = ai.rename(columns={"Inhib Message Size": "m", "Inhib Wait Time (us)": "w", "Inhib Comm Sparsity": "sp",
                         "Run Iteration": "rep"})
ai2["sp"] = ai2.sp.round(9)
ai2 = ai2.merge(sa, on=["App", "m", "w", "sp", "rep"])
ai2["ly"] = np.log(ai2["MPI Time Average"])
gi = ai2.groupby(["App", "m", "w", "sp", "app_sock"]).ly.mean().unstack().dropna()
diff_ai = (gi[1] - gi[0]).rename("d").reset_index().rename(columns={"App": "victim"})
sb = pd.read_csv(OUT_DIR / "socket_app_app.csv")
aa8 = aa[aa["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"}).merge(
    sb, on=["A", "B", "rep"])
long = pd.concat([pd.DataFrame({"victim": aa8.A, "other": aa8.B, "sock": aa8.sock_A,
                                "ly": np.log(aa8["App A MPI Time Average"])}),
                  pd.DataFrame({"victim": aa8.B, "other": aa8.A, "sock": aa8.sock_B,
                                "ly": np.log(aa8["App B MPI Time Average"])})])
long = long[long.victim != long.other]
gd = long.groupby(["victim", "other", "sock"]).ly.mean().unstack().dropna()
diff_aa = (gd[1] - gd[0]).rename("d").reset_index()

p("## S4. Isolated vs co-run socket effect on log T_MPI (sock1 - sock0)")
p("Co-run columns reproduce figs_batch1 C8 (App-Inh: per config with reps on both sockets; App-App: per non-self "
  "directional outcome with reps on both sockets). 'co-run minus isolated' = App-Inh mean minus isolated mean d: "
  "the part of the co-run socket gap not explained by the isolated (baseline) socket gap.")
s4 = pd.DataFrame({
    "iso mean d": S1["dMPI"]["mean d"].astype(float),
    "iso CI lo": S1["dMPI"]["CI lo"].astype(float), "iso CI hi": S1["dMPI"]["CI hi"].astype(float),
    "iso sock1<sock0": S1["dMPI"]["sock1<sock0"],
})
for nm_, dd in (("App-Inh", diff_ai), ("App-App", diff_aa)):
    g = dd.groupby("victim").d
    s4[f"{nm_} n"] = g.size().rename(NAME)
    s4[f"{nm_} median"] = g.median().rename(NAME)
    s4[f"{nm_} mean"] = g.mean().rename(NAME)
s4["App-Inh mean - iso"] = s4["App-Inh mean"] - s4["iso mean d"]
s4["App-App mean - iso"] = s4["App-App mean"] - s4["iso mean d"]
s4 = s4.loc[[NAME[a] for a in APPS]]
s4.index.name = "app"
p(md(s4))
co_ai_mean = diff_ai.groupby("victim").d.mean()
iso_mean = pairs.groupby("App").dMPI.mean()
r = stats.spearmanr(iso_mean.loc[co_ai_mean.index], co_ai_mean)
p(f"Across apps, Spearman(isolated mean d, App-Inh co-run mean d) = {r.statistic:.2f} (p = {r.pvalue:.2g}).")
p(f"Pooled: isolated mean of app means {iso_mean.mean():+.4f}; co-run App-Inh mean of app means "
  f"{co_ai_mean.mean():+.4f}; App-App {diff_aa.groupby('victim').d.mean().mean():+.4f}.")
p()

(OUT_DIR / "socket_numbers.md").write_text(log.getvalue())

# ------------------------------------------------------------------ F-S1
set_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

orderx = [a for a in APPS]
fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
x = np.arange(len(orderx))
C_ISO, C_AI, C_AA = OI["blue"], OI["vermillion"], OI["green"]
rng = np.random.default_rng(0)
for k, a in enumerate(orderx):
    v = pairs[pairs.App == a].dMPI.to_numpy()
    ax.scatter(np.full(len(v), k - 0.22) + rng.uniform(-0.04, 0.04, len(v)), v, s=11, color=C_ISO, alpha=0.6, lw=0,
               zorder=3)
    lo, hi = t_ci(v)
    ax.errorbar(k - 0.22, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=4.5, color=C_ISO,
                mfc="white", mew=1.2, capsize=2.5, lw=1.1, zorder=4)
    for off, dd, col, mk in ((0.08, diff_ai, C_AI, "s"), (0.3, diff_aa, C_AA, "^")):
        w = dd[dd.victim == a].d.to_numpy()
        if len(w):
            q1, q3 = np.percentile(w, [25, 75])
            ax.plot([k + off] * 2, [q1, q3], color=col, lw=1.1, zorder=3)
            ax.plot(k + off, np.median(w), marker=mk, ms=4.5, color=col, ls="", zorder=4)
ax.axhline(0, color=OI["grey"], lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([NAME[a] for a in orderx], rotation=30, ha="right")
ax.set_ylabel(r"$\Delta \log T_\mathrm{MPI}$ (socket 1 $-$ socket 0)")
ax.grid(axis="y", lw=0.4, alpha=0.4)
ax.set_xlim(-0.6, len(orderx) - 0.4)
ax.legend(handles=[
    Line2D([], [], color=C_ISO, marker="o", mfc="white", mew=1.2, ls="-", lw=1.1, ms=4.5,
           label="isolated, pinned: mean, 95% CI (dots = reps)"),
    Line2D([], [], color=C_AI, marker="s", ls="-", lw=1.1, ms=4.5, label="App-Inh co-run: median, IQR"),
    Line2D([], [], color=C_AA, marker="^", ls="-", lw=1.1, ms=4.5, label="App-App co-run: median, IQR")],
    frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handletextpad=0.3,
    columnspacing=0.8, handlelength=1.6)
save(fig, "socket_isolated.pdf")
plt.close(fig)
print("wrote", OUT_DIR / "socket_numbers.md")
