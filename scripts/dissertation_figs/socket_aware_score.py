"""Socket batch 2, part B scoring: socket-blind vs socket-aware refits of the four pipelines.

B1  socket-resolved App-App outcomes: one cell per (victim, aggressor, victim socket s_V) with >= 1 replicate (166).
    True slowdown = max(1, mean T_co over that socket's reps / June isolated mean). Blind models predict the same
    value for both sockets; aware models predict per socket (directional_predictions_socket.csv).
    Extra model: "blind absolute + fixed socket offset" = blind absolute latent + s_V * dbar_V before the floor
    (dbar = pinned isolated sock1 - sock0 mean log difference for the target's time; JMs rule = only apps whose t CI
    excludes 0; all-app dbar as a sensitivity row).
    Metrics: log MAE, delta% vs isolated-speed (constant 1) with a 95% pair-cluster bootstrap CI (45 unordered pairs,
    1,000 draws, seed 923, shared draw matrix; delta% = (method - baseline) / baseline full-sample log MAE), signed
    bias, floor / above-floor log MAE, Spearman, false-benign at 1.05 (unit = unordered pair x socket assignment:
    both endpoints of the same replicates; harmful = true worse endpoint >= 1.05), paired aware - blind log MAE
    difference with the same bootstrap. Splits by s_V; robustness rows on cells with >= 2 reps and with weight = n_reps.
B2  rep-averaged App-App outcomes (the usual 90): aware predictions aggregated over the replicates' actual
    assignments in slowdown space: s_agg = (n0 s(0) + n1 s(1)) / (n0 + n1). Same metrics (FB at the usual pair level).
    Blind refit vs artifact log MAE.

Run dirs are configured in RUNS below (missing runs are skipped). Writes output/socket_aware_scores_{B1,B2}.csv and
appends a "Part B" section to output/socket_aware_numbers.md (rewritten from "## Part B" on every run).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

from common import NAME, OUT_DIR, REPO

EXP = Path.home() / "hpcResearch/experiment_outputs/socket_batch2"
UT = REPO / "notebooks/utils"
# pipeline -> (target, artifact run key, {variant: run dir})
RUNS = {
    "1W-MPI": ("mpi", "baseSnap", {"blind": EXP / "blind_1w_mpi", "aware": EXP / "aware_1w_mpi"}),
    "1W-TT": ("tt", "totalTime", {"blind": EXP / "blind_1w_tt", "aware": EXP / "aware_1w_tt"}),
    "BD-MPI": ("mpi", "bidir", {"blind": EXP / "blind_bd_mpi", "aware": EXP / "aware_bd_mpi"}),
    "BD-TT": ("tt", "bidirTT", {"blind": EXP / "blind_bd_tt", "aware": EXP / "aware_bd_tt"}),
}
PIPE_LABEL = {"1W-MPI": "One-way, MPI time", "1W-TT": "One-way, total time", "BD-MPI": "Bidirectional, MPI time",
              "BD-TT": "Bidirectional, total time"}
# role -> code name per family (one-way / bidirectional)
ROLE_CODE = {
    "constant_1": ("constant_1", "constant_1"),
    "absolute": ("absolute_response", "legacy_absolute"),
    "delta_uniform": ("delta_uniform", "legacy_delta_uniform"),
    "delta_median": ("delta_median", "legacy_delta_median"),
    "delta_ood_kernel": ("delta_ood_kernel", "legacy_delta_ood_kernel"),
    "generic": ("generic_potential", "legacy_generic_uniform"),
    "joint_direct": (None, "joint_profile_direct"), "joint_cal": (None, "joint_profile_calibrated"),
    "bprof_direct": (None, "bidirectional_profile_direct"), "bprof_cal": (None, "bidirectional_profile_calibrated"),
    "bctx_direct": (None, "bidirectional_context_direct"), "bctx_cal": (None, "bidirectional_context_calibrated"),
    "bwall_cal": (None, "bidirectional_profile_wall_calibrated"),
    "bfwd_cal": (None, "bidirectional_context_forward_calibrated"),
}
LABEL = {"constant_1": "Isolated-speed baseline", "absolute": "Absolute response",
         "delta_uniform": "Delta, uniform agg.", "delta_median": "Delta, median agg.",
         "delta_ood_kernel": "Delta, OOD-kernel agg.", "generic": "Generic potential",
         "joint_direct": "Joint", "joint_cal": "Joint, calibrated", "bprof_direct": "Bidir. profile",
         "bprof_cal": "Bidir. profile, calibrated", "bctx_direct": "Bidir. context",
         "bctx_cal": "Bidir. context, calibrated", "bwall_cal": "Wall-clock aux (ablation), cal.",
         "bfwd_cal": "Forward-only tokens (ablation), cal.",
         "offset_jms": "Blind absolute + fixed socket offset (JMs)",
         "offset_all": "Blind absolute + fixed socket offset (all apps)"}
ORDER = ["absolute", "offset_jms", "offset_all", "delta_uniform", "delta_median", "delta_ood_kernel", "generic",
         "joint_direct", "joint_cal", "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal", "bwall_cal", "bfwd_cal"]
LOG105 = np.log(1.05)
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    df = df.copy()
    for c in ("n", "harmful", "CV rank", "App-App rank (resolved)"):
        if c in df:
            df[c] = df[c].astype("Int64").astype(str)
    for i, r in df.iterrows():
        cells = [str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                            ("–" if isinstance(v, (float, np.floating)) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def find_one(root: Path, name: str) -> Path | None:
    """The random_split prediction file of a run root (pipeline output dir or runner root)."""
    if not root.exists():
        return None
    direct = root / name
    if direct.exists():
        return direct
    # one-way runner root: consolidated outputs (filtered to random_split / final evaluation in load_preds)
    cons = {"directional_predictions.csv": "predictions.csv",
            "directional_predictions_socket.csv": "socket_predictions.csv",
            "selection.json": "global_selection.json"}
    if (root / "consolidated").is_dir() and name in cons:
        f = root / "consolidated" / cons[name]
        return f if f.exists() else None
    hits = sorted(h for h in root.rglob(name) if "consolidated" not in h.parts)
    hits = [h for h in hits if "random_split" in str(h)] or hits
    return hits[-1] if hits else None


def load_preds(root: Path, bidir: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    out = []
    for name in ("directional_predictions.csv", "directional_predictions_socket.csv"):
        f = find_one(root, name)
        if f is None:
            out.append(None)
            continue
        d = pd.read_csv(f)
        if "stage" in d:
            d = d[d.stage == "evaluate_random_split"]
        if "evaluation_method" in d:
            d = d[d.evaluation_method == "random_split"]
        if "anchor_budget" in d:
            d = d[d.anchor_budget.astype(str) == "all"]
        if name == "directional_predictions.csv":
            d = d.rename(columns={"victim_id": "victim", "aggressor_id": "aggressor"})
        code2role = {v[1 if bidir else 0]: k for k, v in ROLE_CODE.items() if v[1 if bidir else 0]}
        d = d[d.method.isin(code2role)].assign(role=lambda x: x.method.map(code2role))
        d = d[d.victim != d.aggressor]
        out.append(d)
    return out[0], out[1]


# ------------------------------------------------------------------ truth
cells = pd.read_csv(OUT_DIR / "socket_resolved_appapp.csv")  # victim, aggressor, s_V, n_reps, ratio_mpi, ratio_tt
for t in ("mpi", "tt"):
    cells[f"y_{t}"] = np.log(np.maximum(1.0, cells[f"ratio_{t}"]))
cells["pair"] = [" | ".join(sorted(x)) for x in zip(cells.victim, cells.aggressor)]
PAIRS = np.array(sorted(cells.pair.unique()))
assert len(PAIRS) == 45 and len(cells) == 166
rng = np.random.default_rng(923)
DRAWS = rng.choice(len(PAIRS), size=(1000, len(PAIRS)), replace=True)

# pinned isolated socket offsets dbar per app and target, with JMs selection (t CI excludes 0)
pp = pd.read_csv(OUT_DIR / "socket_pinned_pairs.csv")
DBAR = {}
for t, col in (("mpi", "dMPI"), ("tt", "dApp")):
    rec = {}
    for a, g in pp.groupby("App"):
        x = g[col].to_numpy()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        rec[a] = (x.mean(), (x.mean() + h < 0) or (x.mean() - h > 0))
    DBAR[t] = rec


def cluster_stats(df: pd.DataFrame, err_col: str, w_col: str | None):
    """Per-pair-cluster weighted error sums and weights, aligned to PAIRS."""
    w = df[w_col] if w_col else pd.Series(1.0, index=df.index)
    s = (df[err_col] * w).groupby(df.pair).sum().reindex(PAIRS, fill_value=0.0).to_numpy()
    n = w.groupby(df.pair).sum().reindex(PAIRS, fill_value=0.0).to_numpy()
    return s, n


def boot_delta(df_m, df_b, w_col=None):
    """Delta% = (MAE_m - MAE_b)/MAE_b(full) per draw; returns (point, lo, hi) in %."""
    sm, nm = cluster_stats(df_m, "abs_err", w_col)
    sb, nb = cluster_stats(df_b, "abs_err", w_col)
    base = sb.sum() / nb.sum()
    dm = sm[DRAWS].sum(1) / nm[DRAWS].sum(1) - sb[DRAWS].sum(1) / nb[DRAWS].sum(1)
    return 100 * (sm.sum() / nm.sum() - base) / base, 100 * np.percentile(dm, 2.5) / base, 100 * np.percentile(dm, 97.5) / base


def boot_diff(df_a, df_b, w_col=None):
    """Paired absolute difference in log MAE (a - b), with the pair-cluster CI."""
    sa, na = cluster_stats(df_a, "abs_err", w_col)
    sb, nb = cluster_stats(df_b, "abs_err", w_col)
    d = sa[DRAWS].sum(1) / na[DRAWS].sum(1) - sb[DRAWS].sum(1) / nb[DRAWS].sum(1)
    return sa.sum() / na.sum() - sb.sum() / nb.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def false_benign(df: pd.DataFrame) -> tuple[float, int]:
    """Unit: unordered pair x assignment. Directed cell (V, X, s) pairs with (X, V, 1 - s)."""
    d = df.copy()
    d["assign_key"] = [f"{pr}#{v if s == 0 else x}" for pr, v, x, s in zip(d.pair, d.victim, d.aggressor, d.s_V)]
    g = d.groupby("assign_key").agg(tw=("y", "max"), pw=("pred_log", "max"), n=("y", "size"))
    g = g[g.n == 2]
    harm = g.tw >= LOG105
    return (float(((g.pw < LOG105) & harm).sum() / harm.sum()) if harm.sum() else np.nan), int(harm.sum())


def metrics(df: pd.DataFrame, w_col=None) -> dict:
    w = df[w_col].to_numpy(float) if w_col else np.ones(len(df))
    e = df.pred_log - df.y
    fl = df.y == 0
    return {"n": len(df), "log MAE": float(np.average(np.abs(e), weights=w)),
            "bias": float(np.average(e, weights=w)),
            "floor log MAE": float(np.average(np.abs(e[fl]), weights=w[fl])) if fl.any() else np.nan,
            "above log MAE": float(np.average(np.abs(e[~fl]), weights=w[~fl])) if (~fl).any() else np.nan,
            "Spearman": float(stats.spearmanr(df.pred_log, df.y).statistic)}


# ------------------------------------------------------------------ build resolved predictions per run
def resolved_predictions(pipe: str) -> dict[str, pd.DataFrame]:
    """variant ('blind'/'aware') -> long frame over the 166 cells x roles with pred_log, y, abs_err."""
    target, _, dirs = RUNS[pipe]
    bidir = pipe.startswith("BD")
    base = cells[["victim", "aggressor", "s_V", "n_reps", "pair", f"y_{target}"]].rename(columns={f"y_{target}": "y"})
    out = {}
    for variant, root in dirs.items():
        agg_pred, sock_pred = load_preds(root, bidir)
        if variant == "blind" and agg_pred is not None:
            pr = agg_pred[["role", "victim", "aggressor", "predicted_log_slowdown", "latent_predicted_log_slowdown"]]
            d = base.merge(pr, on=["victim", "aggressor"], validate="many_to_many")
            d = d.rename(columns={"predicted_log_slowdown": "pred_log", "latent_predicted_log_slowdown": "latent"})
            # post-hoc offset models from the blind absolute latent
            ab = d[d.role == "absolute"]
            for tag, only_ci in (("offset_jms", True), ("offset_all", False)):
                off = [DBAR[target][v][0] if (s == 1 and (DBAR[target][v][1] or not only_ci)) else 0.0
                       for v, s in zip(ab.victim, ab.s_V)]
                d = pd.concat([d, ab.assign(role=tag, pred_log=np.maximum(0.0, ab.latent + np.array(off)))])
            out[variant] = d
        elif variant == "aware" and sock_pred is not None:
            pr = sock_pred[["role", "victim", "aggressor", "s_V", "pred_log_slowdown"]].rename(
                columns={"pred_log_slowdown": "pred_log"})
            out[variant] = base.merge(pr, on=["victim", "aggressor", "s_V"], validate="one_to_many")
    for d in out.values():
        d["abs_err"] = (d.pred_log - d.y).abs()
        n = d.groupby("role").size()
        assert (n == 166).all(), n[n != 166]
    return out


def b1_table(pipe: str, preds: dict[str, pd.DataFrame], subset=None, w_col=None) -> pd.DataFrame:
    rows = []
    for variant, d in preds.items():
        if subset is not None:
            d = d[subset(d)]
        base = d[d.role == "constant_1"]
        for role in [r for r in ["constant_1", *ORDER] if r in set(d.role)]:
            g = d[d.role == role]
            m = metrics(g, w_col)
            if role != "constant_1":
                m["Δ%"], m["CI lo"], m["CI hi"] = boot_delta(g, base, w_col)
            fb, nh = false_benign(g)
            rows.append({"pipeline": pipe, "variant": variant, "role": role, **m, "FB@1.05": fb, "harmful": nh})
    return pd.DataFrame(rows)


def paired_aware_blind(pipe, preds, subset=None, w_col=None):
    if not {"blind", "aware"} <= set(preds):
        return pd.DataFrame()
    rows = []
    for role in ORDER:
        if role.startswith("offset"):
            continue
        a, b = preds["aware"], preds["blind"]
        if subset is not None:
            a, b = a[subset(a)], b[subset(b)]
        a, b = a[a.role == role], b[b.role == role]
        if a.empty or b.empty:
            continue
        diff, lo, hi = boot_diff(a, b, w_col)
        rows.append({"pipeline": pipe, "role": role, "aware − blind log MAE": diff, "CI lo": lo, "CI hi": hi,
                     "ratio": a.abs_err.mean() / b.abs_err.mean()})
    # offset model vs aware absolute
    if "blind" in preds and "aware" in preds:
        for tag in ("offset_jms", "offset_all"):
            a = preds["aware"][preds["aware"].role == "absolute"]
            b = preds["blind"][preds["blind"].role == tag]
            if subset is not None:
                a, b = a[subset(a)], b[subset(b)]
            diff, lo, hi = boot_diff(a, b, w_col)
            rows.append({"pipeline": pipe, "role": f"aware absolute vs {tag}", "aware − blind log MAE": diff,
                         "CI lo": lo, "CI hi": hi, "ratio": a.abs_err.mean() / b.abs_err.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B2: rep-averaged outcomes
def b2_predictions(pipe: str, resolved: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    target, _, dirs = RUNS[pipe]
    bidir = pipe.startswith("BD")
    rep = cells.assign(T=lambda x: x[f"ratio_{target}"] * x.n_reps).groupby(["victim", "aggressor", "pair"]).agg(
        T=("T", "sum"), n=("n_reps", "sum")).reset_index()
    rep["y"] = np.log(np.maximum(1.0, rep["T"] / rep["n"]))
    out = {}
    for variant, root in dirs.items():
        agg_pred, sock_pred = load_preds(root, bidir)
        if agg_pred is None:
            continue
        if variant == "aware" and sock_pred is not None:
            s = sock_pred.merge(cells[["victim", "aggressor", "s_V", "n_reps"]], on=["victim", "aggressor", "s_V"],
                                how="left").fillna({"n_reps": 0})
            s["w"] = s.n_reps * s.pred_slowdown
            mine = s.groupby(["role", "victim", "aggressor"]).agg(w=("w", "sum"), n=("n_reps", "sum")).reset_index()
            mine["pred_log_mine"] = np.log(mine.w / mine.n)
            chk = agg_pred.merge(mine, on=["role", "victim", "aggressor"])
            dev = np.abs(chk.predicted_log_slowdown - chk.pred_log_mine).max()
            p(f"- {pipe} aware: pipeline's aggregated prediction vs mine (max |Δ log|) = {dev:.2e}")
        pr = agg_pred[["role", "victim", "aggressor", "predicted_log_slowdown"]].rename(
            columns={"predicted_log_slowdown": "pred_log"})
        if variant == "blind" and "blind" in resolved:
            r = resolved["blind"]
            r = r[r.role.str.startswith("offset")].assign(w=lambda x: x.n_reps * np.exp(x.pred_log))
            r = r.groupby(["role", "victim", "aggressor"]).agg(w=("w", "sum"), n=("n_reps", "sum")).reset_index()
            pr = pd.concat([pr, r.assign(pred_log=np.log(r.w / r.n))[pr.columns]])
        d = rep.merge(pr, on=["victim", "aggressor"], validate="one_to_many")
        out[variant] = d.assign(abs_err=lambda x: (x.pred_log - x.y).abs())
    return out


def fb_repavg(g: pd.DataFrame) -> tuple[float, int]:
    t = g.groupby("pair").agg(tw=("y", "max"), pw=("pred_log", "max"))
    harm = t.tw >= LOG105
    return (float(((t.pw < LOG105) & harm).sum() / harm.sum()) if harm.sum() else np.nan), int(harm.sum())


art = json.loads((OUT_DIR / "artifact_data_20261017.json").read_text())["runs"]

# ------------------------------------------------------------------ main
p("## Part B: results")
p()
p("Runs: " + "; ".join(f"{k}: " + ", ".join(f"{v}={'yes' if find_one(r, 'directional_predictions.csv') else 'missing'}"
                                          for v, r in RUNS[k][2].items()) for k in RUNS))
B1, PAIRED, B2 = [], [], []
for pipe in RUNS:
    preds = resolved_predictions(pipe)
    for lab, subset, w in (("all", None, None), ("s_V=0", lambda d: d.s_V == 0, None),
                           ("s_V=1", lambda d: d.s_V == 1, None), (">=2 reps", lambda d: d.n_reps >= 2, None),
                           ("weight=n_reps", None, "n_reps")):
        t = b1_table(pipe, preds, subset, w)
        if len(t):
            t.insert(1, "subset", lab)
            B1.append(t)
        pt = paired_aware_blind(pipe, preds, subset, w)
        if len(pt):
            pt.insert(1, "subset", lab)
            PAIRED.append(pt)
    for variant, d in b2_predictions(pipe, preds).items():
        base = d[d.role == "constant_1"]
        for role in [r for r in ["constant_1", *ORDER] if r in set(d.role)]:
            g = d[d.role == role]
            m = metrics(g)
            if role != "constant_1":
                m["Δ%"], m["CI lo"], m["CI hi"] = boot_delta(g, base)
            m["FB@1.05"], m["harmful"] = fb_repavg(g)
            ak = RUNS[pipe][1]
            m["artifact log MAE"] = art[ak]["random_split"].get(role, {}).get("log_mae", np.nan)
            B2.append({"pipeline": pipe, "variant": variant, "role": role, **m})

if B1:
    B1 = pd.concat(B1, ignore_index=True)
    B1.to_csv(OUT_DIR / "socket_aware_scores_B1.csv", index=False)
    PAIRED = pd.concat(PAIRED, ignore_index=True) if PAIRED else pd.DataFrame()
    PAIRED.to_csv(OUT_DIR / "socket_aware_paired_B1.csv", index=False)
    p()
    p("### B1. Socket-resolved App-App outcomes (166 cells)")
    for pipe in B1.pipeline.unique():
        for sub in B1.subset.unique():
            t = B1[(B1.pipeline == pipe) & (B1.subset == sub)].copy()
            t["method"] = t.role.map(LABEL) + " [" + t.variant + "]"
            t = t.set_index("method")[["n", "log MAE", "Δ%", "CI lo", "CI hi", "bias", "floor log MAE",
                                       "above log MAE", "Spearman", "FB@1.05", "harmful"]]
            p()
            p(f"#### {PIPE_LABEL[pipe]}: {sub}")
            p(md(t))
            if len(PAIRED):
                q = PAIRED[(PAIRED.pipeline == pipe) & (PAIRED.subset == sub)].set_index("role").drop(
                    columns=["pipeline", "subset"])
                if len(q):
                    p()
                    p("Paired aware − blind (same role), and aware absolute − offset model:")
                    p(md(q))
if B2:
    B2 = pd.DataFrame(B2)
    B2.to_csv(OUT_DIR / "socket_aware_scores_B2.csv", index=False)
    p()
    p("### B2. Rep-averaged App-App outcomes (90 directed non-self; tab:ns_fulldata convention)")
    for pipe in B2.pipeline.unique():
        t = B2[B2.pipeline == pipe].copy()
        t["method"] = t.role.map(LABEL) + " [" + t.variant + "]"
        p()
        p(f"#### {PIPE_LABEL[pipe]}")
        p(md(t.set_index("method")[["n", "log MAE", "artifact log MAE", "Δ%", "CI lo", "CI hi", "bias",
                                     "Spearman", "FB@1.05", "harmful"]]))

# ------------------------------------------------------------------ B3: per-victim aware/blind ratio (resolved cells)
B3_ROLES = {"BD-MPI": ["absolute", "joint_cal", "bprof_cal", "bctx_cal"], "1W-MPI": ["absolute"]}
b3 = []
for pipe, roles in B3_ROLES.items():
    preds = resolved_predictions(pipe)
    if not {"blind", "aware"} <= set(preds):
        continue
    if pipe == "BD-MPI":  # best calibrated = lowest aware B1 log MAE among calibrated joint variants
        cal = preds["aware"][preds["aware"].role.isin(["joint_cal", "bprof_cal", "bctx_cal"])]
        best = cal.groupby("role").abs_err.mean().idxmin()
        roles = ["absolute", best]
        p(f"- B3 best calibrated (BD-MPI, lowest aware resolved log MAE): {LABEL[best]}")
    for role in roles:
        a = preds["aware"][preds["aware"].role == role].groupby("victim").abs_err.mean()
        b = preds["blind"][preds["blind"].role == role].groupby("victim").abs_err.mean()
        sa = preds["aware"][preds["aware"].role == role]
        sb = preds["blind"][preds["blind"].role == role]
        for v in a.index:
            rec = {"pipeline": pipe, "role": role, "victim": NAME[v], "aware log MAE": a[v], "blind log MAE": b[v],
                   "ratio": a[v] / b[v]}
            for s in (0, 1):
                x = sa[(sa.victim == v) & (sa.s_V == s)].abs_err.mean()
                y = sb[(sb.victim == v) & (sb.s_V == s)].abs_err.mean()
                rec[f"ratio s_V={s}"] = x / y
            b3.append(rec)
if b3:
    b3 = pd.DataFrame(b3)
    b3.to_csv(OUT_DIR / "socket_aware_pervictim_B3.csv", index=False)
    p()
    p("### B3. Per-victim aware/blind log-MAE ratio (socket-resolved cells; < 1 = aware better)")
    for (pipe, role), g in b3.groupby(["pipeline", "role"], sort=False):
        p()
        p(f"#### {PIPE_LABEL[pipe]}: {LABEL[role]}")
        p(md(g.set_index("victim").drop(columns=["pipeline", "role"])))


# ------------------------------------------------------------------ B5: CV vs App-App ranking
def cv_scores(pipe: str, root: Path) -> dict[str, float]:
    """role -> crossed App-Inhibitor CV log MAE of the selected recipe (artifact `cv_log_mae` convention)."""
    sel = find_one(root, "selection.json")
    if sel is None:
        return {}
    v = json.loads(sel.read_text()).get("variants", {})
    out = {}
    if pipe.startswith("BD"):
        def agg(name, a):
            return v[name]["aggregation"][a]["validation_log_mae"] if name in v else np.nan
        prim = v.get("legacy_delta", {}).get("aggregation", {}).get("primary_method", "uniform")
        out["delta_" + prim] = agg("legacy_delta", prim)
        out["generic"] = agg("legacy_generic", "uniform")
        for role, name in (("absolute", "legacy_absolute"), ("joint_cal", "joint_profile"),
                           ("bprof_cal", "bidirectional_profile"), ("bctx_cal", "bidirectional_context"),
                           ("bwall_cal", "bidirectional_profile_wall"), ("bfwd_cal", "bidirectional_context_forward")):
            if name in v:
                out[role] = v[name]["early_stopping_validation_log_mae"]
    else:
        ms = json.loads(sel.read_text()).get("model_selection", {})
        prim = json.loads(sel.read_text()).get("aggregations", {}).get("low_rank", {}).get("primary_method", "uniform")
        for role, name in (("absolute", "absolute"), ("generic", "generic"), ("delta_" + prim, "low_rank")):
            if name in ms:
                out[role] = ms[name]["validation_log_mae"]
    return out


b5 = []
for pipe in RUNS:
    preds = resolved_predictions(pipe)
    reps = b2_predictions(pipe, preds) if preds else {}
    for variant, root in RUNS[pipe][2].items():
        cv = cv_scores(pipe, root)
        if not cv or variant not in preds:
            continue
        res_mae = preds[variant].groupby("role").abs_err.mean()
        rep_mae = reps[variant].groupby("role").abs_err.mean() if variant in reps else pd.Series(dtype=float)
        for role, c in cv.items():
            r = role if role in res_mae.index else next((x for x in res_mae.index if x.startswith(role)), None)
            b5.append({"pipeline": pipe, "variant": variant, "role": r or role, "CV log MAE": c,
                       "App-App resolved log MAE": res_mae.get(r, np.nan),
                       "App-App rep-avg log MAE": rep_mae.get(r, np.nan)})
if b5:
    b5 = pd.DataFrame(b5)
    b5.to_csv(OUT_DIR / "socket_aware_cv_B5.csv", index=False)
    p()
    p("### B5. CV (App-Inhibitor crossed CV of the selected recipe) vs App-App error")
    for (pipe, variant), g in b5.groupby(["pipeline", "variant"], sort=False):
        g = g.dropna(subset=["CV log MAE"]).copy()
        g["CV rank"] = g["CV log MAE"].rank().astype(int)
        g["App-App rank (resolved)"] = g["App-App resolved log MAE"].rank().astype(int)
        rho = stats.spearmanr(g["CV log MAE"], g["App-App resolved log MAE"]).statistic if len(g) > 2 else np.nan
        rho2 = stats.spearmanr(g["CV log MAE"], g["App-App rep-avg log MAE"]).statistic if len(g) > 2 else np.nan
        p()
        p(f"#### {PIPE_LABEL[pipe]} [{variant}]: CV top pick = {LABEL.get(g.loc[g['CV log MAE'].idxmin(), 'role'])}; "
          f"Spearman(CV, App-App) resolved {rho:+.2f}, rep-avg {rho2:+.2f}")
        g["method"] = g.role.map(LABEL)
        p(md(g.set_index("method").drop(columns=["pipeline", "variant", "role"])))


num = OUT_DIR / "socket_aware_numbers.md"
txt = num.read_text() if num.exists() else ""
txt = txt.split("\n## Part B")[0].rstrip() + "\n\n" + log.getvalue()
num.write_text(txt)
