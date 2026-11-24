"""Plan v2 scorer (runs on the analysis machine on outputs copied back from agera).

  python scripts/dissertation_figs/v2/v2_score.py --run scripts/dissertation_figs/agera_out/<run_id> \
      [--compare <other run dir>] [--tag <name>]

Metrics per (tier, target, socket mode, method, form, seed), on socket-resolved test cells (unclipped log ratios):
  log MAE; baseline MAE (isolated-speed baseline = predict 0); skill = 1 - MAE/MAE_base; bias = mean(pred - y);
  Spearman(pred, y); false-benign rate at 1.05 = among cells with y >= ln 1.05, fraction predicted < ln 1.05.
Restricted tiers pool all splits of a seed. Deterministic learners (ridge, lasso, svr) are fit once per split and
broadcast to every seed (model SD = 0). Absolute/delta have no BD form: they are listed once as form "none".
Paired pair-cluster bootstrap per seed (1,000 draws, seed 923; clusters = unordered co-runs, carrying every split
in which the pair is a test cell) of MAE(method) - MAE(baseline); also aware - blind and BD - 1W contrasts.
"Consistent" = same sign of the point estimate in >= 9/10 seeds (all seeds if fewer).
Sensitivity: the same metrics without self-pair rows.

Writes output/v2_pred_<tag>_{metrics,summary,contrasts,breakdown,hp_selected,hp_edges}.csv and
output/v2_pred_<tag>.md.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output"
LOG105 = float(np.log(1.05))
NAMES = {"amg": "AMG2023", "beatnik": "Beatnik", "fiesta": "FIESTA", "kripke": "Kripke", "laghos": "Laghos",
         "lammps": "LAMMPS", "minife": "MiniFE", "minivite": "miniVite", "quicksilver": "Quicksilver",
         "tricount": "TriC"}
OUT_OF_HULL = {"quicksilver"}  # flagged in breakdowns (farthest from every inhibitor config)
METHOD_ORDER = ["ridge", "lasso", "svr", "rf", "xgb", "xgb_d124", "mlp", "absolute", "delta"]
DET = {"ridge", "lasso", "svr"}
B, BOOT_SEED = 1000, 923


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def read_preds(run: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(run / "preds" / "*.csv.gz")))
    if not files:
        raise SystemExit(f"no predictions under {run}/preds")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load(run: Path, bundle: Path, xgb_rerun: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """xgb_rerun: follow-up run with the deeper xgb grid; its xgb rows replace the main run's, whose xgb rows are
    kept as method 'xgb_d124' (sensitivity)."""
    cfg = json.loads((run / "run_config.json").read_text())
    rows = pd.read_csv(bundle / "appapp_rows.csv").sort_values(["victim", "corunner", "s_V"]).reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows))
    P = read_preds(run)
    if xgb_rerun is not None:
        X = read_preds(xgb_rerun)
        X = X[X.method == "xgb"]
        P.loc[P.method == "xgb", "method"] = "xgb_d124"
        P = pd.concat([P, X], ignore_index=True)
    P = P.merge(rows[["row_id", "victim", "corunner", "s_V", "is_self", "pair", "y_mpi", "y_tt"]], on="row_id",
                validate="many_to_one")
    P["y"] = np.where(P.target == "mpi", P.y_mpi, P.y_tt)
    P = P.drop(columns=["y_mpi", "y_tt"])
    # broadcast deterministic restricted fits to every seed
    seeds = cfg["seeds"]
    det = P[(P.method.isin(DET)) & (P.tier == "restricted")]
    if len(det):
        rest = P.drop(det.index)
        det = pd.concat([det.assign(seed=s) for s in seeds], ignore_index=True)
        P = pd.concat([rest, det], ignore_index=True)
    return P, cfg


def metrics(g: pd.DataFrame) -> dict:
    e = g.y_pred - g.y
    mae, base = e.abs().mean(), g.y.abs().mean()
    harm = g.y >= LOG105
    return dict(n=len(g), log_mae=mae, base_mae=base, skill=1 - mae / base, bias=e.mean(),
                mean_y=g.y.mean(), mean_pred=g.y_pred.mean(),
                spearman=stats.spearmanr(g.y_pred, g.y).statistic if g.y_pred.nunique() > 1 else np.nan,
                n_harmful=int(harm.sum()),
                fb105=float((g.y_pred[harm] < LOG105).mean()) if harm.any() else np.nan)


KEYS = ["test_tier", "target", "socket", "method", "form"]


def per_seed(P: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for k, g in P.groupby(KEYS + ["seed"]):
        recs.append({**dict(zip(KEYS + ["seed"], k)), **metrics(g)})
    return pd.DataFrame(recs)


def summarize(M: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for k, g in M.groupby(KEYS):
        d = g.log_mae - g.base_mae
        need = 9 if len(g) == 10 else len(g)
        cons = "below" if (d < 0).sum() >= need else ("above" if (d > 0).sum() >= need else "mixed")
        recs.append({**dict(zip(KEYS, k)), "n_seeds": len(g), "n_cells_mean": g.n.mean(),
                     "log_mae_mean": g.log_mae.mean(), "log_mae_sd": g.log_mae.std(ddof=1) if len(g) > 1 else 0.0,
                     "base_mae_mean": g.base_mae.mean(), "skill_mean": g.skill.mean(),
                     "skill_sd": g.skill.std(ddof=1) if len(g) > 1 else 0.0, "bias_mean": g.bias.mean(),
                     "spearman_mean": g.spearman.mean(), "fb105_mean": g.fb105.mean(),
                     "n_harmful_mean": g.n_harmful.mean(), "mean_y": g.mean_y.mean(),
                     "mean_pred": g.mean_pred.mean(),
                     "seeds_better_than_base": int((d < 0).sum()), "consistent_vs_base": cons})
    S = pd.DataFrame(recs)
    S["m_order"] = S.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    return S.sort_values(["target", "test_tier", "socket", "form", "m_order"]).drop(columns="m_order")


def cluster_sums(g: pd.DataFrame, col: str) -> pd.DataFrame:
    return g.assign(a=(g[col] - g.y).abs()).groupby("pair").agg(s=("a", "sum"), n=("a", "size"))


def paired_boot(e1: pd.Series, e2: pd.Series, n: pd.Series, rng_seed=BOOT_SEED):
    """e1, e2: per-cluster sums of |error|; n: per-cluster cell counts (aligned). Returns diff, lo, hi of MAE1-MAE2."""
    e1, e2, n = e1.to_numpy(float), e2.to_numpy(float), n.to_numpy(float)
    point = e1.sum() / n.sum() - e2.sum() / n.sum()
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(len(n), size=(B, len(n)))
    nn = n[idx].sum(1)
    d = e1[idx].sum(1) / nn - e2[idx].sum(1) / nn
    return point, *np.percentile(d, [2.5, 97.5])


def contrasts(P: pd.DataFrame) -> pd.DataFrame:
    recs = []
    # (1) method - baseline
    for k, g in P.groupby(KEYS + ["seed"]):
        cs = g.assign(a=(g.y_pred - g.y).abs(), b=g.y.abs()).groupby("pair").agg(a=("a", "sum"), b=("b", "sum"),
                                                                                  n=("a", "size"))
        pt, lo, hi = paired_boot(cs.a, cs.b, cs.n)
        recs.append({**dict(zip(KEYS + ["seed"], k)), "contrast": "method-baseline", "diff": pt, "lo": lo, "hi": hi})
    # (2) aware - blind and (3) bd - 1w, matched on identical cells
    for name, col, a, b in (("aware-blind", "socket", "aware", "blind"), ("bd-1w", "form", "bd", "1w")):
        keys = [k for k in KEYS if k != col] + ["seed"]
        for k, g in P.groupby(keys):
            if not {a, b} <= set(g[col]):
                continue
            ga, gb = g[g[col] == a], g[g[col] == b]
            m = ga.merge(gb, on=["row_id", "tier", "split", "pair", "y"], suffixes=("_a", "_b"), validate="one_to_one")
            if len(m) != len(ga):
                raise RuntimeError(f"{name}: unmatched cells for {k}")
            cs = m.assign(ea=(m.y_pred_a - m.y).abs(), eb=(m.y_pred_b - m.y).abs()).groupby("pair").agg(
                ea=("ea", "sum"), eb=("eb", "sum"), n=("ea", "size"))
            pt, lo, hi = paired_boot(cs.ea, cs.eb, cs.n)
            recs.append({**dict(zip(keys, k)), col: f"{a}-{b}", "contrast": name, "diff": pt, "lo": lo, "hi": hi})
    C = pd.DataFrame(recs)
    C["ci_excl0"] = np.where(C.hi < 0, "below", np.where(C.lo > 0, "above", "n.s."))
    return C


def contrast_summary(C: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for k, g in C.groupby(KEYS + ["contrast"], dropna=False):
        need = 9 if len(g) == 10 else len(g)
        cons = "neg" if (g["diff"] < 0).sum() >= need else ("pos" if (g["diff"] > 0).sum() >= need else "mixed")
        recs.append({**dict(zip(KEYS + ["contrast"], k)), "n_seeds": len(g), "diff_mean": g["diff"].mean(),
                     "diff_sd": g["diff"].std(ddof=1) if len(g) > 1 else 0.0,
                     "ci_lo_mean": g.lo.mean(), "ci_hi_mean": g.hi.mean(),
                     "seeds_diff_neg": int((g["diff"] < 0).sum()),
                     "seeds_ci_below0": int((g.ci_excl0 == "below").sum()),
                     "seeds_ci_above0": int((g.ci_excl0 == "above").sum()), "consistent_sign": cons})
    return pd.DataFrame(recs)


def breakdown(P: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for by in ("victim", "corunner", "s_V"):
        Q = P[~P.is_self] if by == "s_V" else P
        for k, g in Q.groupby(KEYS + [by, "seed"]):
            e = (g.y_pred - g.y).abs().mean()
            recs.append({**dict(zip(KEYS, k[:-2])), "by": by, "level": k[-2], "seed": k[-1], "n": len(g),
                         "log_mae": e, "base_mae": g.y.abs().mean(), "bias": (g.y_pred - g.y).mean()})
    D = pd.DataFrame(recs)
    out = D.groupby(KEYS + ["by", "level"]).agg(n=("n", "mean"), log_mae=("log_mae", "mean"),
                                                log_mae_sd=("log_mae", "std"), base_mae=("base_mae", "mean"),
                                                bias=("bias", "mean")).reset_index()
    out["out_of_hull"] = out.level.isin(OUT_OF_HULL)
    return out


def hp_tables(run: Path, xgb_rerun: Path | None = None):
    H = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(run / "hp" / "*.csv")))], ignore_index=True)
    if xgb_rerun is not None:
        X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(xgb_rerun / "hp" / "*xgb*.csv")))],
                      ignore_index=True)
        H.loc[H.method == "xgb", "method"] = "xgb_d124"
        H = pd.concat([H, X[X.method == "xgb"]], ignore_index=True)
    sel = H[H.chosen].copy()
    recs = []
    for method, g in H.groupby("method"):
        for col in [c for c in g.columns if c.startswith("hp_") and g[c].notna().any()]:
            vals = g[col].dropna().unique()
            num = pd.to_numeric(pd.Series(vals), errors="coerce")
            chosen = sel[sel.method == method][col]
            n = len(chosen)
            if num.notna().all() and len(vals) > 1:
                lo, hi = num.min(), num.max()
                c = pd.to_numeric(chosen, errors="coerce")
                recs.append(dict(method=method, axis=col[3:], values=sorted(num.tolist()), n_tasks=n,
                                 frac_low_edge=float((c == lo).mean()), frac_high_edge=float((c == hi).mean())))
            else:
                vc = chosen.astype(str).value_counts(normalize=True)
                recs.append(dict(method=method, axis=col[3:], values=sorted(map(str, vals)), n_tasks=n,
                                 frac_low_edge=np.nan, frac_high_edge=np.nan,
                                 categorical_shares=vc.round(3).to_dict()))
    E = pd.DataFrame(recs)
    E["flag_gt30"] = (E.frac_low_edge > 0.3) | (E.frac_high_edge > 0.3)
    dist = []
    for method, g in sel.groupby("method"):
        for col in [c for c in g.columns if c.startswith("hp_") and g[c].notna().any()]:
            vc = g[col].astype(str).value_counts(normalize=True).sort_index()
            for v, share in vc.items():
                dist.append(dict(method=method, axis=col[3:], value=v, share=share, n_tasks=len(g)))
    return sel, E, pd.DataFrame(dist)


def spread(P: pd.DataFrame) -> pd.DataFrame:
    """Per (cell group, seed): SD of predictions across test rows vs SD of the measured y (collapse check)."""
    recs = []
    for k, g in P.groupby(KEYS + ["seed"]):
        recs.append({**dict(zip(KEYS + ["seed"], k)), "sd_pred": g.y_pred.std(), "sd_y": g.y.std(),
                     "range_pred": g.y_pred.max() - g.y_pred.min()})
    R = pd.DataFrame(recs)
    out = R.groupby(KEYS).agg(sd_pred_mean=("sd_pred", "mean"), sd_pred_min=("sd_pred", "min"),
                              sd_pred_max=("sd_pred", "max"), sd_y_mean=("sd_y", "mean"),
                              range_pred_mean=("range_pred", "mean")).reset_index()
    out["sd_ratio"] = out.sd_pred_mean / out.sd_y_mean
    return out


def compare_runs(a: Path, b: Path) -> str:
    fa = sorted(Path(a, "preds").glob("*.csv.gz"))
    nb = {f.name for f in Path(b, "preds").glob("*.csv.gz")}
    same, diff, missing = 0, [], []
    for f in fa:
        if f.name not in nb:
            missing.append(f.name)
            continue
        x, y = pd.read_csv(f), pd.read_csv(Path(b, "preds", f.name))
        if x.shape == y.shape and (x.y_pred.to_numpy() == y.y_pred.to_numpy()).all() and x.row_id.equals(y.row_id):
            same += 1
        else:
            diff.append((f.name, float(np.max(np.abs(x.y_pred.to_numpy() - y.y_pred.to_numpy())))
                         if x.shape == y.shape else "shape"))
    hd = 0
    for f in sorted(Path(a, "hp").glob("*.csv")):
        g = Path(b, "hp", f.name)
        if g.exists() and not pd.read_csv(f).equals(pd.read_csv(g)):
            hd += 1
    return (f"Determinism: {same}/{len(fa)} prediction files bit-identical between {a.name} and {b.name}; "
            f"differing {len(diff)} {diff[:5]}; missing in B {len(missing)}; HP tables differing {hd}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=HERE.parent / "agera_bundle")
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--xgb-rerun", type=Path, help="follow-up xgb run (max_depth 1-8) replacing the main xgb rows")
    args = ap.parse_args()
    tag = args.tag or args.run.name
    P, cfg = load(args.run, args.bundle, args.xgb_rerun)
    lines = [f"# v2 prediction results: {tag}\n", f"Run: `{args.run}`; versions {cfg['versions']}; "
             f"seeds {cfg['seeds']}; tasks {cfg['n_tasks']}.\n"]
    n_err = len(list((args.run / "errors").glob("*.txt"))) if (args.run / "errors").exists() else 0
    lines.append(f"Prediction cells {len(P):,}; error files {n_err}.\n")
    if args.compare:
        lines.append(compare_runs(args.run, args.compare) + "\n")

    M = per_seed(P)
    S = summarize(M)
    M_ns = per_seed(P[~P.is_self])
    S_ns = summarize(M_ns)
    C = contrasts(P)
    CS = contrast_summary(C)
    D = breakdown(P)
    sel, E, HD = hp_tables(args.run, args.xgb_rerun)
    SP = spread(P)
    base = CS[CS.contrast == "method-baseline"][KEYS + ["ci_lo_mean", "ci_hi_mean", "seeds_ci_below0",
                                                         "seeds_ci_above0"]]
    S = S.merge(base, on=KEYS, how="left")
    S_ns = S_ns.merge(CS[CS.contrast == "method-baseline"][KEYS], on=KEYS, how="left")
    if args.xgb_rerun:
        lines.append(f"xgb: rows from the follow-up run `{args.xgb_rerun}` (max_depth 1,2,4,6,8); the main run's "
                     "xgb (max_depth 1,2,4) is kept as method `xgb_d124` (sensitivity).\n")
    for name, frame in (("metrics", M), ("summary", S), ("summary_noself", S_ns), ("contrasts", C),
                        ("contrast_summary", CS), ("breakdown", D), ("hp_selected", sel), ("hp_edges", E),
                        ("hp_distribution", HD), ("pred_spread", SP)):
        frame.to_csv(OUT / f"v2_pred_{tag}_{name}.csv", index=False)

    cols = ["test_tier", "socket", "form", "method", "n_seeds", "n_cells_mean", "log_mae_mean", "log_mae_sd",
            "base_mae_mean", "skill_mean", "skill_sd", "bias_mean", "spearman_mean", "fb105_mean",
            "seeds_better_than_base", "consistent_vs_base", "ci_lo_mean", "ci_hi_mean", "seeds_ci_below0",
            "seeds_ci_above0"]
    for tgt in ("mpi", "tt"):
        s = S[S.target == tgt]
        if len(s):
            lines.append(f"\n## Main table: {'MPI time' if tgt == 'mpi' else 'application time'}\n")
            lines.append(md(s[cols]))
    lines.append("\n## Significant vs baseline: pair-cluster CI of MAE(method) - MAE(baseline) below 0 in >= 9/10 "
                 "seeds\n")
    sig = S[S.seeds_ci_below0 >= 9]
    lines.append(md(sig[["target", "test_tier", "socket", "form", "method", "skill_mean", "skill_sd",
                         "seeds_better_than_base", "seeds_ci_below0", "ci_lo_mean", "ci_hi_mean"]]))
    worse = S[S.seeds_ci_above0 >= 9]
    lines.append(f"\nSignificantly worse than baseline (CI above 0 in >= 9 seeds): {len(worse)} cells.\n")
    lines.append(md(worse[["target", "test_tier", "socket", "form", "method", "skill_mean", "seeds_ci_above0"]]))
    lines.append("\n## Harmful outcomes (measured y >= ln 1.05): the FB@1.05 denominator per seed\n")
    hh = S[(S.method == 'ridge') & (S.socket == 'aware') & (S.form == '1w')]
    hh = hh.assign(share_harmful=hh.n_harmful_mean / hh.n_cells_mean)
    hh_ns = S_ns[(S_ns.method == 'ridge') & (S_ns.socket == 'aware') & (S_ns.form == '1w')]
    hh = hh.merge(hh_ns[["test_tier", "target", "n_harmful_mean", "n_cells_mean"]]
                  .rename(columns={"n_harmful_mean": "n_harmful_noself", "n_cells_mean": "n_cells_noself"}),
                  on=["test_tier", "target"])
    lines.append(md(hh[["target", "test_tier", "n_cells_mean", "n_harmful_mean", "share_harmful",
                        "n_cells_noself", "n_harmful_noself"]], fmt="{:.3f}"))
    rows_all = pd.read_csv(args.bundle / "appapp_rows.csv")
    for t in ("mpi", "tt"):
        y = rows_all[f"y_{t}"]
        lines.append(f"\nAll 176 App-App rows, {t}: {int((y >= LOG105).sum())} harmful "
                     f"({(y >= LOG105).mean():.1%}).")
    lines.append("\n\n## Mean measured vs mean predicted log ratio (response models vs App-App learners)\n")
    mm = S[S.method.isin(["absolute", "delta", "rf", "svr"]) & S.form.isin(["none", "1w"])]
    lines.append(md(mm[["target", "test_tier", "socket", "method", "mean_y", "mean_pred", "bias_mean"]]))
    ai = pd.read_csv(args.bundle / "appinh_resolved.csv")
    ycols = [c for c in ai.columns if c in ("y_mpi", "y_tt")]
    lines.append("\nTraining targets of the response models (App-Inh resolved cells, n = %d): " % len(ai) + "; ".join(
        f"{c}: mean {ai[c].mean():.4f}, median {ai[c].median():.4f} (ratio {np.exp(ai[c].median()):.3f})"
        for c in ycols))
    lines.append("App-App rows (n = 176): " + "; ".join(
        f"y_{t}: mean {rows_all[f'y_{t}'].mean():.4f}, median {rows_all[f'y_{t}'].median():.4f} "
        f"(ratio {np.exp(rows_all[f'y_{t}'].median()):.3f})" for t in ("mpi", "tt")) + "\n")
    lines.append("\n## Sensitivity without self-pairs (skill mean)\n")
    piv = S_ns.pivot_table(index=["target", "test_tier", "socket", "form"], columns="method", values="skill_mean")
    lines.append(md(piv.reset_index()))
    lines.append("\n## Paired contrasts (per-seed pair-cluster bootstrap)\n")
    lines.append(md(CS[["contrast", "target", "test_tier", "socket", "form", "method", "n_seeds", "diff_mean",
                        "diff_sd", "seeds_ci_below0", "seeds_ci_above0", "consistent_sign"]]
                    .sort_values(["contrast", "target", "test_tier", "socket", "form", "method"])))
    lines.append("\n## Prediction spread (collapse check): SD of predictions across test cells / SD of measured y\n")
    sp = SP.pivot_table(index=["target", "test_tier", "socket"], columns=["method"], values="sd_ratio")
    lines.append(md(sp.reset_index(), fmt="{:.2f}"))
    lines.append("\n## HP grid edges (share of tasks choosing the lowest / highest value; flag > 30%)\n")
    lines.append(md(E.drop(columns=[c for c in ("categorical_shares",) if c not in E]), fmt="{:.2f}"))
    lines.append("\n## Selected-HP distribution (share of tasks)\n")
    lines.append(md(HD, fmt="{:.2f}"))
    (OUT / f"v2_pred_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"wrote output/v2_pred_{tag}*.csv / .md")


if __name__ == "__main__":
    main()
