"""Seed batch S0 pilot: final-seed sensitivity of the frozen (fixed_global) BD-MPI blind recipe.

Input: socket_batch2/seed_pilot/P0_bd_mpi_blind (E0_bd_mpi config = the blind CV selection, final_seeds 0..29;
seeds 0-4 are the original 5). The pipeline's ensemble = mean of per-seed latents, then observe (max(0, .)), so any
k-seed sub-ensemble is rebuilt exactly from seed_predictions.csv.

Levels: native = the pipeline's own truth (canonical snapshot, 90 non-self outcomes; the §8 numbers);
        rep-avg = socket-cell truth averaged over replicates (90, as B2); resolved = 166 socket cells (as B1).
Metrics per sub-ensemble: log MAE; Δ% vs the isolated-speed baseline (native); FB@1.05 (native, pair level);
Spearman (native).
Writes output/seed_pilot_{numbers.md,subsets.csv}.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "socket_aware_score.py").read_text()
ns: dict = {"__name__": "socket_aware_score_helpers", "__file__": str(HERE / "socket_aware_score.py")}
exec(compile(src[: src.index("art = json.loads")], "socket_aware_score.py", "exec"), ns)
np, pd, EXP, OUT_DIR, LABEL, md, cells = ns["np"], ns["pd"], ns["EXP"], ns["OUT_DIR"], ns["LABEL"], ns["md"], ns["cells"]
from scipy import stats  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "seed_pilot/P0_bd_mpi_blind"
TARGET = "mpi"
OLD = [0, 1, 2, 3, 4]
SE_GOAL = 0.002
ROLES = ["absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic", "joint_direct", "joint_cal",
         "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal", "bwall_cal", "bfwd_cal"]
CODE2ROLE = {v[1]: k for k, v in ns["ROLE_CODE"].items() if v[1]}
FB_ROLES = ["absolute", "joint_cal", "bprof_cal", "bctx_cal"]
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


sp = pd.read_csv(RUN / "seed_predictions.csv")
sp = sp[(sp.evaluation_method == "random_split") & (sp.anchor_budget.astype(str) == "all")
        & sp.method.isin(CODE2ROLE) & (sp.victim_id != sp.aggressor_id)]
sp = sp.assign(role=sp.method.map(CODE2ROLE))
SEEDS = sorted(sp.seed.unique())
L = sp.pivot_table(index=["role", "victim_id", "aggressor_id"], columns="seed",
                   values="latent_predicted_log_slowdown", aggfunc="first")
assert not L.isna().any().any()
truth = sp.groupby(["role", "victim_id", "aggressor_id"]).true_slowdown.first().reindex(L.index)
roles_present = [r for r in ROLES if r in L.index.get_level_values(0)]

# check: the 5-seed sub-ensemble reproduces the original E0 / CV directional predictions
e0 = pd.read_csv(EXP / "E0_bd_mpi/directional_predictions.csv")
e0 = e0[(e0.evaluation_method == "random_split") & (e0.anchor_budget.astype(str) == "all") & e0.method.isin(CODE2ROLE)
        & (e0.victim_id != e0.aggressor_id)].assign(role=lambda x: x.method.map(CODE2ROLE))
e0 = e0.set_index(["role", "victim_id", "aggressor_id"]).latent_predicted_log_slowdown.reindex(L.index)
dev0 = float(np.abs(L[OLD].mean(axis=1) - e0).max())

# per-role index arrays for the three levels
rep = cells.assign(T=lambda x: x[f"ratio_{TARGET}"] * x.n_reps).groupby(["victim", "aggressor", "pair"]).agg(
    T=("T", "sum"), n=("n_reps", "sum")).reset_index()
rep["y"] = np.log(np.maximum(1.0, rep["T"] / rep["n"]))
LEV = {}
for role in roles_present + ["constant_1"]:
    sub = L.loc[role]
    pos = {k: i for i, k in enumerate(sub.index)}
    rows = np.array([L.index.get_loc((role, *k)) for k in sub.index])
    pair = np.array([" | ".join(sorted(k)) for k in sub.index])
    LEV[(role, "native")] = (rows, np.log(truth.loc[role].to_numpy()), pair)
    r = rep.set_index(["victim", "aggressor"]).reindex(sub.index)
    LEV[(role, "rep-avg")] = (rows, r.y.to_numpy(), pair)
    c = cells[f"y_{TARGET}"].to_numpy()
    ridx = np.array([pos[(v, a)] for v, a in zip(cells.victim, cells.aggressor)])
    LEV[(role, "resolved")] = (rows[ridx], c, None)
LEVELS = ["native", "rep-avg", "resolved"]
Lv = L.to_numpy()
cols = {s: i for i, s in enumerate(L.columns)}


def fb(pred, y, pair):
    d = pd.DataFrame({"pair": pair, "p": pred, "y": y}).groupby("pair").max()
    harm = d.y >= np.log(1.05)
    return float(((d.p < np.log(1.05)) & harm).sum() / harm.sum())


def score(seeds) -> dict:
    lat = Lv[:, [cols[s] for s in seeds]].mean(axis=1)
    pred_all = np.maximum(0.0, lat)
    out = {}
    for (role, lvl), (rows, y, pair) in LEV.items():
        pr = pred_all[rows]
        out[(role, lvl, "log MAE")] = float(np.abs(pr - y).mean())
        if lvl == "native" and role != "constant_1":
            out[(role, lvl, "Spearman")] = float(stats.spearmanr(pr, y).statistic) if np.ptp(pr) > 0 else np.nan
            out[(role, lvl, "floor frac")] = float((pr == 0).mean())
            if role in FB_ROLES:
                out[(role, lvl, "FB@1.05")] = fb(pr, y, pair)
    for role in roles_present:
        out[(role, "native", "Δ%")] = 100 * (out[(role, "native", "log MAE")] / out[("constant_1", "native", "log MAE")] - 1)
    return out


rng = np.random.default_rng(20261109)
N = len(SEEDS)
recs = []
for s in SEEDS:
    recs.append({"kind": "single", "k": 1, "id": str(s), **score([s])})
groups = [SEEDS[i:i + 5] for i in range(0, N - N % 5, 5)]
for g in groups:
    recs.append({"kind": "disjoint5", "k": 5, "id": ",".join(map(str, g)), **score(g)})
KS = [k for k in [1, 2, 3, 5, 8, 10, 15] if k < N] or [1]
for k in KS:
    for _ in range(300 if k > 1 else 0):
        s = sorted(rng.choice(SEEDS, k, replace=False).tolist())
        recs.append({"kind": "random", "k": k, "id": ",".join(map(str, s)), **score(s)})
recs.append({"kind": "full", "k": N, "id": f"0-{N - 1}", **score(SEEDS)})
R = pd.DataFrame(recs)
metric_cols = [c for c in R.columns if isinstance(c, tuple)]
R.rename(columns={c: "|".join(c) for c in metric_cols}).to_csv(OUT_DIR / "seed_pilot_subsets.csv", index=False)


def col(role, lvl, m):
    return (role, lvl, m)


old = R[(R.kind == "disjoint5") & (R.id == "0,1,2,3,4")].iloc[0]
full = R[R.kind == "full"].iloc[0]
r5 = R[(R.kind == "random") & (R.k == 5)]
single = R[R.kind == "single"]
d5 = R[R.kind == "disjoint5"]

p(f"## S0. Seed pilot, BD-MPI blind, frozen CV recipe, {N} final seeds ({RUN.name})")
p()
p(f"- Seeds 0–4 sub-ensemble vs the original 5-seed run (E0 = CV run), max |Δ latent| = {dev0:.1e}.")
p("- Sub-ensemble = mean of per-seed latents, then max(0, ·), exactly as the pipeline builds its ensemble.")
p(f"- Disjoint 5-seed groups: {len(groups)}; random 5-subsets: {len(r5)} (without replacement from {N}).")
p()

for lvl in LEVELS:
    rows = []
    for role in roles_present:
        c = col(role, lvl, "log MAE")
        k_sd = {}
        for k in KS:
            v = R[(R.k == k) & (R.kind.isin(["random", "single"] if k == 1 else ["random"]))][c]
            k_sd[k] = float(v.std(ddof=1)) * np.sqrt((N - 1) / (N - k))  # finite-population correction
        a = float(np.median([k_sd[k] * np.sqrt(k) for k in KS if k <= 10]))
        pct = float((r5[c] < old[c]).mean() * 100)
        rows.append({"method": LABEL[role], "1-seed SD": k_sd[1], "5-seed SD (disjoint)": d5[c].std(ddof=1),
                     "5-seed SD (random, FPC)": k_sd[5], "10-seed SD": k_sd[10],
                     "old 5": old[c], f"full {N}": full[c], "old − full": old[c] - full[c],
                     "old pctile in 5-subsets": pct, "N for SE ≤ 0.002": int(np.ceil((a / SE_GOAL) ** 2))})
    p(f"### log MAE, {lvl} outcomes")
    p(md(pd.DataFrame(rows).set_index("method"), "{:.4f}"))
    p()

p("### §8 headline quantities (native truth): old 5 seeds vs the distribution of 5-seed sub-ensembles")
rows = []
for role in roles_present:
    for m in ("Δ%", "FB@1.05", "Spearman", "floor frac"):
        c = col(role, "native", m)
        if c not in R:
            continue
        v = r5[c].dropna()
        if v.empty:
            continue
        rows.append({"quantity": f"{LABEL[role]}: {m}", "old 5": old[c], f"full {N}": full[c],
                     "5-subset median": v.median(), "5-subset 2.5%": v.quantile(.025), "5-subset 97.5%": v.quantile(.975),
                     "1-seed min": single[c].min(), "1-seed max": single[c].max(),
                     "old pctile": float((v < old[c]).mean() * 100)})
p(md(pd.DataFrame(rows).set_index("quantity"), "{:.4f}"))
p()

p("### Convergence: mean and SD of the k-seed ensemble log MAE (native), by k")
rows = []
for role in roles_present:
    c = col(role, "native", "log MAE")
    r = {"method": LABEL[role]}
    for k in KS:
        v = R[(R.k == k) & (R.kind.isin(["random", "single"] if k == 1 else ["random"]))][c]
        r[f"k={k}"] = f"{v.mean():.4f} ± {v.std(ddof=1) * np.sqrt((N - 1) / (N - k)):.4f}"
    r[f"k={N}"] = f"{full[c]:.4f}"
    rows.append(r)
p(md(pd.DataFrame(rows).set_index("method")))
p()

# key contrasts: seed SD of paired differences
CONTRASTS = [("absolute", "constant_1"), ("joint_cal", "absolute"), ("bprof_cal", "joint_cal"),
             ("joint_direct", "absolute"), ("bctx_direct", "bprof_direct"), ("bctx_cal", "bprof_cal"),
             ("delta_uniform", "absolute"), ("joint_cal", "joint_direct")]
rows = []
for a, b in CONTRASTS:
    if a not in roles_present or (b not in roles_present and b != "constant_1"):
        continue
    ca, cb = col(a, "native", "log MAE"), col(b, "native", "log MAE")
    d1 = single[ca] - single[cb]
    dk = {k: (R[(R.k == k) & (R.kind == "random")][ca] - R[(R.k == k) & (R.kind == "random")][cb]) for k in (5, 10)}
    sd1 = float(d1.std(ddof=1))
    rows.append({"contrast (native log MAE)": f"{LABEL[a]} − {LABEL[b]}", "old 5": old[ca] - old[cb],
                 f"full {N}": full[ca] - full[cb], "1-seed SD": sd1,
                 "5-seed SD": float(dk[5].std(ddof=1)) * np.sqrt((N - 1) / (N - 5)),
                 "10-seed SD": float(dk[10].std(ddof=1)) * np.sqrt((N - 1) / (N - 10)),
                 "sign agrees with full, % of 5-subsets": float((np.sign(dk[5]) == np.sign(full[ca] - full[cb])).mean() * 100)})
p("### Contrasts: seed SD of paired differences")
p(md(pd.DataFrame(rows).set_index("contrast (native log MAE)"), "{:.4f}"))
p()

# SE at N (delete-one jackknife of the full ensemble) and SD(k) ~ c k^-beta fitted on k <= N/3 (FPC-corrected)
p(f"### Seed SE of the full {N}-seed ensemble (delete-one jackknife) and SD(k) power-law fit (k = 2..{N // 3})")
rows = []
jk = [score([t for t in SEEDS if t != s]) for s in SEEDS]
kfit = [k for k in KS if 2 <= k <= N // 3]
for lvl in ("native", "resolved"):
    for role in roles_present:
        c = col(role, lvl, "log MAE")
        v = np.array([j[c] for j in jk])
        se_jk = float(np.sqrt((N - 1) / N * ((v - v.mean()) ** 2).sum()))
        sd = [float(R[(R.k == k) & (R.kind == "random")][c].std(ddof=1)) * np.sqrt((N - 1) / (N - k)) for k in kfit]
        b, a = np.polyfit(np.log(kfit), np.log(sd), 1)
        beta = -b
        n_goal = float(np.exp((a - np.log(SE_GOAL)) / beta)) if beta > 0 else np.inf
        rows.append({"level": lvl, "method": LABEL[role], f"jackknife SE at N={N}": se_jk, "beta": beta,
                     "N for SE ≤ 0.002 (power fit)": int(np.ceil(n_goal)) if np.isfinite(n_goal) else -1})
p(md(pd.DataFrame(rows).set_index(["level", "method"]), "{:.4f}"))
p()
rows = []
for a_, b_ in CONTRASTS:
    if a_ not in roles_present or (b_ not in roles_present and b_ != "constant_1"):
        continue
    v = np.array([j[col(a_, "native", "log MAE")] - j[col(b_, "native", "log MAE")] for j in jk])
    rows.append({"contrast": f"{LABEL[a_]} − {LABEL[b_]}", f"full {N}": full[col(a_, "native", "log MAE")] - full[col(b_, "native", "log MAE")],
                 f"jackknife SE at N={N}": float(np.sqrt((N - 1) / N * ((v - v.mean()) ** 2).sum()))})
p(md(pd.DataFrame(rows).set_index("contrast"), "{:.4f}"))
p()

# §8 claims (BD-MPI-testable): fraction of random k-seed sub-ensembles in which each holds
CLAIMS = [
    ("absolute < baseline", lambda g: g("absolute", "Δ%") < 0),
    ("absolute at the floor on all 90 outcomes", lambda g: g("absolute", "floor frac") == 1),
    ("joint cal FB < absolute FB", lambda g: g("joint_cal", "FB@1.05") < g("absolute", "FB@1.05")),
    ("joint cal < baseline", lambda g: g("joint_cal", "Δ%") < 0),
    ("joint cal < absolute", lambda g: g("joint_cal", "log MAE") < g("absolute", "log MAE")),
    ("delta uniform > baseline", lambda g: g("delta_uniform", "Δ%") > 0),
    ("delta median > baseline", lambda g: g("delta_median", "Δ%") > 0),
    ("generic > baseline", lambda g: g("generic", "Δ%") > 0),
    ("C7 fingerprints hurt, direct (bctx > bprof)", lambda g: g("bctx_direct", "log MAE") > g("bprof_direct", "log MAE")),
    ("C7 fingerprints hurt, cal (bctx cal > bprof cal)", lambda g: g("bctx_cal", "log MAE") > g("bprof_cal", "log MAE")),
    ("C7 joint direct hurts vs absolute", lambda g: g("joint_direct", "log MAE") > g("absolute", "log MAE")),
    ("C7 reverse supervision helps, direct (bprof < joint)", lambda g: g("bprof_direct", "log MAE") < g("joint_direct", "log MAE")),
    ("C7 reverse supervision helps, cal", lambda g: g("bprof_cal", "log MAE") < g("joint_cal", "log MAE")),
    ("calibration helps joint (cal < direct)", lambda g: g("joint_cal", "log MAE") < g("joint_direct", "log MAE")),
]
rows = []
for name, fn in CLAIMS:
    r = {"claim (native, BD-MPI blind)": name, "old 5": bool(fn(lambda a, m: old[col(a, "native", m)]))}
    for k in (5, 10, 15):
        d = R[(R.kind == "random") & (R.k == k)]
        r[f"% of {k}-subsets"] = float(fn(lambda a, m: d[col(a, "native", m)]).mean() * 100)
    r[f"full {N}"] = bool(fn(lambda a, m: full[col(a, "native", m)]))
    rows.append(r)
p("### §8 claims: fraction of random k-seed sub-ensembles in which each holds")
p(md(pd.DataFrame(rows).set_index("claim (native, BD-MPI blind)"), "{:.1f}"))
p()
(OUT_DIR / "seed_pilot_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "seed_pilot_numbers.md")
