"""Seed batch scoring, parts (b) and (c): restricted training-size tiers at N = 30 final seeds.

(b) blind restricted tasks: BD (the B4 blind tasks that succeeded, MPI + TT) and 1W (sizes 4-8 x 10 splits, MPI + TT).
(c) BD socket-aware E1-style and D1 (constant-0 bit) at the 20 C4 cells.

Ensemble = mean of per-seed latents, then max(0, .) (as part a). Outcomes = the pipeline's native non-self directional
outcomes of each tier (one_known / zero_shot), anchor_budget 'all'. Cell log MAE = mean abs log error over the cell's
outcomes; pooled skill over a set of cells = 1 - mean(method cell log MAE) / mean(baseline cell log MAE) (the artifact's
C4 / learning-curve definition).
Uncertainty, shared across all cells (seeds are paired by id across tasks and runs):
- pairs-only 95% CI: the shared 1000 x seed-923 pair-cluster draws (45 unordered pairs), the same draw in every cell;
- two-level 95% CI: each pair draw paired with a seed resample (multinomial, rng 924);
- jackknife seed SE (delete-one-seed) of the deciding quantity, Δ = mean cell log MAE(method) - mean cell log MAE(base);
- 1000 random 5-seed sub-ensembles (rng 925): median and the % in which the claim holds.
Label rule (as part a): sign-unstable if jackknife SE > 0.002 and |Δ| < 2 SE (1.9-2.1 SE: the two-level CI decides);
otherwise 'CI excludes 0' / 'n.s.' by the two-level CI.
Writes output/seed_restricted_numbers.md and output/seed_restricted_{completion,c4,curves,fb,socket}.csv.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "socket_aware_score.py").read_text()
ns: dict = {"__name__": "socket_aware_score_helpers", "__file__": str(HERE / "socket_aware_score.py")}
exec(compile(src[: src.index("art = json.loads")], "socket_aware_score.py", "exec"), ns)
np, pd, OUT_DIR, LABEL, md, CELLS = ns["np"], ns["pd"], ns["OUT_DIR"], ns["LABEL"], ns["md"], ns["cells"]
PAIRS, DRAWS, ROLE_CODE = ns["PAIRS"], ns["DRAWS"], ns["ROLE_CODE"]

SB = Path.home() / "hpcResearch/experiment_outputs/seed_batch"
ART = json.loads((OUT_DIR / "artifact_data_20261017.json").read_text())
C4 = [(s, int(n)) for s, n in ART["common_cells"]]
PIPES = {"1W-MPI": ("1w", "mpi", "baseSnap"), "1W-TT": ("1w", "tt", "totalTime"), "BD-MPI": ("bd", "mpi", "bidir"),
         "BD-TT": ("bd", "tt", "bidirTT")}
ROLES_1W = ["absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic"]
ROLES_BD = ROLES_1W + ["joint_direct", "joint_cal", "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal"]
TIERS = ("one_known", "zero_shot")
SIZES = [4, 5, 6, 7, 8]
SE_GOAL = 0.002
LOG105 = np.log(1.05)
NB = len(DRAWS)
PIDX = {p_: i for i, p_ in enumerate(PAIRS)}
CPAIR = np.stack([np.bincount(d, minlength=len(PAIRS)) for d in DRAWS]).astype(float)  # 1000 x 45
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def pair_of(v, a):
    return " | ".join(sorted((v, a)))


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def jack_se(v):
    v = np.asarray(v, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))


def verdict(est, se, dt):
    lo, hi = ci(dt)
    ci0 = lo > 0 or hi < 0
    ratio = abs(est) / se if se > 0 else np.inf
    boundary = se > SE_GOAL and 1.9 <= ratio <= 2.1
    unstable = se > SE_GOAL and ratio < 2
    v = ("CI excludes 0" if ci0 else "n.s.") if (boundary or not unstable) else "sign-unstable"
    return v, ("YES" if (unstable and ci0 and not boundary) else ""), ("boundary: CI decides" if boundary else "")


# ------------------------------------------------------------------ loading
class Cell:
    """One task x tier: units (role, victim, aggressor[, s_V]) with truth y and a latent matrix L (units x seeds)."""

    def __init__(self, units: pd.DataFrame, L: np.ndarray, seeds: list[int], w=None, agg=None):
        self.u, self.L, self.seeds = units.reset_index(drop=True), L, seeds
        self.pidx = self.u.pair.map(PIDX).to_numpy()
        self.y = self.u.y.to_numpy()
        self.agg = agg  # rep-avg aggregation: (A matrix pairs-units x socket units)

    def preds(self, W):
        P = np.maximum(0.0, self.L @ W)
        if self.agg is not None:
            P = np.log(self.agg @ np.exp(P))
        return P


def code2role(fam):
    i = 0 if fam == "1w" else 1
    return {v[i]: k for k, v in ROLE_CODE.items() if v[i]}


def load_blind(d: Path, tier: str, fam: str, target: str, level: str = "native"):
    f = d / "evaluations" / tier / "seed_predictions.csv"
    df = pd.read_csv(f)
    c2r = code2role(fam)
    df = df[(df.anchor_budget.astype(str) == "all") & df.method.isin(c2r) & (df.victim_id != df.aggressor_id)]
    df = df.assign(role=df.method.map(c2r), victim=df.victim_id, aggressor=df.aggressor_id)
    piv = df.pivot_table(index=["role", "victim", "aggressor"], columns="seed", values="latent_predicted_log_slowdown")
    tru = df.groupby(["victim", "aggressor"]).true_slowdown.first()
    if "constant_1" not in piv.index.get_level_values(0):
        base = pd.DataFrame(0.0, index=pd.MultiIndex.from_tuples([("constant_1", v, a) for v, a in tru.index],
                                                                  names=piv.index.names), columns=piv.columns)
        piv = pd.concat([piv, base])
    piv = piv.dropna()
    u = piv.index.to_frame(index=False)
    u["pair"] = [pair_of(v, a) for v, a in zip(u.victim, u.aggressor)]
    seeds = [int(s) for s in piv.columns]
    if level == "native":
        u["y"] = np.log(np.maximum(1.0, tru.reindex(list(zip(u.victim, u.aggressor))).to_numpy()))
        return Cell(u, piv.to_numpy(), seeds)
    if level == "resolved":  # blind prediction applied to both socket cells; socket-cell truth
        cc = CELLS[["victim", "aggressor", "s_V", f"y_{target}"]].rename(columns={f"y_{target}": "y"})
        uu = u.reset_index().merge(cc, on=["victim", "aggressor"])
        return Cell(uu.drop(columns="index"), piv.to_numpy()[uu["index"].to_numpy()], seeds)
    rep = rep_truth(target)  # rep-avg truth
    uu = u.reset_index().merge(rep[["victim", "aggressor", "y"]], on=["victim", "aggressor"])
    return Cell(uu.drop(columns="index"), piv.to_numpy()[uu["index"].to_numpy()], seeds)


def rep_truth(target):
    rep = CELLS.assign(T=lambda x: x[f"ratio_{target}"] * x.n_reps).groupby(["victim", "aggressor", "pair"]).agg(
        T=("T", "sum"), n=("n_reps", "sum")).reset_index()
    rep["y"] = np.log(np.maximum(1.0, rep["T"] / rep["n"]))
    return rep


def load_socket(d: Path, tier: str, target: str, level: str):
    df = pd.read_csv(d / "evaluations" / tier / "seed_predictions_socket.csv")
    c2r = code2role("bd")
    df = df[df.method.isin(c2r) & (df.victim != df.aggressor)].assign(role=lambda x: x.method.map(c2r))
    piv = df.pivot_table(index=["role", "victim", "aggressor", "s_V"], columns="seed", values="latent").dropna()
    w = df.groupby(["victim", "s_V"]).victim_socket_replicates.first()
    u = piv.index.to_frame(index=False)
    if "constant_1" not in u.role.values:
        b = u[u.role == u.role.iloc[0]].assign(role="constant_1")
        u = pd.concat([u, b], ignore_index=True)
        L = np.vstack([piv.to_numpy(), np.zeros((len(b), piv.shape[1]))])
    else:
        L = piv.to_numpy()
    u["pair"] = [pair_of(v, a) for v, a in zip(u.victim, u.aggressor)]
    seeds = [int(s) for s in piv.columns]
    if level == "resolved":
        cc = CELLS[["victim", "aggressor", "s_V", f"y_{target}"]].rename(columns={f"y_{target}": "y"})
        uu = u.reset_index().merge(cc, on=["victim", "aggressor", "s_V"])
        return Cell(uu.drop(columns="index"), L[uu["index"].to_numpy()], seeds)
    # rep-avg: replicate-weighted slowdown mean over the victim's sockets, in slowdown space
    keys = u[["role", "victim", "aggressor", "pair"]].drop_duplicates().reset_index(drop=True)
    kidx = {k: i for i, k in enumerate(zip(keys.role, keys.victim, keys.aggressor))}
    wt = np.array([w.get((v, s), 0.0) for v, s in zip(u.victim, u.s_V)], float)
    A = np.zeros((len(keys), len(u)))
    for j, k in enumerate(zip(u.role, u.victim, u.aggressor)):
        A[kidx[k], j] = wt[j]
    A = A / A.sum(1, keepdims=True)
    keys = keys.merge(rep_truth(target)[["victim", "aggressor", "y"]], on=["victim", "aggressor"], how="left")
    c = Cell(keys, L, seeds, agg=A)
    return c


# ------------------------------------------------------------------ metric kernels over a set of cells
def cell_stats(cell: Cell, role: str, W: np.ndarray):
    """45 x D pair sums of abs error and 45 counts for one role."""
    m = (cell.u.role == role).to_numpy()
    if not m.any():
        return None, None
    P = cell.preds(W)[m]
    E = np.abs(P - cell.y[m, None])
    S = np.zeros((len(PAIRS), W.shape[1]))
    np.add.at(S, cell.pidx[m], E)
    n = np.bincount(cell.pidx[m], minlength=len(PAIRS)).astype(float)
    return S, n


def pooled(cells: list[Cell], role: str, W: np.ndarray, pair_draws: bool):
    """mean over cells of cell log MAE, per column of W (and per pair draw b when pair_draws: W has NB columns or 1)."""
    vals = []
    for c in cells:
        S, n = cell_stats(c, role, W)
        if S is None:
            vals.append(np.full(NB if pair_draws else W.shape[1], np.nan))
            continue
        if pair_draws:
            if S.shape[1] == 1:
                S = np.repeat(S, NB, 1)
            num = (CPAIR * S.T).sum(1)
            den = CPAIR @ n
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append(np.where(den > 0, num / den, np.nan))
        else:
            vals.append(S.sum(0) / n.sum())
    return np.nanmean(np.vstack(vals), 0)


def fb_pooled(cells: list[Cell], role: str, W: np.ndarray):
    """False-benign rate pooled over cells: unit = unordered pair within a cell; harmful if max true >= 1.05;
    missed if max predicted < 1.05. Returns per-column rate and the harmful-unit count."""
    miss, harm = 0.0, 0
    for c in cells:
        m = (c.u.role == role).to_numpy()
        if not m.any():
            continue
        P = c.preds(W)[m]
        pid = c.pidx[m]
        ty = np.full(len(PAIRS), -np.inf)
        np.maximum.at(ty, pid, c.y[m])
        pm = np.full((len(PAIRS), W.shape[1]), -np.inf)
        np.maximum.at(pm, pid, P)
        h = ty >= LOG105
        harm += int(h.sum())
        miss = miss + (pm[h] < LOG105).sum(0)
    return (miss / harm if harm else np.full(W.shape[1], np.nan)), harm


# ------------------------------------------------------------------ seed weights (shared)
S_N = 30
rng = np.random.default_rng(924)
WF = np.full((S_N, 1), 1.0 / S_N)
WJ = np.full((S_N, S_N), 1.0 / (S_N - 1))
np.fill_diagonal(WJ, 0.0)
WB = rng.multinomial(S_N, np.full(S_N, 1.0 / S_N), size=NB).T / S_N
rng5 = np.random.default_rng(925)
W5 = np.zeros((S_N, 1000))
for j in range(1000):
    W5[rng5.choice(S_N, 5, replace=False), j] = 0.2
W04 = np.zeros((S_N, 1))
W04[:5] = 0.2


def contrast(cells_a, cells_b, role_a, role_b):
    """Δ = mean cell log MAE (a, role_a) - (b, role_b); cells_b may be cells_a (method vs baseline)."""
    g = lambda W, pd_: pooled(cells_a, role_a, W, pd_) - pooled(cells_b, role_b, W, pd_)
    est = float(g(WF, False)[0])
    jk = g(WJ, False)
    dp, dt = g(WF, True), g(WB, True)
    d5 = g(W5, False)
    se = jack_se(jk)
    v, dis, note = verdict(est, se, dt)
    return {"Δ log MAE": est, "CI pairs": "[{:+.4f}, {:+.4f}]".format(*ci(dp)),
            "CI pairs×seeds": "[{:+.4f}, {:+.4f}]".format(*ci(dt)), "seed SE (jackknife)": se,
            "5-seed median Δ": float(np.median(d5)), "% 5-seed draws Δ < 0": float((d5 < 0).mean() * 100),
            "verdict": v, "jackknife vs CI disagree": dis, "note": note}


def skill_row(cells, role):
    b = lambda W, pd_: pooled(cells, "constant_1", W, pd_)
    m = lambda W, pd_: pooled(cells, role, W, pd_)
    sk = lambda W, pd_: 100 * (1 - m(W, pd_) / b(W, pd_))
    s5 = sk(W5, False)
    # wins: cells where the method's cell log MAE < baseline's (N = 30)
    wins = 0
    for c in cells:
        S, n = cell_stats(c, role, WF)
        S0, n0 = cell_stats(c, "constant_1", WF)
        if S is not None and S.sum() / n.sum() < S0.sum() / n0.sum():
            wins += 1
    r = {"skill % (N=30)": float(sk(WF, False)[0]), "skill % CI pairs": "[{:+.1f}, {:+.1f}]".format(*ci(sk(WF, True))),
         "skill % CI pairs×seeds": "[{:+.1f}, {:+.1f}]".format(*ci(sk(WB, True))),
         "skill % seeds 0–4": float(sk(W04, False)[0]), "5-seed median skill %": float(np.median(s5)),
         "5-seed 95%": "[{:+.1f}, {:+.1f}]".format(*ci(s5)), "% 5-seed draws skill > 0": float((s5 > 0).mean() * 100),
         "wins": f"{wins}/{len(cells)}"}
    c = contrast(cells, cells, role, "constant_1")
    r.update({"seed SE (Δ log MAE)": c["seed SE (jackknife)"], "verdict": c["verdict"],
              "jackknife vs CI disagree": c["jackknife vs CI disagree"], "note": c["note"]})
    fb, nh = fb_pooled(cells, role, WF)
    r["FB@1.05"] = float(fb[0])
    r["5-seed FB median"] = float(np.nanmedian(fb_pooled(cells, role, W5)[0]))
    r["harmful units"] = nh
    return r


# ------------------------------------------------------------------ jobs and completion
J = pd.read_csv(SB / "jobs.tsv", sep="\t", keep_default_na=False)
J = J[J.part != "full"].reset_index(drop=True)


def complete(out):
    try:
        return json.loads((Path(out) / "run_report.json").read_text()).get("status") == "complete"
    except Exception:
        return False


J["ok"] = [complete(o) for o in J.out_dir]
J["split"] = ["balanced_" + n.split("_")[-2] for n in J.name]
J["size"] = [int(n.split("_")[-1]) for n in J.name]

p("## Seed batch, parts (b) and (c): restricted tiers at N = 30")
p()
p("Ensemble = mean of per-seed latents, then max(0, ·). Pooled skill = 1 − mean(method cell log MAE)/mean(baseline cell "
  "log MAE). CIs: pairs-only (the shared 1000 × seed-923 pair draws, the same draw in every cell) and two-level (each "
  "pair draw paired with a seed resample). Verdict = the part (a) label rule on Δ = mean cell log MAE(method) − "
  "baseline's (jackknife seed SE > 0.002 and |Δ| < 2 SE → sign-unstable; 1.9–2.1 SE → the two-level CI decides).")
p()
p("### Completion vs the artifact (succeeded tasks per size, sizes 4–8)")
comp = []
for pipe, (fam, t, run) in PIPES.items():
    jj = J[J.name.str.startswith(f"R_blind_{fam}_{t}_") & (J["size"] >= 4)]
    art = pd.DataFrame(ART["runs"][run]["curve"])
    art = art[(art.role == "constant_1") & (art.tier == "one_known")].set_index("size").n_splits
    for s in SIZES:
        q = jj[jj["size"] == s]
        comp.append({"pipeline": pipe, "size": s, "built": len(q), "succeeded": int(q.ok.sum()),
                     "artifact splits": int(art.get(s, 0))})
CP = pd.DataFrame(comp)
tab = CP.pivot_table(index="size", columns="pipeline", values=["succeeded", "artifact splits"])
p(md(tab.astype(int), "{:.0f}"))
p(f"- Totals succeeded (sizes 4–8): {CP.groupby('pipeline').succeeded.sum().to_dict()}; "
  f"artifact: {CP.groupby('pipeline')['artifact splits'].sum().to_dict()}")
c4ok = {pipe: sum(bool(J[(J.name == f"R_blind_{fam}_{t}_{s[-3:]}_{n}")].ok.any()) for s, n in C4)
        for pipe, (fam, t, _) in PIPES.items()}
p(f"- C4 cells succeeded (of 20): {c4ok}")
p()

# ------------------------------------------------------------------ (b) load blind cells
BL = {}  # (pipe, tier, split, size) -> Cell (native)
for pipe, (fam, t, run) in PIPES.items():
    for r in J[J.name.str.startswith(f"R_blind_{fam}_{t}_") & J.ok].itertuples():
        for tier in TIERS:
            BL[(pipe, tier, r.split, r.size)] = load_blind(Path(r.out_dir), tier, fam, t)
p(f"Loaded {len(BL)} blind (task, tier) cells.")
p()


def roles_of(pipe):
    return ROLES_1W if pipe.startswith("1W") else ROLES_BD


# ------------------------------------------------------------------ C4 matched cells
p("### C4: the 20 matched cells (artifact's common cells), blind, N = 30")
C4ROWS = []
for pipe, (fam, t, run) in PIPES.items():
    am = {(m["tier"], m["role"]): m for m in ART["runs"][run]["matched"]}
    for tier in TIERS:
        cl = [BL[(pipe, tier, s, n)] for s, n in C4 if (pipe, tier, s, n) in BL]
        for role in roles_of(pipe):
            r = {"pipeline": pipe, "tier": tier, "method": LABEL[role], "cells": len(cl), **skill_row(cl, role)}
            a = am.get((tier, role))
            r["artifact skill % (5 seeds)"] = 100 * a["skill"] if a else np.nan
            C4ROWS.append(r)
C4T = pd.DataFrame(C4ROWS)
p("Seeds 0–4 must reproduce the artifact's 5-seed C4 skill (column 'artifact skill %').")
p(md(C4T.set_index(["pipeline", "tier", "method"]), "{:.2f}"))
p()

# ------------------------------------------------------------------ learning curves
p("### Learning curves: pooled skill per size over all succeeded splits, and the slope of skill vs size")
CUR, SLO = [], []
for pipe in PIPES:
    for tier in TIERS:
        by = {s: [c for (pp, tt, sp, sz), c in BL.items() if pp == pipe and tt == tier and sz == s] for s in SIZES}
        sizes = [s for s in SIZES if by[s]]
        for role in roles_of(pipe):
            def sk(W, pd_):
                return np.vstack([100 * (1 - pooled(by[s], role, W, pd_) / pooled(by[s], "constant_1", W, pd_))
                                  for s in sizes])  # sizes x D
            full, s5 = sk(WF, False)[:, 0], sk(W5, False)
            for i, s in enumerate(sizes):
                CUR.append({"pipeline": pipe, "tier": tier, "method": LABEL[role], "size": s, "splits": len(by[s]),
                            "skill % (N=30)": full[i], "skill % seeds 0–4": sk(W04, False)[i, 0],
                            "5-seed median skill %": float(np.median(s5[i])),
                            "5-seed 95%": "[{:+.1f}, {:+.1f}]".format(*ci(s5[i]))})
            x = np.array(sizes, float) - np.mean(sizes)
            slope = lambda M: (x[:, None] * (M - M.mean(0))).sum(0) / (x ** 2).sum()  # pp per added app
            est = float(slope(sk(WF, False))[0])
            dt, dp = slope(sk(WB, True)), slope(sk(WF, True))
            se = jack_se(slope(sk(WJ, False)))
            sl5 = slope(s5)
            lo, hi = ci(dt)
            SLO.append({"pipeline": pipe, "tier": tier, "method": LABEL[role], "slope (skill pp / app)": est,
                        "CI pairs": "[{:+.2f}, {:+.2f}]".format(*ci(dp)), "CI pairs×seeds": "[{:+.2f}, {:+.2f}]".format(lo, hi),
                        "seed SE (jackknife)": se, "5-seed median slope": float(np.median(sl5)),
                        "% 5-seed draws slope > 0": float((sl5 > 0).mean() * 100),
                        "'no gain with size' (slope CI not > 0)": "holds" if lo <= 0 else "FAILS (gain)"})
SL = pd.DataFrame(SLO)
p("Slope = OLS of per-size pooled skill (pp) on size (4–8); CI from the same pair/seed draws applied at every size. "
  "'No gain with size' holds when the two-level CI's lower end is ≤ 0.")
p(md(SL.set_index(["pipeline", "tier", "method"]), "{:.2f}"))
p()
p("#### Per-size pooled skill (%)")
CU = pd.DataFrame(CUR)
p(md(CU.pivot_table(index=["pipeline", "tier", "method"], columns="size", values="skill % (N=30)"), "{:+.1f}"))
p()

# ------------------------------------------------------------------ restricted false-benign
p("### Restricted false-benign rate (FB@1.05), pooled over all succeeded cells, sizes 4–8")
FBR = []
for pipe in PIPES:
    for tier in TIERS:
        cl = [c for (pp, tt, sp, sz), c in BL.items() if pp == pipe and tt == tier and sz >= 4]
        for role in ["constant_1"] + roles_of(pipe):
            fb, nh = fb_pooled(cl, role, WF)
            f5 = fb_pooled(cl, role, W5)[0]
            FBR.append({"pipeline": pipe, "tier": tier, "method": LABEL.get(role, role), "cells": len(cl),
                        "harmful units": nh, "FB (N=30)": float(fb[0]), "FB seeds 0–4": float(fb_pooled(cl, role, W04)[0][0]),
                        "5-seed FB median": float(np.median(f5)), "5-seed FB 95%": "[{:.2f}, {:.2f}]".format(*ci(f5))})
FBT = pd.DataFrame(FBR)
p(md(FBT.set_index(["pipeline", "tier", "method"]), "{:.3f}"))
p()

# ------------------------------------------------------------------ (c) socket E1 - D1 / D1 - blind, 20 cells, BD
p("### (c) Socket subsection, BD, the 20 C4 cells: E1 − D1 (information) and D1 − blind (architecture/init)")
SOCK = []
SROLES = ["absolute", "delta_uniform", "delta_median", "generic", "joint_cal", "bprof_cal", "bctx_cal"]
for pipe in ("BD-MPI", "BD-TT"):
    t = PIPES[pipe][1]
    for lvl in ("resolved", "rep-avg"):
        for tier in TIERS:
            arms = {"blind": [], "E1": [], "D1": []}
            for s, n in C4:
                d = {a: J[J.name == f"R_{a}_bd_{t}_{s[-3:]}_{n}"] for a in ("blind", "E1", "D1")}
                if not all(len(x) and x.ok.iloc[0] for x in d.values()):
                    continue
                arms["blind"].append(load_blind(Path(d["blind"].out_dir.iloc[0]), tier, "bd", t, lvl))
                arms["E1"].append(load_socket(Path(d["E1"].out_dir.iloc[0]), tier, t, lvl))
                arms["D1"].append(load_socket(Path(d["D1"].out_dir.iloc[0]), tier, t, lvl))
            for comp_, a, b in (("information E1 − D1", "E1", "D1"), ("architecture/init D1 − blind", "D1", "blind")):
                for role in SROLES:
                    r = contrast(arms[a], arms[b], role, role)
                    SOCK.append({"pipeline": pipe, "comparison": comp_, "level": lvl, "tier": tier,
                                 "method": LABEL[role], "cells": len(arms[a]),
                                 f"{a} log MAE": float(pooled(arms[a], role, WF, False)[0]),
                                 f"{b} log MAE": float(pooled(arms[b], role, WF, False)[0]), **r})
SK = pd.DataFrame(SOCK)
for comp_, g in SK.groupby("comparison", sort=False):
    p(f"#### {comp_}")
    p(md(g.drop(columns="comparison").dropna(axis=1, how="all").set_index(["pipeline", "level", "tier", "method"]),
         "{:.4f}"))
    p()

CP.to_csv(OUT_DIR / "seed_restricted_completion.csv", index=False)
C4T.to_csv(OUT_DIR / "seed_restricted_c4.csv", index=False)
CU.to_csv(OUT_DIR / "seed_restricted_curves.csv", index=False)
SL.to_csv(OUT_DIR / "seed_restricted_slopes.csv", index=False)
FBT.to_csv(OUT_DIR / "seed_restricted_fb.csv", index=False)
SK.to_csv(OUT_DIR / "seed_restricted_socket.csv", index=False)
(OUT_DIR / "seed_restricted_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "seed_restricted_numbers.md")
