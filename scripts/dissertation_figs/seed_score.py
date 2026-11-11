"""Seed batch scoring, part (a): full data, N = 100 final seeds, frozen CV recipes (seed_batch_build.py).

Ensemble = the method as designed: mean of per-seed latents, then the observed floor max(0, .). Socket-aware runs: per
socket cell, then the rep-weighted slowdown mean over the victim's sockets (n_reps from the socket cells), as the pipeline.

For every quantity:
- point estimate from the N-seed ensemble;
- pairs-only 95% CI (45 pair clusters, the shared 1000 x seed-923 draws, seeds fixed);
- two-level 95% CI (the same pair draws, each paired with a seed resample with replacement, rng 924);
- delete-one-seed jackknife SE (contrasts);
- distribution over random k-seed sub-ensembles (k = 1, 5, 10, 15; 1000 each; k = 1 = every seed): median, 2.5-97.5%,
  and the % of draws in which each claim holds.
Socket contrasts between runs pair seeds by id (E1 and D1 share every RNG draw for a given id).

Levels: native (the pipeline's truth, 90 non-self directional outcomes; blind runs; the §8 numbers),
        rep-avg (socket-cell truth averaged over replicates, 90) and resolved (166 socket cells).
Writes output/seed_numbers.md (part a) and output/seed_{methods,claims,contrasts,pervictim,socket}.csv.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "socket_aware_score.py").read_text()
ns: dict = {"__name__": "socket_aware_score_helpers", "__file__": str(HERE / "socket_aware_score.py")}
exec(compile(src[: src.index("art = json.loads")], "socket_aware_score.py", "exec"), ns)
np, pd, OUT_DIR, LABEL, md, cells = ns["np"], ns["pd"], ns["OUT_DIR"], ns["LABEL"], ns["md"], ns["cells"]
PAIRS, DRAWS, DBAR = ns["PAIRS"], ns["DRAWS"], ns["DBAR"]
from scipy import stats  # noqa: E402

import os
OUT_TAG = os.environ.get("SEED_TAG", "")
SB = Path(os.environ.get("SEED_SB", Path.home() / "hpcResearch/experiment_outputs/seed_batch/full"))
LOG105 = np.log(1.05)
PIPES = {"1W-MPI": ("1w", "mpi"), "1W-TT": ("1w", "tt"), "BD-MPI": ("bd", "mpi"), "BD-TT": ("bd", "tt")}
ROLES_1W = ["constant_1", "absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic"]
ROLES_BD = ROLES_1W + ["joint_direct", "joint_cal", "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal", "bwall_cal",
                       "bfwd_cal"]
NEAR_INIT = ["absolute", "joint_direct", "joint_cal", "bprof_direct", "bprof_cal", "bctx_direct", "bctx_cal", "bwall_cal",
             "bfwd_cal"]
KS = [1, 5, 10, 15]
NSUB = 1000
SE_GOAL = 0.002
PIDX = {p: i for i, p in enumerate(PAIRS)}
CPAIR = np.stack([np.bincount(d, minlength=len(PAIRS)) for d in DRAWS]).astype(float)  # 1000 x 45 pair counts
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def pair_of(v, a):
    return " | ".join(sorted((v, a)))


# ------------------------------------------------------------------ loading
class View:
    """Per-seed latents of one run. rows = (role, victim, aggressor[, s_V]); L = rows x seeds."""

    def __init__(self, name: str, fam: str, target: str, aware: bool, d: Path):
        self.name, self.fam, self.target, self.aware = name, fam, target, aware
        idx = 0 if fam == "1w" else 1
        code2role = {v[idx]: k for k, v in ns["ROLE_CODE"].items() if v[idx]}
        self.roles = [r for r in (ROLES_1W if fam == "1w" else ROLES_BD)]
        if not aware:
            sp = pd.read_csv(d / "seed_predictions.csv")
            sp = sp[(sp.evaluation_method == "random_split") & (sp.anchor_budget.astype(str) == "all")]
            sp = sp.rename(columns={"victim_id": "victim", "aggressor_id": "aggressor",
                                    "latent_predicted_log_slowdown": "latent"})
            keys = ["role", "victim", "aggressor"]
        else:
            sp = pd.read_csv(d / "seed_predictions_socket.csv")
            if "evaluation_method" in sp:
                sp = sp[sp.evaluation_method == "random_split"]
            if "anchor_budget" in sp:
                sp = sp[sp.anchor_budget.astype(str) == "all"]
            sp = sp.rename(columns={"victim_id": "victim", "aggressor_id": "aggressor",
                                    "latent_predicted_log_slowdown": "latent"})
            keys = ["role", "victim", "aggressor", "s_V"]
        sp = sp[sp.method.isin(code2role) & (sp.victim != sp.aggressor)].assign(role=lambda x: x.method.map(code2role))
        self.roles = [r for r in self.roles if r in set(sp.role)]
        sp = sp[sp.role.isin(self.roles)]
        Lw = sp.pivot_table(index=keys, columns="seed", values="latent", aggfunc="first")
        assert not Lw.isna().any().any(), name
        self.seeds = list(Lw.columns)
        self.L = Lw.to_numpy(float)
        self.rows = Lw.index.to_frame(index=False)
        self.rowpos = {k: i for i, k in enumerate(Lw.index)}
        if not aware:
            tr = sp.groupby(keys).true_slowdown.first().reindex(Lw.index)
            self.native_y = np.log(tr.to_numpy(float))
        self._levels = {}

    # unit maps per level: (rows index into L or aggregation spec, y, role, pair, victim, assign key)
    def level(self, lvl: str):
        if lvl in self._levels:
            return self._levels[lvl]
        t = self.target
        R = self.rows
        if lvl == "native":
            assert not self.aware
            u = R.assign(row=np.arange(len(R)), y=self.native_y)
            spec = {"kind": "rows", "row": u.row.to_numpy()}
        elif lvl == "resolved":
            c = cells[["victim", "aggressor", "s_V", f"y_{t}"]].rename(columns={f"y_{t}": "y"})
            us = []
            for role in self.roles:
                cc = c.assign(role=role)
                if self.aware:
                    cc["row"] = [self.rowpos[(role, v, a, s)] for v, a, s in zip(cc.victim, cc.aggressor, cc.s_V)]
                else:
                    cc["row"] = [self.rowpos[(role, v, a)] for v, a in zip(cc.victim, cc.aggressor)]
                us.append(cc)
            u = pd.concat(us, ignore_index=True)
            spec = {"kind": "rows", "row": u.row.to_numpy()}
        elif lvl == "rep-avg":
            rep = cells.assign(T=lambda x: x[f"ratio_{t}"] * x.n_reps).groupby(["victim", "aggressor"]).agg(
                T=("T", "sum"), n=("n_reps", "sum")).reset_index()
            rep["y"] = np.log(np.maximum(1.0, rep["T"] / rep["n"]))
            us = []
            if not self.aware:
                for role in self.roles:
                    rr = rep.assign(role=role)
                    rr["row"] = [self.rowpos[(role, v, a)] for v, a in zip(rr.victim, rr.aggressor)]
                    us.append(rr)
                u = pd.concat(us, ignore_index=True)
                spec = {"kind": "rows", "row": u.row.to_numpy()}
            else:
                # rep-weighted slowdown mean over the victim's socket cells
                rows, units, wts = [], [], []
                ulist = []
                for role in self.roles:
                    for j, r in enumerate(rep.itertuples()):
                        k = len(ulist)
                        ulist.append({"role": role, "victim": r.victim, "aggressor": r.aggressor, "y": r.y})
                        cc = cells[(cells.victim == r.victim) & (cells.aggressor == r.aggressor)]
                        for s, n in zip(cc.s_V, cc.n_reps):
                            rows.append(self.rowpos[(role, r.victim, r.aggressor, s)])
                            units.append(k)
                            wts.append(n)
                u = pd.DataFrame(ulist)
                A = np.zeros((len(u), len(R)))
                A[units, rows] = wts
                A /= A.sum(1, keepdims=True)
                spec = {"kind": "agg", "A": A}
        u["pair"] = [pair_of(v, a) for v, a in zip(u.victim, u.aggressor)]
        u["pidx"] = u.pair.map(PIDX).to_numpy()
        if "s_V" in u:
            u["akey"] = [f"{pr}#{v if s == 0 else a}" for pr, v, a, s in zip(u.pair, u.victim, u.aggressor, u.s_V)]
        else:
            u["akey"] = u.pair
        self._levels[lvl] = (u.reset_index(drop=True), spec)
        return self._levels[lvl]

    def preds(self, lvl: str, Wt: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """units x D observed log predictions for seed-weight columns Wt (seeds x D)."""
        u, spec = self.level(lvl)
        Z = self.L @ Wt
        if spec["kind"] == "rows":
            Zu = Z[spec["row"]]
            if offset is not None:
                Zu = Zu + offset[:, None]
            return np.maximum(0.0, Zu)
        P = np.maximum(0.0, Z)
        return np.log(spec["A"] @ np.exp(P))


# ------------------------------------------------------------------ metric kernels
def role_mask(u, role):
    return (u.role == role).to_numpy()


def pair_sums(u, m, E):
    """45 x D per-pair sums of E over mask m, and 45 counts."""
    pid = u.pidx.to_numpy()[m]
    S = np.zeros((len(PAIRS), E.shape[1]))
    np.add.at(S, pid, E[m])
    n = np.bincount(pid, minlength=len(PAIRS)).astype(float)
    return S, n


def fb_units(u, m, P, w=None):
    """FB@1.05 per column: unit = assignment key (resolved) or pair; harmful if true max >= 1.05.
    w: optional 45 x D pair weights (bootstrap)."""
    uu = u[m]
    keys, inv = np.unique(uu.akey.to_numpy(), return_inverse=True)
    kp = np.array([PIDX[k.split("#")[0]] for k in keys])
    ty = np.full(len(keys), -np.inf)
    np.maximum.at(ty, inv, uu.y.to_numpy())
    pm = np.full((len(keys), P.shape[1]), -np.inf)
    np.maximum.at(pm, inv, P[m])
    if "s_V" in uu:  # resolved: keep only complete units (both endpoints present)
        cnt = np.bincount(inv, minlength=len(keys))
        keep = cnt == 2
        ty, pm, kp = ty[keep], pm[keep], kp[keep]
    harm = ty >= LOG105
    miss = (pm < LOG105) & harm[:, None]
    if w is None:
        return miss.sum(0) / max(harm.sum(), 1)
    W = w[kp]  # units x D
    return (W * miss).sum(0) / (W * harm[:, None]).sum(0)


def summarize(view: View, lvl: str, Wt: np.ndarray, extra=False):
    """Per role arrays over D columns: log MAE, bias, floor frac, above-floor MAE, FB, Spearman (if extra)."""
    u, _ = view.level(lvl)
    P = view.preds(lvl, Wt)
    y = u.y.to_numpy()[:, None]
    E = np.abs(P - y)
    out = {}
    for role in view.roles:
        m = role_mask(u, role)
        r = {"log MAE": E[m].mean(0), "bias": (P[m] - y[m]).mean(0), "floor frac": (P[m] == 0).mean(0),
             "FB@1.05": fb_units(u, m, P)}
        ab = m & (u.y.to_numpy() > 0)
        r["above log MAE"] = E[ab].mean(0)
        if extra:
            r["Spearman"] = np.array([stats.spearmanr(P[m][:, j], y[m][:, 0]).statistic if np.ptp(P[m][:, j]) > 0
                                      else np.nan for j in range(P.shape[1])])
        out[role] = r
    return out, u, P


def seed_weights(S: int, kind: str, rng):
    if kind == "full":
        return np.full((S, 1), 1.0 / S)
    if kind == "jack":
        W = np.full((S, S), 1.0 / (S - 1))
        np.fill_diagonal(W, 0.0)
        return W
    if kind == "boot":
        return rng.multinomial(S, np.full(S, 1.0 / S), size=len(DRAWS)).T / S
    k = int(kind)
    if k >= S:  # only in tests with few seeds
        return np.full((S, NSUB), 1.0 / S)
    if k == 1:
        return np.eye(S)
    W = np.zeros((S, NSUB))
    for j in range(NSUB):
        W[rng.choice(S, k, replace=False), j] = 1.0 / k
    return W


def jack_se(v):
    v = np.asarray(v, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))


def boot_logmae(view, lvl, role, Wt, paired_pairs=True):
    """Per-draw log MAE with pair-cluster draws (column b of Wt paired with pair draw b)."""
    u, _ = view.level(lvl)
    P = view.preds(lvl, Wt)
    E = np.abs(P - u.y.to_numpy()[:, None])
    m = role_mask(u, role)
    S, n = pair_sums(u, m, E)
    if S.shape[1] == 1:
        S = np.repeat(S, len(DRAWS), 1)
    return (CPAIR * S.T).sum(1) / (CPAIR @ n)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def logmae(view, lvl, role, Wt):
    u, _ = view.level(lvl)
    P = view.preds(lvl, Wt)
    m = role_mask(u, role)
    return np.abs(P[m] - u.y.to_numpy()[m, None]).mean(0)


def verdict(est, se, dt):
    """One label for claims and contrasts. dt = two-level bootstrap draws of the deciding quantity.
    sign-unstable: jackknife seed SE > SE_GOAL and |est| < 2 SE. Within 1.9-2.1 SE of that boundary the two-level CI
    decides. Otherwise 'CI excludes 0' / 'n.s.' by the two-level CI.
    disagree: the jackknife rule says sign-unstable but the two-level CI excludes 0."""
    lo, hi = ci(dt)
    ci0 = lo > 0 or hi < 0
    ratio = abs(est) / se if se > 0 else np.inf
    boundary = se > SE_GOAL and 1.9 <= ratio <= 2.1
    jk_unstable = se > SE_GOAL and ratio < 2
    if boundary or not jk_unstable:
        v = "CI excludes 0" if ci0 else "n.s."
    else:
        v = "sign-unstable"
    return v, ("YES" if (jk_unstable and ci0 and not boundary) else ""), ("boundary: CI decides" if boundary else "")


# ------------------------------------------------------------------ build all views
rng = np.random.default_rng(924)
VIEWS = {}
for pipe, (fam, t) in PIPES.items():
    for v in ("blind", "awareCV", "E1", "D1"):
        d = SB / f"F_{v}_{fam}_{t}"
        if (d / "run_report.json").exists():
            VIEWS[(pipe, v)] = View(f"{pipe} {v}", fam, t, v != "blind", d)
if (SB / "F_D2_bd_mpi/run_report.json").exists():
    VIEWS[("BD-MPI", "D2")] = View("BD-MPI D2", "bd", "mpi", True, SB / "F_D2_bd_mpi")
S_ALL = {k: len(v.seeds) for k, v in VIEWS.items()}
p("## Seed batch, part (a): full data, frozen CV recipes, many final seeds")
p()
p("Runs: " + ", ".join(f"{k[0]} {k[1]} ({S_ALL[k]} seeds)" for k in VIEWS))
p()
p("- Ensemble = mean of per-seed latents, then max(0, ·) (the method as designed); socket-aware runs per cell, then the "
  "rep-weighted slowdown mean.")
p("- CIs: pairs-only = 45 pair clusters, the shared 1000 × seed-923 draws, seeds fixed; two-level = the same pair draws, "
  "each with a seed resample (with replacement). Seed SE = delete-one-seed jackknife.")
p("- k-seed draws: 1000 random subsets without replacement (k = 1: every seed).")
p()

SEEDW = {}
for key, view in VIEWS.items():
    S = len(view.seeds)
    SEEDW[key] = {k: seed_weights(S, k, rng) for k in ["full", "jack", "boot"] + [str(k) for k in KS]}
# the same seed draws for runs that are compared by seed id (all runs of a pipeline have seeds 0..N-1)
for pipe in PIPES:
    base = (pipe, "blind")
    for key in VIEWS:
        if key[0] == pipe and key != base and base in SEEDW and S_ALL[key] == S_ALL[base]:
            SEEDW[key] = SEEDW[base]

# ------------------------------------------------------------------ §8 method tables (blind, native)
METH, CLAIMS, CONTR, PERV, SOCK = [], [], [], [], []
SUBS = {}
for pipe in PIPES:
    key = (pipe, "blind")
    if key not in VIEWS:
        p(f"- {pipe}: blind run missing")
        continue
    view, W = VIEWS[key], SEEDW[key]
    full, u, Pfull = summarize(view, "native", W["full"], extra=True)
    jack, _, _ = summarize(view, "native", W["jack"])
    subs = {k: summarize(view, "native", W[str(k)], extra=True)[0] for k in KS}
    SUBS[pipe] = (full, subs, jack)
    base_full = full["constant_1"]["log MAE"][0]
    bb_pairs = boot_logmae(view, "native", "constant_1", W["full"])
    bb_two = boot_logmae(view, "native", "constant_1", W["boot"])
    rows = []
    for role in view.roles:
        if role == "constant_1":
            continue
        f = full[role]
        bp = boot_logmae(view, "native", role, W["full"])
        bt = boot_logmae(view, "native", role, W["boot"])
        dp = 100 * (bp - bb_pairs) / base_full
        dt = 100 * (bt - bb_two) / base_full
        d5 = 100 * (subs[5][role]["log MAE"] / subs[5]["constant_1"]["log MAE"] - 1)
        d1 = 100 * (subs[1][role]["log MAE"] / subs[1]["constant_1"]["log MAE"] - 1)
        rec = {"pipeline": pipe, "role": role, "log MAE": f["log MAE"][0],
               "Δ%": 100 * (f["log MAE"][0] / base_full - 1),
               "Δ% CI pairs": "[{:+.1f}, {:+.1f}]".format(*ci(dp)), "Δ% CI pairs×seeds": "[{:+.1f}, {:+.1f}]".format(*ci(dt)),
               "seed SE": jack_se(jack[role]["log MAE"]),
               "5-seed Δ% median": np.median(d5), "5-seed Δ% 95%": "[{:+.1f}, {:+.1f}]".format(*ci(d5)),
               "bias": f["bias"][0], "floor frac": f["floor frac"][0], "above log MAE": f["above log MAE"][0],
               "Spearman": f["Spearman"][0], "FB@1.05": f["FB@1.05"][0],
               "5-seed FB median": np.nanmedian(subs[5][role]["FB@1.05"]),
               "1-seed at floor everywhere": float((subs[1][role]["floor frac"] == 1).mean()),
               "1-seed blown up (Δ% > +25)": float((d1 > 25).mean())}
        rows.append(rec)
    METH += rows
    q = pd.DataFrame(rows).assign(method=lambda x: x.role.map(LABEL)).set_index("method")
    p(f"### {pipe}, blind, native outcomes ({S_ALL[key]} seeds; baseline log MAE {base_full:.4f})")
    p(md(q[["log MAE", "Δ%", "Δ% CI pairs", "Δ% CI pairs×seeds", "seed SE", "5-seed Δ% median", "5-seed Δ% 95%",
            "FB@1.05", "5-seed FB median", "Spearman", "floor frac", "above log MAE",
            "1-seed at floor everywhere", "1-seed blown up (Δ% > +25)"]], "{:.4f}"))
    p()

    # per-victim ratio (C5): absolute (and the other roles) / baseline
    vict = sorted(u.victim.unique())
    for role in [r for r in ("absolute", "joint_cal", "delta_median") if r in view.roles]:
        def ratios(Wt):
            P = view.preds("native", Wt)
            E = np.abs(P - u.y.to_numpy()[:, None])
            out = {}
            for v in vict:
                m = role_mask(u, role) & (u.victim == v).to_numpy()
                b = role_mask(u, "constant_1") & (u.victim == v).to_numpy()
                out[v] = E[m].mean(0) / E[b].mean(0)
            return out
        rf, r5 = ratios(W["full"]), ratios(W["5"])
        for v in vict:
            PERV.append({"pipeline": pipe, "role": role, "victim": v, "ratio": float(rf[v][0]),
                         "5-seed median": float(np.median(r5[v])), "% of 5-seed draws > 1": float((r5[v] > 1).mean() * 100)})

# ------------------------------------------------------------------ claims
def claim_rows(pipe, name, fn, se_fn=None):
    """fn(summary dict) -> bool array; se_fn = (a, b): the deciding quantity is log MAE(a) - log MAE(b)."""
    full, subs, jack = SUBS[pipe]
    r = {"pipeline": pipe, "claim": name, "full N": bool(np.asarray(fn(full))[0])}
    for k in KS:
        v = np.asarray(fn(subs[k]), bool)
        r[f"% of {k}-seed draws"] = float(v.mean() * 100)
    if se_fn is not None:
        a, b = se_fn
        view, W = VIEWS[(pipe, "blind")], SEEDW[(pipe, "blind")]
        est = float(lm(full, a)[0] - lm(full, b)[0])
        se = jack_se(lm(jack, a) - lm(jack, b))
        dt = boot_logmae(view, "native", a, W["boot"]) - boot_logmae(view, "native", b, W["boot"])
        sb = logmae(view, "native", a, W["boot"]) - logmae(view, "native", b, W["boot"])
        v, dis, note = verdict(est, se, dt)
        r.update({"deciding quantity": est, "seed SE (jackknife)": se, "seed SE (seed bootstrap)": float(sb.std(ddof=1)),
                  "CI pairs×seeds": "[{:+.4f}, {:+.4f}]".format(*ci(dt)),
                  "verdict": v if v == "sign-unstable" else ("fails" if not r["full N"] else
                                                             ("holds" if v == "CI excludes 0" else "n.s.")),
                  "jackknife vs CI disagree": dis, "note": note})
    else:
        r["verdict"] = "holds" if r["full N"] else "fails"
    CLAIMS.append(r)


def lm(s, role):
    return s[role]["log MAE"]


def diff_se(a, b):
    return (a, b)


for pipe in SUBS:
    fam = PIPES[pipe][0]
    claim_rows(pipe, "absolute < baseline", lambda s: lm(s, "absolute") < lm(s, "constant_1"), diff_se("absolute", "constant_1"))
    claim_rows(pipe, "absolute at the floor on all outcomes", lambda s: s["absolute"]["floor frac"] == 1)
    for d in ("delta_uniform", "delta_median", "delta_ood_kernel", "generic"):
        claim_rows(pipe, f"{d} > baseline", lambda s, d=d: lm(s, d) > lm(s, "constant_1"), diff_se(d, "constant_1"))
    cands = [r for r in ("absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic", "joint_cal",
                         "bprof_cal", "bctx_cal") if r in VIEWS[(pipe, "blind")].roles]
    delta_idx = [i for i, r in enumerate(cands) if r.startswith("delta")]
    claim_rows(pipe, "a delta variant has the lowest above-floor log MAE",
               lambda s, c=cands, di=delta_idx: np.isin(np.argmin(np.stack([s[r]["above log MAE"] for r in c]), 0), di))
    claim_rows(pipe, "absolute Spearman > 0", lambda s: np.nan_to_num(s["absolute"]["Spearman"], nan=-9) > 0)
    if fam == "bd":
        claim_rows(pipe, "joint cal < baseline", lambda s: lm(s, "joint_cal") < lm(s, "constant_1"),
                   diff_se("joint_cal", "constant_1"))
        claim_rows(pipe, "bprof cal < baseline", lambda s: lm(s, "bprof_cal") < lm(s, "constant_1"),
                   diff_se("bprof_cal", "constant_1"))
        claim_rows(pipe, "joint cal FB < absolute FB", lambda s: s["joint_cal"]["FB@1.05"] < s["absolute"]["FB@1.05"])
        claim_rows(pipe, "joint cal < absolute", lambda s: lm(s, "joint_cal") < lm(s, "absolute"),
                   diff_se("joint_cal", "absolute"))
        claim_rows(pipe, "C7 fingerprints hurt, direct (bctx > bprof)", lambda s: lm(s, "bctx_direct") > lm(s, "bprof_direct"),
                   diff_se("bctx_direct", "bprof_direct"))
        claim_rows(pipe, "C7 fingerprints hurt, calibrated", lambda s: lm(s, "bctx_cal") > lm(s, "bprof_cal"),
                   diff_se("bctx_cal", "bprof_cal"))
        claim_rows(pipe, "C7 joint direct worse than absolute", lambda s: lm(s, "joint_direct") > lm(s, "absolute"),
                   diff_se("joint_direct", "absolute"))
        claim_rows(pipe, "C7 reverse supervision helps, direct (bprof < joint)",
                   lambda s: lm(s, "bprof_direct") < lm(s, "joint_direct"), diff_se("bprof_direct", "joint_direct"))
        claim_rows(pipe, "C7 reverse supervision helps, calibrated", lambda s: lm(s, "bprof_cal") < lm(s, "joint_cal"),
                   diff_se("bprof_cal", "joint_cal"))
        claim_rows(pipe, "calibration helps joint (cal < direct)", lambda s: lm(s, "joint_cal") < lm(s, "joint_direct"),
                   diff_se("joint_cal", "joint_direct"))
        claim_rows(pipe, "calibration lowers FB (joint cal < joint direct)",
                   lambda s: s["joint_cal"]["FB@1.05"] < s["joint_direct"]["FB@1.05"])
C = pd.DataFrame(CLAIMS)
p("### §8 claims (blind, native): full-N verdict and % of k-seed draws in which each holds")
p("Verdict (claims with a deciding contrast): 'sign-unstable' when the deciding quantity's jackknife seed SE at full N "
  "exceeds 0.002 AND |estimate| < 2 SE (when |estimate| is within 1.9-2.1 SE the two-level CI decides instead). "
  "Otherwise: 'fails' if the claim is false at full N; 'holds' if true and the two-level (pairs×seeds) CI excludes 0; "
  "'n.s.' if true but the CI includes 0. Claims without a deciding contrast: holds/fails at full N. "
  "'jackknife vs CI disagree' = YES when the jackknife rule says sign-unstable but the two-level CI excludes 0; "
  "the seed-bootstrap SE (seeds resampled, pairs fixed) is shown next to the jackknife SE.")
p(md(C.set_index(["pipeline", "claim"]), "{:.4f}"))
p()

# ------------------------------------------------------------------ C7 contrasts with two-level CIs (BD)
C7 = [("Joint direct vs legacy absolute", "joint_direct", "absolute"),
      ("Floor-aware calibration (joint cal − joint direct)", "joint_cal", "joint_direct"),
      ("Reverse supervision, direct (bprof − joint)", "bprof_direct", "joint_direct"),
      ("Reverse supervision, calibrated", "bprof_cal", "joint_cal"),
      ("Fingerprints, direct (bctx − bprof)", "bctx_direct", "bprof_direct"),
      ("Fingerprints, calibrated", "bctx_cal", "bprof_cal"),
      ("Wall-clock-only aux, calibrated (bwall − bprof)", "bwall_cal", "bprof_cal"),
      ("Forward-only tokens, calibrated (bfwd − bctx)", "bfwd_cal", "bctx_cal"),
      ("Legacy delta (uniform) vs legacy absolute", "delta_uniform", "absolute"),
      ("Joint cal vs absolute", "joint_cal", "absolute")]
for pipe in ("BD-MPI", "BD-TT"):
    key = (pipe, "blind")
    if key not in VIEWS:
        continue
    view, W = VIEWS[key], SEEDW[key]
    full, subs, jack = SUBS[pipe]
    for name, a, b in C7:
        est = float(lm(full, a)[0] - lm(full, b)[0])
        dp = boot_logmae(view, "native", a, W["full"]) - boot_logmae(view, "native", b, W["full"])
        dt = boot_logmae(view, "native", a, W["boot"]) - boot_logmae(view, "native", b, W["boot"])
        d5 = lm(subs[5], a) - lm(subs[5], b)
        se = jack_se(lm(jack, a) - lm(jack, b))
        sb = logmae(view, "native", a, W["boot"]) - logmae(view, "native", b, W["boot"])
        v, dis, note = verdict(est, se, dt)
        CONTR.append({"pipeline": pipe, "contrast": name, "Δ log MAE": est, "CI pairs": "[{:+.4f}, {:+.4f}]".format(*ci(dp)),
                      "CI pairs×seeds": "[{:+.4f}, {:+.4f}]".format(*ci(dt)), "seed SE (jackknife)": se,
                      "seed SE (seed bootstrap)": float(sb.std(ddof=1)),
                      "5-seed median": float(np.median(d5)), "5-seed 95%": "[{:+.4f}, {:+.4f}]".format(*ci(d5)),
                      "% 5-seed draws same sign": float((np.sign(d5) == np.sign(est)).mean() * 100),
                      "verdict": v, "jackknife vs CI disagree": dis, "note": note})
p("### C7 ablation contrasts (blind, native; method − comparison)")
p(md(pd.DataFrame(CONTR).set_index(["pipeline", "contrast"]), "{:+.4f}"))
p()

PV = pd.DataFrame(PERV)
p("### C5 per-victim error ratio (victim log MAE ÷ baseline's), blind, native")
for role in ("absolute", "joint_cal", "delta_median"):
    q = PV[PV.role == role]
    if q.empty:
        continue
    t = q.pivot_table(index="victim", columns="pipeline", values="ratio").rename(index=ns.get("NAME", {}))
    f5 = q.pivot_table(index="victim", columns="pipeline", values="% of 5-seed draws > 1").rename(index=ns.get("NAME", {}))
    p(f"#### {LABEL[role]}: ratio at full N (and % of 5-seed draws with ratio > 1)")
    tt = t.copy().astype(object)
    for c_ in t.columns:
        tt[c_] = [f"{a:.2f} ({b:.0f}%)" for a, b in zip(t[c_], f5[c_].reindex(t.index))]
    p(md(tt))
    p()


# ------------------------------------------------------------------ socket subsection
def run_diff(pipe, va, vb, lvl, roles, label):
    if (pipe, va) not in VIEWS or (pipe, vb) not in VIEWS:
        return
    A, B = VIEWS[(pipe, va)], VIEWS[(pipe, vb)]
    WA, WB = SEEDW[(pipe, va)], SEEDW[(pipe, vb)]
    sa, ua, _ = summarize(A, lvl, WA["full"])
    sb, ub, _ = summarize(B, lvl, WB["full"])
    ja, _, _ = summarize(A, lvl, WA["jack"]) if A.seeds == B.seeds else (None, None, None)
    jb, _, _ = summarize(B, lvl, WB["jack"]) if A.seeds == B.seeds else (None, None, None)
    a5, _, _ = summarize(A, lvl, WA["5"])
    b5, _, _ = summarize(B, lvl, WB["5"])
    for role in roles:
        if role not in A.roles or role not in B.roles:
            continue
        est = float(lm(sa, role)[0] - lm(sb, role)[0])
        dp = boot_logmae(A, lvl, role, WA["full"]) - boot_logmae(B, lvl, role, WB["full"])
        dt = boot_logmae(A, lvl, role, WA["boot"]) - boot_logmae(B, lvl, role, WB["boot"])
        se = jack_se(lm(ja, role) - lm(jb, role)) if ja is not None else np.nan
        sbo = logmae(A, lvl, role, WA["boot"]) - logmae(B, lvl, role, WB["boot"])
        v, dis, note = verdict(est, se, dt)
        d5 = lm(a5, role) - lm(b5, role)
        SOCK.append({"pipeline": pipe, "comparison": label, "level": lvl, "role": role, "Δ log MAE": est,
                     "CI pairs": "[{:+.4f}, {:+.4f}]".format(*ci(dp)), "CI pairs×seeds": "[{:+.4f}, {:+.4f}]".format(*ci(dt)),
                     "seed SE (jackknife)": se, "seed SE (seed bootstrap)": float(sbo.std(ddof=1)),
                     "verdict": v, "jackknife vs CI disagree": dis, "note": note, "5-seed median": float(np.median(d5)),
                     "% 5-seed draws < 0": float((d5 < 0).mean() * 100),
                     f"{va} log MAE": lm(sa, role)[0], f"{vb} log MAE": lm(sb, role)[0],
                     f"{va} FB": sa[role]["FB@1.05"][0], f"{vb} FB": sb[role]["FB@1.05"][0],
                     f"{va} floor frac": sa[role]["floor frac"][0], f"{vb} floor frac": sb[role]["floor frac"][0]})


SROLES = ["absolute", "delta_uniform", "delta_median", "generic", "joint_cal", "bprof_cal", "bctx_cal"]
for pipe in PIPES:
    for lvl in ("resolved", "rep-avg"):
        run_diff(pipe, "E1", "D1", lvl, SROLES, "information effect E1 − D1")
        run_diff(pipe, "D1", "blind", lvl, SROLES, "init/architecture D1 − blind")
        run_diff(pipe, "awareCV", "blind", lvl, SROLES, "CV-selected aware − blind")
for lvl in ("resolved", "rep-avg"):
    run_diff("BD-MPI", "awareCV", "D2", lvl, ["joint_cal", "bprof_cal", "bctx_cal", "absolute"], "λ effect awareCV(λ10) − D2(λ1)")
    run_diff("BD-MPI", "D2", "E1", lvl, ["joint_cal", "bprof_cal", "bctx_cal", "absolute"], "epoch effect D2 − E1")
SK = pd.DataFrame(SOCK)
p("### Socket subsection (full data): paired run contrasts, seeds paired by id")
for comp, g in SK.groupby("comparison", sort=False) if not SK.empty else []:
    p(f"#### {comp}")
    g = g.assign(method=g.role.map(LABEL)).set_index(["pipeline", "level", "method"]).dropna(axis=1, how="all")
    p(md(g.drop(columns=["comparison", "role"]), "{:.4f}"))
    p()

# offset model: blind absolute latent + pinned isolated socket offset (JMs / all), resolved level
p("#### Offset model (blind absolute + fixed socket offset), resolved cells")
rows = []
for pipe, (fam, t) in PIPES.items():
    key = (pipe, "blind")
    if key not in VIEWS:
        continue
    view, W = VIEWS[key], SEEDW[key]
    u, _ = view.level("resolved")
    m = role_mask(u, "absolute")
    for tag, only_ci in (("offset_jms", True), ("offset_all", False)):
        off = np.array([DBAR[t][v][0] if (s == 1 and r == "absolute" and (DBAR[t][v][1] or not only_ci)) else 0.0
                        for v, s, r in zip(u.victim, u.s_V, u.role)])
        for wk in ("full", "5"):
            P0 = view.preds("resolved", W[wk])
            P1 = view.preds("resolved", W[wk], offset=off)
            y = u.y.to_numpy()[:, None]
            e0, e1 = np.abs(P0 - y)[m].mean(0), np.abs(P1 - y)[m].mean(0)
            if wk == "full":
                rec = {"pipeline": pipe, "model": tag, "absolute log MAE": e0[0], "offset log MAE": e1[0],
                       "offset − absolute": e1[0] - e0[0], "offset floor frac": float((P1[m] == 0).mean())}
            else:
                rec["5-seed median diff"] = float(np.median(e1 - e0))
                rec["% 5-seed draws offset better"] = float((e1 < e0).mean() * 100)
        rows.append(rec)
p(md(pd.DataFrame(rows).set_index(["pipeline", "model"]), "{:.4f}"))
p()

# ------------------------------------------------------------------ k-curve (convergence) for headline claims
KC = [1, 5, 10, 25, 50]
KCURVE = []
rng_k = np.random.default_rng(925)
p("### k-curve: Δ% vs baseline over random k-seed sub-ensembles (blind, native)")
p("1000 random subsets without replacement per k (k = 1: every seed); k = N is the full ensemble. SD at k = 50 of N = 100 "
  "is shrunk by the finite-population factor (subsets overlap).")
for pipe, role in (("BD-MPI", "joint_cal"), ("BD-MPI", "bprof_cal"), ("BD-TT", "joint_cal"), ("BD-TT", "bprof_cal"),
                   ("BD-MPI", "absolute"), ("BD-TT", "absolute")):
    key = (pipe, "blind")
    if key not in VIEWS:
        continue
    view = VIEWS[key]
    S = len(view.seeds)
    base = logmae(view, "native", "constant_1", seed_weights(S, "full", rng_k))[0]
    for k in [k for k in KC if k < S] + [S]:
        Wk = seed_weights(S, "full" if k == S else str(k), rng_k)
        d = 100 * (logmae(view, "native", role, Wk) / base - 1)
        P = view.preds("native", Wk)
        u, _ = view.level("native")
        fl = (P[role_mask(u, role)] == 0).mean(0)
        KCURVE.append({"pipeline": pipe, "method": LABEL[role], "k": k, "mean Δ%": d.mean(),
                       "SD Δ%": d.std(ddof=1) if d.size > 1 else 0.0, "median Δ%": np.median(d),
                       "2.5%": np.percentile(d, 2.5), "97.5%": np.percentile(d, 97.5),
                       "% draws < 0": (d < 0).mean() * 100, "mean floor frac": fl.mean()})
KQ = pd.DataFrame(KCURVE)
p(md(KQ.set_index(["pipeline", "method", "k"]), "{:+.2f}"))
p()

# ------------------------------------------------------------------ floor decomposition
p("### Floor decomposition (blind, native): where the error comes from")
p("Truth at floor = true slowdown < 1 clipped to 1 (y = 0, 'benign'); above floor = y > 0. Bias = mean(pred − truth) on "
  "log slowdown. Seeds 0–4 = the artifact's 5-seed ensemble.")
DEC = []
for pipe in ("BD-MPI", "BD-TT"):
    key = (pipe, "blind")
    if key not in VIEWS:
        continue
    view = VIEWS[key]
    S = len(view.seeds)
    u, _ = view.level("native")
    y = u.y.to_numpy()
    for wname, Wt in (("N = %d" % S, np.full((S, 1), 1.0 / S)),
                      ("seeds 0–4", np.isin(np.asarray(view.seeds), [0, 1, 2, 3, 4]).astype(float)[:, None] / 5)):
        P = view.preds("native", Wt)[:, 0]
        for role in ("constant_1", "absolute", "joint_cal", "bprof_cal"):
            m = role_mask(u, role)
            at, ab = m & (y == 0), m & (y > 0)
            DEC.append({"pipeline": pipe, "ensemble": wname, "method": LABEL.get(role, role),
                        "log MAE": np.abs(P[m] - y[m]).mean(), "bias": (P[m] - y[m]).mean(),
                        "n at floor": int(at.sum()), "log MAE at floor": np.abs(P[at] - y[at]).mean(),
                        "n above": int(ab.sum()), "log MAE above": np.abs(P[ab] - y[ab]).mean(),
                        "bias above": (P[ab] - y[ab]).mean(), "pred floor frac": (P[m] == 0).mean(),
                        "pred floor frac on truth-at-floor": (P[at] == 0).mean(),
                        "share of total abs error from above-floor": np.abs(P[ab] - y[ab]).sum() / np.abs(P[m] - y[m]).sum()})
p(md(pd.DataFrame(DEC).set_index(["pipeline", "ensemble", "method"]), "{:.4f}"))
p()
KQ.to_csv(OUT_DIR / f"seed_kcurve{OUT_TAG}.csv", index=False)
pd.DataFrame(DEC).to_csv(OUT_DIR / f"seed_floor_decomp{OUT_TAG}.csv", index=False)

pd.DataFrame(METH).to_csv(OUT_DIR / f"seed_methods{OUT_TAG}.csv", index=False)
C.to_csv(OUT_DIR / f"seed_claims{OUT_TAG}.csv", index=False)
pd.DataFrame(CONTR).to_csv(OUT_DIR / f"seed_contrasts{OUT_TAG}.csv", index=False)
PV.to_csv(OUT_DIR / f"seed_pervictim{OUT_TAG}.csv", index=False)
SK.to_csv(OUT_DIR / f"seed_socket{OUT_TAG}.csv", index=False)
(OUT_DIR / f"seed_numbers{OUT_TAG}.md").write_text(log.getvalue())
print("wrote", OUT_DIR / f"seed_numbers{OUT_TAG}.md")
