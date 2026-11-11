"""Socket batch 2, control E (epoch-matched final fits): scoring. Run after socket_batch2_epochmatch.py and its runs.

Reuses the helpers of socket_aware_score.py (everything above its main section is exec'd into this namespace, so both
use the same truth table, bootstrap draws and metric code). Writes
  output/socket_epochmatch_scores.csv   (B1 all / s_V=0 / s_V=1 and B2 rep-averaged, per run)
  output/socket_epochmatch_paired.csv   (paired differences with pair-cluster CIs)
  output/socket_epochmatch_numbers.md
Comparisons
  E0  blind fixed_global replay vs blind CV run: max |Δ pred| (must be 0 up to float noise).
  E1  aware at the blind recipe (all 4 pipelines) − blind CV run      → the socket bit alone.
  E1' aware at the blind recipe − aware CV run                         → what the CV recipe change did to aware.
  E2  blind at the aware epochs (BD) − blind CV run                    → the epoch effect alone.
  E2' aware CV run − blind at the aware epochs (BD)                    → socket bit (+ lambda/aggregation) at aware epochs.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "socket_aware_score.py").read_text()
ns: dict = {"__name__": "socket_aware_score_helpers", "__file__": str(HERE / "socket_aware_score.py")}
exec(compile(src[: src.index("art = json.loads")], "socket_aware_score.py", "exec"), ns)
np, pd, EXP, OUT_DIR = ns["np"], ns["pd"], ns["EXP"], ns["OUT_DIR"]
LABEL, ORDER, md = ns["LABEL"], ns["ORDER"], ns["md"]

ROLES = ["absolute", "delta_uniform", "delta_median", "delta_ood_kernel", "generic", "joint_cal", "bprof_cal",
         "bctx_cal", "bwall_cal", "bfwd_cal"]
PIPES = {"1W-MPI": ("mpi", "1w_mpi"), "1W-TT": ("tt", "1w_tt"), "BD-MPI": ("mpi", "bd_mpi"), "BD-TT": ("tt", "bd_tt")}
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def load(pipe: str, kind: str, root: Path):
    """kind 'blind' (aggregate predictions) or 'aware' (socket predictions); returns (resolved B1 frame, B2 frame)."""
    target, _ = PIPES[pipe]
    ns["RUNS"] = {pipe: (target, None, {kind: root})}
    res = ns["resolved_predictions"](pipe)[kind]
    rep = ns["b2_predictions"](pipe, {kind: res})[kind]
    res = res[~res.role.str.startswith("offset")]
    rep = rep[~rep.role.str.startswith("offset")]
    return res, rep


def table(pipe, run, d, rep):
    rows = []
    for lab, sub in (("all", None), ("s_V=0", lambda x: x.s_V == 0), ("s_V=1", lambda x: x.s_V == 1)):
        g0 = d if sub is None else d[sub(d)]
        base = g0[g0.role == "constant_1"]
        for role in [r for r in ROLES if r in set(g0.role)]:
            g = g0[g0.role == role]
            m = ns["metrics"](g)
            m["Δ%"], m["CI lo"], m["CI hi"] = ns["boot_delta"](g, base)
            m["FB@1.05"], m["harmful"] = ns["false_benign"](g)
            m["floor everywhere"] = bool((g.pred_log == 0).all())
            m["frac pred floor"] = float((g.pred_log == 0).mean())
            rows.append({"pipeline": pipe, "run": run, "level": "B1", "subset": lab, "role": role, **m})
    base = rep[rep.role == "constant_1"]
    for role in [r for r in ROLES if r in set(rep.role)]:
        g = rep[rep.role == role]
        m = ns["metrics"](g)
        m["Δ%"], m["CI lo"], m["CI hi"] = ns["boot_delta"](g, base)
        m["FB@1.05"], m["harmful"] = ns["fb_repavg"](g)
        m["floor everywhere"] = bool((g.pred_log == 0).all())
        m["frac pred floor"] = float((g.pred_log == 0).mean())
        rows.append({"pipeline": pipe, "run": run, "level": "B2", "subset": "rep-avg", "role": role, **m})
    return rows


def paired(pipe, name, a, b, a_rep, b_rep):
    rows = []
    for lab, sub in (("all", None), ("s_V=0", lambda x: x.s_V == 0), ("s_V=1", lambda x: x.s_V == 1),
                     ("rep-avg", "B2")):
        for role in ROLES:
            if sub == "B2":
                x, y = a_rep[a_rep.role == role], b_rep[b_rep.role == role]
            else:
                x, y = (a, b) if sub is None else (a[sub(a)], b[sub(b)])
                x, y = x[x.role == role], y[y.role == role]
            if x.empty or y.empty:
                continue
            diff, lo, hi = ns["boot_diff"](x, y)
            rows.append({"pipeline": pipe, "comparison": name, "subset": lab, "role": role, "Δ log MAE": diff,
                         "CI lo": lo, "CI hi": hi, "excl. 0": bool(lo > 0 or hi < 0)})
    return rows


# ------------------------------------------------------------------ E0: fixed_global replay check
p("## Control E: epoch-matched final fits")
p()
p("### E0. fixed_global replay of the blind BD runs (must equal the CV runs)")
e0 = {}
for t in ("mpi", "tt"):
    a = pd.read_csv(EXP / f"E0_bd_{t}/directional_predictions.csv")
    b = pd.read_csv(EXP / f"blind_bd_{t}/directional_predictions.csv")
    k = ["evaluation_method", "anchor_budget", "method", "pair_row_id", "direction"]
    m = a.merge(b, on=k, suffixes=("_e0", "_cv"), validate="one_to_one")
    e0[t] = (len(a), len(b), len(m), float(np.abs(m.predicted_log_slowdown_e0 - m.predicted_log_slowdown_cv).max()),
             float(np.abs(m.latent_predicted_log_slowdown_e0 - m.latent_predicted_log_slowdown_cv).max()))
    p(f"- BD-{t.upper()}: rows E0 {e0[t][0]}, CV {e0[t][1]}, matched {e0[t][2]}; max |Δ pred log| {e0[t][3]:.2e}, "
      f"max |Δ latent| {e0[t][4]:.2e}")
p()

# ------------------------------------------------------------------ recipes
p("### Recipes (final epochs; calibration lambda where it differs)")
for pipe, (t, key) in PIPES.items():
    if pipe.startswith("1W"):
        g = {r: json.loads((EXP / f"{r}_{key}/consolidated/global_selection.json").read_text())["final_epochs"]
             for r in ("blind", "aware")}
        p(f"- {pipe}: blind {g['blind']} | aware {g['aware']}  (E1 aware uses the blind epochs, configs, aggregation)")
    else:
        s = {r: json.loads((EXP / f"{r}_{key}/selection.json").read_text())["variants"] for r in ("blind", "aware")}
        ep = ", ".join(f"{n} {s['blind'][n]['final_epochs']}→{s['aware'][n]['final_epochs']}" for n in s["blind"])
        lam = ", ".join(f"{n} {s['blind'][n]['calibration_lambda']:g}→{s['aware'][n]['calibration_lambda']:g}"
                        for n in s["blind"] if s["blind"][n].get("calibration_lambda") !=
                        s["aware"][n].get("calibration_lambda"))
        p(f"- {pipe} epochs blind→aware: {ep}")
        p(f"  lambda blind→aware: {lam or 'none'}")
p()

# ------------------------------------------------------------------ score
ROWS, PAIRS, D1GAP = [], [], {}
for pipe, (t, key) in PIPES.items():
    runs = {"blind (CV)": ("blind", EXP / f"blind_{key}"), "aware (CV)": ("aware", EXP / f"aware_{key}"),
            "E1 aware @ blind recipe": ("aware", EXP / f"E1_{key}")}
    if pipe.startswith("BD"):
        runs["E2 blind @ aware epochs"] = ("blind", EXP / f"E2_{key}")
    if (EXP / f"D1_{key}/run_report.json").exists():
        runs["D1 const-0 bit @ blind recipe"] = ("aware", EXP / f"D1_{key}")
    if (EXP / f"D2_{key}/run_report.json").exists():
        runs["D2 aware @ aware epochs, blind λ"] = ("aware", EXP / f"D2_{key}")
    L = {name: load(pipe, kind, root) for name, (kind, root) in runs.items()}
    if "D1 const-0 bit @ blind recipe" in L:  # residual socket pathways besides the input bit
        d = L["D1 const-0 bit @ blind recipe"][0]
        w = d.pivot_table(index=["role", "victim", "aggressor"], columns="s_V", values="pred_log")
        gap = (w[0] - w[1]).abs().groupby(level="role").max()
        D1GAP[pipe] = gap
    for name, (d, rep) in L.items():
        ROWS += table(pipe, name, d, rep)
    cmp = [("E1 − blind (socket bit only)", "E1 aware @ blind recipe", "blind (CV)"),
           ("E1 − aware CV (recipe effect on aware)", "E1 aware @ blind recipe", "aware (CV)"),
           ("aware CV − blind CV (protocol, B1)", "aware (CV)", "blind (CV)")]
    if pipe.startswith("BD"):
        cmp += [("E2 − blind (epoch effect only)", "E2 blind @ aware epochs", "blind (CV)"),
                ("aware CV − E2 (socket bit + λ/agg at aware epochs)", "aware (CV)", "E2 blind @ aware epochs")]
    if "D1 const-0 bit @ blind recipe" in L:
        cmp += [("D1 − blind (init/architecture)", "D1 const-0 bit @ blind recipe", "blind (CV)"),
                ("E1 − D1 (socket information)", "E1 aware @ blind recipe", "D1 const-0 bit @ blind recipe")]
    if "D2 aware @ aware epochs, blind λ" in L:
        cmp += [("aware CV − D2 (λ effect at aware epochs)", "aware (CV)", "D2 aware @ aware epochs, blind λ"),
                ("D2 − E1 (epoch effect at λ blind)", "D2 aware @ aware epochs, blind λ", "E1 aware @ blind recipe")]
    for name, a, b in cmp:
        PAIRS += paired(pipe, name, L[a][0], L[b][0], L[a][1], L[b][1])

S = pd.DataFrame(ROWS)
P = pd.DataFrame(PAIRS)
S.to_csv(OUT_DIR / "socket_epochmatch_scores.csv", index=False)
P.to_csv(OUT_DIR / "socket_epochmatch_paired.csv", index=False)

if D1GAP:
    p("### D1 check: max |pred(s_V=0) − pred(s_V=1)| per role (0 = no socket pathway besides the input bit)")
    for pipe, gap in D1GAP.items():
        p(f"- {pipe}: " + ", ".join(f"{LABEL.get(r, r)} {v:.2e}" for r, v in gap.items()))
    p()
p("### Paired differences (log MAE, a − b; 95% pair-cluster CI, 1000 draws seed 923)")
for pipe in PIPES:
    p(f"#### {pipe}")
    for sub in ("all", "s_V=0", "s_V=1", "rep-avg"):
        q = P[(P.pipeline == pipe) & (P.subset == sub)]
        w = q.pivot_table(index="role", columns="comparison", values="Δ log MAE", sort=False)
        ci = q.assign(s=[f"{d:+.4f} [{l:+.4f}, {h:+.4f}]{'*' if e else ''}" for d, l, h, e in
                         zip(q["Δ log MAE"], q["CI lo"], q["CI hi"], q["excl. 0"])])
        w = ci.pivot(index="role", columns="comparison", values="s").reindex([r for r in ROLES if r in set(q.role)])
        w.index = w.index.map(LABEL)
        w.index.name = f"{sub}"
        p(md(w))
        p()
p("(* = CI excludes 0)")
p()
p("### Scores per run (B1 all / s_V split, B2 rep-averaged)")
cols = ["n", "log MAE", "Δ%", "CI lo", "CI hi", "bias", "floor log MAE", "above log MAE", "Spearman", "FB@1.05",
        "harmful", "frac pred floor", "floor everywhere"]
for pipe in PIPES:
    for sub in ("all", "s_V=0", "s_V=1", "rep-avg"):
        q = S[(S.pipeline == pipe) & (S.subset == sub)].copy()
        q["method"] = q.role.map(LABEL) + " [" + q.run + "]"
        q["floor everywhere"] = q["floor everywhere"].map({True: "YES", False: "no"})
        p(f"#### {pipe}, {sub}")
        p(md(q.set_index("method")[cols]))
        p()
p("### Variants that predict exactly the floor for every cell (B1, all 166)")
fl = S[(S.subset == "all") & S["floor everywhere"]]
for (pipe, run), g in fl.groupby(["pipeline", "run"], sort=False):
    p(f"- {pipe} {run}: " + ", ".join(LABEL[r] for r in g.role))
(OUT_DIR / "socket_epochmatch_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "socket_epochmatch_numbers.md")
