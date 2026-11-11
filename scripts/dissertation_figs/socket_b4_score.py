"""Socket batch 2, B4: restricted training-size tiers, bidirectional pipelines, blind vs socket-aware.

Reads the runner roots socket_batch2/b4_{blind,aware}_bd_{mpi,tt} (socket_batch2_b4_bd.sh). Helpers (truth tables,
pair-cluster draws, metrics) come from socket_aware_score.py, exec'd up to its main section.
Outputs output/socket_b4_numbers.md, socket_b4_{cells,paired,training}.csv.

- Completion: evaluate_training_size tasks succeeded, per size; the artifact's 50 = sizes 4-8 x 10 splits.
- C4 format: matched (split, size) cells, sizes 4-8, completed by blind AND aware of that target; per tier
  (one_known / zero_shot); entry = pooled skill 1 - mean(method cell log MAE)/mean(baseline cell log MAE) and wins.
  Two outcome levels: rep-averaged directional outcomes (as the artifact) and socket-resolved cells (as B1).
- Paired aware - blind: outcome rows pooled over the matched cells of a tier, pair-cluster bootstrap (45 unordered
  pairs, the shared 1000 x seed-923 draws).
- Training thinness per size (aware resolved App-Inhibitor panel of each task): rows, (app, config) cells observed on
  one socket only, floor rows; blind: rows and floor rows.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "socket_aware_score.py").read_text()
ns: dict = {"__name__": "socket_aware_score_helpers", "__file__": str(HERE / "socket_aware_score.py")}
exec(compile(src[: src.index("art = json.loads")], "socket_aware_score.py", "exec"), ns)
np, pd, EXP, OUT_DIR, LABEL, md = ns["np"], ns["pd"], ns["EXP"], ns["OUT_DIR"], ns["LABEL"], ns["md"]
cells = ns["cells"]

ROLES = ["absolute", "delta_uniform", "delta_median", "generic", "joint_cal", "bprof_cal", "bctx_cal"]
CODE2ROLE = {v[1]: k for k, v in ns["ROLE_CODE"].items() if v[1] and (k in ROLES or k == "constant_1")}
TIERS = ("one_known", "zero_shot")
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def tasks(root: Path):
    """(split, size, status, outputs dir) for every evaluate_training_size task."""
    out = []
    for t in json.loads((root / "plan.json").read_text()):
        if t["stage"] != "evaluate_training_size":
            continue
        st = root / "tasks" / t["task_id"] / "status.json"
        s = json.loads(st.read_text()) if st.exists() else {"status": "not run"}
        out.append(dict(split=t["split_id"], size=t["training_app_count"], status=s["status"],
                        out=Path(s["output_dir"]) if "output_dir" in s else None,
                        apps=t["config"]["training_apps"]))
    return pd.DataFrame(out)


def rep_truth(target):
    rep = cells.assign(T=lambda x: x[f"ratio_{target}"] * x.n_reps).groupby(["victim", "aggressor", "pair"]).agg(
        T=("T", "sum"), n=("n_reps", "sum")).reset_index()
    rep["y"] = np.log(np.maximum(1.0, rep["T"] / rep["n"]))
    return rep


def load_cell(out: Path, tier: str, aware: bool, target: str):
    """(rep-averaged frame, socket-resolved frame) for one task and tier: role, victim, aggressor, pair, y, pred_log."""
    ev = out / "evaluations" / tier
    d = pd.read_csv(ev / "directional_predictions.csv")
    d = d[(d.anchor_budget.astype(str) == "all") & d.method.isin(CODE2ROLE)]
    d = d.rename(columns={"victim_id": "victim", "aggressor_id": "aggressor"})
    d = d[d.victim != d.aggressor].assign(role=lambda x: x.method.map(CODE2ROLE))
    rep = rep_truth(target).merge(d[["role", "victim", "aggressor", "predicted_log_slowdown"]],
                                  on=["victim", "aggressor"]).rename(columns={"predicted_log_slowdown": "pred_log"})
    base = cells[["victim", "aggressor", "s_V", "n_reps", "pair", f"y_{target}"]].rename(columns={f"y_{target}": "y"})
    if aware:
        s = pd.read_csv(ev / "directional_predictions_socket.csv")
        s = s[s.method.isin(CODE2ROLE)].assign(role=lambda x: x.method.map(CODE2ROLE))
        res = base.merge(s[["role", "victim", "aggressor", "s_V", "pred_log_slowdown"]],
                         on=["victim", "aggressor", "s_V"]).rename(columns={"pred_log_slowdown": "pred_log"})
    else:
        res = base.merge(d[["role", "victim", "aggressor", "predicted_log_slowdown"]],
                         on=["victim", "aggressor"]).rename(columns={"predicted_log_slowdown": "pred_log"})
    for f in (rep, res):
        f["abs_err"] = (f.pred_log - f.y).abs()
    return rep, res


def training_thinness(out: Path, apps: list[str], aware: bool):
    ji = pd.read_csv(out / "selected_inhibitors" / "job_inh.csv")
    ji = ji[ji.job_id.isin(apps)]
    r = {"rows": len(ji), "floor rows": int((ji.slowdown <= 1.0).sum())}
    if aware and "s_V" in ji:
        n_sock = ji.groupby(["job_id", "inhib_id"]).s_V.nunique()
        r["cells"] = len(n_sock)
        r["one-sided cells"] = int((n_sock == 1).sum())
    return r


p("## B4. Restricted training-size tiers, bidirectional, blind vs socket-aware")
p()
CELLROWS, PAIRROWS, TRAIN = [], [], []
for t in ("mpi", "tt"):
    T = {v: tasks(EXP / f"b4_{v}_bd_{t}") for v in ("blind", "aware") if (EXP / f"b4_{v}_bd_{t}/plan.json").exists()}
    if len(T) < 2:
        p(f"- BD-{t.upper()}: roots missing ({sorted(T)})")
        continue
    p(f"### BD-{t.upper()}")
    comp = pd.concat([d.assign(variant=v) for v, d in T.items()])
    tab = comp.assign(ok=comp.status == "succeeded").pivot_table(index="size", columns="variant", values="ok",
                                                                    aggfunc="sum")
    p("Completed evaluate_training_size tasks per size (of 10 splits):")
    p(md(tab.astype(int).rename_axis("size"), "{:.0f}"))
    for v, d in T.items():
        n50 = int(((d["size"] >= 4) & (d.status == "succeeded")).sum())
        p(f"- {v}: sizes 4-8 completed {n50}/50; sizes 2-3 completed {int(((d["size"] < 4) & (d.status == 'succeeded')).sum())}/20; "
          f"other statuses: {dict(d[d.status != 'succeeded'].status.value_counts())}")
    ok = {v: set(zip(d[d.status == "succeeded"].split, d[d.status == "succeeded"]["size"])) for v, d in T.items()}
    matched = sorted(c for c in ok["blind"] & ok["aware"] if c[1] >= 4)
    p(f"- matched cells (sizes 4-8, both variants): {len(matched)}")
    # thinness
    for v, d in T.items():
        for r in d[d.status == "succeeded"].itertuples():
            TRAIN.append({"target": t, "variant": v, "size": r.size, "split": r.split,
                          **training_thinness(r.out, r.apps, v == "aware")})
    # per-cell metrics
    pooled = {}
    for tier in TIERS:
        for v, d in T.items():
            idx = d.set_index(["split", "size"])
            for c in matched:
                rep, res = load_cell(idx.loc[c, "out"], tier, v == "aware", t)
                for lvl, f in (("rep-avg", rep), ("resolved", res)):
                    pooled.setdefault((tier, lvl, v), []).append(f.assign(cell=f"{c[0]}/{c[1]}"))
                    g = f.groupby("role").abs_err.mean()
                    for role in ROLES:
                        if role in g:
                            CELLROWS.append({"target": t, "tier": tier, "level": lvl, "variant": v, "split": c[0],
                                             "size": c[1], "role": role, "log MAE": g[role],
                                             "baseline": g["constant_1"]})
    C = pd.DataFrame([r for r in CELLROWS if r["target"] == t])
    for lvl in ("rep-avg", "resolved"):
        p(f"#### C4 format, {lvl} outcomes ({len(matched)} matched cells): pooled skill (wins/{len(matched)})")
        rows = {}
        for (tier, v, role), g in C[C.level == lvl].groupby(["tier", "variant", "role"]):
            sk = 1 - g["log MAE"].mean() / g.baseline.mean()
            rows.setdefault(LABEL[role], {})[f"{tier} {v}"] = f"{100 * sk:+.1f}% ({int((g['log MAE'] < g.baseline).sum())})"
        p(md(pd.DataFrame(rows).T.reindex([LABEL[r] for r in ROLES]).reindex(
            columns=[f"{tr} {v}" for tr in TIERS for v in ("blind", "aware")])))
        p()
    p("#### Paired aware − blind (log MAE over matched cells, pair-cluster 95% CI)")
    for lvl in ("rep-avg", "resolved"):
        rows = []
        for tier in TIERS:
            a = pd.concat(pooled[(tier, lvl, "aware")])
            b = pd.concat(pooled[(tier, lvl, "blind")])
            for role in ROLES:
                x, y = a[a.role == role], b[b.role == role]
                if x.empty or y.empty:
                    continue
                diff, lo, hi = ns["boot_diff"](x, y)
                fl = lambda f: float((f.pred_log == 0).mean())
                rows.append({"target": t, "level": lvl, "tier": tier, "role": role, "aware − blind": diff, "CI lo": lo,
                             "CI hi": hi, "blind log MAE": y.abs_err.mean(), "aware log MAE": x.abs_err.mean(),
                             "blind floor frac": fl(y), "aware floor frac": fl(x), "n rows": len(x)})
        PAIRROWS += rows
        q = pd.DataFrame(rows)
        q["method"] = q.tier + ": " + q.role.map(LABEL)
        p(f"##### {lvl}")
        p(md(q.set_index("method")[["aware − blind", "CI lo", "CI hi", "blind log MAE", "aware log MAE",
                                    "blind floor frac", "aware floor frac", "n rows"]]))
        p()
    tr = pd.DataFrame([r for r in TRAIN if r["target"] == t])
    p("#### Training panels per size (mean over completed splits): resolved rows, one-sided (app, config) cells, floor rows")
    p(md(tr.groupby(["variant", "size"]).mean(numeric_only=True).reset_index().set_index("variant"), "{:.1f}"))
    p()

pd.DataFrame(CELLROWS).to_csv(OUT_DIR / "socket_b4_cells.csv", index=False)
pd.DataFrame(PAIRROWS).to_csv(OUT_DIR / "socket_b4_paired.csv", index=False)
pd.DataFrame(TRAIN).to_csv(OUT_DIR / "socket_b4_training.csv", index=False)
(OUT_DIR / "socket_b4_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / "socket_b4_numbers.md")
