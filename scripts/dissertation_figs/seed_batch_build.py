"""Seed batch (director GO 2026-11-09): build configs + a job list for the large-seed re-run of §8.

Recipes are FROZEN (fixed_global) at the existing CV selections; only final_seeds changes. Seeds 0-4 are kept, so every
run's 0-4 sub-ensemble must reproduce its 5-seed reference run exactly (checked by seed_batch_check.py).

(a) full data, N = 100 (17 runs): BD {blind, awareCV, E1, D1} x {mpi, tt} + D2_bd_mpi; 1W {blind, awareCV, E1, D1} x {mpi, tt}
    awareCV BD = aware data + the aware CV selection incl. its lambdas (fixed_global; the aware CV runs were crossed_cv runs).
(b) restricted tiers, blind, N = 30: BD = every succeeded B4 blind training-size task (47 x 2); 1W = sizes 4-8 x 10 splits
    x 2 (the random_split resolved config + evaluation_method both + training_apps, as the runner resolves it).
(c) restricted, BD, the 20 C4 cells, N = 30: E1-style aware (blind recipe, aware data, socket bit) and D1 (constant-0 bit).
Writes ~/hpcResearch/experiment_outputs/seed_batch/{configs/,jobs.tsv}; jobs.tsv = name, part, est_seconds, ref_dir, cwd, argv(json).
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
SB2 = HOME / "hpcResearch/experiment_outputs/socket_batch2"
OUT = HOME / "hpcResearch/experiment_outputs/seed_batch"
CFG = OUT / "configs"
UT = HOME / "hpcResearch/repos/coScheduling/notebooks/utils"
PY = "/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python"
N_FULL, N_RESTR = 100, 30
C4_CELLS = [(f"balanced_s{c[1:3]}", int(c.split("/")[1])) for c in
            "s04/8, s05/7, s05/8, s06/6, s06/7, s06/8, s07/4, s07/5, s07/6, s07/7, s07/8, s08/4, s08/5, s08/6, s08/7, "
            "s08/8, s09/4, s09/5, s09/6, s09/7".split(", ")]
CFG.mkdir(parents=True, exist_ok=True)
jobs = []


def est(kind: str, n: int) -> float:
    return {"bd": 60 + 13 * n, "bd_aw": 1.25 * (60 + 13 * n), "1w": 140 + 43 * n, "1w_aw": 1.2 * (140 + 43 * n),
            "bd_r": 60 + 31 * n, "bd_r_aw": 1.15 * (60 + 31 * n), "1w_r": 1.6 * (140 + 43 * n)}[kind]


def add(name, part, kind, n, cfg: dict, argv: list[str], cwd: str, ref: Path | None):
    cfg = dict(cfg)
    cfg["final_seeds"] = list(range(n))
    cp = CFG / f"{name}.json"
    cp.write_text(json.dumps(cfg, indent=2))
    out = OUT / part / name
    argv = list(argv)
    argv[argv.index("--config") + 1] = str(cp)
    argv[argv.index("--output-dir") + 1] = str(out)
    jobs.append({"name": name, "part": part, "est_seconds": round(est(kind, n)), "ref_dir": str(ref) if ref else "",
                 "out_dir": str(out), "cwd": cwd, "argv": json.dumps(argv)})


def from_commands(key: str):
    """(argv, cwd) from socket_batch2 epochmatch_configs/commands.txt."""
    line = [l.split("\t", 1)[1] for l in (SB2 / "epochmatch_configs/commands.txt").read_text().splitlines()
            if l.startswith(f"{key}\t")][0]
    cwd = ""
    if line.startswith("cd "):
        cd, line = line.split(" && ", 1)
        cwd = cd[3:].strip()
    return shlex.split(line), cwd


def runner_task(root: Path, stage: str):
    t = [t for t in json.loads((root / "plan.json").read_text()) if t["stage"] == stage]
    assert len(t) == 1, (root, stage)
    att = sorted((root / "tasks" / t[0]["task_id"] / "attempts").iterdir())[-1]
    return (json.loads((att / "resolved_config.json").read_text()), json.loads((att / "command.json").read_text())["argv"],
            att / "outputs")


# ---------------------------------------------------------------- (a) full data
for t in ("mpi", "tt"):
    for v, key in (("blind", f"E0_bd_{t}"), ("E1", f"E1_bd_{t}"), ("D1", f"D1_bd_{t}")):
        argv, cwd = from_commands(key)
        add(f"F_{v}_bd_{t}", "full", "bd" if v == "blind" else "bd_aw", N_FULL,
            json.loads((SB2 / f"epochmatch_configs/{key}.json").read_text()), argv, cwd, SB2 / key)
    # awareCV: aware data + aware CV selection incl. lambdas
    cfg = json.loads((SB2 / f"epochmatch_configs/E1_bd_{t}.json").read_text())
    aw = json.loads((SB2 / f"aware_bd_{t}/selection.json").read_text())["variants"]
    cfg["fixed_model_configs_by_variant"] = {n: v["model_config"] for n, v in aw.items()}
    cfg["fixed_final_epochs_by_variant"] = {n: int(v["final_epochs"]) for n, v in aw.items()}
    cfg["fixed_aggregations_by_variant"] = {n: v["aggregation"] for n, v in aw.items() if "aggregation" in v}
    cfg["fixed_calibration_lambda_by_variant"] = {n: v["calibration_lambda"] for n, v in aw.items()
                                                  if "calibration_lambda" in v}
    argv, cwd = from_commands(f"E1_bd_{t}")
    add(f"F_awareCV_bd_{t}", "full", "bd_aw", N_FULL, cfg, argv, cwd, SB2 / f"aware_bd_{t}")
argv, cwd = from_commands("D2_bd_mpi")
add("F_D2_bd_mpi", "full", "bd_aw", N_FULL, json.loads((SB2 / "epochmatch_configs/D2_bd_mpi.json").read_text()), argv, cwd,
    SB2 / "D2_bd_mpi")
for t in ("mpi", "tt"):
    for v, root in (("blind", SB2 / f"blind_1w_{t}"), ("awareCV", SB2 / f"aware_1w_{t}")):
        cfg, argv, ref = runner_task(root, "evaluate_random_split")
        add(f"F_{v}_1w_{t}", "full", "1w" if v == "blind" else "1w_aw", N_FULL, cfg, argv, "", ref)
    for v in ("E1", "D1"):
        argv, cwd = from_commands(f"{v}_1w_{t}")
        add(f"F_{v}_1w_{t}", "full", "1w_aw", N_FULL,
            json.loads((SB2 / f"epochmatch_configs/{v}_1w_{t}.json").read_text()), argv, cwd, SB2 / f"{v}_1w_{t}")

# ---------------------------------------------------------------- (b) restricted, blind
for t in ("mpi", "tt"):
    root = SB2 / f"b4_blind_bd_{t}"
    for task in json.loads((root / "plan.json").read_text()):
        if task["stage"] != "evaluate_training_size":
            continue
        st = json.loads((root / "tasks" / task["task_id"] / "status.json").read_text())
        if st["status"] != "succeeded":
            continue
        att = Path(st["output_dir"]).parent
        add(f"R_blind_bd_{t}_{task['split_id'][-3:]}_{task['training_app_count']}", "restricted", "bd_r", N_RESTR,
            json.loads((att / "resolved_config.json").read_text()), json.loads((att / "command.json").read_text())["argv"],
            "", Path(st["output_dir"]))
sys.path.insert(0, str(UT / "delta_response_filteredInhibs_socketAware"))
import run_experiments as rx1  # noqa: E402  (1W runner, for the same nested_orders as its own plan)

bd_orders = {}
for task in json.loads((SB2 / "b4_blind_bd_mpi/plan.json").read_text()):
    if task["stage"] == "evaluate_training_size":
        bd_orders[(task["split_id"], task["training_app_count"])] = task["config"]["training_apps"]
for t in ("mpi", "tt"):
    cfg, argv, _ = runner_task(SB2 / f"blind_1w_{t}", "evaluate_random_split")
    jobs_csv = argv[argv.index("--jobs-csv") + 1]
    orders = rx1.nested_orders(pd.read_csv(jobs_csv), count=10, seed=811)
    for i, order in enumerate(orders):
        for size in range(4, 9):
            split = f"balanced_s{i:02d}"
            apps = order[:size]
            same = sorted(apps) == sorted(bd_orders[(split, size)])
            c = dict(cfg, evaluation_method="both", training_apps=apps)
            add(f"R_blind_1w_{t}_s{i:02d}_{size}", "restricted", "1w_r", N_RESTR, c, argv, "", None)
            jobs[-1]["note"] = "" if same else "training apps differ from BD split"

# ---------------------------------------------------------------- (c) restricted BD aware(E1-style) + D1, 20 C4 cells
for t in ("mpi", "tt"):
    root = SB2 / f"b4_blind_bd_{t}"
    tasks = {(x["split_id"], x["training_app_count"]): x for x in json.loads((root / "plan.json").read_text())
             if x["stage"] == "evaluate_training_size"}
    code = UT / ("delta_response_filteredInhibs_biDirectional_socketAware" if t == "mpi"
                 else "delta_response_filteredInhibs_biDirectional_totalTime_socketAware")
    for split, size in C4_CELLS:
        st = json.loads((root / "tasks" / tasks[(split, size)]["task_id"] / "status.json").read_text())
        assert st["status"] == "succeeded", (t, split, size)
        att = Path(st["output_dir"]).parent
        cfg = json.loads((att / "resolved_config.json").read_text())
        cfg["socket_feature"] = True
        argv0 = json.loads((att / "command.json").read_text())["argv"]
        argv0 = [a.replace(f"{code}/data/", f"{code}/data_socket/") for a in argv0] + ["--socket-feature"]
        assert sum("/data_socket/" in a for a in argv0) == 4
        add(f"R_E1_bd_{t}_{split[-3:]}_{size}", "restricted_socket", "bd_r_aw", N_RESTR, cfg, argv0, "", None)
        d1 = [a.replace(f"{code}/pipeline.py", f"{SB2}/d1_code/{code.name}/pipeline.py") for a in argv0]
        assert d1 != argv0
        add(f"R_D1_bd_{t}_{split[-3:]}_{size}", "restricted_socket", "bd_r_aw", N_RESTR, cfg, d1, "", None)

J = pd.DataFrame(jobs)
J.to_csv(OUT / "jobs.tsv", sep="\t", index=False)
print(J.groupby("part").agg(n=("name", "size"), cpu_h=("est_seconds", lambda s: s.sum() / 3600)))
print("1W splits differing from BD:", int((J.get("note", pd.Series(dtype=str)).fillna("") != "").sum()))
