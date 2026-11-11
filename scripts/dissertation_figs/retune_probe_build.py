"""Re-tune probe (director GO 2026-11-09), BD-MPI: is the near-init null an undertraining artifact?

Stage A (`build A`): long-budget crossed CV, blind, one process per variant. Same folds, cv_seeds, single HP config,
  batch setup and lambda as blind_bd_mpi; only max_epochs = 240 with early stopping off (patience = 240), so every fold
  logs its full validation curve. lambda fixed at the blind-selected value per variant. final_seeds = [0] (the CV run's
  own final fit is not used).
Stage B (`build B`): descriptive App-App curve (NOT used for selection): fixed_global blind (E0 recipe), every variant at
  the same epoch count e in GRID_B, 30 seeds.
Stage C (`build C`, after A): final fits, 30 seeds, at the CV-selected long budget (protocol selector = median of per-fold
  best epochs): blind (E0 recipe), E1 (aware data + socket bit) and D1 (constant-0 bit), epoch-matched. If the
  fold-averaged-curve argmin differs materially (protocol <= 7 and argmin >= 30, or a ratio >= 4) for any variant, C is
  also run at the argmin budget, labelled 'sens'.
Writes ~/hpcResearch/experiment_outputs/retune_probe/{configs/, jobs_<stage>.tsv, selected_epochs.json}.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
SB2 = HOME / "hpcResearch/experiment_outputs/socket_batch2"
PROBE = os.environ.get("PROBE", "retune")  # retune | capacity
OUT = HOME / f"hpcResearch/experiment_outputs/{PROBE}_probe"
# capacity probe: config + epochs by the same crossed CV; original config first (reference), lr/wd/lambda/batch unchanged
CAP_CONFIGS = [dict(hidden_dim=8, embedding_dim=4, rank=2), dict(hidden_dim=16, embedding_dim=8, rank=4),
               dict(hidden_dim=32, embedding_dim=8, rank=4), dict(hidden_dim=32, embedding_dim=16, rank=8)]
CAP_MAX_E, CAP_PATIENCE = 120, 40
CFG = OUT / "configs"
MAX_E = 240
GRID_B = [7, 15, 30, 60, 120, 240]
N_FINAL = 30
CFG.mkdir(parents=True, exist_ok=True)
VARIANTS = ["legacy_delta", "legacy_generic", "legacy_absolute", "joint_profile", "bidirectional_profile",
            "bidirectional_context", "bidirectional_profile_wall", "bidirectional_context_forward"]


def command(key: str):
    line = [l.split("\t", 1)[1] for l in (SB2 / "epochmatch_configs/commands.txt").read_text().splitlines()
            if l.startswith(f"{key}\t")][0]
    cd, line = line.split(" && ", 1)
    return shlex.split(line), cd[3:].strip()


def write_jobs(stage: str, jobs: list[dict]):
    pd.DataFrame(jobs).to_csv(OUT / f"jobs_{stage}.tsv", sep="\t", index=False)
    print(f"{stage}: {len(jobs)} jobs -> {OUT / f'jobs_{stage}.tsv'}")


def job(name, cfg, key):
    argv, cwd = command(key)
    cp = CFG / f"{name}.json"
    cp.write_text(json.dumps(cfg, indent=2))
    out = OUT / "runs" / name
    argv = list(argv)
    argv[argv.index("--config") + 1] = str(cp)
    argv[argv.index("--output-dir") + 1] = str(out)
    return {"name": name, "out_dir": str(out), "cwd": cwd, "argv": json.dumps(argv)}


def base(key):
    return json.loads((SB2 / f"epochmatch_configs/{key}.json").read_text())


stage = sys.argv[1]
if stage == "A":
    blind_cfg = json.loads((SB2 / "blind_bd_mpi/requested_run_config.json").read_text())
    sel = json.loads((SB2 / "blind_bd_mpi/selection.json").read_text())["variants"]
    jobs = []
    for v in VARIANTS:
        c = dict(blind_cfg, variants=[v], max_epochs=MAX_E, patience=MAX_E, final_seeds=[0])
        if PROBE == "capacity":
            m0 = blind_cfg["model_candidates"][0]
            c.update(max_epochs=CAP_MAX_E, patience=CAP_PATIENCE,
                     model_candidates=[{**m0, **x, "aux_hidden_dim": x["hidden_dim"]} for x in CAP_CONFIGS])
        if "calibration_lambda" in sel[v]:
            c["calibration"] = {**c.get("calibration", {}), "calibration_lambda": float(sel[v]["calibration_lambda"]),
                                "select_lambda": False}
        jobs.append(job(f"A_cv_{v}", c, "E0_bd_mpi"))
    write_jobs("A", jobs)
elif stage == "B":
    jobs = []
    for e in GRID_B:
        c = base("E0_bd_mpi")
        c["fixed_final_epochs_by_variant"] = {v: e for v in VARIANTS}
        c["final_seeds"] = list(range(N_FINAL))
        jobs.append(job(f"B_blind_e{e:03d}", c, "E0_bd_mpi"))
    write_jobs("B", jobs)
elif stage == "C":
    rows, sel = [], {}
    for v in VARIANTS:
        d = OUT / "runs" / f"A_cv_{v}"
        f = pd.read_csv(d / "crossed_validation_folds.csv")
        h = pd.read_csv(d / "training_history.csv")
        s = json.loads((d / "selection.json").read_text())["variants"][v]
        cid = int(s["selected_config_id"])
        h = h[h.config_id == cid]
        if PROBE == "retune":
            assert (f.epochs_run == MAX_E).all(), (v, "early stopping was not disabled")
        alive = h.groupby("epoch").size()
        curve = h[h.epoch <= alive[alive == alive.max()].index.max()].groupby("epoch").validation.mean()
        proto = int(s["final_epochs"])
        argmin = int(curve.idxmin())
        material = (proto <= 7 and argmin >= 30) or max(proto, argmin) / max(min(proto, argmin), 1) >= 4
        sel[v] = {"model_config": s["model_config"], "protocol": proto, "fold_avg_argmin": argmin, "material": bool(material),
                  "curve_last_minus_min": float(curve.iloc[-1] - curve.min()),
                  "curve_still_falling_at_end": bool(curve.iloc[-20:].diff().mean() < 0)}
    (OUT / "selected_epochs.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))
    budgets = {"cv": {v: sel[v]["protocol"] for v in VARIANTS}}
    if any(sel[v]["material"] for v in VARIANTS):
        budgets["sens"] = {v: sel[v]["fold_avg_argmin"] for v in VARIANTS}
    jobs = []
    for tag, ep in budgets.items():
        for arm, key in (("blind", "E0_bd_mpi"), ("E1", "E1_bd_mpi"), ("D1", "D1_bd_mpi")):
            c = base(key)
            c["fixed_final_epochs_by_variant"] = ep
            c["final_seeds"] = list(range(N_FINAL))
            if PROBE == "capacity":
                c["fixed_model_configs_by_variant"] = {v: sel[v]["model_config"] for v in VARIANTS}
            jobs.append(job(f"C_{tag}_{arm}", c, key))
    write_jobs("C", jobs)
