"""Seed batch check: every finished job's seeds 0-4 sub-ensemble must reproduce its 5-seed reference run.

For each per-seed file (seed_predictions.csv / seed_predictions_socket.csv) under a job's out_dir, the matching reference
file is directional_predictions(.csv|_socket.csv) in the same relative directory of the job's ref_dir. Compared: the
mean latent over seeds 0-4 vs the reference latent, on the shared id keys. Prints max |Δ| per file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home() / "hpcResearch/experiment_outputs/seed_batch"
KEYS = ["evaluation_method", "anchor_budget", "method", "pair_row_id", "direction", "victim", "aggressor", "victim_id",
        "aggressor_id", "s_V"]
J = pd.read_csv(OUT / "jobs.tsv", sep="\t", keep_default_na=False)
rows = []
for r in J.itertuples():
    out = Path(r.out_dir)
    try:
        if json.loads((out / "run_report.json").read_text()).get("status") != "complete":
            continue
    except Exception:
        continue
    if not r.ref_dir:
        rows.append({"job": r.name, "file": "-", "max |Δ latent|": np.nan, "rows": 0, "note": "no reference"})
        continue
    for f in sorted(out.rglob("seed_predictions*.csv")):
        rel = f.relative_to(out)
        ref = Path(r.ref_dir) / rel.parent / f.name.replace("seed_predictions", "directional_predictions")
        if not ref.exists():
            rows.append({"job": r.name, "file": str(rel), "max |Δ latent|": np.nan, "rows": 0, "note": "ref missing"})
            continue
        a = pd.read_csv(f)
        b = pd.read_csv(ref)
        lat_a = "latent" if "latent" in a else "latent_predicted_log_slowdown"
        lat_b = "latent" if "latent" in b else "latent_predicted_log_slowdown"
        if "victim" in b and "victim_id" in a and "victim" not in a:
            a = a.rename(columns={"victim_id": "victim", "aggressor_id": "aggressor"})
        keys = [k for k in KEYS if k in a and k in b]
        if "anchor_budget" in keys:
            a["anchor_budget"] = a.anchor_budget.astype(str)
            b["anchor_budget"] = b.anchor_budget.astype(str)
        a = a[a.seed.isin([0, 1, 2, 3, 4])].groupby(keys, dropna=False)[lat_a].mean().rename("new")
        b = b.groupby(keys, dropna=False)[lat_b].mean().rename("ref")
        m = pd.concat([a, b], axis=1, join="inner")
        rows.append({"job": r.name, "file": str(rel), "max |Δ latent|": float((m.new - m.ref).abs().max()),
                     "rows": len(m), "note": f"ref rows {len(b)}"})
R = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 500)
print(R.to_string(index=False))
R.to_csv(OUT / "check_seeds0to4.csv", index=False)
bad = R[(R["max |Δ latent|"] > 1e-9)]
print(f"\n{len(R)} files checked; {len(bad)} with max |Δ| > 1e-9")
