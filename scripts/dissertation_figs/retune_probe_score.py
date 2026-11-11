"""Score the re-tune probe (or the capacity probe): usage retune_probe_score.py [retune|capacity]

1. CV selection (stage A, App-Inhibitor crossed CV only): per variant the protocol-selected epochs (median of per-fold
   best epochs), the fold-averaged validation curve argmin (sensitivity), the curve at grid epochs, and whether it is still
   falling at the end. Capacity probe: also the selected model config.
2. Descriptive App-App curve vs epochs (stage B; NOT used for selection), 30-seed ensembles.
3. Final 30-seed fits (stage C) at each budget tag: seed_score.py on a symlink tree (blind -> F_blind_bd_mpi, E1, D1),
   giving the method table (both CIs, floor/blow-up fractions), claims with the unified label rule, k-curve, floor
   decomposition, and the socket E1 - D1 / D1 - blind contrasts.
4. Trigger for the capacity probe (retune only): does joint cal or bprof cal 'help' at any budget
   (claim '<x> cal < baseline' verdict == 'holds', i.e. two-level CI excludes 0 and not sign-unstable)?
Writes output/<probe>_probe_numbers.md and output/<probe>_probe_trigger.json.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = sys.argv[1] if len(sys.argv) > 1 else "retune"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
ROOT = Path(os.environ.get("PROBE_ROOT", Path.home() / f"hpcResearch/experiment_outputs/{PROBE}_probe"))
SB2 = Path.home() / "hpcResearch/experiment_outputs/socket_batch2"
PARTA = Path.home() / "hpcResearch/experiment_outputs/seed_batch/full"
PY = "/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python"
VARIANTS = ["legacy_delta", "legacy_generic", "legacy_absolute", "joint_profile", "bidirectional_profile",
            "bidirectional_context", "bidirectional_profile_wall", "bidirectional_context_forward"]
GRID = [1, 2, 4, 7, 15, 30, 60, 120, 240]
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    df = df.reset_index()
    cells = [[fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("–" if isinstance(
        v, (float, np.floating)) else str(v)) for v in row] for row in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells])


orig = json.loads((SB2 / "blind_bd_mpi/selection.json").read_text())["variants"]
p(f"## {PROBE.capitalize()} probe, BD-MPI (blind CV; final fits blind / E1 / D1, 30 seeds)")
p()
# ------------------------------------------------------------------ 1. CV selection
p("### 1. App-Inhibitor crossed-CV selection (the only selection data)")
rows, curves = [], {}
for v in VARIANTS:
    d = ROOT / "runs" / f"A_cv_{v}"
    if not (d / "selection.json").exists():
        p(f"- {v}: stage A output missing")
        continue
    s = json.loads((d / "selection.json").read_text())["variants"][v]
    f = pd.read_csv(d / "crossed_validation_folds.csv")
    h = pd.read_csv(d / "training_history.csv")
    cid = int(s["selected_config_id"])
    hc = h[h.config_id == cid]
    alive = hc.groupby("epoch").size()
    full_alive = alive[alive == alive.max()].index.max()  # epochs where every fit is still running
    curve = hc[hc.epoch <= full_alive].groupby("epoch").validation.mean()
    curves[v] = curve
    tail = curve.loc[max(1, full_alive - 20):]
    r = {"variant": v, "original epochs (80/patience 10)": int(orig[v]["final_epochs"]),
         "protocol epochs (median per-fold best)": int(s["final_epochs"]),
         "fold-avg argmin": int(curve.idxmin()), "curve epochs with all fits alive": int(full_alive),
         "per-fold best: % at epoch 1": float((f[f.config_id == cid].best_epoch == 1).mean() * 100),
         "grouped CV log MAE at selection": float(s["early_stopping_validation_log_mae"]),
         "original grouped CV log MAE": float(orig[v]["early_stopping_validation_log_mae"]),
         "curve still falling at end": bool(tail.diff().mean() < 0)}
    if PROBE == "capacity":
        r["selected config"] = json.dumps(s["model_config"])
    for e in GRID:
        r[f"curve@{e}"] = float(curve.get(e, np.nan))
    rows.append(r)
SEL = pd.DataFrame(rows)
if not SEL.empty:
    p("Curve = mean per-fit validation error (calibrated inference) over all fits of the selected config, at epochs where "
      "every fit is still running. Protocol = median of per-fold best epochs (the selector); argmin = sensitivity.")
    p(md(SEL.drop(columns=[c for c in SEL if c.startswith("curve@")]).set_index("variant")))
    p()
    p(md(SEL.set_index("variant")[[c for c in SEL if c.startswith("curve@")]]))
    p()
    SEL.to_csv(OUT_DIR / f"{PROBE}_probe_selection.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / f"{PROBE}_probe_cv_curves.csv")

# ------------------------------------------------------------------ 2. descriptive App-App curve
ROLES = {"absolute": "legacy_absolute", "joint_cal": "joint_profile_calibrated", "joint_direct": "joint_profile_direct",
         "bprof_cal": "bidirectional_profile_calibrated", "bctx_cal": "bidirectional_context_calibrated",
         "delta_uniform": "legacy_delta_uniform", "delta_median": "legacy_delta_median", "generic": "legacy_generic_uniform"}
brow = []
src = [("current recipe (part a, 100 seeds)", PARTA / "F_blind_bd_mpi")] + \
      [(f"all variants at {e} epochs (30 seeds)", ROOT / "runs" / f"B_blind_e{e:03d}") for e in (7, 15, 30, 60, 120, 240)]
for lab, d in src:
    mp = d / "metrics_non_self.csv"
    if not mp.exists():
        continue
    m = pd.read_csv(mp)
    m = m[m.evaluation_method == "random_split"].set_index("method")
    base = m.loc["constant_1", "log_mae"]
    r = {"run": lab, "baseline log MAE": f"{base:.4f}"}
    for role, code in ROLES.items():
        if code in m.index:
            r[f"{role} Δ%"] = 100 * (m.loc[code, "log_mae"] / base - 1)
    brow.append(r)
if brow:
    p("### 2. App-App error vs training budget — DESCRIPTIVE ONLY, not used for any selection")
    p("Δ% of the N-seed mean-latent ensemble vs the isolated-speed baseline, native non-self outcomes (pipeline metrics).")
    B = pd.DataFrame(brow)
    p(md(B.set_index("run"), "{:+.1f}"))
    p()
    B.to_csv(OUT_DIR / f"{PROBE}_probe_appapp_curve.csv", index=False)

# ------------------------------------------------------------------ 3. final fits via seed_score.py
trigger = {}
for tag in ("cv", "sens"):
    runs = {a: ROOT / "runs" / f"C_{tag}_{a}" for a in ("blind", "E1", "D1")}
    if not (runs["blind"] / "run_report.json").exists():
        continue
    tree = ROOT / f"score_tree_{tag}"
    tree.mkdir(exist_ok=True)
    for a, d in runs.items():
        link = tree / f"F_{a}_bd_mpi"
        if (d / "run_report.json").exists() and not link.exists():
            link.symlink_to(d)
    otag = f"_{PROBE}_{tag}"
    env = dict(os.environ, SEED_SB=str(tree), SEED_TAG=otag)
    subprocess.run([PY, str(HERE / "seed_score.py")], env=env, check=True, stdout=subprocess.DEVNULL)
    title = "CV-selected budget (protocol selector)" if tag == "cv" else \
        "SENSITIVITY: fold-averaged-argmin budget (not the protocol selector)"
    p(f"### 3{'a' if tag == 'cv' else 'b'}. Final fits, {title}")
    p((OUT_DIR / f"seed_numbers{otag}.md").read_text().split("\n", 1)[1])
    C = pd.read_csv(OUT_DIR / f"seed_claims{otag}.csv")
    for claim in ("joint cal < baseline", "bprof cal < baseline"):
        r = C[(C.pipeline == "BD-MPI") & (C.claim == claim)]
        if len(r):
            trigger[f"{tag}: {claim}"] = r.verdict.iloc[0]

# near-init comparison of single-seed degeneracy
pa = OUT_DIR / "seed_methods.csv"
if pa.exists():
    M = pd.read_csv(pa)
    M = M[M.pipeline == "BD-MPI"].set_index("role")[["1-seed at floor everywhere", "1-seed blown up (Δ% > +25)"]]
    p("### Reference: single-seed degeneracy under the near-init recipe (part a, BD-MPI blind, 100 seeds)")
    p(md(M))
    p()

if PROBE == "retune":
    helps = any(v == "holds" for v in trigger.values())
    trig = {"verdicts": trigger, "a calibrated variant helps": helps, "run capacity probe": not helps}
    (OUT_DIR / "retune_probe_trigger.json").write_text(json.dumps(trig, indent=2))
    p("### 4. Capacity-probe trigger")
    p(f"Verdicts: {trigger}. A calibrated variant helps (two-level CI excludes 0, not sign-unstable): {helps}. "
      f"Capacity probe {'NOT run' if helps else 'started'}.")
(OUT_DIR / f"{PROBE}_probe_numbers.md").write_text(log.getvalue())
print("wrote", OUT_DIR / f"{PROBE}_probe_numbers.md")
