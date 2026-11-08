"""Batch 2 A1: inhibitor panel sizes (full data + restricted tiers) and floor-row counts.

Reproduces run_experiments.nested_orders(jobs, 10, seed=811) and the
select_inhibitors.py objective with the training apps as the selection population.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from common import BIDIR, BIDIR_TT, OUT_DIR, NAME

sys.path.insert(0, str(BIDIR))
from select_inhibitors import communication_features, select_panel  # noqa: E402

out = []


def p(*a):
    print(*a)
    out.append(" ".join(str(x) for x in a))


def panel(jobs, inh, ji, target_apps):
    jobs = jobs[jobs.job_id.isin(target_apps)].sort_values("job_id").reset_index(drop=True)
    resp = ji[ji.job_id.isin(jobs.job_id)]
    counts = resp.groupby("inhib_id").job_id.nunique()
    inh = inh.sort_values("inhib_id").reset_index(drop=True)
    cand = inh[inh.inhib_id.map(counts).fillna(0).eq(len(jobs))].reset_index(drop=True)
    x, z = communication_features(jobs), communication_features(cand)
    mean, scale = x.mean(0), x.std(0)
    active = scale > 1e-8
    scale[~active] = 1.0
    d = (((x[:, None, active] - z[None, :, active]) / scale[active]) ** 2).mean(axis=2)
    sel, _, status = select_panel(d, 10, 0.5)
    return cand.inhib_id.iloc[sel].tolist(), status


for label, root in (("MPI time", BIDIR), ("total time", BIDIR_TT)):
    jobs = pd.read_csv(root / "data/training/jobs.csv")
    inh = pd.read_csv(root / "data/training/inhibitors.csv")
    ji = pd.read_csv(root / "data/training/job_inh.csv")
    ids, status = panel(jobs, inh, ji, jobs.job_id.tolist())
    rows = ji[ji.inhib_id.isin(ids)]
    p(f"## {label}: full-data panel = {len(ids)} inhibitors ({status}); training rows = {len(rows)}, "
      f"floor rows = {int(rows.is_censored.sum())}; all App-Inhibitor floor rows = {int(ji.is_censored.sum())}/{len(ji)}")
    knobs = inh.set_index("inhib_id").loc[ids]
    p("   members (-m bytes, -w us, -s sparsity; -c d): " + "; ".join(
        f"{i} = (-m {int(r.msg_size)}, -w {int(r.wait_time)}, -s {r.comm_sparsity:.9g})" for i, r in knobs.iterrows()))
    p("   per-app floor rows in panel: " + ", ".join(f"{NAME[k]} {int(v)}" for k, v in rows.groupby('job_id').is_censored.sum().items()))
    if label == "MPI time":
        base = np.random.default_rng(811).permutation(jobs.job_id.astype(str).tolist()).tolist()
        orders = [base[o:] + base[:o] for o in range(10)]
        p("   base order (split s00): " + ", ".join(NAME[a] for a in base))
        recs = []
        for s, order in enumerate(orders):
            for size in range(2, 9):
                ids_r, st = panel(jobs, inh, ji, order[:size])
                recs.append(dict(split=f"balanced_s{s:02d}", size=size, panel=len(ids_r), status=st))
        r = pd.DataFrame(recs)
        p("   restricted panel size by training size (min / median / max over 10 splits):")
        for size, g in r.groupby("size"):
            p(f"     size {size}: {g.panel.min()} / {g.panel.median():g} / {g.panel.max()}  "
              f"(splits with panel < 4 CV blocks: {(g.panel < 4).sum()})")
        r.to_csv(OUT_DIR / "restricted_panel_sizes.csv", index=False)

(OUT_DIR / "batch2_panels.md").write_text("\n".join(out) + "\n")
