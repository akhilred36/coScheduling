"""Socket batch 2: socket-resolved training snapshots for the four *_socketAware pipeline copies.

Resolved App-Inhibitor rows = one per (app, inhibitor config, victim socket s_V), with the ratio of aggregate means over
the replicates that had that s_V (isolated baselines: June 4-rep means, unchanged). Built by running each pipeline's
UNMODIFIED prepare functions once per socket on a socket-split aggregate file, then concatenating with
replicate_id = s_V, observation_id = job|inhib|s_V and an explicit `s_V` column. All targets (slowdown, raw ratios,
auxiliary rate changes) are row-wise, so this equals a patched prepare.

pair.csv is the blind evaluation file plus n0_A, n1_A, n0_B, n1_B = number of replicates with that endpoint on
socket 0 / 1 (used to aggregate socket-conditioned predictions to rep-averaged outcomes, B2). Self pairs get 2/2
(each self-pair replicate has one copy per socket; endpoint identity within a self pair is not recoverable).

Outputs:
  notebooks/utils/<copy>/data_socket/{training,evaluation}/   (BD copies)
  notebooks/utils/<copy>/data_socket/{mpi_time,app_time}/       (one-way copies)
  scripts/dissertation_figs/output/socket_resolved_{appinh,appapp}.csv  (resolved cells with rep counts, both targets)
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR, PROCESSED, REPO

UT = REPO / "notebooks/utils"
CONFIG = ["App", "Num Nodes", "Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]
MEAS = ["App Time Average", "MPI Time Average", "Total Messages Sent", "Total Bytes Sent", "Inhib App Time Average",
        "Inhib MPI Time Average", "Inhib Total Messages Sent", "Inhib Total Bytes Sent"]


def load_module(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses look the module up while it executes
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def agg(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = frame.groupby(keys, sort=True)[MEAS]
    out = pd.concat([g.mean().add_suffix(" Mean"), g.median().add_suffix(" Median"), g.std().add_suffix(" Std")], axis=1)
    order = [f"{m} {s}" for m in MEAS for s in ("Mean", "Median", "Std")]
    return out[order].reset_index()


# ------------------------------------------------------------------ socket-split App-Inhibitor aggregates
assign = pd.read_csv(OUT_DIR / "socket_corun_assign.csv")
co = pd.read_csv(PROCESSED / "data_inhib_coscheduled.csv", index_col=0)
co = co[co["Num Nodes"] == 8].copy()
ai = assign[assign.campaign == "AI"].rename(columns={"victim": "App", "rep": "Run Iteration"})
co["_sp"] = co["Inhib Comm Sparsity"].round(9)
co = co.merge(ai[["App", "m", "w", "sp", "Run Iteration", "s_V"]].rename(
    columns={"m": "Inhib Message Size", "w": "Inhib Wait Time (us)", "sp": "_sp"}),
    on=["App", "Inhib Message Size", "Inhib Wait Time (us)", "_sp", "Run Iteration"], how="left", validate="one_to_one")
assert co.s_V.isin([0, 1]).all(), "every App-Inhibitor replicate needs a socket"
co = co.drop(columns="_sp")

# check: blind re-aggregation reproduces processed_data/data_inhib_coscheduled_agg.csv
ref = pd.read_csv(PROCESSED / "data_inhib_coscheduled_agg.csv", index_col=0)
ref = ref[ref["Num Nodes"] == 8]
mine = agg(co, CONFIG)
chk = ref.merge(mine, on=CONFIG, suffixes=("", "_mine"), validate="one_to_one")
assert len(chk) == len(ref) == len(mine), (len(chk), len(ref), len(mine))
err = max(np.nanmax(np.abs(chk[f"{m} Mean"] - chk[f"{m} Mean_mine"]) / np.abs(chk[f"{m} Mean"]).clip(lower=1e-12))
          for m in MEAS)
print(f"blind re-aggregation vs processed agg: {len(chk)} configs, max rel diff of Mean columns {err:.2e}")
assert err < 1e-12

by_sock = agg(co, CONFIG + ["s_V"])
by_sock = by_sock.merge(co.groupby(CONFIG + ["s_V"]).size().rename("n_reps").reset_index(), on=CONFIG + ["s_V"])

# ------------------------------------------------------------------ App-App rep counts per endpoint socket
aa = assign[assign.campaign == "AA"]
cnt = aa.groupby(["victim", "aggressor"]).s_V.agg(n0=lambda s: int((s == 0).sum()), n1=lambda s: int((s == 1).sum()))


def add_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    for e, o in (("A", "B"), ("B", "A")):
        key = list(zip(pairs[f"job{e}_id"], pairs[f"job{o}_id"]))
        self_ = pairs[f"job{e}_id"] == pairs[f"job{o}_id"]
        pairs[f"n0_{e}"] = [2 if s else int(cnt.loc[k, "n0"]) for k, s in zip(key, self_)]
        pairs[f"n1_{e}"] = [2 if s else int(cnt.loc[k, "n1"]) for k, s in zip(key, self_)]
    return pairs


def tag(resp: pd.DataFrame, s: int, has_obs: bool) -> pd.DataFrame:
    resp = resp.copy()
    if has_obs:
        resp["replicate_id"] = s
        resp["observation_id"] = resp["job_id"] + "|" + resp["inhib_id"] + "|" + str(s)
    resp["s_V"] = s
    return resp


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for s in (0, 1):
        by_sock[by_sock.s_V == s].drop(columns=["s_V", "n_reps"]).to_csv(tmp / f"ai_s{s}.csv")

    # ---------------- bidirectional copies
    for name in ("delta_response_filteredInhibs_biDirectional_socketAware",
                 "delta_response_filteredInhibs_biDirectional_totalTime_socketAware"):
        mod = load_module(UT / name / "prepare_bidirectional_data.py", f"prep_{name}")
        out = UT / name / "data_socket"
        parts = []
        for s in (0, 1):
            d = tmp / f"{name}_s{s}"
            mod.prepare_training_data(PROCESSED / "data_isolated_agg.csv", PROCESSED / "data_inhib_isolated_agg.csv",
                                      tmp / f"ai_s{s}.csv", 8, d, legacy_dir=None)
            parts.append(tag(pd.read_csv(d / "job_inh.csv"), s, True))
        (out / "training").mkdir(parents=True, exist_ok=True)
        (out / "evaluation").mkdir(parents=True, exist_ok=True)
        for f in ("jobs.csv", "inhibitors.csv", "inhibitor_id_map.csv"):
            shutil.copy(tmp / f"{name}_s0" / f, out / "training" / f)
        resolved = pd.concat(parts, ignore_index=True).sort_values(["job_id", "inhib_id", "s_V"]).reset_index(drop=True)
        resolved.to_csv(out / "training" / "job_inh.csv", index=False)
        blind_eval = UT / name.replace("_socketAware", "") / "data" / "evaluation" / "pair.csv"
        add_counts(pd.read_csv(blind_eval)).to_csv(out / "evaluation" / "pair.csv", index=False)
        blind = pd.read_csv(UT / name.replace("_socketAware", "") / "data" / "training" / "job_inh.csv")
        print(f"{name}: resolved rows {len(resolved)} (floor {int(resolved.is_censored.sum())}), "
              f"blind rows {len(blind)} (floor {int(blind.is_censored.sum())})")

    # ---------------- one-way copies (identical code; data for both bases)
    mod = load_module(UT / "delta_response_filteredInhibs_totalTime_socketAware" / "prepare_data.py", "prep_1w")
    jobs, inhibitors = mod.build_profiles(mod._read(PROCESSED / "data_isolated_agg.csv", 8),
                                          mod._read(PROCESSED / "data_inhib_isolated_agg.csv", 8))
    for name in ("delta_response_filteredInhibs_socketAware", "delta_response_filteredInhibs_totalTime_socketAware"):
        for basis in ("mpi_time", "app_time"):
            out = UT / name / "data_socket" / basis
            out.mkdir(parents=True, exist_ok=True)
            parts = [tag(mod.build_responses(mod._read(tmp / f"ai_s{s}.csv", 8), jobs, inhibitors, basis), s, False)
                     for s in (0, 1)]
            resolved = pd.concat(parts, ignore_index=True).sort_values(["job_id", "inhib_id", "s_V"])
            jobs.to_csv(out / "jobs.csv", index=False)
            inhibitors.to_csv(out / "inhibitors.csv", index=False)
            resolved.to_csv(out / "job_inh.csv", index=False)
            blind_pairs = UT / "delta_response_filteredInhibs_totalTime" / "data" / basis / "pair.csv"
            add_counts(pd.read_csv(blind_pairs)).to_csv(out / "pair.csv", index=False)
            # blind check: the per-basis blind files used for the reproductions
            blind = pd.read_csv(UT / "delta_response_filteredInhibs_totalTime" / "data" / basis / "job_inh.csv")
            if name.endswith("totalTime_socketAware"):
                print(f"one-way {basis}: resolved rows {len(resolved)} (floor {int((resolved.slowdown == 1).sum())}), "
                      f"blind rows {len(blind)} (floor {int((blind.slowdown == 1).sum())})")

# ------------------------------------------------------------------ resolved cell tables for scoring
iso = pd.read_csv(PROCESSED / "data_isolated.csv", index_col=0)
iso = iso[iso["Num Nodes"] == 8].groupby("App")[["MPI Time Average", "App Time Average"]].mean()
cell = by_sock[CONFIG + ["s_V", "n_reps", "MPI Time Average Mean", "App Time Average Mean"]].copy()
cell["ratio_mpi"] = cell["MPI Time Average Mean"] / cell.App.map(iso["MPI Time Average"])
cell["ratio_tt"] = cell["App Time Average Mean"] / cell.App.map(iso["App Time Average"])
cell.to_csv(OUT_DIR / "socket_resolved_appinh.csv", index=False)

cc = pd.read_csv(PROCESSED / "data_coscheduled.csv", index_col=0)
cc = cc[(cc["Num Nodes"] == 8) & (cc["App A"] != cc["App B"])]
rows = []
for r in cc.to_dict("records"):
    for v, x in (("A", "B"), ("B", "A")):
        rows.append(dict(victim=r[f"App {v}"], aggressor=r[f"App {x}"], rep=r["Run Iteration"],
                         T_mpi=r[f"App {v} MPI Time Average"], T_tt=r[f"App {v} Time Average"]))
d = pd.DataFrame(rows).merge(aa[["victim", "aggressor", "rep", "s_V"]], on=["victim", "aggressor", "rep"],
                             validate="one_to_one")
res = d.groupby(["victim", "aggressor", "s_V"]).agg(n_reps=("rep", "size"), T_mpi=("T_mpi", "mean"),
                                                    T_tt=("T_tt", "mean")).reset_index()
res["ratio_mpi"] = res.T_mpi / res.victim.map(iso["MPI Time Average"])
res["ratio_tt"] = res.T_tt / res.victim.map(iso["App Time Average"])
res.to_csv(OUT_DIR / "socket_resolved_appapp.csv", index=False)
print(f"App-App resolved cells {len(res)}; App-Inh resolved cells {len(cell)}")
