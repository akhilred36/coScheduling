"""Socket batch 2 diagnostics D1/D2 (director-approved 2026-11-09). Writes configs + commands, like socket_batch2_epochmatch.py.

D1_<pipe>  init vs information: the aware code with the socket bit fed as a CONSTANT 0 (both appended columns), so the
           architecture and RNG draws match the aware model but the bit carries no information. Recipe = E1 (blind
           recipe on aware data). Code = a patched copy of each *_socketAware dir under socket_batch2/d1_code/ (the repo
           copies are untouched; only data.py:socket_pair_inputs changes). D1 - blind = init/architecture effect;
           E1 - D1 = information effect.
           Residual socket pathways that do not go through the input bit (e.g. per-socket context token sets) would
           show up as D1 predictions that differ between s_V = 0 and 1; the scorer checks this.
D2_bd_mpi  lambda vs epochs: aware data/code, the AWARE CV recipe (epochs, aggregation) with the BLIND calibration lambdas.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path.home() / "hpcResearch/experiment_outputs/socket_batch2"
UT = Path.home() / "hpcResearch/repos/coScheduling/notebooks/utils"
CFG = ROOT / "epochmatch_configs"
D1CODE = ROOT / "d1_code"
PY = "/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python"
CODES = {"1w_mpi": "delta_response_filteredInhibs_socketAware", "1w_tt": "delta_response_filteredInhibs_totalTime_socketAware",
         "bd_mpi": "delta_response_filteredInhibs_biDirectional_socketAware",
         "bd_tt": "delta_response_filteredInhibs_biDirectional_totalTime_socketAware"}
PATCH = {  # exact return statements of socket_pair_inputs -> constant-0 socket columns
    "1w": ("        np.column_stack([victims, sockets]).astype(np.float32),\n"
           "        np.column_stack([aggressors, 1.0 - sockets]).astype(np.float32),\n",
           "        np.column_stack([victims, np.zeros_like(sockets)]).astype(np.float32),  # D1: constant 0\n"
           "        np.column_stack([aggressors, np.zeros_like(sockets)]).astype(np.float32),  # D1: constant 0\n"),
    "bd": ("    return np.hstack([victims, sockets]), np.hstack([aggressors, 1.0 - sockets])\n",
           "    zero = np.zeros_like(sockets)  # D1: constant-0 socket bit (same shape and RNG, no information)\n"
           "    return np.hstack([victims, zero]), np.hstack([aggressors, zero])\n"),
}
cmds = []
for key, code in CODES.items():
    dst = D1CODE / code
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in (UT / code).glob("*.py"):
        shutil.copy(f, dst / f.name)
    old, new = PATCH[key[:2]]
    src = (dst / "data.py").read_text()
    assert src.count(old) == 1, key
    (dst / "data.py").write_text(src.replace(old, new))
    e1 = CFG / f"E1_{key}.json"
    cfg = json.loads(e1.read_text())
    (CFG / f"D1_{key}.json").write_text(json.dumps(cfg, indent=2))
    # same command as E1, with the patched code dir, config and output dir
    line = [l.split("\t", 1)[1] for l in (CFG / "commands.txt").read_text().splitlines() if l.startswith(f"E1_{key}\t")][0]
    line = (line.replace(f"{UT / code}/pipeline.py", f"{dst}/pipeline.py").replace(f"cd {UT / code} &&", f"cd {dst} &&")
            .replace(f"E1_{key}.json", f"D1_{key}.json").replace(str(ROOT / f"E1_{key}"), str(ROOT / f"D1_{key}")))
    assert str(dst) in line and f"D1_{key}" in line and f"{UT / code}/data_socket" in line
    cmds.append((f"D1_{key}", line))

# D2: aware recipe, blind lambdas (BD-MPI)
cfg = json.loads((CFG / "E1_bd_mpi.json").read_text())  # aware data/base config
aw = json.loads((ROOT / "aware_bd_mpi/selection.json").read_text())["variants"]
bl = json.loads((ROOT / "blind_bd_mpi/selection.json").read_text())["variants"]
cfg["fixed_model_configs_by_variant"] = {n: v["model_config"] for n, v in aw.items()}
cfg["fixed_final_epochs_by_variant"] = {n: int(v["final_epochs"]) for n, v in aw.items()}
cfg["fixed_aggregations_by_variant"] = {n: v["aggregation"] for n, v in aw.items() if "aggregation" in v}
cfg["fixed_calibration_lambda_by_variant"] = {n: bl[n]["calibration_lambda"] for n, v in aw.items()
                                              if "calibration_lambda" in v}
cfg["fixed_selection_source"] = {"recipe_from": "aware_bd_mpi", "lambda_from": "blind_bd_mpi"}
(CFG / "D2_bd_mpi.json").write_text(json.dumps(cfg, indent=2))
line = [l.split("\t", 1)[1] for l in (CFG / "commands.txt").read_text().splitlines() if l.startswith("E1_bd_mpi\t")][0]
cmds.append(("D2_bd_mpi", line.replace("E1_bd_mpi.json", "D2_bd_mpi.json").replace(str(ROOT / "E1_bd_mpi"),
                                                                                    str(ROOT / "D2_bd_mpi"))))
existing = [l for l in (CFG / "commands.txt").read_text().splitlines() if not l.startswith(("D1_", "D2_"))]
(CFG / "commands.txt").write_text("\n".join(existing + [f"{n}\t{c}" for n, c in cmds]) + "\n")
for n, c in cmds:
    print(n, "\n  ", c)
