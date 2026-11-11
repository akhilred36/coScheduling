"""Socket batch 2, control E: epoch-matched final fits (no re-tuning), built with each pipeline's fixed_global mode.

Writes run configs to ~/hpcResearch/experiment_outputs/socket_batch2/epochmatch_configs/ and prints the launch commands.

  E1_<pipe>     AWARE data and socket bit, the BLIND run's full selected recipe (HP config, final epochs, aggregation,
                calibration lambda), all 4 pipelines.
  E2_bd_<t>     BLIND data, the blind recipe but the AWARE selected final epoch counts (bidirectional only).
  E0_bd_<t>     BLIND data, the blind recipe (fixed_global): a control that must reproduce the blind CV run's
                predictions exactly, which shows fixed_global mode replays the CV-selected fit.
One-way: the original evaluate task already ran in fixed_global mode from resolved_config.json, so E1 = the blind
evaluate task's resolved config plus socket_feature on the aware data (E0 is the blind run itself).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path.home() / "hpcResearch/experiment_outputs/socket_batch2"
UT = Path.home() / "hpcResearch/repos/coScheduling/notebooks/utils"
CFG = ROOT / "epochmatch_configs"
CFG.mkdir(exist_ok=True)
PY = "/home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python"
BD = {"mpi": "delta_response_filteredInhibs_biDirectional", "tt": "delta_response_filteredInhibs_biDirectional_totalTime"}
OW = {"mpi": ("delta_response_filteredInhibs", "mpi_time"), "tt": ("delta_response_filteredInhibs_totalTime", "app_time")}
cmds = []


def bd_fixed(base_run: str, recipe_run: str, epoch_run: str) -> dict:
    cfg = json.loads((ROOT / base_run / "requested_run_config.json").read_text())
    rec = json.loads((ROOT / recipe_run / "selection.json").read_text())["variants"]
    ep = json.loads((ROOT / epoch_run / "selection.json").read_text())["variants"]
    cfg["selection_mode"] = "fixed_global"
    cfg["fixed_model_configs_by_variant"] = {n: v["model_config"] for n, v in rec.items()}
    cfg["fixed_final_epochs_by_variant"] = {n: int(ep[n]["final_epochs"]) for n in rec}
    cfg["fixed_aggregations_by_variant"] = {n: v["aggregation"] for n, v in rec.items() if "aggregation" in v}
    cfg["fixed_calibration_lambda_by_variant"] = {n: v["calibration_lambda"] for n, v in rec.items()
                                                  if "calibration_lambda" in v}
    cfg["fixed_selection_source"] = {"recipe_from": recipe_run, "epochs_from": epoch_run, "variants": rec}
    return cfg


for t, code in BD.items():
    for name, base, recipe, epochs, aware in ((f"E0_bd_{t}", f"blind_bd_{t}", f"blind_bd_{t}", f"blind_bd_{t}", False),
                                              (f"E1_bd_{t}", f"aware_bd_{t}", f"blind_bd_{t}", f"blind_bd_{t}", True),
                                              (f"E2_bd_{t}", f"blind_bd_{t}", f"blind_bd_{t}", f"aware_bd_{t}", False)):
        (CFG / f"{name}.json").write_text(json.dumps(bd_fixed(base, recipe, epochs), indent=2))
        data = UT / f"{code}_socketAware/data_socket" if aware else None
        args = [f"cd {UT / (code + '_socketAware')} &&", PY, "pipeline.py", "--output-dir", str(ROOT / name),
                "--config", str(CFG / f"{name}.json")]
        if aware:
            args += ["--jobs-csv", f"{data}/training/jobs.csv", "--inhibitors-csv", f"{data}/training/inhibitors.csv",
                     "--job-inh-csv", f"{data}/training/job_inh.csv", "--pair-csv", f"{data}/evaluation/pair.csv",
                     "--socket-feature"]
        cmds.append((name, " ".join(args)))

for t, (code, basis) in OW.items():
    blind_cfg = None
    for f in glob.glob(str(ROOT / f"blind_1w_{t}/tasks/*/attempts/0001/resolved_config.json")):
        c = json.loads(Path(f).read_text())
        if c.get("selection_mode") == "fixed_global":
            blind_cfg = c
    assert blind_cfg is not None
    blind_cfg["socket_feature"] = True
    name = f"E1_1w_{t}"
    (CFG / f"{name}.json").write_text(json.dumps(blind_cfg, indent=2))
    data = UT / f"{code}_socketAware/data_socket/{basis}"
    cmds.append((name, " ".join([PY, "-u", str(UT / f"{code}_socketAware/pipeline.py"),
                                 "--jobs-csv", f"{data}/jobs.csv", "--inhibitors-csv", f"{data}/inhibitors.csv",
                                 "--job-inh-csv", f"{data}/job_inh.csv", "--config", str(CFG / f"{name}.json"),
                                 "--output-dir", str(ROOT / name), "--pair-csv", f"{data}/pair.csv"])))

with open(CFG / "commands.txt", "w") as fh:
    for name, c in cmds:
        fh.write(f"{name}\t{c}\n")
        print(name, "\n  ", c)
