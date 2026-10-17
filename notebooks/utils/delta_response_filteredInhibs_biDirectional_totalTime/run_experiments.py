"""Plan, run, and consolidate bidirectional filtered-inhibitor experiments.

One global tuning task runs crossed App-Inhibitor validation for every
registered variant (including the wall-clock-only and forward-only-token
ablations) and freezes model-specific recipes: epochs, legacy aggregations,
and calibration lambdas. Fixed-global evaluation tasks then refit weights and
scalers per condition and score random_split plus balanced one_known/zero_shot
training-size conditions. App-App outcomes are never available to tuning.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from pipeline import ABLATION_VARIANTS, DEFAULT_VARIANTS, VARIANT_REGISTRY


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"
DEFAULT_TRAINING = ROOT / "data" / "training"
DEFAULT_EVALUATION = ROOT / "data" / "evaluation"
SCHEMA_VERSION = 1
TUNING_STAGES = ["tune_global"]
EVALUATION_STAGES = ["evaluate_random_split", "evaluate_training_size"]
CODE_FILES = [
    "pipeline.py", "run_experiments.py", "data.py", "contexts.py", "inference.py", "metrics.py",
    "model.py", "training.py", "select_inhibitors.py",
]
REPORT_FILES = {
    "metrics_non_self.csv": "overall_non_self",
    "metrics_repeated_self.csv": "overall_repeated_self",
    "metrics_self_averaged.csv": "overall_self_averaged",
    "metrics_non_self_macro_victim.csv": "non_self_macro_victim",
    "metrics_by_seed.csv": "by_seed",
    "cluster_bootstrap_non_self.csv": "non_self_cluster_bootstrap",
    "paired_method_differences_non_self.csv": "non_self_paired_difference",
}
CONSOLIDATED_TABLES = [
    "ablation_summary.csv", "pair_metrics.csv", "metrics_by_censoring.csv", "auxiliary_metrics.csv",
    "auxiliary_validation_metrics.csv", "matched_fold_comparisons.csv", "method_coverage.csv",
    "calibration_diagnostics.csv", "context_support_manifest.csv",
]


@dataclass(frozen=True)
class ExecutionProfile:
    cv_seeds: list[int]
    final_seeds: list[int]
    max_epochs: int
    patience: int
    batches_per_epoch: int
    batch_size: int
    bootstrap_samples: int
    split_replicates: int
    training_sizes: list[int]


PROFILES = {
    "smoke": ExecutionProfile([0], [0], 2, 1, 1, 32, 20, 1, [2]),
    "standard": ExecutionProfile(list(range(5)), list(range(5)), 80, 10, 4, 128, 1000, 10, list(range(2, 9))),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()[:length]}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_planned_inputs(task: dict[str, Any]) -> None:
    actual = {name: sha256_file(Path(path)) for name, path in task["inputs"].items()}
    planned = task.get("input_sha256", {})
    if actual != planned:
        changed = sorted(name for name in set(actual) | set(planned) if actual.get(name) != planned.get(name))
        raise RuntimeError(f"Experiment inputs changed after planning; create a new plan. Changed: {changed}")


def base_run_config(profile: ExecutionProfile, variants: list[str], context: dict[str, Any],
                    calibration: dict[str, Any], inhibitor_blocks: int = 4) -> dict[str, Any]:
    return {
        "inhibitor_blocks": inhibitor_blocks, "clustering_algorithm": "kmeans", "clustering_seed": 1701,
        "cv_seeds": profile.cv_seeds, "final_seeds": profile.final_seeds,
        "max_epochs": profile.max_epochs, "patience": profile.patience,
        "batches_per_epoch": profile.batches_per_epoch, "batch_size": profile.batch_size,
        "temperatures": [0.1, 0.3, 1.0, 3.0, 10.0], "ood_quantiles": [0.9, 0.95, 0.99],
        "ood_fallbacks": ["uniform", "median"], "bootstrap_samples": profile.bootstrap_samples,
        "bootstrap_seed": 923, "device": "cpu", "selection_mode": "crossed_cv",
        "variants": variants, "lambda_abs": 1.0, "lambda_delta": 1.0, "lambda_aux": 0.1,
        "context": context, "calibration": calibration, "epoch_selection_inference": "calibrated",
    }


def nested_orders(jobs: pd.DataFrame, count: int, seed: int) -> list[list[str]]:
    base = np.random.default_rng(seed).permutation(jobs["job_id"].astype(str).tolist()).tolist()
    return [base[offset:] + base[:offset] for offset in range(count)]


def make_task(*, stage: str, inputs: dict[str, str], config: dict[str, Any], opens_holdout: bool,
              depends_on: str | None = None, candidate_strategy: str = "fixed",
              split_id: str | None = None, training_app_count: int | None = None) -> dict[str, Any]:
    task = {
        "schema_version": SCHEMA_VERSION, "stage": stage, "data_kind": "prepared_bidirectional",
        "inputs": inputs, "input_sha256": {name: sha256_file(Path(path)) for name, path in inputs.items()},
        "config": config, "opens_holdout": opens_holdout, "split_id": split_id,
        "split_strategy": "balanced_random" if split_id else None,
        "training_app_count": training_app_count, "depends_on": depends_on,
        "candidate_strategy": candidate_strategy,
    }
    task["task_id"] = stable_id("run", task)
    return task


def build_plan(root: Path, profile_name: str, jobs_csv: Path, inhibitors_csv: Path, job_inh_csv: Path,
               pair_csv: Path, *, random_split_only: bool = False, variants: list[str] | None = None,
               context: dict[str, Any] | None = None, calibration: dict[str, Any] | None = None,
               frozen_panel_csv: Path | None = None, inhibitor_blocks: int = 4) -> list[dict[str, Any]]:
    profile = PROFILES[profile_name]
    variants = list(variants or [*DEFAULT_VARIANTS, *ABLATION_VARIANTS])
    unknown = sorted(set(variants) - set(VARIANT_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    paths = {"jobs": jobs_csv.resolve(), "inhibitors": inhibitors_csv.resolve(),
             "job_inh": job_inh_csv.resolve(), "pair": pair_csv.resolve()}
    for name, path in [("training_preparation_report", jobs_csv.resolve().parent / "preparation_report.json"),
                       ("evaluation_preparation_report", pair_csv.resolve().parent / "preparation_report.json")]:
        if path.is_file():
            paths[name] = path
    if frozen_panel_csv is not None:
        paths["frozen_panel"] = frozen_panel_csv.resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment inputs do not exist: {missing}")
    jobs = pd.read_csv(paths["jobs"])
    if "job_id" not in jobs or jobs["job_id"].duplicated().any():
        raise ValueError("jobs.csv must contain unique job_id values")
    if profile_name == "standard" and len(jobs) != 10:
        raise ValueError("The standard design requires exactly 10 applications")
    training_inputs = {name: str(path) for name, path in paths.items()
                       if name not in {"pair", "evaluation_preparation_report"}}
    evaluation_inputs = {name: str(path) for name, path in paths.items()}

    def config_for(**updates: Any) -> dict[str, Any]:
        config = base_run_config(profile, variants, context or {}, calibration or {}, inhibitor_blocks)
        if frozen_panel_csv is not None:
            config["frozen_panel_csv"] = str(paths["frozen_panel"])
        config.update(updates)
        return config

    tune = make_task(stage="tune_global", inputs=training_inputs,
                     config=config_for(evaluation_method="random_split", training_apps=None),
                     opens_holdout=False)
    tasks = [tune]
    tasks.append(make_task(stage="evaluate_random_split", inputs=evaluation_inputs,
                           config=config_for(evaluation_method="random_split", training_apps=None),
                           opens_holdout=True, depends_on=tune["task_id"],
                           candidate_strategy="fixed_global_from_dependency", training_app_count=len(jobs)))
    if not random_split_only:
        membership = []
        for split_index, order in enumerate(nested_orders(jobs, min(profile.split_replicates, len(jobs)), 811)):
            split_id = f"balanced_s{split_index:02d}"
            for size in profile.training_sizes:
                if size >= len(jobs):
                    raise ValueError("Restricted training sizes must leave an unknown application")
                membership.extend({"split_id": split_id, "training_app_count": size, "app_id": app,
                                   "addition_rank": rank, "in_training": rank <= size}
                                  for rank, app in enumerate(order, start=1))
                tasks.append(make_task(
                    stage="evaluate_training_size", inputs=evaluation_inputs,
                    config=config_for(evaluation_method="both", training_apps=order[:size]),
                    opens_holdout=True, depends_on=tune["task_id"],
                    candidate_strategy="fixed_global_from_dependency", split_id=split_id,
                    training_app_count=size,
                ))
        pd.DataFrame(membership).to_csv(root / "subset_membership.csv", index=False)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("Experiment plan produced duplicate task IDs")
    return tasks


def plan_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment plan: {root / 'plan.json'}")
    context = {} if args.context_max_support is None else {"max_support": args.context_max_support}
    if args.context_crossfit_folds is not None:
        context["crossfit_folds"] = args.context_crossfit_folds
    calibration = {}
    if args.calibration_lambda:
        calibration = ({"calibration_lambda": args.calibration_lambda[0], "select_lambda": False}
                       if len(args.calibration_lambda) == 1
                       else {"lambda_candidates": args.calibration_lambda, "select_lambda": True})
    tasks = build_plan(root, args.profile, args.jobs_csv, args.inhibitors_csv, args.job_inh_csv, args.pair_csv,
                       random_split_only=args.random_split_only, variants=args.variants, context=context,
                       calibration=calibration, frozen_panel_csv=args.frozen_panel_csv,
                       inhibitor_blocks=args.inhibitor_blocks)
    write_json(root / "experiment_spec.json", {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(), "profile": args.profile,
        "profile_settings": asdict(PROFILES[args.profile]), "task_count": len(tasks),
        "random_split_only": args.random_split_only,
        "variants": tasks[0]["config"]["variants"],
        "panel_policy": "supplied frozen panel" if args.frozen_panel_csv else "recomputed per task from its training applications",
        "holdout_policy": "App-App data are sealed during global tuning and opened only by fixed-selection evaluation tasks.",
        "global_hyperparameter_selection_scope": (
            "Transductive: all evaluation applications' profiles and App-Inhibitor responses may select "
            "the global recipe; App-App outcomes never do."
        ),
    })
    write_json(root / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
    })
    write_json(root / "plan.json", tasks)
    refresh_manifest(root, tasks)
    print(f"Planned {len(tasks)} tasks in {root}")


def task_root(root: Path, task_id: str) -> Path:
    return root / "tasks" / task_id


def latest_status(root: Path, task_id: str) -> dict[str, Any] | None:
    path = task_root(root, task_id) / "status.json"
    return read_json(path) if path.exists() else None


def successful_output(root: Path, task_id: str) -> Path | None:
    status = latest_status(root, task_id)
    if not status or status.get("status") != "succeeded":
        return None
    output = Path(status["output_dir"])
    report = output / "run_report.json"
    success_path = task_root(root, task_id) / "_SUCCESS.json"
    if not report.exists() or not success_path.exists() or read_json(report).get("status") != "complete":
        return None
    if read_json(success_path).get("run_report_sha256") != sha256_file(report):
        return None
    return output


def frozen_recipe(selection: dict[str, Any], selection_path: Path, dependency: str) -> dict[str, Any]:
    """Freeze model-specific recipes from the global tuning selection."""
    variants = selection["variants"]
    return {
        "selection_mode": "fixed_global",
        "fixed_final_epochs_by_variant": {name: entry["final_epochs"] for name, entry in variants.items()},
        "fixed_model_configs_by_variant": {name: entry["model_config"] for name, entry in variants.items()},
        "fixed_aggregations_by_variant": {name: entry["aggregation"] for name, entry in variants.items()
                                          if "aggregation" in entry},
        "fixed_calibration_lambda_by_variant": {name: entry["calibration_lambda"] for name, entry in variants.items()
                                                if "calibration_lambda" in entry},
        "fixed_selection_source": {
            "task_id": dependency, "selection_sha256": sha256_file(selection_path),
            "selection_scope": "all evaluation application profiles and App-Inhibitor responses; no App-App outcomes",
            "searched_settings": ["final_epochs", "legacy aggregation parameters", "calibration_lambda"],
            "fixed_settings": ["architecture", "optimizer", "lambda_abs", "lambda_delta", "lambda_aux", "context recipe"],
        },
    }


def resolve_task_config(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(task["config"]))
    if task["candidate_strategy"] == "fixed":
        return config
    if task["candidate_strategy"] != "fixed_global_from_dependency":
        raise ValueError(f"Unknown candidate strategy: {task['candidate_strategy']}")
    output = successful_output(root, task["depends_on"])
    if output is None:
        raise RuntimeError(f"Dependency {task['depends_on']} has not completed successfully")
    config.update(frozen_recipe(read_json(output / "selection.json"), output / "selection.json", task["depends_on"]))
    return config


def execution_fingerprint(root: Path, task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    verify_planned_inputs(task)
    dependency_hash = None
    if task.get("depends_on"):
        output = successful_output(root, task["depends_on"])
        if output is None:
            raise RuntimeError(f"Dependency {task['depends_on']} has not completed successfully")
        dependency_hash = sha256_file(output / "selection.json")
    return {
        "inputs": {name: sha256_file(Path(path)) for name, path in task["inputs"].items()},
        "code": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "resolved_config": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "dependency_selection": dependency_hash,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "torch": torch.__version__},
    }


def execute_task(root: Path, task: dict[str, Any], resume: bool) -> dict[str, Any]:
    if task.get("depends_on") and successful_output(root, task["depends_on"]) is None:
        return {"task_id": task["task_id"], "status": "blocked"}
    config = resolve_task_config(root, task)
    fingerprint = execution_fingerprint(root, task, config)
    success_path = task_root(root, task["task_id"]) / "_SUCCESS.json"
    if resume and successful_output(root, task["task_id"]) is not None and success_path.exists():
        if read_json(success_path).get("execution_fingerprint") == fingerprint:
            return {"task_id": task["task_id"], "status": "skipped"}
    directory = task_root(root, task["task_id"])
    directory.mkdir(parents=True, exist_ok=True)
    old_status = latest_status(root, task["task_id"])
    attempt = int(old_status.get("attempt", 0) if old_status else 0) + 1
    attempt_dir = directory / "attempts" / f"{attempt:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    output_dir = attempt_dir / "outputs"
    config_path = attempt_dir / "resolved_config.json"
    write_json(config_path, config)
    write_json(attempt_dir / "task_spec.json", task)
    command = [sys.executable, "-u", str(PIPELINE),
               "--jobs-csv", task["inputs"]["jobs"], "--inhibitors-csv", task["inputs"]["inhibitors"],
               "--job-inh-csv", task["inputs"]["job_inh"], "--config", str(config_path),
               "--output-dir", str(output_dir)]
    command += ["--pair-csv", task["inputs"]["pair"]] if task["opens_holdout"] else ["--skip-holdout"]
    write_json(attempt_dir / "command.json", {"argv": command})
    status = {"task_id": task["task_id"], "status": "running", "attempt": attempt, "started_at": utc_now(),
              "output_dir": str(output_dir.resolve()), "pid": os.getpid(), "hostname": platform.node()}
    write_json(directory / "status.json", status)
    started = time.time()
    environment = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                   "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1")}
    with (attempt_dir / "stdout.log").open("w") as stdout, (attempt_dir / "stderr.log").open("w") as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False, env=environment)
    status.update({"status": "succeeded" if completed.returncode == 0 else "failed",
                   "return_code": completed.returncode, "finished_at": utc_now(),
                   "elapsed_seconds": time.time() - started})
    report = output_dir / "run_report.json"
    if completed.returncode == 0 and (not report.exists() or read_json(report).get("status") != "complete"):
        status.update({"status": "failed", "error": "Pipeline did not write a complete run_report.json"})
    write_json(directory / "status.json", status)
    if status["status"] == "succeeded":
        write_json(success_path, {"task_id": task["task_id"], "attempt": attempt,
                                  "resolved_config_sha256": sha256_file(config_path),
                                  "run_report_sha256": sha256_file(report),
                                  "execution_fingerprint": fingerprint})
    return status


def selected_tasks_with_dependencies(tasks: list[dict[str, Any]], stages: set[str], max_tasks: int | None,
                                     root: Path | None = None, resume: bool = False) -> list[dict[str, Any]]:
    selected = [task for task in tasks if task["stage"] in stages]
    if max_tasks is not None:
        if resume and root is not None:
            selected = [task for task in selected if successful_output(root, task["task_id"]) is None]
        selected = selected[:max_tasks]
    by_id = {task["task_id"]: task for task in tasks}
    selected_ids = {task["task_id"] for task in selected}
    pending = [task.get("depends_on") for task in selected]
    while pending:
        dependency = pending.pop()
        if dependency and dependency not in selected_ids:
            selected_ids.add(dependency)
            selected.append(by_id[dependency])
            pending.append(by_id[dependency].get("depends_on"))
    order = {task["task_id"]: index for index, task in enumerate(tasks)}
    return sorted(selected, key=lambda task: order[task["task_id"]])


def run_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    tasks = read_json(root / "plan.json")
    selected = selected_tasks_with_dependencies(tasks, set(args.stages), args.max_tasks, root=root, resume=args.resume)
    print(f"Running {len(selected)} planned tasks with {args.max_workers} worker(s)", flush=True)
    counts = {"succeeded": 0, "failed": 0, "skipped": 0, "blocked": 0}
    remaining = {task["task_id"]: task for task in selected}
    while remaining:
        ready = [task for task in remaining.values()
                 if not task.get("depends_on") or (task["depends_on"] not in remaining
                                                   and successful_output(root, task["depends_on"]) is not None)]
        if not ready:
            break
        batch = ready[: max(1, args.max_workers)]
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(execute_task, root, task, args.resume): task for task in batch}
            started = time.monotonic()
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=args.progress_interval, return_when=FIRST_COMPLETED)
                if not done:
                    print(f"Still running {len(pending)} task(s), elapsed {time.monotonic() - started:.0f}s", flush=True)
                for future in done:
                    result = future.result()
                    counts[result["status"]] = counts.get(result["status"], 0) + 1
                    print(f"{futures[future]['stage']} {result['task_id']}: {result['status']}", flush=True)
                    remaining.pop(result["task_id"], None)
        refresh_manifest(root, tasks)
    if remaining:
        print(f"{len(remaining)} tasks remain blocked by incomplete dependencies")
    print("Summary: " + ", ".join(f"{count} {name}" for name, count in counts.items()))
    consolidate(root, include_predictions=args.include_predictions)


def tune_command(args: argparse.Namespace) -> None:
    args.stages = TUNING_STAGES
    args.include_predictions = False
    run_command(args)


def evaluate_command(args: argparse.Namespace) -> None:
    args.stages = EVALUATION_STAGES
    run_command(args)


def flatten_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    return {
        "task_id": task["task_id"], "stage": task["stage"], "split_id": task.get("split_id"),
        "training_app_count": task.get("training_app_count"),
        "evaluation_method": config.get("evaluation_method"),
        "training_apps": ",".join(config.get("training_apps") or []),
        "variants": ",".join(config.get("variants") or []),
        "context_recipe": canonical_json(config.get("context", {})),
        "calibration_recipe": canonical_json(config.get("calibration", {})),
        "max_epochs": config.get("max_epochs"), "cv_seeds": canonical_json(config.get("cv_seeds", [])),
        "final_seeds": canonical_json(config.get("final_seeds", [])),
        "candidate_strategy": task.get("candidate_strategy"), "depends_on": task.get("depends_on"),
        "input_sha256": canonical_json(task.get("input_sha256", {})),
    }


def refresh_manifest(root: Path, tasks: list[dict[str, Any]]) -> None:
    rows = []
    for task in tasks:
        row = flatten_task(task)
        status = latest_status(root, task["task_id"]) or {}
        row.update({key: status.get(key) for key in ["attempt", "elapsed_seconds", "return_code", "output_dir"]})
        row["status"] = status.get("status", "pending")
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "experiment_manifest.csv", index=False)


def with_provenance(frame: pd.DataFrame, task: dict[str, Any], extra: dict[str, Any] | None = None) -> pd.DataFrame:
    result = frame.copy()
    for column, value in reversed(list({**flatten_task(task), **(extra or {})}.items())):
        result.insert(0, column if column not in result else f"planned_{column}", value)
    return result


def evaluation_directories(output: Path) -> Iterable[Path]:
    nested = output / "evaluations"
    if nested.exists():
        yield from sorted(path for path in nested.iterdir() if path.is_dir())
    elif (output / "directional_predictions.csv").exists():
        yield output


def inference_family(method: str) -> str:
    if method == "constant_1":
        return "baseline"
    if method.endswith("_direct") or method == "legacy_absolute":
        return "direct"
    if method.endswith("_calibrated"):
        return "calibrated"
    return "legacy_anchor_aggregation"


def metrics_to_long(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if "metric" in frame:
        result = frame.rename(columns={"estimate_difference": "estimate"}).copy()
        result["statistic"] = "method_minus_comparison" if "comparison_method" in frame else "estimate"
        result["scope"] = scope
        return result
    metrics = [column for column in frame.columns if column in {
        "log_mae", "log_mse", "log_rmse", "raw_mae", "raw_mape", "raw_mse", "raw_rmse",
        "median_absolute_log_error", "median_multiplicative_error", "spearman", "signed_log_bias"}]
    result = frame.melt(id_vars=[column for column in frame.columns if column not in metrics],
                        value_vars=metrics, var_name="metric", value_name="estimate")
    result["scope"] = scope
    result["statistic"] = "estimate"
    return result


def consolidate(root: Path, include_predictions: bool = True) -> None:
    tasks = read_json(root / "plan.json")
    output_root = root / "consolidated"
    output_root.mkdir(parents=True, exist_ok=True)
    for path in output_root.glob("*.csv"):
        path.unlink()
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in CONSOLIDATED_TABLES}
    failures = []
    global_selection = None
    for task in tasks:
        status = latest_status(root, task["task_id"])
        if not status or status.get("status") != "succeeded":
            if status and status.get("status") == "failed":
                failures.append({**flatten_task(task), **status})
            continue
        output = Path(status["output_dir"])
        report = read_json(output / "run_report.json")
        extra = {
            "panel_source": report["inhibitor_panel"]["panel_source"],
            "panel_ids": "|".join(report["inhibitor_panel"]["selected_ids"]),
            "training_response_rows": read_json(output / "data_validation_summary.json")["model_fitting_app_inhibitor_rows"],
        }
        if task["stage"] in TUNING_STAGES:
            global_selection = {**read_json(output / "selection.json"), "task_id": task["task_id"],
                                "selection_scope": "all application profiles and App-Inhibitor responses; no App-App outcomes"}
            for name in ["matched_fold_comparisons.csv", "ablation_summary.csv", "metrics_by_censoring.csv",
                         "auxiliary_validation_metrics.csv", "calibration_diagnostics.csv", "context_support_manifest.csv"]:
                if (output / name).exists():
                    tables[name].append(with_provenance(pd.read_csv(output / name), task, extra).assign(
                        result_scope="app_inhibitor_crossed_validation"))
        for evaluation_dir in evaluation_directories(output):
            prediction_path = evaluation_dir / "directional_predictions.csv"
            if include_predictions and prediction_path.exists():
                prediction_frames.append(with_provenance(pd.read_csv(prediction_path), task, extra))
            for name, scope in REPORT_FILES.items():
                if (evaluation_dir / name).exists():
                    metric_frames.append(with_provenance(metrics_to_long(pd.read_csv(evaluation_dir / name), scope), task, extra))
            for name in ["pair_metrics.csv", "metrics_by_censoring.csv", "auxiliary_metrics.csv",
                         "method_coverage.csv", "ablation_summary.csv"]:
                if (evaluation_dir / name).exists():
                    frame = pd.read_csv(evaluation_dir / name)
                    if name == "ablation_summary.csv":
                        frame = frame[frame["scope"].astype(str).str.startswith("app_app")]
                    tables[name].append(with_provenance(frame, task, extra).assign(result_scope="app_app"))
    if global_selection is not None:
        write_json(output_root / "global_selection.json", global_selection)
    pd.DataFrame(failures).to_csv(output_root / "failures.csv", index=False)
    for name, frames in tables.items():
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(output_root / name, index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(output_root / "predictions.csv", index=False)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        metrics["inference_family"] = metrics["method"].astype(str).map(inference_family)
        metrics.to_csv(output_root / "metrics_long.csv", index=False)
        accuracy = metrics[(metrics["scope"] == "overall_non_self") & (metrics["statistic"] == "estimate")]
        accuracy.to_csv(output_root / "accuracy_by_eval_method.csv", index=False)
        for family, group in accuracy.groupby("inference_family"):
            group.to_csv(output_root / f"accuracy_{family}.csv", index=False)
        learning = accuracy[accuracy["stage"] == "evaluate_training_size"]
        if not learning.empty:
            learning.groupby(["training_app_count", "evaluation_method", "method", "inference_family", "metric"],
                             as_index=False).agg(estimate=("estimate", "mean"), between_split_sd=("estimate", "std"),
                                                 split_replicate_count=("estimate", "count")).to_csv(
                output_root / "learning_curve_summary.csv", index=False)
    refresh_manifest(root, tasks)
    write_json(output_root / "consolidation_report.json", {
        "created_at": utc_now(),
        "successful_tasks": sum((latest_status(root, task["task_id"]) or {}).get("status") == "succeeded" for task in tasks),
        "failed_tasks": len(failures), "include_predictions": include_predictions,
        "global_selection_available": global_selection is not None,
    })


def consolidate_command(args: argparse.Namespace) -> None:
    consolidate(args.root.resolve(), include_predictions=args.include_predictions)


def status_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    tasks = read_json(root / "plan.json")
    refresh_manifest(root, tasks)
    print(pd.read_csv(root / "experiment_manifest.csv").groupby(["stage", "status"]).size().to_string())


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-predictions", action=argparse.BooleanOptionalAction, default=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write an immutable experiment plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    plan.add_argument("--jobs-csv", type=Path, default=DEFAULT_TRAINING / "jobs.csv")
    plan.add_argument("--inhibitors-csv", type=Path, default=DEFAULT_TRAINING / "inhibitors.csv")
    plan.add_argument("--job-inh-csv", type=Path, default=DEFAULT_TRAINING / "job_inh.csv")
    plan.add_argument("--pair-csv", type=Path, default=DEFAULT_EVALUATION / "pair.csv")
    plan.add_argument("--random-split-only", action="store_true")
    plan.add_argument("--variants", nargs="+", choices=sorted(VARIANT_REGISTRY))
    plan.add_argument("--frozen-panel-csv", type=Path)
    plan.add_argument("--context-max-support", type=int)
    plan.add_argument("--context-crossfit-folds", type=int)
    plan.add_argument("--calibration-lambda", type=float, nargs="+")
    plan.add_argument("--inhibitor-blocks", type=int, default=4)
    plan.set_defaults(function=plan_command)
    tune = subparsers.add_parser("tune", help="run global App-Inhibitor selection")
    add_execution_arguments(tune)
    tune.set_defaults(function=tune_command)
    evaluate = subparsers.add_parser("evaluate", help="run fixed-global evaluations")
    add_execution_arguments(evaluate)
    evaluate.set_defaults(function=evaluate_command)
    consolidate_parser = subparsers.add_parser("consolidate", help="rebuild result tables")
    consolidate_parser.add_argument("--root", type=Path, required=True)
    consolidate_parser.add_argument("--include-predictions", action=argparse.BooleanOptionalAction, default=True)
    consolidate_parser.set_defaults(function=consolidate_command)
    status = subparsers.add_parser("status", help="summarize task status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(function=status_command)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if getattr(args, "max_workers", 1) < 1:
        raise ValueError("--max-workers must be positive")
    try:
        args.function(args)
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume to continue in a new attempt.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
