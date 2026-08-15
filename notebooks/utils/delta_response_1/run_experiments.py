"""Plan, run, and consolidate the paper's focused accuracy experiments.

The runner selects one global low-rank recipe from App-Inhibitor crossed
validation, freezes it, and evaluates random_split plus balanced one_known and
zero_shot training-size conditions on the configured App-App pair data. Pair
outcomes are never available to tuning tasks.
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
from tqdm import tqdm

from metrics import metrics_by_method


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"
DEFAULT_DATA = ROOT.parent / "few_shot" / "data"
SCHEMA_VERSION = 2
MODEL_KINDS = ["low_rank", "generic", "absolute"]
METRICS = [
    "log_mae",
    "log_rmse",
    "raw_mae",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
    "spearman",
]
TUNING_STAGES = ["tune_architecture", "tune_optimizer", "tune_capacity"]
EVALUATION_STAGES = ["evaluate_random_split", "evaluate_training_size"]


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
    "standard": ExecutionProfile(
        list(range(5)),
        list(range(5)),
        80,
        10,
        4,
        128,
        1000,
        10,
        list(range(2, 9)),
    ),
}


DEFAULT_MODEL = {
    "feature_set": "base",
    "hidden_dim": 8,
    "embedding_dim": 4,
    "rank": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_planned_inputs(task: dict[str, Any]) -> None:
    actual = {
        name: sha256_file(Path(path)) for name, path in task["inputs"].items()
    }
    planned = task.get("input_sha256", {})
    if actual != planned:
        changed = sorted(set(actual) | set(planned))
        changed = [name for name in changed if actual.get(name) != planned.get(name)]
        raise RuntimeError(
            "Experiment inputs changed after planning; create a new immutable plan. "
            f"Changed inputs: {changed}"
        )


def architecture_candidates() -> list[dict[str, Any]]:
    return [
        {**DEFAULT_MODEL, "feature_set": feature_set, "rank": rank}
        for feature_set in ["base", "augmented"]
        for rank in [1, 2, 4]
    ]


def candidates_by_kind(low_rank: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "low_rank": low_rank,
        "generic": [DEFAULT_MODEL.copy()],
        "absolute": [DEFAULT_MODEL.copy()],
    }


def base_run_config(profile: ExecutionProfile) -> dict[str, Any]:
    return {
        "inhibitor_blocks": 4,
        "clustering_algorithm": "kmeans",
        "clustering_seed": 1701,
        "cv_seeds": profile.cv_seeds,
        "final_seeds": profile.final_seeds,
        "model_candidates_by_kind": candidates_by_kind([DEFAULT_MODEL.copy()]),
        "max_epochs": profile.max_epochs,
        "patience": profile.patience,
        "batches_per_epoch": profile.batches_per_epoch,
        "batch_size": profile.batch_size,
        "temperatures": [0.1, 0.3, 1.0, 3.0, 10.0],
        "ood_quantiles": [0.9, 0.95, 0.99],
        "ood_fallbacks": ["uniform", "median"],
        "bootstrap_samples": profile.bootstrap_samples,
        "bootstrap_seed": 923,
        "device": "auto",
        "inference_anchor_counts": ["all"],
        "inference_anchor_seed": 1701,
        "selection_mode": "crossed_cv",
    }


def transformed_profiles(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["mpi_time"].to_numpy(float)),
            frame["comm_frac"].to_numpy(float),
            np.log1p(frame["total_msgs"].to_numpy(float)),
            np.log1p(frame["total_bytes"].to_numpy(float)),
        ]
    )


def maximin_order(jobs: pd.DataFrame, start: int) -> list[str]:
    values = transformed_profiles(jobs)
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-8)
    selected = [start]
    remaining = set(range(len(jobs))) - {start}
    while remaining:
        next_index = max(
            remaining,
            key=lambda index: (
                min(np.linalg.norm(values[index] - values[item]) for item in selected),
                -index,
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return jobs.iloc[selected]["job_id"].astype(str).tolist()


def nested_orders(
    jobs: pd.DataFrame,
    count: int,
    seed: int,
    strategy: str = "balanced_random",
) -> list[list[str]]:
    identifiers = jobs["job_id"].astype(str).tolist()
    if strategy == "balanced_random":
        base = np.random.default_rng(seed).permutation(identifiers).tolist()
    elif strategy == "profile_diverse":
        base = maximin_order(jobs, start=seed % len(jobs))
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    return [base[offset:] + base[:offset] for offset in range(count)]


def make_task(
    *,
    stage: str,
    inputs: dict[str, str],
    config: dict[str, Any],
    opens_holdout: bool,
    depends_on: str | None = None,
    candidate_strategy: str = "fixed",
    split_id: str | None = None,
    training_app_count: int | None = None,
) -> dict[str, Any]:
    task = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "data_kind": "real",
        "inputs": inputs,
        "input_sha256": {
            name: sha256_file(Path(path)) for name, path in inputs.items()
        },
        "config": config,
        "opens_holdout": opens_holdout,
        "split_id": split_id,
        "split_strategy": "balanced_random" if split_id else None,
        "training_app_count": training_app_count,
        "depends_on": depends_on,
        "candidate_strategy": candidate_strategy,
    }
    task["task_id"] = stable_id("run", task)
    return task


def build_plan(
    root: Path,
    profile_name: str,
    jobs_csv: Path,
    inhibitors_csv: Path,
    job_inh_csv: Path,
    pair_csv: Path,
) -> list[dict[str, Any]]:
    profile = PROFILES[profile_name]
    paths = {
        "jobs": jobs_csv.resolve(),
        "inhibitors": inhibitors_csv.resolve(),
        "job_inh": job_inh_csv.resolve(),
        "pair": pair_csv.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment inputs do not exist: {missing}")
    jobs = pd.read_csv(paths["jobs"])
    if "job_id" not in jobs or jobs["job_id"].duplicated().any():
        raise ValueError("jobs.csv must contain unique job_id values")
    if profile_name == "standard" and len(jobs) != 10:
        raise ValueError(
            "The preregistered standard design requires exactly 10 applications"
        )

    training_inputs = {
        name: str(path) for name, path in paths.items() if name != "pair"
    }
    evaluation_inputs = {name: str(path) for name, path in paths.items()}
    tasks: list[dict[str, Any]] = []

    architecture_config = base_run_config(profile)
    architecture_config.update(
        {
            "evaluation_method": "random_split",
            "training_apps": None,
            "model_candidates_by_kind": candidates_by_kind(
                architecture_candidates()
            ),
        }
    )
    architecture = make_task(
        stage="tune_architecture",
        inputs=training_inputs,
        config=architecture_config,
        opens_holdout=False,
    )
    tasks.append(architecture)

    optimizer_config = base_run_config(profile)
    optimizer_config.update({"evaluation_method": "random_split", "training_apps": None})
    optimizer = make_task(
        stage="tune_optimizer",
        inputs=training_inputs,
        config=optimizer_config,
        opens_holdout=False,
        depends_on=architecture["task_id"],
        candidate_strategy="optimizer_from_dependency",
    )
    tasks.append(optimizer)

    capacity_config = base_run_config(profile)
    capacity_config.update({"evaluation_method": "random_split", "training_apps": None})
    capacity = make_task(
        stage="tune_capacity",
        inputs=training_inputs,
        config=capacity_config,
        opens_holdout=False,
        depends_on=optimizer["task_id"],
        candidate_strategy="capacity_from_dependency",
    )
    tasks.append(capacity)

    random_config = base_run_config(profile)
    random_config.update({"evaluation_method": "random_split", "training_apps": None})
    tasks.append(
        make_task(
            stage="evaluate_random_split",
            inputs=evaluation_inputs,
            config=random_config,
            opens_holdout=True,
            depends_on=capacity["task_id"],
            candidate_strategy="fixed_global_from_dependency",
            training_app_count=len(jobs),
        )
    )

    membership_rows: list[dict[str, Any]] = []
    orders = nested_orders(
        jobs,
        count=min(profile.split_replicates, len(jobs)),
        seed=811,
    )
    for split_index, order in enumerate(orders):
        split_id = f"balanced_s{split_index:02d}"
        for training_size in profile.training_sizes:
            if training_size >= len(jobs):
                raise ValueError("Restricted training sizes must leave an unknown application")
            training_apps = order[:training_size]
            for addition_rank, app_id in enumerate(order, start=1):
                membership_rows.append(
                    {
                        "split_id": split_id,
                        "split_strategy": "balanced_random",
                        "training_app_count": training_size,
                        "app_id": app_id,
                        "addition_rank": addition_rank,
                        "in_training": addition_rank <= training_size,
                    }
                )
            config = base_run_config(profile)
            config.update(
                {
                    "evaluation_method": "both",
                    "training_apps": training_apps,
                }
            )
            tasks.append(
                make_task(
                    stage="evaluate_training_size",
                    inputs=evaluation_inputs,
                    config=config,
                    opens_holdout=True,
                    depends_on=capacity["task_id"],
                    candidate_strategy="fixed_global_from_dependency",
                    split_id=split_id,
                    training_app_count=training_size,
                )
            )

    pd.DataFrame(membership_rows).to_csv(root / "subset_membership.csv", index=False)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("Experiment plan produced duplicate task IDs")
    return tasks


def plan_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment plan: {root / 'plan.json'}"
        )
    tasks = build_plan(
        root,
        args.profile,
        args.jobs_csv,
        args.inhibitors_csv,
        args.job_inh_csv,
        args.pair_csv,
    )
    write_json(
        root / "experiment_spec.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "profile": args.profile,
            "profile_settings": asdict(PROFILES[args.profile]),
            "task_count": len(tasks),
            "tuning_task_count": sum(task["stage"] in TUNING_STAGES for task in tasks),
            "evaluation_task_count": sum(
                task["stage"] in EVALUATION_STAGES for task in tasks
            ),
            "holdout_policy": (
                "The configured pair data is sealed during global App-Inhibitor "
                "tuning and opened only by fixed-selection evaluation tasks."
            ),
            "global_hyperparameter_selection_scope": (
                "Transductive: all evaluation applications' profiles and App-Inhibitor "
                "responses may select the global recipe; pair outcomes never do."
            ),
        },
    )
    write_json(
        root / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "code_sha256": {
                path.name: sha256_file(path)
                for path in [
                    PIPELINE,
                    Path(__file__).resolve(),
                    ROOT / "data.py",
                    ROOT / "inference.py",
                    ROOT / "metrics.py",
                    ROOT / "model.py",
                    ROOT / "training.py",
                ]
            },
        },
    )
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
    if not report.exists() or not success_path.exists():
        return None
    if read_json(report).get("status") != "complete":
        return None
    success = read_json(success_path)
    if success.get("run_report_sha256") != sha256_file(report):
        return None
    return output


def dependency_selection(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    dependency = task.get("depends_on")
    if not dependency:
        raise RuntimeError(f"Task {task['task_id']} has no selection dependency")
    output = successful_output(root, dependency)
    if output is None:
        raise RuntimeError(f"Dependency {dependency} has not completed successfully")
    return read_json(output / "selection.json")


def accuracy_frozen_selection(selection: dict[str, Any]) -> dict[str, Any]:
    frozen = json.loads(json.dumps(selection))
    aggregation = frozen["aggregations"]["low_rank"]
    accuracy_best = aggregation.get("accuracy_best")
    if accuracy_best is None:
        raise ValueError("Global tuning selection is missing accuracy_best aggregation")
    method = accuracy_best["method"]
    aggregation["primary_method"] = method
    if method == "kernel":
        aggregation["kernel"]["temperature"] = accuracy_best["temperature"]
    elif method == "ood_kernel":
        aggregation["ood_kernel"].update(
            {
                "temperature": accuracy_best["temperature"],
                "quantile": accuracy_best["quantile"],
                "fallback": accuracy_best["fallback"],
            }
        )
    aggregation["focused_runner_selection_rule"] = "minimum_grouped_cv_log_mae"
    return frozen


def execution_fingerprint(
    root: Path,
    task: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    verify_planned_inputs(task)
    dependency = task.get("depends_on")
    dependency_hash = None
    if dependency:
        dependency_output = successful_output(root, dependency)
        if dependency_output is None:
            raise RuntimeError(f"Dependency {dependency} has not completed successfully")
        dependency_hash = sha256_file(dependency_output / "selection.json")
    code_files = [
        PIPELINE,
        Path(__file__).resolve(),
        ROOT / "data.py",
        ROOT / "inference.py",
        ROOT / "metrics.py",
        ROOT / "model.py",
        ROOT / "training.py",
    ]
    return {
        "inputs": {
            name: sha256_file(Path(path)) for name, path in task["inputs"].items()
        },
        "code": {path.name: sha256_file(path) for path in code_files},
        "resolved_config": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "dependency_selection": dependency_hash,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def resolve_task_config(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(task["config"]))
    strategy = task["candidate_strategy"]
    if strategy == "fixed":
        return config
    selection = dependency_selection(root, task)
    selected = selection["model_configs"]
    if strategy == "optimizer_from_dependency":
        low_rank = selected["low_rank"]
        config["model_candidates_by_kind"] = candidates_by_kind(
            [
                {
                    **low_rank,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                }
                for learning_rate in [3e-4, 1e-3, 3e-3]
                for weight_decay in [0.0, 1e-4, 1e-3]
            ]
        )
    elif strategy == "capacity_from_dependency":
        low_rank = selected["low_rank"]
        config["model_candidates_by_kind"] = candidates_by_kind(
            [
                {**low_rank, "hidden_dim": hidden, "embedding_dim": embedding}
                for hidden, embedding in [(4, 2), (8, 4), (16, 8)]
            ]
        )
    elif strategy == "fixed_global_from_dependency":
        selection = accuracy_frozen_selection(selection)
        selected = selection["model_configs"]
        selection_path = successful_output(root, task["depends_on"]) / "selection.json"
        config["selection_mode"] = "fixed_global"
        config["model_candidates_by_kind"] = {
            kind: [selected[kind]] for kind in MODEL_KINDS
        }
        config["fixed_final_epochs_by_kind"] = selection["final_epochs"]
        config["fixed_aggregations_by_kind"] = selection["aggregations"]
        config["fixed_selection_source"] = {
            "task_id": task["depends_on"],
            "selection_sha256": sha256_file(selection_path),
            "selection_scope": (
                "all evaluation application profiles and App-Inhibitor responses; "
                "no App-App outcomes"
            ),
            "model_selection": selection["model_selection"],
        }
    else:
        raise ValueError(f"Unknown candidate strategy: {strategy}")
    return config


def execute_task(root: Path, task: dict[str, Any], resume: bool) -> dict[str, Any]:
    dependency = task.get("depends_on")
    if dependency and successful_output(root, dependency) is None:
        return {"task_id": task["task_id"], "status": "blocked"}
    config = resolve_task_config(root, task)
    fingerprint = execution_fingerprint(root, task, config)
    existing = successful_output(root, task["task_id"])
    success_path = task_root(root, task["task_id"]) / "_SUCCESS.json"
    if resume and existing is not None and success_path.exists():
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

    command = [
        sys.executable,
        "-u",
        str(PIPELINE),
        "--jobs-csv",
        task["inputs"]["jobs"],
        "--inhibitors-csv",
        task["inputs"]["inhibitors"],
        "--job-inh-csv",
        task["inputs"]["job_inh"],
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if task["opens_holdout"]:
        command.extend(["--pair-csv", task["inputs"]["pair"]])
    else:
        command.append("--skip-holdout")
    write_json(attempt_dir / "command.json", {"argv": command})
    status = {
        "task_id": task["task_id"],
        "status": "running",
        "attempt": attempt,
        "started_at": utc_now(),
        "output_dir": str(output_dir.resolve()),
        "pid": os.getpid(),
        "hostname": platform.node(),
    }
    write_json(directory / "status.json", status)
    started = time.time()
    with (attempt_dir / "stdout.log").open("w") as stdout, (
        attempt_dir / "stderr.log"
    ).open("w") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    status.update(
        {
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "finished_at": utc_now(),
            "elapsed_seconds": time.time() - started,
        }
    )
    if completed.returncode == 0:
        report = output_dir / "run_report.json"
        if not report.exists() or read_json(report).get("status") != "complete":
            status["status"] = "failed"
            status["error"] = "Pipeline did not write a complete run_report.json"
    write_json(directory / "status.json", status)
    if status["status"] == "succeeded":
        write_json(
            directory / "_SUCCESS.json",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "resolved_config_sha256": sha256_file(config_path),
                "run_report_sha256": sha256_file(output_dir / "run_report.json"),
                "execution_fingerprint": fingerprint,
            },
        )
    return status


def latest_task_log_line(root: Path, task_id: str) -> str | None:
    status = latest_status(root, task_id)
    if not status or not status.get("output_dir"):
        return None
    stdout_path = Path(status["output_dir"]).parent / "stdout.log"
    if not stdout_path.exists():
        return None
    lines = [line.strip() for line in stdout_path.read_text().splitlines() if line.strip()]
    return lines[-1] if lines else None


def selected_tasks_with_dependencies(
    tasks: list[dict[str, Any]],
    stages: set[str],
    max_tasks: int | None,
    root: Path | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    selected = [task for task in tasks if task["stage"] in stages]
    if max_tasks is not None:
        if resume and root is not None:
            selected = [
                task
                for task in selected
                if successful_output(root, task["task_id"]) is None
            ]
        selected = selected[:max_tasks]
    by_id = {task["task_id"]: task for task in tasks}
    selected_ids = {task["task_id"] for task in selected}
    dependencies = [task.get("depends_on") for task in selected]
    while dependencies:
        dependency = dependencies.pop()
        if dependency and dependency not in selected_ids:
            selected_ids.add(dependency)
            selected.append(by_id[dependency])
            dependencies.append(by_id[dependency].get("depends_on"))
    plan_order = {task["task_id"]: index for index, task in enumerate(tasks)}
    return sorted(selected, key=lambda task: plan_order[task["task_id"]])


def run_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    tasks = read_json(root / "plan.json")
    selected = selected_tasks_with_dependencies(
        tasks,
        set(args.stages),
        args.max_tasks,
        root=root,
        resume=args.resume,
    )
    total_tasks = len(selected)
    completed_tasks = failed_tasks = skipped_tasks = 0
    tqdm.write(
        f"Running {total_tasks} planned tasks with {args.max_workers} worker(s); "
        f"progress heartbeat every {args.progress_interval:g} seconds"
    )
    remaining = {task["task_id"]: task for task in selected}
    while remaining:
        ready = [
            task
            for task in remaining.values()
            if not task.get("depends_on")
            or (
                task["depends_on"] not in remaining
                and successful_output(root, task["depends_on"]) is not None
            )
        ]
        if not ready:
            break
        batch = ready[: max(1, args.max_workers)]
        for task in batch:
            tqdm.write(
                f"Starting {task['task_id']} ({task['stage']}); "
                f"logs: {task_root(root, task['task_id']) / 'attempts'}"
            )
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(execute_task, root, task, args.resume): task
                for task in batch
            }
            started = {future: time.monotonic() for future in futures}
            pending = set(futures)
            while pending:
                done, pending = wait(
                    pending,
                    timeout=args.progress_interval,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    for future in sorted(
                        pending, key=lambda item: futures[item]["task_id"]
                    ):
                        task = futures[future]
                        elapsed = time.monotonic() - started[future]
                        latest = latest_task_log_line(root, task["task_id"])
                        detail = f"; latest log: {latest}" if latest else ""
                        tqdm.write(
                            f"Still running {task['task_id']} ({task['stage']}), "
                            f"elapsed {elapsed:.0f}s{detail}"
                        )
                    continue
                for future in done:
                    result = future.result()
                    status = result["status"]
                    completed_tasks += status == "succeeded"
                    failed_tasks += status == "failed"
                    skipped_tasks += status == "skipped"
                    task = remaining[result["task_id"]]
                    tqdm.write(
                        f"[{completed_tasks + failed_tasks + skipped_tasks}/{total_tasks}] "
                        f"{task['stage']}: {status}"
                    )
                    remaining.pop(result["task_id"], None)
        refresh_manifest(root, tasks)
    if remaining:
        tqdm.write(f"{len(remaining)} tasks remain blocked by incomplete dependencies")
    tqdm.write(
        f"Summary: {completed_tasks} succeeded, {failed_tasks} failed, "
        f"{skipped_tasks} skipped"
    )
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
        "task_id": task["task_id"],
        "stage": task["stage"],
        "split_id": task.get("split_id"),
        "split_strategy": task.get("split_strategy"),
        "training_app_count": task.get("training_app_count"),
        "evaluation_method": config.get("evaluation_method"),
        "training_apps": ",".join(config.get("training_apps") or []),
        "inhibitor_blocks": config.get("inhibitor_blocks"),
        "clustering_seed": config.get("clustering_seed"),
        "cv_seeds": canonical_json(config.get("cv_seeds", [])),
        "final_seeds": canonical_json(config.get("final_seeds", [])),
        "max_epochs": config.get("max_epochs"),
        "candidate_strategy": task.get("candidate_strategy"),
        "depends_on": task.get("depends_on"),
        "input_sha256": canonical_json(task.get("input_sha256", {})),
    }


def refresh_manifest(root: Path, tasks: list[dict[str, Any]]) -> None:
    rows = []
    for task in tasks:
        row = flatten_task(task)
        status = latest_status(root, task["task_id"])
        row.update(
            {
                "status": status.get("status", "pending") if status else "pending",
                "attempt": status.get("attempt") if status else None,
                "elapsed_seconds": status.get("elapsed_seconds") if status else None,
                "return_code": status.get("return_code") if status else None,
                "output_dir": status.get("output_dir") if status else None,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "experiment_manifest.csv", index=False)


def with_provenance(
    frame: pd.DataFrame,
    task: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    provenance = {**flatten_task(task), **(extra or {})}
    for column, value in reversed(list(provenance.items())):
        target = column if column not in result else f"planned_{column}"
        result.insert(0, target, value)
    return result


def evaluation_directories(output: Path) -> Iterable[Path]:
    nested = output / "evaluations"
    if nested.exists():
        yield from sorted(path for path in nested.iterdir() if path.is_dir())
    elif (output / "directional_predictions.csv").exists():
        yield output


def metrics_to_long(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if "metric" in frame:
        result = frame.copy()
        if "estimate_difference" in result:
            result = result.rename(columns={"estimate_difference": "estimate"})
            result["statistic"] = "method_minus_comparison"
        else:
            result["statistic"] = "estimate"
        result["scope"] = scope
        return result
    available = [metric for metric in METRICS if metric in frame]
    identifiers = [column for column in frame.columns if column not in available]
    if not available:
        result = frame.copy()
        result["scope"] = scope
        return result
    result = frame.melt(
        id_vars=identifiers,
        value_vars=available,
        var_name="metric",
        value_name="estimate",
    )
    result["scope"] = scope
    result["statistic"] = "estimate"
    return result


def low_rank_primary_method(selection: dict[str, Any]) -> str:
    primary = selection["aggregations"]["low_rank"]["primary_method"]
    return {
        "uniform": "delta_uniform",
        "median": "delta_median",
        "kernel": "delta_kernel",
        "ood_kernel": "delta_ood_kernel",
    }[primary]


def consolidate(root: Path, include_predictions: bool = True) -> None:
    tasks = read_json(root / "plan.json")
    output_root = root / "consolidated"
    output_root.mkdir(parents=True, exist_ok=True)
    generated_outputs = [
        "accuracy_by_eval_method.csv",
        "accuracy_by_training_size.csv",
        "architecture_search.csv",
        "candidates.csv",
        "failures.csv",
        "fold_metrics.csv",
        "global_selection.json",
        "learning_curve_summary.csv",
        "metrics_long.csv",
        "pair_eligibility.csv",
        "predictions.csv",
        "selections.csv",
    ]
    for filename in generated_outputs:
        path = output_root / filename
        if path.exists():
            path.unlink()
    fold_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    eligibility_frames: list[pd.DataFrame] = []
    native_accuracy: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    random_predictions: pd.DataFrame | None = None
    final_tuning_selection: dict[str, Any] | None = None

    report_files = {
        "metrics_repeated_self.csv": "overall_repeated_self",
        "metrics_self_averaged.csv": "overall_self_averaged",
        "metrics_non_self.csv": "overall_non_self",
        "metrics_per_victim.csv": "per_victim",
        "metrics_macro_victim.csv": "macro_victim",
        "metrics_non_self_per_victim.csv": "non_self_per_victim",
        "metrics_non_self_macro_victim.csv": "non_self_macro_victim",
        "metrics_by_seed.csv": "by_seed",
        "cluster_bootstrap_confidence_intervals.csv": "cluster_bootstrap",
        "paired_method_differences.csv": "paired_difference",
        "cluster_bootstrap_non_self.csv": "non_self_cluster_bootstrap",
        "paired_method_differences_non_self.csv": "non_self_paired_difference",
    }

    for task in tasks:
        status = latest_status(root, task["task_id"])
        if not status or status.get("status") != "succeeded":
            if status and status.get("status") == "failed":
                failures.append({**flatten_task(task), **status})
            continue
        output = Path(status["output_dir"])
        data_summary = read_json(output / "data_validation_summary.json")
        extra = {
            "training_response_rows": data_summary["model_fitting_app_inhibitor_rows"],
            "global_selection_task_id": task.get("depends_on")
            if task["stage"] in EVALUATION_STAGES
            else None,
        }
        resolved_config = read_json(output.parent / "resolved_config.json")
        selection = read_json(output / "selection.json")

        if task["stage"] in TUNING_STAGES:
            candidates = resolved_config["model_candidates_by_kind"]
            for config_id, candidate in enumerate(candidates["low_rank"]):
                candidate_rows.append(
                    {
                        **flatten_task(task),
                        "model_kind": "low_rank",
                        "config_id": config_id,
                        "candidate_id": stable_id("candidate", candidate, 12),
                        "selected": config_id
                        == selection["model_selection"]["low_rank"]["selected_config_id"],
                        **candidate,
                    }
                )
            fold_path = output / "crossed_validation_folds.csv"
            if fold_path.exists():
                folds = pd.read_csv(fold_path)
                low_rank_folds = folds[folds["model_kind"] == "low_rank"].copy()
                for field in DEFAULT_MODEL:
                    low_rank_folds[field] = [
                        candidates["low_rank"][int(config_id)][field]
                        for config_id in low_rank_folds["config_id"]
                    ]
                low_rank_folds["candidate_id"] = [
                    stable_id("candidate", candidates["low_rank"][int(config_id)], 12)
                    for config_id in low_rank_folds["config_id"]
                ]
                fold_frames.append(with_provenance(low_rank_folds, task, extra))
            selection_rows.append(
                {
                    **flatten_task(task),
                    **selection["model_configs"]["low_rank"],
                    **{
                        f"selection_{key}": value
                        for key, value in selection["model_selection"]["low_rank"].items()
                    },
                    "primary_aggregation": selection["aggregations"]["low_rank"][
                        "primary_method"
                    ],
                }
            )
            if task["stage"] == "tune_capacity":
                final_tuning_selection = accuracy_frozen_selection(selection)

        for evaluation_dir in evaluation_directories(output):
            prediction_path = evaluation_dir / "directional_predictions.csv"
            if prediction_path.exists():
                predictions = pd.read_csv(prediction_path)
                if task["stage"] == "evaluate_random_split":
                    random_predictions = predictions
                if include_predictions:
                    prediction_frames.append(with_provenance(predictions, task, extra))
            eligibility_path = evaluation_dir / "evaluation_pair_manifest.csv"
            if eligibility_path.exists():
                eligibility_frames.append(
                    with_provenance(pd.read_csv(eligibility_path), task, extra)
                )
            for filename, scope in report_files.items():
                path = evaluation_dir / filename
                if not path.exists():
                    continue
                long = metrics_to_long(pd.read_csv(path), scope)
                metric_frames.append(with_provenance(long, task, extra))
                if filename == "metrics_non_self.csv":
                    long["comparison_scope"] = "protocol_native"
                    long["fit_evaluation_method"] = long["evaluation_method"]
                    native_accuracy.append(with_provenance(long, task, extra))

    if final_tuning_selection is not None:
        global_selection = {
            **final_tuning_selection,
            "global_selection_id": stable_id(
                "selection",
                {
                    "model_configs": final_tuning_selection["model_configs"],
                    "aggregations": final_tuning_selection["aggregations"],
                    "final_epochs": final_tuning_selection["final_epochs"],
                },
            ),
            "selection_scope": (
                "all evaluation application profiles and App-Inhibitor responses; "
                "no App-App outcomes"
            ),
            "protocol_note": (
                "Weights and model scalers are refit per condition, but global "
                "hyperparameter selection is transductive for one_known and zero_shot."
            ),
        }
        write_json(output_root / "global_selection.json", global_selection)
        primary_method = low_rank_primary_method(final_tuning_selection)
    else:
        primary_method = ""

    matched_accuracy: list[pd.DataFrame] = []
    if random_predictions is not None:
        for task in tasks:
            if task["stage"] != "evaluate_training_size":
                continue
            output = successful_output(root, task["task_id"])
            if output is None:
                continue
            data_summary = read_json(output / "data_validation_summary.json")
            extra = {
                "training_response_rows": data_summary[
                    "model_fitting_app_inhibitor_rows"
                ],
                "global_selection_task_id": task["depends_on"],
            }
            for evaluation_dir in evaluation_directories(output):
                method = evaluation_dir.name
                manifest = pd.read_csv(evaluation_dir / "evaluation_pair_manifest.csv")
                included = manifest[manifest["included"]].copy()
                included = included[included["jobA_id"] != included["jobB_id"]]
                matched = random_predictions[
                    random_predictions["pair_row_id"].isin(included["pair_row_id"])
                ].copy()
                if matched.empty:
                    continue
                metrics = metrics_by_method(matched)
                metrics["evaluation_method"] = method
                long = metrics_to_long(metrics, "overall_non_self")
                long["comparison_scope"] = "matched_random_split"
                long["fit_evaluation_method"] = "random_split"
                matched_accuracy.append(with_provenance(long, task, extra))

    fold_metrics = (
        pd.concat(fold_frames, ignore_index=True, sort=False)
        if fold_frames
        else pd.DataFrame()
    )
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty and not fold_metrics.empty:
        scores = (
            fold_metrics.groupby(["task_id", "config_id"], as_index=False)[
                "best_validation"
            ]
            .mean()
            .rename(columns={"best_validation": "mean_cv_log_mae"})
        )
        architecture_search = candidates.merge(scores, on=["task_id", "config_id"])
    else:
        architecture_search = candidates
    architecture_search.to_csv(output_root / "architecture_search.csv", index=False)
    candidates.to_csv(output_root / "candidates.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_root / "selections.csv", index=False)
    pd.DataFrame(failures).to_csv(output_root / "failures.csv", index=False)
    if not fold_metrics.empty:
        fold_metrics.to_csv(output_root / "fold_metrics.csv", index=False)

    if metric_frames:
        pd.concat(metric_frames, ignore_index=True, sort=False).to_csv(
            output_root / "metrics_long.csv", index=False
        )
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
            output_root / "predictions.csv", index=False
        )
    if eligibility_frames:
        pd.concat(eligibility_frames, ignore_index=True, sort=False).to_csv(
            output_root / "pair_eligibility.csv", index=False
        )

    accuracy_frames = [*native_accuracy, *matched_accuracy]
    if accuracy_frames:
        accuracy = pd.concat(accuracy_frames, ignore_index=True, sort=False)
        accuracy["is_global_primary_low_rank"] = accuracy["method"].eq(primary_method)
        accuracy.to_csv(output_root / "accuracy_by_eval_method.csv", index=False)
        learning = accuracy[
            (accuracy["stage"] == "evaluate_training_size")
            & (accuracy["comparison_scope"] == "protocol_native")
        ].copy()
        learning.to_csv(output_root / "accuracy_by_training_size.csv", index=False)
        group_columns = [
            "training_app_count",
            "evaluation_method",
            "method",
            "metric",
        ]
        summary = learning.groupby(group_columns, as_index=False).agg(
            estimate=("estimate", "mean"),
            between_split_sd=("estimate", "std"),
            split_replicate_count=("estimate", "count"),
        )
        summary["low_support_zero_shot"] = (
            (summary["evaluation_method"] == "zero_shot")
            & (summary["training_app_count"] == 8)
        )
        summary.to_csv(output_root / "learning_curve_summary.csv", index=False)

    refresh_manifest(root, tasks)
    write_json(
        output_root / "consolidation_report.json",
        {
            "created_at": utc_now(),
            "successful_tasks": sum(
                latest_status(root, task["task_id"]) is not None
                and latest_status(root, task["task_id"]).get("status") == "succeeded"
                for task in tasks
            ),
            "include_predictions": include_predictions,
            "global_selection_available": final_tuning_selection is not None,
        },
    )


def consolidate_command(args: argparse.Namespace) -> None:
    consolidate(args.root.resolve(), include_predictions=args.include_predictions)


def status_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    tasks = read_json(root / "plan.json")
    refresh_manifest(root, tasks)
    manifest = pd.read_csv(root / "experiment_manifest.csv")
    print(manifest.groupby(["stage", "status"]).size().to_string())


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="seconds between progress messages while tasks are running",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write an immutable experiment plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    plan.add_argument("--jobs-csv", type=Path, default=DEFAULT_DATA / "jobs.csv")
    plan.add_argument(
        "--inhibitors-csv", type=Path, default=DEFAULT_DATA / "inhibitors.csv"
    )
    plan.add_argument("--job-inh-csv", type=Path, default=DEFAULT_DATA / "job_inh.csv")
    plan.add_argument(
        "--pair-csv",
        type=Path,
        default=DEFAULT_DATA / "pair.csv",
        help="App-App evaluation data",
    )
    plan.set_defaults(function=plan_command)

    tune = subparsers.add_parser("tune", help="select the global low-rank recipe")
    add_execution_arguments(tune)
    tune.set_defaults(function=tune_command)

    evaluate = subparsers.add_parser(
        "evaluate", help="run fixed random-split and training-size evaluations"
    )
    add_execution_arguments(evaluate)
    evaluate.set_defaults(function=evaluate_command)

    consolidate_parser = subparsers.add_parser(
        "consolidate", help="rebuild paper-facing result tables"
    )
    consolidate_parser.add_argument("--root", type=Path, required=True)
    consolidate_parser.add_argument(
        "--include-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    consolidate_parser.set_defaults(function=consolidate_command)

    status = subparsers.add_parser("status", help="summarize task status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(function=status_command)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if getattr(args, "max_workers", 1) < 1:
        raise ValueError("--max-workers must be positive")
    if getattr(args, "progress_interval", 1.0) <= 0:
        raise ValueError("--progress-interval must be positive")
    try:
        args.function(args)
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume to continue in a new attempt.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
