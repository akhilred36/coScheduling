"""Plan, execute, and consolidate delta-response experiments.

Real-data tasks never open pair.csv. End-to-end evaluation is restricted to
generated synthetic pairs unless a future experiment is implemented under a
separate, explicitly authorized holdout protocol.
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


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"
DEFAULT_DATA = ROOT.parent / "few_shot" / "data"
SCHEMA_VERSION = 1
METRICS = [
    "log_mae",
    "log_rmse",
    "raw_mae",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
    "spearman",
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
    synthetic_seeds: list[int]
    split_replicates: int
    training_sizes: list[int]


PROFILES = {
    "smoke": ExecutionProfile(
        [0], [0], 2, 1, 1, 32, 20, [100], 1, [2]
    ),
    "standard": ExecutionProfile(
        list(range(5)),
        list(range(5)),
        80,
        10,
        4,
        128,
        1000,
        list(range(100, 105)),
        10,
        list(range(2, 9)),
    ),
    "extended": ExecutionProfile(
        list(range(10)),
        list(range(10)),
        160,
        20,
        4,
        128,
        5000,
        list(range(100, 105)),
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


BASE_SCENARIO = {
    "n_jobs": 10,
    "n_inhibitors": 236,
    "true_rank": 2,
    "log_noise_sd": 0.05,
    "censor_fraction": 0.25,
    "missing_edge_fraction": 0.0,
    "replicates_per_edge": 1,
    "outlier_fraction": 0.0,
    "target_ood_shift": 0.0,
    "transfer_mismatch": 0.0,
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


def architecture_candidates() -> list[dict[str, Any]]:
    return [
        {
            **DEFAULT_MODEL,
            "feature_set": feature_set,
            "rank": rank,
        }
        for feature_set in ["base", "augmented"]
        for rank in [1, 2, 4]
    ]


def base_run_config(profile: ExecutionProfile) -> dict[str, Any]:
    return {
        "inhibitor_blocks": 4,
        "clustering_algorithm": "kmeans",
        "clustering_seed": 1701,
        "cv_seeds": profile.cv_seeds,
        "final_seeds": profile.final_seeds,
        "model_candidates": [DEFAULT_MODEL],
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
        "inference_anchor_counts": [4, 16, 64, "all"],
        "inference_anchor_seed": 1701,
    }


def stress_scenarios(profile_name: str) -> list[tuple[str, dict[str, Any]]]:
    if profile_name == "smoke":
        variants = [
            ("baseline", {}),
            ("censored_50", {"censor_fraction": 0.5}),
            ("ood_2", {"target_ood_shift": 2.0}),
        ]
    else:
        variants = [("baseline", {})]
        dimensions = {
            "jobs": ("n_jobs", [5, 20]),
            "inhibitors": ("n_inhibitors", [60, 120]),
            "rank": ("true_rank", [1, 4, 8]),
            "noise": ("log_noise_sd", [0.0, 0.15, 0.30]),
            "censored": ("censor_fraction", [0.0, 0.50, 0.75]),
            "missing": ("missing_edge_fraction", [0.10, 0.30, 0.50]),
            "replicates": ("replicates_per_edge", [3, 5]),
            "outliers": ("outlier_fraction", [0.05]),
            "ood": ("target_ood_shift", [1.0, 2.0, 4.0]),
            "mismatch": ("transfer_mismatch", [0.25, 0.50, 1.0]),
        }
        for label, (field, values) in dimensions.items():
            for value in values:
                variants.append((f"{label}_{str(value).replace('.', 'p')}", {field: value}))
        variants.extend(
            [
                (
                    "combined_noisy_censored",
                    {"log_noise_sd": 0.30, "censor_fraction": 0.50},
                ),
                (
                    "combined_sparse_few",
                    {"n_inhibitors": 60, "missing_edge_fraction": 0.50},
                ),
                (
                    "combined_ood_mismatch",
                    {"target_ood_shift": 4.0, "transfer_mismatch": 1.0},
                ),
                (
                    "combined_high_rank",
                    {"true_rank": 8, "log_noise_sd": 0.15},
                ),
            ]
        )
    return [(name, {**BASE_SCENARIO, **changes}) for name, changes in variants]


def transformed_profiles(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["mpi_time"].to_numpy(float)),
            frame["comm_frac"].to_numpy(float),
            np.log1p(frame["total_msgs"].to_numpy(float)),
            np.log1p(frame["total_bytes"].to_numpy(float)),
        ]
    )


def physical_profiles(
    count: int,
    id_column: str,
    prefix: str,
    rng: np.random.Generator,
    *,
    shift: float = 0.0,
) -> pd.DataFrame:
    latent = rng.normal(size=(count, 4))
    latent[:, 0] += 0.35 * shift
    latent[:, 1] += shift
    latent[:, 2] += 0.55 * shift
    latent[:, 3] += 0.40 * shift
    mpi_time = np.exp(3.5 + 0.8 * latent[:, 0])
    comm_frac = 1.0 / (1.0 + np.exp(-(latent[:, 1] - 0.8)))
    total_msgs = np.exp(15.0 + 1.6 * latent[:, 2])
    mean_message_size = np.exp(6.5 + 1.0 * latent[:, 3])
    total_bytes = total_msgs * mean_message_size
    return pd.DataFrame(
        {
            id_column: [f"{prefix}_{index:03d}" for index in range(count)],
            "mpi_time": mpi_time,
            "comm_frac": comm_frac,
            "total_msgs": total_msgs,
            "total_bytes": total_bytes,
        }
    )


def standardized_pair(
    jobs: pd.DataFrame, inhibitors: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    inhibitor_values = transformed_profiles(inhibitors)
    mean = inhibitor_values.mean(axis=0)
    scale = inhibitor_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (
        (transformed_profiles(jobs) - mean) / scale,
        (inhibitor_values - mean) / scale,
    )


def generate_synthetic_data(
    scenario: dict[str, Any], seed: int, output_dir: Path
) -> dict[str, Any]:
    requested = {
        **scenario,
        "seed": seed,
        "schema_version": SCHEMA_VERSION,
        "generator_version": 2,
    }
    summary_path = output_dir / "scenario_summary.json"
    if summary_path.exists():
        cached = read_json(summary_path)
        hashes = cached.get("file_sha256", {})
        cache_valid = cached.get("requested") == requested and all(
            (output_dir / name).exists()
            and sha256_file(output_dir / name) == expected
            for name, expected in hashes.items()
        )
        if cache_valid and hashes:
            return cached

    output_dir.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(seed).spawn(6)
    profile_rng, truth_rng, response_rng, missing_rng, replicate_rng, pair_rng = [
        np.random.default_rng(stream) for stream in streams
    ]
    jobs = physical_profiles(
        int(scenario["n_jobs"]),
        "job_id",
        "job",
        profile_rng,
        shift=float(scenario["target_ood_shift"]),
    )
    inhibitors = physical_profiles(
        int(scenario["n_inhibitors"]), "inhib_id", "inh", profile_rng
    )
    inhibitors["msg_size"] = inhibitors["total_bytes"] / inhibitors["total_msgs"]
    inhibitors["wait_time"] = profile_rng.lognormal(
        mean=-2.0, sigma=1.0, size=len(inhibitors)
    )
    inhibitors["comm_sparsity"] = profile_rng.uniform(0.0, 1.0, size=len(inhibitors))

    job_x, inhibitor_x = standardized_pair(jobs, inhibitors)
    rank = int(scenario["true_rank"])
    global_weight = truth_rng.normal(scale=0.08, size=4)
    victim_weight = truth_rng.normal(scale=0.18, size=(4, rank))
    aggressor_weight = truth_rng.normal(scale=0.18, size=(4, rank))
    victim_bias = truth_rng.normal(scale=0.08, size=len(jobs))
    victim_latent = job_x @ victim_weight
    inhibitor_latent = inhibitor_x @ aggressor_weight
    response_latent = (
        inhibitor_x @ global_weight
        + (victim_latent @ inhibitor_latent.T) / np.sqrt(max(rank, 1))
        + victim_bias[:, None]
    )
    target_floor = float(scenario["censor_fraction"])
    if target_floor == 0.0:
        intercept = 0.05 - float(response_latent.min())
    else:
        intercept = -float(np.quantile(response_latent, target_floor))
    response_latent += intercept
    rows = []
    missing = missing_rng.random(response_latent.shape) < float(
        scenario["missing_edge_fraction"]
    )
    minimum_anchors = min(4, len(inhibitors))
    for job_index in range(len(jobs)):
        keep = np.flatnonzero(~missing[job_index])
        if len(keep) < minimum_anchors:
            missing[job_index, np.argsort(response_latent[job_index])[-minimum_anchors:]] = False
        positive = np.flatnonzero(
            (~missing[job_index]) & (response_latent[job_index] > 0)
        )
        if len(positive) < minimum_anchors:
            forced = np.argsort(response_latent[job_index])[-minimum_anchors:]
            response_latent[job_index, forced] = np.maximum(
                response_latent[job_index, forced], 0.05
            )
            missing[job_index, forced] = False

    replicate_count = int(scenario["replicates_per_edge"])
    response_noise = response_rng.normal(
        scale=float(scenario["log_noise_sd"]),
        size=(replicate_count, *response_latent.shape),
    )
    for job_index, job_id in enumerate(jobs["job_id"]):
        for inhibitor_index, inhib_id in enumerate(inhibitors["inhib_id"]):
            if missing[job_index, inhibitor_index]:
                continue
            for replicate_index in range(replicate_count):
                value = response_latent[job_index, inhibitor_index]
                value += response_noise[replicate_index, job_index, inhibitor_index]
                if replicate_rng.random() < float(scenario["outlier_fraction"]):
                    value += 0.35 * replicate_rng.standard_t(df=2)
                rows.append(
                    [job_id, inhib_id, float(np.exp(np.clip(value, 0.0, 20.0)))]
                )
    responses = pd.DataFrame(rows, columns=["job_id", "inhib_id", "slowdown"])
    uncensored_counts = responses[responses["slowdown"] > 1.0].groupby("job_id").size()
    missing_uncensored = sorted(
        set(jobs["job_id"].astype(str)) - set(uncensored_counts.index.astype(str))
    )
    if missing_uncensored:
        raise RuntimeError(
            "Synthetic scenario produced victims without uncensored anchors: "
            f"{missing_uncensored}"
        )

    pair_rows = []
    mismatch = float(scenario["transfer_mismatch"])
    pair_noise = float(scenario["log_noise_sd"])
    job_aggressor_latent = job_x @ aggressor_weight
    for left in range(len(jobs)):
        for right in range(left, len(jobs)):
            shared_noise = pair_rng.normal(scale=pair_noise * 0.7)
            left_log = (
                job_x[right] @ global_weight
                + victim_latent[left] @ job_aggressor_latent[right] / np.sqrt(max(rank, 1))
                + victim_bias[left]
                + intercept
            )
            right_log = (
                job_x[left] @ global_weight
                + victim_latent[right] @ job_aggressor_latent[left] / np.sqrt(max(rank, 1))
                + victim_bias[right]
                + intercept
            )
            left_log += mismatch * np.tanh(job_x[left, 0] * job_x[right, 2])
            right_log += mismatch * np.tanh(job_x[right, 0] * job_x[left, 2])
            left_log += shared_noise + pair_rng.normal(scale=pair_noise * 0.3)
            right_log += shared_noise + pair_rng.normal(scale=pair_noise * 0.3)
            pair_rows.append(
                [
                    jobs.iloc[left]["job_id"],
                    jobs.iloc[right]["job_id"],
                    float(np.exp(np.clip(left_log, 0.0, 20.0))),
                    float(np.exp(np.clip(right_log, 0.0, 20.0))),
                ]
            )
    pairs = pd.DataFrame(
        pair_rows, columns=["jobA_id", "jobB_id", "slowdown_A", "slowdown_B"]
    )

    jobs.to_csv(output_dir / "jobs.csv", index=False)
    inhibitors.to_csv(output_dir / "inhibitors.csv", index=False)
    responses.to_csv(output_dir / "job_inh.csv", index=False)
    pairs.to_csv(output_dir / "pair.csv", index=False)
    unique_edges = responses[["job_id", "inhib_id"]].drop_duplicates()
    summary = {
        "requested": requested,
        "application_count": len(jobs),
        "inhibitor_count": len(inhibitors),
        "response_rows": len(responses),
        "unique_response_edges": len(unique_edges),
        "replicate_rows": len(responses) - len(unique_edges),
        "realized_censor_fraction": float(responses["slowdown"].eq(1.0).mean()),
        "realized_missing_edge_fraction": float(
            1.0 - len(unique_edges) / (len(jobs) * len(inhibitors))
        ),
        "minimum_uncensored_anchors_per_job": int(uncensored_counts.min()),
        "pair_rows": len(pairs),
        "self_pair_rows": int((pairs["jobA_id"] == pairs["jobB_id"]).sum()),
        "file_sha256": {
            name: sha256_file(output_dir / name)
            for name in ["jobs.csv", "inhibitors.csv", "job_inh.csv", "pair.csv"]
        },
    }
    write_json(summary_path, summary)
    return summary


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
    strategy: str,
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
    data_kind: str,
    inputs: dict[str, str],
    config: dict[str, Any],
    scenario_id: str | None = None,
    scenario: dict[str, Any] | None = None,
    generator_seed: int | None = None,
    split_id: str | None = None,
    split_strategy: str | None = None,
    training_app_count: int | None = None,
    depends_on: str | None = None,
    candidate_strategy: str = "fixed",
) -> dict[str, Any]:
    task = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "data_kind": data_kind,
        "inputs": inputs,
        "input_sha256": {
            name: sha256_file(Path(path)) for name, path in inputs.items()
        },
        "config": config,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "generator_seed": generator_seed,
        "split_id": split_id,
        "split_strategy": split_strategy,
        "training_app_count": training_app_count,
        "depends_on": depends_on,
        "candidate_strategy": candidate_strategy,
    }
    task["task_id"] = stable_id("run", task)
    return task


def build_plan(
    root: Path,
    profile_name: str,
    suites: set[str],
) -> list[dict[str, Any]]:
    profile = PROFILES[profile_name]
    tasks: list[dict[str, Any]] = []
    real_inputs = {
        "jobs": str((DEFAULT_DATA / "jobs.csv").resolve()),
        "inhibitors": str((DEFAULT_DATA / "inhibitors.csv").resolve()),
        "job_inh": str((DEFAULT_DATA / "job_inh.csv").resolve()),
    }

    if "real" in suites:
        config = base_run_config(profile)
        config["evaluation_method"] = "random_split"
        config["training_apps"] = None
        config["model_candidates"] = architecture_candidates()
        architecture = make_task(
            stage="real_architecture",
            data_kind="real",
            inputs=real_inputs,
            config=config,
        )
        tasks.append(architecture)

        optimizer_config = base_run_config(profile)
        optimizer_config.update(
            {"evaluation_method": "random_split", "training_apps": None}
        )
        optimizer = make_task(
            stage="real_optimizer",
            data_kind="real",
            inputs=real_inputs,
            config=optimizer_config,
            depends_on=architecture["task_id"],
            candidate_strategy="optimizer_from_dependency",
        )
        tasks.append(optimizer)

        capacity_config = base_run_config(profile)
        capacity_config.update(
            {"evaluation_method": "random_split", "training_apps": None}
        )
        capacity = make_task(
            stage="real_capacity",
            data_kind="real",
            inputs=real_inputs,
            config=capacity_config,
            depends_on=optimizer["task_id"],
            candidate_strategy="capacity_from_dependency",
        )
        tasks.append(capacity)

        for blocks, cluster_seed in [
            (3, 1701),
            (4, 1701),
            (6, 1701),
            (4, 1702),
            (4, 1703),
        ]:
            fold_config = base_run_config(profile)
            fold_config.update(
                {
                    "evaluation_method": "random_split",
                    "training_apps": None,
                    "inhibitor_blocks": blocks,
                    "clustering_seed": cluster_seed,
                }
            )
            tasks.append(
                make_task(
                    stage="real_fold_robustness",
                    data_kind="real",
                    inputs=real_inputs,
                    config=fold_config,
                    depends_on=capacity["task_id"],
                    candidate_strategy="selected_from_dependency",
                )
            )

    scenario_records: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    if "stress" in suites:
        for scenario_name, scenario in stress_scenarios(profile_name):
            for generator_seed in profile.synthetic_seeds:
                scenario_id = f"{scenario_name}_seed{generator_seed}"
                data_dir = root / "synthetic_data" / scenario_id
                generate_synthetic_data(scenario, generator_seed, data_dir)
                scenario_records[(scenario_name, generator_seed)] = (scenario, data_dir)
                config = base_run_config(profile)
                config.update(
                    {
                        "evaluation_method": "random_split",
                        "training_apps": None,
                        "inhibitor_blocks": min(4, int(scenario["n_inhibitors"])),
                    }
                )
                tasks.append(
                    make_task(
                        stage="synthetic_stress",
                        data_kind="synthetic",
                        inputs={
                            "jobs": str(data_dir / "jobs.csv"),
                            "inhibitors": str(data_dir / "inhibitors.csv"),
                            "job_inh": str(data_dir / "job_inh.csv"),
                            "pair": str(data_dir / "pair.csv"),
                        },
                        config=config,
                        scenario_id=scenario_id,
                        scenario=scenario,
                        generator_seed=generator_seed,
                    )
                )

    membership_rows = []
    if "learning_curve" in suites:
        strategies = ["balanced_random"]
        if profile_name == "extended":
            strategies.append("profile_diverse")
        for generator_seed in profile.synthetic_seeds:
            scenario_name = "baseline"
            scenario = BASE_SCENARIO.copy()
            scenario_id = f"{scenario_name}_seed{generator_seed}"
            data_dir = root / "synthetic_data" / scenario_id
            generate_synthetic_data(scenario, generator_seed, data_dir)
            jobs = pd.read_csv(data_dir / "jobs.csv")
            for strategy_index, strategy in enumerate(strategies):
                orders = nested_orders(
                    jobs,
                    min(profile.split_replicates, len(jobs)),
                    seed=811 + generator_seed + 1009 * strategy_index,
                    strategy=strategy,
                )
                for split_index, order in enumerate(orders):
                    split_id = f"{strategy}_g{generator_seed}_s{split_index:02d}"
                    for training_size in profile.training_sizes:
                        training_apps = order[:training_size]
                        for rank, app_id in enumerate(order):
                            membership_rows.append(
                                {
                                    "scenario_id": scenario_id,
                                    "generator_seed": generator_seed,
                                    "split_id": split_id,
                                    "split_strategy": strategy,
                                    "training_app_count": training_size,
                                    "app_id": app_id,
                                    "addition_rank": rank + 1,
                                    "in_training": rank < training_size,
                                }
                            )
                        config = base_run_config(profile)
                        config.update(
                            {
                                "evaluation_method": "both",
                                "training_apps": training_apps,
                                "model_candidates": [DEFAULT_MODEL],
                            }
                        )
                        tasks.append(
                            make_task(
                                stage="training_size_learning_curve",
                                data_kind="synthetic",
                                inputs={
                                    "jobs": str(data_dir / "jobs.csv"),
                                    "inhibitors": str(data_dir / "inhibitors.csv"),
                                    "job_inh": str(data_dir / "job_inh.csv"),
                                    "pair": str(data_dir / "pair.csv"),
                                },
                                config=config,
                                scenario_id=scenario_id,
                                scenario=scenario,
                                generator_seed=generator_seed,
                                split_id=split_id,
                                split_strategy=strategy,
                                training_app_count=training_size,
                            )
                        )

    if membership_rows:
        pd.DataFrame(membership_rows).to_csv(
            root / "subset_membership.csv", index=False
        )
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
    suites = set(args.suites.split(","))
    unknown = suites - {"real", "stress", "learning_curve"}
    if unknown:
        raise ValueError(f"Unknown suites: {sorted(unknown)}")
    tasks = build_plan(root, args.profile, suites)
    specification = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "profile": args.profile,
        "suites": sorted(suites),
        "holdout_policy": (
            "Real tasks are App-Inhibitor-only; pair evaluation uses generated "
            "synthetic data only."
        ),
        "profile_settings": asdict(PROFILES[args.profile]),
        "task_count": len(tasks),
    }
    write_json(root / "experiment_spec.json", specification)
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
    if not report.exists() or read_json(report).get("status") != "complete":
        return None
    success_path = task_root(root, task_id) / "_SUCCESS.json"
    if not success_path.exists():
        return None
    success = read_json(success_path)
    if success.get("run_report_sha256") != sha256_file(report):
        return None
    return output


def execution_fingerprint(
    root: Path, task: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    dependency = task.get("depends_on")
    dependency_selection = None
    if dependency:
        dependency_output = successful_output(root, dependency)
        if dependency_output is None:
            raise RuntimeError(f"Dependency {dependency} has not completed successfully")
        dependency_selection = sha256_file(dependency_output / "selection.json")
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
        "dependency_selection": dependency_selection,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def dependency_models(root: Path, task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dependency = task.get("depends_on")
    if not dependency:
        return {kind: DEFAULT_MODEL.copy() for kind in ["low_rank", "generic", "absolute"]}
    output = successful_output(root, dependency)
    if output is None:
        raise RuntimeError(f"Dependency {dependency} has not completed successfully")
    selection = read_json(output / "selection.json")
    return {
        kind: dict(model) for kind, model in selection["model_configs"].items()
    }


def resolve_task_config(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(task["config"]))
    strategy = task["candidate_strategy"]
    if strategy == "fixed":
        return config
    selected = dependency_models(root, task)
    if strategy == "optimizer_from_dependency":
        config["model_candidates_by_kind"] = {
            kind: [
                {
                    **model,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                }
                for learning_rate in [3e-4, 1e-3, 3e-3]
                for weight_decay in [0.0, 1e-4, 1e-3]
            ]
            for kind, model in selected.items()
        }
    elif strategy == "capacity_from_dependency":
        config["model_candidates_by_kind"] = {
            kind: [
                {**model, "hidden_dim": hidden, "embedding_dim": embedding}
                for hidden, embedding in [(4, 2), (8, 4), (16, 8)]
            ]
            for kind, model in selected.items()
        }
    elif strategy == "selected_from_dependency":
        config["model_candidates_by_kind"] = {
            kind: [model] for kind, model in selected.items()
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
        success = read_json(success_path)
        if success.get("execution_fingerprint") == fingerprint:
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
    if task["data_kind"] == "real":
        command.append("--skip-holdout")
    else:
        command.extend(["--pair-csv", task["inputs"]["pair"]])
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
    lines = [
        line.strip() for line in stdout_path.read_text().splitlines() if line.strip()
    ]
    return lines[-1] if lines else None


def run_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    tasks = read_json(root / "plan.json")
    requested_stages = set(args.stages.split(",")) if args.stages else None
    selected = [
        task
        for task in tasks
        if requested_stages is None or task["stage"] in requested_stages
    ]
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]
    if requested_stages is not None:
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
        selected.sort(key=lambda task: plan_order[task["task_id"]])

    total_tasks = len(selected)
    completed_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0

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
                f"Starting {task['task_id']} ({task.get('stage', 'unknown')}); "
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
                            f"Still running {task['task_id']} "
                            f"({task.get('stage', 'unknown')}), "
                            f"elapsed {elapsed:.0f}s{detail}"
                        )
                    continue
                for future in done:
                    result = future.result()
                    status = result["status"]
                    if status == "succeeded":
                        completed_tasks += 1
                    elif status == "failed":
                        failed_tasks += 1
                    elif status == "skipped":
                        skipped_tasks += 1

                    stage = remaining[result["task_id"]].get("stage", "unknown")
                    scenario = remaining[result["task_id"]].get("scenario_id", "")
                    display_id = f"{scenario[:20]}... " if scenario else ""
                    tqdm.write(
                        f"[{completed_tasks + failed_tasks + skipped_tasks}/{total_tasks}] "
                        f"{display_id}{stage}: {status}"
                    )

                    remaining.pop(result["task_id"], None)
        refresh_manifest(root, tasks)
    if remaining:
        tqdm.write(f"{len(remaining)} tasks remain blocked by incomplete dependencies")
    tqdm.write(f"Summary: {completed_tasks} succeeded, {failed_tasks} failed, {skipped_tasks} skipped")
    consolidate(root, include_predictions=args.include_predictions)


def flatten_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    scenario = task.get("scenario") or {}
    return {
        "task_id": task["task_id"],
        "stage": task["stage"],
        "data_kind": task["data_kind"],
        "scenario_id": task.get("scenario_id"),
        "generator_seed": task.get("generator_seed"),
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
        **{f"scenario_{key}": value for key, value in scenario.items()},
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


def with_provenance(frame: pd.DataFrame, task: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    provenance = flatten_task(task)
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


def consolidate(root: Path, include_predictions: bool = True) -> None:
    tasks = read_json(root / "plan.json")
    output_root = root / "consolidated"
    output_root.mkdir(parents=True, exist_ok=True)
    fold_frames = []
    candidate_rows = []
    selection_rows = []
    metric_frames = []
    prediction_frames = []
    eligibility_frames = []
    condition_rows = []
    failures = []

    for task in tasks:
        status = latest_status(root, task["task_id"])
        if not status or status.get("status") != "succeeded":
            if status and status.get("status") == "failed":
                failures.append({**flatten_task(task), **status})
            continue
        output = Path(status["output_dir"])
        resolved_config = read_json(output.parent / "resolved_config.json")
        candidates_by_kind = resolved_config.get("model_candidates_by_kind")
        if candidates_by_kind is None:
            candidates_by_kind = {
                kind: resolved_config["model_candidates"]
                for kind in ["low_rank", "generic", "absolute"]
            }
        for kind, candidates in candidates_by_kind.items():
            for config_id, candidate in enumerate(candidates):
                candidate_rows.append(
                    {
                        **flatten_task(task),
                        "model_kind": kind,
                        "config_id": config_id,
                        "candidate_id": stable_id("candidate", candidate, 12),
                        **candidate,
                    }
                )
        fold_path = output / "crossed_validation_folds.csv"
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            for field in DEFAULT_MODEL:
                folds[field] = [
                    candidates_by_kind[str(kind)][int(config_id)][field]
                    for kind, config_id in zip(folds["model_kind"], folds["config_id"])
                ]
            folds["candidate_id"] = [
                stable_id(
                    "candidate",
                    candidates_by_kind[str(kind)][int(config_id)],
                    12,
                )
                for kind, config_id in zip(folds["model_kind"], folds["config_id"])
            ]
            fold_frames.append(with_provenance(folds, task))
        selection_path = output / "selection.json"
        if selection_path.exists():
            selection = read_json(selection_path)
            for kind, model in selection.get("model_configs", {}).items():
                model_selection = selection.get("model_selection", {}).get(kind, {})
                aggregation = selection.get("aggregations", {}).get(kind, {})
                selection_rows.append(
                    {
                        **flatten_task(task),
                        "model_kind": kind,
                        **model,
                        **{f"selection_{key}": value for key, value in model_selection.items()},
                        "primary_aggregation": aggregation.get("primary_method"),
                        "kernel_temperature": aggregation.get("kernel", {}).get(
                            "temperature"
                        ),
                        "ood_temperature": aggregation.get("ood_kernel", {}).get(
                            "temperature"
                        ),
                        "ood_quantile": aggregation.get("ood_kernel", {}).get(
                            "quantile"
                        ),
                        "ood_fallback": aggregation.get("ood_kernel", {}).get(
                            "fallback"
                        ),
                    }
                )
        if task.get("scenario_id"):
            summary = read_json(Path(task["inputs"]["jobs"]).parent / "scenario_summary.json")
            requested = summary.pop("requested")
            file_hashes = summary.pop("file_sha256")
            condition_rows.append(
                {
                    **flatten_task(task),
                    **{f"requested_{key}": value for key, value in requested.items()},
                    **summary,
                    **{f"sha256_{key}": value for key, value in file_hashes.items()},
                }
            )

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
            "paired_method_differences_non_self.csv": (
                "non_self_paired_difference"
            ),
        }
        for evaluation_dir in evaluation_directories(output):
            if include_predictions:
                prediction_path = evaluation_dir / "directional_predictions.csv"
                if prediction_path.exists():
                    prediction_frames.append(
                        with_provenance(pd.read_csv(prediction_path), task)
                    )
            eligibility_path = evaluation_dir / "evaluation_pair_manifest.csv"
            if eligibility_path.exists():
                eligibility_frames.append(
                    with_provenance(pd.read_csv(eligibility_path), task)
                )
            for filename, scope in report_files.items():
                path = evaluation_dir / filename
                if path.exists():
                    metric_frames.append(
                        with_provenance(metrics_to_long(pd.read_csv(path), scope), task)
                    )

    outputs = {
        "fold_metrics.csv": fold_frames,
        "metrics_long.csv": metric_frames,
        "predictions.csv": prediction_frames,
        "pair_eligibility.csv": eligibility_frames,
    }
    for filename, frames in outputs.items():
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                output_root / filename, index=False
            )
        elif (output_root / filename).exists():
            (output_root / filename).unlink()
    pd.DataFrame(selection_rows).to_csv(output_root / "selections.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_root / "candidates.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(
        output_root / "data_conditions.csv", index=False
    )
    pd.DataFrame(failures).to_csv(output_root / "failures.csv", index=False)

    if metric_frames:
        all_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        learning = all_metrics[
            (all_metrics["stage"] == "training_size_learning_curve")
            & (all_metrics["scope"] == "overall_non_self")
        ].copy()
        if not learning.empty:
            learning.to_csv(
                output_root / "learning_curve_replicates.csv", index=False
            )
            group_columns = [
                "training_app_count",
                "evaluation_method",
                "anchor_budget",
                "method",
                "metric",
            ]
            generator_means = (
                learning.groupby([*group_columns, "generator_seed"], as_index=False)[
                    "estimate"
                ]
                .mean()
                .rename(columns={"estimate": "generator_mean"})
            )
            summary_rows = []
            for keys, group in generator_means.groupby(group_columns, sort=False):
                values = group["generator_mean"].to_numpy(float)
                summary_rows.append(
                    {
                        **dict(zip(group_columns, keys)),
                        "estimate": float(values.mean()),
                        "between_generator_sd": float(values.std(ddof=1))
                        if len(values) > 1
                        else np.nan,
                        "between_generator_se": float(values.std(ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else np.nan,
                        "generator_count": len(values),
                        "split_replicate_count": int(
                            learning[
                                np.logical_and.reduce(
                                    [learning[column] == value for column, value in zip(group_columns, keys)]
                                )
                            ]["split_id"].nunique()
                        ),
                    }
                )
            pd.DataFrame(summary_rows).to_csv(
                output_root / "learning_curve_summary.csv", index=False
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write an immutable experiment plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    plan.add_argument(
        "--suites",
        default="real,stress,learning_curve",
        help="comma-separated subset of real,stress,learning_curve",
    )
    plan.set_defaults(function=plan_command)

    run = subparsers.add_parser("run", help="execute planned tasks")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--stages", help="comma-separated stage filter")
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--max-tasks", type=int)
    run.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="seconds between progress messages while tasks are running",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--include-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.set_defaults(function=run_command)

    consolidate_parser = subparsers.add_parser(
        "consolidate", help="rebuild tidy result tables"
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
    return parser.parse_args()


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
