"""Immutable Path A planning, fold membership, and content seals."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from . import __version__
from .config import ExperimentConfig, candidate_registry
from .data import TrainingData, app_profile_weights, inhibitor_blocks, load_training_data


STATIC_FILES = [
    "experiment_spec.json",
    "environment.json",
    "input_hashes.json",
    "code_hashes.json",
    "inhibitor_manifest.csv",
    "fold_manifest.csv",
    "pair_split_manifest.csv",
    "candidate_manifest.csv",
    "rejected_job_inh.csv",
    "duplicate_job_inh_keys.csv",
    "profile_weighting.json",
    "configs/experiment_config.json",
]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(workspace: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Source hashes below bind the package itself. Avoid a repository-wide
        # status scan so planning never traverses unrelated sealed data.
        return {"commit": commit}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None}


def environment_record(workspace: Path) -> dict[str, object]:
    return {
        "delta_response_2": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git": _git_state(workspace),
    }


def code_hashes(package_dir: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in package_dir.glob("*.py")
        if path.name != "__pycache__" and path.is_file()
    )
    return {path.name: sha256_file(path) for path in files}


def _write_fold_manifest(
    path: Path,
    data: TrainingData,
    inhibitor_manifest: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    geometry_columns = [
        "inhib_id",
        "profile_weight",
        "app_nearest_subset",
        "mechanism_family",
        "profile_block",
        "mechanism_block",
    ]
    rows = data.responses.merge(
        inhibitor_manifest[geometry_columns], on="inhib_id", how="left", validate="many_to_one"
    )
    wrote_header = False
    for scheme in config.fold_schemes:
        block_column = f"{scheme}_block"
        for victim_id in data.jobs["job_id"].astype(str):
            for heldout_block in sorted(rows[block_column].unique()):
                job_held = rows["job_id"].eq(victim_id)
                block_held = rows[block_column].eq(heldout_block)
                role = np.select(
                    [job_held & block_held, job_held & ~block_held, ~job_held & ~block_held],
                    ["query", "support", "parameter_train"],
                    default="excluded_held_block",
                )
                base = pd.DataFrame(
                    {
                        "fold_id": f"{scheme}:{victim_id}:block{int(heldout_block)}",
                        "fold_scheme": scheme,
                        "victim_id": victim_id,
                        "heldout_block": int(heldout_block),
                        "response_id": rows["response_id"].to_numpy(dtype=int),
                        "response_job_id": rows["job_id"].astype(str).to_numpy(),
                        "inhib_id": rows["inhib_id"].astype(str).to_numpy(),
                        "inhibitor_block": rows[block_column].to_numpy(dtype=int),
                        "mechanism_family": rows["mechanism_family"].astype(str).to_numpy(),
                        "role": role,
                        "model_job_profile_in_scaler": (~job_held).to_numpy(dtype=bool),
                        "model_inhibitor_profile_in_scaler": (~block_held).to_numpy(dtype=bool),
                        "distance_inhibitor_profile_in_scaler": (~block_held).to_numpy(dtype=bool),
                        "clustering_inhibitor_profile_in_scaler": True,
                        "weighting_job_profile_in_scaler": True,
                        "weighting_inhibitor_profile_in_scaler": True,
                        "censoring_stratum": np.where(rows["is_censored"], "floor", "exact"),
                        "profile_weight": rows["profile_weight"].to_numpy(dtype=float),
                        "app_nearest_subset": rows["app_nearest_subset"].to_numpy(dtype=bool),
                    }
                )
                for seed in config.seeds:
                    manifest = base.copy()
                    manifest.insert(4, "seed", int(seed))
                    manifest.to_csv(path, mode="a", header=not wrote_header, index=False)
                    wrote_header = True


def _candidate_frame(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for candidate in candidate_registry(config):
        row = {key: value for key, value in candidate.items() if key != "parameters"}
        row["parameters_json"] = json.dumps(candidate["parameters"], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def create_plan(
    root: Path,
    *,
    config: ExperimentConfig,
    jobs_csv: Path,
    inhibitors_csv: Path,
    job_inh_csv: Path,
    package_dir: Path,
    workspace: Path,
) -> None:
    """Create a sealed plan that has no pair-data path or pair-data hash."""
    config.validate()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Experiment root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for directory in ["configs", "tasks", "consolidated"]:
        (root / directory).mkdir(exist_ok=False)

    inputs = {
        "jobs_csv": str(jobs_csv.resolve()),
        "inhibitors_csv": str(inhibitors_csv.resolve()),
        "job_inh_csv": str(job_inh_csv.resolve()),
    }
    input_hashes = {
        name: {"path": path, "sha256": sha256_file(path)} for name, path in inputs.items()
    }
    write_json(root / "input_hashes.json", input_hashes)
    write_json(root / "configs" / "experiment_config.json", config.to_dict())
    write_json(root / "environment.json", environment_record(workspace))
    write_json(root / "code_hashes.json", code_hashes(package_dir))
    write_json(
        root / "experiment_spec.json",
        {
            "protocol": "random_split",
            "scientific_path": "Path A: no new App-App data",
            "app_app_labels_used_for_fitting": False,
            "app_app_labels_used_for_selection": False,
            "historical_app_app_status": "diagnostic_only_not_opened",
            "selection_domain": "App-Inhibitor development CV",
            "application_profiles_used_transductively_for_weights": True,
            "calibrated_app_app_scale_identified": False,
            "required_future_evaluation": "newly collected untouched App-App holdout",
        },
    )

    data = load_training_data(jobs_csv, inhibitors_csv, job_inh_csv)
    geometry, geometry_state = app_profile_weights(
        data.jobs,
        data.inhibitors,
        nearest_per_application=config.nearest_per_application,
        ratio_clip=tuple(config.profile_weight_clip),
    )
    blocks = inhibitor_blocks(
        data.inhibitors,
        block_count=config.inhibitor_blocks,
        seed=config.clustering_seed,
        mechanism_column=config.mechanism_column,
    )
    inhibitor_manifest = blocks.merge(geometry, on="inhib_id", validate="one_to_one")
    inhibitor_manifest.to_csv(root / "inhibitor_manifest.csv", index=False)
    write_json(root / "profile_weighting.json", geometry_state)
    _write_fold_manifest(root / "fold_manifest.csv", data, inhibitor_manifest, config)
    pd.DataFrame(
        columns=[
            "pair_cluster_id",
            "endpoint_a",
            "endpoint_b",
            "self_pair",
            "membership",
            "endpoint_a_novel",
            "endpoint_b_novel",
            "status",
        ]
    ).to_csv(root / "pair_split_manifest.csv", index=False)
    _candidate_frame(config).to_csv(root / "candidate_manifest.csv", index=False)
    data.rejected_responses.to_csv(root / "rejected_job_inh.csv", index=False)
    data.duplicate_keys.to_csv(root / "duplicate_job_inh_keys.csv", index=False)

    static_hashes = {name: sha256_file(root / name) for name in STATIC_FILES}
    seal_digest = hashlib.sha256(
        json.dumps(static_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(
        root / "seal.json",
        {
            "sha256": seal_digest,
            "static_files": static_hashes,
            "pair_data_included": False,
        },
    )


def verify_plan(root: Path, package_dir: Path) -> None:
    seal = json.loads((root / "seal.json").read_text())
    if seal.get("pair_data_included") is not False:
        raise RuntimeError("Path A plan seal incorrectly includes pair data")
    recorded = seal["static_files"]
    current = {name: sha256_file(root / name) for name in recorded}
    if current != recorded:
        changed = sorted(name for name in recorded if current.get(name) != recorded[name])
        raise RuntimeError(f"Static experiment files changed after sealing: {changed}")
    digest = hashlib.sha256(
        json.dumps(current, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != seal["sha256"]:
        raise RuntimeError("Experiment seal digest is invalid")
    expected_code = json.loads((root / "code_hashes.json").read_text())
    actual_code = code_hashes(package_dir)
    if actual_code != expected_code:
        raise RuntimeError("Delta Response 2 code changed after planning")
    for record in json.loads((root / "input_hashes.json").read_text()).values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError(f"Input changed after planning: {record['path']}")
    expected_environment = json.loads((root / "environment.json").read_text())
    critical = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    changed_environment = {
        key: (expected_environment.get(key), value)
        for key, value in critical.items()
        if expected_environment.get(key) != value
    }
    if changed_environment:
        raise RuntimeError(
            f"Critical runtime environment changed after planning: {changed_environment}"
        )


def write_completion_seal(root: Path) -> None:
    output_hash_path = root / "output_hashes.json"
    plan_seal_path = root / "seal.json"
    payload = {
        "output_hashes_sha256": sha256_file(output_hash_path),
        "plan_seal_sha256": sha256_file(plan_seal_path),
        "frozen_recipe_sha256": sha256_file(root / "frozen_recipe.json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(root / "completion_seal.json", {**payload, "sha256": digest})


def _experiment_file_roster(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise RuntimeError(f"Experiment contains a directory symlink: {base / name}")
        for name in file_names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Experiment contains a non-regular file: {path}")
            files.add(path.relative_to(root).as_posix())
    return files


def write_output_hashes(root: Path, expected_paths: set[str]) -> None:
    """Hash only the canonical output roster and reject undeclared files."""
    metadata = {"output_hashes.json", "completion_seal.json"}
    if expected_paths & metadata:
        raise ValueError("Completion metadata cannot be part of the output roster")
    actual = _experiment_file_roster(root)
    if actual != expected_paths:
        raise RuntimeError(
            "Experiment output roster differs from the declaration: "
            f"missing={sorted(expected_paths - actual)}, "
            f"unexpected={sorted(actual - expected_paths)}"
        )
    write_json(
        root / "output_hashes.json",
        {name: sha256_file(root / name) for name in sorted(expected_paths)},
    )


def verify_completion(root: Path) -> None:
    seal_path = root / "completion_seal.json"
    if not seal_path.exists():
        raise RuntimeError("Experiment has no completion seal")
    seal = json.loads(seal_path.read_text())
    payload = {
        "output_hashes_sha256": sha256_file(root / "output_hashes.json"),
        "plan_seal_sha256": sha256_file(root / "seal.json"),
        "frozen_recipe_sha256": sha256_file(root / "frozen_recipe.json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if any(seal.get(key) != value for key, value in payload.items()) or seal.get(
        "sha256"
    ) != digest:
        raise RuntimeError("Completion seal is invalid")
    expected_outputs = json.loads((root / "output_hashes.json").read_text())
    expected_roster = set(expected_outputs) | {
        "output_hashes.json",
        "completion_seal.json",
    }
    actual_roster = _experiment_file_roster(root)
    if actual_roster != expected_roster:
        raise RuntimeError(
            "Completed experiment file roster changed: "
            f"missing={sorted(expected_roster - actual_roster)}, "
            f"unexpected={sorted(actual_roster - expected_roster)}"
        )
    changed = [
        name
        for name, expected in expected_outputs.items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"Completed experiment outputs changed: {sorted(changed)}")
