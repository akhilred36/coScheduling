"""Replay a frozen Path A recipe on unlabeled application-profile queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .censoring import fit_censored_intercept
from .data import (
    ProfileScaler,
    load_training_data,
    log_to_slowdown,
    observe_log_slowdown,
    profile_maps,
)
from .inference import (
    LegacyOOD,
    OODCalibration,
    censored_potential_predictions,
    legacy_exact_corrections,
)
from .manifests import sha256_file, verify_completion, verify_plan
from .model import AbsoluteResponseModel, LowRankPotential
from .training import absolute_values, linear_design, response_values


def load_frozen_recipe(root: Path | str) -> dict[str, Any]:
    experiment = Path(root).resolve()
    verify_plan(experiment, Path(__file__).resolve().parent)
    verify_completion(experiment)
    recipe = json.loads((experiment / "frozen_recipe.json").read_text())
    recorded_digest = recipe.pop("recipe_content_sha256")
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if hashlib.sha256(canonical).hexdigest() != recorded_digest:
        raise RuntimeError("Frozen recipe content hash is invalid")
    recipe["recipe_content_sha256"] = recorded_digest
    checkpoint_dir = experiment / recipe["final_training"]["checkpoint_directory"]
    for name, expected in recipe["final_training"]["artifact_hashes"].items():
        path = checkpoint_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")
    expected_models = set(recipe["final_training"].get("model_artifacts", []))
    actual_models = {
        path.name
        for pattern in ["delta_seed*.pt", "absolute_neural_seed*.pt", "absolute_tree_seed*.joblib"]
        for path in checkpoint_dir.glob(pattern)
    }
    if actual_models != expected_models:
        raise RuntimeError(
            "Frozen model checkpoint set changed: "
            f"expected={sorted(expected_models)}, actual={sorted(actual_models)}"
        )
    return recipe


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _planned_data(root: Path):
    inputs = json.loads((root / "input_hashes.json").read_text())
    return load_training_data(
        inputs["jobs_csv"]["path"],
        inputs["inhibitors_csv"]["path"],
        inputs["job_inh_csv"]["path"],
    )


def predict_frozen_recipe(
    root: Path | str,
    victim_id: str,
    aggressor_id: str,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Predict one unlabeled known-profile direction without opening pair outcomes."""
    experiment = Path(root).resolve()
    recipe = load_frozen_recipe(experiment)
    selected = recipe["selected"]
    family = selected["family"]
    parameters = selected["parameters"]
    data = _planned_data(experiment)
    jobs = set(data.jobs["job_id"].astype(str))
    if str(victim_id) not in jobs or str(aggressor_id) not in jobs:
        raise ValueError("victim_id and aggressor_id must have sealed application profiles")
    checkpoint_dir = experiment / recipe["final_training"]["checkpoint_directory"]
    model_artifacts = list(recipe["final_training"].get("model_artifacts", []))
    model_scaler = ProfileScaler.from_state_dict(
        json.loads((checkpoint_dir / "profile_scaler.json").read_text())
    )
    distance_scaler = ProfileScaler.from_state_dict(
        json.loads((checkpoint_dir / "distance_profile_scaler.json").read_text())
    )
    job_profiles, inhibitor_profiles = profile_maps(
        data.jobs, data.inhibitors, model_scaler
    )
    distance_jobs, distance_inhibitors = profile_maps(
        data.jobs, data.inhibitors, distance_scaler
    )
    anchors = data.responses[data.responses["job_id"] == str(victim_id)].reset_index(
        drop=True
    )
    observed = anchors["log_slowdown"].to_numpy(dtype=float)
    anchor_distance_profiles = np.stack(
        [distance_inhibitors[str(value)] for value in anchors["inhib_id"]]
    )
    distances = np.linalg.norm(
        anchor_distance_profiles - distance_jobs[str(aggressor_id)][None, :], axis=1
    )
    latent_by_seed: list[float] = []

    if family == "constant":
        latent_by_seed = [0.0]
    elif family == "global_intercept":
        state = json.loads((checkpoint_dir / "baseline_state.json").read_text())
        latent_by_seed = [float(state["latent_prediction"])]
    elif family == "victim_intercept":
        state = json.loads((checkpoint_dir / "baseline_state.json").read_text())
        result = fit_censored_intercept(
            observed,
            shrinkage=float(state["victim_shrinkage"]),
            prior=float(state["population_prior"]),
        )
        latent_by_seed = [result.correction]
    elif family == "anchor_baseline":
        if selected["method"] == "nearest_anchor":
            latent_by_seed = [float(observed[int(np.argmin(distances))])]
        elif selected["method"] == "robust_median_anchor":
            latent_by_seed = [float(np.median(observed))]
        else:
            raise RuntimeError(f"Unknown frozen anchor method: {selected['method']}")
    elif family == "absolute_linear":
        state = json.loads((checkpoint_dir / "absolute_linear.json").read_text())
        coefficients = np.asarray(state["coefficients"], dtype=float)
        design = linear_design(
            job_profiles[str(victim_id)][None, :],
            job_profiles[str(aggressor_id)][None, :],
        )
        latent_by_seed = [float((design @ coefficients)[0])]
    elif family == "absolute_tree":
        design = linear_design(
            job_profiles[str(victim_id)][None, :],
            job_profiles[str(aggressor_id)][None, :],
        )
        for name in model_artifacts:
            if not name.startswith("absolute_tree_seed"):
                continue
            path = checkpoint_dir / name
            latent_by_seed.append(float(joblib.load(path).predict(design[:, 1:])[0]))
    elif family == "absolute_neural":
        target_device = _device(device)
        victim = job_profiles[str(victim_id)][None, :]
        aggressor = job_profiles[str(aggressor_id)][None, :]
        for name in model_artifacts:
            if not name.startswith("absolute_neural_seed"):
                continue
            path = checkpoint_dir / name
            checkpoint = torch.load(path, map_location=target_device, weights_only=False)
            model = AbsoluteResponseModel(
                int(checkpoint["input_dim"]),
                int(checkpoint["hidden_dim"]),
                int(checkpoint["embedding_dim"]),
                int(checkpoint["rank"]),
            ).to(target_device)
            model.load_state_dict(checkpoint["state_dict"])
            latent_by_seed.append(
                float(absolute_values(model, victim, aggressor, target_device)[0])
            )
    elif family in {
        "legacy_delta",
        "censored_delta",
        "censored_delta_query_shrinkage",
    }:
        target_device = _device(device)
        anchor_profiles = np.stack(
            [inhibitor_profiles[str(value)] for value in anchors["inhib_id"]]
        )
        target_profile = job_profiles[str(aggressor_id)]
        for name in model_artifacts:
            if not name.startswith("delta_seed"):
                continue
            path = checkpoint_dir / name
            checkpoint = torch.load(path, map_location=target_device, weights_only=False)
            model = LowRankPotential(
                int(checkpoint["input_dim"]),
                int(checkpoint["hidden_dim"]),
                int(checkpoint["embedding_dim"]),
                int(checkpoint["rank"]),
            ).to(target_device)
            model.load_state_dict(checkpoint["state_dict"])
            potentials = response_values(
                model,
                job_profiles[str(victim_id)],
                np.vstack([anchor_profiles, target_profile[None, :]]),
                target_device,
            )
            support_potential = potentials[:-1]
            query_potential = potentials[-1:]
            if family == "legacy_delta":
                aggregation = str(parameters["aggregation"])
                if aggregation == "ood":
                    state = recipe["final_training"]["deployment_ood"]
                    if state is None:
                        raise RuntimeError("Frozen OOD method lacks deployment calibration")
                    calibration = OODCalibration(
                        float(state["threshold"]),
                        float(state["width"]),
                        float(state["quantile"]),
                        int(state["reference_count"]),
                    )
                else:
                    calibration = OODCalibration(0.0, 1.0, 0.9, 1)
                corrections = legacy_exact_corrections(
                    observed,
                    support_potential,
                    distances,
                    calibration,
                    ood=LegacyOOD(
                        float(recipe["resolved_experiment_config"]["legacy_ood_temperature"]),
                        float(recipe["resolved_experiment_config"]["legacy_ood_quantile"]),
                        str(recipe["resolved_experiment_config"]["legacy_ood_fallback"]),
                    ),
                )
                correction = corrections[aggregation][0]
                latent_by_seed.append(float(query_potential[0] + correction))
            else:
                prior = (
                    float(checkpoint["population_correction_prior"])
                    if parameters.get("prior_mode") == "population"
                    else 0.0
                )
                latent, _, _ = censored_potential_predictions(
                    observed,
                    support_potential,
                    query_potential,
                    shrinkage=float(parameters["shrinkage"]),
                    prior=prior,
                    query_gamma=float(parameters["query_gamma"]),
                )
                latent_by_seed.append(float(latent[0]))
    else:
        raise RuntimeError(f"Unsupported frozen candidate family: {family}")

    if not latent_by_seed or not np.isfinite(latent_by_seed).all():
        raise RuntimeError("Frozen recipe produced no finite seed predictions")
    latent = float(np.mean(latent_by_seed))
    observed_log = float(observe_log_slowdown(latent))
    return {
        "victim_id": str(victim_id),
        "aggressor_id": str(aggressor_id),
        "method": selected["method"],
        "candidate_id": selected["candidate_id"],
        "seed_latent_predictions": latent_by_seed,
        "latent_predicted_log_slowdown": latent,
        "predicted_log_slowdown": observed_log,
        "predicted_slowdown": float(log_to_slowdown(observed_log)),
        "app_app_outcomes_used": False,
    }
