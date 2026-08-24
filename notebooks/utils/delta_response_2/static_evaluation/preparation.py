"""Pair-blind diagnostic fitting and immutable preparation verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from delta_response_2.data import ProfileScaler, load_training_data, profile_maps
from delta_response_2.frozen import load_frozen_recipe, predict_frozen_recipe
from delta_response_2.inference import (
    LegacyOOD,
    OODCalibration,
    calibrate_ood,
    censored_potential_predictions,
    legacy_exact_corrections,
    population_correction_prior,
    training_distance_reference,
)
from delta_response_2.model import AbsoluteResponseModel, LowRankPotential
from delta_response_2.training import (
    absolute_values,
    assert_finite_model,
    fit_censored_linear,
    linear_design,
    response_values,
    set_seed,
    train_absolute_fixed,
    train_potential_fixed,
)

from .constants import (
    ANALYTIC_METHODS,
    ARCHIVE_METHODS,
    ARCHIVE_ROOT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_METRICS,
    BOOTSTRAP_QUANTILES,
    BOOTSTRAP_SEED,
    CORRECTION_DIAGNOSTIC_COLUMNS,
    DEFAULT_OUTPUT_ROOT,
    DELTA2_METHODS,
    DETERMINISTIC_METHODS,
    DIAGNOSTIC_ARTIFACT_FILES,
    DIAGNOSTIC_FILES,
    DIRECTIONAL_PREDICTION_COLUMNS,
    DIRECTIONAL_ROWS,
    EVIDENCE_LABEL,
    EVIDENCE_STATUS,
    EXPECTED_ARCHIVE_HASHES,
    EXPECTED_INPUT_HASHES,
    FROZEN_COMPLETION_DIGEST,
    FROZEN_PLAN_SEAL_SHA256,
    FROZEN_RECIPE_SHA256,
    FROZEN_ROOT,
    FUTURE_EVIDENCE_REQUIREMENT,
    INPUT_PATHS,
    JOB_INH_PATH,
    JOBS_PATH,
    METHOD_CANDIDATES,
    METRIC_ESTIMATES,
    METRIC_FLAGS,
    NON_SELF_CLUSTERS,
    PAIR_MANIFEST_COLUMNS,
    PAIR_PATH,
    PRIMARY_DIRECTIONAL_ROWS,
    SEEDS,
    SEEDED_METHODS,
    SEED_PREDICTION_COLUMNS,
    SELECTED_CANDIDATE_ID,
    SELF_PAIR_ROWS,
    SHARED_POTENTIAL_METHODS,
    STATIC_EVALUATION_DIR,
    STATIC_PAIR_ROWS,
    TRAINING_SETTINGS,
    UTILS_DIR,
    VALID_APP_INHIBITOR_ROWS,
    INHIBITORS_PATH,
    completed_roster,
    prepared_roster,
)
from .integrity import (
    canonical_digest,
    hash_roster,
    read_canonical_json,
    require_regular_file,
    sha256_file,
    source_hashes,
    verify_exact_roster,
    write_csv,
    write_json,
)
from .synthetic import run_synthetic_self_checks


def _environment_record() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "device": "cpu",
    }


def require_cpu_only(device: str) -> torch.device:
    if device != "cpu":
        raise ValueError("static evaluation preparation accepts only --device cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "prepare requires CUDA to be hidden with CUDA_VISIBLE_DEVICES=\"\""
        )
    if torch.cuda.is_available():
        raise RuntimeError("prepare requires a CPU-only runtime with CUDA unavailable")
    return torch.device("cpu")


def require_fixed_root(root: Path) -> Path:
    candidate = root.expanduser().absolute()
    expected = DEFAULT_OUTPUT_ROOT.absolute()
    if candidate != expected:
        raise ValueError(
            f"the fixed protocol output root is {expected}; received {candidate}"
        )
    return candidate


def _verify_candidate_contract(
    recipe: dict[str, Any], candidate_manifest: pd.DataFrame
) -> None:
    expected_candidates = {
        METHOD_CANDIDATES["delta2_selected"]: {
            "method": "current_delta_ood_h8_e4",
            "family": "legacy_delta",
            "model_key": "delta:8:4",
            "parameters": {
                "aggregation": "ood",
                "embedding_dim": 4,
                "hidden_dim": 8,
            },
        },
        METHOD_CANDIDATES["delta2_profile_only_ood_h32_e16"]: {
            "method": "current_delta_ood_h32_e16",
            "family": "legacy_delta",
            "model_key": "delta:32:16",
            "parameters": {
                "aggregation": "ood",
                "embedding_dim": 16,
                "hidden_dim": 32,
            },
        },
        METHOD_CANDIDATES["delta2_censored_zero_h32_e16"]: {
            "method": "censored_correction_delta_h32_e16_zero_l10p0",
            "family": "censored_delta",
            "model_key": "delta:32:16",
            "parameters": {
                "embedding_dim": 16,
                "hidden_dim": 32,
                "prior_mode": "zero",
                "query_gamma": 1.0,
                "shrinkage": 10.0,
            },
        },
        METHOD_CANDIDATES["delta2_query_shrinkage_h32_e16"]: {
            "method": "censored_correction_delta_with_query_shrinkage_h32_e16_l10p0_g0p75",
            "family": "censored_delta_query_shrinkage",
            "model_key": "delta:32:16",
            "parameters": {
                "embedding_dim": 16,
                "hidden_dim": 32,
                "prior_mode": "population",
                "query_gamma": 0.75,
                "shrinkage": 10.0,
            },
        },
        METHOD_CANDIDATES["delta2_absolute_neural_h32_e16"]: {
            "method": "regularized_absolute_neural_h32_e16",
            "family": "absolute_neural",
            "model_key": "absolute:32:16",
            "parameters": {"embedding_dim": 16, "hidden_dim": 32},
        },
        METHOD_CANDIDATES["delta2_absolute_linear_r0p01"]: {
            "method": "regularized_absolute_linear_r0p01",
            "family": "absolute_linear",
            "model_key": "none",
            "parameters": {"ridge": 0.01},
        },
    }
    required_columns = {
        "candidate_id",
        "method",
        "family",
        "model_key",
        "parameters_json",
    }
    if not required_columns.issubset(candidate_manifest.columns):
        raise RuntimeError("frozen candidate manifest has an unexpected schema")
    indexed = candidate_manifest.set_index("candidate_id", verify_integrity=True)
    for candidate_id, expected in expected_candidates.items():
        if candidate_id not in indexed.index:
            raise RuntimeError(f"prespecified candidate is absent: {candidate_id}")
        row = indexed.loc[candidate_id]
        observed = {
            "method": str(row["method"]),
            "family": str(row["family"]),
            "model_key": str(row["model_key"]),
            "parameters": json.loads(str(row["parameters_json"])),
        }
        if observed != expected:
            raise RuntimeError(
                f"prespecified candidate semantics changed for {candidate_id}: {observed}"
            )

    resolved = recipe.get("resolved_experiment_config", {})
    expected_settings = {
        "feature_set": "base",
        "rank": 2,
        "optimizer_steps": 80,
        "batch_size": 128,
        "checkpoint_interval": 20,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "legacy_ood_temperature": 4.0,
        "legacy_ood_quantile": 0.9,
        "legacy_ood_fallback": "uniform",
        "bootstrap_samples": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seeds": list(SEEDS),
    }
    changed = {
        key: (resolved.get(key), value)
        for key, value in expected_settings.items()
        if resolved.get(key) != value
    }
    if changed:
        raise RuntimeError(f"frozen training settings changed: {changed}")
    if [32, 16] not in resolved.get("capacities", []):
        raise RuntimeError("frozen configuration no longer contains capacity 32/16")
    if 10.0 not in resolved.get("censored_shrinkage", []):
        raise RuntimeError("frozen configuration no longer contains shrinkage 10")
    if 0.75 not in resolved.get("query_gammas", []):
        raise RuntimeError("frozen configuration no longer contains query gamma 0.75")
    if 0.01 not in resolved.get("linear_ridge", []):
        raise RuntimeError("frozen configuration no longer contains linear ridge 0.01")


def verify_frozen_development_root() -> dict[str, Any]:
    require_regular_file(FROZEN_ROOT / "frozen_recipe.json")
    if sha256_file(FROZEN_ROOT / "frozen_recipe.json") != FROZEN_RECIPE_SHA256:
        raise RuntimeError("fixed frozen_recipe.json hash does not match")
    if sha256_file(FROZEN_ROOT / "seal.json") != FROZEN_PLAN_SEAL_SHA256:
        raise RuntimeError("fixed frozen plan seal hash does not match")
    completion = json.loads(
        (FROZEN_ROOT / "completion_seal.json").read_text(encoding="ascii")
    )
    if completion.get("sha256") != FROZEN_COMPLETION_DIGEST:
        raise RuntimeError("fixed frozen completion digest does not match")

    recipe = load_frozen_recipe(FROZEN_ROOT)
    report = json.loads((FROZEN_ROOT / "run_report.json").read_text(encoding="ascii"))
    required_report = {
        "status": "complete",
        "app_app_file_opened": False,
        "candidate_count": 53,
        "fold_schemes": ["profile", "mechanism"],
        "seeds": list(SEEDS),
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "scientific_acceptance": True,
        "valid_app_inhibitor_rows": VALID_APP_INHIBITOR_ROWS,
    }
    mismatches = {
        key: (report.get(key), expected)
        for key, expected in required_report.items()
        if report.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen development report mismatch: {mismatches}")
    if recipe.get("selected", {}).get("candidate_id") != SELECTED_CANDIDATE_ID:
        raise RuntimeError("frozen recipe selected a different candidate")
    if recipe.get("plan_seal_sha256") != FROZEN_PLAN_SEAL_SHA256:
        raise RuntimeError("frozen recipe is bound to a different plan seal")

    candidate_manifest = pd.read_csv(FROZEN_ROOT / "candidate_manifest.csv")
    _verify_candidate_contract(recipe, candidate_manifest)

    replay = predict_frozen_recipe(
        FROZEN_ROOT, "amg", "beatnik", device="cpu"
    )
    if not np.isclose(
        replay["latent_predicted_log_slowdown"],
        0.7722588152923945,
        rtol=0.0,
        atol=1e-10,
    ) or not np.isclose(
        replay["predicted_slowdown"],
        2.1646502808078,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("frozen profile-only replay checksum does not match")
    return recipe


def _verify_training_inputs() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    # Pair data is intentionally absent from this loop. Do not generalize this
    # into iteration over INPUT_PATHS during preparation.
    for name, path in (
        ("jobs.csv", JOBS_PATH),
        ("inhibitors.csv", INHIBITORS_PATH),
        ("job_inh.csv", JOB_INH_PATH),
    ):
        require_regular_file(path)
        actual = sha256_file(path)
        expected = EXPECTED_INPUT_HASHES[name]
        if actual != expected:
            raise RuntimeError(f"fixed training input hash mismatch: {name}")
        records[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": actual,
            "accessed": True,
            "verified": True,
        }
    records["pair.csv"] = {
        "path": str(PAIR_PATH),
        "expected_sha256": EXPECTED_INPUT_HASHES["pair.csv"],
        "observed_sha256": None,
        "accessed": False,
        "verified": False,
    }
    return records


def _full_profile_state(data: Any) -> tuple[
    ProfileScaler,
    ProfileScaler,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    scaler = ProfileScaler("base").fit(
        pd.concat([data.jobs, data.inhibitors], ignore_index=True)
    )
    distance_scaler = ProfileScaler("base").fit(data.inhibitors)
    frozen_checkpoint_dir = FROZEN_ROOT / "tasks" / "final_checkpoints"
    frozen_scaler = json.loads(
        (frozen_checkpoint_dir / "profile_scaler.json").read_text(encoding="ascii")
    )
    frozen_distance = json.loads(
        (frozen_checkpoint_dir / "distance_profile_scaler.json").read_text(
            encoding="ascii"
        )
    )
    if scaler.state_dict() != frozen_scaler:
        raise RuntimeError("final profile scaler differs from the frozen scaler policy")
    if distance_scaler.state_dict() != frozen_distance:
        raise RuntimeError("inhibitor-only distance scaler differs from the frozen scaler")
    job_profiles, inhibitor_profiles = profile_maps(
        data.jobs, data.inhibitors, scaler
    )
    distance_jobs, distance_inhibitors = profile_maps(
        data.jobs, data.inhibitors, distance_scaler
    )
    return (
        scaler,
        distance_scaler,
        job_profiles,
        inhibitor_profiles,
        distance_jobs,
        distance_inhibitors,
    )


def _row_arrays(
    rows: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    victims = np.stack([job_profiles[str(value)] for value in rows["job_id"]])
    aggressors = np.stack(
        [inhibitor_profiles[str(value)] for value in rows["inhib_id"]]
    )
    return victims, aggressors


def _build_ood_reference(
    data: Any,
    distance_inhibitors: dict[str, np.ndarray],
) -> tuple[np.ndarray, OODCalibration]:
    inhibitor_manifest = pd.read_csv(FROZEN_ROOT / "inhibitor_manifest.csv")
    inhibitor_manifest["inhib_id"] = inhibitor_manifest["inhib_id"].astype(str)
    rows = data.responses.merge(
        inhibitor_manifest[["inhib_id", "profile_block"]],
        on="inhib_id",
        how="left",
        validate="many_to_one",
    )
    if rows["profile_block"].isna().any():
        raise RuntimeError("frozen profile blocks do not cover every inhibitor row")
    reference = training_distance_reference(
        rows, distance_inhibitors, block_column="profile_block"
    )
    calibration = calibrate_ood(
        reference, float(TRAINING_SETTINGS["legacy_ood_quantile"])
    )
    if len(reference) != VALID_APP_INHIBITOR_ROWS:
        raise RuntimeError("diagnostic OOD reference must contain exactly 2623 values")
    expected = {
        "threshold": 1.7235320806503296,
        "width": 1.2320533990859985,
        "reference_count": VALID_APP_INHIBITOR_ROWS,
    }
    if (
        not np.isclose(calibration.threshold, expected["threshold"], atol=1e-12)
        or not np.isclose(calibration.width, expected["width"], atol=1e-12)
        or calibration.reference_count != expected["reference_count"]
    ):
        raise RuntimeError("diagnostic OOD calibration differs from the frozen policy")
    frozen_reference = pd.read_csv(
        FROZEN_ROOT
        / "tasks"
        / "final_checkpoints"
        / "deployment_ood_reference.csv"
    )["distance"].to_numpy(dtype=float)
    if reference.shape != frozen_reference.shape or not np.allclose(
        reference, frozen_reference, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("recomputed OOD reference differs from the frozen reference")
    return reference, calibration


def _save_torch_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_potential_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], LowRankPotential]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LowRankPotential(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    assert_finite_model(model)
    return checkpoint, model


def _load_absolute_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], AbsoluteResponseModel]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AbsoluteResponseModel(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    assert_finite_model(model)
    return checkpoint, model


def _validate_potential_replay(
    model: LowRankPotential,
    prior: float,
    data: Any,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    distance_jobs: dict[str, np.ndarray],
    distance_inhibitors: dict[str, np.ndarray],
    calibration: OODCalibration,
    device: torch.device,
) -> None:
    legacy_ood = LegacyOOD(
        float(TRAINING_SETTINGS["legacy_ood_temperature"]),
        float(TRAINING_SETTINGS["legacy_ood_quantile"]),
        str(TRAINING_SETTINGS["legacy_ood_fallback"]),
    )
    all_predictions: list[float] = []
    for victim_id in data.jobs["job_id"].astype(str):
        anchors = data.responses[data.responses["job_id"].eq(victim_id)]
        observed = anchors["log_slowdown"].to_numpy(dtype=float)
        anchor_profiles = np.stack(
            [inhibitor_profiles[str(value)] for value in anchors["inhib_id"]]
        )
        anchor_distances = np.stack(
            [distance_inhibitors[str(value)] for value in anchors["inhib_id"]]
        )
        query_ids = data.jobs["job_id"].astype(str).tolist()
        query_profiles = np.stack([job_profiles[value] for value in query_ids])
        potentials = response_values(
            model,
            job_profiles[victim_id],
            np.vstack([anchor_profiles, query_profiles]),
            device,
        )
        support = potentials[: len(anchors)]
        query = potentials[len(anchors) :]
        for index, query_id in enumerate(query_ids):
            distances = np.linalg.norm(
                anchor_distances - distance_jobs[query_id][None, :], axis=1
            )
            legacy, _ = legacy_exact_corrections(
                observed,
                support,
                distances,
                calibration,
                ood=legacy_ood,
            )["ood"]
            all_predictions.append(float(query[index] + legacy))
        zero, zero_fit, _ = censored_potential_predictions(
            observed,
            support,
            query,
            shrinkage=float(TRAINING_SETTINGS["censored_shrinkage"]),
            prior=0.0,
            query_gamma=1.0,
        )
        shrunk, shrunk_fit, _ = censored_potential_predictions(
            observed,
            support,
            query,
            shrinkage=float(TRAINING_SETTINGS["censored_shrinkage"]),
            prior=prior,
            query_gamma=float(TRAINING_SETTINGS["query_gamma"]),
        )
        if not zero_fit.converged or not shrunk_fit.converged:
            raise RuntimeError("diagnostic censored correction replay did not converge")
        all_predictions.extend(zero.tolist())
        all_predictions.extend(shrunk.tolist())
    if not np.isfinite(all_predictions).all():
        raise FloatingPointError("diagnostic potential profile replay is non-finite")


def _method_panel() -> list[dict[str, object]]:
    roles = {
        "delta2_selected": "primary_new_model",
        "delta2_constant_1": "pair_independent_baseline",
        "delta2_victim_inhibitor_median": "pair_independent_baseline",
        "delta2_nearest_anchor": "pair_independent_baseline",
        "delta2_profile_only_ood_h32_e16": "secondary_design_diagnostic",
        "delta2_censored_zero_h32_e16": "secondary_design_diagnostic",
        "delta2_query_shrinkage_h32_e16": "secondary_design_diagnostic",
        "delta2_absolute_neural_h32_e16": "secondary_design_diagnostic",
        "delta2_absolute_linear_r0p01": "secondary_design_diagnostic",
    }
    parameters: dict[str, dict[str, object]] = {
        "delta2_selected": {
            "family": "legacy_exact_anchor_ood_delta",
            "checkpoint_source": "frozen_compact_v1_h8_e4",
            "anchor_policy": "all_uncensored_victim_anchors",
            "temperature": 4.0,
            "ood_quantile": 0.90,
            "ood_fallback": "uniform_exact_anchor_residual_mean",
        },
        "delta2_constant_1": {
            "family": "analytic",
            "latent_prediction": 0.0,
        },
        "delta2_victim_inhibitor_median": {
            "family": "analytic",
            "anchor_policy": "all_uncensored_victim_app_inhibitor_rows",
            "estimator": "median_observed_log_slowdown",
        },
        "delta2_nearest_anchor": {
            "family": "analytic",
            "anchor_policy": "all_uncensored_victim_app_inhibitor_rows",
            "estimator": "observed_log_slowdown_of_nearest_anchor",
            "distance_scaler": "frozen_inhibitor_only_base_feature_scaler",
        },
        "delta2_profile_only_ood_h32_e16": {
            "family": "legacy_exact_anchor_ood_delta",
            "hidden_dim": 32,
            "embedding_dim": 16,
            "rank": 2,
            "temperature": 4.0,
            "ood_quantile": 0.90,
            "ood_fallback": "uniform_exact_anchor_residual_mean",
        },
        "delta2_censored_zero_h32_e16": {
            "family": "all_anchor_censored_delta",
            "hidden_dim": 32,
            "embedding_dim": 16,
            "rank": 2,
            "shrinkage": 10.0,
            "prior_mode": "zero",
            "query_gamma": 1.0,
        },
        "delta2_query_shrinkage_h32_e16": {
            "family": "all_anchor_censored_delta_query_shrinkage",
            "hidden_dim": 32,
            "embedding_dim": 16,
            "rank": 2,
            "shrinkage": 10.0,
            "prior_mode": "seed_specific_population",
            "query_gamma": 0.75,
        },
        "delta2_absolute_neural_h32_e16": {
            "family": "censored_absolute_neural",
            "hidden_dim": 32,
            "embedding_dim": 16,
            "rank": 2,
        },
        "delta2_absolute_linear_r0p01": {
            "family": "censored_absolute_linear",
            "ridge": 0.01,
            "deterministic": True,
        },
    }
    return [
        {
            "saved_method": method,
            "candidate_id": METHOD_CANDIDATES[method],
            "role": roles[method],
            "seeded": method in SEEDED_METHODS,
            "parameters": parameters[method],
        }
        for method in DELTA2_METHODS
    ]


def _candidate_artifact_mapping() -> dict[str, dict[str, object]]:
    selected_artifacts = [
        str(
            FROZEN_ROOT
            / "tasks"
            / "final_checkpoints"
            / f"delta_seed{seed}.pt"
        )
        for seed in SEEDS
    ]
    potential_artifacts = [f"delta_h32_e16_seed{seed}.pt" for seed in SEEDS]
    absolute_artifacts = [
        f"absolute_neural_h32_e16_seed{seed}.pt" for seed in SEEDS
    ]
    mapping: dict[str, dict[str, object]] = {}
    for method in DELTA2_METHODS:
        if method == "delta2_selected":
            artifacts = selected_artifacts
            source = "frozen_compact_v1"
        elif method in SHARED_POTENTIAL_METHODS:
            artifacts = potential_artifacts
            source = "shared_diagnostic_potential_fit"
        elif method == "delta2_absolute_neural_h32_e16":
            artifacts = absolute_artifacts
            source = "diagnostic_absolute_neural_fit"
        elif method == "delta2_absolute_linear_r0p01":
            artifacts = ["absolute_linear_r0p01.json"]
            source = "diagnostic_censored_linear_fit"
        else:
            artifacts = []
            source = "analytic_pair_independent"
        mapping[method] = {
            "candidate_id": METHOD_CANDIDATES[method],
            "source": source,
            "artifacts": artifacts,
        }
    return mapping


def _evaluation_plan(
    source_records: dict[str, str],
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_status": EVIDENCE_STATUS,
        "future_evidence_requirement": FUTURE_EVIDENCE_REQUIREMENT,
        "evaluation_method": "random_split",
        "evaluation_output_root": str(DEFAULT_OUTPUT_ROOT),
        "training_applications": "all_ten_sealed_applications",
        "unknown_applications": [],
        "valid_app_inhibitor_rows": VALID_APP_INHIBITOR_ROWS,
        "absolute_input_paths": {
            name: str(INPUT_PATHS[name]) for name in EXPECTED_INPUT_HASHES
        },
        "expected_input_hashes": EXPECTED_INPUT_HASHES,
        "frozen_model": {
            "root": str(FROZEN_ROOT),
            "selected_candidate_id": SELECTED_CANDIDATE_ID,
            "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
            "plan_seal_sha256": FROZEN_PLAN_SEAL_SHA256,
            "completion_seal_digest": FROZEN_COMPLETION_DIGEST,
        },
        "method_panel": _method_panel(),
        "archived_comparator_panel": [
            {
                "archive_method": method,
                "comparison_alias": f"delta1_{method}",
                "anchor_budget": "all",
                "retrained": False,
            }
            for method in ARCHIVE_METHODS
        ],
        "candidate_to_artifact_mapping": _candidate_artifact_mapping(),
        "seed_ensemble": {
            "seeds": list(SEEDS),
            "rule": "mean_latent_across_seeds_then_clip_once_at_zero",
            "standard_deviation_ddof": 0,
        },
        "schemas": {
            "seed_predictions_required_columns": list(SEED_PREDICTION_COLUMNS),
            "directional_predictions_required_columns": list(
                DIRECTIONAL_PREDICTION_COLUMNS
            ),
            "correction_diagnostic_columns": list(
                CORRECTION_DIAGNOSTIC_COLUMNS
            ),
            "pair_manifest_columns": list(PAIR_MANIFEST_COLUMNS),
        },
        "pair_contract": {
            "unordered_rows": STATIC_PAIR_ROWS,
            "self_pair_rows": SELF_PAIR_ROWS,
            "non_self_pair_clusters": NON_SELF_CLUSTERS,
            "directional_rows": DIRECTIONAL_ROWS,
            "primary_non_self_directional_rows": PRIMARY_DIRECTIONAL_ROWS,
            "pair_row_id": "original_validated_row_position_0_through_54",
            "pair_cluster_id": "pair_row_id",
            "direction_order": ["A", "B"],
            "self_directions_preserved": True,
            "random_split_name_does_not_request_a_new_split": True,
        },
        "scopes": {
            "primary": "90_non_self_directional_rows",
            "secondary": "all_110_directional_rows",
            "self_diagnostic": "20_self_directional_rows",
            "self_averaged": "10_raw_space_averaged_self_pairs",
            "per_victim": "9_non_self_rows_per_victim",
            "macro_victim": "equal_average_of_ten_victim_estimates",
            "per_seed": "90_non_self_rows_per_seeded_method_and_seed",
            "ensemble": "90_non_self_latent_space_ensemble_rows_per_method",
        },
        "metrics": {
            "primary_metric": "non_self_log_mae",
            "all_point_estimates": list(METRIC_ESTIMATES),
            "defined_flags": list(METRIC_FLAGS),
            "bootstrap_estimates": list(BOOTSTRAP_METRICS),
            "point_diagnostics": [
                "overprediction_count",
                "underprediction_count",
                "calibration_defined",
                "spearman_defined",
                "pearson_defined",
            ],
            "undefined_estimate": "json_null_or_blank_csv_with_explicit_defined_flag",
            "range_tolerance": 1e-12,
            "seed_stability_summaries": [
                "seed_mean",
                "seed_population_standard_deviation_ddof_0",
                "seed_best",
                "seed_worst",
            ],
        },
        "bootstrap": {
            "unit": "original_unordered_pair_cluster_id",
            "cluster_count": NON_SELF_CLUSTERS,
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "draw_size": NON_SELF_CLUSTERS,
            "directions_per_drawn_cluster": 2,
            "shared_draw_matrix": True,
            "interval": "percentile",
            "quantiles": list(BOOTSTRAP_QUANTILES),
            "paired_difference": "first_method_minus_second_method",
        },
        "archived_delta_response_1": {
            "root": str(ARCHIVE_ROOT),
            "expected_paths": {
                name: str(ARCHIVE_ROOT / name) for name in EXPECTED_ARCHIVE_HASHES
            },
            "expected_hashes": EXPECTED_ARCHIVE_HASHES,
            "access_order": "only_after_new_prediction_artifacts_are_hashed",
        },
        "training_settings": TRAINING_SETTINGS,
        "diagnostic_checkpoint_roster": [
            f"diagnostic_checkpoints/{name}" for name in DIAGNOSTIC_FILES
        ],
        "required_complete_output_roster": sorted(completed_roster()),
        "evaluation_source_hashes": source_records,
        "diagnostic_artifact_hashes": artifact_hashes,
        "app_app_outcome_opened": False,
        "no_pair_access_declaration": (
            "No App-App outcome file was opened, resolved, statted, or hashed during "
            "preparation."
        ),
    }


def _preparation_seal_payload(
    root: Path,
    source_records: dict[str, str],
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    checkpoint_dir = root / "diagnostic_checkpoints"
    return {
        "evaluation_plan_sha256": sha256_file(root / "evaluation_plan.json"),
        "input_hashes_sha256": sha256_file(root / "input_hashes.json"),
        "artifact_hashes_metadata_sha256": sha256_file(
            checkpoint_dir / "artifact_hashes.json"
        ),
        "preparation_state_sha256": sha256_file(
            checkpoint_dir / "preparation_state.json"
        ),
        "evaluation_source_hashes": source_records,
        "diagnostic_artifact_hashes": artifact_hashes,
        "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "frozen_plan_seal_sha256": FROZEN_PLAN_SEAL_SHA256,
        "frozen_completion_digest": FROZEN_COMPLETION_DIGEST,
        "exact_preparation_roster": sorted(prepared_roster()),
        "evidence_label": EVIDENCE_LABEL,
        "app_app_outcome_opened": False,
    }


def prepare(root: Path = DEFAULT_OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, object]:
    root = require_fixed_root(root)
    target_device = require_cpu_only(device)
    recipe = verify_frozen_development_root()
    input_records = _verify_training_inputs()
    initial_source_records = source_hashes(STATIC_EVALUATION_DIR)
    synthetic_checks = run_synthetic_self_checks()
    data = load_training_data(JOBS_PATH, INHIBITORS_PATH, JOB_INH_PATH)
    if len(data.responses) != VALID_APP_INHIBITOR_ROWS:
        raise RuntimeError("valid App-Inhibitor fitting row count is not 2623")
    if len(data.jobs) != 10:
        raise RuntimeError("static protocol requires exactly ten application profiles")

    if root.is_symlink() or root.exists():
        raise FileExistsError(f"prepare requires a new absent evaluation root: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise RuntimeError(f"evaluation-root parent is invalid: {root.parent}")
    root.mkdir()
    checkpoint_dir = root / "diagnostic_checkpoints"
    checkpoint_dir.mkdir()

    (
        scaler,
        distance_scaler,
        job_profiles,
        inhibitor_profiles,
        distance_jobs,
        distance_inhibitors,
    ) = _full_profile_state(data)
    write_json(checkpoint_dir / "profile_scaler.json", scaler.state_dict())
    write_json(
        checkpoint_dir / "distance_profile_scaler.json",
        distance_scaler.state_dict(),
    )
    ood_reference, ood_calibration = _build_ood_reference(
        data, distance_inhibitors
    )
    write_csv(
        checkpoint_dir / "delta_h32_e16_ood_reference.csv",
        pd.DataFrame(
            {
                "evidence_label": EVIDENCE_LABEL,
                "distance": ood_reference,
            }
        ),
    )

    potential_curves: list[dict[str, object]] = []
    potential_priors: dict[str, float] = {}
    for seed in SEEDS:
        set_seed(seed)
        model = LowRankPotential(
            scaler.output_dim,
            int(TRAINING_SETTINGS["hidden_dim"]),
            int(TRAINING_SETTINGS["embedding_dim"]),
            int(TRAINING_SETTINGS["rank"]),
        )
        result = train_potential_fixed(
            model,
            data.responses,
            job_profiles,
            inhibitor_profiles,
            seed=seed,
            optimizer_steps=int(TRAINING_SETTINGS["optimizer_steps"]),
            batch_size=int(TRAINING_SETTINGS["batch_size"]),
            learning_rate=float(TRAINING_SETTINGS["learning_rate"]),
            weight_decay=float(TRAINING_SETTINGS["weight_decay"]),
            checkpoint_interval=int(TRAINING_SETTINGS["checkpoint_interval"]),
            device=target_device,
        )
        prior, prior_values = population_correction_prior(
            model,
            data.responses,
            job_profiles,
            inhibitor_profiles,
            target_device,
            query_gamma=float(TRAINING_SETTINGS["query_gamma"]),
        )
        if not np.isfinite(prior) or not np.isfinite(prior_values).all():
            raise FloatingPointError("seed-specific population correction prior is non-finite")
        potential_priors[str(seed)] = float(prior)
        path = checkpoint_dir / f"delta_h32_e16_seed{seed}.pt"
        _save_torch_checkpoint(
            path,
            {
                "seed": seed,
                "state_dict": model.state_dict(),
                "input_dim": scaler.output_dim,
                "hidden_dim": int(TRAINING_SETTINGS["hidden_dim"]),
                "embedding_dim": int(TRAINING_SETTINGS["embedding_dim"]),
                "rank": int(TRAINING_SETTINGS["rank"]),
                "shared_candidate_ids": {
                    method: METHOD_CANDIDATES[method]
                    for method in SHARED_POTENTIAL_METHODS
                },
                "population_correction_prior": float(prior),
                "population_correction_prior_query_gamma": float(
                    TRAINING_SETTINGS["query_gamma"]
                ),
                "population_correction_values": [float(value) for value in prior_values],
                "training_settings": {
                    key: TRAINING_SETTINGS[key]
                    for key in (
                        "optimizer_steps",
                        "batch_size",
                        "learning_rate",
                        "weight_decay",
                        "checkpoint_interval",
                    )
                },
            },
        )
        checkpoint, replay_model = _load_potential_checkpoint(path, target_device)
        if int(checkpoint["seed"]) != seed:
            raise RuntimeError("saved potential checkpoint seed binding changed")
        _validate_potential_replay(
            replay_model,
            float(checkpoint["population_correction_prior"]),
            data,
            job_profiles,
            inhibitor_profiles,
            distance_jobs,
            distance_inhibitors,
            ood_calibration,
            target_device,
        )
        potential_curves.extend(
            {
                "evidence_label": EVIDENCE_LABEL,
                "seed": seed,
                "sampled_examples": int(result["sampled_examples"]),
                "informative_examples": int(result["informative_examples"]),
                **record,
            }
            for record in result["history"]
        )
    write_csv(
        checkpoint_dir / "delta_h32_e16_training_curves.csv",
        pd.DataFrame(potential_curves),
    )

    absolute_curves: list[dict[str, object]] = []
    app_profiles = np.stack(list(job_profiles.values()))
    app_victims = np.repeat(app_profiles, len(app_profiles), axis=0)
    app_aggressors = np.tile(app_profiles, (len(app_profiles), 1))
    for seed in SEEDS:
        set_seed(seed)
        model = AbsoluteResponseModel(
            scaler.output_dim,
            int(TRAINING_SETTINGS["hidden_dim"]),
            int(TRAINING_SETTINGS["embedding_dim"]),
            int(TRAINING_SETTINGS["rank"]),
        )
        result = train_absolute_fixed(
            model,
            data.responses,
            job_profiles,
            inhibitor_profiles,
            seed=seed,
            optimizer_steps=int(TRAINING_SETTINGS["optimizer_steps"]),
            batch_size=int(TRAINING_SETTINGS["batch_size"]),
            learning_rate=float(TRAINING_SETTINGS["learning_rate"]),
            weight_decay=float(TRAINING_SETTINGS["weight_decay"]),
            checkpoint_interval=int(TRAINING_SETTINGS["checkpoint_interval"]),
            device=target_device,
        )
        path = checkpoint_dir / f"absolute_neural_h32_e16_seed{seed}.pt"
        _save_torch_checkpoint(
            path,
            {
                "seed": seed,
                "state_dict": model.state_dict(),
                "input_dim": scaler.output_dim,
                "hidden_dim": int(TRAINING_SETTINGS["hidden_dim"]),
                "embedding_dim": int(TRAINING_SETTINGS["embedding_dim"]),
                "rank": int(TRAINING_SETTINGS["rank"]),
                "candidate_id": METHOD_CANDIDATES[
                    "delta2_absolute_neural_h32_e16"
                ],
                "training_settings": {
                    key: TRAINING_SETTINGS[key]
                    for key in (
                        "optimizer_steps",
                        "batch_size",
                        "learning_rate",
                        "weight_decay",
                        "checkpoint_interval",
                    )
                },
            },
        )
        checkpoint, replay_model = _load_absolute_checkpoint(path, target_device)
        if int(checkpoint["seed"]) != seed:
            raise RuntimeError("saved absolute-neural checkpoint seed binding changed")
        replay = absolute_values(
            replay_model, app_victims, app_aggressors, target_device
        )
        if not np.isfinite(replay).all():
            raise FloatingPointError("absolute-neural profile replay is non-finite")
        absolute_curves.extend(
            {
                "evidence_label": EVIDENCE_LABEL,
                "seed": seed,
                "sampled_examples": int(result["sampled_examples"]),
                "informative_examples": int(result["informative_examples"]),
                **record,
            }
            for record in result["history"]
        )
    write_csv(
        checkpoint_dir / "absolute_neural_h32_e16_training_curves.csv",
        pd.DataFrame(absolute_curves),
    )

    train_victims, train_aggressors = _row_arrays(
        data.responses, job_profiles, inhibitor_profiles
    )
    coefficients, linear_fit = fit_censored_linear(
        linear_design(train_victims, train_aggressors),
        data.responses["log_slowdown"].to_numpy(dtype=float),
        ridge=float(TRAINING_SETTINGS["linear_ridge"]),
    )
    if not linear_fit["converged"] or not np.isfinite(coefficients).all():
        raise RuntimeError("deterministic censored linear diagnostic did not converge")
    linear_replay = linear_design(app_victims, app_aggressors) @ coefficients
    if not np.isfinite(linear_replay).all():
        raise FloatingPointError("deterministic linear profile replay is non-finite")
    write_json(
        checkpoint_dir / "absolute_linear_r0p01.json",
        {
            "evidence_label": EVIDENCE_LABEL,
            "candidate_id": METHOD_CANDIDATES["delta2_absolute_linear_r0p01"],
            "coefficients": coefficients.tolist(),
            "fit": linear_fit,
            "ridge": float(TRAINING_SETTINGS["linear_ridge"]),
            "design": "intercept_victim_aggressor_elementwise_interaction",
        },
    )

    artifact_hashes = hash_roster(checkpoint_dir, DIAGNOSTIC_ARTIFACT_FILES)
    if set(artifact_hashes) != set(DIAGNOSTIC_ARTIFACT_FILES) or len(artifact_hashes) != 16:
        raise RuntimeError("diagnostic artifact hash roster must contain exactly 16 files")
    write_json(checkpoint_dir / "artifact_hashes.json", artifact_hashes)
    preparation_state = {
        "status": "prepared",
        "evidence_label": EVIDENCE_LABEL,
        "app_app_outcome_opened": False,
        "valid_app_inhibitor_rows": len(data.responses),
        "application_count": len(data.jobs),
        "inhibitor_count": len(data.inhibitors),
        "device": "cpu",
        "environment": _environment_record(),
        "training_settings": TRAINING_SETTINGS,
        "candidate_to_artifact_mapping": _candidate_artifact_mapping(),
        "profile_scaler_file": "profile_scaler.json",
        "distance_profile_scaler_file": "distance_profile_scaler.json",
        "distance_scaler_fit_population": "all_inhibitor_profiles_only",
        "ood_reference": {
            "file": "delta_h32_e16_ood_reference.csv",
            "source": "nested_profile_blocks_over_all_valid_app_inhibitor_rows",
            "threshold": ood_calibration.threshold,
            "width": ood_calibration.width,
            "quantile": ood_calibration.quantile,
            "reference_count": ood_calibration.reference_count,
            "temperature": float(TRAINING_SETTINGS["legacy_ood_temperature"]),
            "fallback": str(TRAINING_SETTINGS["legacy_ood_fallback"]),
        },
        "population_correction_prior_query_gamma": float(
            TRAINING_SETTINGS["query_gamma"]
        ),
        "population_correction_priors_by_seed": potential_priors,
        "artifact_hashes": artifact_hashes,
        "synthetic_self_checks": synthetic_checks,
        "frozen_selected_recipe_candidate_id": recipe["selected"]["candidate_id"],
    }
    write_json(checkpoint_dir / "preparation_state.json", preparation_state)

    source_records = source_hashes(STATIC_EVALUATION_DIR)
    if source_records != initial_source_records:
        raise RuntimeError("evaluation-only source changed during preparation")
    write_json(
        root / "input_hashes.json",
        {
            "evidence_label": EVIDENCE_LABEL,
            "app_app_outcome_opened": False,
            "inputs": input_records,
        },
    )
    write_json(
        root / "evaluation_plan.json",
        _evaluation_plan(source_records, artifact_hashes),
    )
    payload = _preparation_seal_payload(
        root, source_records, artifact_hashes
    )
    write_json(
        root / "evaluation_preparation_seal.json",
        {**payload, "sha256": canonical_digest(payload)},
    )
    verify_preparation(root, require_prepared_roster=True)
    return {
        "status": "prepared",
        "root": str(root),
        "evidence_label": EVIDENCE_LABEL,
        "app_app_outcome_opened": False,
        "diagnostic_artifact_count": len(artifact_hashes),
    }


def verify_preparation(
    root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    require_prepared_roster: bool = False,
) -> dict[str, Any]:
    root = require_fixed_root(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"prepared evaluation root is absent or invalid: {root}")
    if require_prepared_roster:
        verify_exact_roster(root, prepared_roster())

    recipe = verify_frozen_development_root()
    current_inputs = _verify_training_inputs()
    plan = read_canonical_json(root / "evaluation_plan.json")
    input_state = read_canonical_json(root / "input_hashes.json")
    checkpoint_dir = root / "diagnostic_checkpoints"
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RuntimeError("diagnostic checkpoint directory is absent or a symlink")
    artifact_hashes = read_canonical_json(checkpoint_dir / "artifact_hashes.json")
    state = read_canonical_json(checkpoint_dir / "preparation_state.json")
    seal = read_canonical_json(root / "evaluation_preparation_seal.json")

    if set(artifact_hashes) != set(DIAGNOSTIC_ARTIFACT_FILES) or len(artifact_hashes) != 16:
        raise RuntimeError("prepared diagnostic artifact roster is not exactly 16 files")
    actual_diagnostic_files = {
        path.name
        for path in checkpoint_dir.iterdir()
        if not path.is_symlink() and path.is_file()
    }
    if actual_diagnostic_files != set(DIAGNOSTIC_FILES):
        raise RuntimeError(
            "diagnostic checkpoint roster changed: "
            f"missing={sorted(set(DIAGNOSTIC_FILES) - actual_diagnostic_files)}, "
            f"unexpected={sorted(actual_diagnostic_files - set(DIAGNOSTIC_FILES))}"
        )
    for path in checkpoint_dir.iterdir():
        require_regular_file(path)
    observed_artifacts = hash_roster(checkpoint_dir, DIAGNOSTIC_ARTIFACT_FILES)
    if observed_artifacts != artifact_hashes:
        raise RuntimeError("diagnostic artifact hashes changed after preparation")

    current_sources = source_hashes(STATIC_EVALUATION_DIR)
    if plan.get("evaluation_source_hashes") != current_sources:
        raise RuntimeError("evaluation-only source changed after preparation")
    expected_plan = _evaluation_plan(current_sources, artifact_hashes)
    if plan != expected_plan:
        changed_keys = sorted(
            key
            for key in set(plan) | set(expected_plan)
            if plan.get(key) != expected_plan.get(key)
        )
        raise RuntimeError(f"prepared evaluation plan changed: {changed_keys}")
    if plan.get("diagnostic_artifact_hashes") != artifact_hashes:
        raise RuntimeError("evaluation plan diagnostic hashes do not match")
    if plan.get("method_panel") != _method_panel():
        raise RuntimeError("prepared method panel differs from the exact contract")
    if plan.get("candidate_to_artifact_mapping") != _candidate_artifact_mapping():
        raise RuntimeError("prepared candidate-to-artifact mapping changed")
    if plan.get("expected_input_hashes") != EXPECTED_INPUT_HASHES:
        raise RuntimeError("prepared expected input hashes changed")
    if plan.get("absolute_input_paths") != {
        name: str(INPUT_PATHS[name]) for name in EXPECTED_INPUT_HASHES
    }:
        raise RuntimeError("prepared absolute input paths changed")
    if plan.get("app_app_outcome_opened") is not False:
        raise RuntimeError("preparation incorrectly declares App-App access")
    frozen = plan.get("frozen_model", {})
    if frozen != {
        "root": str(FROZEN_ROOT),
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "plan_seal_sha256": FROZEN_PLAN_SEAL_SHA256,
        "completion_seal_digest": FROZEN_COMPLETION_DIGEST,
    }:
        raise RuntimeError("prepared frozen-root identifiers changed")

    if input_state.get("app_app_outcome_opened") is not False:
        raise RuntimeError("input state incorrectly declares App-App access")
    if input_state.get("inputs") != current_inputs:
        raise RuntimeError("prepared training-input state changed")
    pair_record = input_state["inputs"]["pair.csv"]
    if pair_record.get("accessed") is not False or pair_record.get("observed_sha256") is not None:
        raise RuntimeError("preparation state indicates forbidden pair access")

    if state.get("status") != "prepared" or state.get("app_app_outcome_opened") is not False:
        raise RuntimeError("diagnostic preparation state is invalid")
    if state.get("training_settings") != TRAINING_SETTINGS:
        raise RuntimeError("diagnostic training settings changed")
    if state.get("candidate_to_artifact_mapping") != _candidate_artifact_mapping():
        raise RuntimeError("diagnostic candidate mapping changed")
    if state.get("artifact_hashes") != artifact_hashes:
        raise RuntimeError("diagnostic preparation state has different artifact hashes")
    if state.get("synthetic_self_checks", {}).get("status") != "passed":
        raise RuntimeError("synthetic protocol checks were not sealed as passed")
    if sorted(map(int, state.get("population_correction_priors_by_seed", {}))) != list(SEEDS):
        raise RuntimeError("seed-specific population priors are incomplete")

    payload = _preparation_seal_payload(root, current_sources, artifact_hashes)
    if {key: seal.get(key) for key in payload} != payload:
        raise RuntimeError("evaluation preparation seal binding is invalid")
    if seal.get("sha256") != canonical_digest(payload):
        raise RuntimeError("evaluation preparation seal digest is invalid")
    return {
        "recipe": recipe,
        "plan": plan,
        "preparation_state": state,
        "artifact_hashes": artifact_hashes,
        "source_hashes": current_sources,
        "preparation_seal_sha256": sha256_file(
            root / "evaluation_preparation_seal.json"
        ),
    }
