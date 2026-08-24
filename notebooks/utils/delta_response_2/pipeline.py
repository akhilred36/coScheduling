"""Random-split-only App-Inhibitor development search and recipe freeze."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor

from .censoring import fit_censored_intercept
from .config import ExperimentConfig, candidate_registry
from .data import (
    ProfileScaler,
    TrainingData,
    load_training_data,
    log_to_slowdown,
    observe_log_slowdown,
    profile_maps,
)
from .inference import (
    LegacyOOD,
    OODCalibration,
    calibrate_ood,
    censored_potential_predictions,
    legacy_exact_corrections,
    population_correction_prior,
    training_distance_reference,
)
from .manifests import (
    sha256_file,
    verify_completion,
    verify_plan,
    write_completion_seal,
    write_json,
    write_output_hashes,
)
from .metrics import (
    METRICS,
    ensemble_metric_table,
    ensemble_predictions,
    metric_tables,
    paired_fold_bootstrap,
)
from .model import AbsoluteResponseModel, LowRankPotential
from .training import (
    absolute_values,
    assert_finite_model,
    fit_censored_linear,
    linear_design,
    response_values,
    set_seed,
    train_absolute_fixed,
    train_potential_fixed,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_planned_data(root: Path) -> tuple[TrainingData, pd.DataFrame]:
    inputs = json.loads((root / "input_hashes.json").read_text())
    data = load_training_data(
        inputs["jobs_csv"]["path"],
        inputs["inhibitors_csv"]["path"],
        inputs["job_inh_csv"]["path"],
    )
    inhibitor_manifest = pd.read_csv(root / "inhibitor_manifest.csv")
    inhibitor_manifest["inhib_id"] = inhibitor_manifest["inhib_id"].astype(str)
    expected = set(data.inhibitors["inhib_id"].astype(str))
    if set(inhibitor_manifest["inhib_id"]) != expected:
        raise RuntimeError("Inhibitor manifest does not match the sealed profiles")
    return data, inhibitor_manifest


def _load_execution_membership(root: Path, seed: int) -> pd.DataFrame:
    columns = [
        "fold_scheme",
        "victim_id",
        "heldout_block",
        "seed",
        "response_id",
        "role",
    ]
    selected = []
    for chunk in pd.read_csv(root / "fold_manifest.csv", usecols=columns, chunksize=100_000):
        rows = chunk[chunk["seed"] == seed]
        if len(rows):
            selected.append(rows)
    if not selected:
        raise RuntimeError("Sealed fold manifest has no execution membership")
    manifest = pd.concat(selected, ignore_index=True)
    if manifest.duplicated(
        ["fold_scheme", "victim_id", "heldout_block", "response_id"]
    ).any():
        raise RuntimeError("Sealed fold manifest has duplicate response membership")
    return manifest


def _concatenate_csv_shards(paths: list[Path], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for index, path in enumerate(paths):
            with path.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, destination)
    temporary.replace(output)


def _row_arrays(
    rows: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([job_profiles[str(value)] for value in rows["job_id"]]),
        np.stack([inhibitor_profiles[str(value)] for value in rows["inhib_id"]]),
    )


class _CSVWriter:
    def __init__(self, path: Path):
        self.path = path
        self.wrote = False

    def write(self, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
        if isinstance(rows, list):
            if not rows:
                return
            frame = pd.DataFrame(rows)
        else:
            frame = rows
            if frame.empty:
                return
        frame.to_csv(self.path, mode="a", header=not self.wrote, index=False)
        self.wrote = True


def _append_predictions(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    queries: pd.DataFrame,
    latent: np.ndarray | float,
    *,
    fold_scheme: str,
    victim_id: str,
    heldout_block: int,
    seed: int,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    values = np.asarray(latent, dtype=float)
    if values.ndim == 0:
        values = np.full(len(queries), float(values))
    if values.shape != (len(queries),) or not np.isfinite(values).all():
        raise ValueError(f"Candidate {candidate['method']} produced invalid predictions")
    observed = observe_log_slowdown(values)
    slowdowns = log_to_slowdown(observed)
    common_diagnostics = diagnostics or {}
    for index, query in enumerate(queries.itertuples(index=False)):
        record = {
            "candidate_id": candidate["candidate_id"],
            "method": candidate["method"],
            "family": candidate["family"],
            "fold_scheme": fold_scheme,
            "victim_id": victim_id,
            "heldout_block": int(heldout_block),
            "seed": int(seed),
            "response_id": int(query.response_id),
            "query_inhib_id": str(query.inhib_id),
            "true_log_slowdown": float(query.log_slowdown),
            "is_censored": bool(query.is_censored),
            "profile_weight": float(query.profile_weight),
            "app_nearest_subset": bool(query.app_nearest_subset),
            "latent_predicted_log_slowdown": float(values[index]),
            "predicted_log_slowdown": float(observed[index]),
            "predicted_slowdown": float(slowdowns[index]),
        }
        for key, diagnostic in common_diagnostics.items():
            if isinstance(diagnostic, np.ndarray):
                record[key] = float(diagnostic[index])
            elif isinstance(diagnostic, (np.integer, np.floating)):
                record[key] = diagnostic.item()
            else:
                record[key] = diagnostic
        rows.append(record)


def _fold_scaler(
    data: TrainingData,
    victim_id: str,
    retained_inhibitors: pd.DataFrame,
    feature_set: str,
) -> ProfileScaler:
    train_jobs = data.jobs[data.jobs["job_id"].astype(str) != victim_id]
    return ProfileScaler(feature_set).fit(
        pd.concat([train_jobs, retained_inhibitors], ignore_index=True)
    )


def _absolute_validation(
    model: torch.nn.Module,
    queries: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    victims, aggressors = _row_arrays(queries, job_profiles, inhibitor_profiles)
    predicted = observe_log_slowdown(absolute_values(model, victims, aggressors, device))
    return {
        "validation_log_mae": float(
            np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy(dtype=float)))
        )
    }


def _delta_validation(
    model: torch.nn.Module,
    victim_id: str,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    support_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    potentials = response_values(
        model,
        job_profiles[victim_id],
        np.vstack([support_profiles, query_profiles]),
        device,
    )
    exact = support["log_slowdown"].to_numpy(dtype=float) > 0
    if exact.any():
        correction = float(
            np.median(
                support.loc[exact, "log_slowdown"].to_numpy(dtype=float)
                - potentials[: len(support)][exact]
            )
        )
    else:
        correction = fit_censored_intercept(
            support["log_slowdown"].to_numpy(dtype=float),
            potentials[: len(support)],
            shrinkage=1.0,
        ).correction
    predicted = observe_log_slowdown(potentials[len(support) :] + correction)
    return {
        "validation_log_mae": float(
            np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy(dtype=float)))
        )
    }


def _classical_fold_predictions(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    train_rows: pd.DataFrame,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    distance_profiles: dict[str, np.ndarray],
    *,
    fold_scheme: str,
    victim_id: str,
    heldout_block: int,
    seed: int,
    config: ExperimentConfig,
) -> None:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_family.setdefault(str(candidate["family"]), []).append(candidate)

    constant = by_family["constant"][0]
    _append_predictions(
        rows,
        constant,
        queries,
        0.0,
        fold_scheme=fold_scheme,
        victim_id=victim_id,
        heldout_block=heldout_block,
        seed=seed,
    )
    train_logs = train_rows["log_slowdown"].to_numpy(dtype=float)
    observed_median = float(np.median(train_logs))
    for candidate in by_family["global_intercept"]:
        parameters = candidate["parameters"]
        if "alpha" in parameters:
            latent = float(parameters["alpha"]) * observed_median
        else:
            latent = fit_censored_intercept(
                train_logs,
                shrinkage=float(parameters["shrinkage"]),
                prior=0.0,
            ).correction
        _append_predictions(
            rows,
            candidate,
            queries,
            latent,
            fold_scheme=fold_scheme,
            victim_id=victim_id,
            heldout_block=heldout_block,
            seed=seed,
        )

    population = fit_censored_intercept(train_logs, shrinkage=0.0).correction
    support_logs = support["log_slowdown"].to_numpy(dtype=float)
    for candidate in by_family["victim_intercept"]:
        result = fit_censored_intercept(
            support_logs,
            shrinkage=float(candidate["parameters"]["shrinkage"]),
            prior=population,
        )
        _append_predictions(
            rows,
            candidate,
            queries,
            result.correction,
            fold_scheme=fold_scheme,
            victim_id=victim_id,
            heldout_block=heldout_block,
            seed=seed,
            diagnostics={
                "correction": result.correction,
                "correction_exact_count": result.exact_count,
                "correction_floor_count": result.floor_count,
                "correction_active_floor_violations": result.active_floor_violations,
            },
        )

    support_distance_values = np.stack(
        [distance_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_distance_values = np.stack(
        [distance_profiles[str(value)] for value in queries["inhib_id"]]
    )
    distance_matrix = np.linalg.norm(
        query_distance_values[:, None, :] - support_distance_values[None, :, :], axis=-1
    )
    nearest_index = distance_matrix.argmin(axis=1)
    nearest_latent = support_logs[nearest_index]
    _append_predictions(
        rows,
        by_family["anchor_baseline"][0],
        queries,
        nearest_latent,
        fold_scheme=fold_scheme,
        victim_id=victim_id,
        heldout_block=heldout_block,
        seed=seed,
        diagnostics={"nearest_anchor_distance": distance_matrix.min(axis=1)},
    )
    _append_predictions(
        rows,
        by_family["anchor_baseline"][1],
        queries,
        float(np.median(support_logs)),
        fold_scheme=fold_scheme,
        victim_id=victim_id,
        heldout_block=heldout_block,
        seed=seed,
    )

    train_victims, train_aggressors = _row_arrays(
        train_rows, job_profiles, inhibitor_profiles
    )
    query_victims, query_aggressors = _row_arrays(
        queries, job_profiles, inhibitor_profiles
    )
    train_design = linear_design(train_victims, train_aggressors)
    query_design = linear_design(query_victims, query_aggressors)
    for candidate in by_family["absolute_linear"]:
        coefficients, fit = fit_censored_linear(
            train_design,
            train_logs,
            ridge=float(candidate["parameters"]["ridge"]),
        )
        _append_predictions(
            rows,
            candidate,
            queries,
            query_design @ coefficients,
            fold_scheme=fold_scheme,
            victim_id=victim_id,
            heldout_block=heldout_block,
            seed=seed,
            diagnostics={"fit_converged": fit["converged"]},
        )

    tree_candidate = by_family["absolute_tree"][0]
    tree = GradientBoostingRegressor(
        loss="huber",
        n_estimators=config.tree_estimators,
        max_depth=config.tree_depth,
        learning_rate=config.tree_learning_rate,
        subsample=0.8,
        random_state=seed,
    )
    tree.fit(train_design[:, 1:], train_logs)
    _append_predictions(
        rows,
        tree_candidate,
        queries,
        tree.predict(query_design[:, 1:]),
        fold_scheme=fold_scheme,
        victim_id=victim_id,
        heldout_block=heldout_block,
        seed=seed,
        diagnostics={"tree_floor_targets_treated_as_observed_boundary": True},
    )


def _neural_fold_predictions(
    rows: list[dict[str, Any]],
    curve_writer: _CSVWriter,
    candidates: list[dict[str, Any]],
    train_rows: pd.DataFrame,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    distance_profiles: dict[str, np.ndarray],
    distance_reference: OODCalibration,
    *,
    fold_scheme: str,
    victim_id: str,
    heldout_block: int,
    seed: int,
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["model_key"] != "none":
            by_model.setdefault(str(candidate["model_key"]), []).append(candidate)
    input_dim = len(next(iter(job_profiles.values())))
    query_victims, query_aggressors = _row_arrays(
        queries, job_profiles, inhibitor_profiles
    )
    support_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    support_distance_values = np.stack(
        [distance_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_distance_values = np.stack(
        [distance_profiles[str(value)] for value in queries["inhib_id"]]
    )
    distance_matrix = np.linalg.norm(
        query_distance_values[:, None, :] - support_distance_values[None, :, :], axis=-1
    )
    legacy_ood = LegacyOOD(
        config.legacy_ood_temperature,
        config.legacy_ood_quantile,
        config.legacy_ood_fallback,
    )

    for hidden, embedding in config.capacities:
        absolute_key = f"absolute:{hidden}:{embedding}"
        set_seed(seed)
        absolute = AbsoluteResponseModel(input_dim, hidden, embedding, config.rank)
        absolute_result = train_absolute_fixed(
            absolute,
            train_rows,
            job_profiles,
            inhibitor_profiles,
            seed=seed,
            optimizer_steps=config.optimizer_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            checkpoint_interval=config.checkpoint_interval,
            device=device,
            validation_fn=lambda model: _absolute_validation(
                model, queries, job_profiles, inhibitor_profiles, device
            ),
        )
        curve_writer.write(
            pd.DataFrame(absolute_result["history"]).assign(
                model_key=absolute_key,
                fold_scheme=fold_scheme,
                victim_id=victim_id,
                heldout_block=heldout_block,
                seed=seed,
                sampled_examples=absolute_result["sampled_examples"],
                informative_examples=absolute_result["informative_examples"],
            )
        )
        absolute_prediction = absolute_values(
            absolute, query_victims, query_aggressors, device
        )
        _append_predictions(
            rows,
            by_model[absolute_key][0],
            queries,
            absolute_prediction,
            fold_scheme=fold_scheme,
            victim_id=victim_id,
            heldout_block=heldout_block,
            seed=seed,
        )

        delta_key = f"delta:{hidden}:{embedding}"
        set_seed(seed)
        potential = LowRankPotential(input_dim, hidden, embedding, config.rank)
        delta_result = train_potential_fixed(
            potential,
            train_rows,
            job_profiles,
            inhibitor_profiles,
            seed=seed,
            optimizer_steps=config.optimizer_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            checkpoint_interval=config.checkpoint_interval,
            device=device,
            validation_fn=lambda model: _delta_validation(
                model,
                victim_id,
                support,
                queries,
                job_profiles,
                inhibitor_profiles,
                device,
            ),
        )
        curve_writer.write(
            pd.DataFrame(delta_result["history"]).assign(
                model_key=delta_key,
                fold_scheme=fold_scheme,
                victim_id=victim_id,
                heldout_block=heldout_block,
                seed=seed,
                sampled_examples=delta_result["sampled_examples"],
                informative_examples=delta_result["informative_examples"],
            )
        )
        potentials = response_values(
            potential,
            job_profiles[victim_id],
            np.vstack([support_profiles, query_profiles]),
            device,
        )
        support_potential = potentials[: len(support)]
        query_potential = potentials[len(support) :]
        support_observed = support["log_slowdown"].to_numpy(dtype=float)
        model_candidates = by_model[delta_key]
        population_gammas = sorted(
            {
                float(candidate["parameters"]["query_gamma"])
                for candidate in model_candidates
                if candidate["parameters"].get("prior_mode") == "population"
            }
        )
        population_by_gamma = {
            gamma: population_correction_prior(
                potential,
                train_rows,
                job_profiles,
                inhibitor_profiles,
                device,
                query_gamma=gamma,
            )
            for gamma in population_gammas
        }
        legacy_candidates = {
            candidate["parameters"]["aggregation"]: candidate
            for candidate in model_candidates
            if candidate["family"] == "legacy_delta"
        }
        legacy_latent = {
            "uniform": np.empty(len(queries)),
            "median": np.empty(len(queries)),
            "ood": np.empty(len(queries)),
        }
        legacy_alpha = np.empty(len(queries))
        legacy_valid = bool((support_observed > 0).any())
        if legacy_valid:
            for query_index in range(len(queries)):
                corrections = legacy_exact_corrections(
                    support_observed,
                    support_potential,
                    distance_matrix[query_index],
                    distance_reference,
                    ood=legacy_ood,
                )
                for aggregation in legacy_latent:
                    correction, diagnostic = corrections[aggregation]
                    legacy_latent[aggregation][query_index] = (
                        query_potential[query_index] + correction
                    )
                    if aggregation == "ood":
                        legacy_alpha[query_index] = diagnostic["ood_alpha"]
        else:
            for aggregation in legacy_latent:
                legacy_latent[aggregation].fill(0.0)
            legacy_alpha.fill(0.0)
        for aggregation, candidate in legacy_candidates.items():
            diagnostics = {
                "anchor_count": len(support),
                "exact_anchor_count": int((support_observed > 0).sum()),
                "floor_anchor_count": int((support_observed == 0).sum()),
                "nearest_any_anchor_distance": distance_matrix.min(axis=1),
                "fit_converged": legacy_valid,
            }
            if legacy_valid:
                exact = support_observed > 0
                diagnostics["nearest_exact_anchor_distance"] = distance_matrix[
                    :, exact
                ].min(axis=1)
            if aggregation == "ood":
                diagnostics["ood_alpha"] = legacy_alpha
            _append_predictions(
                rows,
                candidate,
                queries,
                legacy_latent[aggregation],
                fold_scheme=fold_scheme,
                victim_id=victim_id,
                heldout_block=heldout_block,
                seed=seed,
                diagnostics=diagnostics,
            )

        for candidate in model_candidates:
            if candidate["family"] not in {
                "censored_delta",
                "censored_delta_query_shrinkage",
            }:
                continue
            parameters = candidate["parameters"]
            gamma = float(parameters["query_gamma"])
            if parameters.get("prior_mode") == "population":
                prior, population_values = population_by_gamma[gamma]
            else:
                prior, population_values = 0.0, []
            latent, correction, gauge = censored_potential_predictions(
                support_observed,
                support_potential,
                query_potential,
                shrinkage=float(parameters["shrinkage"]),
                prior=prior,
                query_gamma=float(parameters["query_gamma"]),
            )
            _append_predictions(
                rows,
                candidate,
                queries,
                latent,
                fold_scheme=fold_scheme,
                victim_id=victim_id,
                heldout_block=heldout_block,
                seed=seed,
                diagnostics={
                    "anchor_count": len(support),
                    "exact_anchor_count": correction.exact_count,
                    "floor_anchor_count": correction.floor_count,
                    "correction": correction.correction,
                    "correction_shrinkage": correction.shrinkage,
                    "correction_prior": correction.prior,
                    "correction_active_floor_violations": correction.active_floor_violations,
                    "fit_converged": correction.converged,
                    "potential_gauge_center": gauge,
                    "population_correction_std": (
                        float(np.std(population_values))
                        if population_values
                        else np.nan
                    ),
                    "query_gamma": gamma,
                    "nearest_any_anchor_distance": distance_matrix.min(axis=1),
                },
            )


def run_crossed_search(
    root: Path,
    data: TrainingData,
    inhibitor_manifest: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
) -> Path:
    predictions_path = root / "tasks" / "oof_predictions.csv"
    curves_path = root / "tasks" / "training_curves.csv"
    shard_dir = root / "tasks" / "fold_shards"
    curve_shard_dir = root / "tasks" / "curve_shards"
    status_dir = root / "tasks" / "fold_status"
    shard_dir.mkdir(exist_ok=True)
    curve_shard_dir.mkdir(exist_ok=True)
    status_dir.mkdir(exist_ok=True)
    candidates = candidate_registry(config)
    geometry_columns = [
        "inhib_id",
        "profile_weight",
        "app_nearest_subset",
        "profile_block",
        "mechanism_block",
    ]
    responses = data.responses.merge(
        inhibitor_manifest[geometry_columns],
        on="inhib_id",
        how="left",
        validate="many_to_one",
    )
    execution_membership = _load_execution_membership(root, config.seeds[0])
    fold_keys = execution_membership[
        ["fold_scheme", "victim_id", "heldout_block"]
    ].drop_duplicates()
    total = len(fold_keys) * len(config.seeds)
    completed = 0
    started = time.monotonic()
    prediction_shards: list[Path] = []
    curve_shards: list[Path] = []
    ood_records: list[dict[str, Any]] = []
    for fold_scheme in config.fold_schemes:
        block_column = f"{fold_scheme}_block"
        for victim_id in data.jobs["job_id"].astype(str):
            for heldout_block in sorted(responses[block_column].unique()):
                sealed = execution_membership[
                    (execution_membership["fold_scheme"] == fold_scheme)
                    & (execution_membership["victim_id"] == victim_id)
                    & (execution_membership["heldout_block"] == heldout_block)
                ]
                if len(sealed) != len(responses):
                    raise RuntimeError(
                        f"Sealed membership is incomplete for {fold_scheme}/{victim_id}/{heldout_block}"
                    )
                identifiers = {
                    role: set(frame["response_id"].astype(int))
                    for role, frame in sealed.groupby("role", sort=False)
                }
                retained_inhibitors = data.inhibitors[
                    data.inhibitors["inhib_id"].map(
                        dict(
                            zip(
                                inhibitor_manifest["inhib_id"],
                                inhibitor_manifest[block_column],
                            )
                        )
                    )
                    != heldout_block
                ]
                train_rows = responses[
                    responses["response_id"].isin(identifiers.get("parameter_train", set()))
                ].reset_index(drop=True)
                support = responses[
                    responses["response_id"].isin(identifiers.get("support", set()))
                ].reset_index(drop=True)
                queries = responses[
                    responses["response_id"].isin(identifiers.get("query", set()))
                ].reset_index(drop=True)
                if train_rows.empty or support.empty or queries.empty:
                    raise RuntimeError(
                        f"Invalid sealed fold {fold_scheme}/{victim_id}/{heldout_block}"
                    )
                scaler = _fold_scaler(
                    data, victim_id, retained_inhibitors, config.feature_set
                )
                job_profiles, inhibitor_profiles = profile_maps(
                    data.jobs, data.inhibitors, scaler
                )
                distance_scaler = ProfileScaler("base").fit(retained_inhibitors)
                _, distance_profiles = profile_maps(
                    data.jobs, data.inhibitors, distance_scaler
                )
                reference_values = training_distance_reference(
                    train_rows,
                    distance_profiles,
                    block_column=block_column,
                )
                distance_calibration = calibrate_ood(
                    reference_values, config.legacy_ood_quantile
                )
                ood_records.append(
                    {
                        "fold_scheme": fold_scheme,
                        "victim_id": victim_id,
                        "heldout_block": int(heldout_block),
                        "source": "outer_parameter_training_rows_only",
                        "threshold": distance_calibration.threshold,
                        "width": distance_calibration.width,
                        "quantile": distance_calibration.quantile,
                        "reference_count": distance_calibration.reference_count,
                        "distance_scaler_json": json.dumps(
                            distance_scaler.state_dict(), sort_keys=True
                        ),
                    }
                )
                for seed in config.seeds:
                    task_id = (
                        f"{fold_scheme}__{victim_id}__b{int(heldout_block)}__s{int(seed)}"
                    )
                    prediction_shard = shard_dir / f"{task_id}.csv"
                    curve_shard = curve_shard_dir / f"{task_id}.csv"
                    status_path = status_dir / f"{task_id}.json"
                    prediction_shards.append(prediction_shard)
                    curve_shards.append(curve_shard)
                    if status_path.exists():
                        status = json.loads(status_path.read_text())
                        if (
                            status.get("status") != "complete"
                            or not prediction_shard.is_file()
                            or not curve_shard.is_file()
                            or sha256_file(prediction_shard)
                            != status.get("prediction_sha256")
                            or sha256_file(curve_shard) != status.get("curve_sha256")
                        ):
                            raise RuntimeError(f"Invalid resumable task state: {task_id}")
                        completed += 1
                        continue
                    fold_rows: list[dict[str, Any]] = []
                    curve_temporary = curve_shard.with_suffix(
                        curve_shard.suffix + f".partial.{os.getpid()}"
                    )
                    curve_writer = _CSVWriter(curve_temporary)
                    _classical_fold_predictions(
                        fold_rows,
                        candidates,
                        train_rows,
                        support,
                        queries,
                        job_profiles,
                        inhibitor_profiles,
                        distance_profiles,
                        fold_scheme=fold_scheme,
                        victim_id=victim_id,
                        heldout_block=int(heldout_block),
                        seed=seed,
                        config=config,
                    )
                    _neural_fold_predictions(
                        fold_rows,
                        curve_writer,
                        candidates,
                        train_rows,
                        support,
                        queries,
                        job_profiles,
                        inhibitor_profiles,
                        distance_profiles,
                        distance_calibration,
                        fold_scheme=fold_scheme,
                        victim_id=victim_id,
                        heldout_block=int(heldout_block),
                        seed=seed,
                        config=config,
                        device=device,
                    )
                    methods = {row["candidate_id"] for row in fold_rows}
                    expected = {candidate["candidate_id"] for candidate in candidates}
                    if methods != expected:
                        raise RuntimeError(
                            f"Incomplete candidate predictions: {sorted(expected - methods)}"
                        )
                    counts = pd.Series(
                        [row["candidate_id"] for row in fold_rows]
                    ).value_counts()
                    if not counts.eq(len(queries)).all():
                        raise RuntimeError(f"Incomplete query coverage in task {task_id}")
                    prediction_temporary = prediction_shard.with_suffix(
                        prediction_shard.suffix + f".partial.{os.getpid()}"
                    )
                    pd.DataFrame(fold_rows).to_csv(prediction_temporary, index=False)
                    prediction_temporary.replace(prediction_shard)
                    curve_temporary.replace(curve_shard)
                    write_json(
                        status_path,
                        {
                            "status": "complete",
                            "task_id": task_id,
                            "prediction_sha256": sha256_file(prediction_shard),
                            "curve_sha256": sha256_file(curve_shard),
                            "candidate_count": len(expected),
                            "query_count": len(queries),
                        },
                    )
                    completed += 1
                    if completed == 1 or completed == total or completed % 10 == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"Crossed search: {completed}/{total} fold-seeds "
                            f"({100.0 * completed / total:.1f}%), elapsed {elapsed:.0f}s",
                            flush=True,
                        )
    if len(prediction_shards) != total or any(
        not path.is_file() for path in prediction_shards + curve_shards
    ):
        raise RuntimeError("Crossed search did not complete every sealed fold task")
    _concatenate_csv_shards(prediction_shards, predictions_path)
    _concatenate_csv_shards(curve_shards, curves_path)
    pd.DataFrame(ood_records).to_csv(
        root / "tasks" / "fold_ood_calibrations.csv", index=False
    )
    return predictions_path


def _validate_oof_coverage(
    predictions: pd.DataFrame,
    data: TrainingData,
    config: ExperimentConfig,
    candidates: list[dict[str, Any]],
    root: Path,
) -> None:
    keys = ["candidate_id", "fold_scheme", "seed", "response_id"]
    if predictions.duplicated(keys).any():
        raise RuntimeError("OOF predictions contain duplicate candidate/query rows")
    counts = predictions.groupby(["candidate_id", "fold_scheme", "seed"]).size()
    if not counts.eq(len(data.responses)).all():
        raise RuntimeError("At least one candidate lacks complete OOF query coverage")
    expected_groups = len(candidates) * len(config.fold_schemes) * len(config.seeds)
    if len(counts) != expected_groups:
        raise RuntimeError("OOF coverage has missing candidate/scheme/seed groups")
    if (
        not np.isfinite(predictions["predicted_slowdown"]).all()
        or (predictions["predicted_slowdown"] < 1.0).any()
    ):
        raise RuntimeError("OOF predictions violate the observed support")
    query_chunks = []
    columns = [
        "fold_scheme",
        "victim_id",
        "heldout_block",
        "seed",
        "response_id",
        "inhib_id",
        "role",
    ]
    for chunk in pd.read_csv(root / "fold_manifest.csv", usecols=columns, chunksize=100_000):
        selected = chunk[chunk["role"] == "query"]
        if len(selected):
            query_chunks.append(selected.drop(columns="role"))
    expected_queries = pd.concat(query_chunks, ignore_index=True).rename(
        columns={"inhib_id": "sealed_query_inhib_id"}
    )
    actual_queries = predictions[
        [
            "fold_scheme",
            "victim_id",
            "heldout_block",
            "seed",
            "response_id",
            "query_inhib_id",
        ]
    ].drop_duplicates()
    compared = actual_queries.merge(
        expected_queries,
        on=["fold_scheme", "victim_id", "heldout_block", "seed", "response_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not compared["_merge"].eq("both").all() or not compared[
        "query_inhib_id"
    ].astype(str).eq(compared["sealed_query_inhib_id"].astype(str)).all():
        raise RuntimeError("OOF query metadata does not match the sealed fold manifest")


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        if not keep[index]:
            continue
        dominated = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        if dominated.any():
            keep[index] = False
    return keep


def select_candidate(
    seed_fold_metrics: pd.DataFrame,
    ensemble_query_metrics: pd.DataFrame,
    ensemble_fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    registry = pd.DataFrame(candidates)
    primary_schemes = list(predictions["fold_scheme"].drop_duplicates())
    metric_rows: list[dict[str, Any]] = []
    for candidate_id, candidate_frame in ensemble_fold_metrics.groupby(
        "candidate_id", sort=False
    ):
        row: dict[str, Any] = {"candidate_id": candidate_id}
        robust_values = []
        for scheme in primary_schemes:
            for view in ["unweighted", "app_profile_weighted"]:
                selected = candidate_frame[
                    (candidate_frame["fold_scheme"] == scheme)
                    & (candidate_frame["view"] == view)
                ]
                if selected.empty:
                    raise RuntimeError(f"Missing primary validation view for {candidate_id}")
                prefix = f"{scheme}_{view}"
                for metric in [
                    "log_mae",
                    "raw_mae",
                    "raw_mse",
                    "signed_log_bias",
                    "spearman",
                ]:
                    row[prefix + "_" + metric] = float(selected[metric].mean())
                row[prefix + "_mean_absolute_fold_bias"] = float(
                    selected["signed_log_bias"].abs().mean()
                )
                if view == "unweighted":
                    defined_rank = selected[selected["spearman_defined"].astype(bool)]
                    row[f"{scheme}_defined_fold_rank_fraction"] = float(
                        len(defined_rank) / len(selected)
                    )
                    row[f"{scheme}_macro_defined_fold_spearman"] = (
                        float(defined_rank["spearman"].mean())
                        if len(defined_rank)
                        else 0.0
                    )
                robust_values.append(float(selected["log_mae"].mean()))
            overall_rank = ensemble_query_metrics[
                (ensemble_query_metrics["candidate_id"] == candidate_id)
                & (ensemble_query_metrics["fold_scheme"] == scheme)
                & (ensemble_query_metrics["view"] == "unweighted")
            ]
            if len(overall_rank) != 1:
                raise RuntimeError(f"Missing ensemble rank metric for {candidate_id}")
            row[f"{scheme}_overall_spearman"] = float(overall_rank.iloc[0]["spearman"])
            row[f"{scheme}_spearman_defined"] = bool(
                overall_rank.iloc[0]["spearman_defined"]
            )
        row["robust_log_mae"] = max(robust_values)
        metric_rows.append(row)
    selection = pd.DataFrame(metric_rows).merge(
        registry.drop(columns="parameters"), on="candidate_id", validate="one_to_one"
    )

    collapsed: dict[str, bool] = {}
    unstable: dict[str, bool] = {}
    numerical_valid: dict[str, bool] = {}
    for candidate_id, frame in predictions.groupby("candidate_id", sort=False):
        family = str(frame["family"].iloc[0])
        seed_spread = frame.groupby(["fold_scheme", "seed"])[
            "predicted_log_slowdown"
        ].std(ddof=0)
        collapsed[candidate_id] = bool(
            family in {"absolute_neural", "legacy_delta", "censored_delta", "censored_delta_query_shrinkage"}
            and (seed_spread <= 1e-8).any()
        )
        primary_seed_folds = seed_fold_metrics[
            (seed_fold_metrics["candidate_id"] == candidate_id)
            & seed_fold_metrics["fold_scheme"].isin(primary_schemes)
            & seed_fold_metrics["view"].isin(
                ["unweighted", "app_profile_weighted"]
            )
        ]
        seed_view_scores = (
            primary_seed_folds.groupby(
                ["seed", "fold_scheme", "view"], sort=False
            )["log_mae"]
            .mean()
            .reset_index()
        )
        expected_seed_views = len(primary_schemes) * 2
        views_per_seed = seed_view_scores.groupby("seed").size()
        if views_per_seed.empty or not views_per_seed.eq(expected_seed_views).all():
            raise RuntimeError(f"Missing seed-level primary views for {candidate_id}")
        seed_robust = seed_view_scores.groupby("seed")["log_mae"].max()
        unstable[candidate_id] = bool(
            seed_robust.max()
            > seed_robust.mean() + max(0.05, 0.5 * seed_robust.mean())
        )
        if "fit_converged" in frame:
            reported = frame["fit_converged"].dropna().astype(str).str.lower()
            numerical_valid[candidate_id] = bool(
                reported.empty or reported.isin({"true", "1", "1.0"}).all()
            )
        else:
            numerical_valid[candidate_id] = True
    selection["seed_collapsed"] = selection["candidate_id"].map(collapsed)
    selection["seed_unstable"] = selection["candidate_id"].map(unstable)
    selection["numerical_fit_valid"] = selection["candidate_id"].map(
        numerical_valid
    )
    selection["rank_admissible"] = True
    for scheme in primary_schemes:
        selection["rank_admissible"] &= (
            (selection[f"{scheme}_defined_fold_rank_fraction"] >= 0.8)
            & (selection[f"{scheme}_macro_defined_fold_spearman"] > 0.0)
        )
    selection["base_admissible"] = (
        selection["selectable"].astype(bool)
        & ~selection["seed_collapsed"]
        & ~selection["seed_unstable"]
        & selection["numerical_fit_valid"]
        & np.isfinite(selection["robust_log_mae"])
    )
    selection["admissible"] = selection["base_admissible"] & selection[
        "rank_admissible"
    ]
    negative_result = not selection["admissible"].any()
    if negative_result:
        selection["admissible"] = selection["base_admissible"]
    objective_columns = []
    for scheme in primary_schemes:
        for view in ["unweighted", "app_profile_weighted"]:
            prefix = f"{scheme}_{view}"
            objective_columns.extend(
                [prefix + "_log_mae", prefix + "_raw_mae", prefix + "_raw_mse"]
            )
            objective_columns.append(prefix + "_mean_absolute_fold_bias")
        selection[f"{scheme}_negative_spearman"] = -selection[
            f"{scheme}_macro_defined_fold_spearman"
        ]
        objective_columns.append(f"{scheme}_negative_spearman")
    selection["pareto"] = False
    admissible_index = selection.index[selection["admissible"]]
    if len(admissible_index) == 0:
        raise RuntimeError("No candidate passed finite-range and seed-stability gates")
    selection.loc[admissible_index, "pareto"] = _pareto_mask(
        selection.loc[admissible_index, objective_columns].to_numpy(dtype=float)
    )
    numerical_best_index = selection.loc[selection["admissible"], "robust_log_mae"].idxmin()
    numerical_best = selection.loc[numerical_best_index]
    scheme_view_scores = {
        (scheme, view): float(numerical_best[f"{scheme}_{view}_log_mae"])
        for scheme in primary_schemes
        for view in ["unweighted", "app_profile_weighted"]
    }
    worst_scheme, worst_view = max(scheme_view_scores, key=scheme_view_scores.get)
    best_fold = ensemble_fold_metrics[
        (ensemble_fold_metrics["candidate_id"] == numerical_best["candidate_id"])
        & (ensemble_fold_metrics["fold_scheme"] == worst_scheme)
        & (ensemble_fold_metrics["view"] == worst_view)
    ]
    grouped = best_fold.groupby(["victim_id", "heldout_block"])["log_mae"].mean()
    standard_error = float(grouped.std(ddof=1) / np.sqrt(len(grouped)))
    threshold = float(numerical_best["robust_log_mae"] + standard_error)
    eligible = selection[
        selection["admissible"]
        & selection["pareto"]
        & (selection["robust_log_mae"] <= threshold)
    ]
    chosen = eligible.sort_values(
        ["complexity", "robust_log_mae", "candidate_id"], kind="stable"
    ).iloc[0]
    selection["selected"] = selection["candidate_id"].eq(chosen["candidate_id"])
    selection = selection.sort_values(
        ["selected", "admissible", "pareto", "robust_log_mae"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    result = next(
        candidate for candidate in candidates if candidate["candidate_id"] == chosen["candidate_id"]
    )
    result = {
        **result,
        "robust_log_mae": float(chosen["robust_log_mae"]),
        "numerical_best_candidate_id": str(numerical_best["candidate_id"]),
        "one_standard_error": standard_error,
        "one_standard_error_threshold": threshold,
        "worst_primary_view": f"{worst_scheme}/{worst_view}",
        "deployment_ensemble_rule": "mean_latent_across_configured_seeds_then_clip",
        "scientific_acceptance": not negative_result,
        "negative_result_reason": (
            None
            if not negative_result
            else "No numerically stable candidate had defined rank in at least 80% of folds and positive macro defined-fold Spearman in every scheme"
        ),
        "selection_rule": (
            "finite/boundary, numerical-fit, seed-collapse, seed-instability, and positive-rank gates; "
            "Pareto set over equal-weight fold log MAE/raw MAE/raw MSE/mean absolute fold bias/rank; "
            "minimum complexity within one equal-fold grouped standard error of minimax log MAE"
        ),
    }
    return result, selection


def _full_profile_data(
    data: TrainingData, config: ExperimentConfig
) -> tuple[ProfileScaler, dict[str, np.ndarray], dict[str, np.ndarray]]:
    scaler = ProfileScaler(config.feature_set).fit(
        pd.concat([data.jobs, data.inhibitors], ignore_index=True)
    )
    jobs, inhibitors = profile_maps(data.jobs, data.inhibitors, scaler)
    return scaler, jobs, inhibitors


_SEEDED_FINAL_FAMILIES = {
    "absolute_tree",
    "absolute_neural",
    "legacy_delta",
    "censored_delta",
    "censored_delta_query_shrinkage",
}


def _final_training_schema(
    selected: dict[str, Any], config: ExperimentConfig
) -> tuple[set[str], list[str], list[int]]:
    family = str(selected["family"])
    artifacts = {"profile_scaler.json", "distance_profile_scaler.json"}
    models: list[str] = []
    fitted_seeds = list(config.seeds) if family in _SEEDED_FINAL_FAMILIES else []
    if family in {"constant", "global_intercept", "victim_intercept", "anchor_baseline"}:
        artifacts.add("baseline_state.json")
    elif family == "absolute_linear":
        artifacts.add("absolute_linear.json")
    elif family == "absolute_tree":
        models = [f"absolute_tree_seed{seed}.joblib" for seed in config.seeds]
        artifacts.update(models)
    elif family == "absolute_neural":
        models = [f"absolute_neural_seed{seed}.pt" for seed in config.seeds]
        artifacts.update(models)
        artifacts.add("final_training_curves.csv")
    elif family in {"legacy_delta", "censored_delta", "censored_delta_query_shrinkage"}:
        models = [f"delta_seed{seed}.pt" for seed in config.seeds]
        artifacts.update(models)
        artifacts.add("final_training_curves.csv")
        if family == "legacy_delta" and selected["parameters"].get("aggregation") == "ood":
            artifacts.add("deployment_ood_reference.csv")
    else:
        raise RuntimeError(f"Unsupported final candidate family: {family}")
    return artifacts, models, fitted_seeds


def _checkpoint_file_names(checkpoint_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in checkpoint_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Final-checkpoint directory contains a non-regular file: {path}")
        names.add(path.name)
    return names


def _final_training_binding(
    root: Path, selected: dict[str, Any], config: ExperimentConfig
) -> dict[str, Any]:
    return {
        "candidate_id": selected["candidate_id"],
        "family": selected["family"],
        "method": selected["method"],
        "parameters": selected["parameters"],
        "experiment_config_sha256": sha256_file(
            root / "configs" / "experiment_config.json"
        ),
        "configured_seeds": list(config.seeds),
    }


def _validate_final_training_state(
    checkpoint_dir: Path,
    state: dict[str, Any],
    *,
    binding: dict[str, Any],
    expected_artifacts: set[str],
    expected_models: list[str],
    expected_seeds: list[int],
) -> None:
    if state.get("selection_binding") != binding:
        raise RuntimeError("Final-training state is bound to a different recipe")
    hashes = state.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != expected_artifacts:
        raise RuntimeError("Final-training artifact schema does not match the selected recipe")
    if state.get("model_artifacts") != expected_models:
        raise RuntimeError("Final-training model roster does not match configured seeds")
    if state.get("fitted_seeds") != expected_seeds or state.get(
        "fitted_model_seed_count"
    ) != len(expected_seeds):
        raise RuntimeError("Final-training fitted seeds are incomplete")
    if state.get("final_fit_valid") is not True:
        raise RuntimeError("Final-training state is not numerically valid")
    hash_manifest = checkpoint_dir / "artifact_hashes.json"
    if not hash_manifest.is_file() or json.loads(hash_manifest.read_text()) != hashes:
        raise RuntimeError("Final-training artifact hash manifest is invalid")
    expected_files = expected_artifacts | {
        "artifact_hashes.json",
        "final_training_state.json",
    }
    actual_files = _checkpoint_file_names(checkpoint_dir)
    if actual_files != expected_files:
        raise RuntimeError(
            "Final-training checkpoint roster changed: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    for name, expected in hashes.items():
        if sha256_file(checkpoint_dir / name) != expected:
            raise RuntimeError(f"Incomplete final-training state for artifact {name}")


def _application_profile_pairs(
    job_profiles: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    profiles = np.stack(list(job_profiles.values()))
    return np.repeat(profiles, len(profiles), axis=0), np.tile(profiles, (len(profiles), 1))


def _validate_saved_neural_checkpoint(
    path: Path,
    family: str,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if family == "absolute_neural":
        model: torch.nn.Module = AbsoluteResponseModel(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["embedding_dim"]),
            int(checkpoint["rank"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        assert_finite_model(model)
        victims, aggressors = _application_profile_pairs(job_profiles)
        absolute_values(model, victims, aggressors, device)
        return
    model = LowRankPotential(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    assert_finite_model(model)
    aggressors = np.vstack(
        [np.stack(list(job_profiles.values())), np.stack(list(inhibitor_profiles.values()))]
    )
    for victim in job_profiles.values():
        response_values(model, victim, aggressors, device)


def train_frozen_recipe(
    root: Path,
    selected: dict[str, Any],
    data: TrainingData,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_dir = root / "tasks" / "final_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    state_path = checkpoint_dir / "final_training_state.json"
    expected_artifacts, expected_models, expected_seeds = _final_training_schema(
        selected, config
    )
    binding = _final_training_binding(root, selected, config)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        _validate_final_training_state(
            checkpoint_dir,
            state,
            binding=binding,
            expected_artifacts=expected_artifacts,
            expected_models=expected_models,
            expected_seeds=expected_seeds,
        )
        return state
    allowed_incomplete = expected_artifacts | {"artifact_hashes.json"}
    unexpected = _checkpoint_file_names(checkpoint_dir) - allowed_incomplete
    if unexpected:
        raise RuntimeError(f"Unexpected incomplete final-training files: {sorted(unexpected)}")
    scaler, job_profiles, inhibitor_profiles = _full_profile_data(data, config)
    train_victims, train_aggressors = _row_arrays(
        data.responses, job_profiles, inhibitor_profiles
    )
    design = linear_design(train_victims, train_aggressors)
    family = selected["family"]
    parameters = selected["parameters"]
    training_records: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    model_artifacts: list[str] = []
    fitted_seeds: list[int] = []
    fitted_seed_count = 0

    if family == "constant":
        path = checkpoint_dir / "baseline_state.json"
        write_json(path, {"latent_prediction": 0.0})
        artifact_paths.append(path)
    elif family == "global_intercept":
        if "alpha" in parameters:
            latent = float(parameters["alpha"]) * float(
                np.median(data.responses["log_slowdown"].to_numpy(dtype=float))
            )
            state = {"latent_prediction": latent, "estimator": "shrunk_observed_median"}
        else:
            result = fit_censored_intercept(
                data.responses["log_slowdown"].to_numpy(dtype=float),
                shrinkage=float(parameters["shrinkage"]),
                prior=0.0,
            )
            if not result.converged or not np.isfinite(result.objective):
                raise RuntimeError("Final global censored intercept did not converge")
            state = {"latent_prediction": result.correction, "fit": result.to_dict()}
        path = checkpoint_dir / "baseline_state.json"
        write_json(path, state)
        if not np.isfinite(float(state["latent_prediction"])):
            raise FloatingPointError("Final global intercept is non-finite")
        artifact_paths.append(path)
    elif family == "victim_intercept":
        population = fit_censored_intercept(
            data.responses["log_slowdown"].to_numpy(dtype=float), shrinkage=0.0
        )
        if not population.converged or not np.isfinite(population.objective):
            raise RuntimeError("Final hierarchical population intercept did not converge")
        path = checkpoint_dir / "baseline_state.json"
        write_json(
            path,
            {
                "population_prior": population.correction,
                "victim_shrinkage": float(parameters["shrinkage"]),
                "population_fit": population.to_dict(),
            },
        )
        artifact_paths.append(path)
    elif family == "anchor_baseline":
        path = checkpoint_dir / "baseline_state.json"
        write_json(
            path,
            {
                "method": selected["method"],
                "uses_all_observed_support_anchors": True,
                "floor_values_are_observed_boundary_values": True,
            },
        )
        artifact_paths.append(path)
    elif family == "absolute_linear":
        coefficients, fit = fit_censored_linear(
            design,
            data.responses["log_slowdown"].to_numpy(dtype=float),
            ridge=float(parameters["ridge"]),
        )
        if not fit["converged"] or not np.isfinite(float(fit["objective"])):
            raise RuntimeError("Selected final censored linear model did not converge")
        path = checkpoint_dir / "absolute_linear.json"
        write_json(path, {"coefficients": coefficients.tolist(), "fit": fit})
        app_victims, app_aggressors = _application_profile_pairs(job_profiles)
        if not np.isfinite(linear_design(app_victims, app_aggressors) @ coefficients).all():
            raise FloatingPointError("Final linear model produced non-finite profile predictions")
        artifact_paths.append(path)
    elif family == "absolute_tree":
        for seed in config.seeds:
            model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=config.tree_estimators,
                max_depth=config.tree_depth,
                learning_rate=config.tree_learning_rate,
                subsample=0.8,
                random_state=seed,
            ).fit(design[:, 1:], data.responses["log_slowdown"].to_numpy(dtype=float))
            path = checkpoint_dir / f"absolute_tree_seed{seed}.joblib"
            temporary = path.with_suffix(path.suffix + ".tmp")
            joblib.dump(model, temporary)
            temporary.replace(path)
            app_victims, app_aggressors = _application_profile_pairs(job_profiles)
            app_design = linear_design(app_victims, app_aggressors)
            tree_predictions = joblib.load(path).predict(app_design[:, 1:])
            if not np.isfinite(tree_predictions).all():
                raise FloatingPointError("Final tree produced non-finite profile predictions")
            artifact_paths.append(path)
            model_artifacts.append(path.name)
            fitted_seeds.append(int(seed))
            fitted_seed_count += 1
    elif family == "absolute_neural":
        for seed in config.seeds:
            set_seed(seed)
            model = AbsoluteResponseModel(
                scaler.output_dim,
                int(parameters["hidden_dim"]),
                int(parameters["embedding_dim"]),
                config.rank,
            )
            result = train_absolute_fixed(
                model,
                data.responses,
                job_profiles,
                inhibitor_profiles,
                seed=seed,
                optimizer_steps=config.optimizer_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                checkpoint_interval=config.checkpoint_interval,
                device=device,
            )
            path = checkpoint_dir / f"absolute_neural_seed{seed}.pt"
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(
                {
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "input_dim": scaler.output_dim,
                    "hidden_dim": int(parameters["hidden_dim"]),
                    "embedding_dim": int(parameters["embedding_dim"]),
                    "rank": config.rank,
                },
                temporary,
            )
            temporary.replace(path)
            _validate_saved_neural_checkpoint(
                path, family, job_profiles, inhibitor_profiles, device
            )
            artifact_paths.append(path)
            model_artifacts.append(path.name)
            fitted_seeds.append(int(seed))
            fitted_seed_count += 1
            training_records.extend(
                {"seed": seed, **record} for record in result["history"]
            )
    elif family in {
        "legacy_delta",
        "censored_delta",
        "censored_delta_query_shrinkage",
    }:
        for seed in config.seeds:
            set_seed(seed)
            model = LowRankPotential(
                scaler.output_dim,
                int(parameters["hidden_dim"]),
                int(parameters["embedding_dim"]),
                config.rank,
            )
            result = train_potential_fixed(
                model,
                data.responses,
                job_profiles,
                inhibitor_profiles,
                seed=seed,
                optimizer_steps=config.optimizer_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                checkpoint_interval=config.checkpoint_interval,
                device=device,
            )
            checkpoint: dict[str, Any] = {
                "seed": seed,
                "state_dict": model.state_dict(),
                "input_dim": scaler.output_dim,
                "hidden_dim": int(parameters["hidden_dim"]),
                "embedding_dim": int(parameters["embedding_dim"]),
                "rank": config.rank,
            }
            if family in {"censored_delta", "censored_delta_query_shrinkage"}:
                population_prior, population_values = population_correction_prior(
                    model,
                    data.responses,
                    job_profiles,
                    inhibitor_profiles,
                    device,
                    query_gamma=float(parameters["query_gamma"]),
                )
                checkpoint["population_correction_prior"] = population_prior
                checkpoint["population_correction_values"] = population_values
                if not np.isfinite(population_prior) or not np.isfinite(
                    population_values
                ).all():
                    raise FloatingPointError("Final population correction is non-finite")
            path = checkpoint_dir / f"delta_seed{seed}.pt"
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
            _validate_saved_neural_checkpoint(
                path, family, job_profiles, inhibitor_profiles, device
            )
            artifact_paths.append(path)
            model_artifacts.append(path.name)
            fitted_seeds.append(int(seed))
            fitted_seed_count += 1
            training_records.extend(
                {"seed": seed, **record} for record in result["history"]
            )
    if training_records:
        pd.DataFrame(training_records).to_csv(
            checkpoint_dir / "final_training_curves.csv", index=False
        )
        artifact_paths.append(checkpoint_dir / "final_training_curves.csv")
    scaler_path = checkpoint_dir / "profile_scaler.json"
    write_json(scaler_path, scaler.state_dict())
    artifact_paths.append(scaler_path)
    distance_scaler_path = checkpoint_dir / "distance_profile_scaler.json"
    deployment_distance_scaler = ProfileScaler("base").fit(data.inhibitors)
    write_json(distance_scaler_path, deployment_distance_scaler.state_dict())
    artifact_paths.append(distance_scaler_path)
    deployment_ood: dict[str, Any] | None = None
    if family == "legacy_delta" and parameters.get("aggregation") == "ood":
        inhibitor_manifest = pd.read_csv(root / "inhibitor_manifest.csv")
        inhibitor_manifest["inhib_id"] = inhibitor_manifest["inhib_id"].astype(str)
        full_rows = data.responses.merge(
            inhibitor_manifest[["inhib_id", "profile_block"]],
            on="inhib_id",
            how="left",
            validate="many_to_one",
        )
        _, deployment_distance_profiles = profile_maps(
            data.jobs, data.inhibitors, deployment_distance_scaler
        )
        reference = training_distance_reference(
            full_rows,
            deployment_distance_profiles,
            block_column="profile_block",
        )
        calibration = calibrate_ood(reference, config.legacy_ood_quantile)
        reference_path = checkpoint_dir / "deployment_ood_reference.csv"
        pd.DataFrame({"distance": reference}).to_csv(reference_path, index=False)
        artifact_paths.append(reference_path)
        deployment_ood = {
            "scheme": "profile",
            "source": "nested App-Inhibitor development rows after final fitting",
            "threshold": calibration.threshold,
            "width": calibration.width,
            "quantile": calibration.quantile,
            "reference_count": calibration.reference_count,
            "temperature": config.legacy_ood_temperature,
            "fallback": config.legacy_ood_fallback,
            "reference_file": reference_path.name,
        }
    generated_artifacts = {path.name for path in artifact_paths}
    if generated_artifacts != expected_artifacts:
        raise RuntimeError(
            "Generated final-training artifacts do not match the selected recipe: "
            f"missing={sorted(expected_artifacts - generated_artifacts)}, "
            f"unexpected={sorted(generated_artifacts - expected_artifacts)}"
        )
    hashes = {path.name: sha256_file(path) for path in sorted(artifact_paths)}
    write_json(checkpoint_dir / "artifact_hashes.json", hashes)
    state = {
        "checkpoint_directory": str(checkpoint_dir.relative_to(root)),
        "artifact_hashes": hashes,
        "fitted_model_seed_count": fitted_seed_count,
        "fitted_seeds": fitted_seeds,
        "model_artifacts": model_artifacts,
        "deployment_ood": deployment_ood,
        "final_fit_valid": True,
        "selection_binding": binding,
    }
    write_json(state_path, state)
    _validate_final_training_state(
        checkpoint_dir,
        state,
        binding=binding,
        expected_artifacts=expected_artifacts,
        expected_models=expected_models,
        expected_seeds=expected_seeds,
    )
    return state


def analyze_and_freeze(
    root: Path,
    predictions_path: Path,
    data: TrainingData,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, low_memory=False)
    candidates = candidate_registry(config)
    _validate_oof_coverage(predictions, data, config, candidates, root)
    query_metrics, fold_metrics, per_victim, stability = metric_tables(predictions)
    ensemble = ensemble_predictions(predictions)
    ensemble_metrics = ensemble_metric_table(ensemble)
    (
        ensemble_query_metrics,
        ensemble_fold_metrics,
        ensemble_per_victim,
        _,
    ) = metric_tables(ensemble.assign(seed="ensemble"))
    query_metrics.to_csv(root / "consolidated" / "validation_metrics_by_seed.csv", index=False)
    fold_metrics.to_csv(root / "consolidated" / "fold_metrics.csv", index=False)
    per_victim.to_csv(root / "consolidated" / "metrics_per_victim.csv", index=False)
    stability.to_csv(root / "consolidated" / "seed_stability.csv", index=False)
    ensemble.to_csv(root / "consolidated" / "oof_ensemble_predictions.csv", index=False)
    ensemble_metrics.to_csv(root / "consolidated" / "ensemble_metrics.csv", index=False)
    ensemble_fold_metrics.to_csv(
        root / "consolidated" / "ensemble_fold_metrics.csv", index=False
    )
    ensemble_per_victim.to_csv(
        root / "consolidated" / "ensemble_metrics_per_victim.csv", index=False
    )
    selected, selection_table = select_candidate(
        fold_metrics,
        ensemble_query_metrics,
        ensemble_fold_metrics,
        predictions,
        candidates,
    )
    selection_table.to_csv(root / "consolidated" / "candidate_selection.csv", index=False)
    eligible_family_rows = selection_table[
        selection_table["numerical_fit_valid"] & ~selection_table["seed_collapsed"]
    ]
    family_best = (
        eligible_family_rows.sort_values("robust_log_mae", kind="stable")
        .groupby("family", as_index=False, sort=False)
        .first()
    )
    family_best.to_csv(root / "consolidated" / "ablation_summary.csv", index=False)
    rank_source = ensemble_metrics.pivot_table(
        index="candidate_id",
        columns=["fold_scheme", "view"],
        values="log_mae",
    )
    rank_source.corr(method="spearman").to_csv(
        root / "consolidated" / "validation_view_rank_correlations.csv"
    )

    comparison_ids = [selected["candidate_id"]]
    for family in ["constant", "absolute_linear", "absolute_neural"]:
        choices = selection_table[selection_table["family"] == family]
        choices = choices[
            choices["numerical_fit_valid"]
            & ~choices["seed_collapsed"]
            & ~choices["seed_unstable"]
        ]
        if len(choices):
            comparison_ids.append(str(choices.sort_values("robust_log_mae").iloc[0]["candidate_id"]))
    comparison_ids = list(dict.fromkeys(comparison_ids))
    paired = paired_fold_bootstrap(
        predictions,
        comparison_ids,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    paired.to_csv(root / "consolidated" / "paired_victim_bootstrap.csv", index=False)

    final_training = train_frozen_recipe(root, selected, data, config, device)
    selection_payload = {
        "selected": selected,
        "selection_domain": "App-Inhibitor development CV only",
        "primary_views": [
            f"{scheme}/{view}"
            for scheme in config.fold_schemes
            for view in ["unweighted", "app_profile_weighted"]
        ],
        "profile_weighting_is_transductive": True,
        "deployment_ensemble_rule": "mean_latent_across_configured_seeds_then_clip",
        "app_app_outcomes_used": False,
        "historical_app_app_status": "diagnostic_only_not_opened",
    }
    write_json(root / "consolidated" / "selection.json", selection_payload)
    recipe = {
        **selection_payload,
        "resolved_experiment_config": config.to_dict(),
        "plan_seal_sha256": sha256_file(root / "seal.json"),
        "code_hashes_sha256": sha256_file(root / "code_hashes.json"),
        "environment_sha256": sha256_file(root / "environment.json"),
        "candidate_specification": selected,
        "training_budget": {
            "optimizer_steps": config.optimizer_steps,
            "batch_size": config.batch_size,
            "sampled_examples_per_neural_fit": config.optimizer_steps * config.batch_size,
            "checkpoint_interval": config.checkpoint_interval,
            "early_stopping": False,
        },
        "final_training": final_training,
        "ensemble_rule": "mean latent log prediction across fitted seeds, then clip at zero",
        "support_anchor_policy": (
            "all anchors for censored methods; exact-only anchors only for explicitly named legacy methods"
        ),
        "preserves_potential_invariants": selected["family"]
        in {"legacy_delta", "censored_delta", "censored_delta_query_shrinkage"},
        "calibrated_app_app_scale_identified": False,
        "new_app_app_holdout_required": True,
        "prohibited_claim": "No result from this experiment is an App-App transfer estimate",
    }
    content_hash = sha256_file(root / "consolidated" / "selection.json")
    recipe["selection_file_sha256"] = content_hash
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recipe["recipe_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(root / "frozen_recipe.json", recipe)
    _write_final_report(root, data, config, selected, selection_table, family_best)
    return {"selected": selected, "final_training": final_training}


def _write_final_report(
    root: Path,
    data: TrainingData,
    config: ExperimentConfig,
    selected: dict[str, Any],
    selection_table: pd.DataFrame,
    family_best: pd.DataFrame,
) -> None:
    selected_row = selection_table[
        selection_table["candidate_id"] == selected["candidate_id"]
    ].iloc[0]
    family_lines = []
    for row in family_best.sort_values("robust_log_mae").itertuples(index=False):
        family_lines.append(
            f"| `{row.family}` | `{row.method}` | {row.robust_log_mae:.6f} | "
            f"{bool(row.admissible)} |"
        )
    primary_lines = []
    for scheme in config.fold_schemes:
        for view in ["unweighted", "app_profile_weighted"]:
            value = float(selected_row[f"{scheme}_{view}_log_mae"])
            primary_lines.append(f"| `{scheme}` | `{view}` | {value:.6f} |")
    floor_count = int(data.responses["is_censored"].sum())
    content = "\n".join(
        [
            "# Delta Response 2 Final Report",
            "",
            "## Evidence Status",
            "",
            "This is a Path A App-Inhibitor development result. Historical App-App labels ",
            "were designated diagnostic-only and were not opened by planning, fitting, ",
            "selection, calibration, or reporting. No value below is an App-App transfer ",
            "estimate. A newly collected untouched App-App holdout is still required.",
            "",
            "## Data And Protocol",
            "",
            f"- Applications: {len(data.jobs)}.",
            f"- Inhibitors: {len(data.inhibitors)}.",
            f"- Valid App-Inhibitor rows: {len(data.responses)}.",
            f"- Floor-censored rows: {floor_count} ({100.0 * floor_count / len(data.responses):.2f}%).",
            f"- Fold schemes: {', '.join(config.fold_schemes)}.",
            f"- Seeds: {', '.join(map(str, config.seeds))}.",
            f"- Neural budget: {config.optimizer_steps} optimizer steps and "
            f"{config.optimizer_steps * config.batch_size} sampled examples per fit.",
            "- Application-profile weighting is transductive and uses no response labels.",
            "- Outer query labels never choose checkpoints; all neural fits use fixed steps.",
            "",
            "## Selected Recipe",
            "",
            f"Selected `{selected['method']}` from family `{selected['family']}`.",
            f"The prespecified worst-view development log MAE was {selected['robust_log_mae']:.6f}.",
            f"Scientific acceptance gate passed: {selected['scientific_acceptance']}.",
            "Selection applied finite-boundary and seed-stability gates, Pareto filtering, ",
            "then the declared grouped one-standard-error complexity preference.",
            "",
            "| Fold scheme | Validation view | Selected log MAE |",
            "| --- | --- | ---: |",
            *primary_lines,
            "",
            "## Best Candidate Per Ablation Family",
            "",
            "| Family | Candidate | Worst-view log MAE | Selection-admissible |",
            "| --- | --- | ---: | --- |",
            *family_lines,
            "",
            "## Established In This Run",
            "",
            "- All candidates produced complete, finite OOF predictions at or above slowdown 1.",
            "- Censored candidate families used floor anchors as inequalities rather than invented latent targets; legacy methods remained exact-only by explicit label.",
            "- Profile-weighted, nearest-profile, floor, positive, magnitude, per-victim, and seed views were saved.",
            "- Complete mechanism families remained together when the mechanism fold scheme was enabled.",
            "- Candidate and view rankings are development diagnostics; their stability table is saved separately.",
            "",
            "## Unresolved Transfer Question",
            "",
            "App-Inhibitor labels cannot identify the scale compression previously observed for real ",
            "application aggressors. The selected recipe is frozen for future evaluation, but it must ",
            "not be described as calibrated for App-App prediction until tested once on newly collected, ",
            "sealed pair clusters. A negative future result would support the conclusion that the current ",
            "aggregate profiles do not identify synthetic-to-real transfer.",
            "",
        ]
    )
    (root / "FINAL_REPORT.md").write_text(content)


def _remove_stale_pipeline_temporaries(root: Path) -> None:
    candidates = [
        root / "output_hashes.json",
        root / "output_hashes.json.tmp",
        root / "completion_seal.json.tmp",
        root / "run_report.json.tmp",
        root / "frozen_recipe.json.tmp",
        root / "tasks" / "oof_predictions.csv.tmp",
        root / "tasks" / "training_curves.csv.tmp",
        root / "consolidated" / "selection.json.tmp",
    ]
    for directory, pattern in [
        (root / "tasks" / "fold_shards", "*.partial.*"),
        (root / "tasks" / "curve_shards", "*.partial.*"),
        (root / "tasks" / "fold_status", "*.json.tmp"),
        (root / "tasks" / "final_checkpoints", "*.tmp"),
    ]:
        if directory.is_dir():
            candidates.extend(directory.glob(pattern))
    for path in candidates:
        if path.is_symlink():
            raise RuntimeError(f"Refusing to remove a pipeline symlink: {path}")
        if path.is_file():
            path.unlink()


def _expected_completed_output_paths(
    root: Path, config: ExperimentConfig, final_training: dict[str, Any]
) -> set[str]:
    plan_seal = json.loads((root / "seal.json").read_text())
    expected = set(plan_seal["static_files"]) | {"seal.json"}
    expected.update(
        {
            "tasks/oof_predictions.csv",
            "tasks/training_curves.csv",
            "tasks/fold_ood_calibrations.csv",
            "consolidated/validation_metrics_by_seed.csv",
            "consolidated/fold_metrics.csv",
            "consolidated/metrics_per_victim.csv",
            "consolidated/seed_stability.csv",
            "consolidated/oof_ensemble_predictions.csv",
            "consolidated/ensemble_metrics.csv",
            "consolidated/ensemble_fold_metrics.csv",
            "consolidated/ensemble_metrics_per_victim.csv",
            "consolidated/candidate_selection.csv",
            "consolidated/ablation_summary.csv",
            "consolidated/validation_view_rank_correlations.csv",
            "consolidated/paired_victim_bootstrap.csv",
            "consolidated/selection.json",
            "frozen_recipe.json",
            "FINAL_REPORT.md",
            "run_report.json",
        }
    )
    membership = _load_execution_membership(root, config.seeds[0])
    fold_keys = membership[
        ["fold_scheme", "victim_id", "heldout_block"]
    ].drop_duplicates()
    for fold in fold_keys.itertuples(index=False):
        for seed in config.seeds:
            task_id = (
                f"{fold.fold_scheme}__{fold.victim_id}__"
                f"b{int(fold.heldout_block)}__s{int(seed)}"
            )
            expected.update(
                {
                    f"tasks/fold_shards/{task_id}.csv",
                    f"tasks/curve_shards/{task_id}.csv",
                    f"tasks/fold_status/{task_id}.json",
                }
            )
    checkpoint_directory = str(final_training["checkpoint_directory"])
    for name in final_training["artifact_hashes"]:
        expected.add(f"{checkpoint_directory}/{name}")
    expected.update(
        {
            f"{checkpoint_directory}/artifact_hashes.json",
            f"{checkpoint_directory}/final_training_state.json",
        }
    )
    return expected


def run_experiment(root: Path, package_dir: Path) -> None:
    started = time.time()
    verify_plan(root, package_dir)
    if (root / "completion_seal.json").exists():
        verify_completion(root)
        print(f"Experiment is already complete and verified: {root}", flush=True)
        return
    _remove_stale_pipeline_temporaries(root)
    config = ExperimentConfig.from_json(root / "configs" / "experiment_config.json")
    data, inhibitor_manifest = _load_planned_data(root)
    device = resolve_device(config.device)
    print(
        "Running sealed random_split App-Inhibitor search; no App-App path is available",
        flush=True,
    )
    predictions_path = run_crossed_search(
        root, data, inhibitor_manifest, config, device
    )
    result = analyze_and_freeze(root, predictions_path, data, config, device)
    report = {
        "status": "complete",
        "protocol": "random_split",
        "scientific_path": "Path A",
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "applications": len(data.jobs),
        "inhibitors": len(data.inhibitors),
        "valid_app_inhibitor_rows": len(data.responses),
        "rejected_app_inhibitor_rows": len(data.rejected_responses),
        "floor_rows": int(data.responses["is_censored"].sum()),
        "candidate_count": len(candidate_registry(config)),
        "fold_schemes": config.fold_schemes,
        "seeds": config.seeds,
        "app_app_file_opened": False,
        "app_app_metrics_reported": False,
        "historical_app_app_status": "diagnostic_only",
        "selected_candidate_id": result["selected"]["candidate_id"],
        "selected_method": result["selected"]["method"],
        "scientific_acceptance": result["selected"]["scientific_acceptance"],
        "deployment_ensemble_rule": "mean_latent_across_fitted_seeds_then_clip",
        "new_untouched_app_app_holdout_required": True,
    }
    write_json(root / "run_report.json", report)
    verify_plan(root, package_dir)
    write_output_hashes(
        root,
        _expected_completed_output_paths(root, config, result["final_training"]),
    )
    write_completion_seal(root)
    verify_completion(root)
    print(f"Frozen Path A recipe written to {root / 'frozen_recipe.json'}", flush=True)
