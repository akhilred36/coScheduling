"""Outcome-blind Delta Response 2 prediction construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from delta_response_2.data import (
    ProfileScaler,
    load_training_data,
    log_to_slowdown,
    observe_log_slowdown,
    profile_maps,
)
from delta_response_2.inference import (
    LegacyOOD,
    OODCalibration,
    censored_potential_predictions,
    legacy_exact_corrections,
)
from delta_response_2.model import AbsoluteResponseModel, LowRankPotential
from delta_response_2.training import (
    absolute_values,
    assert_finite_model,
    linear_design,
    response_values,
)

from .constants import (
    CORRECTION_DIAGNOSTIC_COLUMNS,
    DELTA2_METHODS,
    DETERMINISTIC_METHODS,
    DIRECTIONAL_ROWS,
    EVIDENCE_LABEL,
    FROZEN_ROOT,
    INHIBITORS_PATH,
    JOB_INH_PATH,
    JOBS_PATH,
    METHOD_CANDIDATES,
    SEEDS,
    SEEDED_METHODS,
    TRAINING_SETTINGS,
)
from .integrity import read_canonical_json


QUERY_COLUMNS = (
    "evaluation_method",
    "pair_row_id",
    "pair_cluster_id",
    "direction",
    "victim_id",
    "aggressor_id",
    "self_pair",
)
KEY_COLUMNS = (
    "evaluation_method",
    "pair_row_id",
    "pair_cluster_id",
    "direction",
    "victim_id",
    "aggressor_id",
    "self_pair",
)


def _load_potential(path: Path, device: torch.device) -> tuple[dict[str, Any], LowRankPotential]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LowRankPotential(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    assert_finite_model(model)
    model.eval()
    return checkpoint, model


def _load_absolute(path: Path, device: torch.device) -> tuple[dict[str, Any], AbsoluteResponseModel]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AbsoluteResponseModel(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    assert_finite_model(model)
    model.eval()
    return checkpoint, model


def _prediction_values(latent: float) -> tuple[float, float]:
    if not np.isfinite(latent):
        raise FloatingPointError("prediction produced a non-finite latent value")
    observed = float(observe_log_slowdown(latent))
    slowdown = float(log_to_slowdown(observed))
    return observed, slowdown


def _base_row(query: Any, method: str, latent: float) -> dict[str, object]:
    observed, slowdown = _prediction_values(latent)
    return {
        "evaluation_method": str(query.evaluation_method),
        "pair_row_id": int(query.pair_row_id),
        "pair_cluster_id": int(query.pair_cluster_id),
        "direction": str(query.direction),
        "victim_id": str(query.victim_id),
        "aggressor_id": str(query.aggressor_id),
        "self_pair": bool(query.self_pair),
        "method": method,
        "candidate_id": METHOD_CANDIDATES[method],
        "latent_predicted_log_slowdown": float(latent),
        "predicted_log_slowdown": observed,
        "predicted_slowdown": slowdown,
        "evidence_label": EVIDENCE_LABEL,
    }


def _with_diagnostics(
    row: dict[str, object], diagnostics: dict[str, float | int | None]
) -> dict[str, object]:
    for name in CORRECTION_DIAGNOSTIC_COLUMNS:
        value = diagnostics.get(name)
        if value is not None and not np.isfinite(float(value)):
            raise FloatingPointError(f"prediction diagnostic is non-finite: {name}")
        row[name] = value
    return row


def _ranked_exact_anchor_indices(
    anchors: pd.DataFrame, observed: np.ndarray, victim_id: str
) -> np.ndarray:
    usable = anchors.loc[observed > 0].copy()
    usable["_original_index"] = np.flatnonzero(observed > 0)
    if "replicate_id" not in usable:
        usable["replicate_id"] = 0

    def digest(row: pd.Series) -> str:
        identity = (
            f"1701|{victim_id}|{row['inhib_id']}|{int(row['replicate_id'])}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    usable["_anchor_order"] = usable.apply(digest, axis=1)
    ranked = usable.sort_values(
        ["_anchor_order", "inhib_id", "replicate_id"], kind="stable"
    )
    return ranked["_original_index"].to_numpy(dtype=int)


def _validate_queries(queries: pd.DataFrame, valid_ids: set[str]) -> pd.DataFrame:
    if set(queries.columns) != set(QUERY_COLUMNS):
        raise ValueError(
            "prediction queries must contain only outcome-blind identity columns"
        )
    result = queries.loc[:, list(QUERY_COLUMNS)].copy().reset_index(drop=True)
    if len(result) != DIRECTIONAL_ROWS:
        raise ValueError("static prediction query must contain exactly 110 directions")
    if not result["evaluation_method"].eq("random_split").all():
        raise ValueError("all prediction queries must use random_split")
    if set(result["direction"].astype(str)) != {"A", "B"}:
        raise ValueError("prediction directions must be A or B")
    for column in ("victim_id", "aggressor_id"):
        result[column] = result[column].astype(str)
        unknown = sorted(set(result[column]) - valid_ids)
        if unknown:
            raise ValueError(f"prediction query has unknown profile IDs: {unknown}")
    duplicate_keys = result.duplicated(
        ["pair_row_id", "pair_cluster_id", "direction", "victim_id", "aggressor_id"]
    )
    if duplicate_keys.any():
        raise ValueError("prediction query contains duplicate directions")
    return result


def build_delta2_predictions(
    root: Path,
    queries: pd.DataFrame,
    prepared: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict from profile IDs only; this API accepts no outcome columns."""
    device = torch.device("cpu")
    data = load_training_data(JOBS_PATH, INHIBITORS_PATH, JOB_INH_PATH)
    valid_ids = set(data.jobs["job_id"].astype(str))
    query_frame = _validate_queries(queries, valid_ids)
    checkpoint_dir = root / "diagnostic_checkpoints"

    model_scaler = ProfileScaler.from_state_dict(
        read_canonical_json(checkpoint_dir / "profile_scaler.json")
    )
    distance_scaler = ProfileScaler.from_state_dict(
        read_canonical_json(checkpoint_dir / "distance_profile_scaler.json")
    )
    job_profiles, inhibitor_profiles = profile_maps(
        data.jobs, data.inhibitors, model_scaler
    )
    distance_jobs, distance_inhibitors = profile_maps(
        data.jobs, data.inhibitors, distance_scaler
    )

    recipe = prepared["recipe"]
    frozen_checkpoint_dir = (
        FROZEN_ROOT / recipe["final_training"]["checkpoint_directory"]
    )
    selected_models: dict[int, LowRankPotential] = {}
    selected_names = recipe["final_training"]["model_artifacts"]
    for name in selected_names:
        checkpoint, model = _load_potential(frozen_checkpoint_dir / name, device)
        seed = int(checkpoint["seed"])
        if seed in selected_models:
            raise RuntimeError("frozen selected recipe repeats a seed")
        selected_models[seed] = model
    if sorted(selected_models) != list(SEEDS):
        raise RuntimeError("frozen selected checkpoint seeds are incomplete")

    potential_models: dict[int, tuple[dict[str, Any], LowRankPotential]] = {}
    absolute_models: dict[int, AbsoluteResponseModel] = {}
    for seed in SEEDS:
        checkpoint, model = _load_potential(
            checkpoint_dir / f"delta_h32_e16_seed{seed}.pt", device
        )
        if int(checkpoint["seed"]) != seed:
            raise RuntimeError("diagnostic potential checkpoint has the wrong seed")
        expected_shared = {
            method: METHOD_CANDIDATES[method]
            for method in (
                "delta2_profile_only_ood_h32_e16",
                "delta2_censored_zero_h32_e16",
                "delta2_query_shrinkage_h32_e16",
            )
        }
        if checkpoint.get("shared_candidate_ids") != expected_shared:
            raise RuntimeError("diagnostic potential checkpoint candidate binding changed")
        if checkpoint.get("population_correction_prior_query_gamma") != 0.75:
            raise RuntimeError("diagnostic potential checkpoint has the wrong prior gamma")
        potential_models[seed] = (checkpoint, model)

        absolute_checkpoint, absolute_model = _load_absolute(
            checkpoint_dir / f"absolute_neural_h32_e16_seed{seed}.pt", device
        )
        if (
            int(absolute_checkpoint["seed"]) != seed
            or absolute_checkpoint.get("candidate_id")
            != METHOD_CANDIDATES["delta2_absolute_neural_h32_e16"]
        ):
            raise RuntimeError("diagnostic absolute checkpoint binding changed")
        absolute_models[seed] = absolute_model

    linear_state = read_canonical_json(
        checkpoint_dir / "absolute_linear_r0p01.json"
    )
    if (
        linear_state.get("candidate_id")
        != METHOD_CANDIDATES["delta2_absolute_linear_r0p01"]
        or float(linear_state.get("ridge")) != 0.01
    ):
        raise RuntimeError("diagnostic linear artifact binding changed")
    linear_coefficients = np.asarray(linear_state["coefficients"], dtype=float)
    if not np.isfinite(linear_coefficients).all():
        raise FloatingPointError("diagnostic linear coefficients are non-finite")

    selected_ood_state = recipe["final_training"]["deployment_ood"]
    selected_calibration = OODCalibration(
        float(selected_ood_state["threshold"]),
        float(selected_ood_state["width"]),
        float(selected_ood_state["quantile"]),
        int(selected_ood_state["reference_count"]),
    )
    diagnostic_ood_state = prepared["preparation_state"]["ood_reference"]
    diagnostic_calibration = OODCalibration(
        float(diagnostic_ood_state["threshold"]),
        float(diagnostic_ood_state["width"]),
        float(diagnostic_ood_state["quantile"]),
        int(diagnostic_ood_state["reference_count"]),
    )
    legacy_ood = LegacyOOD(
        float(TRAINING_SETTINGS["legacy_ood_temperature"]),
        float(TRAINING_SETTINGS["legacy_ood_quantile"]),
        str(TRAINING_SETTINGS["legacy_ood_fallback"]),
    )

    seed_rows: list[dict[str, object]] = []
    deterministic_rows: list[dict[str, object]] = []
    for query in query_frame.itertuples(index=False):
        victim_id = str(query.victim_id)
        aggressor_id = str(query.aggressor_id)
        anchors = data.responses[data.responses["job_id"].eq(victim_id)].reset_index(
            drop=True
        )
        if anchors.empty:
            raise RuntimeError(f"victim {victim_id} has no App-Inhibitor anchors")
        observed = anchors["log_slowdown"].to_numpy(dtype=float)
        exact = observed > 0
        if not exact.any():
            raise RuntimeError(f"victim {victim_id} has no uncensored anchors")
        anchor_profiles = np.stack(
            [inhibitor_profiles[str(value)] for value in anchors["inhib_id"]]
        )
        distance_anchor_profiles = np.stack(
            [distance_inhibitors[str(value)] for value in anchors["inhib_id"]]
        )
        distances = np.linalg.norm(
            distance_anchor_profiles - distance_jobs[aggressor_id][None, :], axis=1
        )
        common_diagnostics: dict[str, float | int | None] = {
            "anchor_count": int(len(anchors)),
            "exact_anchor_count": int(exact.sum()),
            "floor_anchor_count": int((~exact).sum()),
            "nearest_anchor_distance": float(distances.min()),
            "nearest_exact_anchor_distance": float(distances[exact].min()),
        }
        ranked_exact_indices = _ranked_exact_anchor_indices(
            anchors, observed, victim_id
        )

        deterministic_rows.append(
            _with_diagnostics(
                _base_row(query, "delta2_constant_1", 0.0), {}
            )
        )
        deterministic_rows.append(
            _with_diagnostics(
                _base_row(
                    query,
                    "delta2_victim_inhibitor_median",
                    float(np.median(observed[exact])),
                ),
                common_diagnostics,
            )
        )
        nearest_exact_index = ranked_exact_indices[
            int(np.argmin(distances[ranked_exact_indices]))
        ]
        deterministic_rows.append(
            _with_diagnostics(
                _base_row(
                    query,
                    "delta2_nearest_anchor",
                    float(observed[nearest_exact_index]),
                ),
                common_diagnostics,
            )
        )
        design = linear_design(
            job_profiles[victim_id][None, :], job_profiles[aggressor_id][None, :]
        )
        deterministic_rows.append(
            _with_diagnostics(
                _base_row(
                    query,
                    "delta2_absolute_linear_r0p01",
                    float((design @ linear_coefficients)[0]),
                ),
                {},
            )
        )

        query_profile = job_profiles[aggressor_id]
        for seed in SEEDS:
            selected_potentials = response_values(
                selected_models[seed],
                job_profiles[victim_id],
                np.vstack([anchor_profiles, query_profile[None, :]]),
                device,
            )
            selected_correction, selected_diagnostic = legacy_exact_corrections(
                observed,
                selected_potentials[:-1],
                distances,
                selected_calibration,
                ood=legacy_ood,
            )["ood"]
            selected_row = _base_row(
                query,
                "delta2_selected",
                float(selected_potentials[-1] + selected_correction),
            )
            selected_row["seed"] = seed
            seed_rows.append(
                _with_diagnostics(
                    selected_row,
                    {
                        **common_diagnostics,
                        "ood_alpha": selected_diagnostic["ood_alpha"],
                        "correction": selected_correction,
                    },
                )
            )

            checkpoint, potential_model = potential_models[seed]
            potentials = response_values(
                potential_model,
                job_profiles[victim_id],
                np.vstack([anchor_profiles, query_profile[None, :]]),
                device,
            )
            support_potential = potentials[:-1]
            query_potential = potentials[-1:]
            legacy_correction, legacy_diagnostic = legacy_exact_corrections(
                observed,
                support_potential,
                distances,
                diagnostic_calibration,
                ood=legacy_ood,
            )["ood"]
            legacy_row = _base_row(
                query,
                "delta2_profile_only_ood_h32_e16",
                float(query_potential[0] + legacy_correction),
            )
            legacy_row["seed"] = seed
            seed_rows.append(
                _with_diagnostics(
                    legacy_row,
                    {
                        **common_diagnostics,
                        "ood_alpha": legacy_diagnostic["ood_alpha"],
                        "correction": legacy_correction,
                    },
                )
            )

            zero_latent, zero_fit, _ = censored_potential_predictions(
                observed,
                support_potential,
                query_potential,
                shrinkage=float(TRAINING_SETTINGS["censored_shrinkage"]),
                prior=0.0,
                query_gamma=1.0,
            )
            if not zero_fit.converged:
                raise RuntimeError("zero-prior censored correction did not converge")
            zero_row = _base_row(
                query,
                "delta2_censored_zero_h32_e16",
                float(zero_latent[0]),
            )
            zero_row["seed"] = seed
            seed_rows.append(
                _with_diagnostics(
                    zero_row,
                    {
                        **common_diagnostics,
                        "correction": zero_fit.correction,
                        "correction_prior": zero_fit.prior,
                        "correction_active_floor_violations": zero_fit.active_floor_violations,
                    },
                )
            )

            prior = float(checkpoint["population_correction_prior"])
            shrunk_latent, shrunk_fit, _ = censored_potential_predictions(
                observed,
                support_potential,
                query_potential,
                shrinkage=float(TRAINING_SETTINGS["censored_shrinkage"]),
                prior=prior,
                query_gamma=float(TRAINING_SETTINGS["query_gamma"]),
            )
            if not shrunk_fit.converged:
                raise RuntimeError("query-shrinkage correction did not converge")
            shrunk_row = _base_row(
                query,
                "delta2_query_shrinkage_h32_e16",
                float(shrunk_latent[0]),
            )
            shrunk_row["seed"] = seed
            seed_rows.append(
                _with_diagnostics(
                    shrunk_row,
                    {
                        **common_diagnostics,
                        "correction": shrunk_fit.correction,
                        "correction_prior": shrunk_fit.prior,
                        "correction_active_floor_violations": shrunk_fit.active_floor_violations,
                    },
                )
            )

            absolute_latent = float(
                absolute_values(
                    absolute_models[seed],
                    job_profiles[victim_id][None, :],
                    query_profile[None, :],
                    device,
                )[0]
            )
            absolute_row = _base_row(
                query, "delta2_absolute_neural_h32_e16", absolute_latent
            )
            absolute_row["seed"] = seed
            seed_rows.append(_with_diagnostics(absolute_row, {}))

    seed_frame = pd.DataFrame(seed_rows)
    deterministic_frame = pd.DataFrame(deterministic_rows)
    expected_seed_rows = len(query_frame) * len(SEEDED_METHODS) * len(SEEDS)
    expected_deterministic_rows = len(query_frame) * len(DETERMINISTIC_METHODS)
    if len(seed_frame) != expected_seed_rows:
        raise RuntimeError("seed prediction row count is incomplete")
    if len(deterministic_frame) != expected_deterministic_rows:
        raise RuntimeError("deterministic prediction row count is incomplete")

    ensemble_rows: list[dict[str, object]] = []
    grouping = [*KEY_COLUMNS, "method", "candidate_id"]
    for keys, group in seed_frame.groupby(grouping, sort=False):
        seeds = sorted(group["seed"].astype(int).tolist())
        if seeds != list(SEEDS):
            raise RuntimeError("seeded direction does not contain seeds 0 through 4")
        latent_values = group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
        latent = float(np.mean(latent_values))
        observed, slowdown = _prediction_values(latent)
        row = dict(zip(grouping, keys))
        row.update(
            {
                "latent_predicted_log_slowdown": latent,
                "predicted_log_slowdown": observed,
                "predicted_slowdown": slowdown,
                "ensemble_seed_count": 5,
                "model_seed_log_std": float(np.std(latent_values, ddof=0)),
                "evidence_label": EVIDENCE_LABEL,
            }
        )
        for diagnostic in CORRECTION_DIAGNOSTIC_COLUMNS:
            values = pd.to_numeric(group[diagnostic], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[diagnostic] = float(np.mean(finite)) if len(finite) else None
        ensemble_rows.append(row)

    for deterministic in deterministic_frame.to_dict(orient="records"):
        deterministic["ensemble_seed_count"] = 1
        deterministic["model_seed_log_std"] = 0.0
        ensemble_rows.append(deterministic)
    directional_frame = pd.DataFrame(ensemble_rows)

    _validate_prediction_invariants(seed_frame, directional_frame, query_frame)
    seed_order = [
        *KEY_COLUMNS,
        "method",
        "candidate_id",
        "seed",
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
        *CORRECTION_DIAGNOSTIC_COLUMNS,
        "evidence_label",
    ]
    directional_order = [
        *KEY_COLUMNS,
        "method",
        "candidate_id",
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
        "ensemble_seed_count",
        "model_seed_log_std",
        *CORRECTION_DIAGNOSTIC_COLUMNS,
        "evidence_label",
    ]
    return (
        seed_frame.loc[:, seed_order].sort_values(
            ["pair_row_id", "direction", "method", "seed"], kind="stable"
        ).reset_index(drop=True),
        directional_frame.loc[:, directional_order].sort_values(
            ["pair_row_id", "direction", "method"], kind="stable"
        ).reset_index(drop=True),
    )


def _validate_prediction_invariants(
    seed_frame: pd.DataFrame,
    directional_frame: pd.DataFrame,
    queries: pd.DataFrame,
) -> None:
    required_numeric = (
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
    )
    for frame in (seed_frame, directional_frame):
        if not np.isfinite(frame[list(required_numeric)].to_numpy(dtype=float)).all():
            raise FloatingPointError("prediction table contains non-finite required values")
        if (frame["predicted_log_slowdown"] < 0).any() or (
            frame["predicted_slowdown"] < 1
        ).any():
            raise RuntimeError("prediction table left observed slowdown support")
        for diagnostic in CORRECTION_DIAGNOSTIC_COLUMNS:
            values = pd.to_numeric(frame[diagnostic], errors="coerce")
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise FloatingPointError(f"diagnostic {diagnostic} contains infinity")

    expected_query_keys = set(
        queries[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    if set(directional_frame["method"].astype(str)) != set(DELTA2_METHODS):
        raise RuntimeError("directional table has the wrong Delta Response 2 method roster")
    for method in DELTA2_METHODS:
        frame = directional_frame[directional_frame["method"].eq(method)]
        keys = set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
        if len(frame) != len(queries) or keys != expected_query_keys:
            raise RuntimeError(f"directional prediction coverage is incomplete for {method}")
    for method in SEEDED_METHODS:
        frame = seed_frame[seed_frame["method"].eq(method)]
        counts = frame.groupby(list(KEY_COLUMNS))["seed"].agg(list)
        if len(counts) != len(queries) or any(sorted(value) != list(SEEDS) for value in counts):
            raise RuntimeError(f"seed prediction coverage is incomplete for {method}")

    for method in DELTA2_METHODS:
        self_rows = directional_frame[
            directional_frame["method"].eq(method)
            & directional_frame["self_pair"].astype(bool)
        ]
        for _, group in self_rows.groupby("pair_row_id", sort=False):
            if len(group) != 2 or np.ptp(
                group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
            ) > 1e-12:
                raise RuntimeError(f"self directions differ for method {method}")
    for method in SEEDED_METHODS:
        self_rows = seed_frame[
            seed_frame["method"].eq(method) & seed_frame["self_pair"].astype(bool)
        ]
        for _, group in self_rows.groupby(["pair_row_id", "seed"], sort=False):
            if len(group) != 2 or np.ptp(
                group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
            ) > 1e-12:
                raise RuntimeError(f"seeded self directions differ for method {method}")
