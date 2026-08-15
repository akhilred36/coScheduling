"""End-to-end transitive delta-response experiment.

pair.csv is deliberately not opened until cross-validation, model selection,
aggregation tuning, and final model fitting have all completed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from data import (
    BASE_FEATURES,
    EVALUATION_MODES,
    ProfileScaler,
    TrainingData,
    expand_directional_pairs,
    log_slowdown_to_slowdown,
    load_pair_holdout,
    load_training_data,
    observe_log_slowdown,
    resolve_evaluation_methods,
    select_evaluation_pairs,
    subset_training_data,
)
from inference import (
    OODConfig,
    aggregate_residuals,
    predict_with_anchors,
    rank_inference_anchors,
)
from metrics import (
    cluster_bootstrap,
    metrics_by_method,
    paired_cluster_bootstrap_differences,
    per_victim_metrics,
    self_pair_averaged,
)
from model import AbsoluteResponseModel, GenericPotential, LowRankPotential
from training import (
    profile_maps,
    response_values,
    set_seed,
    train_absolute,
    train_potential,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "few_shot" / "data"
DEFAULT_TRAINING_APPS = [
    "amg",
    "beatnik",
    "fiesta",
    "laghos",
    "lammps",
    "minife",
    "minivite",
]


@dataclass
class ModelConfig:
    feature_set: str = "base"
    hidden_dim: int = 8
    embedding_dim: int = 4
    rank: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class RunConfig:
    evaluation_method: str = "random_split"
    training_apps: list[str] | None = None
    inhibitor_blocks: int = 4
    clustering_algorithm: str = "kmeans"
    clustering_seed: int = 1701
    cv_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    final_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model_candidates: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    model_candidates_by_kind: dict[str, list[ModelConfig]] | None = None
    max_epochs: int = 80
    patience: int = 10
    batches_per_epoch: int = 4
    batch_size: int = 128
    temperatures: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0])
    ood_quantiles: list[float] = field(default_factory=lambda: [0.9, 0.95, 0.99])
    ood_fallbacks: list[str] = field(default_factory=lambda: ["uniform", "median"])
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 923
    device: str = "auto"
    inference_anchor_counts: list[int | str] = field(default_factory=lambda: ["all"])
    inference_anchor_seed: int = 1701


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def load_run_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    raw = json.loads(path.read_text())
    candidates = [ModelConfig(**item) for item in raw.pop("model_candidates", [{}])]
    candidates_by_kind = raw.pop("model_candidates_by_kind", None)
    if candidates_by_kind is not None:
        candidates_by_kind = {
            kind: [ModelConfig(**item) for item in values]
            for kind, values in candidates_by_kind.items()
        }
    return RunConfig(
        model_candidates=candidates,
        model_candidates_by_kind=candidates_by_kind,
        **raw,
    )


def model_candidates_for_kind(config: RunConfig, kind: str) -> list[ModelConfig]:
    if config.model_candidates_by_kind is None:
        return config.model_candidates
    unknown = set(config.model_candidates_by_kind) - {"low_rank", "generic", "absolute"}
    if unknown:
        raise ValueError(f"Unknown model candidate kinds: {sorted(unknown)}")
    candidates = config.model_candidates_by_kind.get(kind)
    if not candidates:
        raise ValueError(f"No model candidates configured for {kind}")
    return candidates


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_training_apps(data: TrainingData, config: RunConfig) -> list[str]:
    """Resolve the applications allowed to influence fitted model parameters."""
    available = list(data.jobs["job_id"].astype(str))
    if config.evaluation_method == "random_split":
        if config.training_apps is not None and set(config.training_apps) != set(available):
            raise ValueError(
                "random_split requires every application to be available for training"
            )
        return available
    requested = config.training_apps
    if requested is None:
        if set(DEFAULT_TRAINING_APPS).issubset(available):
            requested = DEFAULT_TRAINING_APPS
        else:
            raise ValueError(
                "--training-apps is required for one_known and zero_shot on this dataset"
            )
    resolved = list(dict.fromkeys(str(value) for value in requested))
    unknown = sorted(set(resolved) - set(available))
    if unknown:
        raise ValueError(f"Unknown training applications: {unknown}")
    if len(resolved) < 2:
        raise ValueError("At least two training applications are required")
    if set(resolved) == set(available):
        raise ValueError(
            f"{config.evaluation_method} requires at least one unknown application"
        )
    return resolved


def resolve_anchor_budgets(values: list[int | str]) -> list[tuple[str, int | None]]:
    budgets: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and value.lower() == "all":
            label, count = "all", None
        else:
            try:
                count = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Inference anchor counts must be positive integers or 'all'"
                ) from error
            if count <= 0:
                raise ValueError("Inference anchor counts must be positive")
            label = str(count)
        if label in seen:
            raise ValueError(f"Duplicate inference anchor count: {label}")
        seen.add(label)
        budgets.append((label, count))
    if not budgets:
        raise ValueError("At least one inference anchor count is required")
    return budgets


def build_inhibitor_blocks(data: TrainingData, config: RunConfig) -> pd.DataFrame:
    if config.clustering_algorithm != "kmeans":
        raise ValueError("Only the recorded 'kmeans' clustering algorithm is supported")
    scaler = ProfileScaler("base").fit(data.inhibitors)
    profiles = scaler.transform(data.inhibitors)
    labels = KMeans(
        n_clusters=config.inhibitor_blocks,
        random_state=config.clustering_seed,
        n_init=20,
    ).fit_predict(profiles)
    return pd.DataFrame(
        {"inhib_id": data.inhibitors["inhib_id"].astype(str), "block": labels}
    )


def make_model(model_config: ModelConfig, kind: str = "low_rank") -> torch.nn.Module:
    arguments = {
        "input_dim": 4 if model_config.feature_set == "base" else 8,
        "hidden_dim": model_config.hidden_dim,
        "embedding_dim": model_config.embedding_dim,
        "rank": model_config.rank,
    }
    if kind == "low_rank":
        return LowRankPotential(**arguments)
    if kind == "generic":
        return GenericPotential(**arguments)
    if kind == "absolute":
        return AbsoluteResponseModel(**arguments)
    raise ValueError(f"Unknown model kind: {kind}")


def _fold_uniform_log_mae(
    model: torch.nn.Module,
    victim_profile: np.ndarray,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
) -> float:
    support = support[support["log_slowdown"] > 0]
    if support.empty:
        return float("inf")
    support_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    values = response_values(
        model, victim_profile, np.vstack([support_profiles, query_profiles]), device
    )
    support_potential = values[: len(support)]
    query_potential = values[len(support) :]
    correction = np.mean(support["log_slowdown"].to_numpy() - support_potential)
    predicted = observe_log_slowdown(query_potential + correction)
    return float(np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy())))


def _fold_absolute_log_mae(
    model: torch.nn.Module,
    victim_profile: np.ndarray,
    queries: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
) -> float:
    query_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    model.eval()
    victims = np.repeat(victim_profile[None, :], len(queries), axis=0)
    with torch.no_grad():
        latent = model(
            torch.as_tensor(victims, dtype=torch.float32, device=device),
            torch.as_tensor(query_profiles, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    predicted = observe_log_slowdown(latent)
    return float(np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy())))


def _record_fold_predictions(
    records: list[dict[str, Any]],
    model: torch.nn.Module,
    victim_id: str,
    block: int,
    seed: int,
    config_id: int,
    model_kind: str,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    distance_inhibitor_profiles: dict[str, np.ndarray],
    reference: np.ndarray,
    run_config: RunConfig,
    device: torch.device,
) -> None:
    support = support[support["log_slowdown"] > 0].reset_index(drop=True)
    if support.empty:
        return
    specifications: list[tuple[str, float, float, str]] = [
        ("uniform", 1.0, 0.95, "median"),
        ("median", 1.0, 0.95, "median"),
    ]
    specifications.extend(
        ("kernel", temperature, 0.95, "median")
        for temperature in run_config.temperatures
    )
    specifications.extend(
        ("ood_kernel", temperature, quantile, fallback)
        for temperature in run_config.temperatures
        for quantile in run_config.ood_quantiles
        for fallback in run_config.ood_fallbacks
    )
    anchor_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in support["inhib_id"]]
    )
    query_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    distance_anchor_profiles = np.stack(
        [distance_inhibitor_profiles[str(value)] for value in support["inhib_id"]]
    )
    distance_query_profiles = np.stack(
        [distance_inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
    )
    potentials = response_values(
        model,
        job_profiles[victim_id],
        np.vstack([anchor_profiles, query_profiles]),
        device,
    )
    residuals = support["log_slowdown"].to_numpy() - potentials[: len(support)]
    query_potentials = potentials[len(support) :]
    for query_index, (_, query) in enumerate(queries.iterrows()):
        distances = np.linalg.norm(
            distance_anchor_profiles - distance_query_profiles[query_index][None, :],
            axis=1,
        )
        for method, temperature, quantile, fallback in specifications:
            correction, _, _ = aggregate_residuals(
                residuals,
                distances,
                method=method,
                temperature=temperature,
                reference_distances=reference,
                ood_config=OODConfig(quantile, fallback),
            )
            latent_log_slowdown = float(query_potentials[query_index] + correction)
            predicted_log_slowdown = float(observe_log_slowdown(latent_log_slowdown))
            records.append(
                {
                    "config_id": config_id,
                    "model_kind": model_kind,
                    "victim_id": victim_id,
                    "heldout_block": block,
                    "seed": seed,
                    "query_inhib_id": str(query["inhib_id"]),
                    "method": method,
                    "temperature": temperature,
                    "ood_quantile": quantile,
                    "ood_fallback": fallback,
                    "nearest_anchor_distance": float(distances.min()),
                    "true_log_slowdown": float(query["log_slowdown"]),
                    "latent_predicted_log_slowdown": latent_log_slowdown,
                    "predicted_log_slowdown": predicted_log_slowdown,
                    "absolute_log_error": abs(
                        predicted_log_slowdown - query["log_slowdown"]
                    ),
                }
            )


def _common_distance_reference(
    data: TrainingData,
    blocks: pd.DataFrame,
    distance_inhibitor_profiles: dict[str, np.ndarray],
) -> np.ndarray:
    block_by_id = dict(zip(blocks["inhib_id"], blocks["block"]))
    responses = data.responses.assign(
        block=data.responses["inhib_id"].map(block_by_id).astype(int)
    )
    distances = []
    for victim_id in data.jobs["job_id"].astype(str):
        for block in sorted(blocks["block"].unique()):
            support = responses[
                (responses["job_id"] == victim_id)
                & (responses["block"] != block)
                & (responses["log_slowdown"] > 0)
            ].drop_duplicates("inhib_id")
            queries = responses[
                (responses["job_id"] == victim_id) & (responses["block"] == block)
            ].drop_duplicates("inhib_id")
            if support.empty or queries.empty:
                continue
            support_profiles = np.stack(
                [distance_inhibitor_profiles[str(value)] for value in support["inhib_id"]]
            )
            query_profiles = np.stack(
                [distance_inhibitor_profiles[str(value)] for value in queries["inhib_id"]]
            )
            distances.extend(
                np.linalg.norm(
                    query_profiles[:, None, :] - support_profiles[None, :, :], axis=-1
                ).min(axis=1)
            )
    reference = np.asarray(distances, dtype=float)
    if not len(reference) or not np.isfinite(reference).all():
        raise RuntimeError("Could not construct a finite common OOD distance reference")
    return reference


def _candidate_statistics(frame: pd.DataFrame) -> tuple[float, float, int]:
    fold_errors = (
        frame.groupby(["victim_id", "heldout_block", "seed"], as_index=False)[
            "absolute_log_error"
        ]
        .mean()
        .groupby(["victim_id", "heldout_block"])["absolute_log_error"]
        .mean()
    )
    mean = float(fold_errors.mean())
    standard_error = (
        float(fold_errors.std(ddof=1) / np.sqrt(len(fold_errors)))
        if len(fold_errors) > 1
        else 0.0
    )
    return mean, standard_error, len(fold_errors)


def _select_aggregation(predictions: pd.DataFrame) -> dict[str, Any]:
    """Deterministic grouped one-standard-error aggregation selection."""
    candidates: list[dict[str, Any]] = []
    for complexity, method in enumerate(["uniform", "median"]):
        rows = predictions[predictions["method"] == method]
        mean, standard_error, groups = _candidate_statistics(rows)
        candidates.append(
            {
                "method": method,
                "complexity": complexity,
                "parameters": {},
                "validation_log_mae": mean,
                "grouped_standard_error": standard_error,
                "validation_group_count": groups,
                "parameter_order": 0,
            }
        )
    for parameter_order, temperature in enumerate(
        sorted(predictions.loc[predictions["method"] == "kernel", "temperature"].unique())
    ):
        rows = predictions[
            (predictions["method"] == "kernel")
            & (predictions["temperature"] == temperature)
        ]
        mean, standard_error, groups = _candidate_statistics(rows)
        candidates.append(
            {
                "method": "kernel",
                "complexity": 2,
                "parameters": {"temperature": float(temperature)},
                "validation_log_mae": mean,
                "grouped_standard_error": standard_error,
                "validation_group_count": groups,
                "parameter_order": parameter_order,
            }
        )
    ood_values = predictions[predictions["method"] == "ood_kernel"][
        ["temperature", "ood_quantile", "ood_fallback"]
    ].drop_duplicates()
    ood_values = ood_values.sort_values(
        ["temperature", "ood_quantile", "ood_fallback"]
    )
    for parameter_order, values in enumerate(ood_values.itertuples(index=False)):
        rows = predictions[
            (predictions["method"] == "ood_kernel")
            & (predictions["temperature"] == values.temperature)
            & (predictions["ood_quantile"] == values.ood_quantile)
            & (predictions["ood_fallback"] == values.ood_fallback)
        ]
        mean, standard_error, groups = _candidate_statistics(rows)
        candidates.append(
            {
                "method": "ood_kernel",
                "complexity": 3,
                "parameters": {
                    "temperature": float(values.temperature),
                    "quantile": float(values.ood_quantile),
                    "fallback": str(values.ood_fallback),
                },
                "validation_log_mae": mean,
                "grouped_standard_error": standard_error,
                "validation_group_count": groups,
                "parameter_order": parameter_order,
            }
        )
    selected_by_method = {}
    for method in ["uniform", "median", "kernel", "ood_kernel"]:
        family = [candidate for candidate in candidates if candidate["method"] == method]
        best = min(family, key=lambda value: value["validation_log_mae"])
        threshold = best["validation_log_mae"] + best["grouped_standard_error"]
        selected_by_method[method] = min(
            [value for value in family if value["validation_log_mae"] <= threshold],
            key=lambda value: value["parameter_order"],
        )
    family_best = list(selected_by_method.values())
    numerical_best = min(family_best, key=lambda value: value["validation_log_mae"])
    threshold = (
        numerical_best["validation_log_mae"]
        + numerical_best["grouped_standard_error"]
    )
    primary = min(
        [value for value in family_best if value["validation_log_mae"] <= threshold],
        key=lambda value: (value["complexity"], value["parameter_order"]),
    )
    result: dict[str, Any] = {
        method: {
            **value["parameters"],
            "validation_log_mae": value["validation_log_mae"],
            "grouped_standard_error": value["grouped_standard_error"],
            "validation_group_count": value["validation_group_count"],
        }
        for method, value in selected_by_method.items()
    }
    result["primary_method"] = primary["method"]
    result["selection_rule"] = {
        "name": "grouped_one_standard_error",
        "grouping": "victim_id and heldout_block after seed averaging",
        "complexity_order": ["uniform", "median", "kernel", "ood_kernel"],
        "threshold": threshold,
        "numerical_best_method": numerical_best["method"],
    }
    return result


def crossed_validation(
    data: TrainingData,
    blocks: pd.DataFrame,
    config: RunConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[
    dict[str, ModelConfig],
    dict[str, dict[str, Any]],
    dict[str, int],
    pd.DataFrame,
    np.ndarray,
    ProfileScaler,
]:
    block_by_id = dict(zip(blocks["inhib_id"], blocks["block"]))
    responses = data.responses.assign(
        block=data.responses["inhib_id"].map(block_by_id).astype(int)
    )
    prediction_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    distance_scaler = ProfileScaler("base").fit(data.inhibitors)
    _, distance_inhibitor_profiles = profile_maps(
        data.jobs, data.inhibitors, distance_scaler
    )
    distance_reference = _common_distance_reference(
        data, blocks, distance_inhibitor_profiles
    )
    profile_cache: dict[
        tuple[str, str, int], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = {}

    model_kinds = ["low_rank", "generic", "absolute"]
    candidate_lists = {
        kind: model_candidates_for_kind(config, kind) for kind in model_kinds
    }
    valid_fold_groups = 0
    for victim_id in data.jobs["job_id"].astype(str):
        for block in sorted(blocks["block"].unique()):
            victim_rows = responses[responses["job_id"] == victim_id]
            has_support = victim_rows["block"].ne(block).any()
            has_queries = victim_rows["block"].eq(block).any()
            if has_support and has_queries:
                valid_fold_groups += 1
    total_fits = (
        valid_fold_groups
        * len(config.cv_seeds)
        * sum(len(values) for values in candidate_lists.values())
    )
    completed_fits = 0
    progress_started = time.monotonic()
    next_progress = progress_started + 30.0
    for config_id in range(max(len(values) for values in candidate_lists.values())):
        for victim_id in data.jobs["job_id"].astype(str):
            for block in sorted(blocks["block"].unique()):
                retained_inhibitors = data.inhibitors[
                    data.inhibitors["inhib_id"].map(block_by_id) != block
                ]
                train_jobs = data.jobs[data.jobs["job_id"] != victim_id]
                train_rows = responses[
                    (responses["job_id"] != victim_id)
                    & (responses["block"] != block)
                ]
                support = responses[
                    (responses["job_id"] == victim_id)
                    & (responses["block"] != block)
                ]
                queries = responses[
                    (responses["job_id"] == victim_id)
                    & (responses["block"] == block)
                ]
                if support.empty or queries.empty:
                    continue
                for seed in config.cv_seeds:
                    for model_kind in model_kinds:
                        if config_id >= len(candidate_lists[model_kind]):
                            continue
                        model_config = candidate_lists[model_kind][config_id]
                        profile_key = (
                            model_config.feature_set,
                            victim_id,
                            int(block),
                        )
                        if profile_key not in profile_cache:
                            scaler = ProfileScaler(model_config.feature_set).fit(
                                pd.concat(
                                    [train_jobs, retained_inhibitors],
                                    ignore_index=True,
                                )
                            )
                            profile_cache[profile_key] = profile_maps(
                                data.jobs, data.inhibitors, scaler
                            )
                        job_profiles, inhibitor_profiles = profile_cache[profile_key]
                        set_seed(seed)
                        model = make_model(model_config, model_kind)
                        if model_kind == "absolute":
                            validation = lambda candidate: _fold_absolute_log_mae(
                                candidate,
                                job_profiles[victim_id],
                                queries,
                                inhibitor_profiles,
                                device,
                            )
                            training_result = train_absolute(
                                model,
                                train_rows,
                                job_profiles,
                                inhibitor_profiles,
                                seed=seed,
                                epochs=config.max_epochs,
                                batches_per_epoch=config.batches_per_epoch,
                                batch_size=config.batch_size,
                                learning_rate=model_config.learning_rate,
                                weight_decay=model_config.weight_decay,
                                device=device,
                                validation_fn=validation,
                                patience=config.patience,
                            )
                        else:
                            validation = lambda candidate: _fold_uniform_log_mae(
                                candidate,
                                job_profiles[victim_id],
                                support,
                                queries,
                                inhibitor_profiles,
                                device,
                            )
                            training_result = train_potential(
                                model,
                                train_rows,
                                job_profiles,
                                inhibitor_profiles,
                                seed=seed,
                                epochs=config.max_epochs,
                                batches_per_epoch=config.batches_per_epoch,
                                batch_size=config.batch_size,
                                learning_rate=model_config.learning_rate,
                                weight_decay=model_config.weight_decay,
                                device=device,
                                validation_fn=validation,
                                patience=config.patience,
                            )
                        fold_records.append(
                            {
                                "config_id": config_id,
                                "model_kind": model_kind,
                                "victim_id": victim_id,
                                "heldout_block": block,
                                "seed": seed,
                                "support_count": len(support)
                                if model_kind != "absolute"
                                else 0,
                                "uncensored_support_count": int(
                                    (support["log_slowdown"] > 0).sum()
                                )
                                if model_kind != "absolute"
                                else 0,
                                "query_count": len(queries),
                                **{
                                    key: training_result[key]
                                    for key in [
                                        "best_epoch",
                                        "best_validation",
                                        "epochs_run",
                                    ]
                                },
                            }
                        )
                        if model_kind != "absolute":
                            _record_fold_predictions(
                                prediction_records,
                                model,
                                victim_id,
                                int(block),
                                seed,
                                config_id,
                                model_kind,
                                support,
                                queries,
                                job_profiles,
                                inhibitor_profiles,
                                distance_inhibitor_profiles,
                                distance_reference,
                                config,
                                device,
                            )
                        completed_fits += 1
                        now = time.monotonic()
                        if (
                            completed_fits == 1
                            or completed_fits == total_fits
                            or now >= next_progress
                        ):
                            elapsed = now - progress_started
                            print(
                                "Cross-validation progress: "
                                f"{completed_fits}/{total_fits} fits "
                                f"({100.0 * completed_fits / total_fits:.1f}%), "
                                f"elapsed {elapsed:.0f}s",
                                flush=True,
                            )
                            next_progress = now + 30.0

    predictions = pd.DataFrame(prediction_records)
    folds = pd.DataFrame(fold_records)
    if predictions.empty:
        raise RuntimeError("Crossed validation produced no predictions")
    predictions.to_csv(output_dir / "crossed_validation_predictions.csv", index=False)
    folds.to_csv(output_dir / "crossed_validation_folds.csv", index=False)

    pd.DataFrame({"nearest_anchor_distance": distance_reference}).to_csv(
        output_dir / "validation_distance_reference.csv", index=False
    )
    selected_models: dict[str, ModelConfig] = {}
    selected_ids: dict[str, int] = {}
    final_epochs: dict[str, int] = {}
    aggregations: dict[str, dict[str, Any]] = {}
    for model_kind in ["low_rank", "generic", "absolute"]:
        kind_folds = folds[folds["model_kind"] == model_kind]
        config_scores = kind_folds.groupby("config_id")["best_validation"].mean()
        selected_id = int(config_scores.idxmin())
        selected_ids[model_kind] = selected_id
        selected_models[model_kind] = model_candidates_for_kind(config, model_kind)[
            selected_id
        ]
        selected_fold_epochs = kind_folds[kind_folds["config_id"] == selected_id][
            "best_epoch"
        ]
        final_epochs[model_kind] = max(1, int(np.median(selected_fold_epochs)))
        if model_kind != "absolute":
            selected_predictions = predictions[
                (predictions["model_kind"] == model_kind)
                & (predictions["config_id"] == selected_id)
            ]
            aggregations[model_kind] = _select_aggregation(selected_predictions)
            aggregations[model_kind]["model_config_validation_log_mae"] = float(
                config_scores.loc[selected_id]
            )
            aggregations[model_kind]["selected_config_id"] = selected_id
    summary = predictions.groupby(
        [
            "config_id",
            "model_kind",
            "method",
            "temperature",
            "ood_quantile",
            "ood_fallback",
        ],
        as_index=False,
    )["absolute_log_error"].mean()
    summary.to_csv(output_dir / "crossed_validation_summary.csv", index=False)
    return (
        selected_models,
        aggregations,
        final_epochs,
        folds,
        distance_reference,
        distance_scaler,
    )


def train_final_models(
    data: TrainingData,
    model_configs: dict[str, ModelConfig],
    run_config: RunConfig,
    final_epochs: dict[str, int],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, ProfileScaler], dict[int, dict[str, torch.nn.Module]]]:
    scalers = {
        kind: ProfileScaler(model_config.feature_set).fit(
            pd.concat([data.jobs, data.inhibitors], ignore_index=True)
        )
        for kind, model_config in model_configs.items()
    }
    profile_values = {
        kind: profile_maps(data.jobs, data.inhibitors, scaler)
        for kind, scaler in scalers.items()
    }
    models: dict[int, dict[str, torch.nn.Module]] = {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in run_config.final_seeds:
        seed_models: dict[str, torch.nn.Module] = {}
        for kind in ["low_rank", "generic"]:
            set_seed(seed)
            model = make_model(model_configs[kind], kind)
            job_profiles, inhibitor_profiles = profile_values[kind]
            train_potential(
                model,
                data.responses,
                job_profiles,
                inhibitor_profiles,
                seed=seed,
                epochs=final_epochs[kind],
                batches_per_epoch=run_config.batches_per_epoch,
                batch_size=run_config.batch_size,
                learning_rate=model_configs[kind].learning_rate,
                weight_decay=model_configs[kind].weight_decay,
                device=device,
            )
            seed_models[kind] = model
        set_seed(seed)
        absolute = make_model(model_configs["absolute"], "absolute")
        job_profiles, inhibitor_profiles = profile_values["absolute"]
        train_absolute(
            absolute,
            data.responses,
            job_profiles,
            inhibitor_profiles,
            seed=seed,
            epochs=final_epochs["absolute"],
            batches_per_epoch=run_config.batches_per_epoch,
            batch_size=run_config.batch_size,
            learning_rate=model_configs["absolute"].learning_rate,
            weight_decay=model_configs["absolute"].weight_decay,
            device=device,
        )
        seed_models["absolute"] = absolute
        models[seed] = seed_models
        for kind, model in seed_models.items():
            torch.save(
                {
                    "kind": kind,
                    "seed": seed,
                    "epochs": final_epochs[kind],
                    "training_apps": data.jobs["job_id"].astype(str).tolist(),
                    "model_config": asdict(model_configs[kind]),
                    "scaler": scalers[kind].state_dict(),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_dir / f"{kind}_seed{seed}.pt",
            )
    return scalers, models


def _common_diagnostics(
    target_profile: np.ndarray,
    anchors: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    reference: np.ndarray,
) -> dict[str, float]:
    usable = anchors[anchors["log_slowdown"] > 0]
    if usable.empty:
        raise ValueError("At least one uncensored anchor is required for inference")
    anchor_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in usable["inhib_id"]]
    )
    distances = np.linalg.norm(anchor_profiles - target_profile[None, :], axis=1)
    nearest = float(distances.min())
    return {
        "number_of_anchors": len(anchors),
        "uncensored_anchor_count": len(usable),
        "nearest_anchor_distance": nearest,
        "effective_anchor_count": np.nan,
        "weighted_residual_std": np.nan,
        "kernel_component_effective_anchor_count": np.nan,
        "kernel_component_weighted_residual_std": np.nan,
        "unweighted_residual_std": np.nan,
        "fallback_residual_mad": np.nan,
        "ood_score": float(np.mean(reference <= nearest)),
        "ood_alpha": np.nan,
    }


def evaluate_holdout(
    data: TrainingData,
    pairs: pd.DataFrame,
    scalers: dict[str, ProfileScaler],
    models: dict[int, dict[str, torch.nn.Module]],
    aggregations: dict[str, dict[str, Any]],
    distance_reference: np.ndarray,
    distance_scaler: ProfileScaler,
    device: torch.device,
    evaluation_method: str = "random_split",
    known_apps: list[str] | set[str] | None = None,
    inference_anchor_counts: list[int | str] | None = None,
    inference_anchor_seed: int = 1701,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directional = expand_directional_pairs(pairs)
    known = (
        set(data.jobs["job_id"].astype(str))
        if known_apps is None
        else {str(value) for value in known_apps}
    )
    profiles = {
        kind: profile_maps(data.jobs, data.inhibitors, scaler)
        for kind, scaler in scalers.items()
    }
    distance_job_profiles, distance_inhibitor_profiles = profile_maps(
        data.jobs, data.inhibitors, distance_scaler
    )
    reference = distance_reference
    anchors_by_job = {
        str(job_id): group.reset_index(drop=True)
        for job_id, group in data.responses.groupby("job_id", sort=False)
    }
    low_rank_aggregation = aggregations["low_rank"]
    low_rank_specs = {
        "delta_single_anchor": ("single", 1.0, OODConfig()),
        "delta_uniform": ("uniform", 1.0, OODConfig()),
        "delta_median": ("median", 1.0, OODConfig()),
        "delta_kernel": (
            "kernel",
            low_rank_aggregation["kernel"]["temperature"],
            OODConfig(),
        ),
        "delta_ood_kernel": (
            "ood_kernel",
            low_rank_aggregation["ood_kernel"]["temperature"],
            OODConfig(
                low_rank_aggregation["ood_kernel"]["quantile"],
                low_rank_aggregation["ood_kernel"]["fallback"],
            ),
        ),
    }
    generic_aggregation = aggregations["generic"]
    primary = generic_aggregation["primary_method"]
    primary_parameters = generic_aggregation.get(primary, {})
    generic_spec = (
        primary,
        primary_parameters.get("temperature", 1.0),
        OODConfig(
            primary_parameters.get("quantile", 0.95),
            primary_parameters.get("fallback", "median"),
        ),
    )
    anchor_budgets = resolve_anchor_budgets(inference_anchor_counts or ["all"])
    rows: list[dict[str, Any]] = []
    for seed, seed_models in models.items():
        for _, pair in directional.iterrows():
            victim_id = str(pair["victim_id"])
            aggressor_id = str(pair["aggressor_id"])
            ranked_anchors = rank_inference_anchors(
                anchors_by_job[victim_id],
                victim_id=victim_id,
                seed=inference_anchor_seed,
            )
            available_anchor_count = len(ranked_anchors)
            available_inhibitor_count = int(ranked_anchors["inhib_id"].nunique())
            for anchor_budget, anchor_count in anchor_budgets:
                anchors = (
                    ranked_anchors
                    if anchor_count is None
                    else ranked_anchors.iloc[:anchor_count].reset_index(drop=True)
                )
                low_job_profiles, low_inhibitor_profiles = profiles["low_rank"]
                victim_profile = low_job_profiles[victim_id]
                target_profile = low_job_profiles[aggressor_id]
                distance_target_profile = distance_job_profiles[aggressor_id]
                common = _common_diagnostics(
                    distance_target_profile,
                    anchors,
                    distance_inhibitor_profiles,
                    reference,
                )
                common["available_uncensored_anchor_count"] = available_anchor_count
                common["available_uncensored_inhibitor_count"] = (
                    available_inhibitor_count
                )
                common["requested_uncensored_anchor_count"] = (
                    np.nan if anchor_count is None else anchor_count
                )
                common["anchor_budget_feasible"] = (
                    True if anchor_count is None else available_anchor_count >= anchor_count
                )
                common["uncensored_anchor_inhibitor_count"] = int(
                    anchors["inhib_id"].nunique()
                )
                base = {
                    "evaluation_method": evaluation_method,
                    "anchor_budget": anchor_budget,
                    "seed": seed,
                    "pair_row_id": int(pair["pair_row_id"]),
                    "pair_cluster_id": int(pair["pair_cluster_id"]),
                    "direction": pair["direction"],
                    "victim_id": victim_id,
                    "aggressor_id": aggressor_id,
                    "victim_known": victim_id in known,
                    "aggressor_known": aggressor_id in known,
                    "known_endpoint_count": int(victim_id in known)
                    + int(aggressor_id in known),
                    "true_slowdown": float(pair["true_slowdown"]),
                }

                baseline_predictions = {
                    "constant_1": 0.0,
                    "victim_inhibitor_median": float(
                        np.median(anchors["log_slowdown"].to_numpy())
                    ),
                }
                distance_anchor_profiles = np.stack(
                    [
                        distance_inhibitor_profiles[str(value)]
                        for value in anchors["inhib_id"]
                    ]
                )
                nearest_index = int(
                    np.argmin(
                        np.linalg.norm(
                            distance_anchor_profiles
                            - distance_target_profile[None, :],
                            axis=1,
                        )
                    )
                )
                baseline_predictions["nearest_anchor"] = float(
                    anchors.iloc[nearest_index]["log_slowdown"]
                )
                for method, latent_log in baseline_predictions.items():
                    method_diagnostics = dict(common)
                    if method == "nearest_anchor":
                        method_diagnostics["effective_anchor_count"] = 1.0
                        method_diagnostics["weighted_residual_std"] = 0.0
                    rows.append(
                        {
                            **base,
                            "method": method,
                            "latent_predicted_log_slowdown": latent_log,
                            "predicted_log_slowdown": float(
                                observe_log_slowdown(latent_log)
                            ),
                            **method_diagnostics,
                        }
                    )

                absolute = seed_models["absolute"]
                absolute_job_profiles, _ = profiles["absolute"]
                absolute.eval()
                with torch.no_grad():
                    absolute_prediction = absolute(
                        torch.as_tensor(
                            absolute_job_profiles[victim_id][None, :],
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            absolute_job_profiles[aggressor_id][None, :],
                            dtype=torch.float32,
                            device=device,
                        ),
                    ).item()
                rows.append(
                    {
                        **base,
                        "method": "absolute_response",
                        "latent_predicted_log_slowdown": absolute_prediction,
                        "predicted_log_slowdown": float(
                            observe_log_slowdown(absolute_prediction)
                        ),
                        **common,
                    }
                )

                for method_name, (method, temperature, ood_config) in low_rank_specs.items():
                    result = predict_with_anchors(
                        seed_models["low_rank"],
                        victim_profile,
                        target_profile,
                        anchors,
                        low_inhibitor_profiles,
                        method=method,
                        temperature=temperature,
                        reference_distances=reference,
                        ood_config=ood_config,
                        device=device,
                        distance_target_profile=distance_target_profile,
                        distance_inhibitor_profiles=distance_inhibitor_profiles,
                    )
                    result["available_uncensored_anchor_count"] = available_anchor_count
                    result.update(
                        {
                            key: common[key]
                            for key in [
                                "available_uncensored_inhibitor_count",
                                "requested_uncensored_anchor_count",
                                "anchor_budget_feasible",
                                "uncensored_anchor_inhibitor_count",
                            ]
                        }
                    )
                    rows.append({**base, "method": method_name, **result})

                method, temperature, ood_config = generic_spec
                generic_job_profiles, generic_inhibitor_profiles = profiles["generic"]
                result = predict_with_anchors(
                    seed_models["generic"],
                    generic_job_profiles[victim_id],
                    generic_job_profiles[aggressor_id],
                    anchors,
                    generic_inhibitor_profiles,
                    method=method,
                    temperature=temperature,
                    reference_distances=reference,
                    ood_config=ood_config,
                    device=device,
                    distance_target_profile=distance_target_profile,
                    distance_inhibitor_profiles=distance_inhibitor_profiles,
                )
                result["available_uncensored_anchor_count"] = available_anchor_count
                result.update(
                    {
                        key: common[key]
                        for key in [
                            "available_uncensored_inhibitor_count",
                            "requested_uncensored_anchor_count",
                            "anchor_budget_feasible",
                            "uncensored_anchor_inhibitor_count",
                        ]
                    }
                )
                rows.append({**base, "method": "generic_potential", **result})

    seed_predictions = pd.DataFrame(rows)
    seed_predictions["predicted_slowdown"] = log_slowdown_to_slowdown(
        seed_predictions["predicted_log_slowdown"].to_numpy()
    )
    keys = [
        "evaluation_method",
        "anchor_budget",
        "method",
        "pair_row_id",
        "pair_cluster_id",
        "direction",
        "victim_id",
        "aggressor_id",
        "victim_known",
        "aggressor_known",
        "known_endpoint_count",
        "true_slowdown",
    ]
    diagnostic_columns = [
        "number_of_anchors",
        "uncensored_anchor_count",
        "available_uncensored_anchor_count",
        "uncensored_anchor_inhibitor_count",
        "available_uncensored_inhibitor_count",
        "requested_uncensored_anchor_count",
        "anchor_budget_feasible",
        "nearest_anchor_distance",
        "effective_anchor_count",
        "weighted_residual_std",
        "kernel_component_effective_anchor_count",
        "kernel_component_weighted_residual_std",
        "unweighted_residual_std",
        "fallback_residual_mad",
        "ood_score",
        "ood_alpha",
    ]
    ensemble = seed_predictions.groupby(keys, as_index=False, sort=False).agg(
        latent_predicted_log_slowdown=("latent_predicted_log_slowdown", "mean"),
        model_seed_log_std=(
            "latent_predicted_log_slowdown",
            lambda values: float(np.std(values)),
        ),
        **{column: (column, "mean") for column in diagnostic_columns},
    )
    ensemble["predicted_log_slowdown"] = observe_log_slowdown(
        ensemble["latent_predicted_log_slowdown"].to_numpy()
    )
    ensemble["predicted_slowdown"] = log_slowdown_to_slowdown(
        ensemble["predicted_log_slowdown"].to_numpy()
    )
    ensemble["anchor_budget_feasible"] = ensemble["anchor_budget_feasible"].eq(1.0)
    ensemble["absolute_error"] = abs(
        ensemble["predicted_slowdown"] - ensemble["true_slowdown"]
    )
    ensemble["absolute_log_error"] = abs(
        ensemble["predicted_log_slowdown"] - np.log(ensemble["true_slowdown"])
    )
    return ensemble, seed_predictions


def write_evaluation_reports(
    predictions: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    config: RunConfig,
    output_dir: Path,
) -> None:
    evaluation_methods = predictions["evaluation_method"].drop_duplicates().tolist()
    if len(evaluation_methods) != 1:
        raise ValueError("Each evaluation report must contain exactly one evaluation method")
    evaluation_method = evaluation_methods[0]
    predictions.to_csv(output_dir / "directional_predictions.csv", index=False)
    seed_predictions.to_csv(output_dir / "seed_predictions.csv", index=False)
    metrics_by_method(predictions).to_csv(output_dir / "metrics_repeated_self.csv", index=False)
    averaged = self_pair_averaged(predictions)
    metrics_by_method(averaged).to_csv(output_dir / "metrics_self_averaged.csv", index=False)
    per_victim, macro = per_victim_metrics(predictions)
    per_victim.to_csv(output_dir / "metrics_per_victim.csv", index=False)
    macro.to_csv(output_dir / "metrics_macro_victim.csv", index=False)
    non_self = predictions[predictions["victim_id"] != predictions["aggressor_id"]]
    if not non_self.empty:
        metrics_by_method(non_self).to_csv(
            output_dir / "metrics_non_self.csv", index=False
        )
        non_self_per_victim, non_self_macro = per_victim_metrics(non_self)
        non_self_per_victim.to_csv(
            output_dir / "metrics_non_self_per_victim.csv", index=False
        )
        non_self_macro.to_csv(
            output_dir / "metrics_non_self_macro_victim.csv", index=False
        )
        non_self_bootstrap_frames = []
        non_self_difference_frames = []
        for anchor_budget, budget_predictions in non_self.groupby(
            "anchor_budget", sort=False
        ):
            non_self_bootstrap = cluster_bootstrap(
                budget_predictions,
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            )
            non_self_bootstrap.insert(0, "anchor_budget", anchor_budget)
            non_self_bootstrap_frames.append(non_self_bootstrap)
            non_self_differences = paired_cluster_bootstrap_differences(
                budget_predictions,
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            )
            non_self_differences.insert(0, "anchor_budget", anchor_budget)
            non_self_difference_frames.append(non_self_differences)
        non_self_bootstrap = pd.concat(non_self_bootstrap_frames, ignore_index=True)
        non_self_bootstrap.insert(0, "evaluation_method", evaluation_method)
        non_self_bootstrap.to_csv(
            output_dir / "cluster_bootstrap_non_self.csv", index=False
        )
        non_self_differences = pd.concat(
            non_self_difference_frames, ignore_index=True
        )
        non_self_differences.insert(0, "evaluation_method", evaluation_method)
        non_self_differences.to_csv(
            output_dir / "paired_method_differences_non_self.csv", index=False
        )

    bootstrap_frames = []
    difference_frames = []
    for anchor_budget, budget_predictions in predictions.groupby(
        "anchor_budget", sort=False
    ):
        bootstrap = cluster_bootstrap(
            budget_predictions,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
        bootstrap.insert(0, "anchor_budget", anchor_budget)
        bootstrap_frames.append(bootstrap)
        differences = paired_cluster_bootstrap_differences(
            budget_predictions,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
        differences.insert(0, "anchor_budget", anchor_budget)
        difference_frames.append(differences)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap.insert(0, "evaluation_method", evaluation_method)
    bootstrap.to_csv(output_dir / "cluster_bootstrap_confidence_intervals.csv", index=False)
    differences = pd.concat(difference_frames, ignore_index=True)
    differences.insert(0, "evaluation_method", evaluation_method)
    differences.to_csv(output_dir / "paired_method_differences.csv", index=False)

    seed_rows = []
    for (evaluation_method, anchor_budget, method, seed), group in seed_predictions.groupby(
        ["evaluation_method", "anchor_budget", "method", "seed"]
    ):
        result = metrics_by_method(group.assign(method=method)).iloc[0].to_dict()
        seed_rows.append(
            {
                "evaluation_method": evaluation_method,
                "anchor_budget": anchor_budget,
                "method": method,
                "seed": seed,
                **result,
            }
        )
    pd.DataFrame(seed_rows).to_csv(output_dir / "metrics_by_seed.csv", index=False)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(args.config)
    if args.cv_seeds is not None:
        config.cv_seeds = args.cv_seeds
    if args.final_seeds is not None:
        config.final_seeds = args.final_seeds
    if args.inhibitor_blocks is not None:
        config.inhibitor_blocks = args.inhibitor_blocks
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.patience is not None:
        config.patience = args.patience
    if args.batches_per_epoch is not None:
        config.batches_per_epoch = args.batches_per_epoch
    if args.bootstrap_samples is not None:
        config.bootstrap_samples = args.bootstrap_samples
    if getattr(args, "eval_method", None) is not None:
        config.evaluation_method = args.eval_method
    if getattr(args, "training_apps", None) is not None:
        config.training_apps = [
            value.strip()
            for value in args.training_apps.split(",")
            if value.strip()
        ]
    if config.evaluation_method not in EVALUATION_MODES:
        raise ValueError(
            f"evaluation_method must be one of {sorted(EVALUATION_MODES)}"
        )
    evaluation_methods = resolve_evaluation_methods(config.evaluation_method)
    resolve_anchor_budgets(config.inference_anchor_counts)
    device = resolve_device(config.device)

    print("Loading and validating App-Inhibitor training data (pair.csv remains sealed)")
    full_data = load_training_data(
        args.jobs_csv,
        args.inhibitors_csv,
        args.job_inh_csv,
        quarantine_dir=output_dir,
    )
    training_apps = resolve_training_apps(full_data, config)
    config.training_apps = training_apps
    write_json(output_dir / "requested_run_config.json", asdict(config))
    data = subset_training_data(full_data, training_apps)
    unknown_apps = sorted(
        set(full_data.jobs["job_id"].astype(str)) - set(training_apps)
    )
    summary = {
        "applications": len(full_data.jobs),
        "training_application_count": len(training_apps),
        "training_applications": training_apps,
        "unknown_applications": unknown_apps,
        "inhibitors": len(full_data.inhibitors),
        "raw_app_inhibitor_rows": len(full_data.responses)
        + len(full_data.rejected_responses),
        "valid_app_inhibitor_rows": len(full_data.responses),
        "model_fitting_app_inhibitor_rows": len(data.responses),
        "rejected_app_inhibitor_rows": len(full_data.rejected_responses),
        "duplicate_key_rows": len(full_data.duplicate_keys),
        "censored_floor_rows": int(full_data.responses["is_censored"].sum()),
        "anchors_per_application": full_data.responses.groupby("job_id").size().to_dict(),
        "uncensored_anchors_per_application": full_data.responses.groupby("job_id")[
            "is_censored"
        ].apply(lambda values: int((~values).sum())).to_dict(),
    }
    write_json(output_dir / "data_validation_summary.json", summary)
    blocks = build_inhibitor_blocks(data, config)
    blocks.to_csv(output_dir / "inhibitor_blocks.csv", index=False)

    print("Running crossed leave-one-application/block validation")
    (
        model_configs,
        aggregations,
        final_epochs,
        folds,
        distance_reference,
        distance_scaler,
    ) = crossed_validation(data, blocks, config, device, output_dir)
    model_selection = {}
    for kind, model_config in model_configs.items():
        kind_folds = folds[folds["model_kind"] == kind]
        selected_id = int(
            kind_folds.groupby("config_id")["best_validation"].mean().idxmin()
        )
        selected_folds = kind_folds[kind_folds["config_id"] == selected_id]
        model_selection[kind] = {
            "selected_config_id": selected_id,
            "validation_log_mae": float(selected_folds["best_validation"].mean()),
            "final_epochs": final_epochs[kind],
            "selection_source": "crossed App-Inhibitor validation only",
        }
    selection = {
        "model_configs": {
            kind: asdict(model_config) for kind, model_config in model_configs.items()
        },
        "aggregations": aggregations,
        "model_selection": model_selection,
        "final_epochs_from_model_specific_cv_median": final_epochs,
        "crossed_fold_count": len(folds),
        "crossed_fold_count_by_model": folds.groupby("model_kind").size().to_dict(),
        "ood_distance_calibration": {
            "source": "crossed_validation_query_to_uncensored_support_distances",
            "coordinate_system": "single base-feature scaler fit to inhibitor profiles only",
            "scaler": distance_scaler.state_dict(),
            "count": len(distance_reference),
            "median": float(np.median(distance_reference)),
            "maximum": float(np.max(distance_reference)),
        },
    }
    write_json(output_dir / "selection.json", selection)

    print(f"Training final models for {final_epochs} predetermined epochs")
    scalers, models = train_final_models(
        data, model_configs, config, final_epochs, device, output_dir
    )

    common_report = {
        "status": "complete",
        "evaluation_protocol": {
            "mode": config.evaluation_method,
            "methods": evaluation_methods,
            "training_applications": training_apps,
            "unknown_applications": unknown_apps,
            "unknown_application_anchors_available_at_inference": True,
            "app_app_outcomes_used_for_fitting": False,
            "one_known_definition": "exactly_one_known_endpoint",
        },
        "model_seed_count": len(config.final_seeds),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "clustering": {
            "algorithm": config.clustering_algorithm,
            "blocks": config.inhibitor_blocks,
            "seed": config.clustering_seed,
        },
        "target_semantics": {
            "observed_slowdown": "max(1, latent_slowdown)",
            "observed_log_floor": 0.0,
            "latent_speedup_magnitude_identifiable": False,
        },
        **selection,
    }
    if args.skip_holdout:
        report = {
            **common_report,
            "holdout_opened_after_final_training": False,
            "holdout_evaluation": "skipped",
            "pair_rows": None,
            "evaluated_pair_rows": None,
            "directional_outcomes": None,
        }
        write_json(output_dir / "run_report.json", report)
        print(f"App-Inhibitor-only results written to {output_dir}; pair CSV was not opened")
        return

    # This is the sole point where pair.csv is opened. No fitting follows it.
    print(
        "Opening sealed pair.csv for final "
        f"{', '.join(evaluation_methods)} App-App evaluation"
    )
    pairs = load_pair_holdout(args.pair_csv, full_data.jobs)
    pair_counts = {}
    directional_counts = {}
    for evaluation_method in evaluation_methods:
        evaluation_dir = (
            output_dir
            if len(evaluation_methods) == 1
            else output_dir / "evaluations" / evaluation_method
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_pairs, pair_manifest = select_evaluation_pairs(
            pairs, training_apps, evaluation_method
        )
        pair_manifest.to_csv(
            evaluation_dir / "evaluation_pair_manifest.csv", index=False
        )
        predictions, seed_predictions = evaluate_holdout(
            full_data,
            evaluation_pairs,
            scalers,
            models,
            aggregations,
            distance_reference,
            distance_scaler,
            device,
            evaluation_method=evaluation_method,
            known_apps=training_apps,
            inference_anchor_counts=config.inference_anchor_counts,
            inference_anchor_seed=config.inference_anchor_seed,
        )
        write_evaluation_reports(
            predictions, seed_predictions, config, evaluation_dir
        )
        pair_counts[evaluation_method] = len(evaluation_pairs)
        directional_counts[evaluation_method] = len(evaluation_pairs) * 2
    report = {
        **common_report,
        "holdout_opened_after_final_training": True,
        "holdout_evaluation": "completed",
        "pair_rows": len(pairs),
        "evaluated_pair_rows": sum(pair_counts.values()),
        "evaluated_pair_rows_by_method": pair_counts,
        "directional_outcomes": sum(directional_counts.values()),
        "directional_outcomes_by_method": directional_counts,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "run_report.json", report)
    print(f"Results written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, default=DEFAULT_DATA_DIR / "jobs.csv")
    parser.add_argument(
        "--inhibitors-csv", type=Path, default=DEFAULT_DATA_DIR / "inhibitors.csv"
    )
    parser.add_argument(
        "--job-inh-csv", type=Path, default=DEFAULT_DATA_DIR / "job_inh.csv"
    )
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_DATA_DIR / "pair.csv")
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="complete training and App-Inhibitor validation without opening a pair CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, help="JSON RunConfig overrides")
    parser.add_argument(
        "--eval-method",
        "--eval_method",
        dest="eval_method",
        choices=sorted(EVALUATION_MODES),
        help="App-App endpoint-familiarity evaluation tier",
    )
    parser.add_argument(
        "--training-apps",
        "--training_apps",
        dest="training_apps",
        help="comma-separated fitting applications for one_known or zero_shot",
    )
    parser.add_argument("--cv-seeds", type=int, nargs="+")
    parser.add_argument("--final-seeds", type=int, nargs="+")
    parser.add_argument("--inhibitor-blocks", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
