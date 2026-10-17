"""End-to-end bidirectional filtered-inhibitor experiment.

The App-App pair table is deliberately not opened until crossed validation,
model/epoch/aggregation/calibration selection, final fitting, and checkpoint
writing have all completed. Evaluation then reloads every model from its
checkpoint and performs inference and reporting only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from contexts import ContextBuilder, ContextCache, ContextConfig, fit_token_scaler, state_digest, token_dim
from data import (
    AUX_TASKS,
    EVALUATION_MODES,
    AuxiliaryScaler,
    BidirectionalTrainingData,
    ProfileScaler,
    aux_mask_column,
    aux_target_column,
    expand_directional_pairs,
    load_bidirectional_training_data,
    load_pair_holdout,
    log_slowdown_to_slowdown,
    observe_log_slowdown,
    resolve_evaluation_methods,
    select_evaluation_pairs,
    subset_training_data,
    validate_aux_tasks,
)
from inference import (
    CalibrationConfig,
    JointPredictor,
    OODConfig,
    PredictionDiagnostics,
    aggregate_residuals,
    predict_pair_direct,
    predict_with_crossfit_context_calibration,
    predict_with_floor_aware_calibration,
)
from metrics import (
    auxiliary_metrics,
    cluster_bootstrap,
    common_pair_rows,
    metrics_by_censoring,
    metrics_by_method,
    paired_cluster_bootstrap_differences,
    pair_level_predictions,
    pair_metrics,
    per_victim_metrics,
    self_pair_averaged,
)
from model import AbsoluteResponseModel, GenericPotential, JointResponseModel, LowRankPotential
from select_inhibitors import run as select_inhibitor_panel
from training import (
    profile_maps,
    response_values,
    set_seed,
    train_absolute,
    train_joint_response,
    train_potential,
)


HERE = Path(__file__).resolve().parent
DEFAULT_TRAINING_DIR = HERE / "data" / "training"
DEFAULT_EVALUATION_DIR = HERE / "data" / "evaluation"
DEFAULT_TRAINING_APPS = ["amg", "beatnik", "fiesta", "laghos", "lammps", "minife", "minivite"]
CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_AGGREGATIONS = ["uniform", "median", "kernel", "ood_kernel"]


# ---------------------------------------------------------------------------
# Configuration and the model/ablation registry.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    feature_set: str = "base"
    hidden_dim: int = 8
    embedding_dim: int = 4
    rank: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    aux_hidden_dim: int = 8


@dataclass(frozen=True)
class ModelVariantSpec:
    name: str
    family: str
    legacy_kind: str | None = None
    lambda_abs: float = 1.0
    lambda_delta: float = 1.0
    lambda_aux: float = 0.0
    aux_tasks: tuple[str, ...] = ()
    use_context: bool = False
    token_directions: str = "forward_plus_reverse"
    description: str = ""

    @property
    def is_joint(self) -> bool:
        return self.family == "joint"

    @property
    def is_legacy_potential(self) -> bool:
        return self.family == "legacy_potential"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aux_tasks"] = list(self.aux_tasks)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelVariantSpec":
        values = dict(values)
        values["aux_tasks"] = tuple(values.get("aux_tasks", ()))
        return cls(**values)


VARIANT_REGISTRY: dict[str, ModelVariantSpec] = {
    "legacy_delta": ModelVariantSpec(
        "legacy_delta", "legacy_potential", "low_rank",
        description="original low-rank delta-only potential with anchor aggregation"),
    "legacy_generic": ModelVariantSpec(
        "legacy_generic", "legacy_potential", "generic",
        description="original generic delta-only potential with anchor aggregation"),
    "legacy_absolute": ModelVariantSpec(
        "legacy_absolute", "legacy_absolute", "absolute",
        description="original architecture-matched absolute-response model"),
    "joint_profile": ModelVariantSpec(
        "joint_profile", "joint", lambda_aux=0.0,
        description="joint absolute+delta response with a victim level term"),
    "bidirectional_profile": ModelVariantSpec(
        "bidirectional_profile", "joint", lambda_aux=0.1, aux_tasks=AUX_TASKS,
        description="joint response plus forward and reverse bandwidth supervision"),
    "bidirectional_context": ModelVariantSpec(
        "bidirectional_context", "joint", lambda_aux=0.1, aux_tasks=AUX_TASKS, use_context=True,
        description="bidirectional model with measured application response fingerprints"),
    "bidirectional_profile_wall": ModelVariantSpec(
        "bidirectional_profile_wall", "joint", lambda_aux=0.1, aux_tasks=("bw_wall", "msg_wall"),
        description="ablation: wall-clock-only auxiliary tasks (no MPI-time throughput)"),
    "bidirectional_context_forward": ModelVariantSpec(
        "bidirectional_context_forward", "joint", lambda_aux=0.1, aux_tasks=AUX_TASKS,
        use_context=True, token_directions="forward_only",
        description="ablation: fingerprints without reverse (inhibitor-endpoint) evidence"),
}
DEFAULT_VARIANTS = [
    "legacy_delta", "legacy_generic", "legacy_absolute",
    "joint_profile", "bidirectional_profile", "bidirectional_context",
]
ABLATION_VARIANTS = ["bidirectional_profile_wall", "bidirectional_context_forward"]


def inference_methods(spec: ModelVariantSpec) -> list[str]:
    if spec.is_legacy_potential:
        return [f"{spec.name}_{aggregation}" for aggregation in LEGACY_AGGREGATIONS]
    if spec.family == "legacy_absolute":
        return [spec.name]
    return [f"{spec.name}_direct", f"{spec.name}_calibrated"]


@dataclass
class RunConfig:
    max_profiles: int = 10
    evaluation_method: str = "random_split"
    training_apps: list[str] | None = None
    inhibitor_blocks: int = 4
    clustering_algorithm: str = "kmeans"
    clustering_seed: int = 1701
    cv_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    final_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model_candidates: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    model_candidates_by_variant: dict[str, list[ModelConfig]] | None = None
    max_epochs: int = 80
    patience: int = 10
    batches_per_epoch: int = 4
    batch_size: int = 128
    temperatures: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0])
    ood_quantiles: list[float] = field(default_factory=lambda: [0.9, 0.95, 0.99])
    ood_fallbacks: list[str] = field(default_factory=lambda: ["uniform", "median"])
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 923
    device: str = "cpu"
    selection_mode: str = "crossed_cv"
    variants: list[str] = field(default_factory=lambda: list(DEFAULT_VARIANTS))
    lambda_abs: float = 1.0
    lambda_delta: float = 1.0
    lambda_aux: float = 0.1
    context: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    epoch_selection_inference: str = "calibrated"
    frozen_panel_csv: str | None = None
    false_benign_threshold: float = 1.05
    fixed_final_epochs_by_variant: dict[str, int] | None = None
    fixed_aggregations_by_variant: dict[str, dict[str, Any]] | None = None
    fixed_calibration_lambda_by_variant: dict[str, float] | None = None
    fixed_model_configs_by_variant: dict[str, dict[str, Any]] | None = None
    fixed_selection_source: dict[str, Any] | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    raw = json.loads(Path(path).read_text())
    candidates = [ModelConfig(**item) for item in raw.pop("model_candidates", [{}])]
    by_variant = raw.pop("model_candidates_by_variant", None)
    if by_variant is not None:
        by_variant = {name: [ModelConfig(**item) for item in values] for name, values in by_variant.items()}
    return RunConfig(model_candidates=candidates, model_candidates_by_variant=by_variant, **raw)


def resolve_variant_specs(config: RunConfig) -> list[ModelVariantSpec]:
    unknown = sorted(set(config.variants) - set(VARIANT_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; registry: {sorted(VARIANT_REGISTRY)}")
    if not config.variants:
        raise ValueError("At least one variant is required")
    specs = []
    for name in dict.fromkeys(config.variants):
        spec = VARIANT_REGISTRY[name]
        if spec.is_joint:
            spec = replace(
                spec,
                lambda_abs=float(config.lambda_abs),
                lambda_delta=float(config.lambda_delta),
                lambda_aux=float(config.lambda_aux) if spec.aux_tasks else 0.0,
                aux_tasks=validate_aux_tasks(spec.aux_tasks),
            )
        specs.append(spec)
    return specs


def model_candidates_for_variant(config: RunConfig, variant: str) -> list[ModelConfig]:
    if config.model_candidates_by_variant is None:
        return config.model_candidates
    unknown = set(config.model_candidates_by_variant) - set(VARIANT_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown model candidate variants: {sorted(unknown)}")
    candidates = config.model_candidates_by_variant.get(variant, config.model_candidates)
    if not candidates:
        raise ValueError(f"No model candidates configured for {variant}")
    return candidates


def context_config_for(spec: ModelVariantSpec, config: RunConfig) -> ContextConfig:
    settings = {key: value for key, value in config.context.items() if key not in {"enabled", "token_directions"}}
    return ContextConfig.from_dict(
        {**settings, "enabled": spec.use_context, "token_directions": spec.token_directions}
    )


def calibration_config(config: RunConfig) -> CalibrationConfig:
    return CalibrationConfig.from_dict(config.calibration)


def calibration_lambdas(config: RunConfig) -> list[float]:
    calibration = calibration_config(config)
    values = list(calibration.lambda_candidates) if calibration.select_lambda else []
    return sorted(set([float(calibration.calibration_lambda), *values]))


def validate_selection_config(config: RunConfig, specs: list[ModelVariantSpec]) -> None:
    if config.epoch_selection_inference not in {"calibrated", "direct"}:
        raise ValueError("epoch_selection_inference must be 'calibrated' or 'direct'")
    calibration_config(config)
    for spec in specs:
        if spec.use_context:
            context_config_for(spec, config)
    if config.selection_mode == "crossed_cv":
        return
    if config.selection_mode != "fixed_global":
        raise ValueError("selection_mode must be 'crossed_cv' or 'fixed_global'")
    names = {spec.name for spec in specs}
    epochs = config.fixed_final_epochs_by_variant or {}
    if not names.issubset(epochs):
        raise ValueError(f"fixed_global requires fixed epochs for {sorted(names - set(epochs))}")
    if any(not isinstance(epochs[name], int) or epochs[name] <= 0 for name in names):
        raise ValueError("Fixed final epoch counts must be positive integers")
    for spec in specs:
        if config.fixed_model_configs_by_variant is None and len(model_candidates_for_variant(config, spec.name)) != 1:
            raise ValueError(f"fixed_global requires exactly one model candidate for {spec.name}")
        if spec.is_legacy_potential:
            aggregation = (config.fixed_aggregations_by_variant or {}).get(spec.name)
            if aggregation is None:
                raise ValueError(f"fixed_global requires a fixed aggregation for {spec.name}")
            if "temperature" not in aggregation.get("kernel", {}) or not {
                "temperature", "quantile", "fallback"
            }.issubset(aggregation.get("ood_kernel", {})):
                raise ValueError(f"Fixed aggregation for {spec.name} is incomplete")
        if spec.is_joint:
            value = (config.fixed_calibration_lambda_by_variant or {}).get(spec.name)
            if value is None or not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"fixed_global requires a positive calibration lambda for {spec.name}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_training_apps(data: BidirectionalTrainingData, config: RunConfig) -> list[str]:
    available = list(data.jobs["job_id"].astype(str))
    if config.evaluation_method == "random_split":
        if config.training_apps is not None and set(config.training_apps) != set(available):
            raise ValueError("random_split requires every application to be available for training")
        return available
    requested = config.training_apps
    if requested is None:
        if set(DEFAULT_TRAINING_APPS).issubset(available):
            requested = DEFAULT_TRAINING_APPS
        else:
            raise ValueError("--training-apps is required for one_known and zero_shot on this dataset")
    resolved = list(dict.fromkeys(str(value) for value in requested))
    unknown = sorted(set(resolved) - set(available))
    if unknown:
        raise ValueError(f"Unknown training applications: {unknown}")
    if len(resolved) < 2:
        raise ValueError("At least two training applications are required")
    if set(resolved) == set(available):
        raise ValueError(f"{config.evaluation_method} requires at least one unknown application")
    return resolved


def build_inhibitor_blocks(data: BidirectionalTrainingData, config: RunConfig) -> pd.DataFrame:
    if config.clustering_algorithm != "kmeans":
        raise ValueError("Only the recorded 'kmeans' clustering algorithm is supported")
    if config.inhibitor_blocks > len(data.inhibitors):
        raise ValueError("inhibitor_blocks exceeds the number of panel inhibitors")
    scaler = ProfileScaler("base").fit(data.inhibitors)
    labels = KMeans(
        n_clusters=config.inhibitor_blocks, random_state=config.clustering_seed, n_init=20
    ).fit_predict(scaler.transform(data.inhibitors))
    return pd.DataFrame({"inhib_id": data.inhibitors["inhib_id"].astype(str), "block": labels})


# ---------------------------------------------------------------------------
# Variant construction, fitting, and runtime prediction.
# ---------------------------------------------------------------------------


@dataclass
class VariantRuntime:
    spec: ModelVariantSpec
    model_config: ModelConfig
    model: torch.nn.Module
    profile_scaler: ProfileScaler
    job_profiles: dict[str, np.ndarray]
    inhibitor_profiles: dict[str, np.ndarray]
    aux_scaler: AuxiliaryScaler | None
    token_scaler: AuxiliaryScaler | None
    context_config: ContextConfig
    device: torch.device
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    calibration_lambda: float | None = None
    aggregation: dict[str, Any] | None = None

    @property
    def token_width(self) -> int:
        return token_dim(self.context_config, self.profile_scaler.output_dim) if self.spec.use_context else 0

    @property
    def predictor(self) -> JointPredictor:
        return JointPredictor(self.model, self.device, self.token_width)

    def normalization_state(self) -> str:
        return state_digest(
            self.profile_scaler.state_dict(),
            self.token_scaler.state_dict() if self.token_scaler else None,
            self.context_config.to_dict(),
        )

    def builder(
        self, rows: pd.DataFrame, scope: str, *, fold: str = "", cache: ContextCache | None = None
    ) -> ContextBuilder | None:
        if not self.spec.use_context:
            return None
        return ContextBuilder(
            rows, self.inhibitor_profiles, self.token_scaler, self.context_config, scope,
            fold=fold, model_identity=f"{self.spec.name}|{self.normalization_state()}", cache=cache,
        )


def make_model(spec: ModelVariantSpec, model_config: ModelConfig, context: ContextConfig) -> torch.nn.Module:
    input_dim = 4 if model_config.feature_set == "base" else 8
    arguments = {
        "input_dim": input_dim,
        "hidden_dim": model_config.hidden_dim,
        "embedding_dim": model_config.embedding_dim,
        "rank": model_config.rank,
    }
    if spec.legacy_kind == "low_rank":
        return LowRankPotential(**arguments)
    if spec.legacy_kind == "generic":
        return GenericPotential(**arguments)
    if spec.legacy_kind == "absolute":
        return AbsoluteResponseModel(**arguments)
    return JointResponseModel(
        **arguments,
        aux_tasks=spec.aux_tasks,
        aux_hidden_dim=model_config.aux_hidden_dim,
        token_dim=token_dim(context, input_dim) if spec.use_context else None,
        context_hidden_dim=context.token_hidden_dim,
        context_dim=context.context_dim,
    )


def prepare_runtime(
    spec: ModelVariantSpec,
    model_config: ModelConfig,
    config: RunConfig,
    *,
    train_rows: pd.DataFrame,
    scaler_frame: pd.DataFrame,
    jobs: pd.DataFrame,
    inhibitors: pd.DataFrame,
    device: torch.device,
    scaler: ProfileScaler | None = None,
) -> VariantRuntime:
    """Fit every fold-local transformation on training inputs only, then build the model."""
    scaler = scaler or ProfileScaler(model_config.feature_set).fit(scaler_frame)
    job_profiles, inhibitor_profiles = profile_maps(jobs, inhibitors, scaler)
    context = context_config_for(spec, config)
    aux_scaler = AuxiliaryScaler(spec.aux_tasks).fit(train_rows) if spec.aux_tasks else None
    token_scaler = fit_token_scaler(train_rows, context) if spec.use_context else None
    model = make_model(spec, model_config, context)
    return VariantRuntime(
        spec, model_config, model, scaler, job_profiles, inhibitor_profiles,
        aux_scaler, token_scaler, context, device, calibration=calibration_config(config),
    )


def train_runtime(
    runtime: VariantRuntime,
    train_rows: pd.DataFrame,
    config: RunConfig,
    *,
    seed: int,
    epochs: int,
    validation_fn=None,
    patience: int | None = None,
) -> dict[str, Any]:
    spec, model_config = runtime.spec, runtime.model_config
    common = dict(
        seed=seed, epochs=epochs, batches_per_epoch=config.batches_per_epoch,
        batch_size=config.batch_size, learning_rate=model_config.learning_rate,
        weight_decay=model_config.weight_decay, device=runtime.device,
        validation_fn=validation_fn, patience=patience,
    )
    if spec.is_legacy_potential:
        return train_potential(runtime.model, train_rows, runtime.job_profiles, runtime.inhibitor_profiles, **common)
    if spec.family == "legacy_absolute":
        return train_absolute(runtime.model, train_rows, runtime.job_profiles, runtime.inhibitor_profiles, **common)
    token_matrices = None
    if spec.use_context:
        builder = runtime.builder(train_rows, "fold_training_rows")
        token_matrices = {job_id: builder.token_matrix(job_id) for job_id in builder.rows}
    return train_joint_response(
        runtime.model, train_rows, runtime.job_profiles, runtime.inhibitor_profiles,
        aux_tasks=spec.aux_tasks,
        aux_scales=runtime.aux_scaler.vector() if runtime.aux_scaler else None,
        lambda_abs=spec.lambda_abs, lambda_delta=spec.lambda_delta, lambda_aux=spec.lambda_aux,
        token_matrices=token_matrices,
        max_support=runtime.context_config.max_support,
        empty_probability=runtime.context_config.empty_probability,
        **common,
    )


def legacy_aggregation_specs(config: RunConfig) -> list[tuple[str, float, float, str]]:
    specifications: list[tuple[str, float, float, str]] = [
        ("uniform", 1.0, 0.95, "median"),
        ("median", 1.0, 0.95, "median"),
    ]
    specifications.extend(("kernel", value, 0.95, "median") for value in config.temperatures)
    specifications.extend(
        ("ood_kernel", temperature, quantile, fallback)
        for temperature in config.temperatures
        for quantile in config.ood_quantiles
        for fallback in config.ood_fallbacks
    )
    return specifications


def selected_aggregation_specs(aggregation: dict[str, Any]) -> list[tuple[str, float, float, str]]:
    return [
        ("uniform", 1.0, 0.95, "median"),
        ("median", 1.0, 0.95, "median"),
        ("kernel", float(aggregation["kernel"]["temperature"]), 0.95, "median"),
        ("ood_kernel", float(aggregation["ood_kernel"]["temperature"]),
         float(aggregation["ood_kernel"]["quantile"]), str(aggregation["ood_kernel"]["fallback"])),
    ]


def legacy_potential_latents(
    runtime: VariantRuntime,
    victim_id: str,
    anchors: pd.DataFrame,
    target_profiles: np.ndarray,
    target_distance_profiles: np.ndarray,
    distance_inhibitor_profiles: dict[str, np.ndarray],
    reference: np.ndarray,
    specifications: list[tuple[str, float, float, str]],
) -> list[list[dict[str, Any]]] | None:
    """Legacy residual aggregation for every target and aggregation setting."""
    usable = anchors[anchors["log_slowdown"] > 0].reset_index(drop=True)
    if usable.empty:
        return None
    anchor_profiles = np.stack([runtime.inhibitor_profiles[str(value)] for value in usable["inhib_id"]])
    distance_anchors = np.stack([distance_inhibitor_profiles[str(value)] for value in usable["inhib_id"]])
    potentials = response_values(
        runtime.model, runtime.job_profiles[victim_id], np.vstack([anchor_profiles, target_profiles]), runtime.device
    )
    residuals = usable["log_slowdown"].to_numpy() - potentials[: len(usable)]
    results = []
    for index, target_potential in enumerate(potentials[len(usable):]):
        distances = np.linalg.norm(distance_anchors - target_distance_profiles[index][None, :], axis=1)
        rows = []
        for method, temperature, quantile, fallback in specifications:
            correction, _, alpha = aggregate_residuals(
                residuals, distances, method=method, temperature=temperature,
                reference_distances=reference, ood_config=OODConfig(quantile, fallback),
            )
            rows.append(
                {
                    "aggregation": method, "temperature": temperature, "ood_quantile": quantile,
                    "ood_fallback": fallback, "latent": float(target_potential + correction),
                    "nearest_anchor_distance": float(distances.min()), "ood_alpha": alpha,
                    "uncensored_anchor_count": len(usable),
                }
            )
        results.append(rows)
    return results


def absolute_latents(runtime: VariantRuntime, victim_id: str, target_profiles: np.ndarray) -> np.ndarray:
    return response_values(runtime.model, runtime.job_profiles[victim_id], target_profiles, runtime.device)


def joint_predictions(
    runtime: VariantRuntime,
    *,
    victim_id: str,
    anchors: pd.DataFrame,
    builder: ContextBuilder | None,
    target_profiles: np.ndarray,
    target_tokens: list[np.ndarray | None] | None,
    lambdas: list[float],
    include_aux: bool = True,
) -> dict[str, Any]:
    """Direct and floor-aware calibrated latent predictions for one victim."""
    predictor = runtime.predictor
    victim_profile = runtime.job_profiles[victim_id]
    victim_tokens = None
    context_count = 0
    context_ids: list[str] = []
    if builder is not None:
        tokens = builder.application_context(victim_id)
        context_count, context_ids = tokens.count, tokens.inhibitor_ids
        victim_tokens = tokens.features if tokens.count else None
    direct = predictor.response(victim_profile, victim_tokens, target_profiles, target_tokens)
    calibration = runtime.calibration
    calibrated = {}
    for value in lambdas:
        if builder is not None:
            latent, diagnostics = predict_with_crossfit_context_calibration(
                predictor, victim_id, victim_profile, builder, target_profiles, runtime.inhibitor_profiles,
                calibration_lambda=value, folds=runtime.context_config.crossfit_folds,
                seed=runtime.context_config.partition_seed, config=calibration, target_tokens=target_tokens,
            )
        else:
            latent, result = predict_with_floor_aware_calibration(
                predictor, victim_profile, anchors, target_profiles, runtime.inhibitor_profiles,
                calibration_lambda=value, config=calibration, target_tokens=target_tokens,
            )
            diagnostics = PredictionDiagnostics(
                "victim_panel_anchor_calibration",
                calibration_inhibitor_count=result.inhibitor_count,
                exact_calibration_count=result.exact_count,
                floor_calibration_count=result.floor_count,
                calibration_offset=result.offset,
                calibration_converged=result.converged,
                calibration_view_count=1 if result.fallback is None else 0,
                calibration_fallback=result.fallback,
                views=[{"view": 0, "context_inhibitor_ids": "", "context_inhibitor_count": 0,
                        "calibration_inhibitor_ids": "|".join(sorted(anchors["inhib_id"].astype(str).unique())),
                        **result.as_dict()}],
            )
        calibrated[value] = (latent, diagnostics)
    result: dict[str, Any] = {
        "direct": direct, "calibrated": calibrated,
        "context_count": context_count, "context_ids": context_ids,
    }
    if include_aux and runtime.spec.aux_tasks:
        result["aux_forward"] = predictor.bandwidth(victim_profile, victim_tokens, target_profiles, target_tokens)
        result["aux_reverse"] = predictor.bandwidth(
            victim_profile, victim_tokens, target_profiles, target_tokens, reverse=True
        )
    return result


# ---------------------------------------------------------------------------
# Crossed application/inhibitor-block validation.
# ---------------------------------------------------------------------------


def _candidate_statistics(frame: pd.DataFrame) -> tuple[float, float, int]:
    """Average within victim/block/seed, then across seeds, then across groups."""
    fold_errors = (
        frame.groupby(["victim_id", "heldout_block", "seed"], as_index=False)["absolute_log_error"].mean()
        .groupby(["victim_id", "heldout_block"])["absolute_log_error"].mean()
    )
    mean = float(fold_errors.mean())
    standard_error = float(fold_errors.std(ddof=1) / np.sqrt(len(fold_errors))) if len(fold_errors) > 1 else 0.0
    return mean, standard_error, len(fold_errors)


def _select_aggregation(predictions: pd.DataFrame) -> dict[str, Any]:
    """Deterministic grouped one-standard-error aggregation selection (legacy)."""
    candidates: list[dict[str, Any]] = []
    for complexity, method in enumerate(["uniform", "median"]):
        mean, standard_error, groups = _candidate_statistics(predictions[predictions["method"] == method])
        candidates.append({"method": method, "complexity": complexity, "parameters": {},
                           "validation_log_mae": mean, "grouped_standard_error": standard_error,
                           "validation_group_count": groups, "parameter_order": 0})
    kernel = predictions[predictions["method"] == "kernel"]
    for order, temperature in enumerate(sorted(kernel["temperature"].unique())):
        mean, standard_error, groups = _candidate_statistics(kernel[kernel["temperature"] == temperature])
        candidates.append({"method": "kernel", "complexity": 2, "parameters": {"temperature": float(temperature)},
                           "validation_log_mae": mean, "grouped_standard_error": standard_error,
                           "validation_group_count": groups, "parameter_order": order})
    ood = predictions[predictions["method"] == "ood_kernel"]
    values = ood[["temperature", "ood_quantile", "ood_fallback"]].drop_duplicates().sort_values(
        ["temperature", "ood_quantile", "ood_fallback"])
    for order, value in enumerate(values.itertuples(index=False)):
        rows = ood[(ood["temperature"] == value.temperature) & (ood["ood_quantile"] == value.ood_quantile)
                   & (ood["ood_fallback"] == value.ood_fallback)]
        mean, standard_error, groups = _candidate_statistics(rows)
        candidates.append({"method": "ood_kernel", "complexity": 3,
                           "parameters": {"temperature": float(value.temperature),
                                          "quantile": float(value.ood_quantile),
                                          "fallback": str(value.ood_fallback)},
                           "validation_log_mae": mean, "grouped_standard_error": standard_error,
                           "validation_group_count": groups, "parameter_order": order})
    selected_by_method = {}
    for method in LEGACY_AGGREGATIONS:
        family = [candidate for candidate in candidates if candidate["method"] == method]
        best = min(family, key=lambda value: value["validation_log_mae"])
        threshold = best["validation_log_mae"] + best["grouped_standard_error"]
        selected_by_method[method] = min(
            [value for value in family if value["validation_log_mae"] <= threshold],
            key=lambda value: value["parameter_order"],
        )
    family_best = list(selected_by_method.values())
    numerical_best = min(family_best, key=lambda value: value["validation_log_mae"])
    threshold = numerical_best["validation_log_mae"] + numerical_best["grouped_standard_error"]
    primary = min([value for value in family_best if value["validation_log_mae"] <= threshold],
                  key=lambda value: (value["complexity"], value["parameter_order"]))
    result: dict[str, Any] = {
        method: {**value["parameters"], "validation_log_mae": value["validation_log_mae"],
                 "grouped_standard_error": value["grouped_standard_error"],
                 "validation_group_count": value["validation_group_count"]}
        for method, value in selected_by_method.items()
    }
    result["primary_method"] = primary["method"]
    accuracy_best = min(candidates, key=lambda value: value["validation_log_mae"])
    result["accuracy_best"] = {"method": accuracy_best["method"], **accuracy_best["parameters"],
                               "validation_log_mae": accuracy_best["validation_log_mae"]}
    result["selection_rule"] = {"name": "grouped_one_standard_error",
                                "grouping": "victim_id and heldout_block after seed averaging",
                                "complexity_order": LEGACY_AGGREGATIONS, "threshold": threshold}
    return result


def _prediction_record(base: dict[str, Any], latent: float, true_log: float, **extra: Any) -> dict[str, Any]:
    predicted = float(observe_log_slowdown(latent))
    return {
        **base, **extra,
        "latent_predicted_log_slowdown": float(latent),
        "predicted_log_slowdown": predicted,
        "absolute_log_error": abs(predicted - true_log),
        "signed_log_error": predicted - true_log,
    }


def _diagnostic_columns(diagnostics: PredictionDiagnostics) -> dict[str, Any]:
    return {key: value for key, value in diagnostics.row().items()}


def record_fold_predictions(
    records: dict[str, list[dict[str, Any]]],
    runtime: VariantRuntime,
    *,
    fold: dict[str, Any],
    config_id: int,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    builder: ContextBuilder | None,
    distance_inhibitor_profiles: dict[str, np.ndarray],
    reference: np.ndarray,
    config: RunConfig,
) -> None:
    spec = runtime.spec
    victim_id = fold["victim_id"]
    query_ids = queries["inhib_id"].astype(str).tolist()
    query_profiles = np.stack([runtime.inhibitor_profiles[value] for value in query_ids])
    base_rows = [
        {
            "variant": spec.name, "config_id": config_id, "victim_id": victim_id,
            "heldout_block": fold["heldout_block"], "seed": fold["seed"],
            "query_inhib_id": row["inhib_id"], "query_observation_id": row["observation_id"],
            "true_log_slowdown": float(row["log_slowdown"]),
            "query_is_censored": bool(row["is_censored"]),
        }
        for _, row in queries.iterrows()
    ]
    if spec.is_legacy_potential:
        distance_profiles = np.stack([distance_inhibitor_profiles[value] for value in query_ids])
        latents = legacy_potential_latents(
            runtime, victim_id, support, query_profiles, distance_profiles,
            distance_inhibitor_profiles, reference, legacy_aggregation_specs(config),
        )
        if latents is None:
            return
        for base, rows in zip(base_rows, latents):
            for item in rows:
                records["predictions"].append(_prediction_record(
                    {**base, "inference_method": f"{spec.name}_{item['aggregation']}",
                     "support_information_regime": "victim_panel_anchors"},
                    item["latent"], base["true_log_slowdown"],
                    method=item["aggregation"], temperature=item["temperature"],
                    ood_quantile=item["ood_quantile"], ood_fallback=item["ood_fallback"],
                    calibration_inhibitor_count=item["uncensored_anchor_count"],
                ))
        return
    if spec.family == "legacy_absolute":
        for base, latent in zip(base_rows, absolute_latents(runtime, victim_id, query_profiles)):
            records["predictions"].append(_prediction_record(
                {**base, "inference_method": spec.name, "support_information_regime": "isolated_profiles_only"},
                latent, base["true_log_slowdown"],
            ))
        return
    result = joint_predictions(
        runtime, victim_id=victim_id, anchors=support, builder=builder,
        target_profiles=query_profiles, target_tokens=None, lambdas=calibration_lambdas(config),
    )
    direct_regime = "victim_panel_fingerprint" if spec.use_context else "isolated_profiles_only"
    for index, base in enumerate(base_rows):
        records["predictions"].append(_prediction_record(
            {**base, "inference_method": f"{spec.name}_direct", "support_information_regime": direct_regime},
            result["direct"][index], base["true_log_slowdown"],
            context_inhibitor_count=result["context_count"],
        ))
        for value, (latent, diagnostics) in result["calibrated"].items():
            records["predictions"].append(_prediction_record(
                {**base, "inference_method": f"{spec.name}_calibrated"},
                latent[index], base["true_log_slowdown"], calibration_lambda=value,
                **_diagnostic_columns(diagnostics),
            ))
    fold_key = {key: fold[key] for key in ["victim_id", "heldout_block", "seed"]}
    records["context_manifest"].append({
        "variant": spec.name, "config_id": config_id, **fold_key, "scope": "crossed_validation",
        "view": "direct", "context_inhibitor_ids": "|".join(result["context_ids"]),
        "context_inhibitor_count": result["context_count"],
        "query_inhibitor_ids": "|".join(sorted(set(query_ids))),
        "context_query_overlap": len(set(result["context_ids"]) & set(query_ids)),
    })
    for value, (_, diagnostics) in result["calibrated"].items():
        for view in diagnostics.views:
            records["calibration"].append({
                "variant": spec.name, "config_id": config_id, **fold_key,
                "scope": "crossed_validation", "calibration_lambda": value, **view,
            })
            if spec.use_context:
                records["context_manifest"].append({
                    "variant": spec.name, "config_id": config_id, **fold_key, "scope": "crossed_validation",
                    "view": f"crossfit_{view['view']}_lambda_{value}",
                    "context_inhibitor_ids": view["context_inhibitor_ids"],
                    "context_inhibitor_count": view["context_inhibitor_count"],
                    "query_inhibitor_ids": "|".join(sorted(set(query_ids))),
                    "context_query_overlap": len(set(filter(None, view["context_inhibitor_ids"].split("|"))) & set(query_ids)),
                })
    if "aux_forward" in result:
        for index, (_, row) in enumerate(queries.iterrows()):
            for endpoint, values, direction in [("app", result["aux_forward"], "app_under_inhibitor"),
                                                ("inhib", result["aux_reverse"], "inhibitor_under_app")]:
                for task_index, task in enumerate(spec.aux_tasks):
                    records["auxiliary"].append({
                        "variant": spec.name, "config_id": config_id, **fold_key,
                        "query_inhib_id": row["inhib_id"], "endpoint": endpoint, "direction": direction,
                        "task": task, "predicted": float(values[index, task_index]),
                        "true": float(row[aux_target_column(endpoint, task)]) if row[aux_mask_column(endpoint, task)] else np.nan,
                        "valid": bool(row[aux_mask_column(endpoint, task)]),
                    })


def joint_validation_score(
    runtime: VariantRuntime,
    victim_id: str,
    support: pd.DataFrame,
    queries: pd.DataFrame,
    builder: ContextBuilder | None,
    config: RunConfig,
) -> float:
    lambdas = [] if config.epoch_selection_inference == "direct" else [calibration_config(config).calibration_lambda]
    query_profiles = np.stack([runtime.inhibitor_profiles[str(value)] for value in queries["inhib_id"]])
    result = joint_predictions(
        runtime, victim_id=victim_id, anchors=support, builder=builder,
        target_profiles=query_profiles, target_tokens=None, lambdas=lambdas, include_aux=False,
    )
    latent = result["direct"] if not lambdas else result["calibrated"][lambdas[0]][0]
    predicted = observe_log_slowdown(latent)
    return float(np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy())))


def legacy_uniform_score(runtime: VariantRuntime, victim_id: str, support: pd.DataFrame, queries: pd.DataFrame) -> float:
    support = support[support["log_slowdown"] > 0]
    if support.empty:
        return float("inf")
    profiles = np.stack([runtime.inhibitor_profiles[str(value)] for value in [*support["inhib_id"], *queries["inhib_id"]]])
    values = response_values(runtime.model, runtime.job_profiles[victim_id], profiles, runtime.device)
    correction = np.mean(support["log_slowdown"].to_numpy() - values[: len(support)])
    predicted = observe_log_slowdown(values[len(support):] + correction)
    return float(np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy())))


def absolute_score(runtime: VariantRuntime, victim_id: str, queries: pd.DataFrame) -> float:
    profiles = np.stack([runtime.inhibitor_profiles[str(value)] for value in queries["inhib_id"]])
    predicted = observe_log_slowdown(absolute_latents(runtime, victim_id, profiles))
    return float(np.mean(np.abs(predicted - queries["log_slowdown"].to_numpy())))


def _common_distance_reference(
    data: BidirectionalTrainingData, blocks: pd.DataFrame, distance_inhibitor_profiles: dict[str, np.ndarray]
) -> np.ndarray:
    block_by_id = dict(zip(blocks["inhib_id"], blocks["block"]))
    responses = data.responses.assign(block=data.responses["inhib_id"].map(block_by_id).astype(int))
    distances = []
    for victim_id in data.jobs["job_id"].astype(str):
        for block in sorted(blocks["block"].unique()):
            mine = responses[responses["job_id"] == victim_id]
            support = mine[(mine["block"] != block) & (mine["log_slowdown"] > 0)].drop_duplicates("inhib_id")
            queries = mine[mine["block"] == block].drop_duplicates("inhib_id")
            if support.empty or queries.empty:
                continue
            left = np.stack([distance_inhibitor_profiles[str(value)] for value in support["inhib_id"]])
            right = np.stack([distance_inhibitor_profiles[str(value)] for value in queries["inhib_id"]])
            distances.extend(np.linalg.norm(right[:, None, :] - left[None, :, :], axis=-1).min(axis=1))
    reference = np.asarray(distances, dtype=float)
    if not len(reference) or not np.isfinite(reference).all():
        raise RuntimeError("Could not construct a finite common OOD distance reference")
    return reference


def distance_calibration(
    data: BidirectionalTrainingData, blocks: pd.DataFrame
) -> tuple[np.ndarray, ProfileScaler]:
    scaler = ProfileScaler("base").fit(data.inhibitors)
    _, inhibitor_profiles = profile_maps(data.jobs, data.inhibitors, scaler)
    return _common_distance_reference(data, blocks, inhibitor_profiles), scaler


@dataclass
class FoldContext:
    victim_id: str
    heldout_block: int
    train_jobs: pd.DataFrame
    retained_inhibitors: pd.DataFrame
    train_rows: pd.DataFrame
    support: pd.DataFrame
    queries: pd.DataFrame

    @property
    def fold_id(self) -> str:
        return f"{self.victim_id}|block{self.heldout_block}"


def crossed_folds(data: BidirectionalTrainingData, blocks: pd.DataFrame) -> list[FoldContext]:
    """Held-out (application, block) folds; every label of a held-out run is excluded."""
    block_by_id = dict(zip(blocks["inhib_id"], blocks["block"]))
    responses = data.responses.assign(block=data.responses["inhib_id"].map(block_by_id).astype(int))
    folds = []
    for victim_id in data.jobs["job_id"].astype(str):
        for block in sorted(blocks["block"].unique()):
            mine = responses["job_id"] == victim_id
            inside = responses["block"] == block
            fold = FoldContext(
                victim_id=victim_id,
                heldout_block=int(block),
                train_jobs=data.jobs[data.jobs["job_id"] != victim_id],
                retained_inhibitors=data.inhibitors[data.inhibitors["inhib_id"].map(block_by_id) != block],
                train_rows=responses[~mine & ~inside].drop(columns="block"),
                support=responses[mine & ~inside].drop(columns="block"),
                queries=responses[mine & inside].drop(columns="block"),
            )
            if not fold.support.empty and not fold.queries.empty:
                folds.append(fold)
    return folds


def crossed_validation(
    data: BidirectionalTrainingData,
    blocks: pd.DataFrame,
    config: RunConfig,
    specs: list[ModelVariantSpec],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    distance_reference, distance_scaler = distance_calibration(data, blocks)
    _, distance_inhibitor_profiles = profile_maps(data.jobs, data.inhibitors, distance_scaler)
    folds = crossed_folds(data, blocks)
    candidates = {spec.name: model_candidates_for_variant(config, spec.name) for spec in specs}
    records: dict[str, list[dict[str, Any]]] = {
        "predictions": [], "auxiliary": [], "context_manifest": [], "calibration": [], "history": []
    }
    fold_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    scaler_cache: dict[tuple[str, str], ProfileScaler] = {}
    context_cache = ContextCache()
    total = sum(len(candidates[spec.name]) for spec in specs) * len(folds) * len(config.cv_seeds)
    completed = 0
    started = time.monotonic()
    next_progress = started + 30.0
    for config_id in range(max(len(values) for values in candidates.values())):
        for fold in folds:
            trainable = fold.train_rows.groupby("job_id")["inhib_id"].nunique().ge(2).any()
            uncensored_support = bool((fold.support["log_slowdown"] > 0).any())
            for seed in config.cv_seeds:
                for spec in specs:
                    if config_id >= len(candidates[spec.name]):
                        continue
                    base = {"variant": spec.name, "config_id": config_id, "victim_id": fold.victim_id,
                            "heldout_block": fold.heldout_block, "seed": seed}
                    reason = None
                    if spec.is_legacy_potential and not uncensored_support:
                        reason = "no_uncensored_support"
                    elif spec.is_legacy_potential and not trainable:
                        reason = "no_training_victim_with_two_inhibitors"
                    if reason is not None:
                        completed += 1
                        skipped.append({**base, "reason": reason, "support_count": len(fold.support),
                                        "query_count": len(fold.queries)})
                        continue
                    model_config = candidates[spec.name][config_id]
                    key = (model_config.feature_set, fold.fold_id)
                    if key not in scaler_cache:
                        scaler_cache[key] = ProfileScaler(model_config.feature_set).fit(
                            pd.concat([fold.train_jobs, fold.retained_inhibitors], ignore_index=True)
                        )
                    set_seed(seed)
                    runtime = prepare_runtime(
                        spec, model_config, config, train_rows=fold.train_rows, scaler_frame=None,
                        jobs=data.jobs, inhibitors=data.inhibitors, device=device, scaler=scaler_cache[key],
                    )
                    builder = runtime.builder(fold.support, "fold_victim_permitted_support",
                                              fold=fold.fold_id, cache=context_cache)
                    if spec.is_joint:
                        validation = lambda _, r=runtime, b=builder, f=fold: joint_validation_score(
                            r, f.victim_id, f.support, f.queries, b, config)
                    elif spec.is_legacy_potential:
                        validation = lambda _, r=runtime, f=fold: legacy_uniform_score(r, f.victim_id, f.support, f.queries)
                    else:
                        validation = lambda _, r=runtime, f=fold: absolute_score(r, f.victim_id, f.queries)
                    result = train_runtime(
                        runtime, fold.train_rows, config, seed=seed, epochs=config.max_epochs,
                        validation_fn=validation, patience=config.patience,
                    )
                    for row in result["history"]:
                        records["history"].append({**base, **row})
                    fold_records.append({
                        **base,
                        "support_count": len(fold.support),
                        "uncensored_support_count": int((fold.support["log_slowdown"] > 0).sum()),
                        "support_inhibitor_count": int(fold.support["inhib_id"].nunique()),
                        "query_count": len(fold.queries),
                        "training_rows": len(fold.train_rows),
                        **{name: result[name] for name in ["best_epoch", "best_validation", "epochs_run"]},
                    })
                    record_fold_predictions(
                        records, runtime, fold={**base}, config_id=config_id,
                        support=fold.support, queries=fold.queries, builder=builder,
                        distance_inhibitor_profiles=distance_inhibitor_profiles,
                        reference=distance_reference, config=config,
                    )
                    completed += 1
                    now = time.monotonic()
                    if completed == 1 or completed == total or now >= next_progress:
                        print(f"Cross-validation progress: {completed}/{total} fits "
                              f"({100.0 * completed / max(total, 1):.1f}%), elapsed {now - started:.0f}s", flush=True)
                        next_progress = now + 30.0

    predictions = pd.DataFrame(records["predictions"])
    folds_frame = pd.DataFrame(fold_records)
    if predictions.empty or folds_frame.empty:
        raise RuntimeError("Crossed validation produced no predictions")
    pd.DataFrame(skipped, columns=["variant", "config_id", "victim_id", "heldout_block", "seed", "reason",
                                   "support_count", "query_count"]).to_csv(output_dir / "skipped_validation_folds.csv", index=False)
    predictions.to_csv(output_dir / "crossed_validation_predictions.csv", index=False)
    folds_frame.to_csv(output_dir / "crossed_validation_folds.csv", index=False)
    pd.DataFrame(records["history"]).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(records["auxiliary"]).to_csv(output_dir / "auxiliary_validation_predictions.csv", index=False)
    pd.DataFrame(records["context_manifest"]).to_csv(output_dir / "context_support_manifest.csv", index=False)
    pd.DataFrame(records["calibration"]).to_csv(output_dir / "calibration_diagnostics.csv", index=False)
    pd.DataFrame({"nearest_anchor_distance": distance_reference}).to_csv(
        output_dir / "validation_distance_reference.csv", index=False)

    selection = select_from_validation(predictions, folds_frame, specs, config, candidates)
    selected = selected_validation_predictions(predictions, selection, specs)
    selected.to_csv(output_dir / "selected_validation_predictions.csv", index=False)
    write_validation_reports(selected, pd.DataFrame(records["auxiliary"]), selection, specs, folds_frame,
                             pd.DataFrame(skipped), output_dir)
    selection["distance_reference"] = distance_reference
    selection["distance_scaler"] = distance_scaler
    return selection


def select_from_validation(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    specs: list[ModelVariantSpec],
    config: RunConfig,
    candidates: dict[str, list[ModelConfig]],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    calibration = calibration_config(config)
    for spec in specs:
        variant_folds = folds[folds["variant"] == spec.name]
        if variant_folds.empty:
            raise RuntimeError(f"No eligible validation folds for {spec.name}")
        scores = variant_folds.groupby("config_id")["best_validation"].mean()
        selected_id = int(scores.idxmin())
        chosen = variant_folds[variant_folds["config_id"] == selected_id]
        entry: dict[str, Any] = {
            "spec": spec.to_dict(),
            "selected_config_id": selected_id,
            "model_config": asdict(candidates[spec.name][selected_id]),
            "final_epochs": max(1, int(np.median(chosen["best_epoch"]))),
            "final_epoch_source": "median selected best epoch across crossed folds",
            "early_stopping_validation_log_mae": float(chosen["best_validation"].mean()),
            "evaluated_fit_count": int(len(chosen)),
            "evaluated_group_count": int(chosen[["victim_id", "heldout_block"]].drop_duplicates().shape[0]),
            "selection_source": "crossed App-Inhibitor validation only",
        }
        rows = predictions[(predictions["variant"] == spec.name) & (predictions["config_id"] == selected_id)]
        if spec.is_legacy_potential:
            entry["aggregation"] = _select_aggregation(rows)
        if spec.is_joint:
            calibrated = rows[rows["inference_method"] == f"{spec.name}_calibrated"]
            lambda_scores = {
                float(value): _candidate_statistics(group)[0]
                for value, group in calibrated.groupby("calibration_lambda")
            }
            if calibration.select_lambda:
                best = min(lambda_scores, key=lambda value: (lambda_scores[value], value))
                source = "searched: minimum grouped App-Inhibitor CV log MAE"
            else:
                best = float(calibration.calibration_lambda)
                source = "fixed"
            entry["calibration_lambda"] = best
            entry["calibration_lambda_source"] = source
            entry["calibration_lambda_validation_log_mae"] = lambda_scores
            entry["direct_validation_log_mae"] = _candidate_statistics(
                rows[rows["inference_method"] == f"{spec.name}_direct"])[0]
        variants[spec.name] = entry
    return {"variants": variants}


def selected_validation_predictions(
    predictions: pd.DataFrame, selection: dict[str, Any], specs: list[ModelVariantSpec]
) -> pd.DataFrame:
    frames = []
    for spec in specs:
        entry = selection["variants"][spec.name]
        rows = predictions[(predictions["variant"] == spec.name)
                           & (predictions["config_id"] == entry["selected_config_id"])]
        if spec.is_legacy_potential:
            aggregation = entry["aggregation"]
            for method, temperature, quantile, fallback in selected_aggregation_specs(aggregation):
                chosen = rows[rows["method"] == method]
                if method in {"kernel", "ood_kernel"}:
                    chosen = chosen[np.isclose(chosen["temperature"], temperature)]
                if method == "ood_kernel":
                    chosen = chosen[np.isclose(chosen["ood_quantile"], quantile) & (chosen["ood_fallback"] == fallback)]
                frames.append(chosen)
        elif spec.is_joint:
            frames.append(rows[rows["inference_method"] == f"{spec.name}_direct"])
            calibrated = rows[rows["inference_method"] == f"{spec.name}_calibrated"]
            frames.append(calibrated[np.isclose(calibrated["calibration_lambda"], entry["calibration_lambda"])])
        else:
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _group_errors(frame: pd.DataFrame) -> pd.Series:
    return (frame.groupby(["victim_id", "heldout_block", "seed"])["absolute_log_error"].mean()
            .groupby(["victim_id", "heldout_block"]).mean())


def ablation_contrasts(methods: set[str], legacy_primary: str | None) -> list[tuple[str, str, str]]:
    contrasts = [
        ("legacy_delta_vs_legacy_absolute", legacy_primary or "legacy_delta_uniform", "legacy_absolute"),
        ("joint_absolute_delta_learning", "joint_profile_direct", "legacy_absolute"),
        ("floor_aware_calibration", "joint_profile_calibrated", "joint_profile_direct"),
        ("reverse_direction_supervision_direct", "bidirectional_profile_direct", "joint_profile_direct"),
        ("reverse_direction_supervision_calibrated", "bidirectional_profile_calibrated", "joint_profile_calibrated"),
        ("measured_response_fingerprints_direct", "bidirectional_context_direct", "bidirectional_profile_direct"),
        ("measured_response_fingerprints_calibrated", "bidirectional_context_calibrated", "bidirectional_profile_calibrated"),
        ("context_calibration", "bidirectional_context_calibrated", "bidirectional_context_direct"),
        ("wall_clock_only_auxiliary_direct", "bidirectional_profile_wall_direct", "bidirectional_profile_direct"),
        ("wall_clock_only_auxiliary_calibrated", "bidirectional_profile_wall_calibrated", "bidirectional_profile_calibrated"),
        ("forward_only_context_tokens_direct", "bidirectional_context_forward_direct", "bidirectional_context_direct"),
        ("forward_only_context_tokens_calibrated", "bidirectional_context_forward_calibrated", "bidirectional_context_calibrated"),
    ]
    return [item for item in contrasts if item[1] in methods and item[2] in methods]


def write_validation_reports(
    selected: pd.DataFrame,
    auxiliary: pd.DataFrame,
    selection: dict[str, Any],
    specs: list[ModelVariantSpec],
    folds: pd.DataFrame,
    skipped: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = selected.copy()
    frame["true_slowdown"] = np.exp(frame["true_log_slowdown"])
    frame["predicted_slowdown"] = np.exp(frame["predicted_log_slowdown"])
    frame["method"] = frame["inference_method"]
    metrics_by_censoring(frame, ["inference_method"]).to_csv(output_dir / "metrics_by_censoring.csv", index=False)
    group_errors = {method: _group_errors(group) for method, group in frame.groupby("inference_method")}
    common = set.intersection(*(set(values.index) for values in group_errors.values()))
    rows = []
    for method, errors in group_errors.items():
        matched = errors[errors.index.isin(common)]
        rows.append({
            "inference_method": method, "comparison_method": None,
            "own_group_count": len(errors), "own_grouped_log_mae": float(errors.mean()),
            "common_group_count": len(matched), "common_grouped_log_mae": float(matched.mean()) if len(matched) else np.nan,
            "difference": np.nan, "difference_standard_error": np.nan,
        })
    methods = sorted(group_errors)
    for index, method in enumerate(methods):
        for comparison in methods[index + 1:]:
            shared = group_errors[method].index.intersection(group_errors[comparison].index)
            difference = group_errors[method].loc[shared] - group_errors[comparison].loc[shared]
            rows.append({
                "inference_method": method, "comparison_method": comparison,
                "own_group_count": len(group_errors[method]), "own_grouped_log_mae": np.nan,
                "common_group_count": len(shared),
                "common_grouped_log_mae": np.nan,
                "difference": float(difference.mean()) if len(shared) else np.nan,
                "difference_standard_error": float(difference.std(ddof=1) / np.sqrt(len(shared))) if len(shared) > 1 else np.nan,
            })
    pd.DataFrame(rows).to_csv(output_dir / "matched_fold_comparisons.csv", index=False)
    legacy = selection["variants"].get("legacy_delta", {}).get("aggregation", {})
    primary = f"legacy_delta_{legacy['primary_method']}" if legacy else None
    ablation = []
    for contrast, method, comparison in ablation_contrasts(set(group_errors), primary):
        shared = group_errors[method].index.intersection(group_errors[comparison].index)
        difference = group_errors[method].loc[shared] - group_errors[comparison].loc[shared]
        ablation.append({
            "scope": "app_inhibitor_crossed_validation", "contrast": contrast, "method": method,
            "comparison_method": comparison, "metric": "grouped_log_mae",
            "method_estimate": float(group_errors[method].loc[shared].mean()) if len(shared) else np.nan,
            "comparison_estimate": float(group_errors[comparison].loc[shared].mean()) if len(shared) else np.nan,
            "estimate_difference": float(difference.mean()) if len(shared) else np.nan,
            "standard_error": float(difference.std(ddof=1) / np.sqrt(len(shared))) if len(shared) > 1 else np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "matched_units": len(shared),
            "unit": "victim/heldout_block groups after seed averaging",
        })
    pd.DataFrame(ablation).to_csv(output_dir / "ablation_summary.csv", index=False)
    if not auxiliary.empty:
        auxiliary_metrics(auxiliary, ["variant", "endpoint", "direction", "task"]).assign(
            grouping="endpoint_direction_task").to_csv(output_dir / "auxiliary_validation_metrics.csv", index=False)
        by_app = auxiliary_metrics(auxiliary, ["variant", "victim_id", "endpoint", "task"]).assign(grouping="application")
        block = auxiliary_metrics(auxiliary, ["variant", "heldout_block", "endpoint", "task"]).assign(grouping="inhibitor_block")
        pd.concat([pd.read_csv(output_dir / "auxiliary_validation_metrics.csv"), by_app, block],
                  ignore_index=True, sort=False).to_csv(output_dir / "auxiliary_validation_metrics.csv", index=False)
    eligibility = []
    for spec in specs:
        evaluated = folds[folds["variant"] == spec.name]
        skipped_rows = skipped[skipped["variant"] == spec.name] if not skipped.empty else skipped
        eligibility.append({
            "variant": spec.name,
            "evaluated_fits": len(evaluated),
            "evaluated_groups": int(evaluated[["victim_id", "heldout_block"]].drop_duplicates().shape[0]),
            "skipped_fits": len(skipped_rows),
            "skipped_groups": int(skipped_rows[["victim_id", "heldout_block"]].drop_duplicates().shape[0]) if len(skipped_rows) else 0,
        })
    pd.DataFrame(eligibility).to_csv(output_dir / "validation_group_eligibility.csv", index=False)


# ---------------------------------------------------------------------------
# Final fitting, checkpoints, and reload.
# ---------------------------------------------------------------------------


def input_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    hashes: dict[str, Any] = {name: sha256_file(path) for name, path in paths.items() if Path(path).is_file()}
    report = Path(paths["jobs"]).parent / "preparation_report.json"
    if report.is_file():
        hashes["preparation_report"] = sha256_file(report)
        hashes["preparation_source_sha256"] = json.loads(report.read_text()).get("source_sha256", {})
    return hashes


def checkpoint_payload(
    runtime: VariantRuntime,
    *,
    seed: int,
    epochs: int,
    training_apps: list[str],
    inhibitor_ids: list[str],
    config: RunConfig,
    data_hashes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "variant": runtime.spec.to_dict(),
        "seed": seed,
        "epochs": epochs,
        "training_apps": training_apps,
        "selected_inhibitor_ids": inhibitor_ids,
        "model_config": asdict(runtime.model_config),
        "scaler": runtime.profile_scaler.state_dict(),
        "auxiliary_scaler": runtime.aux_scaler.state_dict() if runtime.aux_scaler else None,
        "context_token_scaler": runtime.token_scaler.state_dict() if runtime.token_scaler else None,
        "context_config": runtime.context_config.to_dict(),
        "calibration": {
            **calibration_config(config).to_dict(),
            "selected_calibration_lambda": runtime.calibration_lambda,
        },
        "aggregation": runtime.aggregation,
        "data_hashes": data_hashes,
        "model_state_dict": runtime.model.state_dict(),
    }


def load_checkpoint(
    path: Path | str,
    jobs: pd.DataFrame,
    inhibitors: pd.DataFrame,
    device: torch.device,
) -> tuple[VariantRuntime, dict[str, Any]]:
    """Rebuild a runtime from a checkpoint without refitting any scaler."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible checkpoint schema {payload.get('schema_version')!r} in {path}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    spec = ModelVariantSpec.from_dict(payload["variant"])
    model_config = ModelConfig(**payload["model_config"])
    context = ContextConfig.from_dict(payload["context_config"])
    scaler = ProfileScaler.from_state_dict(payload["scaler"])
    job_profiles, inhibitor_profiles = profile_maps(jobs, inhibitors, scaler)
    model = make_model(spec, model_config, context)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    runtime = VariantRuntime(
        spec, model_config, model, scaler, job_profiles, inhibitor_profiles,
        AuxiliaryScaler.from_state_dict(payload["auxiliary_scaler"]) if payload["auxiliary_scaler"] else None,
        AuxiliaryScaler.from_state_dict(payload["context_token_scaler"]) if payload["context_token_scaler"] else None,
        context, device,
        calibration=CalibrationConfig.from_dict(
            {key: value for key, value in payload["calibration"].items() if key != "selected_calibration_lambda"}
        ),
        calibration_lambda=payload["calibration"].get("selected_calibration_lambda"),
        aggregation=payload.get("aggregation"),
    )
    return runtime, payload


def train_final_models(
    data: BidirectionalTrainingData,
    full_data: BidirectionalTrainingData,
    specs: list[ModelVariantSpec],
    selection: dict[str, Any],
    config: RunConfig,
    device: torch.device,
    output_dir: Path,
    data_hashes: dict[str, Any],
) -> dict[int, dict[str, Path]]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_apps = data.jobs["job_id"].astype(str).tolist()
    inhibitor_ids = data.inhibitors["inhib_id"].astype(str).tolist()
    scaler_frame = pd.concat([data.jobs, data.inhibitors], ignore_index=True)
    paths: dict[int, dict[str, Path]] = {}
    history = []
    for seed in config.final_seeds:
        paths[seed] = {}
        for spec in specs:
            entry = selection["variants"][spec.name]
            set_seed(seed)
            runtime = prepare_runtime(
                spec, ModelConfig(**entry["model_config"]), config, train_rows=data.responses,
                scaler_frame=scaler_frame, jobs=full_data.jobs, inhibitors=full_data.inhibitors, device=device,
            )
            runtime.calibration_lambda = entry.get("calibration_lambda")
            runtime.aggregation = entry.get("aggregation")
            result = train_runtime(runtime, data.responses, config, seed=seed, epochs=entry["final_epochs"])
            history.extend({"variant": spec.name, "seed": seed, **row} for row in result["history"])
            path = checkpoint_dir / f"{spec.name}_seed{seed}.pt"
            torch.save(
                checkpoint_payload(runtime, seed=seed, epochs=entry["final_epochs"], training_apps=training_apps,
                                   inhibitor_ids=inhibitor_ids, config=config, data_hashes=data_hashes),
                path,
            )
            paths[seed][spec.name] = path
    pd.DataFrame(history).to_csv(output_dir / "final_training_history.csv", index=False)
    return paths


# ---------------------------------------------------------------------------
# App-App evaluation (inference and reporting only).
# ---------------------------------------------------------------------------


def evaluate_bidirectional_holdout(
    full_data: BidirectionalTrainingData,
    pairs: pd.DataFrame,
    checkpoints: dict[int, dict[str, Path]],
    distance_reference: np.ndarray,
    distance_scaler: ProfileScaler,
    device: torch.device,
    *,
    evaluation_method: str,
    known_apps: list[str] | set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    directional = expand_directional_pairs(pairs)
    known = {str(value) for value in known_apps}
    distance_job_profiles, distance_inhibitor_profiles = profile_maps(full_data.jobs, full_data.inhibitors, distance_scaler)
    anchors_by_job = {str(job): group.reset_index(drop=True) for job, group in full_data.responses.groupby("job_id")}
    rows: list[dict[str, Any]] = []
    auxiliary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for seed, variant_paths in checkpoints.items():
        for variant, path in variant_paths.items():
            runtime, _ = load_checkpoint(path, full_data.jobs, full_data.inhibitors, device)
            spec = runtime.spec
            builder = runtime.builder(full_data.responses, "permitted_inference_panel")
            aggressor_tokens: dict[str, np.ndarray | None] = {}
            if builder is not None:
                for job_id in full_data.jobs["job_id"].astype(str):
                    tokens = builder.application_context(job_id)
                    aggressor_tokens[job_id] = tokens.features if tokens.count else None
            pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
            for _, pair in directional.iterrows():
                victim_id, aggressor_id = str(pair["victim_id"]), str(pair["aggressor_id"])
                base = {
                    "evaluation_method": evaluation_method, "anchor_budget": "all", "seed": seed,
                    "pair_row_id": int(pair["pair_row_id"]), "pair_cluster_id": int(pair["pair_cluster_id"]),
                    "direction": pair["direction"], "victim_id": victim_id, "aggressor_id": aggressor_id,
                    "victim_known": victim_id in known, "aggressor_known": aggressor_id in known,
                    "known_endpoint_count": int(victim_id in known) + int(aggressor_id in known),
                    "victim_support_source": "training_application" if victim_id in known
                    else "unknown_application_inference_support",
                    "true_slowdown": float(pair["true_slowdown"]), "variant": spec.name,
                }
                target = runtime.job_profiles[aggressor_id][None, :]
                anchors = anchors_by_job[victim_id]
                if spec.is_legacy_potential:
                    latents = legacy_potential_latents(
                        runtime, victim_id, anchors, target, distance_job_profiles[aggressor_id][None, :],
                        distance_inhibitor_profiles, distance_reference, selected_aggregation_specs(runtime.aggregation),
                    )
                    if latents is None:
                        continue
                    for item in latents[0]:
                        rows.append({**base, "method": f"{spec.name}_{item['aggregation']}",
                                     "inference_method": f"{spec.name}_{item['aggregation']}",
                                     "support_information_regime": "victim_panel_anchors",
                                     "latent_predicted_log_slowdown": item["latent"],
                                     "calibration_inhibitor_count": item["uncensored_anchor_count"],
                                     "nearest_anchor_distance": item["nearest_anchor_distance"]})
                    continue
                if spec.family == "legacy_absolute":
                    rows.append({**base, "method": spec.name, "inference_method": spec.name,
                                 "support_information_regime": "isolated_profiles_only",
                                 "latent_predicted_log_slowdown": float(absolute_latents(runtime, victim_id, target)[0])})
                    continue
                target_tokens = [aggressor_tokens.get(aggressor_id)] if builder is not None else None
                result = joint_predictions(
                    runtime, victim_id=victim_id, anchors=anchors, builder=builder,
                    target_profiles=target, target_tokens=target_tokens, lambdas=[runtime.calibration_lambda],
                )
                key = tuple(sorted((victim_id, aggressor_id)))
                if key not in pair_cache:
                    left, right = key
                    forward = predict_pair_direct(runtime.predictor, runtime.job_profiles[left], aggressor_tokens.get(left),
                                                  runtime.job_profiles[right], aggressor_tokens.get(right))
                    swapped = predict_pair_direct(runtime.predictor, runtime.job_profiles[right], aggressor_tokens.get(right),
                                                  runtime.job_profiles[left], aggressor_tokens.get(left))
                    pair_cache[key] = {"residual": max(abs(forward["latent_a"] - swapped["latent_b"]),
                                                       abs(forward["latent_b"] - swapped["latent_a"]))}
                aux = {}
                if "aux_forward" in result:
                    aux = {f"predicted_{task}_log_degradation": float(result["aux_forward"][0, index])
                           for index, task in enumerate(spec.aux_tasks)}
                    for index, task in enumerate(spec.aux_tasks):
                        auxiliary_rows.append({
                            "evaluation_method": evaluation_method, "seed": seed, "variant": spec.name,
                            "method": f"{spec.name}_direct", "pair_row_id": base["pair_row_id"],
                            "direction": base["direction"], "victim_id": victim_id, "aggressor_id": aggressor_id,
                            "task": task, "predicted": float(result["aux_forward"][0, index]),
                            "true": pair[f"true_{task}_log_degradation"], "valid": bool(pair[f"true_{task}_valid"]),
                        })
                direct_regime = ("victim_and_aggressor_panel_fingerprints" if builder is not None
                                 else "isolated_profiles_only")
                rows.append({**base, "method": f"{spec.name}_direct", "inference_method": f"{spec.name}_direct",
                             "support_information_regime": direct_regime,
                             "latent_predicted_log_slowdown": float(result["direct"][0]),
                             "context_inhibitor_count": result["context_count"],
                             "swap_equivariance_residual": pair_cache[key]["residual"], **aux})
                latent, diagnostics = result["calibrated"][runtime.calibration_lambda]
                regime = diagnostics.support_information_regime
                if builder is not None:
                    regime += "_with_fixed_aggressor_fingerprint"
                rows.append({**base, "method": f"{spec.name}_calibrated", "inference_method": f"{spec.name}_calibrated",
                             "latent_predicted_log_slowdown": float(latent[0]),
                             "calibration_lambda": runtime.calibration_lambda,
                             **_diagnostic_columns(diagnostics), "support_information_regime": regime})
                for view in diagnostics.views:
                    calibration_rows.append({"scope": f"app_app_{evaluation_method}", "seed": seed, "variant": spec.name,
                                             "victim_id": victim_id, "aggressor_id": aggressor_id,
                                             "pair_row_id": base["pair_row_id"], "direction": base["direction"],
                                             "calibration_lambda": runtime.calibration_lambda, **view})
        # Constant baseline once per seed.
        for _, pair in directional.iterrows():
            victim_id, aggressor_id = str(pair["victim_id"]), str(pair["aggressor_id"])
            rows.append({
                "evaluation_method": evaluation_method, "anchor_budget": "all", "seed": seed,
                "pair_row_id": int(pair["pair_row_id"]), "pair_cluster_id": int(pair["pair_cluster_id"]),
                "direction": pair["direction"], "victim_id": victim_id, "aggressor_id": aggressor_id,
                "victim_known": victim_id in known, "aggressor_known": aggressor_id in known,
                "known_endpoint_count": int(victim_id in known) + int(aggressor_id in known),
                "victim_support_source": "training_application" if victim_id in known
                else "unknown_application_inference_support",
                "true_slowdown": float(pair["true_slowdown"]), "variant": "constant_1", "method": "constant_1",
                "inference_method": "constant_1", "support_information_regime": "none",
                "latent_predicted_log_slowdown": 0.0,
            })
    seed_predictions = pd.DataFrame(rows)
    seed_predictions["predicted_log_slowdown"] = observe_log_slowdown(seed_predictions["latent_predicted_log_slowdown"].to_numpy())
    seed_predictions["predicted_slowdown"] = log_slowdown_to_slowdown(seed_predictions["predicted_log_slowdown"].to_numpy())
    keys = ["evaluation_method", "anchor_budget", "method", "inference_method", "variant",
            "support_information_regime", "pair_row_id", "pair_cluster_id", "direction", "victim_id",
            "aggressor_id", "victim_known", "aggressor_known", "known_endpoint_count",
            "victim_support_source", "true_slowdown"]
    numeric = [column for column in [
        "context_inhibitor_count", "calibration_inhibitor_count", "exact_calibration_count",
        "floor_calibration_count", "calibration_offset", "calibration_view_count", "calibration_lambda",
        "nearest_anchor_distance", "swap_equivariance_residual",
        *[f"predicted_{task}_log_degradation" for task in AUX_TASKS],
    ] if column in seed_predictions]
    seed_predictions["calibration_converged"] = seed_predictions.get("calibration_converged", np.nan)
    aggregations = {column: (column, "mean") for column in numeric}
    ensemble = seed_predictions.groupby(keys, as_index=False, sort=False, dropna=False).agg(
        latent_predicted_log_slowdown=("latent_predicted_log_slowdown", "mean"),
        model_seed_log_std=("latent_predicted_log_slowdown", lambda values: float(np.std(values))),
        model_seed_count=("seed", "nunique"),
        calibration_converged=("calibration_converged",
                               lambda values: bool(pd.Series(values).dropna().astype(bool).all())
                               if pd.Series(values).notna().any() else np.nan),
        **aggregations,
    )
    ensemble["predicted_log_slowdown"] = observe_log_slowdown(ensemble["latent_predicted_log_slowdown"].to_numpy())
    ensemble["predicted_slowdown"] = log_slowdown_to_slowdown(ensemble["predicted_log_slowdown"].to_numpy())
    ensemble["absolute_error"] = (ensemble["predicted_slowdown"] - ensemble["true_slowdown"]).abs()
    ensemble["absolute_log_error"] = (ensemble["predicted_log_slowdown"] - np.log(ensemble["true_slowdown"])).abs()
    ensemble["signed_log_error"] = ensemble["predicted_log_slowdown"] - np.log(ensemble["true_slowdown"])
    auxiliary = pd.DataFrame(auxiliary_rows)
    if not auxiliary.empty:
        auxiliary = auxiliary.groupby(
            ["evaluation_method", "variant", "method", "pair_row_id", "direction", "victim_id", "aggressor_id", "task"],
            as_index=False, sort=False,
        ).agg(predicted=("predicted", "mean"), true=("true", "first"), valid=("valid", "first"))
    return ensemble, seed_predictions, auxiliary, calibration_rows


def write_evaluation_reports(
    predictions: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    auxiliary: pd.DataFrame,
    calibration_rows: list[dict[str, Any]],
    config: RunConfig,
    output_dir: Path,
    validation_ablation: pd.DataFrame | None,
    legacy_primary: str | None,
) -> None:
    evaluation_method = predictions["evaluation_method"].iloc[0]
    predictions.to_csv(output_dir / "directional_predictions.csv", index=False)
    seed_predictions.to_csv(output_dir / "seed_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_diagnostics.csv", index=False)
    common, coverage = common_pair_rows(predictions)
    coverage.to_csv(output_dir / "method_coverage.csv", index=False)
    available_non_self = predictions[predictions["victim_id"] != predictions["aggressor_id"]]
    if not available_non_self.empty:
        # Each method on its own evaluable rows; compare methods only on common rows.
        metrics_by_method(available_non_self).to_csv(output_dir / "metrics_non_self_all_available.csv", index=False)
    if common.empty:
        raise RuntimeError("No pair cluster is evaluable by every method")
    metrics_by_method(common).to_csv(output_dir / "metrics_repeated_self.csv", index=False)
    metrics_by_method(self_pair_averaged(common)).to_csv(output_dir / "metrics_self_averaged.csv", index=False)
    per_victim, macro = per_victim_metrics(common)
    per_victim.to_csv(output_dir / "metrics_per_victim.csv", index=False)
    macro.to_csv(output_dir / "metrics_macro_victim.csv", index=False)
    is_self = common["victim_id"] == common["aggressor_id"]
    non_self, self_pairs = common[~is_self], common[is_self]
    censoring = [metrics_by_censoring(non_self, ["evaluation_method", "method"]).assign(pair_scope="non_self")]
    if not self_pairs.empty:
        censoring.append(metrics_by_censoring(self_pairs, ["evaluation_method", "method"]).assign(pair_scope="self"))
    pd.concat(censoring, ignore_index=True).to_csv(output_dir / "metrics_by_censoring.csv", index=False)
    pair_frames = []
    for scope, frame in [("non_self", non_self), ("self", self_pairs)]:
        if not frame.empty:
            pair_frames.append(pair_level_predictions(frame).assign(pair_scope=scope))
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(output_dir / "pair_predictions.csv", index=False)
    pd.concat([pair_metrics(group).assign(pair_scope=scope) for scope, group in pairs.groupby("pair_scope")],
              ignore_index=True).to_csv(output_dir / "pair_metrics.csv", index=False)
    if not auxiliary.empty:
        auxiliary.to_csv(output_dir / "auxiliary_predictions.csv", index=False)
        auxiliary_metrics(auxiliary, ["evaluation_method", "method", "task"]).to_csv(
            output_dir / "auxiliary_metrics.csv", index=False)
    ablation_rows = [] if validation_ablation is None else [validation_ablation]
    if not non_self.empty:
        metrics_by_method(non_self).to_csv(output_dir / "metrics_non_self.csv", index=False)
        non_self_per_victim, non_self_macro = per_victim_metrics(non_self)
        non_self_per_victim.to_csv(output_dir / "metrics_non_self_per_victim.csv", index=False)
        non_self_macro.to_csv(output_dir / "metrics_non_self_macro_victim.csv", index=False)
        bootstrap = cluster_bootstrap(non_self, samples=config.bootstrap_samples, seed=config.bootstrap_seed)
        bootstrap.insert(0, "anchor_budget", "all")
        bootstrap.insert(0, "evaluation_method", evaluation_method)
        bootstrap.to_csv(output_dir / "cluster_bootstrap_non_self.csv", index=False)
        differences = paired_cluster_bootstrap_differences(
            non_self, samples=config.bootstrap_samples, seed=config.bootstrap_seed)
        differences.insert(0, "anchor_budget", "all")
        differences.insert(0, "evaluation_method", evaluation_method)
        differences.to_csv(output_dir / "paired_method_differences_non_self.csv", index=False)
        lookup = differences[differences["metric"] == "log_mae"].set_index(["method", "comparison_method"])
        point = metrics_by_method(non_self).set_index("method")["log_mae"]
        app_rows = []
        for contrast, method, comparison in ablation_contrasts(set(non_self["method"]), legacy_primary):
            if (method, comparison) in lookup.index:
                row, sign = lookup.loc[(method, comparison)], 1.0
            else:
                row, sign = lookup.loc[(comparison, method)], -1.0
            low, high = sorted([sign * row["ci_lower"], sign * row["ci_upper"]])
            app_rows.append({
                "scope": f"app_app_{evaluation_method}_non_self", "contrast": contrast, "method": method,
                "comparison_method": comparison, "metric": "log_mae",
                "method_estimate": float(point[method]), "comparison_estimate": float(point[comparison]),
                "estimate_difference": sign * float(row["estimate_difference"]), "standard_error": np.nan,
                "ci_lower": low, "ci_upper": high, "matched_units": int(row["cluster_count"]),
                "unit": "unordered pair clusters (paired cluster bootstrap)",
            })
        ablation_rows.append(pd.DataFrame(app_rows))
    bootstrap = cluster_bootstrap(common, samples=config.bootstrap_samples, seed=config.bootstrap_seed)
    bootstrap.insert(0, "anchor_budget", "all")
    bootstrap.insert(0, "evaluation_method", evaluation_method)
    bootstrap.to_csv(output_dir / "cluster_bootstrap_confidence_intervals.csv", index=False)
    differences = paired_cluster_bootstrap_differences(common, samples=config.bootstrap_samples, seed=config.bootstrap_seed)
    differences.insert(0, "anchor_budget", "all")
    differences.insert(0, "evaluation_method", evaluation_method)
    differences.to_csv(output_dir / "paired_method_differences.csv", index=False)
    if ablation_rows:
        pd.concat(ablation_rows, ignore_index=True, sort=False).to_csv(output_dir / "ablation_summary.csv", index=False)
    seed_common = seed_predictions[seed_predictions["pair_row_id"].isin(common["pair_row_id"].unique())]
    seed_rows = []
    for (method, seed), group in seed_common.groupby(["method", "seed"]):
        if set(group["pair_row_id"]) != set(common["pair_row_id"]):
            continue
        seed_rows.append({"evaluation_method": evaluation_method, "anchor_budget": "all", "seed": seed,
                          **metrics_by_method(group.assign(method=method)).iloc[0].to_dict()})
    pd.DataFrame(seed_rows).to_csv(output_dir / "metrics_by_seed.csv", index=False)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def apply_cli_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    for name in ["cv_seeds", "final_seeds", "inhibitor_blocks", "max_epochs", "patience",
                 "batches_per_epoch", "bootstrap_samples", "max_profiles"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    if getattr(args, "eval_method", None) is not None:
        config.evaluation_method = args.eval_method
    if getattr(args, "training_apps", None) is not None:
        config.training_apps = [value.strip() for value in args.training_apps.split(",") if value.strip()]
    if getattr(args, "variants", None):
        config.variants = list(args.variants)
    if getattr(args, "frozen_panel_csv", None) is not None:
        config.frozen_panel_csv = str(Path(args.frozen_panel_csv).resolve())
    if getattr(args, "context_max_support", None) is not None:
        config.context = {**config.context, "max_support": args.context_max_support}
    if getattr(args, "context_crossfit_folds", None) is not None:
        config.context = {**config.context, "crossfit_folds": args.context_crossfit_folds}
    if getattr(args, "calibration_lambda", None):
        values = [float(value) for value in args.calibration_lambda]
        if len(values) == 1:
            config.calibration = {**config.calibration, "calibration_lambda": values[0], "select_lambda": False}
        else:
            config.calibration = {**config.calibration, "lambda_candidates": values, "select_lambda": True,
                                  "calibration_lambda": config.calibration.get("calibration_lambda", 1.0)}
    return config


def fixed_selection(config: RunConfig, specs: list[ModelVariantSpec]) -> dict[str, Any]:
    variants = {}
    for spec in specs:
        model_config = (config.fixed_model_configs_by_variant or {}).get(spec.name)
        if model_config is None:
            model_config = asdict(model_candidates_for_variant(config, spec.name)[0])
        entry: dict[str, Any] = {
            "spec": spec.to_dict(), "selected_config_id": 0, "model_config": model_config,
            "final_epochs": int(config.fixed_final_epochs_by_variant[spec.name]),
            "final_epoch_source": "fixed_global_selection",
            "selection_source": "globally frozen App-Inhibitor selection",
        }
        if spec.is_legacy_potential:
            entry["aggregation"] = config.fixed_aggregations_by_variant[spec.name]
        if spec.is_joint:
            entry["calibration_lambda"] = float(config.fixed_calibration_lambda_by_variant[spec.name])
            entry["calibration_lambda_source"] = "fixed_global_selection"
        variants[spec.name] = entry
    return {"variants": variants}


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = apply_cli_overrides(load_run_config(args.config), args)
    if type(config.max_profiles) is not int or not 1 <= config.max_profiles <= 10:
        raise ValueError("max_profiles must be an integer in [1, 10]")
    if config.evaluation_method not in EVALUATION_MODES:
        raise ValueError(f"evaluation_method must be one of {sorted(EVALUATION_MODES)}")
    evaluation_methods = resolve_evaluation_methods(config.evaluation_method)
    specs = resolve_variant_specs(config)
    validate_selection_config(config, specs)
    device = resolve_device(config.device)
    paths = {"jobs": Path(args.jobs_csv).resolve(), "inhibitors": Path(args.inhibitors_csv).resolve(),
             "job_inh": Path(args.job_inh_csv).resolve()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared training inputs are missing: {missing}. Run "
            "`prepare_bidirectional_data.py training` first."
        )
    data_hashes = input_hashes(paths)

    print("Loading prepared App-Inhibitor training data (App-App outcomes remain sealed)")
    full_data = load_bidirectional_training_data(
        paths["jobs"], paths["inhibitors"], paths["job_inh"], quarantine_dir=output_dir,
        preparation_report=paths["jobs"].parent / "preparation_report.json",
    )
    training_apps = resolve_training_apps(full_data, config)
    panel_dir = output_dir / "selected_inhibitors"
    panel_report = select_inhibitor_panel(argparse.Namespace(
        jobs_csv=paths["jobs"], inhibitors_csv=paths["inhibitors"], job_inh_csv=paths["job_inh"],
        output_dir=panel_dir, max_profiles=config.max_profiles, tail_weight=0.5,
        target_apps=",".join(training_apps), frozen_panel_csv=config.frozen_panel_csv,
    ))
    full_data = load_bidirectional_training_data(
        panel_dir / "jobs.csv", panel_dir / "inhibitors.csv", panel_dir / "job_inh.csv",
        preparation_report=paths["jobs"].parent / "preparation_report.json",
    )
    if full_data.inhibitors["inhib_id"].nunique() > 10:
        raise RuntimeError("Filtered pipeline exceeded its inhibitor panel limit")
    print(f"Training and inference restricted to {len(full_data.inhibitors)} inhibitor profiles "
          f"({panel_report['panel_source']} panel)")
    config.training_apps = training_apps
    write_json(output_dir / "requested_run_config.json", asdict(config))
    data = subset_training_data(full_data, training_apps)
    unknown_apps = sorted(set(full_data.jobs["job_id"].astype(str)) - set(training_apps))
    summary = {
        "applications": len(full_data.jobs),
        "training_application_count": len(training_apps),
        "training_applications": training_apps,
        "unknown_applications": unknown_apps,
        "inhibitors": len(full_data.inhibitors),
        "valid_app_inhibitor_rows": len(full_data.responses),
        "model_fitting_app_inhibitor_rows": len(data.responses),
        "rejected_app_inhibitor_rows": len(full_data.rejected_responses),
        "duplicate_key_rows": len(full_data.duplicate_keys),
        "censored_floor_rows": int(full_data.responses["is_censored"].sum()),
        "auxiliary_valid_rows": {
            aux_target_column(endpoint, task): int(full_data.responses[aux_mask_column(endpoint, task)].sum())
            for endpoint in ["app", "inhib"] for task in AUX_TASKS
        },
        "observation_unit": full_data.preparation_metadata.get("details", {}).get("observation_unit"),
        "rate_estimator": full_data.preparation_metadata.get("details", {}).get("rate_estimator"),
        "input_hashes": data_hashes,
    }
    write_json(output_dir / "data_validation_summary.json", summary)
    blocks = build_inhibitor_blocks(data, config)
    blocks.to_csv(output_dir / "inhibitor_blocks.csv", index=False)
    write_json(output_dir / "variant_registry.json", {
        "requested": [spec.to_dict() for spec in specs],
        "inference_methods": {spec.name: inference_methods(spec) for spec in specs},
        "available": {name: spec.to_dict() for name, spec in VARIANT_REGISTRY.items()},
    })

    validation_ablation = None
    if config.selection_mode == "crossed_cv":
        print("Running crossed leave-one-application/block validation")
        selection = crossed_validation(data, blocks, config, specs, device, output_dir)
        distance_reference = selection.pop("distance_reference")
        distance_scaler = selection.pop("distance_scaler")
        validation_ablation = pd.read_csv(output_dir / "ablation_summary.csv")
    else:
        print("Using globally frozen selection; skipping crossed CV")
        selection = fixed_selection(config, specs)
        distance_reference, distance_scaler = distance_calibration(data, blocks)
        pd.DataFrame({"nearest_anchor_distance": distance_reference}).to_csv(
            output_dir / "validation_distance_reference.csv", index=False)
    selection.update({
        "selection_mode": config.selection_mode,
        "fixed_selection_source": config.fixed_selection_source,
        "selection_metric": "grouped application-slowdown log MAE (victim/block/seed averaging); auxiliary errors are diagnostic only",
        "epoch_selection_inference": config.epoch_selection_inference,
        "calibration": calibration_config(config).to_dict(),
        "fixed_settings": {"lambda_abs": config.lambda_abs, "lambda_delta": config.lambda_delta,
                           "lambda_aux": config.lambda_aux, "context": config.context},
        "ood_distance_calibration": {
            "scaler": distance_scaler.state_dict(), "count": len(distance_reference),
            "median": float(np.median(distance_reference)),
        },
    })
    write_json(output_dir / "selection.json", selection)

    print("Training final models for selected epoch counts: "
          + ", ".join(f"{name}={entry['final_epochs']}" for name, entry in selection["variants"].items()))
    checkpoints = train_final_models(data, full_data, specs, selection, config, device, output_dir, data_hashes)
    legacy_entry = selection["variants"].get("legacy_delta", {}).get("aggregation")
    legacy_primary = f"legacy_delta_{legacy_entry['primary_method']}" if legacy_entry else None

    common_report = {
        "status": "complete",
        "inhibitor_panel": panel_report,
        "validation_support_policy": (
            "legacy potential folds without uncensored support or a trainable distinct-inhibitor episode "
            "are skipped; absolute and joint folds are retained (joint calibration handles all-floor support)"
        ),
        "evaluation_protocol": {
            "mode": config.evaluation_method, "methods": evaluation_methods,
            "training_applications": training_apps, "unknown_applications": unknown_apps,
            "unknown_application_support_access": (
                "forward and reverse App-Inhibitor panel measurements of unknown applications are used "
                "only as inference-time anchors and fingerprints; never for parameter or scaler fitting"
            ),
            "app_app_outcomes_used_for_fitting_or_selection": False,
            "app_app_outcomes_used_for_context": False,
            "one_known_definition": "exactly_one_known_endpoint",
            "panel_design_scope": panel_report["panel_design_scope"],
            "global_hyperparameter_selection_scope": (
                (config.fixed_selection_source or {}).get("selection_scope")
                if config.selection_mode == "fixed_global" else "training applications in this run only"
            ),
        },
        "variants": [spec.name for spec in specs],
        "model_seed_count": len(config.final_seeds),
        "device": str(device),
        "clustering": {"algorithm": config.clustering_algorithm, "blocks": config.inhibitor_blocks,
                       "seed": config.clustering_seed},
        "target_semantics": {
            "observed_slowdown": "max(1, latent_slowdown)", "observed_log_floor": 0.0,
            "auxiliary": "signed log(rate_iso) - log(rate_co); positive = degradation",
            "context_direct_note": "bidirectional_context_direct is an absolute-response predictor with measured support features",
        },
        "data_hashes": data_hashes,
        "selection": selection,
    }
    if args.skip_holdout:
        write_json(output_dir / "run_report.json", {
            **common_report, "holdout_opened_after_final_training": False, "holdout_evaluation": "skipped",
            "elapsed_seconds": time.time() - started,
        })
        print(f"App-Inhibitor-only results written to {output_dir}; App-App data were not opened")
        return

    # Sole point where App-App outcomes are parsed. Checkpoints are already written.
    print(f"Opening App-App evaluation data for final {', '.join(evaluation_methods)} evaluation")
    pairs = load_pair_holdout(args.pair_csv, full_data.jobs)
    counts = {}
    for evaluation_method in evaluation_methods:
        evaluation_dir = output_dir if len(evaluation_methods) == 1 else output_dir / "evaluations" / evaluation_method
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_pairs, manifest = select_evaluation_pairs(pairs, training_apps, evaluation_method)
        manifest.to_csv(evaluation_dir / "evaluation_pair_manifest.csv", index=False)
        predictions, seed_predictions, auxiliary, calibration_rows = evaluate_bidirectional_holdout(
            full_data, evaluation_pairs, checkpoints, distance_reference, distance_scaler, device,
            evaluation_method=evaluation_method, known_apps=training_apps,
        )
        write_evaluation_reports(predictions, seed_predictions, auxiliary, calibration_rows, config,
                                 evaluation_dir, validation_ablation, legacy_primary)
        counts[evaluation_method] = len(evaluation_pairs)
    write_json(output_dir / "run_report.json", {
        **common_report, "holdout_opened_after_final_training": True, "holdout_evaluation": "completed",
        "pair_rows": len(pairs), "evaluated_pair_rows_by_method": counts,
        "app_app_input_sha256": sha256_file(args.pair_csv),
        "elapsed_seconds": time.time() - started,
    })
    print(f"Results written to {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-profiles", type=int, choices=range(1, 11),
                        help="shared training/inference inhibitor panel cap (default: 10)")
    parser.add_argument("--jobs-csv", type=Path, default=DEFAULT_TRAINING_DIR / "jobs.csv")
    parser.add_argument("--inhibitors-csv", type=Path, default=DEFAULT_TRAINING_DIR / "inhibitors.csv")
    parser.add_argument("--job-inh-csv", type=Path, default=DEFAULT_TRAINING_DIR / "job_inh.csv")
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_EVALUATION_DIR / "pair.csv")
    parser.add_argument("--skip-holdout", action="store_true",
                        help="complete training and App-Inhibitor validation without opening App-App data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "filtered_fit")
    parser.add_argument("--config", type=Path, help="JSON RunConfig overrides")
    parser.add_argument("--eval-method", "--eval_method", dest="eval_method", choices=sorted(EVALUATION_MODES))
    parser.add_argument("--training-apps", "--training_apps", dest="training_apps",
                        help="comma-separated fitting applications for one_known or zero_shot")
    parser.add_argument("--cv-seeds", type=int, nargs="+")
    parser.add_argument("--final-seeds", type=int, nargs="+")
    parser.add_argument("--inhibitor-blocks", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANT_REGISTRY),
                        help=f"model variants (default: {' '.join(DEFAULT_VARIANTS)})")
    parser.add_argument("--frozen-panel-csv", type=Path, help="CSV with inhib_id; skip panel recomputation")
    parser.add_argument("--context-max-support", type=int)
    parser.add_argument("--context-crossfit-folds", type=int)
    parser.add_argument("--calibration-lambda", type=float, nargs="+",
                        help="one value fixes lambda; several values are searched by App-Inhibitor CV")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
