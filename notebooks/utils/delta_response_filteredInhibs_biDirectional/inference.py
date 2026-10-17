"""Anchor aggregation, floor-aware calibration, and prediction diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from contexts import ContextBuilder, crossfit_partition, stack_contexts
from data import observe_log_slowdown
from training import response_values


@dataclass(frozen=True)
class OODConfig:
    quantile: float = 0.95
    fallback: str = "median"


def rank_inference_anchors(
    anchors: pd.DataFrame,
    *,
    victim_id: str,
    seed: int,
) -> pd.DataFrame:
    """Return a stable, nested ordering of the victim's uncensored anchors."""
    usable = anchors[anchors["log_slowdown"] > 0].copy()
    if usable.empty:
        raise ValueError("Latent residual calibration requires an uncensored anchor")
    if "replicate_id" not in usable:
        usable["replicate_id"] = 0

    def digest(row: pd.Series) -> str:
        identity = (
            f"{seed}|{victim_id}|{row['inhib_id']}|{int(row['replicate_id'])}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    usable["_anchor_order"] = usable.apply(digest, axis=1)
    return (
        usable.sort_values(
            ["_anchor_order", "inhib_id", "replicate_id"], kind="stable"
        )
        .drop(columns="_anchor_order")
        .reset_index(drop=True)
    )


def softmax_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -(distances**2) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def nearest_neighbor_reference(profiles: np.ndarray) -> np.ndarray:
    if len(profiles) < 2:
        return np.asarray([0.0])
    distances = np.linalg.norm(profiles[:, None, :] - profiles[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def ood_alpha(distance: float, reference: np.ndarray, quantile: float) -> float:
    threshold = float(np.quantile(reference, quantile))
    positive = reference[reference > 0]
    width = float(np.median(positive)) if len(positive) else max(threshold, 1.0)
    if distance <= threshold:
        return 1.0
    return float(np.exp(-(distance - threshold) / max(width, 1e-8)))


def aggregate_residuals(
    residuals: np.ndarray,
    distances: np.ndarray,
    method: str,
    *,
    temperature: float,
    reference_distances: np.ndarray,
    ood_config: OODConfig,
) -> tuple[float, np.ndarray, float]:
    count = len(residuals)
    if count == 0:
        raise ValueError("At least one uncensored anchor is required")
    uniform = np.full(count, 1.0 / count)
    if method == "single":
        index = int(np.argmin(distances))
        weights = np.zeros(count)
        weights[index] = 1.0
        return float(residuals[index]), weights, 1.0
    if method == "uniform":
        return float(residuals.mean()), uniform, 1.0
    if method == "median":
        return float(np.median(residuals)), np.full(count, np.nan), 1.0
    kernel_weights = softmax_weights(distances, temperature)
    local = float(kernel_weights @ residuals)
    if method == "kernel":
        return local, kernel_weights, 1.0
    if method != "ood_kernel":
        raise ValueError(f"Unknown aggregation method: {method}")
    nearest = float(distances.min())
    alpha = ood_alpha(nearest, reference_distances, ood_config.quantile)
    if ood_config.fallback == "median":
        fallback = float(np.median(residuals))
    elif ood_config.fallback == "uniform":
        fallback = float(residuals.mean())
    else:
        raise ValueError("OOD fallback must be 'uniform' or 'median'")
    if ood_config.fallback == "median" and alpha < 1.0:
        effective_weights = np.full(count, np.nan)
    else:
        effective_weights = alpha * kernel_weights + (1.0 - alpha) * uniform
    return alpha * local + (1.0 - alpha) * fallback, effective_weights, alpha


def predict_with_anchors(
    model: torch.nn.Module,
    victim_profile: np.ndarray,
    target_profile: np.ndarray,
    anchors: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    method: str,
    temperature: float,
    reference_distances: np.ndarray,
    ood_config: OODConfig,
    device: torch.device,
    distance_target_profile: np.ndarray | None = None,
    distance_inhibitor_profiles: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    total_anchor_count = len(anchors)
    anchors = anchors[anchors["log_slowdown"] > 0].reset_index(drop=True)
    if anchors.empty:
        raise ValueError("Latent residual calibration requires an uncensored anchor")
    anchor_profiles = np.stack(
        [inhibitor_profiles[str(value)] for value in anchors["inhib_id"]]
    )
    aggressors = np.vstack([anchor_profiles, target_profile[None, :]])
    potentials = response_values(model, victim_profile, aggressors, device)
    anchor_potential = potentials[:-1]
    target_potential = float(potentials[-1])
    residuals = anchors["log_slowdown"].to_numpy(dtype=float) - anchor_potential
    distance_target = (
        target_profile if distance_target_profile is None else distance_target_profile
    )
    distance_profiles = (
        inhibitor_profiles
        if distance_inhibitor_profiles is None
        else distance_inhibitor_profiles
    )
    distance_anchors = np.stack(
        [distance_profiles[str(value)] for value in anchors["inhib_id"]]
    )
    distances = np.linalg.norm(distance_anchors - distance_target[None, :], axis=1)
    correction, weights, alpha = aggregate_residuals(
        residuals,
        distances,
        method,
        temperature=temperature,
        reference_distances=reference_distances,
        ood_config=ood_config,
    )
    nearest = float(distances.min())
    percentile = float(np.mean(reference_distances <= nearest))
    kernel_weights = softmax_weights(distances, temperature)
    kernel_mean = float(kernel_weights @ residuals)
    kernel_dispersion = float(
        np.sqrt(kernel_weights @ ((residuals - kernel_mean) ** 2))
    )
    effective_count = (
        float(1.0 / np.sum(weights**2)) if np.isfinite(weights).all() else np.nan
    )
    weighted_dispersion = (
        float(np.sqrt(weights @ ((residuals - correction) ** 2)))
        if np.isfinite(weights).all()
        else np.nan
    )
    latent_prediction = target_potential + correction
    return {
        "latent_predicted_log_slowdown": latent_prediction,
        "predicted_log_slowdown": float(observe_log_slowdown(latent_prediction)),
        "number_of_anchors": total_anchor_count,
        "uncensored_anchor_count": len(anchors),
        "nearest_anchor_distance": nearest,
        "effective_anchor_count": effective_count,
        "weighted_residual_std": weighted_dispersion,
        "kernel_component_effective_anchor_count": float(
            1.0 / np.sum(kernel_weights**2)
        ),
        "kernel_component_weighted_residual_std": kernel_dispersion,
        "unweighted_residual_std": float(np.std(residuals)),
        "fallback_residual_mad": float(
            np.median(np.abs(residuals - np.median(residuals)))
        ),
        "ood_score": percentile,
        "ood_alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Floor-aware scalar calibration for level-identified (joint) models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    calibration_lambda: float = 1.0
    lambda_candidates: tuple[float, ...] = (0.1, 1.0, 10.0)
    select_lambda: bool = True
    weights: str = "uniform"
    huber_delta: float = 1.0
    tolerance: float = 1e-10
    max_iterations: int = 200

    def __post_init__(self) -> None:
        values = [float(self.calibration_lambda), *map(float, self.lambda_candidates)]
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Calibration lambdas must be finite and positive")
        if self.weights != "uniform":
            raise ValueError("Only uniform configuration weights are implemented")
        object.__setattr__(self, "lambda_candidates", tuple(float(v) for v in self.lambda_candidates))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lambda_candidates"] = list(self.lambda_candidates)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "CalibrationConfig":
        values = dict(values or {})
        if "lambda_candidates" in values:
            values["lambda_candidates"] = tuple(values["lambda_candidates"])
        return cls(**values)


@dataclass
class CalibrationResult:
    offset: float
    converged: bool
    iterations: int
    anchor_count: int
    inhibitor_count: int
    exact_count: int
    floor_count: int
    fallback: str | None
    derivative_at_offset: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configuration_weights(inhibitor_ids: list[str] | np.ndarray) -> np.ndarray:
    """Equal total weight per inhibitor configuration, split among replicates."""
    identifiers = [str(value) for value in inhibitor_ids]
    if not identifiers:
        return np.zeros(0)
    counts = pd.Series(identifiers).value_counts()
    weights = np.asarray([1.0 / counts[value] for value in identifiers], dtype=float)
    return weights / weights.sum()


def _calibration_derivative(
    offset: float,
    latent: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    regularization: float,
    delta: float,
) -> float:
    shifted = latent + offset
    exact = observed > 0
    residual = np.where(exact, shifted - observed, np.maximum(shifted, 0.0))
    psi = np.clip(residual, -delta, delta)
    return float(np.sum(weights * psi) + regularization * offset)


def fit_floor_aware_offset(
    anchor_latent: np.ndarray,
    anchor_logs: np.ndarray,
    *,
    inhibitor_ids: list[str] | np.ndarray | None = None,
    weights: np.ndarray | None = None,
    calibration_lambda: float = 1.0,
    huber_delta: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> CalibrationResult:
    """argmin_c sum_J w_J l_J(c) + lambda/2 c^2 by derivative-root bisection.

    l_J is Huber(r_J + c - y_J) for exact anchors and Huber(max(0, r_J + c))
    for floor anchors. The objective is convex; positive lambda makes the
    root of its monotone derivative unique.
    """
    latent = np.asarray(anchor_latent, dtype=float).reshape(-1)
    observed = np.asarray(anchor_logs, dtype=float).reshape(-1)
    if latent.shape != observed.shape:
        raise ValueError("Anchor predictions and observations must align")
    if not np.isfinite(latent).all() or not np.isfinite(observed).all():
        raise ValueError("Calibration inputs must be finite")
    if (observed < 0).any():
        raise ValueError("Observed anchor log slowdowns must be non-negative")
    if not np.isfinite(calibration_lambda) or calibration_lambda <= 0:
        raise ValueError("calibration_lambda must be finite and positive")
    count = len(latent)
    identifiers = (
        [str(value) for value in inhibitor_ids] if inhibitor_ids is not None
        else [str(index) for index in range(count)]
    )
    if count == 0:
        return CalibrationResult(0.0, True, 0, 0, 0, 0, 0, "empty_support", 0.0)
    if weights is None:
        weights = configuration_weights(identifiers)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != latent.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Calibration weights must be finite, non-negative, and aligned")
    weights = weights / weights.sum()

    def derivative(value: float) -> float:
        return _calibration_derivative(value, latent, observed, weights, calibration_lambda, huber_delta)

    low, high = -1.0, 1.0
    for _ in range(2000):
        if derivative(low) <= 0:
            break
        low *= 2.0
    for _ in range(2000):
        if derivative(high) >= 0:
            break
        high *= 2.0
    if derivative(low) > 0 or derivative(high) < 0:
        raise FloatingPointError("Could not bracket the calibration derivative root")
    iterations = 0
    while high - low > tolerance and iterations < max_iterations:
        middle = 0.5 * (low + high)
        if derivative(middle) > 0:
            high = middle
        else:
            low = middle
        iterations += 1
    offset = 0.5 * (low + high)
    if not np.isfinite(offset):
        raise FloatingPointError("Calibration produced a non-finite offset")
    return CalibrationResult(
        offset=float(offset),
        converged=bool(high - low <= tolerance),
        iterations=iterations,
        anchor_count=count,
        inhibitor_count=len(set(identifiers)),
        exact_count=int((observed > 0).sum()),
        floor_count=int((observed <= 0).sum()),
        fallback=None,
        derivative_at_offset=derivative(offset),
    )


@dataclass
class PredictionDiagnostics:
    support_information_regime: str
    context_inhibitor_count: float = 0.0
    calibration_inhibitor_count: float = 0.0
    exact_calibration_count: float = 0.0
    floor_calibration_count: float = 0.0
    calibration_offset: float = np.nan
    calibration_converged: bool | float = np.nan
    calibration_view_count: int = 0
    calibration_fallback: str | None = None
    views: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("views")
        return result


class JointPredictor:
    """Numpy-facing wrapper around a JointResponseModel."""

    def __init__(self, model: torch.nn.Module, device: torch.device, token_width: int = 0):
        self.model = model
        self.device = device
        self.token_width = token_width

    def _tensor(self, values: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

    def embed(self, profiles: np.ndarray, tokens: list[np.ndarray | None] | None) -> torch.Tensor:
        profiles = np.asarray(profiles, dtype=np.float32)
        context = None
        if self.model.uses_context and tokens is not None:
            padded, mask = stack_contexts(tokens, self.token_width)
            context = (self._tensor(padded), self._tensor(mask, torch.bool))
        return self.model.embed(self._tensor(profiles), context)

    def response(
        self,
        victim_profile: np.ndarray,
        victim_tokens: np.ndarray | None,
        aggressor_profiles: np.ndarray,
        aggressor_tokens: list[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            victim = self.embed(np.asarray(victim_profile)[None, :], [victim_tokens])
            aggressors = self.embed(aggressor_profiles, aggressor_tokens)
            values = self.model.response_from_embeddings(
                victim.expand(len(aggressors), -1), aggressors
            )
        return values.detach().cpu().numpy().astype(float)

    def bandwidth(
        self,
        victim_profile: np.ndarray,
        victim_tokens: np.ndarray | None,
        aggressor_profiles: np.ndarray,
        aggressor_tokens: list[np.ndarray | None] | None = None,
        reverse: bool = False,
    ) -> np.ndarray:
        """D_t(victim, aggressor); ``reverse=True`` returns D_t(aggressor, victim)."""
        self.model.eval()
        with torch.no_grad():
            victim = self.embed(np.asarray(victim_profile)[None, :], [victim_tokens])
            aggressors = self.embed(aggressor_profiles, aggressor_tokens)
            victim = victim.expand(len(aggressors), -1)
            values = (
                self.model.bandwidth_from_embeddings(aggressors, victim)
                if reverse
                else self.model.bandwidth_from_embeddings(victim, aggressors)
            )
        return values.detach().cpu().numpy().astype(float)


def predict_pair_direct(
    predictor: JointPredictor,
    profile_a: np.ndarray,
    tokens_a: np.ndarray | None,
    profile_b: np.ndarray,
    tokens_b: np.ndarray | None,
) -> dict[str, Any]:
    """Evaluate the one directional response function in both orders."""
    result = {
        "latent_a": float(predictor.response(profile_a, tokens_a, profile_b[None, :], [tokens_b])[0]),
        "latent_b": float(predictor.response(profile_b, tokens_b, profile_a[None, :], [tokens_a])[0]),
    }
    if getattr(predictor.model, "bandwidth_head", None) is not None:
        result["aux_a"] = predictor.bandwidth(profile_a, tokens_a, profile_b[None, :], [tokens_b])[0]
        result["aux_b"] = predictor.bandwidth(profile_b, tokens_b, profile_a[None, :], [tokens_a])[0]
    return result


def _anchor_arrays(
    anchors: pd.DataFrame, inhibitor_profiles: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    identifiers = anchors["inhib_id"].astype(str).tolist()
    profiles = np.stack([inhibitor_profiles[value] for value in identifiers]) if identifiers else None
    return profiles, anchors["log_slowdown"].to_numpy(dtype=float), identifiers


def predict_with_floor_aware_calibration(
    predictor: JointPredictor,
    victim_profile: np.ndarray,
    anchors: pd.DataFrame,
    target_profiles: np.ndarray,
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    calibration_lambda: float,
    config: CalibrationConfig = CalibrationConfig(),
    victim_tokens: np.ndarray | None = None,
    target_tokens: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, CalibrationResult]:
    """One calibration view: anchors and targets share the victim's context."""
    target_latent = predictor.response(victim_profile, victim_tokens, target_profiles, target_tokens)
    profiles, observed, identifiers = _anchor_arrays(anchors, inhibitor_profiles)
    if profiles is None:
        result = fit_floor_aware_offset(np.empty(0), np.empty(0), calibration_lambda=calibration_lambda)
        return target_latent, result
    anchor_latent = predictor.response(victim_profile, victim_tokens, profiles, None)
    result = fit_floor_aware_offset(
        anchor_latent, observed, inhibitor_ids=identifiers,
        calibration_lambda=calibration_lambda, huber_delta=config.huber_delta,
        tolerance=config.tolerance, max_iterations=config.max_iterations,
    )
    return target_latent + result.offset, result


def predict_with_crossfit_context_calibration(
    predictor: JointPredictor,
    victim_id: str,
    victim_profile: np.ndarray,
    builder: ContextBuilder,
    target_profiles: np.ndarray,
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    calibration_lambda: float,
    folds: int,
    seed: int,
    config: CalibrationConfig = CalibrationConfig(),
    target_tokens: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, PredictionDiagnostics]:
    """Cross-fitted calibration: a label never both builds and calibrates.

    View k builds the victim fingerprint from S_k = panel minus C_k and fits
    the offset on anchors C_k; anchors and targets share that fixed context.
    View-level latent predictions are averaged before the observation floor.
    """
    support_rows = builder.support_rows(victim_id)
    groups = crossfit_partition(builder.support_ids(victim_id), victim_id, folds, seed)
    diagnostics = PredictionDiagnostics("crossfit_victim_fingerprint_calibration")
    if not groups:
        latent = predictor.response(victim_profile, None, target_profiles, target_tokens)
        diagnostics.calibration_offset = 0.0
        diagnostics.calibration_converged = True
        diagnostics.calibration_fallback = "empty_support"
        return latent, diagnostics
    if len(groups) == 1:
        # A single available inhibitor: empty fingerprint, that inhibitor calibrates.
        views = [([], groups[0])]
    else:
        views = [
            ([value for other in groups if other is not group for value in other], group)
            for group in groups
        ]
    latents = []
    for index, (context_ids, calibration_ids) in enumerate(views):
        if set(context_ids) & set(calibration_ids):
            raise RuntimeError("Cross-fitted context and calibration sets must be disjoint")
        tokens = builder.application_context(
            victim_id, include_inhib_ids=context_ids, exclude_inhib_ids=calibration_ids
        )
        anchors = support_rows[support_rows["inhib_id"].astype(str).isin(calibration_ids)]
        latent, result = predict_with_floor_aware_calibration(
            predictor, victim_profile, anchors, target_profiles, inhibitor_profiles,
            calibration_lambda=calibration_lambda, config=config,
            victim_tokens=tokens.features if tokens.count else None,
            target_tokens=target_tokens,
        )
        latents.append(latent)
        diagnostics.views.append(
            {
                "view": index,
                "context_inhibitor_ids": "|".join(tokens.inhibitor_ids),
                "calibration_inhibitor_ids": "|".join(sorted(calibration_ids)),
                "context_inhibitor_count": tokens.count,
                **result.as_dict(),
            }
        )
    views_frame = pd.DataFrame(diagnostics.views)
    diagnostics.calibration_view_count = len(views)
    diagnostics.context_inhibitor_count = float(views_frame["context_inhibitor_count"].mean())
    diagnostics.calibration_inhibitor_count = float(views_frame["inhibitor_count"].mean())
    diagnostics.exact_calibration_count = float(views_frame["exact_count"].mean())
    diagnostics.floor_calibration_count = float(views_frame["floor_count"].mean())
    diagnostics.calibration_offset = float(views_frame["offset"].mean())
    diagnostics.calibration_converged = bool(views_frame["converged"].all())
    return np.mean(np.stack(latents), axis=0), diagnostics
