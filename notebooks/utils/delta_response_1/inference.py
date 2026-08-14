"""Anchor aggregation and prediction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data import observe_log_slowdown
from training import response_values


@dataclass(frozen=True)
class OODConfig:
    quantile: float = 0.95
    fallback: str = "median"


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
