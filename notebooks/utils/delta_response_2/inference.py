"""Legacy and all-anchor censored potential calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .censoring import CensoredInterceptResult, fit_censored_intercept, gauge_center
from .training import response_values


@dataclass(frozen=True)
class LegacyOOD:
    temperature: float = 4.0
    quantile: float = 0.90
    fallback: str = "uniform"


@dataclass(frozen=True)
class OODCalibration:
    threshold: float
    width: float
    quantile: float
    reference_count: int


def softmax_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Distances must be a non-empty finite vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -(values**2) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def calibrate_ood(reference: np.ndarray, quantile: float) -> OODCalibration:
    values = np.asarray(reference, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or not 0 < quantile < 1:
        raise ValueError("Invalid OOD reference or quantile")
    threshold = float(np.quantile(values, quantile))
    positive = values[values > 0]
    width = float(np.median(positive)) if len(positive) else max(threshold, 1.0)
    return OODCalibration(threshold, max(width, 1e-8), quantile, len(values))


def ood_alpha(
    distance: float,
    reference: np.ndarray | OODCalibration,
    quantile: float | None = None,
) -> float:
    calibration = (
        reference
        if isinstance(reference, OODCalibration)
        else calibrate_ood(reference, 0.95 if quantile is None else quantile)
    )
    if distance <= calibration.threshold:
        return 1.0
    return float(np.exp(-(distance - calibration.threshold) / calibration.width))


def legacy_exact_corrections(
    support_observed: np.ndarray,
    support_potential: np.ndarray,
    support_distances: np.ndarray,
    reference_distances: np.ndarray | OODCalibration,
    *,
    ood: LegacyOOD = LegacyOOD(),
) -> dict[str, tuple[float, dict[str, float]]]:
    """Reproduce positive-only residual aggregation as explicit legacy methods."""
    observed = np.asarray(support_observed, dtype=float)
    potential = np.asarray(support_potential, dtype=float)
    distances = np.asarray(support_distances, dtype=float)
    exact = observed > 0
    if not exact.any():
        raise ValueError("Legacy residual calibration requires an exact anchor")
    residuals = observed[exact] - potential[exact]
    exact_distances = distances[exact]
    uniform = float(residuals.mean())
    median = float(np.median(residuals))
    kernel_weights = softmax_weights(exact_distances, ood.temperature)
    kernel = float(kernel_weights @ residuals)
    calibration = (
        reference_distances
        if isinstance(reference_distances, OODCalibration)
        else calibrate_ood(reference_distances, ood.quantile)
    )
    alpha = ood_alpha(float(exact_distances.min()), calibration)
    fallback = uniform if ood.fallback == "uniform" else median
    ood_value = alpha * kernel + (1.0 - alpha) * fallback
    common = {
        "anchor_count": float(len(observed)),
        "exact_anchor_count": float(exact.sum()),
        "floor_anchor_count": float((~exact).sum()),
        "nearest_exact_anchor_distance": float(exact_distances.min()),
        "residual_std": float(np.std(residuals)),
    }
    return {
        "uniform": (uniform, {**common, "ood_alpha": 1.0}),
        "median": (median, {**common, "ood_alpha": 1.0}),
        "ood": (float(ood_value), {**common, "ood_alpha": alpha}),
    }


def censored_potential_predictions(
    support_observed: np.ndarray,
    support_potential: np.ndarray,
    query_potential: np.ndarray,
    *,
    shrinkage: float,
    prior: float,
    query_gamma: float = 1.0,
) -> tuple[np.ndarray, CensoredInterceptResult, float]:
    """Gauge-center R, fit all-anchor c, and optionally shrink query variation."""
    if not 0 <= query_gamma <= 1:
        raise ValueError("query_gamma must lie in [0, 1]")
    centered_support, center = gauge_center(np.asarray(support_potential, dtype=float))
    centered_support = query_gamma * centered_support
    centered_query = query_gamma * (np.asarray(query_potential, dtype=float) - center)
    result = fit_censored_intercept(
        np.asarray(support_observed, dtype=float),
        centered_support,
        shrinkage=shrinkage,
        prior=prior,
    )
    latent = result.correction + centered_query
    if not np.isfinite(latent).all():
        raise FloatingPointError("Censored potential prediction is non-finite")
    return latent, result, center


def population_correction_prior(
    model: torch.nn.Module,
    training_rows: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    device: torch.device,
    *,
    query_gamma: float = 1.0,
) -> tuple[float, list[float]]:
    """Estimate a fold-local hierarchical center using parameter-training victims."""
    corrections: list[float] = []
    for victim_id, rows in training_rows.groupby("job_id", sort=False):
        profiles = np.stack(
            [inhibitor_profiles[str(value)] for value in rows["inhib_id"]]
        )
        potentials = response_values(
            model, job_profiles[str(victim_id)], profiles, device
        )
        centered, _ = gauge_center(potentials)
        centered = query_gamma * centered
        observed = rows["log_slowdown"].to_numpy(dtype=float)
        result = fit_censored_intercept(
            observed,
            centered,
            shrinkage=0.0 if (observed > 0).any() else 1.0,
        )
        corrections.append(result.correction)
    if not corrections:
        raise ValueError("A population correction requires training victims")
    return float(np.median(corrections)), corrections


def training_distance_reference(
    training_rows: pd.DataFrame,
    distance_profiles: dict[str, np.ndarray],
    *,
    block_column: str,
) -> np.ndarray:
    """Build a nested OOD reference using only outer parameter-training rows."""
    if block_column not in training_rows:
        raise ValueError(f"Training rows are missing block column {block_column}")
    rows = training_rows
    distances: list[float] = []
    for victim_id in rows["job_id"].drop_duplicates():
        victim_rows = rows[rows["job_id"] == victim_id]
        for block in sorted(rows[block_column].unique()):
            support = victim_rows[
                (victim_rows[block_column] != block)
                & (victim_rows["log_slowdown"] > 0)
            ]
            queries = victim_rows[victim_rows[block_column] == block]
            if support.empty or queries.empty:
                continue
            support_values = np.stack(
                [distance_profiles[str(value)] for value in support["inhib_id"]]
            )
            query_values = np.stack(
                [distance_profiles[str(value)] for value in queries["inhib_id"]]
            )
            distances.extend(
                np.linalg.norm(
                    query_values[:, None, :] - support_values[None, :, :], axis=-1
                ).min(axis=1)
            )
    result = np.asarray(distances, dtype=float)
    if len(result) == 0:
        identifiers = rows["inhib_id"].astype(str).drop_duplicates().tolist()
        profiles = np.stack([distance_profiles[value] for value in identifiers])
        if len(profiles) < 2:
            raise RuntimeError("OOD calibration requires at least two training inhibitors")
        pairwise = np.linalg.norm(
            profiles[:, None, :] - profiles[None, :, :], axis=-1
        )
        np.fill_diagonal(pairwise, np.inf)
        result = pairwise.min(axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Could not build a finite training-only distance reference")
    return result
