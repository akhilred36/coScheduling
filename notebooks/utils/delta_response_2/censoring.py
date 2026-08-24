"""Convex one-dimensional calibration with exact and floor-censored anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CensoredInterceptResult:
    correction: float
    objective: float
    exact_count: int
    floor_count: int
    active_floor_violations: int
    shrinkage: float
    prior: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(
        absolute <= delta,
        0.5 * values**2,
        delta * (absolute - 0.5 * delta),
    )


def _objective(
    correction: float,
    potentials: np.ndarray,
    observed: np.ndarray,
    shrinkage: float,
    prior: float,
    delta: float,
) -> float:
    exact = observed > 0
    exact_residual = potentials[exact] + correction - observed[exact]
    floor_violation = np.maximum(0.0, potentials[~exact] + correction)
    return float(
        _huber(exact_residual, delta).sum()
        + _huber(floor_violation, delta).sum()
        + 0.5 * shrinkage * (correction - prior) ** 2
    )


def _gradient(
    correction: float,
    potentials: np.ndarray,
    observed: np.ndarray,
    shrinkage: float,
    prior: float,
    delta: float,
) -> float:
    exact = observed > 0
    exact_residual = potentials[exact] + correction - observed[exact]
    floor_latent = potentials[~exact] + correction
    exact_gradient = np.clip(exact_residual, -delta, delta).sum()
    floor_gradient = np.clip(np.maximum(floor_latent, 0.0), 0.0, delta).sum()
    return float(exact_gradient + floor_gradient + shrinkage * (correction - prior))


def fit_censored_intercept(
    observed_log: np.ndarray,
    potentials: np.ndarray | None = None,
    *,
    shrinkage: float = 0.0,
    prior: float = 0.0,
    huber_delta: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 256,
) -> CensoredInterceptResult:
    """Fit c where exact rows target p+c=y and floors constrain p+c<=0."""
    observed = np.asarray(observed_log, dtype=float)
    values = np.zeros_like(observed) if potentials is None else np.asarray(potentials, dtype=float)
    if observed.ndim != 1 or values.shape != observed.shape or len(observed) == 0:
        raise ValueError("observed_log and potentials must be non-empty matching vectors")
    if not np.isfinite(observed).all() or not np.isfinite(values).all():
        raise ValueError("Censored calibration inputs must be finite")
    if (observed < 0).any():
        raise ValueError("Observed log slowdowns must be non-negative")
    if shrinkage < 0 or huber_delta <= 0:
        raise ValueError("shrinkage must be non-negative and huber_delta positive")
    exact_count = int((observed > 0).sum())
    if exact_count == 0 and shrinkage == 0:
        raise ValueError("An all-floor censored intercept requires positive shrinkage")

    scale = max(1.0, float(np.max(np.abs(observed - values))), abs(float(prior)))
    low = float(prior - 2.0 * scale)
    high = float(prior + 2.0 * scale)
    for _ in range(128):
        if _gradient(low, values, observed, shrinkage, prior, huber_delta) <= 0:
            break
        low -= scale
        scale *= 2.0
    scale = max(1.0, float(np.max(np.abs(observed - values))), abs(float(prior)))
    for _ in range(128):
        if _gradient(high, values, observed, shrinkage, prior, huber_delta) >= 0:
            break
        high += scale
        scale *= 2.0
    else:
        raise RuntimeError("Could not bracket censored-intercept optimum")

    correction = 0.5 * (low + high)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        correction = 0.5 * (low + high)
        gradient = _gradient(
            correction, values, observed, shrinkage, prior, huber_delta
        )
        if abs(gradient) <= tolerance or high - low <= tolerance:
            converged = True
            break
        if gradient > 0:
            high = correction
        else:
            low = correction
    if not np.isfinite(correction):
        raise FloatingPointError("Censored calibration produced a non-finite result")
    floor_latent = values[observed == 0] + correction
    return CensoredInterceptResult(
        correction=float(correction),
        objective=_objective(
            correction, values, observed, shrinkage, prior, huber_delta
        ),
        exact_count=exact_count,
        floor_count=int((observed == 0).sum()),
        active_floor_violations=int((floor_latent > tolerance).sum()),
        shrinkage=float(shrinkage),
        prior=float(prior),
        iterations=iterations,
        converged=converged,
    )


def gauge_center(potentials: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(potentials, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Gauge reference must be a non-empty finite vector")
    center = float(values.mean())
    return values - center, center
