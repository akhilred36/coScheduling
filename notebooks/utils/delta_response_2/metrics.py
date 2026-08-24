"""Magnitude, calibration, rank, seed, and paired-fold metrics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import log_to_slowdown, observe_log_slowdown


METRICS = [
    "log_mae",
    "log_mse",
    "log_rmse",
    "raw_mae",
    "raw_mape",
    "raw_mse",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
    "signed_log_bias",
    "calibration_intercept",
    "calibration_slope",
    "spearman",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def compute_metrics(
    frame: pd.DataFrame,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, float | bool | int]:
    true_log = frame["true_log_slowdown"].to_numpy(dtype=float)
    predicted_log = frame["predicted_log_slowdown"].to_numpy(dtype=float)
    if len(frame) == 0:
        raise ValueError("Metrics require at least one prediction")
    if (
        not np.isfinite(true_log).all()
        or not np.isfinite(predicted_log).all()
        or (true_log < 0).any()
        or (predicted_log < 0).any()
    ):
        raise ValueError("Log slowdowns must be finite and non-negative")
    metric_weights = (
        np.ones(len(frame), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if (
        metric_weights.shape != (len(frame),)
        or not np.isfinite(metric_weights).all()
        or (metric_weights < 0).any()
        or metric_weights.sum() <= 0
    ):
        raise ValueError("Metric weights must be finite, non-negative, and nonzero")

    true = log_to_slowdown(true_log)
    predicted = log_to_slowdown(predicted_log)
    log_error = predicted_log - true_log
    raw_error = predicted - true
    if len(frame) > 1 and np.ptp(predicted_log) > 1e-12 and np.ptp(true_log) > 1e-12:
        correlation = float(spearmanr(predicted_log, true_log).correlation)
        correlation_defined = bool(np.isfinite(correlation))
    else:
        correlation = 0.0
        correlation_defined = False
    if not correlation_defined:
        correlation = 0.0

    design = np.column_stack([np.ones(len(frame)), predicted_log])
    weighted_design = design * np.sqrt(metric_weights)[:, None]
    weighted_target = true_log * np.sqrt(metric_weights)
    if np.ptp(predicted_log) > 1e-12:
        calibration = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        calibration_defined = bool(np.isfinite(calibration).all())
    else:
        calibration = np.asarray([_weighted_mean(true_log, metric_weights), 0.0])
        calibration_defined = False
    return {
        "n": len(frame),
        "weight_sum": float(metric_weights.sum()),
        "effective_n": float(metric_weights.sum() ** 2 / np.sum(metric_weights**2)),
        "log_mae": _weighted_mean(np.abs(log_error), metric_weights),
        "log_mse": _weighted_mean(log_error**2, metric_weights),
        "log_rmse": float(np.sqrt(_weighted_mean(log_error**2, metric_weights))),
        "raw_mae": _weighted_mean(np.abs(raw_error), metric_weights),
        "raw_mape": 100.0 * _weighted_mean(np.abs(raw_error) / true, metric_weights),
        "raw_mse": _weighted_mean(raw_error**2, metric_weights),
        "raw_rmse": float(np.sqrt(_weighted_mean(raw_error**2, metric_weights))),
        "median_absolute_log_error": float(np.median(np.abs(log_error))),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(log_error)))),
        "signed_log_bias": _weighted_mean(log_error, metric_weights),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
        "calibration_defined": calibration_defined,
        "spearman": correlation,
        "spearman_defined": correlation_defined,
        "rank_weighting": "unweighted",
        "median_weighting": "unweighted",
    }


def validation_views(
    predictions: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, np.ndarray | None]]:
    positive = predictions[predictions["true_log_slowdown"] > 0]
    floor = predictions[predictions["true_log_slowdown"] == 0]
    nearest = predictions[predictions["app_nearest_subset"].astype(bool)]
    views: list[tuple[str, pd.DataFrame, np.ndarray | None]] = [
        ("unweighted", predictions, None),
        (
            "app_profile_weighted",
            predictions,
            predictions["profile_weight"].to_numpy(dtype=float),
        ),
        ("app_profile_nearest", nearest, None),
        ("floor", floor, None),
        ("positive", positive, None),
    ]
    if len(positive):
        positive_values = positive["true_log_slowdown"].to_numpy(dtype=float)
        first, second = np.quantile(positive_values, [1.0 / 3.0, 2.0 / 3.0])
        views.extend(
            [
                (
                    "positive_low",
                    positive[positive["true_log_slowdown"] <= first],
                    None,
                ),
                (
                    "positive_mid",
                    positive[
                        (positive["true_log_slowdown"] > first)
                        & (positive["true_log_slowdown"] <= second)
                    ],
                    None,
                ),
                (
                    "positive_high",
                    positive[positive["true_log_slowdown"] > second],
                    None,
                ),
            ]
        )
    return [(name, frame, weights) for name, frame, weights in views if len(frame)]


def metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return query-weighted, fold, per-victim, and seed-stability tables."""
    query_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    victim_rows: list[dict[str, object]] = []
    grouping = ["candidate_id", "method", "fold_scheme", "seed"]
    for keys, group in predictions.groupby(grouping, sort=False):
        base = dict(zip(grouping, keys))
        for view, selected, weights in validation_views(group):
            query_rows.append(
                {**base, "view": view, "aggregation": "query_weighted", **compute_metrics(selected, weights=weights)}
            )
        for (victim_id, heldout_block), fold in group.groupby(
            ["victim_id", "heldout_block"], sort=False
        ):
            for view, selected, weights in validation_views(fold):
                fold_rows.append(
                    {
                        **base,
                        "victim_id": victim_id,
                        "heldout_block": heldout_block,
                        "view": view,
                        **compute_metrics(selected, weights=weights),
                    }
                )
        for victim_id, victim in group.groupby("victim_id", sort=False):
            for view, selected, weights in validation_views(victim):
                victim_rows.append(
                    {
                        **base,
                        "victim_id": victim_id,
                        "view": view,
                        **compute_metrics(selected, weights=weights),
                    }
                )

    query_metrics = pd.DataFrame(query_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    per_victim = pd.DataFrame(victim_rows)
    stability_rows: list[dict[str, object]] = []
    for keys, group in query_metrics.groupby(
        ["candidate_id", "method", "fold_scheme", "view"], sort=False
    ):
        base = dict(zip(["candidate_id", "method", "fold_scheme", "view"], keys))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            if metric == "spearman":
                worst = float(np.min(values))
                best = float(np.max(values))
            elif metric == "signed_log_bias":
                worst = float(values[np.argmax(np.abs(values))])
                best = float(values[np.argmin(np.abs(values))])
            elif metric == "calibration_slope":
                worst = float(values[np.argmax(np.abs(values - 1.0))])
                best = float(values[np.argmin(np.abs(values - 1.0))])
            elif metric == "calibration_intercept":
                worst = float(values[np.argmax(np.abs(values))])
                best = float(values[np.argmin(np.abs(values))])
            else:
                worst = float(np.max(values))
                best = float(np.min(values))
            stability_rows.append(
                {
                    **base,
                    "metric": metric,
                    "seed_mean": float(np.mean(values)),
                    "seed_std": float(np.std(values)),
                    "seed_worst": worst,
                    "seed_best": best,
                    "seed_count": len(values),
                }
            )
    return query_metrics, fold_metrics, per_victim, pd.DataFrame(stability_rows)


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "candidate_id",
        "method",
        "fold_scheme",
        "victim_id",
        "heldout_block",
        "response_id",
        "query_inhib_id",
        "true_log_slowdown",
        "is_censored",
        "profile_weight",
        "app_nearest_subset",
    ]
    ensemble = predictions.groupby(keys, as_index=False, sort=False).agg(
        latent_predicted_log_slowdown=("latent_predicted_log_slowdown", "mean"),
        model_seed_log_std=("latent_predicted_log_slowdown", lambda values: float(np.std(values))),
    )
    ensemble["predicted_log_slowdown"] = observe_log_slowdown(
        ensemble["latent_predicted_log_slowdown"].to_numpy(dtype=float)
    )
    ensemble["predicted_slowdown"] = log_to_slowdown(
        ensemble["predicted_log_slowdown"].to_numpy(dtype=float)
    )
    return ensemble


def ensemble_metric_table(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["candidate_id", "method", "fold_scheme"]
    for keys, group in ensemble.groupby(grouping, sort=False):
        base = dict(zip(grouping, keys))
        for view, selected, weights in validation_views(group):
            rows.append({**base, "view": view, **compute_metrics(selected, weights=weights)})
    return pd.DataFrame(rows)


def paired_fold_bootstrap(
    predictions: pd.DataFrame,
    candidate_ids: list[str],
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Paired descriptive intervals clustered conservatively by victim."""
    subset = predictions[predictions["candidate_id"].isin(candidate_ids)].copy()
    ensemble = ensemble_predictions(subset)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for scheme, scheme_frame in ensemble.groupby("fold_scheme", sort=False):
        clusters = list(scheme_frame["victim_id"].drop_duplicates())
        frames = {
            candidate: frame
            for candidate, frame in scheme_frame.groupby("candidate_id", sort=False)
        }
        expected = set(clusters)
        for candidate, frame in frames.items():
            actual = set(frame["victim_id"].drop_duplicates())
            if actual != expected:
                raise ValueError(f"Candidate {candidate} does not cover every fold cluster")
        draws = rng.integers(0, len(clusters), size=(samples, len(clusters)))
        point = {
            candidate: compute_metrics(frame)["log_mae"]
            for candidate, frame in frames.items()
        }
        sampled: dict[str, list[float]] = {candidate: [] for candidate in frames}
        grouped = {
            candidate: {
                cluster: group
                for cluster, group in frame.groupby("victim_id", sort=False)
            }
            for candidate, frame in frames.items()
        }
        for draw in draws:
            selected_clusters = [clusters[index] for index in draw]
            for candidate in frames:
                replicate = pd.concat(
                    [grouped[candidate][cluster] for cluster in selected_clusters],
                    ignore_index=True,
                )
                sampled[candidate].append(float(compute_metrics(replicate)["log_mae"]))
        for left, right in combinations(frames, 2):
            differences = np.asarray(sampled[left]) - np.asarray(sampled[right])
            rows.append(
                {
                    "fold_scheme": scheme,
                    "method": left,
                    "comparison_method": right,
                    "metric": "log_mae",
                    "estimate_difference": float(point[left] - point[right]),
                    "ci_lower": float(np.quantile(differences, 0.025)),
                    "ci_upper": float(np.quantile(differences, 0.975)),
                    "bootstrap_samples": samples,
                    "cluster_count": len(clusters),
                    "cluster_unit": "victim_id",
                    "interval_status": "descriptive_development_only",
                }
            )
    return pd.DataFrame(rows)
