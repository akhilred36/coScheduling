"""Evaluation metrics and cluster-aware confidence intervals."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRIC_COLUMNS = [
    "log_mae",
    "log_rmse",
    "raw_mae",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
    "spearman",
]


def metric_strata(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["evaluation_method", "anchor_budget"]
        if column in frame.columns
    ]


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["true_slowdown"].to_numpy(dtype=float)
    predicted = frame["predicted_slowdown"].to_numpy(dtype=float)
    if not np.isfinite(true).all() or (true < 1.0).any():
        raise ValueError("True slowdowns must be finite and at least 1")
    if not np.isfinite(predicted).all() or (predicted < 1.0).any():
        raise ValueError("Predicted slowdowns must be finite and at least 1")
    log_error = np.log(predicted) - np.log(true)
    raw_error = predicted - true
    if len(frame) > 1 and np.ptp(predicted) > 1e-12 and np.ptp(true) > 1e-12:
        correlation = spearmanr(predicted, true).correlation
        correlation_defined = bool(np.isfinite(correlation))
    else:
        correlation = 0.0
        correlation_defined = False
    if not correlation_defined:
        correlation = 0.0
    return {
        "n": len(frame),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "median_absolute_log_error": float(np.median(np.abs(log_error))),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(log_error)))),
        "spearman": float(correlation),
        "spearman_defined": correlation_defined,
    }


def metrics_by_method(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [*metric_strata(predictions), "method"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(group_columns, keys)), **compute_metrics(group)})
    return pd.DataFrame(rows)


def self_pair_averaged(predictions: pd.DataFrame) -> pd.DataFrame:
    non_self = predictions[predictions["victim_id"] != predictions["aggressor_id"]].copy()
    self_pairs = predictions[predictions["victim_id"] == predictions["aggressor_id"]]
    if self_pairs.empty:
        return predictions.copy()
    group_columns = [
        *metric_strata(predictions),
        "method",
        "pair_row_id",
        "victim_id",
        "aggressor_id",
    ]
    numeric = [
        column
        for column in self_pairs.select_dtypes(include=[np.number]).columns
        if column not in group_columns
    ]
    averaged = self_pairs.groupby(group_columns, as_index=False)[numeric].mean()
    averaged["direction"] = "self_average"
    return pd.concat([non_self, averaged], ignore_index=True, sort=False)


def per_victim_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    strata = metric_strata(predictions)
    group_columns = [*strata, "method", "victim_id"]
    for keys, group in predictions.groupby(group_columns):
        rows.append({**dict(zip(group_columns, keys)), **compute_metrics(group)})
    detailed = pd.DataFrame(rows)
    macro_columns = [*strata, "method"]
    macro = detailed.groupby(macro_columns, as_index=False)[METRIC_COLUMNS].mean()
    macro.insert(len(macro_columns), "victim_id", "macro_average")
    return detailed, macro


def cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    pair_ids = predictions["pair_row_id"].drop_duplicates().to_numpy()
    method_clusters = {
        method: set(frame["pair_row_id"])
        for method, frame in predictions.groupby("method", sort=False)
    }
    expected = set(pair_ids)
    if any(clusters != expected for clusters in method_clusters.values()):
        raise ValueError("Every method must contain the same pair clusters")
    sampled_clusters = cluster_bootstrap_draws(pair_ids, samples=samples, seed=seed)
    rows = []
    for method, method_frame in predictions.groupby("method", sort=False):
        draws = {metric: [] for metric in METRIC_COLUMNS}
        for sampled in sampled_clusters:
            replicate = resample_pair_clusters(method_frame, sampled)
            result = compute_metrics(replicate)
            for metric in METRIC_COLUMNS:
                draws[metric].append(result[metric])
        point = compute_metrics(method_frame)
        for metric in METRIC_COLUMNS:
            values = np.asarray(draws[metric], dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_samples": samples,
                    "cluster_count": len(pair_ids),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_draws(
    pair_ids: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    """Generate the one shared cluster draw matrix used by every method."""
    pair_ids = np.asarray(pair_ids)
    if len(pair_ids) == 0:
        raise ValueError("Bootstrap requires at least one pair cluster")
    rng = np.random.default_rng(seed)
    return rng.choice(pair_ids, size=(samples, len(pair_ids)), replace=True)


def resample_pair_clusters(frame: pd.DataFrame, sampled_ids: np.ndarray) -> pd.DataFrame:
    """Resample complete original pair rows, retaining all directional outcomes."""
    groups = {key: value for key, value in frame.groupby("pair_row_id", sort=False)}
    missing = set(np.asarray(sampled_ids).tolist()) - set(groups)
    if missing:
        raise ValueError(f"Bootstrap draw references missing pair clusters: {sorted(missing)}")
    return pd.concat([groups[value] for value in sampled_ids], ignore_index=True)


def paired_cluster_bootstrap_differences(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Paired method-minus-comparison intervals from shared pair-cluster draws."""
    methods = list(predictions["method"].drop_duplicates())
    pair_ids = predictions["pair_row_id"].drop_duplicates().to_numpy()
    frames = {
        method: predictions[predictions["method"] == method]
        for method in methods
    }
    expected = set(pair_ids)
    if any(set(frame["pair_row_id"]) != expected for frame in frames.values()):
        raise ValueError("Every method must contain the same pair clusters")
    sampled_clusters = cluster_bootstrap_draws(pair_ids, samples=samples, seed=seed)
    sampled_metrics: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in METRIC_COLUMNS} for method in methods
    }
    for sampled in sampled_clusters:
        for method, frame in frames.items():
            result = compute_metrics(resample_pair_clusters(frame, sampled))
            for metric in METRIC_COLUMNS:
                sampled_metrics[method][metric].append(result[metric])
    points = {method: compute_metrics(frame) for method, frame in frames.items()}
    rows = []
    for method, comparison in combinations(methods, 2):
        for metric in METRIC_COLUMNS:
            differences = np.asarray(sampled_metrics[method][metric]) - np.asarray(
                sampled_metrics[comparison][metric]
            )
            finite = differences[np.isfinite(differences)]
            rows.append(
                {
                    "method": method,
                    "comparison_method": comparison,
                    "metric": metric,
                    "estimate_difference": points[method][metric]
                    - points[comparison][metric],
                    "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_samples": samples,
                    "cluster_count": len(pair_ids),
                }
            )
    return pd.DataFrame(rows)
