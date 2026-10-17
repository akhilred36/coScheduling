"""Evaluation metrics and cluster-aware confidence intervals."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


METRIC_COLUMNS = [
    "log_mae",
    "log_mse",
    "log_rmse",
    "raw_mae",
    "raw_mape",
    "raw_mse",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
    "spearman",
    "signed_log_bias",
]
FALSE_BENIGN_THRESHOLD = 1.05


def metric_strata(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["evaluation_method", "anchor_budget"]
        if column in frame.columns
    ]


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return metrics_from_arrays(
        frame["true_slowdown"].to_numpy(dtype=float),
        frame["predicted_slowdown"].to_numpy(dtype=float),
    )


def metrics_from_arrays(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if not np.isfinite(true).all() or (true < 1.0).any():
        raise ValueError("True slowdowns must be finite and at least 1")
    if not np.isfinite(predicted).all() or (predicted < 1.0).any():
        raise ValueError("Predicted slowdowns must be finite and at least 1")
    log_error = np.log(predicted) - np.log(true)
    raw_error = predicted - true
    if len(true) > 1 and np.ptp(predicted) > 1e-12 and np.ptp(true) > 1e-12:
        # Spearman = Pearson correlation of average-tie ranks (scipy semantics).
        correlation = float(np.corrcoef(rankdata(predicted), rankdata(true))[0, 1])
        correlation_defined = bool(np.isfinite(correlation))
    else:
        correlation = 0.0
        correlation_defined = False
    if not correlation_defined:
        correlation = 0.0
    return {
        "n": len(true),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_mse": float(np.mean(log_error**2)),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "raw_mape": float(100.0 * np.mean(np.abs(raw_error) / true)),
        "raw_mse": float(np.mean(raw_error**2)),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "median_absolute_log_error": float(np.median(np.abs(log_error))),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(log_error)))),
        "spearman": float(correlation),
        "spearman_defined": correlation_defined,
        "signed_log_bias": float(np.mean(log_error)),
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


class _ClusterResampler:
    """Precomputed pair-cluster row positions for fast shared-draw resampling."""

    def __init__(self, frame: pd.DataFrame):
        self.true = frame["true_slowdown"].to_numpy(dtype=float)
        self.predicted = frame["predicted_slowdown"].to_numpy(dtype=float)
        self.positions = {
            key: np.asarray(index, dtype=int)
            for key, index in frame.reset_index(drop=True).groupby("pair_row_id", sort=False).indices.items()
        }

    def metrics(self, sampled_ids: np.ndarray) -> dict[str, float]:
        missing = set(np.asarray(sampled_ids).tolist()) - set(self.positions)
        if missing:
            raise ValueError(f"Bootstrap draw references missing pair clusters: {sorted(missing)}")
        rows = np.concatenate([self.positions[value] for value in sampled_ids])
        return metrics_from_arrays(self.true[rows], self.predicted[rows])


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
        resampler = _ClusterResampler(method_frame)
        draws = {metric: [] for metric in METRIC_COLUMNS}
        for sampled in sampled_clusters:
            result = resampler.metrics(sampled)
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
    resamplers = {method: _ClusterResampler(frame) for method, frame in frames.items()}
    for sampled in sampled_clusters:
        for method in frames:
            result = resamplers[method].metrics(sampled)
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


def common_pair_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict to pair clusters present for every method; report coverage."""
    clusters = predictions.groupby("method")["pair_row_id"].apply(set)
    common = set.intersection(*clusters.tolist()) if len(clusters) else set()
    evaluated = set(predictions["pair_row_id"])
    coverage = pd.DataFrame(
        {
            "method": clusters.index,
            "pair_clusters": [len(value) for value in clusters],
            "missing_pair_clusters": [len(evaluated - value) for value in clusters],
            "common_pair_clusters": len(common),
            "excluded_from_common_pair_clusters": [len(value - common) for value in clusters],
        }
    )
    return predictions[predictions["pair_row_id"].isin(common)].copy(), coverage


def metrics_by_censoring(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Floor-case and above-floor directional diagnostics."""
    frame = predictions.copy()
    frame["censoring_stratum"] = np.where(frame["true_slowdown"] <= 1.0, "floor", "above_floor")
    rows = []
    for stratum_frame, label in [
        (frame, "all"),
        *((group, name) for name, group in frame.groupby("censoring_stratum")),
    ]:
        for keys, group in stratum_frame.groupby(group_columns, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            log_error = np.log(group["predicted_slowdown"]) - np.log(group["true_slowdown"])
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "censoring_stratum": label,
                    "n": len(group),
                    "log_mae": float(np.mean(np.abs(log_error))),
                    "signed_log_bias": float(np.mean(log_error)),
                    "log_rmse": float(np.sqrt(np.mean(log_error**2))),
                }
            )
    return pd.DataFrame(rows)


def auxiliary_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """MAE/RMSE/bias of auxiliary log-rate degradations on valid labels only."""
    valid = frame[frame["valid"].astype(bool)].copy()
    rows = []
    for keys, group in valid.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        error = group["predicted"].to_numpy(float) - group["true"].to_numpy(float)
        correlation = (
            float(np.corrcoef(group["predicted"], group["true"])[0, 1])
            if len(group) > 2 and group["predicted"].std() > 0 and group["true"].std() > 0
            else np.nan
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "pearson": correlation,
            }
        )
    return pd.DataFrame(rows)


def pair_level_predictions(
    directional: pd.DataFrame,
    *,
    threshold: float = FALSE_BENIGN_THRESHOLD,
    group_columns: tuple[str, ...] = ("evaluation_method", "method"),
) -> pd.DataFrame:
    """Severity, imbalance, worst endpoint, and false-benign flags per pair."""
    keys = [column for column in group_columns if column in directional]
    rows = []
    for key_values, group in directional.groupby([*keys, "pair_row_id"], sort=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        by_direction = group.drop_duplicates("direction").set_index("direction")
        if not {"A", "B"}.issubset(by_direction.index):
            continue
        a, b = by_direction.loc["A"], by_direction.loc["B"]
        true_a, true_b = np.log(a["true_slowdown"]), np.log(b["true_slowdown"])
        pred_a, pred_b = a["predicted_log_slowdown"], b["predicted_log_slowdown"]
        true_worst, predicted_worst = max(true_a, true_b), max(pred_a, pred_b)
        rows.append(
            {
                **dict(zip([*keys, "pair_row_id"], key_values)),
                "jobA_id": a["victim_id"],
                "jobB_id": b["victim_id"],
                "true_log_slowdown_A": true_a,
                "true_log_slowdown_B": true_b,
                "predicted_log_slowdown_A": pred_a,
                "predicted_log_slowdown_B": pred_b,
                "true_severity": 0.5 * (true_a + true_b),
                "predicted_severity": 0.5 * (pred_a + pred_b),
                "true_imbalance": true_a - true_b,
                "predicted_imbalance": pred_a - pred_b,
                "true_worst_log_slowdown": true_worst,
                "predicted_worst_log_slowdown": predicted_worst,
                "worst_endpoint_abs_log_error": abs(predicted_worst - true_worst),
                "false_benign": bool(true_worst >= np.log(threshold) and predicted_worst < np.log(threshold)),
                "truly_harmful": bool(true_worst >= np.log(threshold)),
                "swap_equivariance_residual": float(
                    np.nanmax([a.get("swap_equivariance_residual", np.nan),
                               b.get("swap_equivariance_residual", np.nan)])
                ) if "swap_equivariance_residual" in group and group["swap_equivariance_residual"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pair_metrics(pairs: pd.DataFrame, group_columns: tuple[str, ...] = ("evaluation_method", "method")) -> pd.DataFrame:
    keys = [column for column in group_columns if column in pairs]
    rows = []
    for key_values, group in pairs.groupby(keys, sort=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        harmful = group[group["truly_harmful"]]
        ranking = (
            float(spearmanr(group["predicted_worst_log_slowdown"], group["true_worst_log_slowdown"]).correlation)
            if len(group) > 1 and group["predicted_worst_log_slowdown"].std() > 0
            and group["true_worst_log_slowdown"].std() > 0 else np.nan
        )
        rows.append(
            {
                **dict(zip(keys, key_values)),
                "pairs": len(group),
                "severity_mae": float((group["predicted_severity"] - group["true_severity"]).abs().mean()),
                "imbalance_mae": float((group["predicted_imbalance"] - group["true_imbalance"]).abs().mean()),
                "imbalance_sign_agreement": float(
                    (np.sign(group["predicted_imbalance"]) == np.sign(group["true_imbalance"])).mean()
                ),
                "worst_endpoint_log_mae": float(group["worst_endpoint_abs_log_error"].mean()),
                "worst_endpoint_spearman": ranking,
                "truly_harmful_pairs": len(harmful),
                "false_benign_rate": float(harmful["false_benign"].mean()) if len(harmful) else np.nan,
                "false_benign_threshold": FALSE_BENIGN_THRESHOLD,
                "max_swap_equivariance_residual": float(group["swap_equivariance_residual"].max())
                if group["swap_equivariance_residual"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
