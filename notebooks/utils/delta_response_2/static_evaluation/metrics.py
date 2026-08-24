"""Static-evaluation metrics with explicit undefined-estimate semantics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .constants import (
    EVIDENCE_LABEL,
    ERROR_METRICS,
    METRIC_ESTIMATES,
    METRIC_FLAGS,
)


RANGE_TOLERANCE = 1e-12


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, bool]:
    if (
        len(left) < 2
        or np.ptp(left) <= RANGE_TOLERANCE
        or np.ptp(right) <= RANGE_TOLERANCE
    ):
        return None, False
    value = float(np.corrcoef(left, right)[0, 1])
    return (value, True) if np.isfinite(value) else (None, False)


def compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    if len(frame) == 0:
        raise ValueError("metrics require at least one observation")
    required = {
        "true_log_slowdown",
        "predicted_log_slowdown",
        "true_slowdown",
        "predicted_slowdown",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metric frame is missing columns: {missing}")
    y = frame["true_log_slowdown"].to_numpy(dtype=float)
    p = frame["predicted_log_slowdown"].to_numpy(dtype=float)
    s = frame["true_slowdown"].to_numpy(dtype=float)
    q = frame["predicted_slowdown"].to_numpy(dtype=float)
    if (
        not np.isfinite(np.column_stack([y, p, s, q])).all()
        or (y < 0).any()
        or (p < 0).any()
        or (s < 1).any()
        or (q < 1).any()
    ):
        raise ValueError("metric inputs must be finite and on observed support")
    if not np.allclose(np.exp(y), s, rtol=1e-10, atol=1e-12):
        raise ValueError("true raw and log slowdowns are inconsistent")
    if not np.allclose(np.exp(p), q, rtol=1e-10, atol=1e-12):
        raise ValueError("predicted raw and log slowdowns are inconsistent")

    error = p - y
    raw_error = q - s
    pearson, pearson_defined = _correlation(p, y)
    if (
        len(p) < 2
        or np.ptp(p) <= RANGE_TOLERANCE
        or np.ptp(y) <= RANGE_TOLERANCE
    ):
        spearman, spearman_defined = None, False
    else:
        spearman, spearman_defined = _correlation(rankdata(p), rankdata(y))

    if len(p) >= 2 and np.ptp(p) > RANGE_TOLERANCE:
        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(len(p)), p]), y, rcond=None
        )[0]
        calibration_defined = bool(np.isfinite(coefficients).all())
    else:
        coefficients = np.asarray([np.nan, np.nan])
        calibration_defined = False
    intercept = float(coefficients[0]) if calibration_defined else None
    slope = float(coefficients[1]) if calibration_defined else None

    return {
        "n": int(len(frame)),
        "log_mae": float(np.mean(np.abs(error))),
        "log_mse": float(np.mean(error**2)),
        "log_rmse": float(np.sqrt(np.mean(error**2))),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "raw_mape": float(100.0 * np.mean(np.abs(raw_error) / s)),
        "raw_mse": float(np.mean(raw_error**2)),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "median_absolute_log_error": float(np.median(np.abs(error))),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(error)))),
        "signed_log_bias": float(np.mean(error)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_defined": calibration_defined,
        "spearman": spearman,
        "spearman_defined": spearman_defined,
        "pearson": pearson,
        "pearson_defined": pearson_defined,
        "mean_true_log_slowdown": float(np.mean(y)),
        "mean_predicted_log_slowdown": float(np.mean(p)),
        "overprediction_count": int(np.sum(error > 0)),
        "underprediction_count": int(np.sum(error < 0)),
    }


def metric_rows(
    predictions: pd.DataFrame,
    *,
    scope: str,
    group_columns: Iterable[str] = ("method", "candidate_id"),
) -> pd.DataFrame:
    grouping = list(group_columns)
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(grouping, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(grouping, key_values))
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "evaluation_method": "random_split",
                "scope": scope,
                **base,
                **compute_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def self_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[predictions["self_pair"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    grouping = ["method", "candidate_id", "pair_cluster_id"]
    for keys, group in selected.groupby(grouping, sort=False):
        if len(group) != 2:
            raise ValueError("each self pair must have exactly two directions per method")
        predicted_log = group["predicted_log_slowdown"].to_numpy(dtype=float)
        latent = group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
        if np.ptp(predicted_log) > 1e-12 or np.ptp(latent) > 1e-12:
            raise ValueError("self directions do not have an identical model prediction")
        true_slowdown = float(group["true_slowdown"].mean())
        first = group.iloc[0]
        rows.append(
            {
                "method": keys[0],
                "candidate_id": keys[1],
                "pair_cluster_id": int(keys[2]),
                "self_pair": True,
                "victim_id": str(first["victim_id"]),
                "aggressor_id": str(first["aggressor_id"]),
                "latent_predicted_log_slowdown": float(latent[0]),
                "predicted_log_slowdown": float(predicted_log[0]),
                "predicted_slowdown": float(np.exp(predicted_log[0])),
                "true_slowdown": true_slowdown,
                "true_log_slowdown": float(np.log(true_slowdown)),
            }
        )
    return pd.DataFrame(rows)


def macro_victim_metrics(per_victim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, candidate_id), group in per_victim.groupby(
        ["method", "candidate_id"], sort=False
    ):
        if group["victim_id"].nunique() != 10 or len(group) != 10:
            raise ValueError(f"method {method} does not have ten per-victim rows")
        row: dict[str, object] = {
            "evidence_label": EVIDENCE_LABEL,
            "evaluation_method": "random_split",
            "scope": "non_self_macro_victim",
            "method": method,
            "candidate_id": candidate_id,
            "victim_count": 10,
            "n": 10,
        }
        for metric in METRIC_ESTIMATES:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            row[metric] = float(np.mean(values[finite])) if finite.any() else None
        calibration_count = int(group["calibration_defined"].astype(bool).sum())
        spearman_count = int(group["spearman_defined"].astype(bool).sum())
        pearson_count = int(group["pearson_defined"].astype(bool).sum())
        row.update(
            {
                "calibration_defined": calibration_count > 0,
                "calibration_defined_victim_count": calibration_count,
                "spearman_defined": spearman_count > 0,
                "spearman_defined_victim_count": spearman_count,
                "pearson_defined": pearson_count > 0,
                "pearson_defined_victim_count": pearson_count,
                "defined_victim_count": (
                    calibration_count
                    if calibration_count == spearman_count == pearson_count
                    else None
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, candidate_id), group in metrics_by_seed.groupby(
        ["method", "candidate_id"], sort=False
    ):
        if sorted(group["seed"].astype(int).tolist()) != [0, 1, 2, 3, 4]:
            raise ValueError(f"seed metrics are incomplete for {method}")
        for metric in METRIC_ESTIMATES:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values):
                if metric in {"signed_log_bias", "calibration_intercept"}:
                    best = finite_values[np.argmin(np.abs(finite_values))]
                    worst = finite_values[np.argmax(np.abs(finite_values))]
                    best_worst_definition = "closest_to_zero_farthest_from_zero"
                elif metric == "calibration_slope":
                    best = finite_values[np.argmin(np.abs(finite_values - 1.0))]
                    worst = finite_values[np.argmax(np.abs(finite_values - 1.0))]
                    best_worst_definition = "closest_to_one_farthest_from_one"
                elif metric in {"spearman", "pearson"}:
                    best = np.max(finite_values)
                    worst = np.min(finite_values)
                    best_worst_definition = "maximum_minimum"
                elif metric in ERROR_METRICS:
                    best = np.min(finite_values)
                    worst = np.max(finite_values)
                    best_worst_definition = "minimum_maximum"
                else:
                    best = None
                    worst = None
                    best_worst_definition = "not_prespecified"
                mean = float(np.mean(finite_values))
                standard_deviation = float(np.std(finite_values, ddof=0))
                best_value = float(best) if best is not None else None
                worst_value = float(worst) if worst is not None else None
            else:
                mean = None
                standard_deviation = None
                best_value = None
                worst_value = None
                best_worst_definition = "undefined"
            rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "evaluation_method": "random_split",
                    "scope": "non_self_per_seed",
                    "method": method,
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "seed_mean": mean,
                    "seed_standard_deviation": standard_deviation,
                    "seed_best": best_value,
                    "seed_worst": worst_value,
                    "seed_count": 5,
                    "defined_seed_count": int(len(finite_values)),
                    "best_worst_definition": best_worst_definition,
                }
            )
    return pd.DataFrame(rows)


def validate_metric_null_semantics(frame: pd.DataFrame) -> None:
    for flag, estimates in {
        "calibration_defined": ("calibration_intercept", "calibration_slope"),
        "spearman_defined": ("spearman",),
        "pearson_defined": ("pearson",),
    }.items():
        if flag not in frame:
            continue
        for row in frame.itertuples(index=False):
            defined = bool(getattr(row, flag))
            for estimate in estimates:
                value = getattr(row, estimate)
                finite = value is not None and not pd.isna(value) and np.isfinite(float(value))
                if defined != finite:
                    raise ValueError(f"{estimate}/{flag} null semantics are inconsistent")
