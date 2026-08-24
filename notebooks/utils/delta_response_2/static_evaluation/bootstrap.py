"""Shared paired bootstrap clustered by original unordered static pair."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_METRICS,
    BOOTSTRAP_QUANTILES,
    BOOTSTRAP_SEED,
    EVIDENCE_LABEL,
    NON_SELF_CLUSTERS,
)
from .metrics import compute_metrics


SIGN_CONVENTION = "first_method_minus_second_method"
ERROR_SIGN_CONVENTION = "negative_error_metric_difference_favors_first_method"


def make_cluster_draws(
    cluster_ids: Sequence[int],
    *,
    samples: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    clusters = np.asarray(list(cluster_ids), dtype=int)
    if len(clusters) == 0 or len(np.unique(clusters)) != len(clusters):
        raise ValueError("bootstrap cluster IDs must be non-empty and unique")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    draws = np.random.default_rng(seed).integers(
        0, len(clusters), size=(samples, len(clusters))
    )
    return clusters, draws


def bootstrap_tables(
    predictions: pd.DataFrame,
    method_order: Sequence[str],
    *,
    samples: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    expected_cluster_count: int | None = NON_SELF_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    selected = predictions[~predictions["self_pair"].astype(bool)].copy()
    methods = list(method_order)
    if set(selected["method"].astype(str)) != set(methods):
        raise ValueError("bootstrap method roster does not match the declared method order")
    cluster_ids = sorted(selected["pair_cluster_id"].astype(int).unique().tolist())
    if expected_cluster_count is not None and len(cluster_ids) != expected_cluster_count:
        raise ValueError(
            f"bootstrap requires exactly {expected_cluster_count} unordered clusters"
        )
    clusters, draws = make_cluster_draws(cluster_ids, samples=samples, seed=seed)

    by_method: dict[str, pd.DataFrame] = {}
    positions: dict[str, dict[int, np.ndarray]] = {}
    expected_keys: set[tuple[int, str, str, str]] | None = None
    expected_targets: pd.DataFrame | None = None
    identity_columns = [
        "pair_cluster_id",
        "direction",
        "victim_id",
        "aggressor_id",
    ]
    for method in methods:
        frame = selected[selected["method"].eq(method)].reset_index(drop=True)
        keys = set(
            frame[identity_columns].itertuples(index=False, name=None)
        )
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"method {method} has different directional membership")
        targets = frame[
            [*identity_columns, "true_log_slowdown", "true_slowdown"]
        ].sort_values(identity_columns, kind="stable").reset_index(drop=True)
        if expected_targets is None:
            expected_targets = targets
        elif (
            not targets[identity_columns].equals(expected_targets[identity_columns])
            or not np.allclose(
                targets[["true_log_slowdown", "true_slowdown"]].to_numpy(dtype=float),
                expected_targets[["true_log_slowdown", "true_slowdown"]].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"method {method} has different paired true outcomes")
        grouped = {
            int(cluster): group.index.to_numpy(dtype=int)
            for cluster, group in frame.groupby("pair_cluster_id", sort=False)
        }
        if set(grouped) != set(cluster_ids) or any(len(value) != 2 for value in grouped.values()):
            raise ValueError(f"method {method} does not preserve both pair directions")
        by_method[method] = frame
        positions[method] = grouped

    replicate_values: dict[str, dict[str, np.ndarray]] = {
        method: {
            metric: np.full(samples, np.nan, dtype=float)
            for metric in BOOTSTRAP_METRICS
        }
        for method in methods
    }
    point_values = {
        method: compute_metrics(frame) for method, frame in by_method.items()
    }
    for replicate_index, draw in enumerate(draws):
        sampled_clusters = clusters[draw]
        for method in methods:
            row_positions = np.concatenate(
                [positions[method][int(cluster)] for cluster in sampled_clusters]
            )
            estimates = compute_metrics(by_method[method].iloc[row_positions])
            for metric in BOOTSTRAP_METRICS:
                value = estimates[metric]
                if value is not None:
                    replicate_values[method][metric][replicate_index] = float(value)

    interval_rows: list[dict[str, object]] = []
    for method in methods:
        candidate_id = str(by_method[method]["candidate_id"].iloc[0])
        for metric in BOOTSTRAP_METRICS:
            finite = replicate_values[method][metric]
            finite = finite[np.isfinite(finite)]
            lower, upper = (
                np.quantile(finite, BOOTSTRAP_QUANTILES)
                if len(finite)
                else (np.nan, np.nan)
            )
            point = point_values[method][metric]
            interval_rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "evaluation_method": "random_split",
                    "scope": "non_self_ensemble",
                    "method": method,
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "point_estimate": point,
                    "ci_lower": float(lower) if np.isfinite(lower) else None,
                    "ci_upper": float(upper) if np.isfinite(upper) else None,
                    "defined_replicate_count": int(len(finite)),
                    "bootstrap_draws": int(samples),
                    "bootstrap_seed": int(seed),
                    "cluster_count": int(len(clusters)),
                    "cluster_unit": "pair_cluster_id",
                    "interval": "percentile_2.5_97.5",
                }
            )

    paired_rows: list[dict[str, object]] = []
    for first in methods:
        for second in methods:
            if first == second:
                continue
            for metric in BOOTSTRAP_METRICS:
                differences = (
                    replicate_values[first][metric] - replicate_values[second][metric]
                )
                finite = differences[np.isfinite(differences)]
                lower, upper = (
                    np.quantile(finite, BOOTSTRAP_QUANTILES)
                    if len(finite)
                    else (np.nan, np.nan)
                )
                first_point = point_values[first][metric]
                second_point = point_values[second][metric]
                point = (
                    float(first_point) - float(second_point)
                    if first_point is not None and second_point is not None
                    else None
                )
                paired_rows.append(
                    {
                        "evidence_label": EVIDENCE_LABEL,
                        "evaluation_method": "random_split",
                        "scope": "non_self_ensemble",
                        "first_method": first,
                        "second_method": second,
                        "metric": metric,
                        "estimate_difference": point,
                        "ci_lower": float(lower) if np.isfinite(lower) else None,
                        "ci_upper": float(upper) if np.isfinite(upper) else None,
                        "defined_replicate_count": int(len(finite)),
                        "bootstrap_draws": int(samples),
                        "bootstrap_seed": int(seed),
                        "cluster_count": int(len(clusters)),
                        "cluster_unit": "pair_cluster_id",
                        "difference_definition": SIGN_CONVENTION,
                        "error_metric_sign_convention": ERROR_SIGN_CONVENTION,
                    }
                )
    return pd.DataFrame(interval_rows), pd.DataFrame(paired_rows), replicate_values
