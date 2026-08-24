"""Pair-blind in-memory protocol self-checks used before preparation sealing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_tables, make_cluster_draws
from .metrics import compute_metrics
from .pairs import expand_directions, validate_pair_frame


def run_synthetic_self_checks() -> dict[str, object]:
    pair_fixture = pd.DataFrame(
        {
            "jobA_id": ["a", "a", "a", "b", "b", "c"],
            "jobB_id": ["a", "b", "c", "b", "c", "c"],
            "slowdown_A": [1.0, 2.0, 1.5, 1.1, 1.4, 1.2],
            "slowdown_B": [1.0, 1.8, 1.3, 1.2, 1.6, 1.1],
        }
    )
    validated = validate_pair_frame(
        pair_fixture,
        {"a", "b", "c"},
        expected_rows=6,
        expected_self_rows=3,
    )
    directions = expand_directions(validated)
    if len(directions) != 12 or int((~directions["self_pair"]).sum()) != 6:
        raise RuntimeError("synthetic direction expansion failed")
    for cluster, group in directions.groupby("pair_cluster_id"):
        if len(group) != 2 or group["pair_cluster_id"].nunique() != 1:
            raise RuntimeError(f"synthetic cluster {cluster} lost a direction")

    metric_fixture = pd.DataFrame(
        {
            "true_log_slowdown": [0.0, np.log(2.0), np.log(4.0)],
            "predicted_log_slowdown": [0.0, np.log(4.0), np.log(2.0)],
            "true_slowdown": [1.0, 2.0, 4.0],
            "predicted_slowdown": [1.0, 4.0, 2.0],
        }
    )
    metrics = compute_metrics(metric_fixture)
    expected_log_mae = 2.0 * np.log(2.0) / 3.0
    if not np.isclose(metrics["log_mae"], expected_log_mae):
        raise RuntimeError("synthetic log-MAE check failed")
    constant = metric_fixture.assign(predicted_log_slowdown=0.0, predicted_slowdown=1.0)
    undefined = compute_metrics(constant)
    if (
        undefined["spearman_defined"]
        or undefined["pearson_defined"]
        or undefined["calibration_defined"]
        or undefined["spearman"] is not None
        or undefined["calibration_slope"] is not None
    ):
        raise RuntimeError("synthetic undefined-metric semantics failed")

    non_self = directions[~directions["self_pair"]].copy()
    prediction_frames = []
    for method, offset in [("better", 0.0), ("worse", 0.2)]:
        frame = non_self.copy()
        frame["method"] = method
        frame["candidate_id"] = "synthetic"
        frame["predicted_log_slowdown"] = np.maximum(
            0.0, frame["true_log_slowdown"].to_numpy(dtype=float) + offset
        )
        frame["predicted_slowdown"] = np.exp(frame["predicted_log_slowdown"])
        prediction_frames.append(frame)
    bootstrap, paired, _ = bootstrap_tables(
        pd.concat(prediction_frames, ignore_index=True),
        ["better", "worse"],
        samples=40,
        seed=923,
        expected_cluster_count=3,
    )
    contrast = paired[
        paired["first_method"].eq("better")
        & paired["second_method"].eq("worse")
        & paired["metric"].eq("log_mae")
    ].iloc[0]
    if not contrast["estimate_difference"] < 0 or not contrast["ci_upper"] < 0:
        raise RuntimeError("synthetic paired-bootstrap sign check failed")
    clusters, draws = make_cluster_draws([1, 2, 4], samples=40, seed=923)
    if clusters.tolist() != [1, 2, 4] or draws.shape != (40, 3):
        raise RuntimeError("synthetic shared draw-matrix check failed")
    if len(bootstrap) != 2 * 16:
        raise RuntimeError("synthetic bootstrap metric roster failed")
    return {
        "status": "passed",
        "uses_filesystem": False,
        "pair_parser": True,
        "direction_expansion": True,
        "metric_semantics": True,
        "paired_pair_cluster_bootstrap": True,
    }
