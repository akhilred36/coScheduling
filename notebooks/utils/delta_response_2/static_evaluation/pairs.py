"""Strict static-pair validation and deterministic direction expansion."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from .constants import (
    DIRECTIONAL_ROWS,
    EVIDENCE_LABEL,
    NON_SELF_CLUSTERS,
    PAIR_MANIFEST_COLUMNS,
    PAIR_REQUIRED_COLUMNS,
    PRIMARY_DIRECTIONAL_ROWS,
    SELF_PAIR_ROWS,
    STATIC_PAIR_ROWS,
)


def validate_pair_frame(
    frame: pd.DataFrame,
    valid_application_ids: Collection[str],
    *,
    expected_rows: int = STATIC_PAIR_ROWS,
    expected_self_rows: int | None = SELF_PAIR_ROWS,
) -> pd.DataFrame:
    missing = sorted(set(PAIR_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"pair table is missing required columns: {missing}")
    result = frame.loc[:, list(PAIR_REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    if len(result) != expected_rows:
        raise ValueError(f"pair table must contain exactly {expected_rows} rows")

    for column in ("jobA_id", "jobB_id"):
        if result[column].isna().any():
            raise ValueError(f"pair table contains a null {column}")
        result[column] = result[column].astype(str)
        unknown = sorted(set(result[column]) - {str(value) for value in valid_application_ids})
        if unknown:
            raise ValueError(f"pair table contains unknown application IDs: {unknown}")

    for column in ("slowdown_A", "slowdown_B"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 1.0).any():
            raise ValueError(f"{column} must be finite and at least one")
        result[column] = values

    unordered = [tuple(sorted((left, right))) for left, right in result[["jobA_id", "jobB_id"]].itertuples(index=False, name=None)]
    if len(unordered) != len(set(unordered)):
        raise ValueError("pair table contains a duplicate or reversed duplicate")

    result.insert(0, "pair_row_id", np.arange(len(result), dtype=int))
    result.insert(1, "pair_cluster_id", result["pair_row_id"].to_numpy(dtype=int))
    result["self_pair"] = result["jobA_id"].eq(result["jobB_id"])
    if expected_self_rows is not None and int(result["self_pair"].sum()) != expected_self_rows:
        raise ValueError(f"pair table must contain exactly {expected_self_rows} self pairs")
    return result


def pair_manifest(validated_pairs: pd.DataFrame) -> pd.DataFrame:
    manifest = validated_pairs[
        ["pair_row_id", "pair_cluster_id", "jobA_id", "jobB_id", "self_pair"]
    ].copy()
    manifest.insert(0, "evaluation_method", "random_split")
    manifest["jobA_known"] = True
    manifest["jobB_known"] = True
    manifest["known_endpoint_count"] = 2
    manifest["included"] = True
    manifest["evidence_label"] = EVIDENCE_LABEL
    return manifest.loc[:, list(PAIR_MANIFEST_COLUMNS)]


def expand_directions(validated_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in validated_pairs.itertuples(index=False):
        common = {
            "evaluation_method": "random_split",
            "pair_row_id": int(row.pair_row_id),
            "pair_cluster_id": int(row.pair_cluster_id),
            "self_pair": bool(row.self_pair),
        }
        rows.append(
            {
                **common,
                "direction": "A",
                "victim_id": str(row.jobA_id),
                "aggressor_id": str(row.jobB_id),
                "true_slowdown": float(row.slowdown_A),
            }
        )
        rows.append(
            {
                **common,
                "direction": "B",
                "victim_id": str(row.jobB_id),
                "aggressor_id": str(row.jobA_id),
                "true_slowdown": float(row.slowdown_B),
            }
        )
    result = pd.DataFrame(rows)
    result["true_log_slowdown"] = np.log(result["true_slowdown"].to_numpy(dtype=float))
    if not np.isfinite(result["true_log_slowdown"]).all():
        raise ValueError("direction expansion produced non-finite targets")
    if len(validated_pairs) == STATIC_PAIR_ROWS:
        if len(result) != DIRECTIONAL_ROWS:
            raise RuntimeError("static direction expansion did not produce 110 rows")
        if int((~result["self_pair"]).sum()) != PRIMARY_DIRECTIONAL_ROWS:
            raise RuntimeError("static direction expansion did not produce 90 non-self rows")
        if result.loc[~result["self_pair"], "pair_cluster_id"].nunique() != NON_SELF_CLUSTERS:
            raise RuntimeError("static direction expansion did not preserve 45 clusters")
    return result


def outcome_blind_queries(directions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "evaluation_method",
        "pair_row_id",
        "pair_cluster_id",
        "direction",
        "victim_id",
        "aggressor_id",
        "self_pair",
    ]
    return directions.loc[:, columns].copy()
