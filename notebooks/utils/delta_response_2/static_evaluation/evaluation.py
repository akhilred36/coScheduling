"""Static evaluation execution, output sealing, and complete-state verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from delta_response_2.data import read_csv
from delta_response_2.frozen import predict_frozen_recipe

from .archive import load_archived_comparison, verify_archive_hashes
from .bootstrap import bootstrap_tables
from .constants import (
    ARCHIVE_METHODS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    COMPLETION_METADATA_FILES,
    CORRECTION_DIAGNOSTIC_COLUMNS,
    DEFAULT_OUTPUT_ROOT,
    DELTA2_METHODS,
    DIRECTIONAL_PREDICTION_COLUMNS,
    DIRECTIONAL_ROWS,
    EVALUATION_PRODUCT_FILES,
    EVIDENCE_LABEL,
    EVIDENCE_STATUS,
    EXPECTED_ARCHIVE_HASHES,
    EXPECTED_INPUT_HASHES,
    FROZEN_RECIPE_SHA256,
    FROZEN_ROOT,
    FUTURE_EVIDENCE_REQUIREMENT,
    METHOD_CANDIDATES,
    NAMESPACED_ARCHIVE_METHODS,
    NON_SELF_CLUSTERS,
    PAIR_PATH,
    PRIMARY_DIRECTIONAL_ROWS,
    SEEDS,
    SEEDED_METHODS,
    SEED_PREDICTION_COLUMNS,
    completed_roster,
    prepared_roster,
)
from .integrity import (
    canonical_digest,
    hash_roster,
    read_canonical_json,
    regular_file_roster,
    require_regular_file,
    sha256_file,
    verify_exact_roster,
    write_csv,
    write_json,
)
from .metrics import (
    macro_victim_metrics,
    metric_rows,
    seed_stability,
    self_averaged_predictions,
    validate_metric_null_semantics,
)
from .pairs import (
    expand_directions,
    outcome_blind_queries,
    pair_manifest,
    validate_pair_frame,
)
from .prediction import (
    KEY_COLUMNS,
    _validate_prediction_invariants,
    build_delta2_predictions,
)
from .preparation import require_cpu_only, verify_preparation
from .reporting import (
    design_diagnostic_table,
    hypothesis_results,
    render_report,
)


ALL_METHODS = (*DELTA2_METHODS, *NAMESPACED_ARCHIVE_METHODS)
PREDICTION_ARTIFACTS = (
    "evaluation_pair_manifest.csv",
    "seed_predictions.csv",
    "directional_predictions.csv",
)


def _attach_outcomes(
    predictions: pd.DataFrame,
    directions: pd.DataFrame,
    *,
    seeded: bool,
) -> pd.DataFrame:
    truth = directions[
        [*KEY_COLUMNS, "true_log_slowdown", "true_slowdown"]
    ].copy()
    result = predictions.merge(
        truth,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("prediction rows could not be attached to every static outcome")
    result = result.drop(columns="_merge")
    required_numeric = [
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
        "true_log_slowdown",
        "true_slowdown",
    ]
    if not np.isfinite(result[required_numeric].to_numpy(dtype=float)).all():
        raise FloatingPointError("prediction or target value is non-finite")
    if not np.allclose(
        np.exp(result["predicted_log_slowdown"].to_numpy(dtype=float)),
        result["predicted_slowdown"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("prediction raw/log values are inconsistent")
    if not np.allclose(
        np.exp(result["true_log_slowdown"].to_numpy(dtype=float)),
        result["true_slowdown"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("target raw/log values are inconsistent")
    required = SEED_PREDICTION_COLUMNS if seeded else DIRECTIONAL_PREDICTION_COLUMNS
    missing = sorted(set(required) - set(result.columns))
    if missing:
        raise RuntimeError(f"prediction output is missing required columns: {missing}")
    order = [
        *required,
        *CORRECTION_DIAGNOSTIC_COLUMNS,
        "evidence_label",
    ]
    return result.loc[:, order]


def _validate_ensemble_construction(
    seed_predictions: pd.DataFrame, directional_predictions: pd.DataFrame
) -> None:
    for method in SEEDED_METHODS:
        seeds = seed_predictions[seed_predictions["method"].eq(method)]
        ensemble = directional_predictions[
            directional_predictions["method"].eq(method)
        ]
        grouped = seeds.groupby(list(KEY_COLUMNS), sort=False)
        if len(grouped) != DIRECTIONAL_ROWS or len(ensemble) != DIRECTIONAL_ROWS:
            raise RuntimeError(f"ensemble coverage is incomplete for {method}")
        expected_rows: list[dict[str, object]] = []
        for keys, group in grouped:
            if sorted(group["seed"].astype(int)) != list(SEEDS):
                raise RuntimeError(f"seed roster changed for {method}")
            latent_values = group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
            latent = float(np.mean(latent_values))
            expected_rows.append(
                {
                    **dict(zip(KEY_COLUMNS, keys)),
                    "latent": latent,
                    "observed": max(0.0, latent),
                    "standard_deviation": float(np.std(latent_values, ddof=0)),
                }
            )
        expected = pd.DataFrame(expected_rows)
        checked = ensemble.merge(
            expected,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(checked) != DIRECTIONAL_ROWS or not np.allclose(
            checked["latent_predicted_log_slowdown"], checked["latent"], atol=1e-12
        ) or not np.allclose(
            checked["predicted_log_slowdown"], checked["observed"], atol=1e-12
        ) or not np.allclose(
            checked["model_seed_log_std"], checked["standard_deviation"], atol=1e-12
        ):
            raise RuntimeError(f"latent ensemble invariant failed for {method}")


def _validate_recomputed_baseline_parity(
    delta2: pd.DataFrame, archive: pd.DataFrame
) -> list[dict[str, object]]:
    mappings = {
        "delta2_constant_1": "delta1_constant_1",
        "delta2_victim_inhibitor_median": "delta1_victim_inhibitor_median",
        "delta2_nearest_anchor": "delta1_nearest_anchor",
    }
    identity = [
        "pair_row_id",
        "pair_cluster_id",
        "direction",
        "victim_id",
        "aggressor_id",
    ]
    value_columns = [
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
    ]
    records: list[dict[str, object]] = []
    for new_method, archived_method in mappings.items():
        new_rows = delta2[delta2["method"].eq(new_method)][
            [*identity, *value_columns]
        ]
        archived_rows = archive[archive["method"].eq(archived_method)][
            [*identity, *value_columns]
        ]
        checked = new_rows.merge(
            archived_rows,
            on=identity,
            how="inner",
            validate="one_to_one",
            suffixes=("_new", "_archive"),
        )
        if len(checked) != DIRECTIONAL_ROWS:
            raise RuntimeError(f"baseline parity membership differs for {new_method}")
        maximum = 0.0
        for column in value_columns:
            differences = np.abs(
                checked[f"{column}_new"].to_numpy(dtype=float)
                - checked[f"{column}_archive"].to_numpy(dtype=float)
            )
            maximum = max(maximum, float(np.max(differences)))
            if not np.allclose(
                checked[f"{column}_new"].to_numpy(dtype=float),
                checked[f"{column}_archive"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"recomputed baseline differs from the archive: {new_method}/{column}"
                )
        records.append(
            {
                "new_method": new_method,
                "archived_method": archived_method,
                "row_count": DIRECTIONAL_ROWS,
                "maximum_absolute_difference": maximum,
                "status": "passed",
            }
        )
    return records


def _external_oracle_checks(
    directions: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    directional_predictions: pd.DataFrame,
) -> list[dict[str, object]]:
    checks = directions[~directions["self_pair"].astype(bool)].iloc[:10]
    if len(checks) < 10:
        raise RuntimeError("fewer than ten non-self directions are available for replay")
    records: list[dict[str, object]] = []
    for row in checks.itertuples(index=False):
        oracle = predict_frozen_recipe(
            FROZEN_ROOT,
            str(row.victim_id),
            str(row.aggressor_id),
            device="cpu",
        )
        key_mask = (
            seed_predictions["pair_row_id"].eq(int(row.pair_row_id))
            & seed_predictions["direction"].eq(str(row.direction))
            & seed_predictions["method"].eq("delta2_selected")
        )
        observed_seeds = seed_predictions[key_mask].sort_values("seed")[
            "latent_predicted_log_slowdown"
        ].to_numpy(dtype=float)
        expected_seeds = np.asarray(oracle["seed_latent_predictions"], dtype=float)
        ensemble = directional_predictions[
            directional_predictions["pair_row_id"].eq(int(row.pair_row_id))
            & directional_predictions["direction"].eq(str(row.direction))
            & directional_predictions["method"].eq("delta2_selected")
        ]
        if (
            observed_seeds.shape != (5,)
            or expected_seeds.shape != (5,)
            or not np.allclose(observed_seeds, expected_seeds, rtol=0.0, atol=1e-10)
            or len(ensemble) != 1
            or not np.isclose(
                float(ensemble.iloc[0]["latent_predicted_log_slowdown"]),
                float(oracle["latent_predicted_log_slowdown"]),
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise RuntimeError(
                f"external frozen replay disagrees for {row.victim_id}/{row.aggressor_id}"
            )
        records.append(
            {
                "pair_row_id": int(row.pair_row_id),
                "direction": str(row.direction),
                "victim_id": str(row.victim_id),
                "aggressor_id": str(row.aggressor_id),
                "maximum_seed_absolute_difference": float(
                    np.max(np.abs(observed_seeds - expected_seeds))
                ),
                "status": "passed",
            }
        )
    return records


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {method: index for index, method in enumerate(ALL_METHODS)}
    result = frame.copy()
    result["_method_order"] = result["method"].map(order)
    if result["_method_order"].isna().any():
        raise RuntimeError("table contains a method outside the frozen method panel")
    secondary = [
        column
        for column in ("victim_id", "seed", "metric")
        if column in result.columns
    ]
    return result.sort_values(
        ["_method_order", *secondary], kind="stable"
    ).drop(columns="_method_order").reset_index(drop=True)


def _build_metric_products(
    delta2: pd.DataFrame,
    archive: pd.DataFrame,
    seed_predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    common_columns = [
        *KEY_COLUMNS,
        "method",
        "candidate_id",
        "latent_predicted_log_slowdown",
        "predicted_log_slowdown",
        "predicted_slowdown",
        "true_log_slowdown",
        "true_slowdown",
        "ensemble_seed_count",
        "model_seed_log_std",
    ]
    combined = pd.concat(
        [delta2[common_columns], archive[common_columns]], ignore_index=True
    )
    if set(combined["method"]) != set(ALL_METHODS):
        raise RuntimeError("combined metric frame has the wrong method roster")
    counts = combined.groupby("method").size()
    if not counts.eq(DIRECTIONAL_ROWS).all():
        raise RuntimeError("each combined method must have 110 directional predictions")
    non_self = combined[~combined["self_pair"].astype(bool)]
    if not non_self.groupby("method").size().eq(PRIMARY_DIRECTIONAL_ROWS).all():
        raise RuntimeError("each primary method must have 90 non-self directions")

    metrics_non_self = _ordered(
        metric_rows(non_self, scope="non_self_ensemble")
    )
    per_victim = _ordered(
        metric_rows(
            non_self,
            scope="non_self_per_victim",
            group_columns=("method", "candidate_id", "victim_id"),
        )
    )
    victim_counts = per_victim.groupby("method").size()
    if not victim_counts.eq(10).all() or not per_victim["n"].eq(9).all():
        raise RuntimeError("per-victim scope must contain ten rows of nine directions")
    macro = _ordered(macro_victim_metrics(per_victim))
    metrics_all = _ordered(metric_rows(combined, scope="all_directional"))
    self_rows = combined[combined["self_pair"].astype(bool)]
    metrics_self = _ordered(metric_rows(self_rows, scope="self_directional"))
    self_average = self_averaged_predictions(combined)
    metrics_self_averaged = _ordered(
        metric_rows(self_average, scope="self_raw_averaged")
    )
    if not metrics_self["n"].eq(20).all() or not metrics_self_averaged["n"].eq(10).all():
        raise RuntimeError("self scopes have incorrect observation counts")

    seed_non_self = seed_predictions[~seed_predictions["self_pair"].astype(bool)]
    metrics_by_seed = _ordered(
        metric_rows(
            seed_non_self,
            scope="non_self_per_seed",
            group_columns=("method", "candidate_id", "seed"),
        )
    )
    if len(metrics_by_seed) != len(SEEDED_METHODS) * len(SEEDS) or not metrics_by_seed[
        "n"
    ].eq(PRIMARY_DIRECTIONAL_ROWS).all():
        raise RuntimeError("per-seed metrics have incorrect coverage")
    stability = _ordered(seed_stability(metrics_by_seed))

    for frame in (
        metrics_non_self,
        per_victim,
        macro,
        metrics_all,
        metrics_self,
        metrics_self_averaged,
        metrics_by_seed,
    ):
        validate_metric_null_semantics(frame)
    return {
        "metrics_non_self.csv": metrics_non_self,
        "metrics_non_self_per_victim.csv": per_victim,
        "metrics_non_self_macro_victim.csv": macro,
        "metrics_all.csv": metrics_all,
        "metrics_self.csv": metrics_self,
        "metrics_self_averaged.csv": metrics_self_averaged,
        "metrics_by_seed.csv": metrics_by_seed,
        "seed_stability.csv": stability,
        "_combined": combined,
    }


def _historical_checks(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, object]:
    expected_log_mae = {
        "delta1_absolute_response": 0.097189,
        "delta1_constant_1": 0.103012,
        "delta1_nearest_anchor": 0.193877,
        "delta1_delta_median": 0.211887,
        "delta1_delta_single_anchor": 0.229221,
        "delta1_generic_potential": 0.237422,
        "delta1_delta_uniform": 0.239282,
        "delta1_delta_ood_kernel": 0.251164,
        "delta1_delta_kernel": 0.261504,
        "delta1_victim_inhibitor_median": 0.380145,
    }
    observed: dict[str, float] = {}
    for method, expected in expected_log_mae.items():
        rows = metrics[metrics["method"].eq(method)]
        if len(rows) != 1:
            raise RuntimeError(f"historical metric row is absent: {method}")
        value = float(rows.iloc[0]["log_mae"])
        if not np.isclose(value, expected, rtol=0.0, atol=1.1e-6):
            raise RuntimeError(
                f"historical log MAE mismatch for {method}: {value} != {expected}"
            )
        observed[method] = value
    ood = metrics[metrics["method"].eq("delta1_delta_ood_kernel")].iloc[0]
    additional = {
        "mean_true_log_slowdown": (float(ood["mean_true_log_slowdown"]), 0.103012),
        "mean_predicted_log_slowdown": (
            float(ood["mean_predicted_log_slowdown"]),
            0.340791,
        ),
        "overprediction_count": (float(ood["overprediction_count"]), 78.0),
        "spearman": (float(ood["spearman"]), 0.418745),
    }
    for name, (value, expected) in additional.items():
        if not np.isclose(value, expected, rtol=0.0, atol=1.1e-6):
            raise RuntimeError(f"historical check mismatch for {name}: {value}")
    interval = bootstrap[
        bootstrap["method"].eq("delta1_delta_ood_kernel")
        & bootstrap["metric"].eq("log_mae")
    ].iloc[0]
    if not np.allclose(
        [interval["ci_lower"], interval["ci_upper"]],
        [0.209078, 0.294224],
        rtol=0.0,
        atol=1.1e-6,
    ):
        raise RuntimeError("historical Delta Response 1 OOD interval did not reproduce")
    contrast = paired[
        paired["first_method"].eq("delta1_delta_ood_kernel")
        & paired["second_method"].eq("delta1_absolute_response")
        & paired["metric"].eq("log_mae")
    ].iloc[0]
    if not np.allclose(
        [
            contrast["estimate_difference"],
            contrast["ci_lower"],
            contrast["ci_upper"],
        ],
        [0.153975, 0.106098, 0.203071],
        rtol=0.0,
        atol=1.1e-6,
    ):
        raise RuntimeError("historical OOD-minus-absolute paired check did not reproduce")
    return {
        "status": "passed",
        "log_mae": observed,
        "additional_checks": {key: value[0] for key, value in additional.items()},
        "ood_log_mae_interval": [
            float(interval["ci_lower"]),
            float(interval["ci_upper"]),
        ],
        "ood_minus_absolute_log_mae": float(contrast["estimate_difference"]),
        "ood_minus_absolute_interval": [
            float(contrast["ci_lower"]),
            float(contrast["ci_upper"]),
        ],
    }


def _paired_summary(
    paired: pd.DataFrame, first: str, second: str, metric: str = "log_mae"
) -> dict[str, object]:
    row = paired[
        paired["first_method"].eq(first)
        & paired["second_method"].eq(second)
        & paired["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError(f"paired summary is missing: {first}/{second}/{metric}")
    value = row.iloc[0]
    return {
        "first_method": first,
        "second_method": second,
        "metric": metric,
        "estimate_difference": float(value["estimate_difference"]),
        "ci_lower": float(value["ci_lower"]),
        "ci_upper": float(value["ci_upper"]),
        "difference_definition": value["difference_definition"],
    }


def _assert_frame_matches(
    name: str, observed: pd.DataFrame, expected: pd.DataFrame
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"completed product failed semantic replay: {name}") from error


def _assert_nested_matches(name: str, observed: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"completed report binding differs: {name}")
        for key, value in expected.items():
            _assert_nested_matches(f"{name}.{key}", observed[key], value)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"completed report binding differs: {name}")
        for index, value in enumerate(expected):
            _assert_nested_matches(f"{name}[{index}]", observed[index], value)
        return
    if isinstance(expected, (float, np.floating)):
        if observed is None or not np.isclose(
            float(observed), float(expected), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"completed report binding differs: {name}")
        return
    if observed != expected:
        raise RuntimeError(f"completed report binding differs: {name}")


def _read_product_csv(root: Path, name: str) -> pd.DataFrame:
    require_regular_file(root / name)
    return pd.read_csv(root / name, low_memory=False)


def _verify_completed_products(
    root: Path,
    prepared: dict[str, Any],
    archive_hashes: dict[str, str],
    report: dict[str, Any],
) -> None:
    raw_pairs = read_csv(PAIR_PATH)
    data_ids = {
        str(value)
        for value in pd.read_csv(
            prepared["plan"]["absolute_input_paths"]["jobs.csv"]
        )["job_id"]
    }
    validated_pairs = validate_pair_frame(raw_pairs, data_ids)
    expected_manifest = pair_manifest(validated_pairs)
    directions = expand_directions(validated_pairs)
    observed_manifest = _read_product_csv(root, "evaluation_pair_manifest.csv")
    _assert_frame_matches(
        "evaluation_pair_manifest.csv", observed_manifest, expected_manifest
    )

    queries = outcome_blind_queries(directions)
    seed_blind, directional_blind = build_delta2_predictions(root, queries, prepared)
    expected_seed = _attach_outcomes(seed_blind, directions, seeded=True)
    expected_directional = _attach_outcomes(
        directional_blind, directions, seeded=False
    )
    observed_seed = _read_product_csv(root, "seed_predictions.csv")
    observed_directional = _read_product_csv(root, "directional_predictions.csv")
    _assert_frame_matches("seed_predictions.csv", observed_seed, expected_seed)
    _assert_frame_matches(
        "directional_predictions.csv", observed_directional, expected_directional
    )
    _validate_prediction_invariants(observed_seed, observed_directional, queries)
    _validate_ensemble_construction(observed_seed, observed_directional)
    oracle_checks = _external_oracle_checks(
        directions, observed_seed, observed_directional
    )

    expected_archive = load_archived_comparison(
        expected_manifest, directions, archive_hashes
    )
    observed_archive = _read_product_csv(root, "archived_delta1_comparison.csv")
    _assert_frame_matches(
        "archived_delta1_comparison.csv", observed_archive, expected_archive
    )
    baseline_parity = _validate_recomputed_baseline_parity(
        observed_directional, observed_archive
    )

    products = _build_metric_products(
        observed_directional, observed_archive, observed_seed
    )
    combined = products.pop("_combined")
    for name, expected in products.items():
        _assert_frame_matches(name, _read_product_csv(root, name), expected)

    expected_bootstrap, expected_paired, _ = bootstrap_tables(
        combined,
        ALL_METHODS,
        samples=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        expected_cluster_count=NON_SELF_CLUSTERS,
    )
    observed_bootstrap = _read_product_csv(
        root, "cluster_bootstrap_non_self.csv"
    )
    observed_paired = _read_product_csv(
        root, "paired_method_differences_non_self.csv"
    )
    _assert_frame_matches(
        "cluster_bootstrap_non_self.csv",
        observed_bootstrap,
        expected_bootstrap,
    )
    _assert_frame_matches(
        "paired_method_differences_non_self.csv",
        observed_paired,
        expected_paired,
    )

    historical_checks = _historical_checks(
        products["metrics_non_self.csv"], expected_bootstrap, expected_paired
    )
    hypotheses = hypothesis_results(
        products["metrics_non_self.csv"],
        expected_bootstrap,
        expected_paired,
        products["seed_stability.csv"],
    )
    expected_design = design_diagnostic_table(
        products["metrics_non_self.csv"], expected_paired
    )
    _assert_frame_matches(
        "design_diagnostic_comparison.csv",
        _read_product_csv(root, "design_diagnostic_comparison.csv"),
        expected_design,
    )
    expected_report_text = render_report(
        products["metrics_non_self.csv"],
        products["metrics_non_self_per_victim.csv"],
        products["seed_stability.csv"],
        expected_bootstrap,
        expected_paired,
        hypotheses,
    )
    report_path = root / "STATIC_RANDOM_SPLIT_REPORT.md"
    require_regular_file(report_path)
    if report_path.read_text(encoding="ascii") != expected_report_text:
        raise RuntimeError("completed Markdown report differs from semantic replay")
    if any(
        prohibited in expected_report_text
        for prohibited in (
            "untouched test set",
            "unbiased generalization estimate",
            "confirmatory App-App result",
        )
    ):
        raise RuntimeError("completed Markdown report contains prohibited evidence language")

    selected = products["metrics_non_self.csv"][
        products["metrics_non_self.csv"]["method"].eq("delta2_selected")
    ].iloc[0]
    expected_selected_metrics = {
        "log_mae": float(selected["log_mae"]),
        "raw_mae": float(selected["raw_mae"]),
        "signed_log_bias": float(selected["signed_log_bias"]),
        "spearman": (
            float(selected["spearman"])
            if bool(selected["spearman_defined"])
            else None
        ),
        "spearman_defined": bool(selected["spearman_defined"]),
        "calibration_intercept": (
            float(selected["calibration_intercept"])
            if bool(selected["calibration_defined"])
            else None
        ),
        "calibration_slope": (
            float(selected["calibration_slope"])
            if bool(selected["calibration_defined"])
            else None
        ),
        "calibration_defined": bool(selected["calibration_defined"]),
        "mean_true_log_slowdown": float(selected["mean_true_log_slowdown"]),
        "mean_predicted_log_slowdown": float(
            selected["mean_predicted_log_slowdown"]
        ),
        "overprediction_count": int(selected["overprediction_count"]),
        "underprediction_count": int(selected["underprediction_count"]),
    }
    expected_paired_summaries = {
        "versus_delta1_ood_delta": _paired_summary(
            expected_paired, "delta2_selected", "delta1_delta_ood_kernel"
        ),
        "versus_constant_one": _paired_summary(
            expected_paired, "delta2_selected", "delta2_constant_1"
        ),
        "versus_delta1_absolute_response": _paired_summary(
            expected_paired, "delta2_selected", "delta1_absolute_response"
        ),
    }
    _assert_nested_matches(
        "selected_non_self_metrics",
        report.get("selected_non_self_metrics"),
        expected_selected_metrics,
    )
    _assert_nested_matches(
        "primary_paired_log_mae_differences",
        report.get("primary_paired_log_mae_differences"),
        expected_paired_summaries,
    )
    _assert_nested_matches("hypotheses", report.get("hypotheses"), hypotheses)
    _assert_nested_matches(
        "historical_delta1_checks",
        report.get("historical_delta1_checks"),
        historical_checks,
    )
    _assert_nested_matches(
        "external_profile_oracle_checks",
        report.get("external_profile_oracle_checks"),
        oracle_checks,
    )
    _assert_nested_matches(
        "recomputed_baseline_parity",
        report.get("recomputed_baseline_parity"),
        baseline_parity,
    )


def _completion_payload(
    root: Path,
    prepared: dict[str, Any],
    pair_hash: str,
    archive_hashes: dict[str, str],
    prediction_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "preparation_seal_sha256": prepared["preparation_seal_sha256"],
        "pair_sha256": pair_hash,
        "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "archived_delta_response_1_hashes": archive_hashes,
        "evaluation_source_hashes": prepared["source_hashes"],
        "diagnostic_artifact_hashes": prepared["artifact_hashes"],
        "new_prediction_artifact_hashes": prediction_hashes,
        "output_hashes_sha256": sha256_file(root / "output_hashes.json"),
        "exact_output_roster": sorted(completed_roster()),
        "evidence_label": EVIDENCE_LABEL,
    }


def _seal_completion(
    root: Path,
    prepared: dict[str, Any],
    pair_hash: str,
    archive_hashes: dict[str, str],
    prediction_hashes: dict[str, str],
) -> None:
    output_names = completed_roster() - set(COMPLETION_METADATA_FILES)
    actual = regular_file_roster(root)
    if actual != output_names:
        raise RuntimeError(
            "pre-completion output roster mismatch: "
            f"missing={sorted(output_names - actual)}, unexpected={sorted(actual - output_names)}"
        )
    write_json(root / "output_hashes.json", hash_roster(root, output_names))
    payload = _completion_payload(
        root, prepared, pair_hash, archive_hashes, prediction_hashes
    )
    write_json(
        root / "completion_seal.json",
        {**payload, "sha256": canonical_digest(payload)},
    )
    verify_complete(root)


def evaluate(root: Path = DEFAULT_OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, object]:
    require_cpu_only(device)
    # This is the complete pair-blind gate. No operation above or inside this
    # call touches the pair path.
    prepared = verify_preparation(root, require_prepared_roster=True)

    write_json(
        root / "run_report.json",
        {
            "status": "evaluation_started",
            "evidence_label": EVIDENCE_LABEL,
            "app_app_outcome_access_started": True,
            "pair_access_status": "may_have_been_opened_after_this_marker",
            "preparation_seal_sha256": prepared["preparation_seal_sha256"],
        },
    )

    require_regular_file(PAIR_PATH)
    pair_hash = sha256_file(PAIR_PATH)
    if pair_hash != EXPECTED_INPUT_HASHES["pair.csv"]:
        raise RuntimeError("historical static pair hash does not match the fixed contract")
    raw_pairs = read_csv(PAIR_PATH)
    data_ids = {
        str(value)
        for value in pd.read_csv(
            prepared["plan"]["absolute_input_paths"]["jobs.csv"]
        )["job_id"]
    }
    validated_pairs = validate_pair_frame(raw_pairs, data_ids)
    manifest = pair_manifest(validated_pairs)
    directions = expand_directions(validated_pairs)
    write_csv(root / "evaluation_pair_manifest.csv", manifest)

    queries = outcome_blind_queries(directions)
    seed_blind, directional_blind = build_delta2_predictions(
        root, queries, prepared
    )
    seed_predictions = _attach_outcomes(seed_blind, directions, seeded=True)
    directional_predictions = _attach_outcomes(
        directional_blind, directions, seeded=False
    )
    _validate_ensemble_construction(seed_predictions, directional_predictions)
    write_csv(root / "seed_predictions.csv", seed_predictions)
    write_csv(root / "directional_predictions.csv", directional_predictions)

    prediction_hashes = hash_roster(root, PREDICTION_ARTIFACTS)
    oracle_checks = _external_oracle_checks(
        directions, seed_predictions, directional_predictions
    )
    if hash_roster(root, PREDICTION_ARTIFACTS) != prediction_hashes:
        raise RuntimeError("new prediction artifacts changed after freezing")

    # Archive access begins only here, after all new prediction artifacts have
    # stable hashes and ten profile-only oracle checks have passed.
    archive_hashes = verify_archive_hashes()
    archive = load_archived_comparison(
        manifest, directions, archive_hashes
    )
    write_csv(root / "archived_delta1_comparison.csv", archive)
    baseline_parity = _validate_recomputed_baseline_parity(
        directional_predictions, archive
    )

    products = _build_metric_products(
        directional_predictions, archive, seed_predictions
    )
    combined = products.pop("_combined")
    bootstrap, paired, _ = bootstrap_tables(
        combined,
        ALL_METHODS,
        samples=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        expected_cluster_count=NON_SELF_CLUSTERS,
    )
    historical_checks = _historical_checks(
        products["metrics_non_self.csv"], bootstrap, paired
    )
    hypotheses = hypothesis_results(
        products["metrics_non_self.csv"],
        bootstrap,
        paired,
        products["seed_stability.csv"],
    )
    design = design_diagnostic_table(
        products["metrics_non_self.csv"], paired
    )

    for name, frame in products.items():
        write_csv(root / name, frame)
    write_csv(root / "cluster_bootstrap_non_self.csv", bootstrap)
    write_csv(root / "paired_method_differences_non_self.csv", paired)
    write_csv(root / "design_diagnostic_comparison.csv", design)

    report_text = render_report(
        products["metrics_non_self.csv"],
        products["metrics_non_self_per_victim.csv"],
        products["seed_stability.csv"],
        bootstrap,
        paired,
        hypotheses,
    )
    (root / "STATIC_RANDOM_SPLIT_REPORT.md").write_text(
        report_text, encoding="ascii"
    )
    selected = products["metrics_non_self.csv"][
        products["metrics_non_self.csv"]["method"].eq("delta2_selected")
    ].iloc[0]
    run_report = {
        "status": "complete",
        "evidence_label": EVIDENCE_LABEL,
        "evidence_status": EVIDENCE_STATUS,
        "future_evidence_requirement": FUTURE_EVIDENCE_REQUIREMENT,
        "evaluation_method": "random_split",
        "pair_sha256": pair_hash,
        "pair_rows": 55,
        "self_pair_rows": 10,
        "non_self_pair_clusters": 45,
        "directional_rows": 110,
        "primary_directional_rows": 90,
        "valid_app_inhibitor_rows": 2623,
        "seeds": list(SEEDS),
        "seed_ensemble_rule": "mean_latent_across_seeds_then_clip_once_at_zero",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "pair_cluster_id",
            "cluster_count": NON_SELF_CLUSTERS,
            "interval": "percentile_2.5_97.5",
        },
        "selected_candidate_id": METHOD_CANDIDATES["delta2_selected"],
        "selected_non_self_metrics": {
            "log_mae": float(selected["log_mae"]),
            "raw_mae": float(selected["raw_mae"]),
            "signed_log_bias": float(selected["signed_log_bias"]),
            "spearman": (
                float(selected["spearman"])
                if bool(selected["spearman_defined"])
                else None
            ),
            "spearman_defined": bool(selected["spearman_defined"]),
            "calibration_intercept": (
                float(selected["calibration_intercept"])
                if bool(selected["calibration_defined"])
                else None
            ),
            "calibration_slope": (
                float(selected["calibration_slope"])
                if bool(selected["calibration_defined"])
                else None
            ),
            "calibration_defined": bool(selected["calibration_defined"]),
            "mean_true_log_slowdown": float(selected["mean_true_log_slowdown"]),
            "mean_predicted_log_slowdown": float(
                selected["mean_predicted_log_slowdown"]
            ),
            "overprediction_count": int(selected["overprediction_count"]),
            "underprediction_count": int(selected["underprediction_count"]),
        },
        "primary_paired_log_mae_differences": {
            "versus_delta1_ood_delta": _paired_summary(
                paired, "delta2_selected", "delta1_delta_ood_kernel"
            ),
            "versus_constant_one": _paired_summary(
                paired, "delta2_selected", "delta2_constant_1"
            ),
            "versus_delta1_absolute_response": _paired_summary(
                paired, "delta2_selected", "delta1_absolute_response"
            ),
        },
        "hypotheses": hypotheses,
        "external_profile_oracle_checks": oracle_checks,
        "historical_delta1_checks": historical_checks,
        "recomputed_baseline_parity": baseline_parity,
        "new_prediction_artifact_hashes": prediction_hashes,
        "archived_delta_response_1_hashes": archive_hashes,
        "preparation_seal_sha256": prepared["preparation_seal_sha256"],
        "app_app_outcomes_used_for_prediction_or_fitting": False,
        "app_app_outcomes_used_for_metrics_only": True,
    }
    write_json(root / "run_report.json", run_report)
    if hash_roster(root, PREDICTION_ARTIFACTS) != prediction_hashes:
        raise RuntimeError("new prediction artifacts changed during analysis")
    _seal_completion(
        root,
        prepared,
        pair_hash,
        archive_hashes,
        prediction_hashes,
    )
    return run_report


def verify_complete(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    prepared = verify_preparation(root, require_prepared_roster=False)
    require_regular_file(PAIR_PATH)
    pair_hash = sha256_file(PAIR_PATH)
    if pair_hash != EXPECTED_INPUT_HASHES["pair.csv"]:
        raise RuntimeError("complete evaluation pair input changed")
    archive_hashes = verify_archive_hashes()
    output_hashes = read_canonical_json(root / "output_hashes.json")
    seal = read_canonical_json(root / "completion_seal.json")

    expected_hashed = completed_roster() - set(COMPLETION_METADATA_FILES)
    if set(output_hashes) != expected_hashed:
        raise RuntimeError("completion output-hash roster is not exact")
    current_hashes = hash_roster(root, expected_hashed)
    if current_hashes != output_hashes:
        raise RuntimeError("one or more completed evaluation outputs changed")
    verify_exact_roster(root, completed_roster())
    prediction_hashes = {
        name: output_hashes[name] for name in PREDICTION_ARTIFACTS
    }
    payload = _completion_payload(
        root, prepared, pair_hash, archive_hashes, prediction_hashes
    )
    if {key: seal.get(key) for key in payload} != payload:
        raise RuntimeError("completion seal binding is invalid")
    if seal.get("sha256") != canonical_digest(payload):
        raise RuntimeError("completion seal digest is invalid")
    report = read_canonical_json(root / "run_report.json")
    if report.get("status") != "complete" or report.get("pair_sha256") != pair_hash:
        raise RuntimeError("completed run report is invalid")
    if report.get("new_prediction_artifact_hashes") != prediction_hashes:
        raise RuntimeError("run report prediction hashes differ from completion seal")
    _verify_completed_products(root, prepared, archive_hashes, report)
    return report
