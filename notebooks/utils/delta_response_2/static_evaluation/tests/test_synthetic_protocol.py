from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from delta_response_2.static_evaluation.bootstrap import (
    bootstrap_tables,
    make_cluster_draws,
)
from delta_response_2.static_evaluation.archive import load_archived_comparison
from delta_response_2.static_evaluation.constants import (
    DELTA2_METHODS,
    DIAGNOSTIC_ARTIFACT_FILES,
    DIAGNOSTIC_FILES,
    EVALUATION_PRODUCT_FILES,
    EXPECTED_ARCHIVE_HASHES,
    METHOD_CANDIDATES,
    NAMESPACED_ARCHIVE_METHODS,
    SEEDS,
    SEEDED_METHODS,
    completed_roster,
    prepared_roster,
)
from delta_response_2.static_evaluation.evaluation import (
    ALL_METHODS,
    _build_metric_products,
    _validate_ensemble_construction,
)
from delta_response_2.static_evaluation.integrity import (
    read_canonical_json,
    write_json,
)
from delta_response_2.static_evaluation.metrics import (
    compute_metrics,
    self_averaged_predictions,
)
from delta_response_2.static_evaluation.pairs import (
    expand_directions,
    pair_manifest,
    validate_pair_frame,
)
from delta_response_2.static_evaluation.reporting import (
    design_diagnostic_table,
    hypothesis_results,
    render_report,
)
from delta_response_2.static_evaluation.synthetic import run_synthetic_self_checks


def fixture_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jobA_id": ["a", "a", "a", "b", "b", "c"],
            "jobB_id": ["a", "b", "c", "b", "c", "c"],
            "slowdown_A": [1.0, 2.0, 1.5, 1.1, 1.4, 1.2],
            "slowdown_B": [1.0, 1.8, 1.3, 1.2, 1.6, 1.1],
        }
    )


class PairProtocolTests(unittest.TestCase):
    def test_parser_manifest_and_direction_expansion(self) -> None:
        pairs = validate_pair_frame(
            fixture_pairs(),
            {"a", "b", "c"},
            expected_rows=6,
            expected_self_rows=3,
        )
        self.assertEqual(pairs["pair_row_id"].tolist(), list(range(6)))
        self.assertTrue(pairs["pair_cluster_id"].equals(pairs["pair_row_id"]))
        manifest = pair_manifest(pairs)
        self.assertTrue(manifest["included"].all())
        self.assertTrue(manifest["jobA_known"].all())
        directions = expand_directions(pairs)
        self.assertEqual(len(directions), 12)
        self.assertEqual(int(directions["self_pair"].sum()), 6)
        for pair_id, group in directions.groupby("pair_row_id"):
            self.assertEqual(group["direction"].tolist(), ["A", "B"])
            self.assertEqual(group["pair_cluster_id"].tolist(), [pair_id, pair_id])

    def test_reversed_duplicate_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "jobA_id": ["a", "b"],
                "jobB_id": ["b", "a"],
                "slowdown_A": [1.0, 1.1],
                "slowdown_B": [1.2, 1.3],
            }
        )
        with self.assertRaisesRegex(ValueError, "reversed duplicate"):
            validate_pair_frame(
                frame,
                {"a", "b"},
                expected_rows=2,
                expected_self_rows=0,
            )

    def test_invalid_slowdown_and_unknown_id_are_rejected(self) -> None:
        bad = fixture_pairs()
        bad.loc[0, "slowdown_A"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite and at least one"):
            validate_pair_frame(
                bad,
                {"a", "b", "c"},
                expected_rows=6,
                expected_self_rows=3,
            )
        unknown = fixture_pairs()
        unknown.loc[0, "jobA_id"] = "other"
        with self.assertRaisesRegex(ValueError, "unknown application"):
            validate_pair_frame(
                unknown,
                {"a", "b", "c"},
                expected_rows=6,
                expected_self_rows=3,
            )


class MetricTests(unittest.TestCase):
    def test_exact_metric_definitions(self) -> None:
        frame = pd.DataFrame(
            {
                "true_log_slowdown": [0.0, np.log(2.0), np.log(4.0)],
                "predicted_log_slowdown": [0.0, np.log(4.0), np.log(2.0)],
                "true_slowdown": [1.0, 2.0, 4.0],
                "predicted_slowdown": [1.0, 4.0, 2.0],
            }
        )
        result = compute_metrics(frame)
        self.assertAlmostEqual(result["log_mae"], 2.0 * np.log(2.0) / 3.0)
        self.assertAlmostEqual(result["raw_mae"], 4.0 / 3.0)
        self.assertAlmostEqual(result["raw_mape"], 50.0)
        self.assertEqual(result["overprediction_count"], 1)
        self.assertEqual(result["underprediction_count"], 1)
        self.assertTrue(result["spearman_defined"])
        self.assertTrue(result["pearson_defined"])
        self.assertTrue(result["calibration_defined"])

    def test_undefined_metrics_are_null_with_false_flags(self) -> None:
        frame = pd.DataFrame(
            {
                "true_log_slowdown": [0.0, np.log(2.0), np.log(3.0)],
                "predicted_log_slowdown": [0.0, 0.0, 0.0],
                "true_slowdown": [1.0, 2.0, 3.0],
                "predicted_slowdown": [1.0, 1.0, 1.0],
            }
        )
        result = compute_metrics(frame)
        for estimate, flag in (
            ("spearman", "spearman_defined"),
            ("pearson", "pearson_defined"),
            ("calibration_intercept", "calibration_defined"),
            ("calibration_slope", "calibration_defined"),
        ):
            self.assertIsNone(result[estimate])
            self.assertFalse(result[flag])

    def test_near_constant_range_is_undefined_before_ranking(self) -> None:
        predicted_log = np.asarray([0.1, 0.1 + 5e-13, 0.1 + 9e-13])
        frame = pd.DataFrame(
            {
                "true_log_slowdown": [0.0, 0.2, 0.4],
                "predicted_log_slowdown": predicted_log,
                "true_slowdown": np.exp([0.0, 0.2, 0.4]),
                "predicted_slowdown": np.exp(predicted_log),
            }
        )
        result = compute_metrics(frame)
        self.assertFalse(result["spearman_defined"])
        self.assertIsNone(result["spearman"])
        self.assertFalse(result["pearson_defined"])
        self.assertFalse(result["calibration_defined"])

    def test_self_pair_truth_is_averaged_in_raw_space(self) -> None:
        frame = pd.DataFrame(
            {
                "method": ["m", "m"],
                "candidate_id": ["c", "c"],
                "pair_cluster_id": [4, 4],
                "self_pair": [True, True],
                "victim_id": ["a", "a"],
                "aggressor_id": ["a", "a"],
                "latent_predicted_log_slowdown": [0.2, 0.2],
                "predicted_log_slowdown": [0.2, 0.2],
                "predicted_slowdown": [np.exp(0.2), np.exp(0.2)],
                "true_slowdown": [1.0, 3.0],
                "true_log_slowdown": [0.0, np.log(3.0)],
            }
        )
        averaged = self_averaged_predictions(frame)
        self.assertEqual(len(averaged), 1)
        self.assertEqual(averaged.iloc[0]["true_slowdown"], 2.0)
        self.assertAlmostEqual(averaged.iloc[0]["true_log_slowdown"], np.log(2.0))


class BootstrapTests(unittest.TestCase):
    def _predictions(self) -> pd.DataFrame:
        pairs = validate_pair_frame(
            fixture_pairs(),
            {"a", "b", "c"},
            expected_rows=6,
            expected_self_rows=3,
        )
        directions = expand_directions(pairs)
        directions = directions[~directions["self_pair"]].copy()
        rows = []
        for method, offset in (("better", 0.0), ("worse", 0.2)):
            frame = directions.copy()
            frame["method"] = method
            frame["candidate_id"] = "synthetic"
            frame["predicted_log_slowdown"] = (
                frame["true_log_slowdown"].to_numpy(dtype=float) + offset
            )
            frame["predicted_slowdown"] = np.exp(frame["predicted_log_slowdown"])
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    def test_draw_matrix_is_reproducible(self) -> None:
        clusters_a, draws_a = make_cluster_draws([1, 2, 4], samples=20, seed=923)
        clusters_b, draws_b = make_cluster_draws([1, 2, 4], samples=20, seed=923)
        np.testing.assert_array_equal(clusters_a, clusters_b)
        np.testing.assert_array_equal(draws_a, draws_b)
        self.assertEqual(draws_a.shape, (20, 3))

    def test_paired_cluster_interval_uses_both_directions(self) -> None:
        intervals, paired, replicates = bootstrap_tables(
            self._predictions(),
            ["better", "worse"],
            samples=50,
            seed=923,
            expected_cluster_count=3,
        )
        self.assertEqual(len(intervals), 32)
        contrast = paired[
            paired["first_method"].eq("better")
            & paired["second_method"].eq("worse")
            & paired["metric"].eq("log_mae")
        ].iloc[0]
        self.assertLess(contrast["estimate_difference"], 0)
        self.assertLess(contrast["ci_upper"], 0)
        self.assertEqual(contrast["cluster_unit"], "pair_cluster_id")
        self.assertEqual(replicates["better"]["log_mae"].shape, (50,))


class IntegrityTests(unittest.TestCase):
    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            path.write_text('{"value": NaN}\n', encoding="ascii")
            with self.assertRaises(ValueError):
                read_canonical_json(path)

    def test_canonical_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            value = {"b": [2, 1], "a": False}
            write_json(path, value)
            self.assertEqual(read_canonical_json(path), value)
            self.assertEqual(json.loads(path.read_text(encoding="ascii")), value)

    def test_canonical_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text('{"value": 1, "value": 2}\n', encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_canonical_json(path)

    def test_all_in_memory_preparation_checks_pass(self) -> None:
        result = run_synthetic_self_checks()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["uses_filesystem"])

    def test_exact_protocol_rosters_are_fixed(self) -> None:
        self.assertEqual(len(DIAGNOSTIC_ARTIFACT_FILES), 16)
        self.assertEqual(len(DIAGNOSTIC_FILES), 18)
        self.assertEqual(len(DELTA2_METHODS), 9)
        self.assertEqual(len(SEEDED_METHODS), 5)
        self.assertEqual(len(NAMESPACED_ARCHIVE_METHODS), 10)
        self.assertEqual(
            METHOD_CANDIDATES["delta2_selected"],
            "current_delta_ood_h8_e4__de045c14efb9",
        )
        self.assertEqual(
            prepared_roster(),
            completed_roster()
            - set(EVALUATION_PRODUCT_FILES)
            - {"output_hashes.json", "completion_seal.json"},
        )


class SyntheticHarnessIntegrationTests(unittest.TestCase):
    @staticmethod
    def _directions() -> pd.DataFrame:
        identifiers = [f"app{index}" for index in range(10)]
        rows = []
        for left_index, left in enumerate(identifiers):
            for right_index in range(left_index, len(identifiers)):
                right = identifiers[right_index]
                rows.append(
                    {
                        "jobA_id": left,
                        "jobB_id": right,
                        "slowdown_A": 1.0 + 0.01 * (left_index + right_index),
                        "slowdown_B": 1.0 + 0.015 * (left_index + right_index),
                    }
                )
        pairs = validate_pair_frame(pd.DataFrame(rows), identifiers)
        return expand_directions(pairs)

    @classmethod
    def _prediction_tables(
        cls,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        directions = cls._directions()
        directional_rows = []
        for method_index, method in enumerate(ALL_METHODS):
            candidate_id = METHOD_CANDIDATES.get(
                method, "archived_delta_response_1"
            )
            for row in directions.itertuples(index=False):
                true_log = float(row.true_log_slowdown)
                victim_index = int(str(row.victim_id).removeprefix("app"))
                aggressor_index = int(str(row.aggressor_id).removeprefix("app"))
                predicted_log = max(
                    0.0,
                    0.01 * (victim_index + aggressor_index)
                    + 0.002 * (method_index - 5),
                )
                directional_rows.append(
                    {
                        "evaluation_method": "random_split",
                        "pair_row_id": int(row.pair_row_id),
                        "pair_cluster_id": int(row.pair_cluster_id),
                        "direction": str(row.direction),
                        "victim_id": str(row.victim_id),
                        "aggressor_id": str(row.aggressor_id),
                        "self_pair": bool(row.self_pair),
                        "method": method,
                        "candidate_id": candidate_id,
                        "latent_predicted_log_slowdown": predicted_log,
                        "predicted_log_slowdown": predicted_log,
                        "predicted_slowdown": float(np.exp(predicted_log)),
                        "true_log_slowdown": true_log,
                        "true_slowdown": float(row.true_slowdown),
                        "ensemble_seed_count": 5 if method in SEEDED_METHODS else 1,
                        "model_seed_log_std": 0.0,
                    }
                )
        directional = pd.DataFrame(directional_rows)

        seed_rows = []
        for method_index, method in enumerate(SEEDED_METHODS):
            for seed in SEEDS:
                shift = 0.001 * (seed - 2)
                for row in directions.itertuples(index=False):
                    victim_index = int(str(row.victim_id).removeprefix("app"))
                    aggressor_index = int(str(row.aggressor_id).removeprefix("app"))
                    latent = max(
                        0.0,
                        0.01 * (victim_index + aggressor_index)
                        + 0.002 * method_index
                        + shift,
                    )
                    observed = max(0.0, latent)
                    seed_rows.append(
                        {
                            "evaluation_method": "random_split",
                            "pair_row_id": int(row.pair_row_id),
                            "pair_cluster_id": int(row.pair_cluster_id),
                            "direction": str(row.direction),
                            "victim_id": str(row.victim_id),
                            "aggressor_id": str(row.aggressor_id),
                            "self_pair": bool(row.self_pair),
                            "method": method,
                            "candidate_id": METHOD_CANDIDATES[method],
                            "seed": seed,
                            "latent_predicted_log_slowdown": latent,
                            "predicted_log_slowdown": observed,
                            "predicted_slowdown": float(np.exp(observed)),
                            "true_log_slowdown": float(row.true_log_slowdown),
                            "true_slowdown": float(row.true_slowdown),
                        }
                    )
        return directions, directional, pd.DataFrame(seed_rows)

    def test_all_metric_scopes_have_exact_synthetic_coverage(self) -> None:
        _, directional, seeds = self._prediction_tables()
        delta2 = directional[directional["method"].isin(DELTA2_METHODS)]
        archive = directional[
            directional["method"].isin(NAMESPACED_ARCHIVE_METHODS)
        ]
        products = _build_metric_products(delta2, archive, seeds)
        self.assertEqual(len(products["metrics_non_self.csv"]), 19)
        self.assertTrue(products["metrics_non_self.csv"]["n"].eq(90).all())
        self.assertEqual(len(products["metrics_non_self_per_victim.csv"]), 190)
        self.assertTrue(
            products["metrics_non_self_per_victim.csv"]["n"].eq(9).all()
        )
        self.assertEqual(len(products["metrics_non_self_macro_victim.csv"]), 19)
        self.assertTrue(products["metrics_self.csv"]["n"].eq(20).all())
        self.assertTrue(products["metrics_self_averaged.csv"]["n"].eq(10).all())
        self.assertEqual(len(products["metrics_by_seed.csv"]), 25)
        self.assertTrue(products["metrics_by_seed.csv"]["n"].eq(90).all())

        intervals, paired, _ = bootstrap_tables(
            products["_combined"],
            ALL_METHODS,
            samples=5,
            seed=923,
            expected_cluster_count=45,
        )
        self.assertEqual(len(intervals), 19 * 16)
        self.assertEqual(len(paired), 19 * 18 * 16)
        self.assertTrue(intervals["cluster_count"].eq(45).all())
        hypotheses = hypothesis_results(
            products["metrics_non_self.csv"],
            intervals,
            paired,
            products["seed_stability.csv"],
        )
        self.assertEqual(set(hypotheses), {"H1", "H2", "H3", "H4", "H5"})
        design = design_diagnostic_table(
            products["metrics_non_self.csv"], paired
        )
        self.assertEqual(len(design), 9)
        report = render_report(
            products["metrics_non_self.csv"],
            products["metrics_non_self_per_victim.csv"],
            products["seed_stability.csv"],
            intervals,
            paired,
            hypotheses,
        )
        self.assertIn("retrospective static random_split evaluation", report)
        self.assertIn("## H1-H5 Decisions", report)

    def test_latent_seed_ensemble_invariant(self) -> None:
        directions, _, seeds = self._prediction_tables()
        ensemble_rows = []
        for keys, group in seeds.groupby(
            [
                "evaluation_method",
                "pair_row_id",
                "pair_cluster_id",
                "direction",
                "victim_id",
                "aggressor_id",
                "self_pair",
                "method",
                "candidate_id",
            ],
            sort=False,
        ):
            latent_values = group["latent_predicted_log_slowdown"].to_numpy(dtype=float)
            latent = float(np.mean(latent_values))
            ensemble_rows.append(
                {
                    **dict(
                        zip(
                            [
                                "evaluation_method",
                                "pair_row_id",
                                "pair_cluster_id",
                                "direction",
                                "victim_id",
                                "aggressor_id",
                                "self_pair",
                                "method",
                                "candidate_id",
                            ],
                            keys,
                        )
                    ),
                    "latent_predicted_log_slowdown": latent,
                    "predicted_log_slowdown": max(0.0, latent),
                    "model_seed_log_std": float(np.std(latent_values, ddof=0)),
                }
            )
        _validate_ensemble_construction(seeds, pd.DataFrame(ensemble_rows))

    def test_historical_archive_schema_needs_no_derived_truth_columns(self) -> None:
        directions = self._directions()
        manifest_pairs = pd.DataFrame(
            {
                "jobA_id": directions[directions["direction"].eq("A")]["victim_id"].to_numpy(),
                "jobB_id": directions[directions["direction"].eq("A")]["aggressor_id"].to_numpy(),
                "slowdown_A": directions[directions["direction"].eq("A")]["true_slowdown"].to_numpy(),
                "slowdown_B": directions[directions["direction"].eq("B")]["true_slowdown"].to_numpy(),
            }
        )
        validated = validate_pair_frame(
            manifest_pairs,
            {f"app{index}" for index in range(10)},
        )
        manifest = pair_manifest(validated)

        rows = []
        for method in (
            "constant_1",
            "victim_inhibitor_median",
            "nearest_anchor",
            "absolute_response",
            "delta_single_anchor",
            "delta_uniform",
            "delta_median",
            "delta_kernel",
            "delta_ood_kernel",
            "generic_potential",
        ):
            for row in directions.itertuples(index=False):
                predicted_log = 0.5 * float(row.true_log_slowdown)
                rows.append(
                    {
                        "evaluation_method": "random_split",
                        "anchor_budget": "all",
                        "method": method,
                        "pair_row_id": int(row.pair_row_id),
                        "pair_cluster_id": int(row.pair_cluster_id),
                        "direction": str(row.direction),
                        "victim_id": str(row.victim_id),
                        "aggressor_id": str(row.aggressor_id),
                        "true_slowdown": float(row.true_slowdown),
                        "latent_predicted_log_slowdown": predicted_log,
                        "predicted_log_slowdown": predicted_log,
                        "predicted_slowdown": float(np.exp(predicted_log)),
                        "model_seed_log_std": 0.01,
                    }
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest.loc[:, [
                "evaluation_method",
                "pair_row_id",
                "jobA_id",
                "jobB_id",
            ]].to_csv(root / "evaluation_pair_manifest.csv", index=False)
            pd.DataFrame(rows).to_csv(root / "directional_predictions.csv", index=False)
            with patch(
                "delta_response_2.static_evaluation.archive.ARCHIVE_ROOT", root
            ):
                result = load_archived_comparison(
                    manifest,
                    directions,
                    EXPECTED_ARCHIVE_HASHES,
                )
        self.assertEqual(len(result), 1100)
        self.assertEqual(int(result["self_pair"].sum()), 200)
        self.assertTrue(result["ensemble_seed_count"].eq(5).all())


if __name__ == "__main__":
    unittest.main()
