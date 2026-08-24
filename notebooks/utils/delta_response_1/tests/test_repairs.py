from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = MODULE_DIR / "audit_outputs"
sys.path.insert(0, str(MODULE_DIR))

import inference
import metrics
import pipeline
from data import (
    ProfileScaler,
    TrainingData,
    load_pair_holdout,
    log_slowdown_to_slowdown,
    observe_log_slowdown,
    select_evaluation_pairs,
    subset_training_data,
)
from model import GenericPotential
from training import EpisodeSampler, censored_absolute_huber_loss, censored_delta_huber_loss


def profiles(id_column: str, identifiers: list[str], offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for index, identifier in enumerate(identifiers):
        value = index + 1.0 + offset
        rows.append([identifier, value, min(value / 10.0, 1.0), value * 10, value * 100])
    return pd.DataFrame(
        rows,
        columns=[id_column, "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
    )


def crossed_data() -> tuple[TrainingData, pd.DataFrame]:
    jobs = profiles("job_id", ["v0", "v1", "v2"])
    inhibitors = profiles("inhib_id", ["i0", "i1", "i2", "i3"], 0.25)
    rows = []
    for victim_index, victim in enumerate(jobs["job_id"]):
        for inhibitor_index, inhibitor in enumerate(inhibitors["inhib_id"]):
            slowdown = 1.0 + 0.05 * victim_index + 0.02 * inhibitor_index
            rows.append(
                [
                    victim,
                    inhibitor,
                    slowdown,
                    np.log(slowdown),
                    slowdown == 1.0,
                    0,
                ]
            )
    responses = pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "inhib_id",
            "slowdown",
            "log_slowdown",
            "is_censored",
            "replicate_id",
        ],
    )
    blocks = pd.DataFrame({"inhib_id": inhibitors["inhib_id"], "block": [0, 0, 1, 1]})
    return TrainingData(jobs, inhibitors, responses, pd.DataFrame(), pd.DataFrame()), blocks


class CensoringAndBoundaryTests(unittest.TestCase):
    def test_censored_losses_apply_exact_and_one_sided_constraints(self) -> None:
        target_logs = torch.tensor(
            [[0.2, 0.5], [0.4, 0.0], [0.0, 0.3], [0.0, 0.0]]
        )
        satisfying = torch.tensor([0.3, -0.5, 0.4, 9.0])
        violating = torch.tensor([0.0, 0.2, -0.2, -9.0])
        self.assertEqual(float(censored_delta_huber_loss(satisfying, target_logs)), 0.0)
        self.assertGreater(float(censored_delta_huber_loss(violating, target_logs)), 0.0)
        exact = censored_absolute_huber_loss(
            torch.tensor([0.2, -2.0]), torch.tensor([0.2, 0.0])
        )
        violation = censored_absolute_huber_loss(
            torch.tensor([0.0, 0.5]), torch.tensor([0.2, 0.0])
        )
        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(violation), 0.0)

    def test_observation_mapping_and_exp_guards(self) -> None:
        observed = observe_log_slowdown(np.asarray([-100.0, 0.0, 0.5]))
        np.testing.assert_allclose(observed, [0.0, 0.0, 0.5])
        slowdown = log_slowdown_to_slowdown(observed)
        self.assertTrue(np.isfinite(slowdown).all())
        self.assertTrue((slowdown >= 1.0).all())
        with self.assertRaises(ValueError):
            observe_log_slowdown(np.asarray([np.nan]))
        with self.assertRaises(ValueError):
            log_slowdown_to_slowdown(np.asarray([-0.1]))
        with self.assertRaises(OverflowError):
            log_slowdown_to_slowdown(np.asarray([1000.0]))

    def test_pair_loader_rejects_reversed_duplicates(self) -> None:
        jobs = profiles("job_id", ["a", "b"])
        pairs = pd.DataFrame(
            [
                ["a", "b", 1.1, 1.2],
                ["b", "a", 1.3, 1.4],
            ],
            columns=["jobA_id", "jobB_id", "slowdown_A", "slowdown_B"],
        )
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            path = Path(directory) / "pairs.csv"
            pairs.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate unordered pairs"):
                load_pair_holdout(path, jobs)


class EvaluationProtocolTests(unittest.TestCase):
    def test_endpoint_tiers_are_canonical_and_disjoint(self) -> None:
        rows = []
        pair_id = 0
        for left_index, left in enumerate(["a", "b", "c", "d"]):
            for right in ["a", "b", "c", "d"][left_index:]:
                rows.append([pair_id, pair_id, left, right, 1.1, 1.2])
                pair_id += 1
        pairs = pd.DataFrame(
            rows,
            columns=[
                "pair_row_id",
                "pair_cluster_id",
                "jobA_id",
                "jobB_id",
                "slowdown_A",
                "slowdown_B",
            ],
        )
        selected = {}
        expected_counts = {
            "random_split": {2},
            "one_known": {1},
            "zero_shot": {0},
        }
        for method in expected_counts:
            frame, manifest = select_evaluation_pairs(pairs, {"a", "b"}, method)
            selected[method] = set(frame["pair_row_id"])
            included = manifest[manifest["included"]]
            self.assertEqual(
                set(included["known_endpoint_count"]), expected_counts[method]
            )
            expanded = pipeline.expand_directional_pairs(frame)
            self.assertTrue(
                (expanded.groupby("pair_row_id")["direction"].nunique() == 2).all()
            )
        self.assertFalse(selected["random_split"] & selected["one_known"])
        self.assertFalse(selected["random_split"] & selected["zero_shot"])
        self.assertFalse(selected["one_known"] & selected["zero_shot"])

    def test_training_subset_excludes_unknown_profiles_and_responses(self) -> None:
        data, _ = crossed_data()
        subset = subset_training_data(data, ["v0", "v2"])
        self.assertEqual(set(subset.jobs["job_id"]), {"v0", "v2"})
        self.assertEqual(set(subset.responses["job_id"]), {"v0", "v2"})
        self.assertEqual(set(subset.inhibitors["inhib_id"]), {"i0", "i1", "i2", "i3"})

    def test_anchor_budget_metrics_are_not_pooled(self) -> None:
        predictions = pd.DataFrame(
            {
                "evaluation_method": ["zero_shot"] * 4,
                "anchor_budget": ["4", "4", "all", "all"],
                "method": ["model"] * 4,
                "victim_id": ["a"] * 4,
                "aggressor_id": ["b"] * 4,
                "pair_row_id": [0, 1, 0, 1],
                "true_slowdown": [2.0] * 4,
                "predicted_slowdown": [2.0, 2.0, 1.0, 1.0],
            }
        )
        result = metrics.metrics_by_method(predictions)
        self.assertEqual(len(result), 2)
        by_budget = result.set_index("anchor_budget")["log_mae"]
        self.assertAlmostEqual(by_budget["4"], 0.0)
        self.assertGreater(by_budget["all"], 0.0)
        all_metrics = result.set_index("anchor_budget").loc["all"]
        self.assertAlmostEqual(all_metrics["raw_mae"], 1.0)
        self.assertAlmostEqual(all_metrics["raw_mape"], 50.0)
        self.assertAlmostEqual(all_metrics["raw_mse"], 1.0)
        self.assertAlmostEqual(all_metrics["raw_rmse"], 1.0)
        self.assertAlmostEqual(
            all_metrics["log_mse"], all_metrics["log_rmse"] ** 2
        )

    def test_random_split_resolves_every_application_for_training(self) -> None:
        data, _ = crossed_data()
        config = pipeline.RunConfig(evaluation_method="random_split")
        self.assertEqual(
            pipeline.resolve_training_apps(data, config), ["v0", "v1", "v2"]
        )
        config.training_apps = ["v0", "v1"]
        with self.assertRaisesRegex(ValueError, "every application"):
            pipeline.resolve_training_apps(data, config)

    def test_restricted_modes_fit_known_apps_and_infer_with_full_data(self) -> None:
        data, blocks = crossed_data()
        pair_rows = []
        pair_id = 0
        job_ids = data.jobs["job_id"].tolist()
        for left_index, left in enumerate(job_ids):
            for right in job_ids[left_index:]:
                pair_rows.append([pair_id, pair_id, left, right, 1.1, 1.2])
                pair_id += 1
        pairs = pd.DataFrame(
            pair_rows,
            columns=[
                "pair_row_id",
                "pair_cluster_id",
                "jobA_id",
                "jobB_id",
                "slowdown_A",
                "slowdown_B",
            ],
        )
        distance_scaler = ProfileScaler("base").fit(data.inhibitors)
        model_configs = {
            kind: pipeline.ModelConfig()
            for kind in ["low_rank", "generic", "absolute"]
        }
        folds = pd.DataFrame(
            [
                {
                    "model_kind": kind,
                    "config_id": 0,
                    "best_validation": 0.1,
                }
                for kind in model_configs
            ]
        )
        events = []

        def fake_crossed(training_data, *_args):
            events.append(("crossed", set(training_data.jobs["job_id"])))
            return (
                model_configs,
                {},
                {kind: 1 for kind in model_configs},
                folds,
                np.asarray([1.0]),
                distance_scaler,
            )

        def fake_final(training_data, *_args):
            events.append(("final", set(training_data.jobs["job_id"])))
            return {}, {}

        def fake_pair_load(*_args):
            events.append(("pair_load", None))
            return pairs

        def fake_evaluate(full_data, selected_pairs, *_args, **kwargs):
            events.append(
                (
                    "evaluate",
                    set(full_data.responses["job_id"]),
                    len(selected_pairs),
                    kwargs["evaluation_method"],
                    set(kwargs["known_apps"]),
                )
            )
            return pd.DataFrame(), pd.DataFrame()

        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            for method, expected_pairs in [("one_known", 2), ("zero_shot", 1)]:
                args = argparse.Namespace(
                    jobs_csv=Path("unused-jobs.csv"),
                    inhibitors_csv=Path("unused-inhibitors.csv"),
                    job_inh_csv=Path("unused-responses.csv"),
                    pair_csv=Path("unused-pairs.csv"),
                    output_dir=Path(directory) / method,
                    config=None,
                    eval_method=method,
                    training_apps="v0,v1",
                    cv_seeds=[0],
                    final_seeds=[0],
                    inhibitor_blocks=2,
                    max_epochs=1,
                    patience=1,
                    batches_per_epoch=1,
                    bootstrap_samples=2,
                    skip_holdout=False,
                )
                with mock.patch.object(
                    pipeline, "load_training_data", return_value=data
                ), mock.patch.object(
                    pipeline, "build_inhibitor_blocks", return_value=blocks
                ), mock.patch.object(
                    pipeline, "crossed_validation", side_effect=fake_crossed
                ), mock.patch.object(
                    pipeline, "train_final_models", side_effect=fake_final
                ), mock.patch.object(
                    pipeline, "load_pair_holdout", side_effect=fake_pair_load
                ), mock.patch.object(
                    pipeline, "evaluate_holdout", side_effect=fake_evaluate
                ), mock.patch.object(pipeline, "write_evaluation_reports"):
                    pipeline.run(args)

                mode_events = events[-4:]
                self.assertEqual([event[0] for event in mode_events], [
                    "crossed",
                    "final",
                    "pair_load",
                    "evaluate",
                ])
                self.assertEqual(mode_events[0][1], {"v0", "v1"})
                self.assertEqual(mode_events[1][1], {"v0", "v1"})
                self.assertEqual(mode_events[3][1], {"v0", "v1", "v2"})
                self.assertEqual(mode_events[3][2], expected_pairs)
                self.assertEqual(mode_events[3][3], method)
                self.assertEqual(mode_events[3][4], {"v0", "v1"})


class DiagnosticAndOODTests(unittest.TestCase):
    def test_weighted_dispersion_and_median_effective_count(self) -> None:
        anchors = pd.DataFrame(
            {"inhib_id": ["a", "b", "c"], "log_slowdown": [0.1, 0.5, 1.2]}
        )
        model_profiles = {
            "a": np.zeros(4, dtype=np.float32),
            "b": np.ones(4, dtype=np.float32),
            "c": np.full(4, 2.0, dtype=np.float32),
        }
        distance_profiles = {
            "a": np.asarray([0.0, 0.0, 0.0, 0.0]),
            "b": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "c": np.asarray([3.0, 0.0, 0.0, 0.0]),
        }

        def zeros(_model, _victim, aggressors, _device):
            return np.zeros(len(aggressors))

        with mock.patch.object(inference, "response_values", side_effect=zeros):
            kernel = inference.predict_with_anchors(
                mock.Mock(),
                np.zeros(4),
                np.zeros(4),
                anchors,
                model_profiles,
                method="kernel",
                temperature=1.0,
                reference_distances=np.asarray([0.1, 0.2]),
                ood_config=inference.OODConfig(),
                device=torch.device("cpu"),
                distance_target_profile=np.asarray([0.1, 0.0, 0.0, 0.0]),
                distance_inhibitor_profiles=distance_profiles,
            )
            median = inference.predict_with_anchors(
                mock.Mock(),
                np.zeros(4),
                np.zeros(4),
                anchors,
                model_profiles,
                method="median",
                temperature=1.0,
                reference_distances=np.asarray([0.1, 0.2]),
                ood_config=inference.OODConfig(),
                device=torch.device("cpu"),
                distance_target_profile=np.asarray([0.1, 0.0, 0.0, 0.0]),
                distance_inhibitor_profiles=distance_profiles,
            )
        self.assertTrue(np.isfinite(kernel["weighted_residual_std"]))
        self.assertTrue(np.isnan(median["effective_anchor_count"]))
        self.assertTrue(np.isnan(median["weighted_residual_std"]))
        self.assertTrue(np.isfinite(median["kernel_component_effective_anchor_count"]))
        self.assertTrue(np.isfinite(median["fallback_residual_mad"]))

    def test_distances_use_common_coordinate_when_model_scalers_differ(self) -> None:
        anchors = pd.DataFrame(
            {"inhib_id": ["x", "y"], "log_slowdown": [0.1, 0.2]}
        )
        model_profiles = {"x": np.zeros(4), "y": np.zeros(4)}
        distance_profiles = {
            "x": np.asarray([0.0, 0.0, 0.0, 0.0]),
            "y": np.asarray([3.0, 4.0, 0.0, 0.0]),
        }
        captured: dict[str, np.ndarray] = {}
        original = inference.aggregate_residuals

        def capture(residuals, distances, method, **kwargs):
            captured["distances"] = distances.copy()
            return original(residuals, distances, method, **kwargs)

        with mock.patch.object(
            inference, "response_values", return_value=np.zeros(3)
        ), mock.patch.object(inference, "aggregate_residuals", side_effect=capture):
            inference.predict_with_anchors(
                mock.Mock(),
                np.full(4, 99.0),
                np.full(4, -99.0),
                anchors,
                model_profiles,
                method="kernel",
                temperature=1.0,
                reference_distances=np.asarray([1.0]),
                ood_config=inference.OODConfig(),
                device=torch.device("cpu"),
                distance_target_profile=np.asarray([0.0, 4.0, 0.0, 0.0]),
                distance_inhibitor_profiles=distance_profiles,
            )
        np.testing.assert_allclose(captured["distances"], [4.0, 3.0])


class SelectionAndLeakageTests(unittest.TestCase):
    def test_fixed_global_selection_skips_crossed_validation(self) -> None:
        data, blocks = crossed_data()
        aggregation = {
            "uniform": {},
            "median": {},
            "kernel": {"temperature": 1.0},
            "ood_kernel": {
                "temperature": 1.0,
                "quantile": 0.95,
                "fallback": "median",
            },
            "primary_method": "uniform",
        }
        config = {
            "evaluation_method": "random_split",
            "selection_mode": "fixed_global",
            "model_candidates_by_kind": {
                kind: [{}] for kind in ["low_rank", "generic", "absolute"]
            },
            "fixed_final_epochs_by_kind": {
                "low_rank": 2,
                "generic": 3,
                "absolute": 4,
            },
            "fixed_aggregations_by_kind": {
                "low_rank": aggregation,
                "generic": aggregation,
            },
            "fixed_selection_source": {
                "selection_scope": "all application App-Inhibitor data",
                "model_selection": {
                    kind: {"validation_log_mae": 0.1}
                    for kind in ["low_rank", "generic", "absolute"]
                },
            },
            "final_seeds": [0],
            "inference_anchor_counts": ["all"],
            "device": "cpu",
        }
        distance_scaler = ProfileScaler("base").fit(data.inhibitors)
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = argparse.Namespace(
                jobs_csv=Path("unused-jobs.csv"),
                inhibitors_csv=Path("unused-inhibitors.csv"),
                job_inh_csv=Path("unused-responses.csv"),
                pair_csv=Path("must-not-open.csv"),
                output_dir=root / "outputs",
                config=config_path,
                eval_method=None,
                training_apps=None,
                cv_seeds=None,
                final_seeds=None,
                inhibitor_blocks=None,
                max_epochs=None,
                patience=None,
                batches_per_epoch=None,
                bootstrap_samples=None,
                skip_holdout=True,
            )
            with mock.patch.object(
                pipeline, "load_training_data", return_value=data
            ), mock.patch.object(
                pipeline, "build_inhibitor_blocks", return_value=blocks
            ), mock.patch.object(
                pipeline,
                "distance_calibration",
                return_value=(np.asarray([0.5, 1.0]), distance_scaler),
            ), mock.patch.object(
                pipeline, "crossed_validation"
            ) as crossed, mock.patch.object(
                pipeline, "train_final_models", return_value=({}, {})
            ) as final, mock.patch.object(pipeline, "load_pair_holdout") as pair_loader:
                pipeline.run(args)

            crossed.assert_not_called()
            pair_loader.assert_not_called()
            self.assertEqual(final.call_args.args[3], config["fixed_final_epochs_by_kind"])
            selection = json.loads((args.output_dir / "selection.json").read_text())
            self.assertEqual(selection["selection_mode"], "fixed_global")
            self.assertEqual(selection["crossed_fold_count"], 0)
            self.assertEqual(selection["final_epoch_source"], "fixed_global_selection")

    def test_model_candidates_can_be_configured_per_model_kind(self) -> None:
        raw = {
            "model_candidates_by_kind": {
                "low_rank": [{"rank": 1}],
                "generic": [{"feature_set": "augmented"}],
                "absolute": [{"hidden_dim": 16}],
            }
        }
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            config = pipeline.load_run_config(path)
        self.assertEqual(
            pipeline.model_candidates_for_kind(config, "low_rank")[0].rank, 1
        )
        self.assertEqual(
            pipeline.model_candidates_for_kind(config, "generic")[0].feature_set,
            "augmented",
        )
        self.assertEqual(
            pipeline.model_candidates_for_kind(config, "absolute")[0].hidden_dim,
            16,
        )

    def test_crossed_rows_scalers_and_model_specific_epochs(self) -> None:
        data, blocks = crossed_data()
        config = pipeline.RunConfig(
            cv_seeds=[7],
            final_seeds=[7],
            max_epochs=3,
            patience=1,
            batches_per_epoch=1,
            batch_size=4,
            temperatures=[1.0],
            ood_quantiles=[0.95],
            ood_fallbacks=["uniform"],
            bootstrap_samples=2,
            device="cpu",
        )
        training_rows: list[tuple[str, pd.DataFrame]] = []
        scaler_rows: list[pd.DataFrame] = []
        original_fit = pipeline.ProfileScaler.fit

        def record_fit(scaler, frame):
            scaler_rows.append(frame.copy())
            return original_fit(scaler, frame)

        def fake_potential(model, responses, *args, **kwargs):
            kind = "generic" if isinstance(model, GenericPotential) else "low_rank"
            training_rows.append((kind, responses.copy()))
            EpisodeSampler(responses, args[0], args[1], kwargs["seed"])
            epoch = 2 if kind == "generic" else 1
            return {"best_epoch": epoch, "best_validation": 0.2, "epochs_run": epoch}

        def fake_absolute(_model, responses, *args, **kwargs):
            training_rows.append(("absolute", responses.copy()))
            return {"best_epoch": 3, "best_validation": 0.3, "epochs_run": 3}

        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory, mock.patch.object(
            pipeline.ProfileScaler, "fit", record_fit
        ), mock.patch.object(
            pipeline, "train_potential", side_effect=fake_potential
        ), mock.patch.object(
            pipeline, "train_absolute", side_effect=fake_absolute
        ):
            result = pipeline.crossed_validation(
                data, blocks, config, torch.device("cpu"), Path(directory)
            )
        _, _, epochs, folds, _, _ = result
        self.assertEqual(epochs, {"low_rank": 1, "generic": 2, "absolute": 3})
        self.assertEqual(set(folds["model_kind"]), {"low_rank", "generic", "absolute"})
        self.assertEqual(len(scaler_rows), 7)
        block_map = dict(zip(blocks["inhib_id"], blocks["block"]))
        for index, (victim, block) in enumerate(
            (victim, block) for victim in ["v0", "v1", "v2"] for block in [0, 1]
        ):
            fitted = scaler_rows[index + 1]
            self.assertNotIn(victim, set(fitted.get("job_id", pd.Series(dtype=str)).dropna()))
            inhibitor_ids = fitted.get("inhib_id", pd.Series(dtype=str)).dropna()
            self.assertFalse(inhibitor_ids.map(block_map).eq(block).any())
            for kind, train in training_rows[index * 3 : index * 3 + 3]:
                self.assertNotIn(victim, set(train["job_id"]))
                self.assertFalse(train["inhib_id"].map(block_map).eq(block).any(), kind)

    def test_sampler_keeps_inhibitors_distinct_with_response_replicates(self) -> None:
        jobs = {"v": np.zeros(4, dtype=np.float32)}
        inhibitors = {
            "a": np.zeros(4, dtype=np.float32),
            "b": np.ones(4, dtype=np.float32),
        }
        responses = pd.DataFrame(
            {
                "job_id": ["v", "v", "v"],
                "inhib_id": ["a", "a", "b"],
                "log_slowdown": [0.1, 0.2, 0.3],
            }
        )
        sampler = EpisodeSampler(responses, jobs, inhibitors, seed=3)
        _, anchors, queries, _ = sampler.sample(200)
        self.assertTrue(np.all(np.any(anchors.numpy() != queries.numpy(), axis=1)))

    def test_one_standard_error_rule_prefers_simpler_method(self) -> None:
        rows = []
        for victim in ["a", "b", "c", "d"]:
            for method, error in [("uniform", 0.101), ("median", 0.102), ("kernel", 0.1), ("ood_kernel", 0.099)]:
                rows.append(
                    {
                        "victim_id": victim,
                        "heldout_block": 0,
                        "seed": 0,
                        "method": method,
                        "temperature": 1.0,
                        "ood_quantile": 0.95,
                        "ood_fallback": "uniform",
                        "absolute_log_error": error + (0.02 if victim == "d" else 0.0),
                    }
                )
        selected = pipeline._select_aggregation(pd.DataFrame(rows))
        self.assertEqual(selected["primary_method"], "uniform")
        self.assertEqual(selected["accuracy_best"]["method"], "ood_kernel")
        self.assertEqual(
            selected["selection_rule"]["complexity_order"],
            ["uniform", "median", "kernel", "ood_kernel"],
        )


class BootstrapAndHoldoutTests(unittest.TestCase):
    def test_bootstrap_draws_are_paired_and_keep_directions_together(self) -> None:
        rows = []
        for method, offset in [("a", 0.0), ("b", 0.1)]:
            for pair_id in range(3):
                for direction in ["A", "B"]:
                    rows.append(
                        [method, pair_id, direction, 1.0 + pair_id, 1.1 + pair_id + offset]
                    )
        frame = pd.DataFrame(
            rows,
            columns=["method", "pair_row_id", "direction", "true_slowdown", "predicted_slowdown"],
        )
        captured: list[np.ndarray] = []
        original = metrics.resample_pair_clusters

        def record(sample_frame, sampled):
            captured.append(np.asarray(sampled).copy())
            result = original(sample_frame, sampled)
            counts = result.groupby("pair_row_id")["direction"].nunique()
            self.assertTrue((counts == 2).all())
            return result

        with mock.patch.object(metrics, "resample_pair_clusters", side_effect=record):
            metrics.cluster_bootstrap(frame, samples=4, seed=9)
        for left, right in zip(captured[:4], captured[4:]):
            np.testing.assert_array_equal(left, right)
        differences = metrics.paired_cluster_bootstrap_differences(
            frame, samples=20, seed=9
        )
        self.assertFalse(differences.empty)
        self.assertEqual(set(differences["cluster_count"]), {3})

    def test_skip_holdout_completes_without_pair_access(self) -> None:
        synthetic = AUDIT_DIR / "synthetic_data"
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            output = Path(directory) / "run"
            args = argparse.Namespace(
                jobs_csv=synthetic / "jobs.csv",
                inhibitors_csv=synthetic / "inhibitors.csv",
                job_inh_csv=synthetic / "job_inh.csv",
                pair_csv=AUDIT_DIR / "must_not_open.csv",
                output_dir=output,
                config=None,
                cv_seeds=[0],
                final_seeds=[0],
                inhibitor_blocks=2,
                max_epochs=1,
                patience=1,
                batches_per_epoch=1,
                bootstrap_samples=2,
                skip_holdout=True,
            )
            with mock.patch.object(pipeline, "load_pair_holdout") as loader:
                pipeline.run(args)
            loader.assert_not_called()
            report = json.loads((output / "run_report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["holdout_opened_after_final_training"])
            self.assertIsNone(report["pair_rows"])


if __name__ == "__main__":
    unittest.main()
