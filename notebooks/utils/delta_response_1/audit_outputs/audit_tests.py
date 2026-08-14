from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


AUDIT_DIR = Path(__file__).resolve().parent
MODULE_DIR = AUDIT_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

import data as data_module
import inference
import metrics
import pipeline
from data import ProfileScaler, TrainingData, expand_directional_pairs, load_training_data
from model import AbsoluteResponseModel, LowRankPotential
from training import EpisodeSampler


def make_profiles(id_column: str, ids: list[str], offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for index, identifier in enumerate(ids):
        value = float(index + 1) + offset
        rows.append([identifier, value, min(0.1 * value, 1.0), 10.0 * value, 100.0 * value])
    return pd.DataFrame(
        rows,
        columns=[id_column, "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
    )


def make_crossed_data() -> tuple[TrainingData, pd.DataFrame]:
    jobs = make_profiles("job_id", ["v0", "v1", "v2"])
    inhibitors = make_profiles("inhib_id", ["i0", "i1", "i2", "i3"], 0.25)
    rows = []
    for victim_index, victim in enumerate(jobs["job_id"]):
        for inhibitor_index, inhibitor_id in enumerate(inhibitors["inhib_id"]):
            slowdown = 1.0 + 0.1 * victim_index + 0.01 * inhibitor_index
            rows.append([victim, inhibitor_id, slowdown, np.log(slowdown), 0])
    responses = pd.DataFrame(
        rows,
        columns=["job_id", "inhib_id", "slowdown", "log_slowdown", "replicate_id"],
    )
    blocks = pd.DataFrame({"inhib_id": inhibitors["inhib_id"], "block": [0, 0, 1, 1]})
    return TrainingData(jobs, inhibitors, responses, pd.DataFrame(), pd.DataFrame()), blocks


class CrossedValidationLeakageTests(unittest.TestCase):
    def test_fold_exclusions_scaler_and_fixed_queries(self) -> None:
        data, blocks = make_crossed_data()
        config = pipeline.RunConfig(
            cv_seeds=[13],
            final_seeds=[13],
            max_epochs=1,
            patience=1,
            batches_per_epoch=1,
            batch_size=4,
            temperatures=[1.0],
            ood_quantiles=[0.95],
            ood_fallbacks=["uniform"],
            bootstrap_samples=2,
            device="cpu",
        )
        scaler_frames: list[pd.DataFrame] = []
        training_frames: list[pd.DataFrame] = []
        original_fit = pipeline.ProfileScaler.fit

        def recording_fit(scaler: ProfileScaler, frame: pd.DataFrame) -> ProfileScaler:
            scaler_frames.append(frame.copy())
            return original_fit(scaler, frame)

        def fake_train(model: torch.nn.Module, responses: pd.DataFrame, *args, **kwargs):
            training_frames.append(responses.copy())
            return {"best_epoch": 1, "best_validation": 0.0, "epochs_run": 1, "history": []}

        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory, mock.patch.object(
            pipeline.ProfileScaler, "fit", recording_fit
        ), mock.patch.object(
            pipeline, "train_potential", side_effect=fake_train
        ), mock.patch.object(pipeline, "train_absolute", side_effect=fake_train):
            output = Path(directory)
            _, _, _, folds, _, _ = pipeline.crossed_validation(
                data, blocks, config, torch.device("cpu"), output
            )

            saved_predictions = pd.read_csv(output / "crossed_validation_predictions.csv")

        expected = [(victim, block) for victim in ["v0", "v1", "v2"] for block in [0, 1]]
        self.assertEqual(len(training_frames), len(expected) * 3)
        self.assertEqual(len(scaler_frames), len(expected) + 1)
        block_map = dict(zip(blocks["inhib_id"], blocks["block"]))
        response_blocks = data.responses["inhib_id"].map(block_map)
        for index, (victim, block) in enumerate(expected):
            fitted = scaler_frames[index + 1]
            for train in training_frames[index * 3 : index * 3 + 3]:
                self.assertNotIn(victim, set(train["job_id"]))
                self.assertFalse(train["inhib_id"].map(block_map).eq(block).any())
            self.assertNotIn(victim, set(fitted["job_id"].dropna()))
            self.assertFalse(fitted["inhib_id"].dropna().map(block_map).eq(block).any())
            expected_queries = set(
                data.responses.loc[
                    data.responses["job_id"].eq(victim) & response_blocks.eq(block), "inhib_id"
                ]
            )
            predicted_queries = set(
                saved_predictions.query(
                    "victim_id == @victim and heldout_block == @block"
                )["query_inhib_id"].astype(str)
            )
            self.assertEqual(predicted_queries, expected_queries)
            for train in training_frames[index * 3 : index * 3 + 3]:
                train_keys = set(zip(train["job_id"], train["inhib_id"]))
                self.assertTrue(all((victim, query) not in train_keys for query in expected_queries))
        self.assertEqual(len(folds), len(expected) * 3)


class SamplingTests(unittest.TestCase):
    def test_victims_uniform_and_inhibitors_distinct(self) -> None:
        jobs = make_profiles("job_id", ["a", "b", "c"])
        inhibitors = make_profiles("inhib_id", [f"i{i}" for i in range(8)], 0.3)
        scaler = ProfileScaler().fit(pd.concat([jobs, inhibitors], ignore_index=True))
        job_values = dict(zip(jobs["job_id"], scaler.transform(jobs)))
        inhibitor_values = dict(zip(inhibitors["inhib_id"], scaler.transform(inhibitors)))
        rows = []
        for victim, count in [("a", 2), ("b", 5), ("c", 8)]:
            for inhibitor_id in inhibitors["inhib_id"].iloc[:count]:
                rows.append([victim, inhibitor_id, 0.1])
        responses = pd.DataFrame(rows, columns=["job_id", "inhib_id", "log_slowdown"])
        sampler = EpisodeSampler(responses, job_values, inhibitor_values, seed=71)
        victim, anchor, query, _ = sampler.sample(60000)
        observed = victim.numpy()
        counts = [np.all(np.isclose(observed, value), axis=1).sum() for value in job_values.values()]
        self.assertLess(max(abs(np.asarray(counts) - 20000)), 700)
        self.assertTrue(np.all(np.any(anchor.numpy() != query.numpy(), axis=1)))


class StructuralAndInferenceTests(unittest.TestCase):
    def test_identity_antisymmetry_and_path_consistency(self) -> None:
        torch.manual_seed(5)
        model = LowRankPotential()
        victim = torch.randn(20, 4)
        left = torch.randn(20, 4)
        middle = torch.randn(20, 4)
        right = torch.randn(20, 4)
        identity = model(victim, left, left)
        forward = model(victim, left, right)
        reverse = model(victim, right, left)
        path = model(victim, left, middle) + model(victim, middle, right)
        torch.testing.assert_close(identity, torch.zeros_like(identity), rtol=0, atol=0)
        torch.testing.assert_close(forward, -reverse, rtol=0, atol=0)
        torch.testing.assert_close(forward, path, rtol=1e-6, atol=1e-6)

    def test_kernel_distance_is_input_profile_geometry(self) -> None:
        anchors = pd.DataFrame(
            {"inhib_id": ["x", "y"], "log_slowdown": [0.01, 0.1]}
        )
        profiles = {
            "x": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "y": np.asarray([3.0, 4.0, 0.0, 0.0], dtype=np.float32),
        }
        target = np.asarray([0.0, 4.0, 0.0, 0.0], dtype=np.float32)
        captured: dict[str, np.ndarray] = {}
        original = inference.aggregate_residuals

        def capture(residuals, distances, method, **kwargs):
            captured["distances"] = distances.copy()
            return original(residuals, distances, method, **kwargs)

        with mock.patch.object(inference, "response_values", return_value=np.zeros(3)), mock.patch.object(
            inference, "aggregate_residuals", side_effect=capture
        ):
            inference.predict_with_anchors(
                LowRankPotential(),
                np.zeros(4, dtype=np.float32),
                target,
                anchors,
                profiles,
                method="kernel",
                temperature=1.0,
                reference_distances=np.asarray([1.0]),
                ood_config=inference.OODConfig(),
                device=torch.device("cpu"),
            )
        np.testing.assert_allclose(captured["distances"], [4.0, 3.0])

    def test_all_low_rank_aggregations_share_model_object(self) -> None:
        data, _ = make_crossed_data()
        scaler = ProfileScaler().fit(pd.concat([data.jobs, data.inhibitors], ignore_index=True))
        low_rank = LowRankPotential()
        generic = LowRankPotential()
        absolute = AbsoluteResponseModel()
        seen: list[int] = []

        def fake_predict(model, *args, **kwargs):
            seen.append(id(model))
            return {
                "latent_predicted_log_slowdown": 0.0,
                "predicted_log_slowdown": 0.0,
                "number_of_anchors": 4,
                "uncensored_anchor_count": 4,
                "nearest_anchor_distance": 1.0,
                "effective_anchor_count": 1.0,
                "weighted_residual_std": 0.0,
                "kernel_component_effective_anchor_count": 1.0,
                "kernel_component_weighted_residual_std": 0.0,
                "unweighted_residual_std": 0.0,
                "fallback_residual_mad": 0.0,
                "ood_score": 0.5,
                "ood_alpha": 1.0,
            }

        pairs = pd.DataFrame(
            {
                "pair_row_id": [0],
                "jobA_id": ["v0"],
                "jobB_id": ["v1"],
                "slowdown_A": [1.1],
                "slowdown_B": [1.2],
            }
        )
        aggregation = {
            "primary_method": "uniform",
            "uniform": {"validation_log_mae": 0.1},
            "kernel": {"temperature": 1.0},
            "ood_kernel": {"temperature": 1.0, "quantile": 0.95, "fallback": "median"},
        }
        aggregations = {"low_rank": aggregation, "generic": aggregation}
        scalers = {"low_rank": scaler, "generic": scaler, "absolute": scaler}
        with mock.patch.object(pipeline, "predict_with_anchors", side_effect=fake_predict):
            pipeline.evaluate_holdout(
                data,
                pairs,
                scalers,
                {0: {"low_rank": low_rank, "generic": generic, "absolute": absolute}},
                aggregations,
                np.asarray([1.0]),
                scaler,
                torch.device("cpu"),
            )
        self.assertEqual(seen[:5], [id(low_rank)] * 5)
        self.assertEqual(seen[5], id(generic))
        self.assertEqual(seen[6:11], [id(low_rank)] * 5)
        self.assertEqual(seen[11], id(generic))


class HoldoutAndDataTests(unittest.TestCase):
    def test_holdout_load_follows_selection_and_final_training(self) -> None:
        source = inspect.getsource(pipeline.run)
        self.assertLess(source.index("crossed_validation("), source.index("train_final_models("))
        self.assertLess(source.index("train_final_models("), source.index("load_pair_holdout("))
        self.assertLess(source.index("load_pair_holdout("), source.index("evaluate_holdout("))

    def test_replicates_preserved_and_bad_rows_quarantined(self) -> None:
        jobs = make_profiles("job_id", ["a"])
        inhibitors = make_profiles("inhib_id", ["i0", "i1"])
        responses = pd.DataFrame(
            [
                ["a", "i0", 1.1],
                ["a", "i0", 1.2],
                ["a", "missing", 1.3],
                ["missing", "i1", 1.4],
                ["a", "i1", 0.0],
            ],
            columns=["job_id", "inhib_id", "slowdown"],
        )
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            jobs.to_csv(root / "jobs.csv", index=False)
            inhibitors.to_csv(root / "inhibitors.csv", index=False)
            responses.to_csv(root / "responses.csv", index=False)
            with self.assertWarns(Warning):
                loaded = load_training_data(
                    root / "jobs.csv", root / "inhibitors.csv", root / "responses.csv"
                )
        self.assertEqual(len(loaded.responses), 2)
        self.assertEqual(loaded.responses["replicate_id"].tolist(), [0, 1])
        self.assertEqual(len(loaded.duplicate_keys), 2)
        self.assertEqual(len(loaded.rejected_responses), 3)

    def test_directional_expansion_and_self_average(self) -> None:
        pairs = pd.DataFrame(
            {
                "pair_row_id": [0],
                "jobA_id": ["a"],
                "jobB_id": ["a"],
                "slowdown_A": [1.0],
                "slowdown_B": [3.0],
            }
        )
        expanded = expand_directional_pairs(pairs)
        self.assertEqual(expanded["true_slowdown"].tolist(), [1.0, 3.0])
        predictions = expanded.assign(method="m", predicted_slowdown=2.0)
        averaged = metrics.self_pair_averaged(predictions)
        self.assertEqual(len(averaged), 1)
        self.assertEqual(float(averaged.iloc[0]["true_slowdown"]), 2.0)
        self.assertEqual(float(averaged.iloc[0]["predicted_slowdown"]), 2.0)

    def test_bootstrap_uses_original_rows_as_clusters(self) -> None:
        rows = []
        for pair_id in range(5):
            for direction in ["A", "B"]:
                rows.append(["m", pair_id, direction, 1.0 + pair_id, 1.1 + pair_id])
        frame = pd.DataFrame(
            rows,
            columns=["method", "pair_row_id", "direction", "true_slowdown", "predicted_slowdown"],
        )
        result = metrics.cluster_bootstrap(frame, samples=20, seed=4)
        self.assertEqual(set(result["cluster_count"]), {5})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    unittest.main(argv=[sys.argv[0]], verbosity=2)
