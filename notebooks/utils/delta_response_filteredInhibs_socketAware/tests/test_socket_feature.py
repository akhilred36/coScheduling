from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import inference
import pipeline
import run_experiments
from data import TrainingData, require_socket_resolution, socket_pair_inputs
from inference import OODConfig, predict_with_anchors
from pipeline import ModelConfig, RunConfig, make_model
from training import (
    EpisodeSampler,
    SocketEpisodeSampler,
    censored_absolute_huber_loss,
)


def _responses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["v"] * 5,
            "inhib_id": ["a", "a", "b", "b", "c"],
            "s_V": [0, 1, 0, 1, 1],
            "log_slowdown": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


class SocketInputTests(unittest.TestCase):
    def test_endpoint_bits_and_dims(self) -> None:
        victim = np.arange(4, dtype=np.float32)
        aggressors = np.ones((3, 4), dtype=np.float32)
        victims, aggressor_inputs = socket_pair_inputs(victim, aggressors, [0, 1, 1])
        self.assertEqual(victims.shape, (3, 5))
        self.assertEqual(aggressor_inputs.shape, (3, 5))
        np.testing.assert_array_equal(victims[:, :4], np.tile(victim, (3, 1)))
        np.testing.assert_array_equal(victims[:, 4], [0, 1, 1])
        np.testing.assert_array_equal(aggressor_inputs[:, 4], [1, 0, 0])

    def test_model_input_dim_gains_one_socket_bit(self) -> None:
        for feature_set, base_dim in [("base", 4), ("augmented", 8)]:
            config = ModelConfig(feature_set=feature_set)
            for kind in ["low_rank", "generic", "absolute"]:
                blind = make_model(config, kind)
                aware = make_model(config, kind, socket_feature=True)
                self.assertEqual(blind.encoder.network[0].in_features, base_dim)
                self.assertEqual(aware.encoder.network[0].in_features, base_dim + 1)

    def test_socket_sampler_uses_distinct_configs_and_own_sockets(self) -> None:
        jobs = {"v": np.zeros(4, dtype=np.float32)}
        inhibitors = {
            name: np.full(4, value, dtype=np.float32)
            for name, value in [("a", 1.0), ("b", 2.0), ("c", 3.0)]
        }
        sampler = SocketEpisodeSampler(_responses(), jobs, inhibitors, seed=3)
        anchor_victim, anchor, query_victim, query, targets = sampler.sample(300)
        for tensor in [anchor_victim, anchor, query_victim, query]:
            self.assertEqual(tuple(tensor.shape), (300, 5))
        self.assertTrue(torch.equal(anchor_victim[:, 4], 1 - anchor[:, 4]))
        self.assertTrue(torch.equal(query_victim[:, 4], 1 - query[:, 4]))
        self.assertTrue(torch.all(anchor[:, 0] != query[:, 0]))
        # Each target is the resolved row of (config, victim socket).
        lookup = {
            (1.0 + "abc".index(row.inhib_id), row.s_V): row.log_slowdown
            for row in _responses().itertuples()
        }
        for side, victim in [(anchor, anchor_victim), (query, query_victim)]:
            column = 0 if side is anchor else 1
            for index in range(300):
                key = (float(side[index, 0]), int(victim[index, 4]))
                self.assertAlmostEqual(
                    float(targets[index, column]), lookup[key], places=6
                )
        # Both sockets of a two-socket config are sampled.
        a_rows = anchor[:, 0] == 1.0
        self.assertEqual(set(anchor_victim[a_rows, 4].tolist()), {0.0, 1.0})

    def test_weighted_absolute_loss(self) -> None:
        latent = torch.tensor([0.5, 0.2, -0.1])
        observed = torch.tensor([0.4, 0.0, 0.3])
        unweighted = censored_absolute_huber_loss(latent, observed)
        uniform = censored_absolute_huber_loss(latent, observed, torch.ones(3))
        torch.testing.assert_close(unweighted, uniform)
        weighted = censored_absolute_huber_loss(
            latent, observed, torch.tensor([1.0, 0.5, 0.5])
        )
        per_row = torch.nn.functional.huber_loss(
            torch.tensor([0.1, 0.2, -0.4]), torch.zeros(3), reduction="none"
        )
        torch.testing.assert_close(weighted, (per_row * torch.tensor([0.5, 0.25, 0.25])).sum())

    def test_anchor_inference_scores_each_anchor_at_its_socket(self) -> None:
        anchors = _responses()
        profiles = {name: np.full(4, value, dtype=np.float32) for name, value in [("a", 1), ("b", 2), ("c", 3)]}
        calls = []

        def fake_response(_model, victims, aggressors, _device):
            calls.append((np.asarray(victims), np.asarray(aggressors)))
            return aggressors[:, 0] + 10 * victims[:, -1]

        with mock.patch.object(inference, "response_values", side_effect=fake_response):
            results = {
                socket: predict_with_anchors(
                    None,
                    np.zeros(4, dtype=np.float32),
                    np.full(4, 5.0, dtype=np.float32),
                    anchors,
                    profiles,
                    method="uniform",
                    temperature=1.0,
                    reference_distances=np.asarray([1.0]),
                    ood_config=OODConfig(),
                    device=torch.device("cpu"),
                    victim_socket=socket,
                )
                for socket in [0, 1]
            }
        victims, aggressors = calls[0]
        np.testing.assert_array_equal(victims[:, 4], [0, 1, 0, 1, 1, 0])
        np.testing.assert_array_equal(aggressors[:, 4], 1 - victims[:, 4])
        # Distances are socket-free (5-D profiles never enter them).
        self.assertEqual(results[0]["nearest_anchor_distance"], results[1]["nearest_anchor_distance"])
        self.assertAlmostEqual(
            results[1]["latent_predicted_log_slowdown"]
            - results[0]["latent_predicted_log_slowdown"],
            10.0,
            places=5,
        )

    def test_socket_resolution_keys_replicates_by_s_v(self) -> None:
        responses = _responses().assign(slowdown=1.1)
        responses["replicate_id"] = responses.groupby(["job_id", "inhib_id"]).cumcount()
        data = TrainingData(pd.DataFrame(), pd.DataFrame(), responses.iloc[[1, 2, 4]], None, None)
        resolved = require_socket_resolution(data)
        np.testing.assert_array_equal(resolved.responses["replicate_id"], [1, 0, 1])
        duplicated = TrainingData(None, None, pd.concat([responses, responses.iloc[[0]]]), None, None)
        with self.assertRaises(ValueError):
            require_socket_resolution(duplicated)


class SocketAggregationTests(unittest.TestCase):
    def test_rep_weighted_average_in_slowdown_space(self) -> None:
        pairs = pd.DataFrame(
            {"pair_row_id": [0], "n0_A": [3], "n1_A": [1], "n0_B": [0], "n1_B": [2]}
        )
        keys = {
            "evaluation_method": "random_split",
            "anchor_budget": "all",
            "method": "m",
            "pair_row_id": 0,
            "pair_cluster_id": 0,
            "victim_known": True,
            "aggressor_known": True,
            "known_endpoint_count": 2,
            "true_slowdown": 1.5,
        }
        rows = []
        for direction, victim, aggressor in [("A", "x", "y"), ("B", "y", "x")]:
            for socket, latent in [(0, np.log(1.2)), (1, np.log(2.0))]:
                rows.append({**keys, "direction": direction, "victim_id": victim,
                             "aggressor_id": aggressor, "s_V": socket,
                             "latent_predicted_log_slowdown": latent,
                             "model_seed_log_std": 0.0,
                             "anchor_budget_feasible": True,
                             "predicted_log_slowdown": latent,
                             "predicted_slowdown": float(np.exp(latent)),
                             "absolute_error": 0.0, "absolute_log_error": 0.0})
        column_order = list(keys)[:5] + ["direction", "victim_id", "aggressor_id"] + list(keys)[5:]
        resolved = pd.DataFrame(rows)
        resolved = resolved[column_order + [c for c in resolved.columns if c not in column_order]]
        seed = resolved.drop(columns=["model_seed_log_std", "absolute_error", "absolute_log_error"]).assign(seed=0)
        ensemble, seed_out = pipeline.aggregate_socket_predictions(resolved, seed, pairs)
        self.assertNotIn("s_V", ensemble)
        self.assertEqual(list(ensemble.columns), [c for c in resolved.columns if c != "s_V"])
        np.testing.assert_allclose(ensemble["predicted_slowdown"], [(3 * 1.2 + 2.0) / 4, 2.0])
        np.testing.assert_allclose(ensemble["predicted_log_slowdown"], np.log(ensemble["predicted_slowdown"]))
        np.testing.assert_allclose(ensemble["absolute_error"], abs(ensemble["predicted_slowdown"] - 1.5))
        self.assertTrue(ensemble["anchor_budget_feasible"].all())
        np.testing.assert_allclose(seed_out["predicted_slowdown"], ensemble["predicted_slowdown"])


class FlagOffTests(unittest.TestCase):
    def test_blind_records_and_plans_omit_the_flag(self) -> None:
        self.assertNotIn("socket_feature", pipeline.run_config_record(RunConfig()))
        self.assertTrue(pipeline.run_config_record(RunConfig(socket_feature=True))["socket_feature"])
        profile = run_experiments.PROFILES["smoke"]
        self.assertNotIn("socket_feature", run_experiments.base_run_config(profile))
        self.assertTrue(run_experiments.base_run_config(profile, True)["socket_feature"])
        self.assertIsNone(pipeline._row_sockets(False, _responses()))

    def test_blind_sampler_and_inputs_are_unchanged(self) -> None:
        jobs = {"v": np.zeros(4, dtype=np.float32)}
        inhibitors = {name: np.full(4, value, dtype=np.float32) for name, value in [("a", 1), ("b", 2), ("c", 3)]}
        batch = EpisodeSampler(_responses(), jobs, inhibitors, seed=3).sample(16)
        self.assertEqual(len(batch), 4)
        self.assertEqual(tuple(batch[0].shape), (16, 4))
        self.assertEqual(make_model(ModelConfig(), "low_rank").encoder.network[0].in_features, 4)


if __name__ == "__main__":
    unittest.main()
