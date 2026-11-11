"""Socket-aware mode: per-endpoint socket inputs, resolved sampling, and the blind path."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import pipeline
from contexts import ContextBuilder, ContextConfig, fit_token_scaler, token_dim
from data import socket_pair_inputs
from inference import JointPredictor, configuration_weights
from model import JointResponseModel, LowRankPotential
from training import (
    EpisodeSampler,
    JointEpisodeSampler,
    cell_balanced_row_weights,
    response_values,
)


def resolved_responses() -> tuple[pd.DataFrame, dict, dict]:
    """Two apps x four configurations; i3 of app b has only socket 1."""
    rows = []
    for job in ["a", "b"]:
        for index in range(4):
            sockets = [1] if (job, index) == ("b", 3) else [0, 1]
            for socket in sockets:
                row = {"job_id": job, "inhib_id": f"i{index}", "s_V": socket, "replicate_id": socket,
                       "observation_id": f"{job}|i{index}|{socket}",
                       "log_slowdown": 0.1 * index + 0.05 * socket, "is_censored": index == 0}
                for endpoint in ["app", "inhib"]:
                    for task in ["bw_wall", "bw_mpi", "msg_wall"]:
                        row[f"{endpoint}_{task}_log_degradation"] = 0.05 * index - 0.1
                        row[f"{endpoint}_{task}_valid"] = True
                rows.append(row)
    rng = np.random.default_rng(1)
    jobs = {job: rng.normal(size=4).astype(np.float32) for job in ["a", "b"]}
    inhibitors = {f"i{index}": rng.normal(size=4).astype(np.float32) for index in range(4)}
    return pd.DataFrame(rows), jobs, inhibitors


class SocketInputTests(unittest.TestCase):
    def test_each_endpoint_gets_its_own_socket_bit(self) -> None:
        victim = np.arange(4, dtype=np.float32)
        aggressors = np.ones((3, 4), dtype=np.float32)
        victims, augmented = socket_pair_inputs(victim, aggressors, [0, 1, 1])
        self.assertEqual(victims.shape, (3, 5))
        self.assertEqual(augmented.shape, (3, 5))
        np.testing.assert_array_equal(victims[:, -1], [0, 1, 1])
        np.testing.assert_array_equal(augmented[:, -1], [1, 0, 0])
        np.testing.assert_array_equal(victims[:, :4], np.tile(victim, (3, 1)))

    def test_joint_sampler_inputs_carry_row_sockets(self) -> None:
        responses, jobs, inhibitors = resolved_responses()
        sampler = JointEpisodeSampler(responses, jobs, inhibitors, aux_tasks=("bw_wall",), seed=0,
                                      socket_feature=True)
        batch = sampler.sample(64)
        for key in ["victim", "victim_query", "anchor", "query"]:
            self.assertEqual(batch[key].shape, (64, 5))
        anchor_rows = [episode["anchor_row"] for episode in batch["episodes"]]
        query_rows = [episode["query_row"] for episode in batch["episodes"]]
        np.testing.assert_array_equal(batch["victim"][:, -1], responses["s_V"].to_numpy()[anchor_rows])
        np.testing.assert_array_equal(batch["victim_query"][:, -1], responses["s_V"].to_numpy()[query_rows])
        np.testing.assert_array_equal(batch["anchor"][:, -1], 1 - batch["victim"][:, -1])
        np.testing.assert_array_equal(batch["query"][:, -1], 1 - batch["victim_query"][:, -1])

    def test_legacy_sampler_draws_distinct_configs_then_a_socket_row(self) -> None:
        responses, jobs, inhibitors = resolved_responses()
        sampler = EpisodeSampler(responses, jobs, inhibitors, seed=0, socket_feature=True)
        victim_anchor, victim_query, anchor, query, targets = sampler.sample_resolved(256)
        for tensor in [victim_anchor, victim_query, anchor, query]:
            self.assertEqual(tuple(tensor.shape), (256, 5))
        torch.testing.assert_close(anchor[:, -1], 1 - victim_anchor[:, -1])
        torch.testing.assert_close(query[:, -1], 1 - victim_query[:, -1])
        self.assertTrue(set(victim_anchor[:, -1].tolist()) == {0.0, 1.0})
        profiles = {name: torch.as_tensor(value) for name, value in inhibitors.items()}
        for row in range(256):
            anchor_id = next(name for name, value in profiles.items() if torch.equal(anchor[row, :4], value))
            query_id = next(name for name, value in profiles.items() if torch.equal(query[row, :4], value))
            self.assertNotEqual(anchor_id, query_id)

    def test_absolute_weights_give_each_cell_blind_weight(self) -> None:
        responses, _, _ = resolved_responses()
        weights = pd.Series(cell_balanced_row_weights(responses), index=responses.index)
        rows_per_victim = responses.groupby("job_id").size()
        # Victim uniform, row uniform within victim: expected cell weight = w * n_cell / N_victim.
        expected = (weights / responses["job_id"].map(rows_per_victim)).groupby(
            [responses["job_id"], responses["inhib_id"]]).sum()
        for job_id, values in expected.groupby(level=0):
            np.testing.assert_allclose(values.to_numpy(), 1.0 / values.size)

    def test_calibration_weights_split_a_config_across_sockets(self) -> None:
        weights = configuration_weights(["i0", "i0", "i1"])
        np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])

    def test_tokens_per_socket_cell_and_config_level_exclusion(self) -> None:
        responses, _, inhibitors = resolved_responses()
        config = ContextConfig(enabled=True)
        scaler = fit_token_scaler(responses, config)
        builder = ContextBuilder(responses, inhibitors, scaler, config, socket_feature=True)
        self.assertEqual(builder.width, token_dim(config, 4) + 1)
        tokens = builder.application_context("a")
        self.assertEqual(tokens.count, 8)
        np.testing.assert_array_equal(tokens.features[:, -1], [0, 1] * 4)
        held_out = builder.application_context("a", exclude_inhib_ids=["i1"])
        self.assertNotIn("i1", held_out.inhibitor_ids)
        self.assertEqual(held_out.count, 6)
        self.assertEqual(builder.support_ids("b"), ["i0", "i1", "i2", "i3"])

    def test_aware_predictor_requires_and_uses_sockets(self) -> None:
        torch.manual_seed(0)
        predictor = JointPredictor(JointResponseModel(input_dim=5), torch.device("cpu"), socket_feature=True)
        victim, aggressors = np.ones(4, np.float32), np.zeros((1, 4), np.float32)
        with self.assertRaises(ValueError):
            predictor.response(victim, None, aggressors)
        values = predictor.response(victim, None, np.repeat(aggressors, 2, axis=0), victim_sockets=[0, 1])
        self.assertNotAlmostEqual(values[0], values[1])

    def test_replicate_weighted_socket_average_in_slowdown_space(self) -> None:
        frame = pd.DataFrame({
            "method": ["m", "m"], "pair_row_id": [0, 0], "s_V": [0, 1], "socket_replicates": [1, 3],
            "latent_predicted_log_slowdown": [np.log(1.2), -0.1],
            "predicted_log_slowdown": [np.log(1.2), 0.0], "predicted_slowdown": [1.2, 1.0],
        })
        result = pipeline.replicate_weighted_socket_average(frame, ["method", "pair_row_id"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["predicted_slowdown"].iloc[0]), (1.2 + 3 * 1.0) / 4)
        self.assertAlmostEqual(float(result["predicted_log_slowdown"].iloc[0]), np.log((1.2 + 3.0) / 4))
        self.assertNotIn("s_V", result)


class BlindPathTests(unittest.TestCase):
    def test_flag_off_ignores_socket_column(self) -> None:
        responses, jobs, inhibitors = resolved_responses()
        blind = responses.drop(columns="s_V")
        left = JointEpisodeSampler(responses, jobs, inhibitors, aux_tasks=(), seed=4).sample(32)
        right = JointEpisodeSampler(blind, jobs, inhibitors, aux_tasks=(), seed=4).sample(32)
        self.assertNotIn("victim_query", left)
        for key in ["victim", "anchor", "query", "anchor_log", "query_log"]:
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(left["victim"].shape[1], 4)
        first = EpisodeSampler(responses, jobs, inhibitors, seed=2).sample(16)
        second = EpisodeSampler(blind, jobs, inhibitors, seed=2).sample(16)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_flag_off_model_and_config_are_unchanged(self) -> None:
        config = pipeline.RunConfig()
        self.assertFalse(config.socket_feature)
        self.assertNotIn("socket_feature", pipeline.run_config_dict(config))
        self.assertEqual(pipeline.socket_record(False), {})
        spec = pipeline.VARIANT_REGISTRY["bidirectional_context"]
        context = ContextConfig(enabled=True)
        blind = pipeline.make_model(spec, pipeline.ModelConfig(), context)
        aware = pipeline.make_model(spec, pipeline.ModelConfig(), context, socket_feature=True)
        self.assertEqual(blind.encoder.network[0].in_features, 4)
        self.assertEqual(aware.encoder.network[0].in_features, 5)
        self.assertEqual(aware.context_encoder.token_network[0].in_features,
                         blind.context_encoder.token_network[0].in_features + 1)

    def test_blind_response_values_match_explicit_broadcast(self) -> None:
        torch.manual_seed(0)
        model = LowRankPotential()
        victim, aggressors = np.ones(4, np.float32), np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
        with torch.no_grad():
            expected = model.response(torch.as_tensor(np.tile(victim, (3, 1))), torch.as_tensor(aggressors)).numpy()
        np.testing.assert_array_equal(response_values(model, victim, aggressors, torch.device("cpu")), expected)


if __name__ == "__main__":
    unittest.main()
