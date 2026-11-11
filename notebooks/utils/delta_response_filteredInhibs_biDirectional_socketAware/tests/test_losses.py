from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import JointResponseModel
from training import (
    JointEpisodeSampler,
    batch_to_tensors,
    identified_delta_huber_loss,
    joint_loss,
    masked_absolute_huber_loss,
    masked_auxiliary_loss,
)


def toy_responses() -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    for job in ["a", "b"]:
        for index in range(4):
            row = {"job_id": job, "inhib_id": f"i{index}", "log_slowdown": 0.1 * index,
                   "observation_id": f"{job}|i{index}|0"}
            for endpoint in ["app", "inhib"]:
                for task in ["bw_wall", "bw_mpi", "msg_wall"]:
                    row[f"{endpoint}_{task}_log_degradation"] = 0.05 * index - 0.1
                    row[f"{endpoint}_{task}_valid"] = True
            rows.append(row)
    rng = np.random.default_rng(1)
    jobs = {job: rng.normal(size=4).astype(np.float32) for job in ["a", "b"]}
    inhibitors = {f"i{index}": rng.normal(size=4).astype(np.float32) for index in range(4)}
    return pd.DataFrame(rows), jobs, inhibitors


class AbsoluteAndDeltaLossTests(unittest.TestCase):
    def test_floor_absolute_loss_is_one_sided(self) -> None:
        latent = torch.tensor([-0.5, 0.4, 0.2], requires_grad=True)
        observed = torch.tensor([0.0, 0.0, 0.5])
        loss = masked_absolute_huber_loss(latent, observed, torch.ones(3))
        loss.backward()
        self.assertEqual(float(latent.grad[0]), 0.0)
        self.assertGreater(float(latent.grad[1]), 0.0)
        self.assertLess(float(latent.grad[2]), 0.0)

    def test_both_floor_delta_pairs_identify_nothing(self) -> None:
        prediction = torch.tensor([0.3, -0.2], requires_grad=True)
        loss, count = identified_delta_huber_loss(prediction, torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool))
        self.assertEqual(count, 0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros(2)))

    def test_one_sided_delta_directions(self) -> None:
        valid = torch.ones(2, dtype=torch.bool)
        # Query at floor: delta must not exceed -anchor (= -0.4).
        loss, count = identified_delta_huber_loss(torch.tensor([-0.5, -0.3]), torch.tensor([0.4, 0.4]),
                                                  torch.zeros(2), valid)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss), 0.5 * (0.5 * 0.1**2), places=6)
        # Anchor at floor: delta must be at least the query log (= 0.3).
        loss, _ = identified_delta_huber_loss(torch.tensor([0.5, 0.1]), torch.zeros(2), torch.tensor([0.3, 0.3]), valid)
        self.assertAlmostEqual(float(loss), 0.5 * (0.5 * 0.2**2), places=6)
        # Invalid (same-inhibitor) pairs are excluded from the denominator.
        loss, count = identified_delta_huber_loss(torch.tensor([0.5, 9.0]), torch.tensor([0.1, 0.1]),
                                                  torch.tensor([0.2, 0.2]), torch.tensor([True, False]))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss), 0.5 * 0.4**2, places=6)


class AuxiliaryLossTests(unittest.TestCase):
    def test_all_masked_auxiliary_tasks_are_finite_and_differentiable(self) -> None:
        prediction = torch.randn(3, 2, requires_grad=True)
        loss, counts = masked_auxiliary_loss(prediction, torch.full((3, 2), float("nan")),
                                             torch.zeros(3, 2, dtype=torch.bool), torch.ones(2))
        self.assertEqual(counts, [0, 0])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_mean_within_task_then_across_available_tasks(self) -> None:
        prediction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = torch.zeros(2, 3)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        loss, counts = masked_auxiliary_loss(prediction, target, mask, torch.tensor([2.0, 1.0, 1.0]))
        # Task 0: mean(huber(0.5), huber(0)) = 0.0625; task 1: 0; task 2 unavailable.
        self.assertEqual(counts, [2, 1, 0])
        self.assertAlmostEqual(float(loss), 0.0625 / 2, places=6)

    def test_reverse_labels_reach_shared_role_projections(self) -> None:
        torch.manual_seed(0)
        responses, jobs, inhibitors = toy_responses()
        for task in ["bw_wall", "bw_mpi", "msg_wall"]:
            responses[f"app_{task}_valid"] = False
        sampler = JointEpisodeSampler(responses, jobs, inhibitors, aux_tasks=("bw_wall", "bw_mpi", "msg_wall"), seed=0)
        batch = batch_to_tensors(sampler.sample(16), torch.device("cpu"))
        model = JointResponseModel(aux_tasks=("bw_wall", "bw_mpi", "msg_wall"))
        loss, details = joint_loss(model, batch, lambda_abs=0.0, lambda_delta=0.0, lambda_aux=1.0,
                                   aux_scales=torch.ones(3))
        loss.backward()
        self.assertGreater(sum(details["auxiliary_valid_counts"]), 0)
        self.assertGreater(float(model.aggressor_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.victim_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder.network[0].weight.grad.abs().sum()), 0.0)


class SamplerTests(unittest.TestCase):
    def test_support_and_queries_are_disjoint_and_distinct(self) -> None:
        responses, jobs, inhibitors = toy_responses()
        responses = pd.concat([responses, responses.assign(observation_id=responses["observation_id"] + "r")])
        matrices = {job: (sorted(group["inhib_id"].unique()), np.zeros((4, 3), np.float32))
                    for job, group in responses.groupby("job_id")}
        sampler = JointEpisodeSampler(responses, jobs, inhibitors, aux_tasks=(), seed=3, token_matrices=matrices,
                                      max_support=7, empty_probability=0.25)
        sizes = []
        for _ in range(400):
            episode = sampler.sample_episode("a")
            self.assertNotIn(episode["anchor_id"], episode["support_ids"])
            self.assertNotIn(episode["query_id"], episode["support_ids"])
            self.assertNotEqual(episode["anchor_id"], episode["query_id"])
            sizes.append(len(episode["support_ids"]))
        self.assertLessEqual(max(sizes), 2)
        self.assertTrue(0.15 < np.mean(np.asarray(sizes) == 0) < 0.35)
        self.assertEqual(set(sizes), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
