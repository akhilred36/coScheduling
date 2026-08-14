from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = MODULE_DIR / "audit_outputs"
sys.path.insert(0, str(MODULE_DIR))

from data import ProfileScaler, expand_directional_pairs, load_training_data
from inference import OODConfig, aggregate_residuals, ood_alpha
from model import LowRankPotential


class ModelTests(unittest.TestCase):
    def test_delta_invariants_are_exact(self) -> None:
        torch.manual_seed(2)
        model = LowRankPotential()
        victim = torch.randn(5, 4)
        left = torch.randn(5, 4)
        right = torch.randn(5, 4)
        identity = model(victim, left, left)
        forward = model(victim, left, right)
        reverse = model(victim, right, left)
        middle = torch.randn(5, 4)
        path = model(victim, left, middle) + model(victim, middle, right)
        self.assertTrue(torch.equal(identity, torch.zeros_like(identity)))
        self.assertTrue(torch.equal(forward, -reverse))
        torch.testing.assert_close(forward, path, rtol=1e-6, atol=1e-6)


class AggregationTests(unittest.TestCase):
    def test_uniform_median_kernel_and_ood(self) -> None:
        residuals = np.asarray([0.0, 1.0, 10.0])
        distances = np.asarray([0.1, 1.0, 2.0])
        reference = np.asarray([0.1, 0.2, 0.3])
        uniform, weights, _ = aggregate_residuals(
            residuals,
            distances,
            "uniform",
            temperature=1.0,
            reference_distances=reference,
            ood_config=OODConfig(),
        )
        median, _, _ = aggregate_residuals(
            residuals,
            distances,
            "median",
            temperature=1.0,
            reference_distances=reference,
            ood_config=OODConfig(),
        )
        kernel, _, _ = aggregate_residuals(
            residuals,
            distances,
            "kernel",
            temperature=0.01,
            reference_distances=reference,
            ood_config=OODConfig(),
        )
        self.assertAlmostEqual(uniform, 11.0 / 3.0)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(median, 1.0)
        self.assertLess(kernel, 0.01)
        _, median_weights, _ = aggregate_residuals(
            residuals,
            distances,
            "median",
            temperature=1.0,
            reference_distances=reference,
            ood_config=OODConfig(),
        )
        self.assertTrue(np.isnan(median_weights).all())
        self.assertEqual(ood_alpha(0.1, reference, 0.95), 1.0)
        self.assertLess(ood_alpha(2.0, reference, 0.95), 1.0)


class DataTests(unittest.TestCase):
    def test_validation_quarantines_unknown_inhibitors(self) -> None:
        jobs = pd.DataFrame(
            [["a", 1.0, 0.5, 2.0, 4.0]],
            columns=["job_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
        )
        inhibitors = pd.DataFrame(
            [["i", 1.0, 0.5, 2.0, 4.0]],
            columns=["inhib_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
        )
        responses = pd.DataFrame(
            [["a", "i", 1.2], ["a", "missing", 1.3]],
            columns=["job_id", "inhib_id", "slowdown"],
        )
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            jobs.to_csv(root / "jobs.csv", index=False)
            inhibitors.to_csv(root / "inhibitors.csv", index=False)
            responses.to_csv(root / "responses.csv", index=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = load_training_data(
                    root / "jobs.csv", root / "inhibitors.csv", root / "responses.csv"
                )
        self.assertEqual(len(loaded.responses), 1)
        self.assertEqual(len(loaded.rejected_responses), 1)

    def test_scaler_and_directional_expansion(self) -> None:
        profiles = pd.DataFrame(
            [[1.0, 0.25, 10.0, 100.0], [3.0, 0.5, 20.0, 400.0]],
            columns=["mpi_time", "comm_frac", "total_msgs", "total_bytes"],
        )
        scaler = ProfileScaler("augmented").fit(profiles)
        transformed = scaler.transform(profiles)
        self.assertEqual(transformed.shape, (2, 8))
        self.assertTrue(np.isfinite(transformed).all())
        pairs = pd.DataFrame(
            {
                "pair_row_id": [7],
                "jobA_id": ["a"],
                "jobB_id": ["b"],
                "slowdown_A": [1.1],
                "slowdown_B": [1.2],
            }
        )
        expanded = expand_directional_pairs(pairs)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(set(expanded["victim_id"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
