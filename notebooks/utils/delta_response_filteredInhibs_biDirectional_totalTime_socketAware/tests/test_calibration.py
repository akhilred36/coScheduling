from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from contexts import ContextBuilder, ContextConfig, fit_token_scaler
from inference import (
    CalibrationConfig,
    configuration_weights,
    fit_floor_aware_offset,
    predict_with_crossfit_context_calibration,
)


def objective(offset: float, latent: np.ndarray, observed: np.ndarray, weights: np.ndarray, lam: float) -> float:
    def huber(value: np.ndarray) -> np.ndarray:
        return np.where(np.abs(value) <= 1, 0.5 * value**2, np.abs(value) - 0.5)

    residual = np.where(observed > 0, latent + offset - observed, np.maximum(latent + offset, 0.0))
    return float(np.sum(weights * huber(residual)) + 0.5 * lam * offset**2)


class OffsetTests(unittest.TestCase):
    def assert_minimizer(self, latent, observed, lam=1.0, ids=None) -> float:
        latent, observed = np.asarray(latent, float), np.asarray(observed, float)
        result = fit_floor_aware_offset(latent, observed, inhibitor_ids=ids, calibration_lambda=lam)
        self.assertTrue(result.converged)
        weights = configuration_weights(ids if ids is not None else [str(i) for i in range(len(latent))])
        best = objective(result.offset, latent, observed, weights, lam)
        for step in [-1e-4, 1e-4]:
            self.assertLessEqual(best, objective(result.offset + step, latent, observed, weights, lam) + 1e-12)
        return result.offset

    def test_mixed_exact_and_floor_anchors(self) -> None:
        offset = self.assert_minimizer([0.5, -0.2], [0.0, 0.3])
        self.assertAlmostEqual(offset, 0.0, places=8)
        result = fit_floor_aware_offset(np.array([0.5, -0.2]), np.array([0.0, 0.3]), calibration_lambda=1.0)
        self.assertEqual((result.exact_count, result.floor_count), (1, 1))

    def test_all_floor_support_pulls_predictions_down(self) -> None:
        offset = self.assert_minimizer([0.5, 0.4], [0.0, 0.0], lam=0.1)
        self.assertAlmostEqual(offset, -0.25 / 0.6, places=8)
        self.assertLess(offset, 0.0)
        # Already-benign floor anchors need no correction.
        self.assertAlmostEqual(self.assert_minimizer([-0.3, -1.0], [0.0, 0.0]), 0.0, places=8)

    def test_single_anchor_and_regularization_shrinkage(self) -> None:
        offset = self.assert_minimizer([0.1], [0.6], lam=1.0)
        self.assertAlmostEqual(offset, 0.25, places=8)
        self.assertGreater(self.assert_minimizer([0.1], [0.6], lam=0.1), offset)
        self.assertLess(self.assert_minimizer([0.1], [0.6], lam=10.0), offset)

    def test_empty_support_falls_back_to_zero(self) -> None:
        result = fit_floor_aware_offset(np.empty(0), np.empty(0), calibration_lambda=1.0)
        self.assertEqual(result.offset, 0.0)
        self.assertEqual(result.fallback, "empty_support")

    def test_deterministic_finite_and_validated(self) -> None:
        latent, observed = np.array([30.0, -40.0, 2.0]), np.array([0.0, 5.0, 0.0])
        first = fit_floor_aware_offset(latent, observed, calibration_lambda=0.1)
        second = fit_floor_aware_offset(latent, observed, calibration_lambda=0.1)
        self.assertEqual(first.offset, second.offset)
        self.assertTrue(np.isfinite(first.offset))
        self.assert_minimizer(latent, observed, lam=0.1)
        with self.assertRaises(ValueError):
            fit_floor_aware_offset(np.array([np.nan]), np.array([0.1]), calibration_lambda=1.0)
        with self.assertRaises(ValueError):
            fit_floor_aware_offset(np.array([0.0]), np.array([0.1]), calibration_lambda=0.0)
        with self.assertRaises(ValueError):
            CalibrationConfig(lambda_candidates=(0.1, -1.0))

    def test_equal_configuration_weighting_with_replicates(self) -> None:
        np.testing.assert_allclose(configuration_weights(["a", "a", "b"]), [0.25, 0.25, 0.5])
        single = fit_floor_aware_offset(np.array([0.0, 0.0]), np.array([0.2, 0.6]),
                                        inhibitor_ids=["a", "b"], calibration_lambda=1.0)
        replicated = fit_floor_aware_offset(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.2, 0.6]),
                                            inhibitor_ids=["a", "a", "a", "b"], calibration_lambda=1.0)
        self.assertAlmostEqual(single.offset, replicated.offset, places=9)


class RecordingPredictor:
    """Records the victim context used for every response call."""

    def __init__(self) -> None:
        self.calls = []

    def response(self, victim_profile, victim_tokens, aggressor_profiles, aggressor_tokens=None):
        signature = None if victim_tokens is None else victim_tokens.tobytes()
        self.calls.append((signature, len(aggressor_profiles)))
        return np.zeros(len(aggressor_profiles))


class CrossfitContextTests(unittest.TestCase):
    def test_anchor_and_target_share_each_views_fixed_context(self) -> None:
        rows = pd.DataFrame({
            "job_id": "a", "inhib_id": [f"i{index}" for index in range(5)],
            "observation_id": [f"a|i{index}|0" for index in range(5)],
            "log_slowdown": [0.0, 0.1, 0.2, 0.0, 0.4], "is_censored": [True, False, False, True, False],
        })
        for endpoint in ["app", "inhib"]:
            for task in ["bw_wall", "bw_mpi", "msg_wall"]:
                rows[f"{endpoint}_{task}_log_degradation"] = 0.1
                rows[f"{endpoint}_{task}_valid"] = True
        profiles = {f"i{index}": np.full(4, index, dtype=np.float32) for index in range(5)}
        config = ContextConfig(enabled=True)
        builder = ContextBuilder(rows, profiles, fit_token_scaler(rows, config), config)
        predictor = RecordingPredictor()
        _, diagnostics = predict_with_crossfit_context_calibration(
            predictor, "a", np.zeros(4), builder, np.zeros((1, 4)), profiles,
            calibration_lambda=1.0, folds=3, seed=7,
        )
        self.assertEqual(diagnostics.calibration_view_count, 3)
        self.assertEqual(len(predictor.calls), 6)
        for target_call, anchor_call in zip(predictor.calls[::2], predictor.calls[1::2]):
            self.assertEqual(target_call[0], anchor_call[0])
        self.assertEqual(len({call[0] for call in predictor.calls[::2]}), 3)


if __name__ == "__main__":
    unittest.main()
