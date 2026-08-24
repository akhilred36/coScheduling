from __future__ import annotations

import unittest

import numpy as np
import torch

from delta_response_2.censoring import fit_censored_intercept, gauge_center
from delta_response_2.inference import censored_potential_predictions
from delta_response_2.model import LowRankPotential


class CensoredCorrectionTests(unittest.TestCase):
    def test_all_exact_matches_quadratic_huber_mean(self) -> None:
        observed = np.asarray([0.2, 0.4, 0.6])
        potentials = np.asarray([0.1, 0.1, 0.1])
        result = fit_censored_intercept(observed, potentials)
        self.assertAlmostEqual(result.correction, 0.3, places=8)
        self.assertEqual(result.exact_count, 3)
        self.assertEqual(result.floor_count, 0)

    def test_mixed_anchors_use_floor_inequalities(self) -> None:
        exact_only = fit_censored_intercept(np.asarray([0.5]), np.asarray([0.0]))
        mixed = fit_censored_intercept(
            np.asarray([0.5, 0.0, 0.0]), np.asarray([0.0, 0.2, 0.4])
        )
        self.assertLess(mixed.correction, exact_only.correction)
        self.assertEqual(mixed.floor_count, 2)

    def test_mostly_censored_and_all_floor_are_finite(self) -> None:
        mostly = fit_censored_intercept(
            np.asarray([0.2, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.1, -0.1, 0.2, 0.3]),
            shrinkage=10.0,
        )
        self.assertTrue(np.isfinite(mostly.correction))
        with self.assertRaises(ValueError):
            fit_censored_intercept(np.zeros(3), np.zeros(3))
        all_floor = fit_censored_intercept(
            np.zeros(3), np.zeros(3), shrinkage=1.0
        )
        self.assertTrue(np.isfinite(all_floor.correction))
        self.assertLessEqual(all_floor.correction, 1e-8)

    def test_adding_floor_anchors_cannot_raise_unshrunk_correction(self) -> None:
        observed = np.asarray([0.3, 0.6])
        potentials = np.asarray([0.0, 0.1])
        original = fit_censored_intercept(observed, potentials)
        augmented = fit_censored_intercept(
            np.concatenate([observed, [0.0, 0.0]]),
            np.concatenate([potentials, [0.2, 0.5]]),
        )
        self.assertLessEqual(augmented.correction, original.correction + 1e-9)

    def test_adding_floor_anchors_cannot_raise_shrunk_correction(self) -> None:
        original = fit_censored_intercept(
            np.asarray([0.3, 0.6]),
            np.asarray([0.0, 0.1]),
            shrinkage=10.0,
            prior=0.2,
        )
        augmented = fit_censored_intercept(
            np.asarray([0.3, 0.6, 0.0, 0.0]),
            np.asarray([0.0, 0.1, 0.2, 0.5]),
            shrinkage=10.0,
            prior=0.2,
        )
        self.assertLessEqual(augmented.correction, original.correction + 1e-9)

    def test_gauge_translation_does_not_change_prediction(self) -> None:
        observed = np.asarray([0.3, 0.0, 0.5])
        support = np.asarray([-0.2, 0.1, 0.4])
        query = np.asarray([0.25, -0.1])
        first, _, _ = censored_potential_predictions(
            observed, support, query, shrinkage=10.0, prior=0.2
        )
        second, _, _ = censored_potential_predictions(
            observed, support + 7.0, query + 7.0, shrinkage=10.0, prior=0.2
        )
        np.testing.assert_allclose(first, second, atol=1e-10)
        centered, center = gauge_center(support)
        self.assertAlmostEqual(float(centered.mean()), 0.0, places=12)
        self.assertAlmostEqual(center, float(support.mean()))

    def test_support_correction_is_independent_of_query_potential(self) -> None:
        support_observed = np.asarray([0.0, 0.2, 0.5])
        support_potential = np.asarray([-0.1, 0.1, 0.3])
        query_potential = np.asarray([0.2, 0.8])
        first, correction, _ = censored_potential_predictions(
            support_observed,
            support_potential,
            query_potential,
            shrinkage=10.0,
            prior=0.1,
        )
        changed_query_potential = np.asarray([-20.0, 30.0])
        second, repeated, _ = censored_potential_predictions(
            support_observed,
            support_potential,
            changed_query_potential,
            shrinkage=10.0,
            prior=0.1,
        )
        self.assertEqual(correction.correction, repeated.correction)
        np.testing.assert_allclose(
            second - first, changed_query_potential - query_potential
        )


class PotentialInvariantTests(unittest.TestCase):
    def test_identity_antisymmetry_and_path_consistency(self) -> None:
        torch.manual_seed(4)
        model = LowRankPotential(4, 8, 4, 2)
        victim = torch.randn(6, 4)
        left = torch.randn(6, 4)
        middle = torch.randn(6, 4)
        right = torch.randn(6, 4)
        self.assertTrue(torch.equal(model(victim, left, left), torch.zeros(6)))
        torch.testing.assert_close(
            model(victim, left, right), -model(victim, right, left)
        )
        torch.testing.assert_close(
            model(victim, left, middle) + model(victim, middle, right),
            model(victim, left, right),
        )


if __name__ == "__main__":
    unittest.main()
