from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from inference import JointPredictor, predict_pair_direct
from model import JointResponseModel, LowRankPotential, SupportSetEncoder


def context(batch: int, tokens: int, width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[0, tokens - 1:] = False
    return torch.randn(batch, tokens, width, generator=generator), mask


class ModelInvariantTests(unittest.TestCase):
    def test_legacy_delta_invariants_are_exact(self) -> None:
        torch.manual_seed(2)
        model = LowRankPotential()
        victim, left, right, middle = (torch.randn(5, 4) for _ in range(4))
        self.assertTrue(torch.equal(model(victim, left, left), torch.zeros(5)))
        self.assertTrue(torch.equal(model(victim, left, right), -model(victim, right, left)))
        torch.testing.assert_close(model(victim, left, right),
                                   model(victim, left, middle) + model(victim, middle, right), rtol=1e-6, atol=1e-6)

    def test_joint_delta_invariants_hold_for_fixed_contexts(self) -> None:
        torch.manual_seed(3)
        model = JointResponseModel(aux_tasks=("bw_wall", "bw_mpi"), token_dim=6)
        victim, left, right, middle = (torch.randn(4, 4) for _ in range(4))
        victim_context = context(4, 3, 6, 0)
        identity = model.delta(victim, left, left, victim_context)
        self.assertTrue(torch.equal(identity, torch.zeros_like(identity)))
        forward = model.delta(victim, left, right, victim_context)
        torch.testing.assert_close(forward, -model.delta(victim, right, left, victim_context), rtol=0, atol=1e-7)
        path = model.delta(victim, left, middle, victim_context) + model.delta(victim, middle, right, victim_context)
        torch.testing.assert_close(forward, path, rtol=1e-6, atol=1e-6)
        expected = model.response(victim, right, victim_context) - model.response(victim, left, victim_context)
        torch.testing.assert_close(forward, expected, rtol=0, atol=1e-7)

    def test_empty_context_contributes_exactly_zero(self) -> None:
        torch.manual_seed(4)
        model = JointResponseModel(token_dim=5)
        profiles = torch.randn(3, 4)
        empty = (torch.zeros(3, 0, 5), torch.zeros(3, 0, dtype=torch.bool))
        masked = (torch.randn(3, 2, 5), torch.zeros(3, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(model.embed(profiles, empty), model.embed(profiles)))
        self.assertTrue(torch.equal(model.embed(profiles, masked), model.embed(profiles)))
        present = (torch.randn(3, 2, 5), torch.ones(3, 2, dtype=torch.bool))
        self.assertFalse(torch.equal(model.embed(profiles, present), model.embed(profiles)))

    def test_set_encoder_is_permutation_invariant(self) -> None:
        torch.manual_seed(5)
        encoder = SupportSetEncoder(7)
        tokens, mask = torch.randn(2, 5, 7), torch.ones(2, 5, dtype=torch.bool)
        order = torch.tensor([3, 0, 4, 1, 2])
        pooled, count = encoder(tokens, mask)
        shuffled, _ = encoder(tokens[:, order], mask[:, order])
        torch.testing.assert_close(pooled, shuffled, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(count, torch.tensor([5.0, 5.0])))

    def test_swap_equivariance_and_asymmetry(self) -> None:
        torch.manual_seed(6)
        model = JointResponseModel(aux_tasks=("bw_wall", "bw_mpi", "msg_wall"), token_dim=6)
        predictor = JointPredictor(model, torch.device("cpu"), token_width=6)
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=4).astype(np.float32), rng.normal(size=4).astype(np.float32)
        tokens_a, tokens_b = rng.normal(size=(2, 6)).astype(np.float32), None
        forward = predict_pair_direct(predictor, a, tokens_a, b, tokens_b)
        swapped = predict_pair_direct(predictor, b, tokens_b, a, tokens_a)
        self.assertAlmostEqual(forward["latent_a"], swapped["latent_b"], places=6)
        self.assertAlmostEqual(forward["latent_b"], swapped["latent_a"], places=6)
        np.testing.assert_allclose(forward["aux_a"], swapped["aux_b"], atol=1e-6)
        np.testing.assert_allclose(forward["aux_b"], swapped["aux_a"], atol=1e-6)
        self.assertNotAlmostEqual(forward["latent_a"], forward["latent_b"], places=4)
        self.assertFalse(np.allclose(forward["aux_a"], forward["aux_b"]))
        reverse = predictor.bandwidth(a, tokens_a, b[None, :], [tokens_b], reverse=True)[0]
        np.testing.assert_allclose(reverse, forward["aux_b"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
