from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contexts import ContextBuilder, ContextCache, ContextConfig, build_support_tokens, crossfit_partition, fit_token_scaler
from data import load_bidirectional_training_data, subset_training_data
from inference import CalibrationConfig, JointPredictor, predict_with_crossfit_context_calibration
from make_synthetic_data import prepare_synthetic
from model import JointResponseModel
import pipeline
from training import profile_maps


def setUpModule() -> None:
    warnings.simplefilter("ignore")


class ContextFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = prepare_synthetic(cls.temporary.name)
        cls.data = load_bidirectional_training_data(cls.paths["jobs"], cls.paths["inhibitors"], cls.paths["job_inh"])
        cls.config = ContextConfig(enabled=True)
        cls.scaler = pipeline.ProfileScaler("base").fit(pd.concat([cls.data.jobs, cls.data.inhibitors]))
        cls.job_profiles, cls.inhibitor_profiles = profile_maps(cls.data.jobs, cls.data.inhibitors, cls.scaler)
        cls.token_scaler = fit_token_scaler(cls.data.responses, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def with_replicates(self) -> pd.DataFrame:
        rows = self.data.responses[self.data.responses["job_id"] == "alpha"]
        replicate = rows.assign(replicate_id=1, observation_id=rows["observation_id"] + "r",
                                log_slowdown=rows["log_slowdown"] + 0.01)
        return pd.concat([rows, replicate], ignore_index=True)


class TokenTests(ContextFixture):
    def test_tokens_are_order_invariant_and_pool_replicates(self) -> None:
        rows = self.with_replicates()
        tokens = build_support_tokens(rows, self.inhibitor_profiles, self.token_scaler, self.config)
        shuffled = build_support_tokens(rows.sample(frac=1.0, random_state=4), self.inhibitor_profiles,
                                        self.token_scaler, self.config)
        np.testing.assert_array_equal(tokens.features, shuffled.features)
        self.assertEqual(tokens.count, rows["inhib_id"].nunique())
        self.assertEqual(len(tokens.observation_ids), len(rows))
        first = rows[rows["inhib_id"] == tokens.inhibitor_ids[0]]
        profile_dim = len(self.inhibitor_profiles[tokens.inhibitor_ids[0]])
        self.assertAlmostEqual(float(tokens.features[0, profile_dim]), float(first["log_slowdown"].mean()), places=5)

    def test_forward_only_tokens_drop_reverse_evidence(self) -> None:
        rows = self.data.responses[self.data.responses["job_id"] == "alpha"]
        forward = ContextConfig(enabled=True, token_directions="forward_only")
        both = build_support_tokens(rows, self.inhibitor_profiles, self.token_scaler, self.config)
        only = build_support_tokens(rows, self.inhibitor_profiles, fit_token_scaler(self.data.responses, forward), forward)
        self.assertEqual(both.features.shape[1] - only.features.shape[1], 6)
        mutated = rows.copy()
        mutated["inhib_bw_wall_log_degradation"] += 5.0
        changed = build_support_tokens(mutated, self.inhibitor_profiles,
                                       fit_token_scaler(self.data.responses, forward), forward)
        np.testing.assert_array_equal(only.features, changed.features)

    def test_query_inhibitor_excluded_across_all_replicates(self) -> None:
        rows = self.with_replicates()
        builder = ContextBuilder(rows, self.inhibitor_profiles, self.token_scaler, self.config)
        query = sorted(rows["inhib_id"].unique())[2]
        tokens = builder.application_context("alpha", exclude_inhib_ids=[query])
        self.assertNotIn(query, tokens.inhibitor_ids)
        self.assertFalse(any(value.split("|")[1] == query for value in tokens.observation_ids))
        with self.assertRaises(ValueError):
            builder.application_context("alpha", include_inhib_ids=[query], exclude_inhib_ids=[query])

    def test_mutating_query_label_does_not_change_its_context(self) -> None:
        rows = self.data.responses[self.data.responses["job_id"] == "alpha"].copy()
        query = sorted(rows["inhib_id"].unique())[0]
        before = ContextBuilder(rows, self.inhibitor_profiles, self.token_scaler, self.config).application_context(
            "alpha", exclude_inhib_ids=[query])
        rows.loc[rows["inhib_id"] == query, ["log_slowdown", "app_bw_wall_log_degradation"]] = 9.0
        after = ContextBuilder(rows, self.inhibitor_profiles, self.token_scaler, self.config).application_context(
            "alpha", exclude_inhib_ids=[query])
        np.testing.assert_array_equal(before.features, after.features)

    def test_empty_context(self) -> None:
        builder = ContextBuilder(self.data.responses[self.data.responses["job_id"] == "alpha"],
                                 self.inhibitor_profiles, self.token_scaler, self.config)
        tokens = builder.application_context("bravo")
        self.assertEqual(tokens.count, 0)
        self.assertEqual(tokens.features.shape[0], 0)

    def test_cache_key_names_every_identity_component(self) -> None:
        key = ContextCache.key(fold="f", application="a", support_inhib_ids=["j2", "j1"],
                               support_observation_ids=["o2", "o1"], normalization_state="n", model_identity="m")
        self.assertEqual(key, ("f", "a", ("j1", "j2"), ("o1", "o2"), "n", "m"))
        for field, value in [("fold", "g"), ("normalization_state", "x"), ("model_identity", "y")]:
            other = dict(fold="f", application="a", support_inhib_ids=["j1", "j2"],
                         support_observation_ids=["o1", "o2"], normalization_state="n", model_identity="m")
            other[field] = value
            self.assertNotEqual(key, ContextCache.key(**other))

    def test_crossfit_partition_is_deterministic_disjoint_and_complete(self) -> None:
        ids = [f"i{index}" for index in range(8)]
        groups = crossfit_partition(ids, "alpha", 3)
        self.assertEqual(groups, crossfit_partition(list(reversed(ids)), "alpha", 3))
        self.assertEqual(len(groups), 3)
        self.assertEqual(sorted(value for group in groups for value in group), ids)
        self.assertEqual(len(crossfit_partition(ids[:2], "alpha", 3)), 2)


class CrossfitTests(ContextFixture):
    def test_crossfit_context_and_calibration_sets_are_disjoint(self) -> None:
        torch.manual_seed(0)
        rows = self.data.responses[self.data.responses["job_id"] == "alpha"]
        builder = ContextBuilder(rows, self.inhibitor_profiles, self.token_scaler, self.config)
        width = builder.width
        predictor = JointPredictor(JointResponseModel(token_dim=width), torch.device("cpu"), width)
        latent, diagnostics = predict_with_crossfit_context_calibration(
            predictor, "alpha", self.job_profiles["alpha"], builder,
            np.stack([self.inhibitor_profiles[value] for value in rows["inhib_id"][:2]]), self.inhibitor_profiles,
            calibration_lambda=1.0, folds=3, seed=1701,
        )
        self.assertEqual(latent.shape, (2,))
        self.assertEqual(diagnostics.calibration_view_count, 3)
        calibration_union = set()
        for view in diagnostics.views:
            context_ids = set(filter(None, view["context_inhibitor_ids"].split("|")))
            calibration_ids = set(view["calibration_inhibitor_ids"].split("|"))
            self.assertFalse(context_ids & calibration_ids)
            self.assertEqual(context_ids | calibration_ids, set(rows["inhib_id"]))
            calibration_union |= calibration_ids
        self.assertEqual(calibration_union, set(rows["inhib_id"]))

    def test_single_inhibitor_uses_empty_fingerprint_and_calibrates_on_it(self) -> None:
        rows = self.data.responses[self.data.responses["job_id"] == "alpha"].iloc[:1]
        builder = ContextBuilder(rows, self.inhibitor_profiles, self.token_scaler, self.config)
        predictor = JointPredictor(JointResponseModel(token_dim=builder.width), torch.device("cpu"), builder.width)
        _, diagnostics = predict_with_crossfit_context_calibration(
            predictor, "alpha", self.job_profiles["alpha"], builder,
            self.inhibitor_profiles[rows["inhib_id"].iloc[0]][None, :], self.inhibitor_profiles,
            calibration_lambda=1.0, folds=3, seed=1, config=CalibrationConfig(),
        )
        self.assertEqual(diagnostics.calibration_view_count, 1)
        self.assertEqual(diagnostics.context_inhibitor_count, 0)
        self.assertEqual(diagnostics.calibration_inhibitor_count, 1)


class FoldLeakageTests(ContextFixture):
    def test_heldout_block_and_application_never_reach_training_inputs(self) -> None:
        config = pipeline.RunConfig(inhibitor_blocks=3)
        blocks = pipeline.build_inhibitor_blocks(self.data, config)
        block_by_id = dict(zip(blocks["inhib_id"], blocks["block"]))
        spec = pipeline.resolve_variant_specs(pipeline.RunConfig(variants=["bidirectional_context"]))[0]
        for fold in pipeline.crossed_folds(self.data, blocks):
            self.assertNotIn(fold.victim_id, set(fold.train_rows["job_id"]))
            self.assertFalse(fold.train_rows["inhib_id"].map(block_by_id).eq(fold.heldout_block).any())
            self.assertFalse(fold.support["inhib_id"].map(block_by_id).eq(fold.heldout_block).any())
            self.assertTrue(fold.queries["inhib_id"].map(block_by_id).eq(fold.heldout_block).all())
            runtime = pipeline.prepare_runtime(
                spec, pipeline.ModelConfig(), pipeline.RunConfig(variants=["bidirectional_context"]),
                train_rows=fold.train_rows,
                scaler_frame=pd.concat([fold.train_jobs, fold.retained_inhibitors]),
                jobs=self.data.jobs, inhibitors=self.data.inhibitors, device=torch.device("cpu"),
            )
            mutated = self.data.responses.copy()
            outside = (mutated["job_id"] == fold.victim_id) | mutated["inhib_id"].map(block_by_id).eq(fold.heldout_block)
            mutated.loc[outside, [c for c in mutated if c.endswith("_log_degradation")]] = 50.0
            mutated_rows = mutated[~outside]
            self.assertEqual(pipeline.AuxiliaryScaler(spec.aux_tasks).fit(mutated_rows).scales_,
                             runtime.aux_scaler.scales_)
            self.assertEqual(fit_token_scaler(mutated_rows, runtime.context_config).scales_,
                             runtime.token_scaler.scales_)
            builder = runtime.builder(fold.train_rows, "training")
            for job_id in builder.rows:
                ids, _ = builder.token_matrix(job_id)
                self.assertFalse(any(block_by_id[value] == fold.heldout_block for value in ids))
            break

    def test_unknown_applications_do_not_fit_scalers(self) -> None:
        training = subset_training_data(self.data, ["alpha", "bravo", "delta"])
        self.assertEqual(set(training.responses["job_id"]), {"alpha", "bravo", "delta"})
        scaler = pipeline.ProfileScaler("base").fit(pd.concat([training.jobs, training.inhibitors]))
        jobs = self.data.jobs.copy()
        jobs.loc[jobs["job_id"] == "echo", "mpi_time"] *= 100
        other = pipeline.ProfileScaler("base").fit(pd.concat([
            jobs[jobs["job_id"].isin(training.jobs["job_id"])], training.inhibitors]))
        np.testing.assert_array_equal(scaler.mean_, other.mean_)


if __name__ == "__main__":
    unittest.main()
