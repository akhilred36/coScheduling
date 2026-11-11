from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_bidirectional_training_data
from make_synthetic_data import prepare_synthetic
import pipeline


TINY = ["--cv-seeds", "0", "--final-seeds", "0", "1", "--inhibitor-blocks", "2", "--max-epochs", "3",
        "--patience", "2", "--batches-per-epoch", "1", "--bootstrap-samples", "20"]
ALL_VARIANTS = [*pipeline.DEFAULT_VARIANTS, *pipeline.ABLATION_VARIANTS]


def setUpModule() -> None:
    warnings.simplefilter("ignore")


def run_pipeline(paths: dict[str, Path], output: Path, *extra: str) -> None:
    arguments = ["--jobs-csv", str(paths["jobs"]), "--inhibitors-csv", str(paths["inhibitors"]),
                 "--job-inh-csv", str(paths["job_inh"]), "--pair-csv", str(paths["pair"]),
                 "--output-dir", str(output), *TINY, *extra]
    with redirect_stdout(io.StringIO()):
        pipeline.run(pipeline.parse_args(arguments))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.paths = prepare_synthetic(cls.root / "synthetic")
        cls.full = cls.root / "full"
        original = pipeline.load_pair_holdout
        cls.checkpoints_at_open = []

        def guarded(path, jobs):
            cls.checkpoints_at_open.extend(sorted((cls.full / "checkpoints").glob("*.pt")))
            return original(path, jobs)

        with patch.object(pipeline, "load_pair_holdout", guarded):
            run_pipeline(cls.paths, cls.full, "--variants", *ALL_VARIANTS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_holdout_free_fit_never_opens_app_app_data(self) -> None:
        output = self.root / "skip"
        with patch.object(pipeline, "load_pair_holdout", side_effect=AssertionError("opened")):
            run_pipeline(self.paths, output, "--skip-holdout", "--variants", "joint_profile", "legacy_absolute")
        report = json.loads((output / "run_report.json").read_text())
        self.assertEqual(report["holdout_evaluation"], "skipped")
        self.assertFalse((output / "directional_predictions.csv").exists())
        self.assertTrue((output / "checkpoints" / "joint_profile_seed0.pt").exists())

    def test_app_app_opened_only_after_every_checkpoint_was_written(self) -> None:
        self.assertEqual(len(self.checkpoints_at_open), 2 * len(ALL_VARIANTS))

    def test_full_evaluation_outputs_all_methods(self) -> None:
        predictions = pd.read_csv(self.full / "directional_predictions.csv")
        expected = {"constant_1"}
        for name in ALL_VARIANTS:
            expected |= set(pipeline.inference_methods(pipeline.VARIANT_REGISTRY[name]))
        self.assertEqual(set(predictions["method"]), expected)
        coverage = pd.read_csv(self.full / "method_coverage.csv").set_index("method")
        legacy = coverage.index.str.startswith(("legacy_delta", "legacy_generic"))
        self.assertTrue((coverage.loc[~legacy, "missing_pair_clusters"] == 0).all())
        self.assertTrue((coverage.loc[legacy, "missing_pair_clusters"] > 0).all())
        # The all-floor application has no uncensored anchor for legacy aggregation.
        charlie = predictions[(predictions["victim_id"] == "charlie")]
        self.assertFalse(charlie["method"].str.startswith("legacy_delta").any())
        self.assertTrue(charlie["method"].eq("joint_profile_calibrated").any())
        available = pd.read_csv(self.full / "metrics_non_self_all_available.csv").set_index("method")
        self.assertGreater(available.loc["joint_profile_calibrated", "n"], available.loc["legacy_delta_uniform", "n"])
        for name in ["auxiliary_validation_predictions.csv", "auxiliary_validation_metrics.csv",
                     "context_support_manifest.csv", "calibration_diagnostics.csv", "pair_predictions.csv",
                     "pair_metrics.csv", "metrics_by_censoring.csv", "matched_fold_comparisons.csv",
                     "ablation_summary.csv", "auxiliary_predictions.csv", "training_history.csv"]:
            self.assertTrue((self.full / name).exists(), name)
        joint = predictions[predictions["method"].str.startswith("bidirectional_context")]
        self.assertTrue(joint["support_information_regime"].str.contains("fingerprint").all())
        pairs = pd.read_csv(self.full / "pair_predictions.csv")
        direct = pairs[pairs["method"].str.endswith("_direct")]
        self.assertLess(float(direct["swap_equivariance_residual"].max()), 1e-5)
        self.assertEqual(set(pairs["pair_scope"]), {"non_self", "self"})
        auxiliary = pd.read_csv(self.full / "auxiliary_predictions.csv")
        self.assertEqual(set(auxiliary["direction"]), {"A", "B"})
        self.assertFalse(auxiliary["valid"].all())
        ablation = pd.read_csv(self.full / "ablation_summary.csv")
        self.assertIn("forward_only_context_tokens_direct", set(ablation["contrast"]))
        self.assertIn("wall_clock_only_auxiliary_direct", set(ablation["contrast"]))
        self.assertEqual(set(ablation["scope"]), {"app_inhibitor_crossed_validation", "app_app_random_split_non_self"})

    def test_all_floor_support_is_validated_by_joint_models(self) -> None:
        folds = pd.read_csv(self.full / "crossed_validation_folds.csv")
        charlie = folds[folds["victim_id"] == "charlie"]
        self.assertTrue((charlie[charlie["variant"] == "joint_profile"]["uncensored_support_count"] == 0).any())
        skipped = pd.read_csv(self.full / "skipped_validation_folds.csv")
        self.assertTrue(set(skipped["variant"]) <= {"legacy_delta", "legacy_generic"})
        self.assertIn("charlie", set(skipped["victim_id"]))
        predictions = pd.read_csv(self.full / "selected_validation_predictions.csv")
        calibrated = predictions[(predictions["victim_id"] == "charlie")
                                 & (predictions["inference_method"] == "joint_profile_calibrated")]
        self.assertFalse(calibrated.empty)
        self.assertTrue(np.isfinite(calibrated["calibration_offset"]).all())

    def test_context_manifest_never_overlaps_queries(self) -> None:
        manifest = pd.read_csv(self.full / "context_support_manifest.csv")
        self.assertTrue((manifest["context_query_overlap"] == 0).all())

    def test_checkpoint_reload_reproduces_predictions_and_rejects_bad_schema(self) -> None:
        data = load_bidirectional_training_data(self.full / "selected_inhibitors" / "jobs.csv",
                                                self.full / "selected_inhibitors" / "inhibitors.csv",
                                                self.full / "selected_inhibitors" / "job_inh.csv")
        seeds = pd.read_csv(self.full / "seed_predictions.csv")
        for variant in ["bidirectional_context", "legacy_absolute"]:
            runtime, payload = pipeline.load_checkpoint(self.full / "checkpoints" / f"{variant}_seed0.pt",
                                                        data.jobs, data.inhibitors, torch.device("cpu"))
            self.assertEqual(payload["selected_inhibitor_ids"], data.inhibitors["inhib_id"].tolist())
            self.assertIn("preparation_report", payload["data_hashes"])
            method = f"{variant}_direct" if variant.startswith("bidirectional") else variant
            row = seeds[(seeds["method"] == method) & (seeds["seed"] == 0)].iloc[0]
            target = runtime.job_profiles[row["aggressor_id"]][None, :]
            if runtime.spec.is_joint:
                builder = runtime.builder(data.responses, "reload")
                victim = builder.application_context(row["victim_id"]).features
                aggressor = builder.application_context(row["aggressor_id"]).features
                value = runtime.predictor.response(runtime.job_profiles[row["victim_id"]], victim, target, [aggressor])[0]
            else:
                value = pipeline.absolute_latents(runtime, row["victim_id"], target)[0]
            self.assertAlmostEqual(float(value), float(row["latent_predicted_log_slowdown"]), places=5)
        payload = torch.load(self.full / "checkpoints" / "joint_profile_seed0.pt", weights_only=False)
        payload["schema_version"] = 999
        bad = self.root / "bad.pt"
        torch.save(payload, bad)
        with self.assertRaisesRegex(ValueError, "Incompatible checkpoint schema"):
            pipeline.load_checkpoint(bad, data.jobs, data.inhibitors, torch.device("cpu"))

    def test_restricted_tiers_and_fixed_global_replay(self) -> None:
        selection = json.loads((self.full / "selection.json").read_text())
        variants = ["legacy_delta", "joint_profile", "bidirectional_context"]
        config = {
            "selection_mode": "fixed_global", "variants": variants,
            "fixed_final_epochs_by_variant": {name: selection["variants"][name]["final_epochs"] for name in variants},
            "fixed_model_configs_by_variant": {name: selection["variants"][name]["model_config"] for name in variants},
            "fixed_aggregations_by_variant": {"legacy_delta": selection["variants"]["legacy_delta"]["aggregation"]},
            "fixed_calibration_lambda_by_variant": {
                name: selection["variants"][name]["calibration_lambda"] for name in ["joint_profile", "bidirectional_context"]},
            "fixed_selection_source": {"selection_scope": "synthetic test"},
        }
        config_path = self.root / "fixed.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "restricted"
        run_pipeline(self.paths, output, "--config", str(config_path), "--eval-method", "both",
                     "--training-apps", "alpha,charlie,delta,echo")
        report = json.loads((output / "run_report.json").read_text())
        self.assertEqual(report["selection"]["selection_mode"], "fixed_global")
        self.assertFalse((output / "crossed_validation_predictions.csv").exists())
        summary = json.loads((output / "data_validation_summary.json").read_text())
        self.assertEqual(summary["unknown_applications"], ["bravo", "foxtrot"])
        payload = torch.load(output / "checkpoints" / "bidirectional_context_seed0.pt", weights_only=False)
        self.assertEqual(sorted(payload["training_apps"]), ["alpha", "charlie", "delta", "echo"])
        self.assertTrue(set(payload["context_token_scaler"]["counts"].values()))
        for tier in ["one_known", "zero_shot"]:
            predictions = pd.read_csv(output / "evaluations" / tier / "directional_predictions.csv")
            unknown = predictions[~predictions["victim_known"]]
            self.assertFalse(unknown.empty)
            self.assertTrue((unknown["victim_support_source"] == "unknown_application_inference_support").all())
        zero = pd.read_csv(output / "evaluations" / "zero_shot" / "directional_predictions.csv")
        self.assertTrue((zero["known_endpoint_count"] == 0).all())
        selected = pd.read_csv(output / "selected_inhibitors" / "application_coverage.csv")
        self.assertEqual(sorted(selected["job_id"]), ["alpha", "charlie", "delta", "echo"])


if __name__ == "__main__":
    unittest.main()
