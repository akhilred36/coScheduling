from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from delta_response_2.config import ExperimentConfig, candidate_registry
from delta_response_2.data import ProfileScaler, app_profile_weights, load_training_data, profile_maps
from delta_response_2.frozen import predict_frozen_recipe
from delta_response_2.manifests import (
    create_plan,
    verify_completion,
    verify_plan,
)
from delta_response_2.model import LowRankPotential
from delta_response_2.pipeline import run_experiment
from delta_response_2.training import train_potential_fixed


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[2]


def synthetic_inputs(directory: Path) -> tuple[Path, Path, Path]:
    jobs = pd.DataFrame(
        {
            "job_id": ["a", "b", "c"],
            "mpi_time": [1.0, 2.0, 3.0],
            "comm_frac": [0.1, 0.2, 0.3],
            "total_msgs": [100.0, 200.0, 300.0],
            "total_bytes": [1000.0, 3000.0, 7000.0],
        }
    )
    inhibitors = pd.DataFrame(
        {
            "inhib_id": ["i0", "i1", "i2", "i3"],
            "msg_size": [8, 8, 16, 16],
            "mpi_time": [0.5, 1.0, 2.0, 4.0],
            "comm_frac": [0.05, 0.1, 0.2, 0.4],
            "total_msgs": [50.0, 100.0, 200.0, 400.0],
            "total_bytes": [500.0, 1200.0, 3000.0, 8000.0],
        }
    )
    response_rows = []
    for job_index, job_id in enumerate(jobs["job_id"]):
        for inhibitor_index, inhib_id in enumerate(inhibitors["inhib_id"]):
            value = 0.0 if (job_index + inhibitor_index) % 4 == 0 else 0.1 + 0.05 * (job_index + inhibitor_index)
            response_rows.append(
                {"job_id": job_id, "inhib_id": inhib_id, "slowdown": float(np.exp(value))}
            )
    responses = pd.DataFrame(response_rows)
    paths = (directory / "jobs.csv", directory / "inhibitors.csv", directory / "job_inh.csv")
    jobs.to_csv(paths[0], index=False)
    inhibitors.to_csv(paths[1], index=False)
    responses.to_csv(paths[2], index=False)
    return paths


class ProtocolTests(unittest.TestCase):
    def test_candidate_registry_has_required_ablations_and_no_app_app_calibrator(self) -> None:
        methods = [row["method"] for row in candidate_registry(ExperimentConfig())]
        self.assertIn("constant_1", methods)
        self.assertTrue(any(value.startswith("regularized_absolute_linear") for value in methods))
        self.assertTrue(any(value.startswith("censored_correction_delta_h") for value in methods))
        self.assertTrue(
            any(value.startswith("censored_correction_delta_with_query_shrinkage") for value in methods)
        )
        self.assertFalse(any("calibrated_delta_rank" in value for value in methods))

    def test_profile_weights_are_label_free_normalized_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs_path, inhibitors_path, responses_path = synthetic_inputs(Path(temporary))
            data = load_training_data(jobs_path, inhibitors_path, responses_path)
            first, state = app_profile_weights(data.jobs, data.inhibitors, nearest_per_application=2)
            changed = data.responses.copy()
            changed["log_slowdown"] = np.linspace(0, 100, len(changed))
            second, _ = app_profile_weights(data.jobs, data.inhibitors, nearest_per_application=2)
            np.testing.assert_allclose(first["profile_weight"], second["profile_weight"])
            self.assertAlmostEqual(float(first["profile_weight"].mean()), 1.0)
            self.assertTrue(np.isfinite(first["profile_weight"]).all())
            self.assertFalse(state["uses_outcome_labels"])
            self.assertTrue(state["declared_transductive"])

    def test_sealed_manifest_has_roles_scaler_exclusions_and_no_pair_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            jobs, inhibitors, responses = synthetic_inputs(temporary_path)
            root = temporary_path / "experiment"
            config = ExperimentConfig(
                inhibitor_blocks=2,
                fold_schemes=["profile"],
                seeds=[0],
                capacities=[[8, 4]],
                optimizer_steps=1,
                batch_size=4,
                checkpoint_interval=1,
                nearest_per_application=2,
                bootstrap_samples=5,
                device="cpu",
            )
            create_plan(
                root,
                config=config,
                jobs_csv=jobs,
                inhibitors_csv=inhibitors,
                job_inh_csv=responses,
                package_dir=PACKAGE,
                workspace=WORKSPACE,
            )
            verify_plan(root, PACKAGE)
            inputs = json.loads((root / "input_hashes.json").read_text())
            self.assertEqual(set(inputs), {"jobs_csv", "inhibitors_csv", "job_inh_csv"})
            environment = json.loads((root / "environment.json").read_text())
            self.assertEqual(set(environment["git"]), {"commit"})
            seal = json.loads((root / "seal.json").read_text())
            self.assertIn("rejected_job_inh.csv", seal["static_files"])
            self.assertIn("duplicate_job_inh_keys.csv", seal["static_files"])
            pair_manifest = pd.read_csv(root / "pair_split_manifest.csv")
            self.assertTrue(pair_manifest.empty)
            inhibitor_manifest = pd.read_csv(root / "inhibitor_manifest.csv")
            self.assertTrue(
                inhibitor_manifest.groupby("mechanism_family")["mechanism_block"]
                .nunique()
                .eq(1)
                .all()
            )
            folds = pd.read_csv(root / "fold_manifest.csv")
            held = folds[folds["role"].isin(["support", "query"])]
            self.assertFalse(held["model_job_profile_in_scaler"].astype(bool).any())
            queries = folds[folds["role"] == "query"]
            self.assertFalse(
                queries["model_inhibitor_profile_in_scaler"].astype(bool).any()
            )
            self.assertFalse(
                queries["distance_inhibitor_profile_in_scaler"].astype(bool).any()
            )
            self.assertTrue(
                queries["clustering_inhibitor_profile_in_scaler"].astype(bool).all()
            )
            self.assertEqual(set(folds["role"]), {"query", "support", "parameter_train", "excluded_held_block"})
            with (root / "candidate_manifest.csv").open("a") as handle:
                handle.write("tamper\n")
            with self.assertRaises(RuntimeError):
                verify_plan(root, PACKAGE)

    def test_fixed_step_training_runs_exact_budget_and_saves_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs_path, inhibitors_path, responses_path = synthetic_inputs(Path(temporary))
            data = load_training_data(jobs_path, inhibitors_path, responses_path)
            scaler = ProfileScaler("base").fit(pd.concat([data.jobs, data.inhibitors]))
            jobs, inhibitors = profile_maps(data.jobs, data.inhibitors, scaler)
            model = LowRankPotential(4, 8, 4, 2)
            result = train_potential_fixed(
                model,
                data.responses,
                jobs,
                inhibitors,
                seed=0,
                optimizer_steps=3,
                batch_size=8,
                learning_rate=0.001,
                weight_decay=0.0,
                checkpoint_interval=2,
                device=torch.device("cpu"),
                validation_fn=lambda _: {"validation_log_mae": 1.0},
            )
            self.assertEqual(result["optimizer_steps"], 3)
            self.assertEqual(result["sampled_examples"], 24)
            self.assertEqual(result["history"][-1]["step"], 3)
            self.assertGreater(result["informative_examples"], 0)

    def test_outer_query_labels_do_not_change_fold_models_or_ood(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first_inputs = base / "first_inputs"
            second_inputs = base / "second_inputs"
            first_inputs.mkdir()
            second_inputs.mkdir()
            first_jobs, first_inhibitors, first_responses = synthetic_inputs(first_inputs)
            second_jobs, second_inhibitors, second_responses = synthetic_inputs(second_inputs)
            changed = pd.read_csv(second_responses)
            changed.loc[
                changed["job_id"].eq("a") & changed["inhib_id"].eq("i0"),
                "slowdown",
            ] = float(np.exp(1.5))
            changed.to_csv(second_responses, index=False)
            config = ExperimentConfig(
                inhibitor_blocks=2,
                fold_schemes=["profile"],
                seeds=[0],
                capacities=[[8, 4]],
                optimizer_steps=1,
                batch_size=4,
                checkpoint_interval=1,
                nearest_per_application=2,
                bootstrap_samples=5,
                tree_estimators=2,
                device="cpu",
            )
            first_root = base / "first_experiment"
            second_root = base / "second_experiment"
            for root, jobs, inhibitors, responses in [
                (first_root, first_jobs, first_inhibitors, first_responses),
                (second_root, second_jobs, second_inhibitors, second_responses),
            ]:
                create_plan(
                    root,
                    config=config,
                    jobs_csv=jobs,
                    inhibitors_csv=inhibitors,
                    job_inh_csv=responses,
                    package_dir=PACKAGE,
                    workspace=WORKSPACE,
                )
                run_experiment(root, PACKAGE)

            manifest = pd.read_csv(first_root / "inhibitor_manifest.csv")
            heldout_block = int(
                manifest.loc[manifest["inhib_id"].eq("i0"), "profile_block"].iloc[0]
            )
            task_id = f"profile__a__b{heldout_block}__s0.csv"
            first_predictions = pd.read_csv(
                first_root / "tasks" / "fold_shards" / task_id
            ).sort_values(["candidate_id", "response_id"])
            second_predictions = pd.read_csv(
                second_root / "tasks" / "fold_shards" / task_id
            ).sort_values(["candidate_id", "response_id"])
            self.assertFalse(
                np.allclose(
                    first_predictions["true_log_slowdown"],
                    second_predictions["true_log_slowdown"],
                )
            )
            np.testing.assert_allclose(
                first_predictions["latent_predicted_log_slowdown"],
                second_predictions["latent_predicted_log_slowdown"],
                atol=1e-10,
            )
            first_ood = pd.read_csv(first_root / "tasks" / "fold_ood_calibrations.csv")
            second_ood = pd.read_csv(second_root / "tasks" / "fold_ood_calibrations.csv")
            first_ood = first_ood[
                first_ood["victim_id"].eq("a")
                & first_ood["heldout_block"].eq(heldout_block)
            ]
            second_ood = second_ood[
                second_ood["victim_id"].eq("a")
                & second_ood["heldout_block"].eq(heldout_block)
            ]
            np.testing.assert_allclose(
                first_ood[["threshold", "width", "reference_count"]],
                second_ood[["threshold", "width", "reference_count"]],
            )

    def test_tiny_end_to_end_freeze_replay_and_completion_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            jobs, inhibitors, responses = synthetic_inputs(temporary_path)
            root = temporary_path / "experiment"
            config = ExperimentConfig(
                inhibitor_blocks=2,
                fold_schemes=["profile"],
                seeds=[0],
                capacities=[[8, 4]],
                optimizer_steps=1,
                batch_size=4,
                checkpoint_interval=1,
                nearest_per_application=2,
                bootstrap_samples=5,
                tree_estimators=2,
                device="cpu",
            )
            create_plan(
                root,
                config=config,
                jobs_csv=jobs,
                inhibitors_csv=inhibitors,
                job_inh_csv=responses,
                package_dir=PACKAGE,
                workspace=WORKSPACE,
            )
            run_experiment(root, PACKAGE)
            verify_completion(root)
            prediction = predict_frozen_recipe(root, "a", "b", device="cpu")
            self.assertGreaterEqual(prediction["predicted_slowdown"], 1.0)
            self.assertFalse(prediction["app_app_outcomes_used"])
            original_responses = responses.read_bytes()
            responses.write_bytes(original_responses + b"\n")
            with self.assertRaises(RuntimeError):
                predict_frozen_recipe(root, "a", "b", device="cpu")
            responses.write_bytes(original_responses)
            checkpoint_dir = root / "tasks" / "final_checkpoints"
            extra_checkpoint = checkpoint_dir / "delta_seed999.pt"
            extra_checkpoint.write_bytes(b"not sealed")
            with self.assertRaises(RuntimeError):
                predict_frozen_recipe(root, "a", "b", device="cpu")
            extra_checkpoint.unlink()
            run_experiment(root, PACKAGE)
            report = root / "FINAL_REPORT.md"
            report.write_text(report.read_text() + "tamper\n")
            with self.assertRaises(RuntimeError):
                verify_completion(root)

    def test_source_never_imports_inherited_runtime(self) -> None:
        for path in PACKAGE.glob("*.py"):
            source = path.read_text()
            self.assertNotIn("import delta_response_1", source)
            self.assertNotIn("from delta_response_1", source)


if __name__ == "__main__":
    unittest.main()
