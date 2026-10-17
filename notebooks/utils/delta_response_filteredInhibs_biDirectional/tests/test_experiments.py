from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_data import prepare_synthetic
import run_experiments


def setUpModule() -> None:
    warnings.simplefilter("ignore")


def execution_args(root: Path, **overrides) -> argparse.Namespace:
    values = dict(root=root, max_workers=2, max_tasks=None, progress_interval=30.0, resume=True,
                  include_predictions=True)
    values.update(overrides)
    return argparse.Namespace(**values)


class PlanTests(unittest.TestCase):
    def test_task_identity_includes_context_and_calibration_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = prepare_synthetic(root / "synthetic")
            common = (paths["jobs"], paths["inhibitors"], paths["job_inh"], paths["pair"])
            first = run_experiments.build_plan(root, "smoke", *common)
            second = run_experiments.build_plan(root, "smoke", *common, context={"max_support": 3})
            third = run_experiments.build_plan(root, "smoke", *common, calibration={"calibration_lambda": 10.0,
                                                                                     "select_lambda": False})
            self.assertNotEqual(first[0]["task_id"], second[0]["task_id"])
            self.assertNotEqual(first[0]["task_id"], third[0]["task_id"])
            self.assertEqual([task["stage"] for task in first],
                             ["tune_global", "evaluate_random_split", "evaluate_training_size"])
            self.assertIn("training_preparation_report", first[0]["inputs"])
            self.assertNotIn("pair", first[0]["inputs"])
            self.assertIn("evaluation_preparation_report", first[1]["inputs"])
            self.assertIn("bidirectional_context_forward", first[0]["config"]["variants"])
            self.assertTrue(all(task["depends_on"] == first[0]["task_id"] for task in first[1:]))


class RunnerTests(unittest.TestCase):
    def test_smoke_experiment_runs_resumes_and_consolidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            paths = prepare_synthetic(Path(directory) / "synthetic")
            with redirect_stdout(io.StringIO()):
                run_experiments.plan_command(argparse.Namespace(
                    root=root, profile="smoke", jobs_csv=paths["jobs"], inhibitors_csv=paths["inhibitors"],
                    job_inh_csv=paths["job_inh"], pair_csv=paths["pair"], random_split_only=False,
                    variants=None, frozen_panel_csv=None, context_max_support=None,
                    context_crossfit_folds=None, calibration_lambda=None, inhibitor_blocks=2,
                ))
                run_experiments.tune_command(execution_args(root))
                run_experiments.evaluate_command(execution_args(root))
            tasks = json.loads((root / "plan.json").read_text())
            manifest = pd.read_csv(root / "experiment_manifest.csv")
            self.assertTrue((manifest["status"] == "succeeded").all(), manifest[["stage", "status"]])
            evaluation = next(task for task in tasks if task["stage"] == "evaluate_random_split")
            output = run_experiments.successful_output(root, evaluation["task_id"])
            resolved = json.loads((output.parent / "resolved_config.json").read_text())
            self.assertEqual(resolved["selection_mode"], "fixed_global")
            self.assertIn("bidirectional_context", resolved["fixed_calibration_lambda_by_variant"])
            first_attempts = {task["task_id"]: run_experiments.latest_status(root, task["task_id"])["attempt"]
                              for task in tasks}
            capture = io.StringIO()
            with redirect_stdout(capture):
                run_experiments.evaluate_command(execution_args(root))
            self.assertIn("skipped", capture.getvalue())
            self.assertEqual(first_attempts, {task["task_id"]: run_experiments.latest_status(root, task["task_id"])["attempt"]
                                              for task in tasks})
            consolidated = root / "consolidated"
            for name in ["global_selection.json", "metrics_long.csv", "accuracy_by_eval_method.csv",
                         "accuracy_direct.csv", "accuracy_calibrated.csv", "ablation_summary.csv",
                         "pair_metrics.csv", "matched_fold_comparisons.csv", "predictions.csv",
                         "learning_curve_summary.csv"]:
                self.assertTrue((consolidated / name).exists(), name)
            ablation = pd.read_csv(consolidated / "ablation_summary.csv")
            self.assertEqual(set(ablation["result_scope"]), {"app_inhibitor_crossed_validation", "app_app"})
            predictions = pd.read_csv(consolidated / "predictions.csv")
            self.assertIn("support_information_regime", predictions)
            self.assertEqual(set(predictions["panel_source"]), {"recomputed"})
            (Path(paths["job_inh"])).write_text(Path(paths["job_inh"]).read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                run_experiments.execution_fingerprint(root, tasks[0], tasks[0]["config"])


if __name__ == "__main__":
    unittest.main()
