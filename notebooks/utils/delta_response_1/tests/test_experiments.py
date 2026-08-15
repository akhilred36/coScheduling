from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = MODULE_DIR / "audit_outputs"
sys.path.insert(0, str(MODULE_DIR))

import run_experiments


class SyntheticExperimentTests(unittest.TestCase):
    def test_synthetic_generation_is_reproducible_and_valid(self) -> None:
        scenario = {
            **run_experiments.BASE_SCENARIO,
            "n_jobs": 4,
            "n_inhibitors": 12,
            "censor_fraction": 0.5,
            "missing_edge_fraction": 0.25,
            "replicates_per_edge": 3,
        }
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_summary = run_experiments.generate_synthetic_data(scenario, 41, first)
            second_summary = run_experiments.generate_synthetic_data(scenario, 41, second)
            self.assertEqual(first_summary["file_sha256"], second_summary["file_sha256"])
            jobs = pd.read_csv(first / "jobs.csv")
            inhibitors = pd.read_csv(first / "inhibitors.csv")
            responses = pd.read_csv(first / "job_inh.csv")
            pairs = pd.read_csv(first / "pair.csv")
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len(inhibitors), 12)
            self.assertEqual(len(pairs), 10)
            self.assertTrue((responses["slowdown"] >= 1.0).all())
            self.assertGreater(first_summary["replicate_rows"], 0)

    def test_response_only_stresses_do_not_change_pair_truth(self) -> None:
        baseline = {**run_experiments.BASE_SCENARIO, "n_jobs": 4, "n_inhibitors": 12}
        stressed = {
            **baseline,
            "missing_edge_fraction": 0.25,
            "replicates_per_edge": 3,
            "outlier_fraction": 0.05,
        }
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            run_experiments.generate_synthetic_data(baseline, 53, first)
            run_experiments.generate_synthetic_data(stressed, 53, second)
            self.assertEqual(
                run_experiments.sha256_file(first / "pair.csv"),
                run_experiments.sha256_file(second / "pair.csv"),
            )

    def test_nested_orders_are_balanced_prefixes(self) -> None:
        jobs = pd.DataFrame(
            {
                "job_id": [f"j{index}" for index in range(5)],
                "mpi_time": [1, 2, 3, 4, 5],
                "comm_frac": [0.1, 0.2, 0.3, 0.4, 0.5],
                "total_msgs": [10, 20, 30, 40, 50],
                "total_bytes": [100, 200, 300, 400, 500],
            }
        )
        orders = run_experiments.nested_orders(
            jobs, count=5, seed=7, strategy="balanced_random"
        )
        self.assertEqual(len(orders), 5)
        self.assertTrue(all(len(set(order)) == 5 for order in orders))
        for size in range(1, 6):
            counts = {
                app_id: sum(app_id in order[:size] for order in orders)
                for app_id in jobs["job_id"]
            }
            self.assertEqual(set(counts.values()), {size})

    def test_smoke_learning_curve_plan_uses_synthetic_both_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            tasks = run_experiments.build_plan(
                root, "smoke", {"learning_curve"}
            )
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task["config"]["evaluation_method"], "both")
            self.assertEqual(task["training_app_count"], 2)
            self.assertEqual(task["data_kind"], "synthetic")
            self.assertIn("pair", task["inputs"])
            membership = pd.read_csv(root / "subset_membership.csv")
            self.assertEqual(int(membership["in_training"].sum()), 2)

    def test_run_command_reports_progress_for_long_task(self) -> None:
        task = {
            "task_id": "run_slow",
            "stage": "test_stage",
            "depends_on": None,
        }
        args = argparse.Namespace(
            root=None,
            stages=None,
            max_tasks=None,
            max_workers=1,
            progress_interval=0.01,
            resume=True,
            include_predictions=False,
        )
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            args.root = root
            run_experiments.write_json(root / "plan.json", [task])

            def slow_task(*_args: object) -> dict[str, str]:
                time.sleep(0.04)
                return {"task_id": "run_slow", "status": "succeeded"}

            output = io.StringIO()
            with (
                patch.object(run_experiments, "execute_task", side_effect=slow_task),
                patch.object(run_experiments, "refresh_manifest"),
                patch.object(run_experiments, "consolidate"),
                redirect_stdout(output),
            ):
                run_experiments.run_command(args)

            text = output.getvalue()
            self.assertIn("Starting run_slow (test_stage)", text)
            self.assertIn("Still running run_slow (test_stage)", text)

    def test_main_handles_keyboard_interrupt_without_traceback(self) -> None:
        args = argparse.Namespace(max_workers=1, progress_interval=30.0)
        args.function = Mock(side_effect=KeyboardInterrupt)
        output = io.StringIO()
        with patch.object(
            run_experiments, "parse_args", return_value=args
        ), redirect_stdout(output):
            return_code = run_experiments.main()
        self.assertEqual(return_code, 130)
        self.assertIn("Re-run with --resume", output.getvalue())


if __name__ == "__main__":
    unittest.main()
