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


def ten_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [f"j{index}" for index in range(10)],
            "mpi_time": [index + 1 for index in range(10)],
            "comm_frac": [(index + 1) / 20 for index in range(10)],
            "total_msgs": [(index + 1) * 10 for index in range(10)],
            "total_bytes": [(index + 1) * 100 for index in range(10)],
        }
    )


def write_plan_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "jobs": root / "jobs.csv",
        "inhibitors": root / "inhibitors.csv",
        "job_inh": root / "job_inh.csv",
        "pair": root / "pair.csv",
    }
    ten_jobs().to_csv(paths["jobs"], index=False)
    pd.DataFrame(
        [["i0", 1.0, 0.1, 10, 100]],
        columns=["inhib_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
    ).to_csv(paths["inhibitors"], index=False)
    pd.DataFrame(
        [["j0", "i0", 1.1]], columns=["job_id", "inhib_id", "slowdown"]
    ).to_csv(paths["job_inh"], index=False)
    pd.DataFrame(
        [["j0", "j0", 1.1, 1.1]],
        columns=["jobA_id", "jobB_id", "slowdown_A", "slowdown_B"],
    ).to_csv(paths["pair"], index=False)
    return paths


def selected_aggregation() -> dict[str, object]:
    return {
        "uniform": {},
        "median": {},
        "kernel": {"temperature": 1.0},
        "ood_kernel": {
            "temperature": 1.0,
            "quantile": 0.95,
            "fallback": "median",
        },
        "primary_method": "uniform",
        "accuracy_best": {
            "method": "kernel",
            "temperature": 1.0,
            "validation_log_mae": 0.1,
        },
    }


class FocusedExperimentTests(unittest.TestCase):
    def test_nested_orders_are_balanced_prefixes(self) -> None:
        jobs = ten_jobs()
        orders = run_experiments.nested_orders(jobs, count=10, seed=7)
        self.assertEqual(len(orders), 10)
        self.assertTrue(all(len(set(order)) == 10 for order in orders))
        for size in range(2, 9):
            counts = {
                app_id: sum(app_id in order[:size] for order in orders)
                for app_id in jobs["job_id"]
            }
            self.assertEqual(set(counts.values()), {size})

    def test_standard_plan_has_one_global_selection_and_focused_matrix(self) -> None:
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            paths = write_plan_inputs(root)
            tasks = run_experiments.build_plan(
                root,
                "standard",
                paths["jobs"],
                paths["inhibitors"],
                paths["job_inh"],
                paths["pair"],
            )
            self.assertEqual(len(tasks), 74)
            tuning = [task for task in tasks if task["stage"] in run_experiments.TUNING_STAGES]
            random = [task for task in tasks if task["stage"] == "evaluate_random_split"]
            restricted = [task for task in tasks if task["stage"] == "evaluate_training_size"]
            self.assertEqual(len(tuning), 3)
            self.assertEqual(len(random), 1)
            self.assertEqual(len(restricted), 70)
            global_task_id = tuning[-1]["task_id"]
            self.assertTrue(
                all(task["depends_on"] == global_task_id for task in [*random, *restricted])
            )
            self.assertEqual(random[0]["config"]["evaluation_method"], "random_split")
            self.assertTrue(
                all(task["config"]["evaluation_method"] == "both" for task in restricted)
            )
            self.assertTrue(
                all(task["config"]["inference_anchor_counts"] == ["all"] for task in tasks)
            )
            self.assertEqual(
                {task["training_app_count"] for task in restricted}, set(range(2, 9))
            )
            self.assertEqual(
                {
                    size: sum(task["training_app_count"] == size for task in restricted)
                    for size in range(2, 9)
                },
                {size: 10 for size in range(2, 9)},
            )
            membership = pd.read_csv(root / "subset_membership.csv")
            for size in range(2, 9):
                rows = membership[membership["training_app_count"] == size]
                counts = rows[rows["in_training"]].groupby("app_id").size()
                self.assertEqual(set(counts.index), set(ten_jobs()["job_id"]))
                self.assertEqual(set(counts.to_numpy()), {size})
            paths["pair"].write_text("changed after planning\n")
            with self.assertRaisesRegex(RuntimeError, "create a new immutable plan"):
                run_experiments.verify_planned_inputs(random[0])

    def test_evaluation_config_freezes_dependency_selection(self) -> None:
        selection = {
            "model_configs": {
                kind: {**run_experiments.DEFAULT_MODEL, "rank": 4 if kind == "low_rank" else 2}
                for kind in run_experiments.MODEL_KINDS
            },
            "aggregations": {
                "low_rank": selected_aggregation(),
                "generic": selected_aggregation(),
            },
            "final_epochs": {"low_rank": 11, "generic": 12, "absolute": 13},
            "model_selection": {
                kind: {"validation_log_mae": 0.1}
                for kind in run_experiments.MODEL_KINDS
            },
        }
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            output = root / "dependency"
            output.mkdir()
            run_experiments.write_json(output / "selection.json", selection)
            task = {
                "task_id": "evaluation",
                "depends_on": "tuning",
                "candidate_strategy": "fixed_global_from_dependency",
                "config": run_experiments.base_run_config(
                    run_experiments.PROFILES["smoke"]
                ),
            }
            with patch.object(
                run_experiments, "dependency_selection", return_value=selection
            ), patch.object(
                run_experiments, "successful_output", return_value=output
            ):
                config = run_experiments.resolve_task_config(root, task)
            self.assertEqual(config["selection_mode"], "fixed_global")
            self.assertEqual(config["fixed_final_epochs_by_kind"]["low_rank"], 11)
            self.assertEqual(
                config["model_candidates_by_kind"]["low_rank"][0]["rank"], 4
            )
            self.assertEqual(
                config["fixed_aggregations_by_kind"]["low_rank"]["primary_method"],
                "kernel",
            )
            self.assertEqual(config["inference_anchor_counts"], ["all"])
            self.assertIn("no App-App outcomes", config["fixed_selection_source"]["selection_scope"])

    def test_cli_requires_explicit_pair_holdout(self) -> None:
        with self.assertRaises(SystemExit):
            run_experiments.parse_args(["plan", "--root", "out"])
        args = run_experiments.parse_args(
            ["plan", "--root", "out", "--pair-csv", "new_pair.csv"]
        )
        self.assertEqual(args.command, "plan")
        self.assertEqual(args.pair_csv, Path("new_pair.csv"))
        with self.assertRaisesRegex(ValueError, "historical pair.csv is contaminated"):
            run_experiments.build_plan(
                Path("unused"),
                "smoke",
                Path("jobs.csv"),
                Path("inhibitors.csv"),
                Path("job_inh.csv"),
                run_experiments.DEFAULT_DATA / "pair.csv",
            )

    def test_consolidation_removes_stale_optional_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=AUDIT_DIR) as directory:
            root = Path(directory)
            consolidated = root / "consolidated"
            consolidated.mkdir()
            (consolidated / "predictions.csv").write_text("stale\n")
            (consolidated / "learning_curve_summary.csv").write_text("stale\n")
            run_experiments.write_json(root / "plan.json", [])
            run_experiments.consolidate(root, include_predictions=False)
            self.assertFalse((consolidated / "predictions.csv").exists())
            self.assertFalse((consolidated / "learning_curve_summary.csv").exists())

    def test_run_command_reports_progress_for_long_task(self) -> None:
        task = {
            "task_id": "run_slow",
            "stage": "test_stage",
            "depends_on": None,
        }
        args = argparse.Namespace(
            root=None,
            stages=["test_stage"],
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

    def test_bounded_resume_advances_past_completed_tasks(self) -> None:
        tasks = [
            {"task_id": f"run_{index}", "stage": "evaluate", "depends_on": None}
            for index in range(3)
        ]
        with patch.object(
            run_experiments,
            "successful_output",
            side_effect=lambda _root, task_id: Path("complete")
            if task_id == "run_0"
            else None,
        ):
            selected = run_experiments.selected_tasks_with_dependencies(
                tasks,
                {"evaluate"},
                max_tasks=1,
                root=Path("root"),
                resume=True,
            )
        self.assertEqual([task["task_id"] for task in selected], ["run_1"])

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
