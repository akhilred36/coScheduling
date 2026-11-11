"""Coverage, measurement integrity, and outcome-independence checks."""

import argparse
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_training_data
from select_inhibitors import communication_features, coverage_score, run, select_panel


class SelectionTests(unittest.TestCase):
    def test_runtime_normalization_is_duration_invariant(self):
        frame = pd.DataFrame({"mpi_time": [10., 100.], "comm_frac": [.1, .1],
                              "total_msgs": [1000., 10000.],
                              "total_bytes": [100000., 1000000.]})
        features = communication_features(frame)
        np.testing.assert_allclose(features[0], features[1])
        self.assertAlmostEqual(features[0, 1], np.log1p(10.))

    def test_nearest_union_is_global_optimum_and_not_a_quota(self):
        d = np.array([[0., 5., 7., 0.], [0., 3., 8., 0.], [4., 0., 3., 4.]])
        selected, _, status = select_panel(d)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(status, "global_optimum_pointwise_lower_bound")
        np.testing.assert_equal(d[:, selected].min(axis=1), d.min(axis=1))
        for budget in [0, 11]:
            with self.assertRaises(ValueError):
                select_panel(d, budget)

    def test_budget_limited_panel_is_one_swap_local_optimum(self):
        rng = np.random.default_rng(4)
        x, z = rng.normal(size=(12, 3)), rng.normal(size=(9, 3))
        d = ((x[:, None] - z[None]) ** 2).mean(axis=2)
        for budget in [1, 2, 3]:
            selected, _, _ = select_panel(d, budget)
            self.assertLessEqual(len(selected), budget)
            score = coverage_score(d[:, selected].min(axis=1), .5)
            for old, new in itertools.product(selected, set(range(9)) - set(selected)):
                trial = [j for j in selected if j != old] + [new]
                self.assertGreaterEqual(coverage_score(d[:, trial].min(axis=1), .5), score - 1e-12)
            self.assertEqual(selected, select_panel(d, budget)[0])

    def test_invalid_runtime_is_not_silently_fabricated(self):
        frame = pd.DataFrame({"mpi_time": [0.], "comm_frac": [0.],
                              "total_msgs": [10.], "total_bytes": [10.]})
        with self.assertRaises(ValueError):
            communication_features(frame)

    def test_outputs_preserve_floor_replicates_and_ignore_outcome_magnitude(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = pd.DataFrame({"job_id": ["a", "b"], "mpi_time": [10., 20.],
                                 "comm_frac": [.1, .3], "total_msgs": [100., 200.],
                                 "total_bytes": [10000., 40000.]})
            inhibitors = jobs.rename(columns={"job_id": "inhib_id"}).copy()
            inhibitors["inhib_id"] = ["i", "j"]
            # Exact profile match but no measurements: must not be selected.
            missing = inhibitors.iloc[[0]].copy()
            missing["inhib_id"] = "h"
            inhibitors = pd.concat([inhibitors, missing], ignore_index=True)
            responses = pd.DataFrame({"job_id": ["a", "a", "b", "b", "a"],
                                      "inhib_id": ["i", "j", "i", "j", "i"],
                                      "slowdown": [1., 2., 1., 1.2, 1.1]})
            jobs.to_csv(root / "jobs.csv", index=False)
            inhibitors.to_csv(root / "inhibitors.csv", index=False)
            responses.to_csv(root / "job_inh.csv", index=False)
            args = argparse.Namespace(jobs_csv=root / "jobs.csv",
                                      inhibitors_csv=root / "inhibitors.csv",
                                      job_inh_csv=root / "job_inh.csv",
                                      output_dir=root / "first", target_apps=None,
                                      max_profiles=10, tail_weight=.5)
            first = run(args)
            self.assertEqual(first["selected_ids"], ["i", "j"])
            self.assertEqual(first["excluded_missing_target_responses"], ["h"])
            loaded = load_training_data(args.output_dir / "jobs.csv",
                                        args.output_dir / "inhibitors.csv",
                                        args.output_dir / "job_inh.csv")
            self.assertEqual(len(loaded.responses), 5)
            self.assertEqual(int(loaded.responses["is_censored"].sum()), 2)
            self.assertEqual(len(loaded.duplicate_keys), 2)
            json.loads((args.output_dir / "selection_report.json").read_text())
            # Reorder every input and change all slowdown values, preserving validity.
            jobs.iloc[::-1].to_csv(args.jobs_csv, index=False)
            inhibitors.iloc[::-1].to_csv(args.inhibitors_csv, index=False)
            responses["slowdown"] = [3., 1., 4., 1., 3.2]
            responses.iloc[::-1].to_csv(args.job_inh_csv, index=False)
            args.output_dir = root / "second"
            second = run(args)
            self.assertEqual(first["selected_ids"], second["selected_ids"])
            self.assertEqual(first["selected_coverage"], second["selected_coverage"])


if __name__ == "__main__":
    unittest.main()
