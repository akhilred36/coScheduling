from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from data import load_pair_holdout, load_training_data
import prepare_data


APPS = {"alpha": (100.0, 10.0), "bravo": (200.0, 50.0)}  # name: (app time, MPI time)
INHIBITORS = [(20, 5000, 0.01), (2000, 0, 0.002237136)]


def write_snapshot(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    isolated, inhib_isolated, coscheduled, pairs = [], [], [], []
    for name, (app_time, mpi_time) in APPS.items():
        for nodes in [8, 1]:
            isolated.append({"App": name, "Num Nodes": nodes, "App Time Average Mean": app_time,
                             "MPI Time Average Mean": mpi_time, "Total Messages Sent Mean": 1e6,
                             "Total Bytes Sent Mean": 1e9})
    for size, wait, sparsity in INHIBITORS:
        inhib_isolated.append({"Num Nodes": 8, "Inhib Message Size": size, "Inhib Wait Time (us)": wait,
                               "Inhib Comm Sparsity": sparsity, "Inhib App Time Average Mean": 360.0,
                               "Inhib MPI Time Average Mean": 8.0, "Inhib Total Messages Sent Mean": 1e7,
                               "Inhib Total Bytes Sent Mean": 1e10})
    for name, (app_time, mpi_time) in APPS.items():
        for index, (size, wait, sparsity) in enumerate(INHIBITORS):
            # Inhibitor 0 slows application time but not MPI time; inhibitor 1 the reverse.
            coscheduled.append({"App": name, "Num Nodes": 8, "Inhib Message Size": size,
                                "Inhib Wait Time (us)": wait, "Inhib Comm Sparsity": sparsity,
                                "App Time Average Mean": app_time * (1.2 if index == 0 else 0.95),
                                "MPI Time Average Mean": mpi_time * (0.9 if index == 0 else 1.5)})
    pairs.append({"App A": "alpha", "App B": "bravo", "Num Nodes": 8,
                  "App A Time Average Mean": 110.0, "App B Time Average Mean": 190.0,
                  "App A MPI Time Average Mean": 9.0, "App B MPI Time Average Mean": 60.0})
    paths = {
        "isolated_apps_csv": root / "data_isolated_agg.csv",
        "isolated_inhibitors_csv": root / "data_inhib_isolated_agg.csv",
        "app_inhibitor_csv": root / "data_inhib_coscheduled_agg.csv",
        "app_app_csv": root / "data_coscheduled_agg.csv",
    }
    for frame, key in [(isolated, "isolated_apps_csv"), (inhib_isolated, "isolated_inhibitors_csv"),
                       (coscheduled, "app_inhibitor_csv"), (pairs, "app_app_csv")]:
        pd.DataFrame(frame).to_csv(paths[key])
    return paths


class PrepareDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.paths = write_snapshot(root / "processed")
        cls.output = root / "prepared"
        cls.reports = prepare_data.prepare(cls.output, 8, **cls.paths)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def read(self, basis: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.output / basis / f"{name}.csv")

    def test_slowdown_targets_follow_time_basis(self) -> None:
        app = self.read("app_time", "job_inh").set_index(["job_id", "inhib_id"])["slowdown"]
        mpi = self.read("mpi_time", "job_inh").set_index(["job_id", "inhib_id"])["slowdown"]
        first, second = "8_20_5000_0.01", "8_2000_0_0.002237136"
        for job in APPS:
            self.assertAlmostEqual(app[(job, first)], 1.2)
            self.assertEqual(app[(job, second)], 1.0)
            self.assertEqual(mpi[(job, first)], 1.0)
            self.assertAlmostEqual(mpi[(job, second)], 1.5)

    def test_pair_slowdowns_follow_time_basis(self) -> None:
        app = self.read("app_time", "pair").iloc[0]
        mpi = self.read("mpi_time", "pair").iloc[0]
        self.assertAlmostEqual(app["slowdown_A"], 1.1)
        self.assertEqual(app["slowdown_B"], 1.0)
        self.assertAlmostEqual(app["raw_time_ratio_B"], 0.95)
        self.assertEqual(mpi["slowdown_A"], 1.0)
        self.assertAlmostEqual(mpi["slowdown_B"], 1.2)

    def test_profiles_are_identical_across_bases(self) -> None:
        for name in ["jobs", "inhibitors"]:
            pd.testing.assert_frame_equal(self.read("app_time", name), self.read("mpi_time", name))
        jobs = self.read("app_time", "jobs")
        self.assertEqual(len(jobs), len(APPS))  # 1-node rows dropped
        np.testing.assert_allclose(jobs["comm_frac"], jobs["mpi_time"] / jobs["app_time"])
        self.assertEqual(sorted(self.read("app_time", "inhibitors")["inhib_id"]),
                         ["8_2000_0_0.002237136", "8_20_5000_0.01"])

    def test_outputs_load_in_pipeline_and_report_provenance(self) -> None:
        for basis in prepare_data.TIME_BASES:
            directory = self.output / basis
            data = load_training_data(directory / "jobs.csv", directory / "inhibitors.csv", directory / "job_inh.csv")
            self.assertEqual(len(data.responses), 4)
            self.assertEqual(int(data.responses["is_censored"].sum()), 2)
            self.assertEqual(len(load_pair_holdout(directory / "pair.csv", data.jobs)), 1)
            report = json.loads((directory / "preparation_report.json").read_text())
            self.assertEqual(report["time_basis"], basis)
            self.assertEqual(set(report["source_sha256"]), set(self.paths))

    def test_duplicate_configuration_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_snapshot(Path(directory) / "processed")
            frame = pd.read_csv(paths["app_inhibitor_csv"], index_col=0)
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(paths["app_inhibitor_csv"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_data.prepare(Path(directory) / "out", 8, **paths)


if __name__ == "__main__":
    unittest.main()
