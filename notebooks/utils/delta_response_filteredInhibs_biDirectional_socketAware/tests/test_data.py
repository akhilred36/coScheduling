from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_bidirectional_data as prep
from data import AuxiliaryScaler, load_bidirectional_training_data, load_pair_holdout
from make_synthetic_data import APPS, inhibitor_rows, prepare_synthetic, write_synthetic_snapshot
import select_inhibitors


def setUpModule() -> None:
    warnings.simplefilter("ignore")


class CanonicalIdTests(unittest.TestCase):
    def test_formatting_is_canonical(self) -> None:
        self.assertEqual(prep.canonical_inhibitor_id(8, 20, 5000, 0.01), "8_20_5000_0.01")
        self.assertEqual(prep.canonical_inhibitor_id(8.0, 20.0, 0.0, 1.0), "8_20_0_1")
        self.assertEqual(prep.canonical_inhibitor_id("8", 200, 200000, 0.002237136), "8_200_200000_0.002237136")
        self.assertEqual(prep.canonical_inhibitor_id(8, 4e6, 1e6, 0.50), "8_4000000_1000000_0.5")
        with self.assertRaises(ValueError):
            prep.canonical_inhibitor_id(8, 20.5, 0, 0.1)
        with self.assertRaises(ValueError):
            prep.canonical_inhibitor_id(8, 20, 0, float("nan"))


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = prepare_synthetic(cls.temporary.name, legacy=True)
        cls.responses = pd.read_csv(cls.paths["job_inh"])
        cls.audit = pd.read_csv(cls.paths["training_dir"] / "observation_audit.csv")
        cls.report = json.loads((cls.paths["training_dir"] / "preparation_report.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def row(self, job: str, inhib: str) -> pd.Series:
        return self.responses[(self.responses["job_id"] == job) & (self.responses["inhib_id"] == inhib)].iloc[0]

    def test_rates_and_log_degradations_for_both_endpoints(self) -> None:
        audit = self.audit.set_index("observation_id")
        for _, row in self.responses.iterrows():
            values = audit.loc[row["observation_id"]]
            for endpoint in ["app", "inhib"]:
                if not row[f"{endpoint}_bw_wall_valid"]:
                    continue
                expected = np.log(values[f"{endpoint}_iso_total_bytes"] / values[f"{endpoint}_iso_runtime"]) - np.log(
                    values[f"{endpoint}_co_total_bytes"] / values[f"{endpoint}_co_runtime"])
                self.assertAlmostEqual(row[f"{endpoint}_bw_wall_log_degradation"], expected, places=10)
                expected = np.log(values[f"{endpoint}_iso_total_msgs"] / values[f"{endpoint}_iso_runtime"]) - np.log(
                    values[f"{endpoint}_co_total_msgs"] / values[f"{endpoint}_co_runtime"])
                self.assertAlmostEqual(row[f"{endpoint}_msg_wall_log_degradation"], expected, places=10)

    def test_negative_degradation_is_preserved(self) -> None:
        self.assertTrue((self.responses["app_bw_wall_log_degradation"] < 0).any())
        self.assertTrue((self.responses["inhib_bw_wall_log_degradation"] < 0).any())

    def test_mpi_throughput_equals_timing_ratio_when_bytes_fixed(self) -> None:
        valid = self.responses[self.responses["app_bw_mpi_valid"]]
        np.testing.assert_allclose(valid["app_bw_mpi_log_degradation"], np.log(valid["raw_mpi_time_ratio"]), atol=1e-12)
        self.assertTrue((valid["log_slowdown"] >= 0).all())
        self.assertTrue((valid.loc[valid["is_censored"], "app_bw_mpi_log_degradation"] <= 0).all())

    def test_changing_inhibitor_work_volume(self) -> None:
        inhibitors = inhibitor_rows()
        row = self.row("delta", prep.canonical_inhibitor_id(8, inhibitors[3]["Inhib Message Size"],
                                                          inhibitors[3]["Inhib Wait Time (us)"],
                                                          inhibitors[3]["Inhib Comm Sparsity"]))
        aggressiveness = APPS["delta"][5]
        # Fixed inhibitor runtime: wall-bandwidth degradation equals the byte-volume change.
        self.assertAlmostEqual(row["inhib_bw_wall_log_degradation"],
                               aggressiveness * inhibitors[3]["pressure"] - 0.03, places=9)

    def test_zero_rate_masks_only_that_task(self) -> None:
        masked = self.responses[~self.responses["app_bw_wall_valid"]]
        self.assertEqual(len(masked), 1)
        self.assertEqual(masked.iloc[0]["job_id"], "bravo")
        self.assertTrue(np.isfinite(masked.iloc[0]["slowdown"]))
        self.assertTrue(masked.iloc[0]["app_msg_wall_valid"])
        availability = pd.read_csv(self.paths["training_dir"] / "auxiliary_target_availability.csv")
        self.assertIn("zero_rate", availability.set_index("target_column").loc["app_bw_wall_log_degradation", "mask_reasons"])

    def test_aggregate_metadata_and_filters_recorded(self) -> None:
        details = self.report["details"]
        self.assertEqual(details["observation_unit"], "aggregate_configuration")
        self.assertEqual(details["rate_estimator"], "ratio_of_aggregate_means")
        self.assertFalse(details["app_app_data_read"])
        self.assertEqual(set(self.report["source_sha256"]), {"isolated_apps_csv", "isolated_inhibitors_csv", "app_inhibitor_csv"})
        self.assertTrue((self.responses["replicate_id"] == 0).all())
        jobs = pd.read_csv(self.paths["jobs"])
        self.assertEqual(len(jobs), len(APPS))
        self.assertTrue((jobs["num_nodes"] == 8).all())
        np.testing.assert_allclose(jobs["comm_frac"], jobs["mpi_time"] / jobs["runtime"])
        pairs = pd.read_csv(self.paths["pair"])
        self.assertTrue((pairs["num_nodes"] == 8).all())

    def test_source_inconsistency_detected(self) -> None:
        report = pd.read_csv(self.paths["training_dir"] / "source_consistency_report.csv")
        legacy = report[(report["check"] == "legacy_inhibitor_profile") & ~report["consistent"]]
        self.assertEqual(legacy["entity"].tolist(), ["8_20_0_0.01"])
        self.assertEqual(legacy["field"].tolist(), ["mpi_time"])
        self.assertGreaterEqual(self.report["details"]["source_inconsistency_count"], 1)

    def test_training_preparation_never_reads_app_app_data(self) -> None:
        read_paths = []
        original = prep._read

        def recording(path):
            read_paths.append(Path(path).resolve())
            return original(path)

        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "_read", recording):
            paths = write_synthetic_snapshot(Path(directory) / "snapshot")
            prep.prepare_training_data(paths["isolated_apps"], paths["isolated_inhibitors"], paths["app_inhibitor"],
                                       8, Path(directory) / "out")
        self.assertNotIn(paths["app_app"].resolve(), read_paths)
        self.assertEqual(set(read_paths), {paths[name].resolve() for name in ["isolated_apps", "isolated_inhibitors", "app_inhibitor"]})

    def test_ambiguous_joins_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_synthetic_snapshot(Path(directory) / "snapshot")
            frame = pd.read_csv(paths["app_inhibitor"], index_col=0)
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(paths["app_inhibitor"])
            prep.prepare_training_data(paths["isolated_apps"], paths["isolated_inhibitors"], paths["app_inhibitor"],
                                       8, Path(directory) / "out")
            rejected = pd.read_csv(Path(directory) / "out" / "rejected_observations.csv")
            self.assertEqual(len(rejected), 2)
            self.assertTrue((rejected["rejection_reason"] == "ambiguous_duplicate_configuration").all())
            isolated = pd.read_csv(paths["isolated_inhibitors"], index_col=0)
            pd.concat([isolated, isolated.iloc[[0]]], ignore_index=True).to_csv(paths["isolated_inhibitors"])
            with self.assertRaisesRegex(ValueError, "ambiguous duplicate"):
                prep.prepare_training_data(paths["isolated_apps"], paths["isolated_inhibitors"],
                                           paths["app_inhibitor"], 8, Path(directory) / "out2")

    def test_missing_join_distinguished_from_invalid_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_synthetic_snapshot(Path(directory) / "snapshot")
            frame = pd.read_csv(paths["app_inhibitor"], index_col=0)
            frame.loc[0, "Inhib Message Size"] = 777
            frame.loc[1, "MPI Time Average Mean"] = -1.0
            frame.to_csv(paths["app_inhibitor"])
            prep.prepare_training_data(paths["isolated_apps"], paths["isolated_inhibitors"], paths["app_inhibitor"],
                                       8, Path(directory) / "out")
            reasons = set(pd.read_csv(Path(directory) / "out" / "rejected_observations.csv")["rejection_reason"])
        self.assertEqual(reasons, {"missing_join_inhibitor_profile", "invalid_slowdown_measurement"})

    def test_loader_keeps_rows_with_missing_auxiliary_labels(self) -> None:
        data = load_bidirectional_training_data(self.paths["jobs"], self.paths["inhibitors"], self.paths["job_inh"])
        self.assertEqual(len(data.responses), len(self.responses))
        self.assertFalse(data.responses["app_bw_wall_valid"].all())
        self.assertTrue(data.responses.loc[~data.responses["app_bw_wall_valid"], "app_bw_wall_log_degradation"].isna().all())
        pairs = load_pair_holdout(self.paths["pair"], data.jobs)
        self.assertIn("A_bw_wall_valid", pairs)

    def test_auxiliary_scaler_pools_endpoints_with_minimum(self) -> None:
        frame = pd.DataFrame({
            "app_bw_wall_log_degradation": [0.0, 0.0, np.nan], "app_bw_wall_valid": [True, True, False],
            "inhib_bw_wall_log_degradation": [1.0, -1.0, 5.0], "inhib_bw_wall_valid": [True, True, False],
        })
        scaler = AuxiliaryScaler(["bw_wall"]).fit(frame)
        self.assertAlmostEqual(scaler.scale("bw_wall"), float(np.std([0.0, 0.0, 1.0, -1.0])))
        self.assertEqual(scaler.counts_["bw_wall"], 4)
        empty = AuxiliaryScaler(["bw_mpi"]).fit(frame)
        self.assertEqual(empty.scale("bw_mpi"), 0.1)
        restored = AuxiliaryScaler.from_state_dict(scaler.state_dict())
        self.assertEqual(restored.scales_, scaler.scales_)


class SelectionTests(unittest.TestCase):
    def test_auxiliary_labels_do_not_change_panel_and_frozen_panel_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = prepare_synthetic(directory)
            arguments = dict(jobs_csv=paths["jobs"], inhibitors_csv=paths["inhibitors"], max_profiles=4,
                             tail_weight=0.5, target_apps=None, frozen_panel_csv=None)
            first = select_inhibitors.run(argparse.Namespace(**arguments, output_dir=Path(directory) / "a",
                                                             job_inh_csv=paths["job_inh"]))
            responses = pd.read_csv(paths["job_inh"])
            for column in [column for column in responses if column.endswith("_log_degradation")]:
                responses[column] = np.nan
            mutated = Path(directory) / "mutated.csv"
            responses.to_csv(mutated, index=False)
            second = select_inhibitors.run(argparse.Namespace(**arguments, output_dir=Path(directory) / "b",
                                                              job_inh_csv=mutated))
            self.assertEqual(first["selected_ids"], second["selected_ids"])
            self.assertEqual(first["panel_source"], "recomputed")
            exported = pd.read_csv(Path(directory) / "a" / "job_inh.csv")
            self.assertIn("inhib_bw_mpi_log_degradation", exported)
            self.assertIn("observation_id", exported)
            frozen = Path(directory) / "frozen.csv"
            pd.DataFrame({"inhib_id": first["selected_ids"][:2]}).to_csv(frozen, index=False)
            third = select_inhibitors.run(argparse.Namespace(**{**arguments, "frozen_panel_csv": frozen},
                                                             output_dir=Path(directory) / "c", job_inh_csv=paths["job_inh"]))
            self.assertEqual(third["panel_source"], "supplied")
            self.assertEqual(sorted(third["selected_ids"]), sorted(first["selected_ids"][:2]))


if __name__ == "__main__":
    unittest.main()
