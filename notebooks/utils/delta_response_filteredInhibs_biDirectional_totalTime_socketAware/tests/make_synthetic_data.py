"""Deterministic synthetic processed-data snapshot for the regression suite.

Writes files with the same schema as ``processed_data/*_agg.csv``:

* two susceptibility/aggressiveness patterns (sensitive-quiet vs robust-loud);
* both bandwidth proxies, with inhibitor work volume that changes under load;
* negative auxiliary targets (increased achieved rate);
* missing auxiliary labels (zero co-scheduled byte counts);
* one application whose App-Inhibitor support is entirely at the floor;
* App-App pairs with unequal directional effects; and a 1-node row that the
  8-node filter must drop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


APPS = {
    # name: (mpi_time, runtime, messages, bytes, susceptibility, aggressiveness)
    "alpha": (10.0, 100.0, 1.0e6, 1.0e10, 0.80, 0.10),
    "bravo": (20.0, 120.0, 4.0e6, 4.0e10, 0.60, 0.20),
    "charlie": (5.0, 150.0, 1.0e5, 1.0e9, 0.00, 0.15),  # all-floor support
    "delta": (40.0, 110.0, 8.0e6, 9.0e10, 0.10, 0.90),
    "echo": (30.0, 130.0, 6.0e6, 7.0e10, 0.20, 0.70),
    "foxtrot": (15.0, 140.0, 2.0e6, 2.0e10, 0.45, 0.45),
}
MESSAGE_SIZES = [20, 2000, 200000]
WAIT_TIMES = [0, 1000]
SPARSITIES = [0.01, 0.5]


def deterministic_noise(*keys: object, scale: float = 0.02) -> float:
    text = "|".join(str(key) for key in keys)
    value = sum((index + 1) * ord(character) for index, character in enumerate(text)) % 997
    return scale * (value / 996.0 - 0.5)


def inhibitor_rows() -> list[dict[str, float]]:
    rows = []
    for size in MESSAGE_SIZES:
        for wait in WAIT_TIMES:
            for sparsity in SPARSITIES:
                pressure = (np.log10(size) / 5.3) * (1.0 if wait == 0 else 0.6) * (0.5 + sparsity)
                runtime = 100.0
                messages = 1.0e7 * (1.0 if wait == 0 else 0.2) * (0.2 + sparsity)
                rows.append({
                    "Inhib Message Size": size, "Inhib Wait Time (us)": wait,
                    "Inhib Comm Sparsity": sparsity, "pressure": pressure,
                    "runtime": runtime, "mpi": 5.0 + 40.0 * pressure,
                    "messages": messages, "bytes": messages * size,
                })
    return rows


def write_synthetic_snapshot(directory: Path | str, legacy: bool = False) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    isolated = []
    for name, (mpi, runtime, messages, volume, _, _) in APPS.items():
        for nodes, factor in [(8, 1.0), (1, 0.5)]:
            isolated.append({
                "App": name, "Num Nodes": nodes,
                "App Time Average Mean": runtime * factor, "App Time Average Median": runtime * factor,
                "App Time Average Std": 0.0,
                "MPI Time Average Mean": mpi * factor, "MPI Time Average Median": mpi * factor,
                "MPI Time Average Std": 0.0,
                "Total Messages Sent Mean": messages, "Total Messages Sent Median": messages,
                "Total Messages Sent Std": 0.0,
                "Total Bytes Sent Mean": volume, "Total Bytes Sent Median": volume, "Total Bytes Sent Std": 0.0,
            })
    inhibitors = inhibitor_rows()
    inhib_isolated = []
    for row in inhibitors:
        record = {"Num Nodes": 8, **{key: row[key] for key in ["Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]}}
        for label, value in [("App Time Average", row["runtime"]), ("MPI Time Average", row["mpi"]),
                             ("Total Messages Sent", row["messages"]), ("Total Bytes Sent", row["bytes"])]:
            record.update({f"Inhib {label} Mean": value, f"Inhib {label} Median": value, f"Inhib {label} Std": 0.0})
        inhib_isolated.append(record)
    coscheduled = []
    for name, (mpi, runtime, messages, volume, susceptibility, aggressiveness) in APPS.items():
        for index, row in enumerate(inhibitors):
            log_ratio = susceptibility * row["pressure"] - 0.05 + deterministic_noise(name, index)
            mpi_co = mpi * np.exp(log_ratio)
            runtime_co = runtime + (mpi_co - mpi)
            app_bytes = 0.0 if (name == "bravo" and index == 0) else volume
            inhib_bytes = row["bytes"] * np.exp(-(aggressiveness * row["pressure"]) + 0.03)
            inhib_messages = row["messages"] * np.exp(-(aggressiveness * row["pressure"]) + 0.03)
            inhib_mpi = row["mpi"] * np.exp(0.5 * aggressiveness * row["pressure"])
            record = {"App": name, "Num Nodes": 8,
                      **{key: row[key] for key in ["Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]}}
            for label, value in [("App Time Average", runtime_co), ("MPI Time Average", mpi_co),
                                 ("Total Messages Sent", messages), ("Total Bytes Sent", app_bytes)]:
                record.update({f"{label} Mean": value, f"{label} Median": value, f"{label} Std": 0.0})
            for label, value in [("App Time Average", row["runtime"]), ("MPI Time Average", inhib_mpi),
                                 ("Total Messages Sent", inhib_messages), ("Total Bytes Sent", inhib_bytes)]:
                record.update({f"Inhib {label} Mean": value, f"Inhib {label} Median": value, f"Inhib {label} Std": 0.0})
            coscheduled.append(record)
    pairs = []
    names = list(APPS)
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            record = {"App A": left, "App B": right, "Num Nodes": 8}
            for endpoint, victim, aggressor in [("A", left, right), ("B", right, left)]:
                mpi, runtime, messages, volume, susceptibility, _ = APPS[victim]
                pressure = APPS[aggressor][5]
                log_ratio = susceptibility * pressure * 0.8 - 0.03 + deterministic_noise(victim, aggressor)
                mpi_co = mpi * np.exp(log_ratio)
                values = [("Time Average", runtime + mpi_co - mpi), ("MPI Time Average", mpi_co),
                          ("Total Messages Sent", messages), ("Total Bytes Sent", volume)]
                for label, value in values:
                    record.update({f"App {endpoint} {label} Mean": value, f"App {endpoint} {label} Median": value,
                                   f"App {endpoint} {label} Std": 0.0})
            if left == right:
                # Mirrors the processed snapshot: App B byte counts are zero on self-pairs.
                for statistic in ["Mean", "Median"]:
                    record[f"App B Total Bytes Sent {statistic}"] = 0.0
            pairs.append(record)
    pairs.append({**pairs[0], "Num Nodes": 1})
    paths = {
        "isolated_apps": root / "data_isolated_agg.csv",
        "isolated_inhibitors": root / "data_inhib_isolated_agg.csv",
        "app_inhibitor": root / "data_inhib_coscheduled_agg.csv",
        "app_app": root / "data_coscheduled_agg.csv",
    }
    pd.DataFrame(isolated).to_csv(paths["isolated_apps"])
    pd.DataFrame(inhib_isolated).to_csv(paths["isolated_inhibitors"])
    pd.DataFrame(coscheduled).to_csv(paths["app_inhibitor"])
    pd.DataFrame(pairs).to_csv(paths["app_app"])
    if legacy:
        legacy_dir = root / "legacy"
        legacy_dir.mkdir(exist_ok=True)
        rows = []
        for row in inhibitors:
            rows.append({"inhib_id": f"legacy_{len(rows)}", "msg_size": row["Inhib Message Size"],
                         "wait_time": row["Inhib Wait Time (us)"], "comm_sparsity": row["Inhib Comm Sparsity"],
                         "mpi_time": row["mpi"], "comm_frac": row["mpi"] / row["runtime"],
                         "total_msgs": row["messages"], "total_bytes": row["bytes"]})
        rows[0]["mpi_time"] *= 0.5  # deliberate provenance discrepancy
        pd.DataFrame(rows).to_csv(legacy_dir / "inhibitors.csv")
        paths["legacy_dir"] = legacy_dir
    return paths


def prepare_synthetic(directory: Path | str, legacy: bool = False) -> dict[str, Path]:
    """Write the snapshot and run both preparation entry points."""
    from prepare_bidirectional_data import prepare_pair_evaluation_data, prepare_training_data

    root = Path(directory)
    paths = write_synthetic_snapshot(root / "processed", legacy=legacy)
    training = root / "prepared" / "training"
    evaluation = root / "prepared" / "evaluation"
    prepare_training_data(paths["isolated_apps"], paths["isolated_inhibitors"], paths["app_inhibitor"], 8,
                          training, legacy_dir=paths.get("legacy_dir"))
    prepare_pair_evaluation_data(paths["app_app"], training / "jobs.csv", 8, evaluation)
    return {**paths, "training_dir": training, "evaluation_dir": evaluation,
            "jobs": training / "jobs.csv", "inhibitors": training / "inhibitors.csv",
            "job_inh": training / "job_inh.csv", "pair": evaluation / "pair.csv"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print(prepare_synthetic(arguments.output_dir, legacy=True))
