"""Rebuild pipeline inputs from one processed-data snapshot for either time basis.

Writes ``jobs.csv``, ``inhibitors.csv``, ``job_inh.csv``, and ``pair.csv`` with
the schema produced by ``notebooks/few_shot.ipynb`` (plus diagnostic columns
the loader ignores). Only the slowdown target depends on the time basis:

* ``app_time``: ``slowdown = max(1, T_app_co / T_app_iso)`` (total application
  time, mpiP ``AppTime``). This is the default input of this pipeline.
* ``mpi_time``: ``slowdown = max(1, T_MPI_co / T_MPI_iso)``, the original
  target, rebuilt from the same snapshot for a matched comparison.

Isolated profile features (``mpi_time``, ``comm_frac``, ``total_msgs``,
``total_bytes``) are identical for both bases. Every row is an aggregate
configuration (mean of per-run averages), not a run-level replicate.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROCESSED = HERE.parents[2] / "processed_data"
DEFAULT_OUTPUT_ROOT = HERE / "data"
TIME_BASES = {
    "app_time": "App Time Average Mean",
    "mpi_time": "MPI Time Average Mean",
}
INHIB_CONFIG = ["Num Nodes", "Inhib Message Size", "Inhib Wait Time (us)", "Inhib Comm Sparsity"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, num_nodes: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
    return frame[pd.to_numeric(frame["Num Nodes"], errors="coerce") == num_nodes].reset_index(drop=True)


def canonical_inhibitor_id(num_nodes, msg_size, wait_time, comm_sparsity) -> str:
    """``{nodes}_{msg size}_{wait us}_{sparsity}`` with integers and normalized decimals."""
    integers = []
    for value in (num_nodes, msg_size, wait_time):
        number = float(value)
        if not np.isfinite(number) or number != np.round(number):
            raise ValueError(f"inhibitor configuration value must be an integer, got {value!r}")
        integers.append(str(int(number)))
    sparsity = format(Decimal(repr(float(comm_sparsity))).normalize(), "f")
    return "_".join([*integers, sparsity])


def inhibitor_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [canonical_inhibitor_id(*values) for values in frame[INHIB_CONFIG].itertuples(index=False)],
        index=frame.index,
    )


def clipped_ratio(co: pd.Series, iso: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    co = co.to_numpy(dtype=float)
    iso = iso.to_numpy(dtype=float)
    if not (np.isfinite(co).all() and np.isfinite(iso).all()) or (co <= 0).any() or (iso <= 0).any():
        raise ValueError("timing values must be finite and positive")
    raw = co / iso
    return np.maximum(1.0, raw), raw


def build_profiles(isolated_apps: pd.DataFrame, isolated_inhibitors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = pd.DataFrame(
        {
            "job_id": isolated_apps["App"].astype(str),
            "mpi_time": isolated_apps["MPI Time Average Mean"],
            "comm_frac": isolated_apps["MPI Time Average Mean"] / isolated_apps["App Time Average Mean"],
            "total_msgs": isolated_apps["Total Messages Sent Mean"],
            "total_bytes": isolated_apps["Total Bytes Sent Mean"],
            "app_time": isolated_apps["App Time Average Mean"],
        }
    )
    inhibitors = pd.DataFrame(
        {
            "inhib_id": inhibitor_ids(isolated_inhibitors),
            "msg_size": isolated_inhibitors["Inhib Message Size"],
            "wait_time": isolated_inhibitors["Inhib Wait Time (us)"],
            "comm_sparsity": isolated_inhibitors["Inhib Comm Sparsity"],
            "mpi_time": isolated_inhibitors["Inhib MPI Time Average Mean"],
            "comm_frac": isolated_inhibitors["Inhib MPI Time Average Mean"]
            / isolated_inhibitors["Inhib App Time Average Mean"],
            "total_msgs": isolated_inhibitors["Inhib Total Messages Sent Mean"],
            "total_bytes": isolated_inhibitors["Inhib Total Bytes Sent Mean"],
            "app_time": isolated_inhibitors["Inhib App Time Average Mean"],
        }
    )
    for name, frame, key in [("applications", jobs, "job_id"), ("inhibitors", inhibitors, "inhib_id")]:
        if frame[key].duplicated().any():
            raise ValueError(f"isolated {name} contain duplicate configurations")
        if not frame["comm_frac"].between(0.0, 1.0, inclusive="right").all():
            raise ValueError(f"isolated {name} have MPI time exceeding application time")
    return (
        jobs.sort_values("job_id").reset_index(drop=True),
        inhibitors.sort_values("inhib_id").reset_index(drop=True),
    )


def build_responses(app_inhibitor: pd.DataFrame, jobs: pd.DataFrame, inhibitors: pd.DataFrame, basis: str) -> pd.DataFrame:
    column = TIME_BASES[basis]
    responses = pd.DataFrame({"job_id": app_inhibitor["App"].astype(str), "inhib_id": inhibitor_ids(app_inhibitor)})
    if responses.duplicated().any():
        raise ValueError("App-Inhibitor data contain duplicate configurations")
    missing = ~responses["job_id"].isin(jobs["job_id"]) | ~responses["inhib_id"].isin(inhibitors["inhib_id"])
    if missing.any():
        raise ValueError(f"App-Inhibitor rows without isolated profiles: {responses[missing].to_dict('records')[:5]}")
    iso = responses["job_id"].map(jobs.set_index("job_id")[basis])
    responses["slowdown"], responses["raw_time_ratio"] = clipped_ratio(app_inhibitor[column], iso)
    return responses


def build_pairs(app_app: pd.DataFrame, jobs: pd.DataFrame, basis: str) -> pd.DataFrame:
    pairs = pd.DataFrame({"jobA_id": app_app["App A"].astype(str), "jobB_id": app_app["App B"].astype(str)})
    iso = jobs.set_index("job_id")[basis]
    for endpoint in ["A", "B"]:
        column = f"App {endpoint} Time Average Mean" if basis == "app_time" else f"App {endpoint} MPI Time Average Mean"
        slowdown, raw = clipped_ratio(app_app[column], pairs[f"job{endpoint}_id"].map(iso))
        pairs[f"slowdown_{endpoint}"] = slowdown
        pairs[f"raw_time_ratio_{endpoint}"] = raw
    return pairs


def prepare(
    output_root: Path,
    num_nodes: int = 8,
    isolated_apps_csv: Path = PROCESSED / "data_isolated_agg.csv",
    isolated_inhibitors_csv: Path = PROCESSED / "data_inhib_isolated_agg.csv",
    app_inhibitor_csv: Path = PROCESSED / "data_inhib_coscheduled_agg.csv",
    app_app_csv: Path = PROCESSED / "data_coscheduled_agg.csv",
) -> dict[str, dict]:
    sources = {
        "isolated_apps_csv": Path(isolated_apps_csv),
        "isolated_inhibitors_csv": Path(isolated_inhibitors_csv),
        "app_inhibitor_csv": Path(app_inhibitor_csv),
        "app_app_csv": Path(app_app_csv),
    }
    jobs, inhibitors = build_profiles(_read(sources["isolated_apps_csv"], num_nodes),
                                      _read(sources["isolated_inhibitors_csv"], num_nodes))
    app_inhibitor = _read(sources["app_inhibitor_csv"], num_nodes)
    app_app = _read(sources["app_app_csv"], num_nodes)
    reports = {}
    for basis in TIME_BASES:
        output = Path(output_root) / basis
        output.mkdir(parents=True, exist_ok=True)
        responses = build_responses(app_inhibitor, jobs, inhibitors, basis)
        pairs = build_pairs(app_app, jobs, basis)
        jobs.to_csv(output / "jobs.csv", index=False)
        inhibitors.to_csv(output / "inhibitors.csv", index=False)
        responses.to_csv(output / "job_inh.csv", index=False)
        pairs.to_csv(output / "pair.csv", index=False)
        non_self = pairs[pairs["jobA_id"] != pairs["jobB_id"]]
        target = "T_app" if basis == "app_time" else "T_MPI"
        reports[basis] = {
            "time_basis": basis,
            "slowdown": f"max(1, {target}_co / {target}_iso); raw_time_ratio is the unclipped ratio",
            "profile_features": "mpi_time, comm_frac = T_MPI / T_app, total_msgs, total_bytes (identical for both bases)",
            "observation_unit": "aggregate_configuration (mean of per-run averages)",
            "num_nodes": int(num_nodes),
            "row_counts": {
                "jobs": len(jobs),
                "inhibitors": len(inhibitors),
                "job_inh": len(responses),
                "job_inh_floor": int((responses["slowdown"] == 1.0).sum()),
                "pairs": len(pairs),
                "self_pairs": int(len(pairs) - len(non_self)),
                "non_self_directional_floor": int((non_self[["slowdown_A", "slowdown_B"]] == 1.0).to_numpy().sum()),
            },
            "source_paths": {name: str(path.resolve()) for name, path in sources.items()},
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        }
        (output / "preparation_report.json").write_text(json.dumps(reports[basis], indent=2) + "\n")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--isolated-apps-csv", type=Path, default=PROCESSED / "data_isolated_agg.csv")
    parser.add_argument("--isolated-inhibitors-csv", type=Path, default=PROCESSED / "data_inhib_isolated_agg.csv")
    parser.add_argument("--app-inhibitor-csv", type=Path, default=PROCESSED / "data_inhib_coscheduled_agg.csv")
    parser.add_argument("--app-app-csv", type=Path, default=PROCESSED / "data_coscheduled_agg.csv")
    args = parser.parse_args()
    reports = prepare(args.output_root, args.num_nodes, args.isolated_apps_csv,
                      args.isolated_inhibitors_csv, args.app_inhibitor_csv, args.app_app_csv)
    for basis, report in reports.items():
        print(basis, json.dumps(report["row_counts"]))
    print(f"Outputs: {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
