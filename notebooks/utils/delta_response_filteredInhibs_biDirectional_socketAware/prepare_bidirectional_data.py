"""Rebuild isolated profiles and bidirectional responses from one processed snapshot.

Two separate entry points keep the training/evaluation boundary explicit:

* ``training`` reads isolated application profiles, isolated inhibitor
  profiles, and App-Inhibitor co-scheduled aggregates. It never reads App-App
  data.
* ``evaluation`` reads the App-App aggregate together with the prepared
  application profiles and writes rich directional labels for final scoring.

The default inputs are aggregate configuration tables, so every observation is
an ``aggregate_configuration`` and rates are ``ratio_of_aggregate_means``; the
rows are not run-level replicates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import (
    AUX_TASK_DEFINITIONS,
    AUX_TASKS,
    aux_mask_column,
    aux_target_column,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PROCESSED = REPOSITORY / "processed_data"
LEGACY_DIR = HERE.parent / "few_shot" / "data"
DEFAULT_TRAINING_DIR = HERE / "data" / "training"
DEFAULT_EVALUATION_DIR = HERE / "data" / "evaluation"
SCHEMA_VERSION = 1
OBSERVATION_UNIT = "aggregate_configuration"
RATE_ESTIMATOR = "ratio_of_aggregate_means"
QUANTITIES = ("runtime", "mpi_time", "total_msgs", "total_bytes")
APP_SOURCE = {
    "runtime": "App Time Average Mean",
    "mpi_time": "MPI Time Average Mean",
    "total_msgs": "Total Messages Sent Mean",
    "total_bytes": "Total Bytes Sent Mean",
}
INHIB_SOURCE = {key: f"Inhib {value}" for key, value in APP_SOURCE.items()}
INHIB_CONFIG = {
    "num_nodes": "Num Nodes",
    "msg_size": "Inhib Message Size",
    "wait_time": "Inhib Wait Time (us)",
    "comm_sparsity": "Inhib Comm Sparsity",
}
UNITS = {
    "runtime": "seconds (aggregate mean of per-run averages)",
    "mpi_time": "seconds (aggregate mean of per-run averages)",
    "total_msgs": "messages",
    "total_bytes": "bytes",
    "slowdown": "dimensionless, max(1, co MPI time / isolated MPI time)",
    "log_degradation": "natural log ratio, log(rate_iso) - log(rate_co)",
}
LEGACY_TOLERANCE = 1e-6


def pair_source_columns(endpoint: str) -> dict[str, str]:
    return {
        "runtime": f"App {endpoint} Time Average Mean",
        "mpi_time": f"App {endpoint} MPI Time Average Mean",
        "total_msgs": f"App {endpoint} Total Messages Sent Mean",
        "total_bytes": f"App {endpoint} Total Bytes Sent Mean",
    }


@dataclass
class PreparationReport:
    kind: str
    output_dir: str
    row_counts: dict[str, int] = field(default_factory=dict)
    source_sha256: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def canonical_integer(value: Any, name: str = "value") -> str:
    number = float(value)
    if not np.isfinite(number) or number != np.round(number):
        raise ValueError(f"{name} must be a finite integer, got {value!r}")
    return str(int(np.round(number)))


def canonical_decimal(value: Any, name: str = "value") -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    text = format(Decimal(repr(number)).normalize(), "f")
    return "0" if text in {"-0", ""} else text


def canonical_inhibitor_id(
    num_nodes: Any, msg_size: Any, wait_time: Any, comm_sparsity: Any
) -> str:
    """The single ID formatter used for isolated and co-scheduled inhibitor rows."""
    return "_".join(
        [
            canonical_integer(num_nodes, "Num Nodes"),
            canonical_integer(msg_size, "Inhib Message Size"),
            canonical_integer(wait_time, "Inhib Wait Time (us)"),
            canonical_decimal(comm_sparsity, "Inhib Comm Sparsity"),
        ]
    )


def inhibitor_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            canonical_inhibitor_id(*values)
            for values in frame[list(INHIB_CONFIG.values())].itertuples(index=False)
        ],
        index=frame.index,
        dtype=object,
    )


def _filter_nodes(frame: pd.DataFrame, num_nodes: int) -> pd.DataFrame:
    return frame[pd.to_numeric(frame["Num Nodes"], errors="coerce") == num_nodes].copy()


def _profile_table(
    frame: pd.DataFrame,
    identifiers: pd.Series,
    id_column: str,
    source: dict[str, str],
    num_nodes: int,
    name: str,
) -> pd.DataFrame:
    duplicated = identifiers.duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"{name} has ambiguous duplicate configurations: "
            f"{sorted(set(identifiers[duplicated]))}"
        )
    table = pd.DataFrame({id_column: identifiers.to_numpy(dtype=str)})
    for quantity, column in source.items():
        table[quantity] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = table[list(QUANTITIES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite isolated profile values")
    if (table["runtime"] <= 0).any() or (table["mpi_time"] <= 0).any():
        raise ValueError(f"{name} requires positive runtime and MPI time denominators")
    if (table[["total_msgs", "total_bytes"]] < 0).any().any():
        raise ValueError(f"{name} contains negative message counts or byte volumes")
    table["comm_frac"] = table["mpi_time"] / table["runtime"]
    if not table["comm_frac"].between(0.0, 1.0).all():
        bad = table.loc[~table["comm_frac"].between(0.0, 1.0), id_column].tolist()
        raise ValueError(f"{name} has MPI time exceeding runtime for {bad}")
    table["num_nodes"] = int(num_nodes)
    return table


def build_application_profiles(isolated_apps: pd.DataFrame, num_nodes: int) -> pd.DataFrame:
    _require(isolated_apps, ["App", "Num Nodes", *APP_SOURCE.values()], "isolated apps")
    frame = _filter_nodes(isolated_apps, num_nodes)
    if frame.empty:
        raise ValueError(f"No isolated application rows for Num Nodes == {num_nodes}")
    identifiers = frame["App"].astype(str).reset_index(drop=True)
    table = _profile_table(
        frame.reset_index(drop=True), identifiers, "job_id", APP_SOURCE, num_nodes,
        "isolated application profiles",
    )
    columns = ["job_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes", "runtime", "num_nodes"]
    return table[columns].sort_values("job_id").reset_index(drop=True)


def build_inhibitor_profiles(
    isolated_inhibitors: pd.DataFrame, num_nodes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require(
        isolated_inhibitors,
        [*INHIB_CONFIG.values(), *INHIB_SOURCE.values()],
        "isolated inhibitors",
    )
    frame = _filter_nodes(isolated_inhibitors, num_nodes).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No isolated inhibitor rows for Num Nodes == {num_nodes}")
    identifiers = inhibitor_ids(frame)
    table = _profile_table(
        frame, identifiers, "inhib_id", INHIB_SOURCE, num_nodes, "isolated inhibitor profiles"
    )
    for key in ["msg_size", "wait_time", "comm_sparsity"]:
        table[key] = pd.to_numeric(frame[INHIB_CONFIG[key]]).to_numpy()
    mapping = pd.DataFrame(
        {
            "inhib_id": identifiers.to_numpy(dtype=str),
            **{key: frame[column].to_numpy() for key, column in INHIB_CONFIG.items()},
        }
    )
    columns = [
        "inhib_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes", "runtime",
        "num_nodes", "msg_size", "wait_time", "comm_sparsity",
    ]
    return (
        table[columns].sort_values("inhib_id").reset_index(drop=True),
        mapping.sort_values("inhib_id").reset_index(drop=True),
    )


def log_rate_degradation(
    volume_iso: np.ndarray,
    time_iso: np.ndarray,
    volume_co: np.ndarray,
    time_co: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signed log degradation, validity, reason) computed in log space.

    A zero or invalid rate makes the log ratio unavailable for that row only.
    """
    arrays = [np.asarray(value, dtype=float) for value in (volume_iso, time_iso, volume_co, time_co)]
    volume_iso, time_iso, volume_co, time_co = arrays
    reason = np.full(volume_iso.shape, "", dtype=object)
    finite = np.isfinite(volume_iso) & np.isfinite(time_iso) & np.isfinite(volume_co) & np.isfinite(time_co)
    reason[~finite] = "non_finite_measurement"
    nonpositive_time = finite & ((time_iso <= 0) | (time_co <= 0))
    reason[(reason == "") & nonpositive_time] = "nonpositive_time_denominator"
    negative = finite & ((volume_iso < 0) | (volume_co < 0))
    reason[(reason == "") & negative] = "negative_count_or_volume"
    zero = finite & ((volume_iso == 0) | (volume_co == 0))
    reason[(reason == "") & zero] = "zero_rate"
    valid = reason == ""
    result = np.full(volume_iso.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid] = (
            np.log(volume_iso[valid]) - np.log(time_iso[valid])
            - np.log(volume_co[valid]) + np.log(time_co[valid])
        )
    still_valid = valid & np.isfinite(result)
    reason[valid & ~still_valid] = "non_finite_log_ratio"
    result[~still_valid] = np.nan
    return result, still_valid, reason


def clipped_slowdown(
    mpi_co: np.ndarray, mpi_iso: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mpi_co = np.asarray(mpi_co, dtype=float)
    mpi_iso = np.asarray(mpi_iso, dtype=float)
    valid = np.isfinite(mpi_co) & np.isfinite(mpi_iso) & (mpi_iso > 0) & (mpi_co > 0)
    raw = np.full(mpi_co.shape, np.nan)
    raw[valid] = mpi_co[valid] / mpi_iso[valid]
    valid &= np.isfinite(raw)
    slowdown = np.where(valid, np.maximum(1.0, raw), np.nan)
    return slowdown, raw, valid


def _legacy_consistency(
    jobs: pd.DataFrame,
    inhibitors: pd.DataFrame,
    responses: pd.DataFrame,
    legacy_dir: Path,
    num_nodes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def compare(check: str, entity: str, field_name: str, reference: float, rebuilt: float) -> None:
        denominator = max(abs(reference), abs(rebuilt), 1e-300)
        relative = abs(reference - rebuilt) / denominator
        rows.append(
            {
                "check": check,
                "entity": entity,
                "field": field_name,
                "reference_source": str(legacy_dir),
                "reference_value": reference,
                "rebuilt_value": rebuilt,
                "relative_difference": relative,
                "expected_invariant": True,
                "consistent": bool(relative <= LEGACY_TOLERANCE),
            }
        )

    fields = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
    legacy_jobs_path = legacy_dir / "jobs.csv"
    if legacy_jobs_path.is_file():
        legacy_jobs = _read(legacy_jobs_path).set_index("job_id")
        rebuilt = jobs.set_index("job_id")
        for job_id in sorted(set(legacy_jobs.index) & set(rebuilt.index)):
            for name in fields:
                compare("legacy_application_profile", job_id, name,
                        float(legacy_jobs.loc[job_id, name]), float(rebuilt.loc[job_id, name]))
    legacy_inhibitors_path = legacy_dir / "inhibitors.csv"
    if legacy_inhibitors_path.is_file():
        legacy = _read(legacy_inhibitors_path)
        if {"msg_size", "wait_time", "comm_sparsity"}.issubset(legacy.columns):
            legacy = legacy.assign(
                canonical_id=[
                    canonical_inhibitor_id(num_nodes, *values)
                    for values in legacy[["msg_size", "wait_time", "comm_sparsity"]].itertuples(index=False)
                ]
            ).set_index("canonical_id")
            rebuilt = inhibitors.set_index("inhib_id")
            for inhib_id in sorted(set(legacy.index) & set(rebuilt.index)):
                for name in fields:
                    compare("legacy_inhibitor_profile", inhib_id, name,
                            float(legacy.loc[inhib_id, name]), float(rebuilt.loc[inhib_id, name]))
            legacy_responses_path = legacy_dir / "job_inh.csv"
            if legacy_responses_path.is_file():
                legacy_ids = dict(zip(legacy["inhib_id"].astype(str), legacy.index))
                old = _read(legacy_responses_path)
                old["canonical_id"] = old["inhib_id"].astype(str).map(legacy_ids)
                old = old.dropna(subset=["canonical_id"])
                old = old[~old.duplicated(["job_id", "canonical_id"], keep=False)]
                merged = old.merge(
                    responses[["job_id", "inhib_id", "slowdown"]],
                    left_on=["job_id", "canonical_id"],
                    right_on=["job_id", "inhib_id"],
                    suffixes=("_legacy", ""),
                )
                for row in merged.itertuples(index=False):
                    compare("legacy_app_inhibitor_slowdown", f"{row.job_id}|{row.canonical_id}",
                            "slowdown", float(row.slowdown_legacy), float(row.slowdown))
    return rows


def _work_volume_consistency(responses: pd.DataFrame, audit: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize whether co-scheduled work volumes match isolated volumes."""
    rows = []
    joined = responses[["observation_id"]].merge(audit, on="observation_id", validate="one_to_one")
    for endpoint, entity_column, invariant in [("app", "job_id", True), ("inhib", "inhib_id", False)]:
        for quantity in ["total_msgs", "total_bytes"]:
            iso = joined[f"{endpoint}_iso_{quantity}"].to_numpy(float)
            co = joined[f"{endpoint}_co_{quantity}"].to_numpy(float)
            relative = np.abs(co - iso) / np.maximum(np.maximum(np.abs(co), np.abs(iso)), 1e-300)
            frame = pd.DataFrame({"entity": joined[entity_column], "relative": relative})
            for entity, group in frame.groupby("entity"):
                worst = float(group["relative"].max())
                rows.append(
                    {
                        "check": f"{endpoint}_work_volume_co_vs_isolated",
                        "entity": entity,
                        "field": quantity,
                        "reference_source": "isolated profile",
                        "reference_value": np.nan,
                        "rebuilt_value": np.nan,
                        "relative_difference": worst,
                        "expected_invariant": invariant,
                        "consistent": bool(worst <= LEGACY_TOLERANCE) if invariant else True,
                    }
                )
    return rows


def prepare_training_data(
    isolated_apps_csv: Path | str,
    isolated_inhibitors_csv: Path | str,
    app_inhibitor_csv: Path | str,
    num_nodes: int,
    output_dir: Path | str,
    legacy_dir: Path | str | None = None,
) -> PreparationReport:
    """Build jobs/inhibitors/job_inh from one snapshot. Never reads App-App data."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "isolated_apps_csv": Path(isolated_apps_csv),
        "isolated_inhibitors_csv": Path(isolated_inhibitors_csv),
        "app_inhibitor_csv": Path(app_inhibitor_csv),
    }
    jobs = build_application_profiles(_read(sources["isolated_apps_csv"]), num_nodes)
    inhibitors, id_map = build_inhibitor_profiles(_read(sources["isolated_inhibitors_csv"]), num_nodes)

    co_all = _read(sources["app_inhibitor_csv"])
    _require(
        co_all,
        ["App", *INHIB_CONFIG.values(), *APP_SOURCE.values(), *INHIB_SOURCE.values()],
        "App-Inhibitor co-scheduled data",
    )
    co = _filter_nodes(co_all, num_nodes).reset_index(drop=True)
    co["job_id"] = co["App"].astype(str)
    co["inhib_id"] = inhibitor_ids(co).to_numpy(dtype=str)
    reasons = np.full(len(co), "", dtype=object)
    ambiguous = co.duplicated(["job_id", "inhib_id"], keep=False).to_numpy()
    reasons[ambiguous] = "ambiguous_duplicate_configuration"
    missing_app = ~co["job_id"].isin(jobs["job_id"]).to_numpy()
    reasons[(reasons == "") & missing_app] = "missing_join_app_profile"
    missing_inhib = ~co["inhib_id"].isin(inhibitors["inhib_id"]).to_numpy()
    reasons[(reasons == "") & missing_inhib] = "missing_join_inhibitor_profile"

    app_iso = jobs.set_index("job_id")
    inhib_iso = inhibitors.set_index("inhib_id")
    joinable = reasons == ""
    audit = pd.DataFrame(index=co.index)
    for quantity in QUANTITIES:
        audit[f"app_iso_{quantity}"] = co["job_id"].map(app_iso[quantity]).to_numpy(float)
        audit[f"app_co_{quantity}"] = pd.to_numeric(co[APP_SOURCE[quantity]], errors="coerce").to_numpy(float)
        audit[f"inhib_iso_{quantity}"] = co["inhib_id"].map(inhib_iso[quantity]).to_numpy(float)
        audit[f"inhib_co_{quantity}"] = pd.to_numeric(co[INHIB_SOURCE[quantity]], errors="coerce").to_numpy(float)

    slowdown, raw, slowdown_valid = clipped_slowdown(audit["app_co_mpi_time"], audit["app_iso_mpi_time"])
    reasons[(reasons == "") & ~slowdown_valid] = "invalid_slowdown_measurement"

    responses = pd.DataFrame(
        {
            "job_id": co["job_id"],
            "inhib_id": co["inhib_id"],
            "replicate_id": 0,
            "num_nodes": int(num_nodes),
            "slowdown": slowdown,
            "log_slowdown": np.log(slowdown),
            "is_censored": raw <= 1.0,
            "raw_mpi_time_ratio": raw,
        }
    )
    responses.insert(2, "observation_id", responses["job_id"] + "|" + responses["inhib_id"] + "|0")
    availability_rows = []
    for endpoint in ["app", "inhib"]:
        for task in AUX_TASKS:
            definition = AUX_TASK_DEFINITIONS[task]
            value, valid, reason = log_rate_degradation(
                audit[f"{endpoint}_iso_{definition['numerator']}"],
                audit[f"{endpoint}_iso_{definition['denominator']}"],
                audit[f"{endpoint}_co_{definition['numerator']}"],
                audit[f"{endpoint}_co_{definition['denominator']}"],
            )
            responses[aux_target_column(endpoint, task)] = value
            responses[aux_mask_column(endpoint, task)] = valid
            audit[f"{endpoint}_{task}_mask_reason"] = reason
            kept = reasons == ""
            counts = pd.Series(reason[kept]).replace("", "valid").value_counts().to_dict()
            availability_rows.append(
                {
                    "endpoint": endpoint,
                    "task": task,
                    "target_column": aux_target_column(endpoint, task),
                    "observations": int(kept.sum()),
                    "valid": int(valid[kept].sum()),
                    "masked": int((~valid[kept]).sum()),
                    "negative_increased_rate": int((valid[kept] & (value[kept] < 0)).sum()),
                    "zero_change": int((valid[kept] & (value[kept] == 0)).sum()),
                    "mask_reasons": json.dumps({key: int(v) for key, v in counts.items() if key != "valid"}),
                }
            )
    audit.insert(0, "observation_id", responses["observation_id"])
    audit.insert(1, "job_id", responses["job_id"])
    audit.insert(2, "inhib_id", responses["inhib_id"])
    audit["raw_mpi_time_ratio"] = raw
    audit["slowdown"] = slowdown
    audit["rejection_reason"] = reasons

    rejected = pd.concat(
        [co.loc[reasons != "", ["job_id", "inhib_id"]].reset_index(drop=True),
         audit.loc[reasons != ""].drop(columns=["job_id", "inhib_id"]).reset_index(drop=True)],
        axis=1,
    )
    kept = responses.loc[reasons == ""].reset_index(drop=True)
    kept_audit = audit.loc[reasons == ""].reset_index(drop=True)
    missing_apps = sorted(set(jobs["job_id"]) - set(kept["job_id"]))
    if missing_apps:
        raise ValueError(f"Applications have no valid App-Inhibitor responses: {missing_apps}")

    consistency = _work_volume_consistency(kept, kept_audit)
    if legacy_dir is not None and Path(legacy_dir).is_dir():
        consistency.extend(_legacy_consistency(jobs, inhibitors, kept, Path(legacy_dir), num_nodes))
    consistency_frame = pd.DataFrame(
        consistency,
        columns=["check", "entity", "field", "reference_source", "reference_value",
                 "rebuilt_value", "relative_difference", "expected_invariant", "consistent"],
    )

    jobs.to_csv(output / "jobs.csv", index=False)
    inhibitors.to_csv(output / "inhibitors.csv", index=False)
    kept.to_csv(output / "job_inh.csv", index=False)
    id_map.to_csv(output / "inhibitor_id_map.csv", index=False)
    rejected.to_csv(output / "rejected_observations.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(output / "auxiliary_target_availability.csv", index=False)
    kept_audit.to_csv(output / "observation_audit.csv", index=False)
    consistency_frame.to_csv(output / "source_consistency_report.csv", index=False)

    inconsistent = consistency_frame[~consistency_frame["consistent"].astype(bool)]
    report = PreparationReport(
        kind="training",
        output_dir=str(output.resolve()),
        row_counts={
            "isolated_application_profiles": len(jobs),
            "isolated_inhibitor_profiles": len(inhibitors),
            "app_inhibitor_source_rows": len(co_all),
            "app_inhibitor_rows_after_node_filter": len(co),
            "prepared_observations": len(kept),
            "rejected_observations": len(rejected),
            "censored_floor_observations": int(kept["is_censored"].sum()),
        },
        source_sha256={name: sha256_file(path) for name, path in sources.items()},
        details={
            "schema_version": SCHEMA_VERSION,
            "app_app_data_read": False,
            "num_nodes": int(num_nodes),
            "filters": {"Num Nodes": int(num_nodes)},
            "source_paths": {name: str(path.resolve()) for name, path in sources.items()},
            "observation_unit": OBSERVATION_UNIT,
            "rate_estimator": RATE_ESTIMATOR,
            "replicate_semantics": "one aggregate configuration per (job_id, inhib_id); replicate_id = 0; not run-level replicates",
            "inhibitor_id_format": "{Num Nodes}_{Inhib Message Size}_{Inhib Wait Time (us)}_{decimal-normalized Inhib Comm Sparsity}",
            "target_definitions": {
                "slowdown": "max(1, T_MPI_co / T_MPI_iso) for the application endpoint",
                "log_slowdown": "log(slowdown)",
                "is_censored": "raw_mpi_time_ratio <= 1 (observation at the slowdown floor)",
                "raw_mpi_time_ratio": "unclipped T_MPI_co / T_MPI_iso; diagnostic only",
                "auxiliary": {
                    task: {**definition, "target": "log(rate_iso) - log(rate_co); positive = degradation; not clipped"}
                    for task, definition in AUX_TASK_DEFINITIONS.items()
                },
                "comm_frac": "T_MPI / T_runtime of the matching aggregate means",
            },
            "units": UNITS,
            "rejection_reasons": pd.Series(reasons[reasons != ""]).value_counts().to_dict(),
            "auxiliary_availability": availability_rows,
            "legacy_comparison_dir": str(Path(legacy_dir).resolve()) if legacy_dir and Path(legacy_dir).is_dir() else None,
            "source_inconsistency_count": int(len(inconsistent)),
            "source_inconsistent_checks": inconsistent.groupby("check").size().to_dict() if len(inconsistent) else {},
        },
    )
    (output / "preparation_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=_json_default) + "\n"
    )
    return report


def prepare_pair_evaluation_data(
    app_app_csv: Path | str,
    prepared_jobs_csv: Path | str,
    num_nodes: int,
    output_dir: Path | str,
) -> PreparationReport:
    """Build directional App-App labels for final evaluation only."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = _read(prepared_jobs_csv)
    _require(jobs, ["job_id", *QUANTITIES], "prepared jobs")
    jobs["job_id"] = jobs["job_id"].astype(str)
    if jobs["job_id"].duplicated().any():
        raise ValueError("prepared jobs contain duplicate job_id values")
    source = _read(app_app_csv)
    _require(source, ["App A", "App B", "Num Nodes", *pair_source_columns("A").values(),
                      *pair_source_columns("B").values()], "App-App data")
    pairs = _filter_nodes(source, num_nodes).reset_index(drop=True)
    pairs["jobA_id"] = pairs["App A"].astype(str)
    pairs["jobB_id"] = pairs["App B"].astype(str)
    reasons = np.full(len(pairs), "", dtype=object)
    unordered = pairs.apply(lambda row: tuple(sorted((row["jobA_id"], row["jobB_id"]))), axis=1)
    reasons[unordered.duplicated(keep=False).to_numpy()] = "ambiguous_duplicate_unordered_pair"
    missing = ~(pairs["jobA_id"].isin(jobs["job_id"]) & pairs["jobB_id"].isin(jobs["job_id"])).to_numpy()
    reasons[(reasons == "") & missing] = "missing_join_app_profile"
    iso = jobs.set_index("job_id")
    result = pd.DataFrame({"jobA_id": pairs["jobA_id"], "jobB_id": pairs["jobB_id"], "num_nodes": int(num_nodes)})
    audit = pd.DataFrame({"jobA_id": pairs["jobA_id"], "jobB_id": pairs["jobB_id"]})
    for endpoint in ["A", "B"]:
        columns = pair_source_columns(endpoint)
        for quantity in QUANTITIES:
            audit[f"{endpoint}_iso_{quantity}"] = pairs[f"job{endpoint}_id"].map(iso[quantity]).to_numpy(float)
            audit[f"{endpoint}_co_{quantity}"] = pd.to_numeric(pairs[columns[quantity]], errors="coerce").to_numpy(float)
        slowdown, raw, valid = clipped_slowdown(audit[f"{endpoint}_co_mpi_time"], audit[f"{endpoint}_iso_mpi_time"])
        reasons[(reasons == "") & ~valid] = f"invalid_slowdown_measurement_{endpoint}"
        result[f"slowdown_{endpoint}"] = slowdown
        result[f"log_slowdown_{endpoint}"] = np.log(slowdown)
        result[f"is_censored_{endpoint}"] = raw <= 1.0
        result[f"raw_mpi_time_ratio_{endpoint}"] = raw
    availability_rows = []
    for endpoint in ["A", "B"]:
        for task in AUX_TASKS:
            definition = AUX_TASK_DEFINITIONS[task]
            value, valid, reason = log_rate_degradation(
                audit[f"{endpoint}_iso_{definition['numerator']}"],
                audit[f"{endpoint}_iso_{definition['denominator']}"],
                audit[f"{endpoint}_co_{definition['numerator']}"],
                audit[f"{endpoint}_co_{definition['denominator']}"],
            )
            result[aux_target_column(endpoint, task)] = value
            result[aux_mask_column(endpoint, task)] = valid
            audit[f"{endpoint}_{task}_mask_reason"] = reason
            kept = reasons == ""
            availability_rows.append(
                {"endpoint": endpoint, "task": task, "observations": int(kept.sum()),
                 "valid": int(valid[kept].sum()), "masked": int((~valid[kept]).sum()),
                 "negative_increased_rate": int((valid[kept] & (value[kept] < 0)).sum())}
            )
    audit["rejection_reason"] = reasons
    kept = result.loc[reasons == ""].reset_index(drop=True)
    kept.to_csv(output / "pair.csv", index=False)
    audit.to_csv(output / "pair_observation_audit.csv", index=False)
    audit.loc[reasons != ""].to_csv(output / "rejected_pairs.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(output / "auxiliary_target_availability.csv", index=False)
    report = PreparationReport(
        kind="evaluation",
        output_dir=str(output.resolve()),
        row_counts={
            "app_app_source_rows": len(source),
            "app_app_rows_after_node_filter": len(pairs),
            "prepared_pairs": len(kept),
            "rejected_pairs": int((reasons != "").sum()),
            "self_pairs": int((kept["jobA_id"] == kept["jobB_id"]).sum()),
        },
        source_sha256={
            "app_app_csv": sha256_file(app_app_csv),
            "prepared_jobs_csv": sha256_file(prepared_jobs_csv),
        },
        details={
            "schema_version": SCHEMA_VERSION,
            "num_nodes": int(num_nodes),
            "observation_unit": OBSERVATION_UNIT,
            "rate_estimator": RATE_ESTIMATOR,
            "use": "final evaluation only; never used for fitting, selection, or context construction",
            "rejection_reasons": pd.Series(reasons[reasons != ""]).value_counts().to_dict(),
            "auxiliary_availability": availability_rows,
            "units": UNITS,
        },
    )
    (output / "preparation_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=_json_default) + "\n"
    )
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("training", help="prepare App-Inhibitor training data")
    training.add_argument("--isolated-apps-csv", type=Path, default=PROCESSED / "data_isolated_agg.csv")
    training.add_argument("--isolated-inhibitors-csv", type=Path, default=PROCESSED / "data_inhib_isolated_agg.csv")
    training.add_argument("--app-inhibitor-csv", type=Path, default=PROCESSED / "data_inhib_coscheduled_agg.csv")
    training.add_argument("--num-nodes", type=int, default=8)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    training.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR,
                          help="optional legacy few_shot data used only for the source-consistency report")
    training.add_argument("--no-legacy-comparison", action="store_true")
    evaluation = commands.add_parser("evaluation", help="prepare App-App evaluation labels")
    evaluation.add_argument("--app-app-csv", type=Path, default=PROCESSED / "data_coscheduled_agg.csv")
    evaluation.add_argument("--jobs-csv", type=Path, default=DEFAULT_TRAINING_DIR / "jobs.csv")
    evaluation.add_argument("--num-nodes", type=int, default=8)
    evaluation.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "training":
        report = prepare_training_data(
            args.isolated_apps_csv, args.isolated_inhibitors_csv, args.app_inhibitor_csv,
            args.num_nodes, args.output_dir,
            legacy_dir=None if args.no_legacy_comparison else args.legacy_dir,
        )
    else:
        report = prepare_pair_evaluation_data(args.app_app_csv, args.jobs_csv, args.num_nodes, args.output_dir)
    print(json.dumps(report.row_counts, indent=2))
    if report.kind == "training":
        print(f"Source inconsistencies flagged: {report.details['source_inconsistency_count']}")
    print(f"Outputs: {report.output_dir}")


if __name__ == "__main__":
    main()
