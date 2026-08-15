"""Leakage-safe loading and profile transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


BASE_FEATURES = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
OBSERVED_LOG_FLOOR = 0.0
MAX_EXP_LOG = float(np.log(np.finfo(np.float64).max))
DERIVED_FEATURES = [
    "estimated_runtime",
    "mean_message_size",
    "message_rate",
    "byte_rate",
]


def _read_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def observe_log_slowdown(latent_log_slowdown: np.ndarray | float) -> np.ndarray:
    """Map finite latent logs to the clipped observed target support."""
    values = np.asarray(latent_log_slowdown, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Latent log-slowdown predictions must be finite")
    return np.maximum(values, OBSERVED_LOG_FLOOR)


def log_slowdown_to_slowdown(observed_log_slowdown: np.ndarray | float) -> np.ndarray:
    """Exponentiate observed logs after explicit support and overflow checks."""
    values = np.asarray(observed_log_slowdown, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Observed log-slowdown predictions must be finite")
    if (values < OBSERVED_LOG_FLOOR).any():
        raise ValueError("Observed log-slowdown predictions must be at least zero")
    if (values > MAX_EXP_LOG).any():
        raise OverflowError(
            f"Log-slowdown prediction exceeds the finite exp limit {MAX_EXP_LOG:.6g}"
        )
    result = np.exp(values)
    if not np.isfinite(result).all() or (result < 1.0).any():
        raise FloatingPointError("Slowdown prediction left the finite observed support")
    return result


def _validate_profiles(frame: pd.DataFrame, id_column: str, name: str) -> None:
    _require_columns(frame, [id_column, *BASE_FEATURES], name)
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"{name}.{id_column} must be non-null and unique")
    values = frame[BASE_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} profile values must be finite and non-negative")
    if not frame["comm_frac"].between(0.0, 1.0).all():
        raise ValueError(f"{name}.comm_frac must lie in [0, 1]")


@dataclass
class TrainingData:
    jobs: pd.DataFrame
    inhibitors: pd.DataFrame
    responses: pd.DataFrame
    rejected_responses: pd.DataFrame
    duplicate_keys: pd.DataFrame


EVALUATION_METHODS = {"random_split", "one_known", "zero_shot"}
EVALUATION_MODES = EVALUATION_METHODS | {"both"}


def resolve_evaluation_methods(mode: str) -> list[str]:
    if mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation mode must be one of {sorted(EVALUATION_MODES)}")
    return ["one_known", "zero_shot"] if mode == "both" else [mode]


def subset_training_data(data: TrainingData, training_apps: list[str]) -> TrainingData:
    """Restrict model-fitting inputs to an explicit set of applications."""
    requested = list(dict.fromkeys(str(value) for value in training_apps))
    available = set(data.jobs["job_id"].astype(str))
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Training applications are missing from jobs.csv: {unknown}")
    if len(requested) < 2:
        raise ValueError("At least two training applications are required")
    selected = set(requested)
    jobs = data.jobs[data.jobs["job_id"].astype(str).isin(selected)].copy()
    responses = data.responses[
        data.responses["job_id"].astype(str).isin(selected)
    ].copy()
    missing_responses = sorted(selected - set(responses["job_id"].astype(str)))
    if missing_responses:
        raise ValueError(
            "Training applications have no valid App-Inhibitor responses: "
            f"{missing_responses}"
        )
    rejected = data.rejected_responses.copy()
    if "job_id" in rejected:
        rejected = rejected[rejected["job_id"].astype(str).isin(selected)]
    duplicates = data.duplicate_keys.copy()
    if "job_id" in duplicates:
        duplicates = duplicates[duplicates["job_id"].astype(str).isin(selected)]
    return TrainingData(
        jobs.reset_index(drop=True),
        data.inhibitors.copy(),
        responses.reset_index(drop=True),
        rejected.reset_index(drop=True),
        duplicates.reset_index(drop=True),
    )


def select_evaluation_pairs(
    pairs: pd.DataFrame,
    known_apps: list[str] | set[str],
    evaluation_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select canonical App-App rows for one endpoint-familiarity tier."""
    if evaluation_method not in EVALUATION_METHODS:
        raise ValueError(
            f"evaluation_method must be one of {sorted(EVALUATION_METHODS)}"
        )
    known = {str(value) for value in known_apps}
    left_known = pairs["jobA_id"].astype(str).isin(known)
    right_known = pairs["jobB_id"].astype(str).isin(known)
    if evaluation_method == "random_split":
        included = left_known & right_known
    elif evaluation_method == "one_known":
        included = left_known ^ right_known
    else:
        included = ~left_known & ~right_known

    manifest = pairs[
        ["pair_row_id", "pair_cluster_id", "jobA_id", "jobB_id"]
    ].copy()
    manifest["evaluation_method"] = evaluation_method
    manifest["jobA_known"] = left_known.to_numpy(dtype=bool)
    manifest["jobB_known"] = right_known.to_numpy(dtype=bool)
    manifest["known_endpoint_count"] = (
        left_known.astype(int) + right_known.astype(int)
    ).to_numpy()
    manifest["included"] = included.to_numpy(dtype=bool)

    selected = pairs.loc[included].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"No App-App rows are eligible for evaluation_method={evaluation_method!r}"
        )
    return selected, manifest


class ProfileScaler:
    """Transforms physical profiles and standardizes with fold-local statistics."""

    def __init__(self, feature_set: str = "base", epsilon: float = 1e-12):
        if feature_set not in {"base", "augmented"}:
            raise ValueError("feature_set must be 'base' or 'augmented'")
        self.feature_set = feature_set
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        return 4 if self.feature_set == "base" else 8

    def _physical_transform(self, frame: pd.DataFrame) -> np.ndarray:
        mpi_time = frame["mpi_time"].to_numpy(dtype=float)
        comm_frac = frame["comm_frac"].to_numpy(dtype=float)
        total_msgs = frame["total_msgs"].to_numpy(dtype=float)
        total_bytes = frame["total_bytes"].to_numpy(dtype=float)
        columns = [
            np.log1p(mpi_time),
            comm_frac,
            np.log1p(total_msgs),
            np.log1p(total_bytes),
        ]
        if self.feature_set == "augmented":
            runtime = mpi_time / np.maximum(comm_frac, self.epsilon)
            mean_size = total_bytes / np.maximum(total_msgs, self.epsilon)
            columns.extend(
                [
                    np.log1p(runtime),
                    np.log1p(mean_size),
                    np.log1p(total_msgs / np.maximum(runtime, self.epsilon)),
                    np.log1p(total_bytes / np.maximum(runtime, self.epsilon)),
                ]
            )
        return np.column_stack(columns)

    def fit(self, frame: pd.DataFrame) -> "ProfileScaler":
        values = self._physical_transform(frame)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ProfileScaler must be fit before transform")
        return ((self._physical_transform(frame) - self.mean_) / self.scale_).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "epsilon": self.epsilon,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ProfileScaler":
        scaler = cls(str(state["feature_set"]), float(state["epsilon"]))
        scaler.mean_ = np.asarray(state["mean"], dtype=float)
        scaler.scale_ = np.asarray(state["scale"], dtype=float)
        return scaler


def load_training_data(
    jobs_csv: Path | str,
    inhibitors_csv: Path | str,
    responses_csv: Path | str,
    quarantine_dir: Path | str | None = None,
) -> TrainingData:
    """Load App-Inhibitor data without opening the App-App holdout."""
    jobs = _read_csv(jobs_csv)
    inhibitors = _read_csv(inhibitors_csv)
    responses = _read_csv(responses_csv)
    _validate_profiles(jobs, "job_id", "jobs.csv")
    _validate_profiles(inhibitors, "inhib_id", "inhibitors.csv")
    _require_columns(responses, ["job_id", "inhib_id", "slowdown"], "job_inh.csv")

    jobs["job_id"] = jobs["job_id"].astype(str)
    inhibitors["inhib_id"] = inhibitors["inhib_id"].astype(str)
    responses["job_id"] = responses["job_id"].astype(str)
    responses["inhib_id"] = responses["inhib_id"].astype(str)
    slowdown = pd.to_numeric(responses["slowdown"], errors="coerce")
    reasons = np.full(len(responses), "", dtype=object)
    invalid_slowdown = ~np.isfinite(slowdown) | (slowdown < 1.0)
    reasons[invalid_slowdown] = "below_observed_support_or_non_finite_slowdown"
    unknown_job = ~responses["job_id"].isin(jobs["job_id"])
    reasons[(reasons == "") & unknown_job] = "missing_job_profile"
    unknown_inhibitor = ~responses["inhib_id"].isin(inhibitors["inhib_id"])
    reasons[(reasons == "") & unknown_inhibitor] = "missing_inhibitor_profile"

    rejected = responses.loc[reasons != ""].copy()
    rejected["rejection_reason"] = reasons[reasons != ""]
    valid = responses.loc[reasons == ""].copy()
    valid["slowdown"] = slowdown.loc[reasons == ""].astype(float)
    valid["log_slowdown"] = np.log(valid["slowdown"])
    valid["is_censored"] = valid["slowdown"].eq(1.0)
    valid["replicate_id"] = valid.groupby(["job_id", "inhib_id"]).cumcount()
    duplicate_mask = valid.duplicated(["job_id", "inhib_id"], keep=False)
    duplicates = valid.loc[duplicate_mask].copy()

    if len(rejected):
        warnings.warn(f"Quarantined {len(rejected)} invalid App-Inhibitor rows")
    if len(duplicates):
        warnings.warn(
            f"Detected {len(duplicates)} rows with duplicate App-Inhibitor keys; "
            "they are preserved as replicates"
        )
    counts = valid.groupby("job_id").size()
    missing_jobs = sorted(set(jobs["job_id"]) - set(counts.index))
    if missing_jobs:
        raise ValueError(f"Applications have no valid response anchors: {missing_jobs}")

    if quarantine_dir is not None:
        output = Path(quarantine_dir)
        output.mkdir(parents=True, exist_ok=True)
        rejected.to_csv(output / "rejected_job_inh.csv", index=False)
        duplicates.to_csv(output / "duplicate_job_inh_keys.csv", index=False)

    return TrainingData(jobs, inhibitors, valid.reset_index(drop=True), rejected, duplicates)


def load_pair_holdout(path: Path | str, jobs: pd.DataFrame) -> pd.DataFrame:
    """Open and validate pair.csv. Call only after every model choice is fixed."""
    pairs = _read_csv(path)
    required = ["jobA_id", "jobB_id", "slowdown_A", "slowdown_B"]
    _require_columns(pairs, required, "pair.csv")
    pairs["jobA_id"] = pairs["jobA_id"].astype(str)
    pairs["jobB_id"] = pairs["jobB_id"].astype(str)
    known = set(jobs["job_id"].astype(str))
    unknown = (set(pairs["jobA_id"]) | set(pairs["jobB_id"])) - known
    if unknown:
        raise ValueError(f"pair.csv references unknown jobs: {sorted(unknown)}")
    values = pairs[["slowdown_A", "slowdown_B"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 1.0).any():
        raise ValueError("pair.csv slowdowns must be finite and at least 1")
    canonical = pairs.apply(
        lambda row: tuple(sorted((str(row["jobA_id"]), str(row["jobB_id"])))), axis=1
    )
    duplicate = canonical.duplicated(keep=False)
    if duplicate.any():
        keys = sorted(set(canonical[duplicate]))
        raise ValueError(
            "pair.csv contains duplicate unordered pairs (including reversals): "
            f"{keys}"
        )
    pairs.insert(0, "pair_row_id", np.arange(len(pairs), dtype=int))
    pairs.insert(1, "pair_cluster_id", pairs["pair_row_id"])
    return pairs


def expand_directional_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame(
        {
            "pair_row_id": pairs["pair_row_id"],
            "pair_cluster_id": pairs.get("pair_cluster_id", pairs["pair_row_id"]),
            "direction": "A",
            "victim_id": pairs["jobA_id"],
            "aggressor_id": pairs["jobB_id"],
            "true_slowdown": pairs["slowdown_A"],
        }
    )
    b = pd.DataFrame(
        {
            "pair_row_id": pairs["pair_row_id"],
            "pair_cluster_id": pairs.get("pair_cluster_id", pairs["pair_row_id"]),
            "direction": "B",
            "victim_id": pairs["jobB_id"],
            "aggressor_id": pairs["jobA_id"],
            "true_slowdown": pairs["slowdown_B"],
        }
    )
    return pd.concat([a, b], ignore_index=True)
