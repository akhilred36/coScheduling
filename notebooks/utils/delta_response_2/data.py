"""App-Inhibitor loading, profile transforms, and label-free target geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


BASE_FEATURES = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
DERIVED_FEATURES = [
    "estimated_runtime",
    "mean_message_size",
    "message_rate",
    "byte_rate",
]
OBSERVED_LOG_FLOOR = 0.0
MAX_EXP_LOG = float(np.log(np.finfo(np.float64).max))


def read_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def observe_log_slowdown(latent: np.ndarray | float) -> np.ndarray:
    values = np.asarray(latent, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Latent log-slowdown predictions must be finite")
    return np.maximum(values, OBSERVED_LOG_FLOOR)


def log_to_slowdown(observed_log: np.ndarray | float) -> np.ndarray:
    values = np.asarray(observed_log, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Observed log-slowdown predictions must be finite")
    if (values < OBSERVED_LOG_FLOOR).any():
        raise ValueError("Observed log-slowdown predictions must be non-negative")
    if (values > MAX_EXP_LOG).any():
        raise OverflowError("Log-slowdown prediction exceeds the finite exp limit")
    result = np.exp(values)
    if not np.isfinite(result).all() or (result < 1.0).any():
        raise FloatingPointError("Prediction left the observed slowdown support")
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


@dataclass(frozen=True)
class TrainingData:
    jobs: pd.DataFrame
    inhibitors: pd.DataFrame
    responses: pd.DataFrame
    rejected_responses: pd.DataFrame
    duplicate_keys: pd.DataFrame


class ProfileScaler:
    """Physical transform followed by fold-local standardization."""

    def __init__(self, feature_set: str = "base", epsilon: float = 1e-12):
        if feature_set not in {"base", "augmented"}:
            raise ValueError("feature_set must be 'base' or 'augmented'")
        self.feature_set = feature_set
        self.epsilon = float(epsilon)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        return 4 if self.feature_set == "base" else 8

    def physical_transform(self, frame: pd.DataFrame) -> np.ndarray:
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
        values = np.column_stack(columns)
        if not np.isfinite(values).all():
            raise ValueError("Profile transform produced non-finite values")
        return values

    def fit(self, frame: pd.DataFrame) -> "ProfileScaler":
        values = self.physical_transform(frame)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ProfileScaler must be fit before transform")
        return ((self.physical_transform(frame) - self.mean_) / self.scale_).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "epsilon": self.epsilon,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
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
) -> TrainingData:
    """Load only App-Inhibitor inputs; this module has no pair-data API."""
    jobs = read_csv(jobs_csv)
    inhibitors = read_csv(inhibitors_csv)
    responses = read_csv(responses_csv)
    _validate_profiles(jobs, "job_id", "jobs.csv")
    _validate_profiles(inhibitors, "inhib_id", "inhibitors.csv")
    _require_columns(responses, ["job_id", "inhib_id", "slowdown"], "job_inh.csv")

    jobs["job_id"] = jobs["job_id"].astype(str)
    inhibitors["inhib_id"] = inhibitors["inhib_id"].astype(str)
    responses["job_id"] = responses["job_id"].astype(str)
    responses["inhib_id"] = responses["inhib_id"].astype(str)
    responses.insert(0, "source_row", np.arange(len(responses), dtype=int))

    slowdown = pd.to_numeric(responses["slowdown"], errors="coerce")
    reasons = np.full(len(responses), "", dtype=object)
    invalid = ~np.isfinite(slowdown) | (slowdown < 1.0)
    reasons[invalid] = "below_observed_support_or_non_finite_slowdown"
    unknown_job = ~responses["job_id"].isin(jobs["job_id"])
    reasons[(reasons == "") & unknown_job] = "missing_job_profile"
    unknown_inhibitor = ~responses["inhib_id"].isin(inhibitors["inhib_id"])
    reasons[(reasons == "") & unknown_inhibitor] = "missing_inhibitor_profile"

    rejected = responses.loc[reasons != ""].copy()
    rejected["rejection_reason"] = reasons[reasons != ""]
    valid = responses.loc[reasons == ""].copy()
    valid["slowdown"] = slowdown.loc[reasons == ""].astype(float)
    valid["log_slowdown"] = np.log(valid["slowdown"])
    valid["is_censored"] = valid["log_slowdown"].eq(0.0)
    valid["replicate_id"] = valid.groupby(["job_id", "inhib_id"]).cumcount()
    valid.insert(0, "response_id", np.arange(len(valid), dtype=int))
    duplicates = valid.loc[
        valid.duplicated(["job_id", "inhib_id"], keep=False)
    ].copy()

    if len(rejected):
        warnings.warn(f"Quarantined {len(rejected)} invalid App-Inhibitor rows")
    if len(duplicates):
        warnings.warn(
            f"Preserved {len(duplicates)} duplicate App-Inhibitor rows as replicates"
        )
    missing_jobs = sorted(set(jobs["job_id"]) - set(valid["job_id"]))
    if missing_jobs:
        raise ValueError(f"Applications have no valid responses: {missing_jobs}")
    return TrainingData(
        jobs.reset_index(drop=True),
        inhibitors.reset_index(drop=True),
        valid.reset_index(drop=True),
        rejected.reset_index(drop=True),
        duplicates.reset_index(drop=True),
    )


def profile_maps(
    jobs: pd.DataFrame, inhibitors: pd.DataFrame, scaler: ProfileScaler
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        dict(zip(jobs["job_id"].astype(str), scaler.transform(jobs))),
        dict(zip(inhibitors["inhib_id"].astype(str), scaler.transform(inhibitors))),
    )


def app_profile_weights(
    jobs: pd.DataFrame,
    inhibitors: pd.DataFrame,
    *,
    nearest_per_application: int = 10,
    ratio_clip: tuple[float, float] = (0.05, 20.0),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute label-free density-ratio weights toward application profiles."""
    if nearest_per_application <= 0:
        raise ValueError("nearest_per_application must be positive")
    low, high = map(float, ratio_clip)
    if not 0 < low <= high:
        raise ValueError("ratio_clip must contain positive ordered bounds")

    scaler = ProfileScaler("base").fit(inhibitors)
    inhibitor_values = scaler.transform(inhibitors).astype(float)
    app_values = scaler.transform(jobs).astype(float)
    app_distances = np.linalg.norm(
        inhibitor_values[:, None, :] - app_values[None, :, :], axis=-1
    )
    inhibitor_distances = np.linalg.norm(
        inhibitor_values[:, None, :] - inhibitor_values[None, :, :], axis=-1
    )
    np.fill_diagonal(inhibitor_distances, np.inf)
    bandwidth = float(np.median(app_distances.min(axis=1)))
    bandwidth = max(bandwidth, 1e-6)
    numerator = np.exp(-(app_distances**2) / (2.0 * bandwidth**2)).mean(axis=1)
    denominator_kernel = np.exp(
        -(inhibitor_distances**2) / (2.0 * bandwidth**2)
    )
    denominator_kernel[~np.isfinite(inhibitor_distances)] = 0.0
    denominator = denominator_kernel.sum(axis=1) / max(len(inhibitors) - 1, 1)
    raw = numerator / np.maximum(denominator, 1e-12)
    clipped = np.clip(raw, low, high)
    normalized = clipped / clipped.mean()

    nearest_count = np.zeros(len(inhibitors), dtype=int)
    k = min(nearest_per_application, len(inhibitors))
    for app_index in range(len(jobs)):
        nearest_count[np.argsort(app_distances[:, app_index], kind="stable")[:k]] += 1
    nearest_app_index = app_distances.argmin(axis=1)
    result = pd.DataFrame(
        {
            "inhib_id": inhibitors["inhib_id"].astype(str),
            "nearest_app_id": jobs.iloc[nearest_app_index]["job_id"].to_numpy(),
            "nearest_app_distance": app_distances.min(axis=1),
            "app_nearest_count": nearest_count,
            "app_nearest_subset": nearest_count > 0,
            "profile_weight_raw": raw,
            "profile_weight_clipped": clipped,
            "profile_weight": normalized,
        }
    )
    state = {
        "declared_transductive": True,
        "uses_outcome_labels": False,
        "application_count": len(jobs),
        "inhibitor_count": len(inhibitors),
        "nearest_per_application": k,
        "ratio_clip": [low, high],
        "bandwidth": bandwidth,
        "effective_sample_size": float(normalized.sum() ** 2 / np.sum(normalized**2)),
        "scaler": scaler.state_dict(),
    }
    return result, state


def inhibitor_blocks(
    inhibitors: pd.DataFrame,
    *,
    block_count: int,
    seed: int,
    mechanism_column: str = "msg_size",
) -> pd.DataFrame:
    """Create profile K-means and complete-family mechanism blocks."""
    if block_count < 2 or block_count > len(inhibitors):
        raise ValueError("block_count must be between 2 and the inhibitor count")
    scaler = ProfileScaler("base").fit(inhibitors)
    values = scaler.transform(inhibitors)
    profile = KMeans(
        n_clusters=block_count,
        random_state=seed,
        n_init=20,
    ).fit_predict(values)
    result = pd.DataFrame(
        {
            "inhib_id": inhibitors["inhib_id"].astype(str),
            "profile_block": profile.astype(int),
        }
    )

    if mechanism_column not in inhibitors:
        raise ValueError(
            f"Requested mechanism-family column is absent: {mechanism_column}"
        )
    if inhibitors[mechanism_column].nunique() < block_count:
        raise ValueError(
            f"{mechanism_column} has fewer families than the requested block count"
        )
    family = inhibitors[mechanism_column].astype(str)
    counts = family.value_counts().rename_axis("family").reset_index(name="count")
    counts["tie"] = counts["family"].map(
        lambda value: int(
            hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()[:8], 16
        )
    )
    counts = counts.sort_values(["count", "tie", "family"], ascending=[False, True, True])
    loads = [0] * block_count
    family_block: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        block = min(range(block_count), key=lambda index: (loads[index], index))
        family_block[str(row.family)] = block
        loads[block] += int(row.count)
    result["mechanism_family"] = family.to_numpy()
    result["mechanism_block"] = family.map(family_block).to_numpy(dtype=int)
    return result
