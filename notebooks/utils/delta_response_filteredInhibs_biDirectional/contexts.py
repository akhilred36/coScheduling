"""Application response fingerprints built from permitted App-Inhibitor support.

A fingerprint token summarizes one inhibitor *configuration* (all replicates
pooled) measured against one application: the inhibitor's isolated profile,
the application's slowdown evidence (incoming susceptibility), and both
endpoints' rate changes (incoming and outgoing evidence). Inhibitor endpoints
never receive fingerprints: their context is empty by definition.

Leakage rule: callers hand a ``ContextBuilder`` only rows they are permitted
to use, and query inhibitor IDs are excluded across all of their replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from data import AUX_TASKS, AuxiliaryScaler, aux_mask_column, aux_target_column, validate_aux_tasks


TOKEN_DIRECTIONS = {
    "forward_plus_reverse": ("app", "inhib"),
    "forward_only": ("app",),
}


@dataclass(frozen=True)
class ContextConfig:
    enabled: bool = False
    token_directions: str = "forward_plus_reverse"
    token_tasks: tuple[str, ...] = AUX_TASKS
    max_support: int = 7
    empty_probability: float = 0.25
    crossfit_folds: int = 3
    token_hidden_dim: int = 8
    context_dim: int = 4
    partition_seed: int = 1701

    def __post_init__(self) -> None:
        if self.token_directions not in TOKEN_DIRECTIONS:
            raise ValueError(f"token_directions must be one of {sorted(TOKEN_DIRECTIONS)}")
        object.__setattr__(self, "token_tasks", validate_aux_tasks(self.token_tasks))
        if int(self.max_support) < 0:
            raise ValueError("max_support must be non-negative")
        if not 0.0 <= float(self.empty_probability) <= 1.0:
            raise ValueError("empty_probability must lie in [0, 1]")
        if int(self.crossfit_folds) < 1:
            raise ValueError("crossfit_folds must be positive")

    @property
    def endpoints(self) -> tuple[str, ...]:
        return TOKEN_DIRECTIONS[self.token_directions]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["token_tasks"] = list(self.token_tasks)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ContextConfig":
        values = dict(values or {})
        if "token_tasks" in values:
            values["token_tasks"] = tuple(values["token_tasks"])
        return cls(**values)


def token_feature_names(config: ContextConfig, profile_dim: int) -> list[str]:
    names = [f"inhibitor_profile_{index}" for index in range(profile_dim)]
    names += ["app_mean_log_slowdown", "app_floor_fraction"]
    for endpoint in config.endpoints:
        for task in config.token_tasks:
            names += [f"{endpoint}_{task}_scaled_mean", f"{endpoint}_{task}_available"]
    return names


def token_dim(config: ContextConfig, profile_dim: int) -> int:
    return len(token_feature_names(config, profile_dim))


@dataclass
class SupportTokens:
    application: str
    inhibitor_ids: list[str]
    observation_ids: list[str]
    features: np.ndarray

    @property
    def count(self) -> int:
        return len(self.inhibitor_ids)


def fit_token_scaler(training_rows: pd.DataFrame, config: ContextConfig) -> AuxiliaryScaler:
    """Training-only scales for auxiliary token features (slowdown is unscaled)."""
    return AuxiliaryScaler(config.token_tasks, per_endpoint=True).fit(training_rows)


def _token_table(
    rows: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    token_scaler: AuxiliaryScaler,
    config: ContextConfig,
) -> dict[str, tuple[list[str], np.ndarray, dict[str, list[str]]]]:
    """Per application: sorted inhibitor IDs, one token row each, and observation IDs."""
    if rows.empty:
        return {}
    frame = pd.DataFrame(
        {
            "job_id": rows["job_id"].astype(str).to_numpy(),
            "inhib_id": rows["inhib_id"].astype(str).to_numpy(),
            "log": rows["log_slowdown"].to_numpy(dtype=float),
            "floor": rows["is_censored"].astype(bool).to_numpy(dtype=float),
        }
    )
    aux_columns = []
    for endpoint in config.endpoints:
        for task in config.token_tasks:
            mask = rows[aux_mask_column(endpoint, task)].astype(bool).to_numpy()
            values = rows[aux_target_column(endpoint, task)].to_numpy(dtype=float)
            name = f"{endpoint}:{task}"
            frame[f"{name}:sum"] = np.where(mask, values, 0.0)
            frame[f"{name}:count"] = mask.astype(float)
            aux_columns.append((name, endpoint, task))
    grouped = frame.groupby(["job_id", "inhib_id"], sort=True)
    sums = grouped.sum(numeric_only=True)
    sizes = grouped.size().to_numpy(dtype=float)
    observations = rows.assign(_job=frame["job_id"].to_numpy(), _inhib=frame["inhib_id"].to_numpy()).groupby(
        ["_job", "_inhib"], sort=True
    )["observation_id"].apply(lambda values: sorted(values.astype(str)))
    profile_matrix = np.stack([inhibitor_profiles[value] for value in sums.index.get_level_values("inhib_id")])
    columns = [profile_matrix, (sums["log"].to_numpy() / sizes)[:, None], (sums["floor"].to_numpy() / sizes)[:, None]]
    for name, endpoint, task in aux_columns:
        count = sums[f"{name}:count"].to_numpy()
        mean = np.divide(sums[f"{name}:sum"].to_numpy(), count, out=np.zeros_like(count), where=count > 0)
        columns.append((mean / token_scaler.scale(task, endpoint))[:, None])
        columns.append((count > 0).astype(float)[:, None])
    matrix = np.hstack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Support tokens must be finite")
    result: dict[str, tuple[list[str], np.ndarray, dict[str, list[str]]]] = {}
    jobs = sums.index.get_level_values("job_id").to_numpy()
    inhibs = sums.index.get_level_values("inhib_id").to_numpy()
    for job_id in dict.fromkeys(jobs):
        selected = np.flatnonzero(jobs == job_id)
        ids = inhibs[selected].tolist()
        result[str(job_id)] = (
            ids, matrix[selected], {value: observations.loc[(job_id, value)] for value in ids}
        )
    return result


def build_support_tokens(
    support_rows: pd.DataFrame,
    inhibitor_profiles: dict[str, np.ndarray],
    token_scaler: AuxiliaryScaler,
    config: ContextConfig,
    application: str = "",
) -> SupportTokens:
    """One token per inhibitor configuration, replicates pooled by averaging."""
    profile_dim = len(next(iter(inhibitor_profiles.values())))
    width = token_dim(config, profile_dim)
    if support_rows.empty:
        return SupportTokens(application, [], [], np.zeros((0, width), dtype=np.float32))
    if support_rows["job_id"].astype(str).nunique() != 1:
        raise ValueError("Support tokens must describe exactly one application")
    table = _token_table(support_rows, inhibitor_profiles, token_scaler, config)
    ids, matrix, observations = next(iter(table.values()))
    return SupportTokens(
        application or str(support_rows["job_id"].iloc[0]), ids,
        sorted(value for inhib_id in ids for value in observations[inhib_id]), matrix,
    )


class ContextBuilder:
    """Builds application fingerprints from an explicit permitted-support table.

    Token rows are per configuration and do not depend on one another, so the
    builder precomputes each permitted (application, inhibitor) token once and
    serves subsets through a ``ContextCache`` keyed by fold, application,
    support IDs, support observation IDs, normalization state, and model
    identity. Only raw token tensors are cached, never learned embeddings.
    """

    def __init__(
        self,
        permitted_rows: pd.DataFrame,
        inhibitor_profiles: dict[str, np.ndarray],
        token_scaler: AuxiliaryScaler,
        config: ContextConfig,
        scope: str = "unspecified",
        *,
        fold: str = "",
        model_identity: str = "",
        cache: ContextCache | None = None,
    ):
        self.rows = {
            str(job_id): group.reset_index(drop=True)
            for job_id, group in permitted_rows.groupby("job_id", sort=True)
        }
        self.inhibitor_profiles = inhibitor_profiles
        self.token_scaler = token_scaler
        self.config = config
        self.scope = scope
        self.fold = fold
        self.model_identity = model_identity
        self.cache = cache if cache is not None else ContextCache()
        self.normalization_state = state_digest(token_scaler.state_dict(), config.to_dict())
        self.width = token_dim(config, len(next(iter(inhibitor_profiles.values()))))
        self.tables = _token_table(permitted_rows, inhibitor_profiles, token_scaler, config)

    def support_ids(self, application: str) -> list[str]:
        entry = self.tables.get(str(application))
        return [] if entry is None else list(entry[0])

    def support_rows(self, application: str) -> pd.DataFrame:
        rows = self.rows.get(str(application))
        return rows.copy() if rows is not None else pd.DataFrame()

    def application_context(
        self,
        application: str,
        *,
        include_inhib_ids: list[str] | set[str] | None = None,
        exclude_inhib_ids: list[str] | set[str] = (),
    ) -> SupportTokens:
        application = str(application)
        excluded = {str(value) for value in exclude_inhib_ids}
        ids, matrix, observations = self.tables.get(
            application, ([], np.zeros((0, self.width), np.float32), {})
        )
        if include_inhib_ids is not None:
            included = {str(value) for value in include_inhib_ids}
            overlap = included & excluded
            if overlap:
                raise ValueError(f"Support and excluded query inhibitors overlap: {sorted(overlap)}")
        else:
            included = set(ids)
        chosen = [index for index, value in enumerate(ids) if value in included and value not in excluded]
        support_ids = [ids[index] for index in chosen]
        support_observations = sorted(value for inhib_id in support_ids for value in observations[inhib_id])
        key = ContextCache.key(
            fold=self.fold, application=application, support_inhib_ids=support_ids,
            support_observation_ids=support_observations,
            normalization_state=self.normalization_state, model_identity=self.model_identity,
        )
        return self.cache.get_or_build(
            key,
            lambda: SupportTokens(application, support_ids, support_observations,
                                  matrix[chosen] if chosen else np.zeros((0, self.width), np.float32)),
        )

    def token_matrix(self, application: str) -> tuple[list[str], np.ndarray]:
        """Per-configuration tokens for every permitted inhibitor of one app."""
        tokens = self.application_context(application)
        return tokens.inhibitor_ids, tokens.features


def build_application_context(
    builder: ContextBuilder,
    application: str,
    *,
    include_inhib_ids: list[str] | set[str] | None = None,
    exclude_inhib_ids: list[str] | set[str] = (),
) -> SupportTokens:
    return builder.application_context(
        application, include_inhib_ids=include_inhib_ids, exclude_inhib_ids=exclude_inhib_ids
    )


def stack_contexts(
    token_sets: list[np.ndarray | None], feature_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-size token sets into [batch, max_tokens, feature] plus a mask."""
    sizes = [0 if tokens is None else len(tokens) for tokens in token_sets]
    width = max(sizes) if sizes else 0
    padded = np.zeros((len(token_sets), width, feature_dim), dtype=np.float32)
    mask = np.zeros((len(token_sets), width), dtype=bool)
    for index, tokens in enumerate(token_sets):
        if tokens is not None and len(tokens):
            padded[index, : len(tokens)] = tokens
            mask[index, : len(tokens)] = True
    return padded, mask


def state_digest(*states: Any) -> str:
    text = json.dumps(states, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ContextCache:
    """Caches raw support tensors; the key names every input that defines them."""

    store: dict[tuple, SupportTokens] = field(default_factory=dict)

    @staticmethod
    def key(
        *,
        fold: str,
        application: str,
        support_inhib_ids: list[str],
        support_observation_ids: list[str],
        normalization_state: str,
        model_identity: str,
    ) -> tuple:
        return (
            str(fold),
            str(application),
            tuple(sorted(str(value) for value in support_inhib_ids)),
            tuple(sorted(str(value) for value in support_observation_ids)),
            str(normalization_state),
            str(model_identity),
        )

    def get_or_build(self, key: tuple, build: Callable[[], SupportTokens]) -> SupportTokens:
        if key not in self.store:
            self.store[key] = build()
        return self.store[key]


def crossfit_partition(
    inhibitor_ids: list[str], victim_id: str, folds: int, seed: int = 1701
) -> list[list[str]]:
    """Deterministically split distinct inhibitor IDs into up to ``folds`` groups."""
    distinct = sorted(dict.fromkeys(str(value) for value in inhibitor_ids))
    if not distinct:
        return []
    ordered = sorted(
        distinct,
        key=lambda value: hashlib.sha256(f"{seed}|{victim_id}|{value}".encode()).hexdigest(),
    )
    count = min(int(folds), len(ordered))
    groups = [ordered[index::count] for index in range(count)]
    return [sorted(group) for group in groups]
