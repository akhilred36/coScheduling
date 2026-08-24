"""Static experiment configuration and compact candidate registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    protocol: str = "random_split"
    app_app_status: str = "historical_diagnostic_only"
    inhibitor_blocks: int = 4
    clustering_seed: int = 1701
    mechanism_column: str = "msg_size"
    fold_schemes: list[str] = field(default_factory=lambda: ["profile", "mechanism"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    capacities: list[list[int]] = field(
        default_factory=lambda: [[8, 4], [16, 8], [32, 16]]
    )
    feature_set: str = "base"
    rank: int = 2
    optimizer_steps: int = 80
    batch_size: int = 128
    checkpoint_interval: int = 20
    learning_rate: float = 0.003
    weight_decay: float = 0.001
    nearest_per_application: int = 10
    profile_weight_clip: list[float] = field(default_factory=lambda: [0.05, 20.0])
    censored_shrinkage: list[float] = field(default_factory=lambda: [0.0, 10.0, 100.0])
    hierarchical_shrinkage: list[float] = field(default_factory=lambda: [10.0, 100.0])
    query_gammas: list[float] = field(default_factory=lambda: [0.5, 0.75])
    linear_ridge: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    median_shrinkage: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    legacy_ood_temperature: float = 4.0
    legacy_ood_quantile: float = 0.90
    legacy_ood_fallback: str = "uniform"
    tree_estimators: int = 50
    tree_depth: int = 2
    tree_learning_rate: float = 0.05
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 923
    device: str = "auto"

    def validate(self) -> None:
        if self.protocol != "random_split":
            raise ValueError("Delta Response 2 supports only random_split")
        if self.app_app_status != "historical_diagnostic_only":
            raise ValueError("This Path A implementation requires diagnostic-only App-App data")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if (
            set(self.fold_schemes) - {"profile", "mechanism"}
            or not self.fold_schemes
            or len(set(self.fold_schemes)) != len(self.fold_schemes)
        ):
            raise ValueError("fold_schemes may contain only profile and mechanism")
        if self.optimizer_steps <= 0 or self.checkpoint_interval <= 0 or self.batch_size <= 0:
            raise ValueError("Fixed training budget fields must be positive")
        if any(len(values) != 2 or min(values) <= 0 for values in self.capacities):
            raise ValueError("capacities must contain positive [hidden, embedding] pairs")
        if self.feature_set not in {"base", "augmented"}:
            raise ValueError("feature_set must be base or augmented")
        if self.rank <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("rank/LR must be positive and weight decay non-negative")
        if (
            any(value < 0 for value in self.censored_shrinkage)
            or len(set(self.censored_shrinkage)) != len(self.censored_shrinkage)
        ):
            raise ValueError("censored_shrinkage must be non-negative")
        if (
            any(value <= 0 for value in self.hierarchical_shrinkage)
            or len(set(self.hierarchical_shrinkage)) != len(self.hierarchical_shrinkage)
        ):
            raise ValueError("hierarchical_shrinkage must be positive and unique")
        if (
            any(not 0 <= value <= 1 for value in self.query_gammas)
            or len(set(self.query_gammas)) != len(self.query_gammas)
        ):
            raise ValueError("query_gammas must lie in [0, 1]")
        if any(value < 0 for value in self.linear_ridge):
            raise ValueError("linear_ridge must be non-negative")
        if any(not 0 <= value <= 1 for value in self.median_shrinkage):
            raise ValueError("median_shrinkage must lie in [0, 1]")
        if (
            self.legacy_ood_temperature <= 0
            or not 0 < self.legacy_ood_quantile < 1
            or self.legacy_ood_fallback not in {"uniform", "median"}
        ):
            raise ValueError("Invalid legacy OOD configuration")
        if (
            self.tree_estimators <= 0
            or self.tree_depth <= 0
            or self.tree_learning_rate <= 0
            or self.bootstrap_samples <= 0
        ):
            raise ValueError("Tree and bootstrap settings must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path | str) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        config = cls(**raw)
        config.validate()
        return config


def _number(value: float) -> str:
    return str(value).replace(".", "p")


def candidate_registry(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Return every prespecified candidate; shared model keys avoid duplicate fits."""
    config.validate()
    candidates: list[dict[str, Any]] = []

    def add(
        method: str,
        family: str,
        complexity: int,
        parameters: dict[str, Any] | None = None,
        model_key: str = "none",
        selectable: bool = True,
    ) -> None:
        payload = {
            "method": method,
            "family": family,
            "complexity": complexity,
            "parameters": parameters or {},
            "model_key": model_key,
            "selectable": selectable,
        }
        complete_specification = {"candidate": payload, "experiment": config.to_dict()}
        digest = hashlib.sha256(
            json.dumps(
                complete_specification, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:12]
        candidates.append({"candidate_id": f"{method}__{digest}", **payload})

    add("constant_1", "constant", 0, selectable=False)
    for alpha in config.median_shrinkage:
        add(
            f"global_observed_median_shrink_{_number(alpha)}",
            "global_intercept",
            1,
            {"alpha": alpha},
        )
    for shrinkage in config.hierarchical_shrinkage:
        add(
            f"global_censored_intercept_l{_number(shrinkage)}",
            "global_intercept",
            2,
            {"shrinkage": shrinkage},
        )
        add(
            f"victim_hierarchical_censored_l{_number(shrinkage)}",
            "victim_intercept",
            3,
            {"shrinkage": shrinkage},
        )
    add("nearest_anchor", "anchor_baseline", 2)
    add("robust_median_anchor", "anchor_baseline", 2)
    for ridge in config.linear_ridge:
        add(
            f"regularized_absolute_linear_r{_number(ridge)}",
            "absolute_linear",
            4,
            {"ridge": ridge},
        )
    add(
        "small_absolute_tree",
        "absolute_tree",
        5,
        {
            "estimators": config.tree_estimators,
            "depth": config.tree_depth,
            "learning_rate": config.tree_learning_rate,
        },
    )
    for capacity_index, (hidden, embedding) in enumerate(config.capacities):
        suffix = f"h{hidden}_e{embedding}"
        absolute_key = f"absolute:{hidden}:{embedding}"
        delta_key = f"delta:{hidden}:{embedding}"
        add(
            f"regularized_absolute_neural_{suffix}",
            "absolute_neural",
            60 + capacity_index,
            {"hidden_dim": hidden, "embedding_dim": embedding},
            absolute_key,
        )
        for aggregation in ["uniform", "median", "ood"]:
            add(
                f"current_delta_{aggregation}_{suffix}",
                "legacy_delta",
                70 + capacity_index,
                {
                    "hidden_dim": hidden,
                    "embedding_dim": embedding,
                    "aggregation": aggregation,
                },
                delta_key,
            )
        for shrinkage in config.censored_shrinkage:
            prior_mode = "none" if shrinkage == 0 else "population"
            add(
                f"censored_correction_delta_{suffix}_{prior_mode}_l{_number(shrinkage)}",
                "censored_delta",
                80 + capacity_index,
                {
                    "hidden_dim": hidden,
                    "embedding_dim": embedding,
                    "shrinkage": shrinkage,
                    "prior_mode": prior_mode,
                    "query_gamma": 1.0,
                },
                delta_key,
            )
        for shrinkage in config.hierarchical_shrinkage:
            add(
                f"censored_correction_delta_{suffix}_zero_l{_number(shrinkage)}",
                "censored_delta",
                80 + capacity_index,
                {
                    "hidden_dim": hidden,
                    "embedding_dim": embedding,
                    "shrinkage": shrinkage,
                    "prior_mode": "zero",
                    "query_gamma": 1.0,
                },
                delta_key,
            )
        for shrinkage in config.hierarchical_shrinkage:
            for gamma in config.query_gammas:
                add(
                    "censored_correction_delta_with_query_shrinkage_"
                    f"{suffix}_l{_number(shrinkage)}_g{_number(gamma)}",
                    "censored_delta_query_shrinkage",
                    90 + capacity_index,
                    {
                        "hidden_dim": hidden,
                        "embedding_dim": embedding,
                        "shrinkage": shrinkage,
                        "prior_mode": "population",
                        "query_gamma": gamma,
                    },
                    delta_key,
                )
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Candidate IDs are not unique")
    return candidates
