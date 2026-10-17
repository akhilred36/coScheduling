from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch


AUDIT_DIR = Path(__file__).resolve().parent
MODULE_DIR = AUDIT_DIR.parent
RUN_DIR = AUDIT_DIR / "primary_full_default"
OUTPUT_DIR = AUDIT_DIR / "post_holdout"
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(MODULE_DIR))
BOOTSTRAP_SAMPLES = 1000

from data import ProfileScaler
from metrics import METRIC_COLUMNS, cluster_bootstrap, metrics_by_method, per_victim_metrics, self_pair_averaged
from model import LowRankPotential


predictions = pd.read_csv(RUN_DIR / "directional_predictions.csv")
seed_predictions = pd.read_csv(RUN_DIR / "seed_predictions.csv")
cv = pd.read_csv(RUN_DIR / "crossed_validation_predictions.csv")
seed_metrics = pd.read_csv(RUN_DIR / "metrics_by_seed.csv")
selection = json.loads((RUN_DIR / "selection.json").read_text())
rejected = pd.read_csv(RUN_DIR / "rejected_job_inh.csv")

primary = "delta_ood_kernel"
primary_cv_name = "ood_kernel"
primary_selection = selection["aggregation"][primary_cv_name]


def quantiles(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def log_mae(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            np.abs(
                frame["predicted_log_slowdown"].to_numpy(dtype=float)
                - np.log(frame["true_slowdown"].to_numpy(dtype=float))
            )
        )
    )


# Paired method comparisons use identical resampled pair-row clusters.
rng = np.random.default_rng(20260814)
pair_ids = np.sort(predictions["pair_row_id"].unique())
method_groups = {
    method: {pair_id: group for pair_id, group in frame.groupby("pair_row_id")}
    for method, frame in predictions.groupby("method")
}
primary_point = log_mae(predictions[predictions["method"] == primary])
comparison_rows = []
for comparator in sorted(set(method_groups) - {primary}):
    comparator_frame = predictions[predictions["method"] == comparator]
    point_difference = primary_point - log_mae(comparator_frame)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        primary_draw = pd.concat([method_groups[primary][value] for value in sampled])
        comparator_draw = pd.concat([method_groups[comparator][value] for value in sampled])
        draws.append(log_mae(primary_draw) - log_mae(comparator_draw))
    lower, upper = quantiles(draws)
    comparison_rows.append(
        {
            "primary": primary,
            "comparator": comparator,
            "log_mae_difference_primary_minus_comparator": point_difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "probability_primary_better": float(np.mean(np.asarray(draws) < 0)),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "cluster_count": len(pair_ids),
        }
    )
paired = pd.DataFrame(comparison_rows)
paired.to_csv(OUTPUT_DIR / "paired_log_mae_comparisons.csv", index=False)


# Add the missing self-averaged per-victim, macro, and clustered intervals.
averaged = self_pair_averaged(predictions)
averaged_per_victim, averaged_macro = per_victim_metrics(averaged)
metrics_by_method(averaged).to_csv(OUTPUT_DIR / "self_averaged_metrics.csv", index=False)
averaged_per_victim.to_csv(OUTPUT_DIR / "self_averaged_per_victim.csv", index=False)
averaged_macro.to_csv(OUTPUT_DIR / "self_averaged_macro_victim.csv", index=False)
cluster_bootstrap(averaged, samples=BOOTSTRAP_SAMPLES, seed=20260814).to_csv(
    OUTPUT_DIR / "self_averaged_cluster_bootstrap.csv", index=False
)


# Summarize variation in whole-holdout performance over initialization seeds.
seed_variation = (
    seed_metrics.groupby("method", as_index=False)
    .agg(
        seed_count=("seed", "nunique"),
        log_mae_mean=("log_mae", "mean"),
        log_mae_std=("log_mae", "std"),
        log_mae_min=("log_mae", "min"),
        log_mae_max=("log_mae", "max"),
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
    )
)
seed_variation.to_csv(OUTPUT_DIR / "model_seed_variation.csv", index=False)


# Select exactly the CV rows used for the frozen aggregation choices.
cv_selected = {
    "delta_uniform": cv[cv["method"].eq("uniform")],
    "delta_median": cv[cv["method"].eq("median")],
    "delta_kernel": cv[
        cv["method"].eq("kernel")
        & np.isclose(cv["temperature"], selection["aggregation"]["kernel"]["temperature"])
    ],
    "delta_ood_kernel": cv[
        cv["method"].eq("ood_kernel")
        & np.isclose(cv["temperature"], primary_selection["temperature"])
        & np.isclose(cv["ood_quantile"], primary_selection["quantile"])
        & cv["ood_fallback"].eq(primary_selection["fallback"])
    ],
}
validation_transfer_rows = []
for method, frame in cv_selected.items():
    final_frame = predictions[predictions["method"].eq(method)]
    validation_transfer_rows.append(
        {
            "method": method,
            "crossed_validation_log_mae": float(frame["absolute_log_error"].mean()),
            "app_app_log_mae": log_mae(final_frame),
            "app_app_minus_cv": log_mae(final_frame) - float(frame["absolute_log_error"].mean()),
        }
    )
validation_transfer = pd.DataFrame(validation_transfer_rows)
validation_transfer.to_csv(OUTPUT_DIR / "validation_vs_app_app.csv", index=False)
method_rank_correlation = float(
    spearmanr(
        validation_transfer["crossed_validation_log_mae"],
        validation_transfer["app_app_log_mae"],
    ).correlation
)

cv_victim = (
    cv_selected[primary]
    .groupby("victim_id", as_index=False)["absolute_log_error"]
    .mean()
    .rename(columns={"absolute_log_error": "cv_log_mae"})
)
app_victim = (
    predictions[predictions["method"].eq(primary)]
    .groupby("victim_id", as_index=False)["absolute_log_error"]
    .mean()
    .rename(columns={"absolute_log_error": "app_app_log_mae"})
)
victim_transfer = cv_victim.merge(app_victim, on="victim_id", validate="one_to_one")
victim_transfer_correlation = float(
    spearmanr(victim_transfer["cv_log_mae"], victim_transfer["app_app_log_mae"]).correlation
)
victim_transfer.to_csv(OUTPUT_DIR / "primary_per_victim_validation_vs_app_app.csv", index=False)


# Diagnostic associations are post-holdout, descriptive, and cluster-bootstrapped.
primary_frame = predictions[predictions["method"].eq(primary)].copy()
diagnostic_names = [
    "nearest_anchor_distance",
    "ood_score",
    "effective_anchor_count",
    "per_anchor_prediction_std",
    "per_anchor_prediction_mad",
    "model_seed_log_std",
]
diagnostic_rows = []
rng = np.random.default_rng(20260815)
primary_groups = {key: value for key, value in primary_frame.groupby("pair_row_id")}
for diagnostic in diagnostic_names:
    finite = primary_frame[["victim_id", diagnostic, "absolute_log_error"]].dropna()
    if len(finite) > 1 and finite[diagnostic].nunique() > 1:
        point = float(spearmanr(finite[diagnostic], finite["absolute_log_error"]).correlation)
        ranked = finite.assign(
            diagnostic_rank=finite.groupby("victim_id")[diagnostic].rank(pct=True),
            error_rank=finite.groupby("victim_id")["absolute_log_error"].rank(pct=True),
        )
        within = float(spearmanr(ranked["diagnostic_rank"], ranked["error_rank"]).correlation)
        draws = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            draw = pd.concat([primary_groups[value] for value in sampled], ignore_index=True)
            draw = draw[[diagnostic, "absolute_log_error"]].dropna()
            if len(draw) > 1 and draw[diagnostic].nunique() > 1:
                draws.append(float(spearmanr(draw[diagnostic], draw["absolute_log_error"]).correlation))
        lower, upper = quantiles(draws)
    else:
        point = within = lower = upper = np.nan
    diagnostic_rows.append(
        {
            "method": primary,
            "diagnostic": diagnostic,
            "n_finite": len(finite),
            "unique_values": finite[diagnostic].nunique(),
            "spearman_with_absolute_log_error": point,
            "cluster_ci_lower": lower,
            "cluster_ci_upper": upper,
            "within_victim_rank_spearman": within,
        }
    )
diagnostic_correlations = pd.DataFrame(diagnostic_rows)
diagnostic_correlations.to_csv(OUTPUT_DIR / "primary_diagnostic_correlations.csv", index=False)


# Verify invariants on every saved low-rank checkpoint using actual normalized profiles.
profile_rows = []
jobs_csv = MODULE_DIR.parent / "few_shot" / "data" / "jobs.csv"
inhibitors_csv = MODULE_DIR.parent / "few_shot" / "data" / "inhibitors.csv"
jobs = pd.read_csv(jobs_csv).loc[:, lambda value: ~value.columns.str.match(r"^Unnamed")]
inhibitors = pd.read_csv(inhibitors_csv).loc[:, lambda value: ~value.columns.str.match(r"^Unnamed")]
for checkpoint_path in sorted((RUN_DIR / "checkpoints").glob("low_rank_seed*.pt")):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = ProfileScaler.from_state_dict(checkpoint["scaler"])
    job_values = torch.as_tensor(scaler.transform(jobs), dtype=torch.float32)
    inhibitor_values = torch.as_tensor(scaler.transform(inhibitors), dtype=torch.float32)
    model = LowRankPotential(**{
        "input_dim": 4,
        "hidden_dim": checkpoint["model_config"]["hidden_dim"],
        "embedding_dim": checkpoint["model_config"]["embedding_dim"],
        "rank": checkpoint["model_config"]["rank"],
    })
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    victim = job_values.repeat_interleave(3, dim=0)
    left = inhibitor_values[:30]
    middle = inhibitor_values[30:60]
    right = inhibitor_values[60:90]
    with torch.no_grad():
        identity = model(victim, left, left)
        forward = model(victim, left, right)
        reverse = model(victim, right, left)
        path = model(victim, left, middle) + model(victim, middle, right)
    profile_rows.append(
        {
            "seed": checkpoint["seed"],
            "max_identity_error": float(identity.abs().max()),
            "max_antisymmetry_error": float((forward + reverse).abs().max()),
            "max_path_consistency_error": float((forward - path).abs().max()),
        }
    )
invariants = pd.DataFrame(profile_rows)
invariants.to_csv(OUTPUT_DIR / "checkpoint_invariants.csv", index=False)


# Self-pair behavior and OOD coverage summaries.
self_rows = predictions[
    predictions["method"].eq(primary)
    & predictions["victim_id"].eq(predictions["aggressor_id"])
]
self_prediction_spread = self_rows.groupby("pair_row_id")["predicted_log_slowdown"].agg(
    lambda values: float(np.ptp(values))
)
self_truth_spread = self_rows.groupby("pair_row_id")["true_slowdown"].agg(
    lambda values: float(np.ptp(np.log(values)))
)

summary = {
    "analysis_is_post_holdout_only": True,
    "primary_method": primary,
    "real_pair_rows": int(predictions["pair_row_id"].nunique()),
    "directional_rows_per_method": int(len(primary_frame)),
    "all_predictions_finite_and_positive": bool(
        np.isfinite(predictions["predicted_slowdown"]).all()
        and (predictions["predicted_slowdown"] > 0).all()
    ),
    "validation_to_app_app_method_rank_spearman": method_rank_correlation,
    "primary_per_victim_validation_to_app_app_spearman": victim_transfer_correlation,
    "primary_ood": {
        "alpha_below_one_count": int((primary_frame["ood_alpha"] < 1).sum()),
        "alpha_median": float(primary_frame["ood_alpha"].median()),
        "ood_score_median": float(primary_frame["ood_score"].median()),
        "ood_score_at_least_0_95_count": int((primary_frame["ood_score"] >= 0.95).sum()),
        "distance_min": float(primary_frame["nearest_anchor_distance"].min()),
        "distance_median": float(primary_frame["nearest_anchor_distance"].median()),
        "distance_max": float(primary_frame["nearest_anchor_distance"].max()),
        "effective_anchor_count_finite": int(primary_frame["effective_anchor_count"].notna().sum()),
    },
    "self_pairs": {
        "count": int(self_rows["pair_row_id"].nunique()),
        "max_directional_prediction_log_difference": float(self_prediction_spread.max()),
        "median_observed_directional_log_difference": float(self_truth_spread.median()),
        "max_observed_directional_log_difference": float(self_truth_spread.max()),
    },
    "rejected_response_reasons": rejected["rejection_reason"].value_counts().to_dict(),
    "invariants": {
        "max_identity_error": float(invariants["max_identity_error"].max()),
        "max_antisymmetry_error": float(invariants["max_antisymmetry_error"].max()),
        "max_path_consistency_error": float(invariants["max_path_consistency_error"].max()),
    },
}
(OUTPUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
