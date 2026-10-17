#!/usr/bin/env python3
"""Summarize and plot a consolidated bidirectional experiment root.

Reads ``<root>/consolidated`` (written by ``run_experiments.py``) and produces
accuracy-by-tier, learning-curve, ablation, pair-level, and censoring plots,
with direct and calibrated inference reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_prediction_errors import MUTED, SERIES, plot_ablation, save, style


FOCUS_BY_FAMILY = {
    "direct": ["legacy_absolute", "joint_profile_direct", "bidirectional_profile_direct", "bidirectional_context_direct"],
    "calibrated": ["joint_profile_calibrated", "bidirectional_profile_calibrated",
                   "bidirectional_context_calibrated", "bidirectional_context_forward_calibrated"],
}


def load(consolidated: Path, name: str) -> pd.DataFrame:
    path = consolidated / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def plot_accuracy_by_tier(accuracy: pd.DataFrame, directory: Path) -> pd.DataFrame:
    frame = accuracy[accuracy["metric"] == "log_mae"]
    summary = frame.groupby(["evaluation_method", "method", "inference_family"], as_index=False)["estimate"].mean()
    tiers = list(dict.fromkeys(summary["evaluation_method"]))
    order = summary.groupby("method")["estimate"].mean().sort_values().index.tolist()
    figure, axes = plt.subplots(1, len(tiers), figsize=(4.2 * len(tiers), 0.3 * len(order) + 1.6),
                                squeeze=False, sharey=True)
    for axis, tier in zip(axes[0], tiers):
        values = summary[summary["evaluation_method"] == tier].set_index("method").reindex(order)
        axis.barh(np.arange(len(order)), values["estimate"], color=SERIES[0], height=0.6)
        axis.set_title(tier, fontsize=9)
        axis.set_xlabel("non-self log MAE (mean over tasks)")
    axes[0, 0].set_yticks(np.arange(len(order)), order)
    axes[0, 0].invert_yaxis()
    save(figure, directory, "accuracy_by_tier")
    return summary


def plot_learning_curves(learning: pd.DataFrame, directory: Path) -> None:
    if learning.empty:
        return
    frame = learning[learning["metric"] == "log_mae"]
    if frame.empty:
        return
    tiers = list(dict.fromkeys(frame["evaluation_method"]))
    families = [family for family in FOCUS_BY_FAMILY if frame["method"].isin(FOCUS_BY_FAMILY[family]).any()]
    figure, axes = plt.subplots(len(families), len(tiers), figsize=(4.4 * len(tiers), 3.2 * len(families)),
                                squeeze=False, sharex=True)
    for row, family in enumerate(families):
        for column, tier in enumerate(tiers):
            axis = axes[row, column]
            for color, method in zip(SERIES, FOCUS_BY_FAMILY[family]):
                values = frame[(frame["evaluation_method"] == tier) & (frame["method"] == method)].sort_values("training_app_count")
                if values.empty:
                    continue
                axis.plot(values["training_app_count"], values["estimate"], color=color, linewidth=2, marker="o",
                          markersize=4, label=method)
                if values["between_split_sd"].notna().any():
                    axis.fill_between(values["training_app_count"], values["estimate"] - values["between_split_sd"],
                                      values["estimate"] + values["between_split_sd"], color=color, alpha=0.12, linewidth=0)
            axis.set_title(f"{tier} - {family}", fontsize=9)
            axis.set_xlabel("training applications")
            if column == 0:
                axis.set_ylabel("non-self log MAE")
        axes[row, -1].legend(fontsize=7)
    save(figure, directory, "learning_curves")


def plot_pair_metrics(pairs: pd.DataFrame, directory: Path) -> None:
    if pairs.empty:
        return
    frame = pairs[pairs["pair_scope"] == "non_self"] if "pair_scope" in pairs else pairs
    if frame.empty:
        return
    summary = frame.groupby("method", as_index=False)[["severity_mae", "imbalance_mae", "worst_endpoint_log_mae",
                                                       "false_benign_rate"]].mean().sort_values("worst_endpoint_log_mae")
    figure, axes = plt.subplots(1, 4, figsize=(14, 0.3 * len(summary) + 1.6), sharey=True)
    for axis, column, color in zip(axes, ["severity_mae", "imbalance_mae", "worst_endpoint_log_mae", "false_benign_rate"],
                                   SERIES):
        axis.barh(np.arange(len(summary)), summary[column], color=color, height=0.6)
        axis.set_xlabel(column.replace("_", " "))
    axes[0].set_yticks(np.arange(len(summary)), summary["method"])
    axes[0].invert_yaxis()
    save(figure, directory, "pair_level_metrics")
    summary.to_csv(directory / "pair_level_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="experiment root containing consolidated/")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    consolidated = args.root / "consolidated"
    if not consolidated.is_dir():
        raise FileNotFoundError(f"Missing consolidated results: {consolidated}")
    directory = args.output_dir or args.root / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    style()
    report: dict[str, object] = {}
    accuracy = load(consolidated, "accuracy_by_eval_method.csv")
    if not accuracy.empty:
        summary = plot_accuracy_by_tier(accuracy, directory)
        summary.to_csv(directory / "accuracy_by_tier_summary.csv", index=False)
        report["best_by_tier"] = {
            tier: group.sort_values("estimate").iloc[0][["method", "estimate"]].to_dict()
            for tier, group in summary.groupby("evaluation_method")
        }
    plot_learning_curves(load(consolidated, "learning_curve_summary.csv"), directory)
    ablation = load(consolidated, "ablation_summary.csv")
    if not ablation.empty:
        if "stage" in ablation:
            ablation = ablation[ablation["stage"].isin(["tune_global", "evaluate_random_split"])]
        plot_ablation(ablation, directory)
        ablation.to_csv(directory / "ablation_primary.csv", index=False)
    plot_pair_metrics(load(consolidated, "pair_metrics.csv"), directory)
    (directory / "analysis_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Analysis written to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
