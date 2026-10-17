"""Compare a filtered fit with the isolated-speed (slowdown=1) baseline.

Reads completed predictions only; never fits or selects models from App-App outcomes.
The primary delta aggregation is taken from App-Inhibitor selection.json.
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

from metrics import compute_metrics


def plot_run(run_dir: Path, output_dir: Path | None = None) -> pd.DataFrame:
    output_dir = output_dir or run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((run_dir / "selection.json").read_text())
    delta = "delta_" + selection["aggregations"]["low_rank"]["primary_method"]
    methods = ["constant_1", delta, "absolute_response"]
    labels = ["Isolated speed\n(slowdown = 1)", "Filtered delta\n" + delta.removeprefix("delta_"),
              "Filtered absolute"]
    colors = ["#64748b", "#2563eb", "#ea580c"]
    frame = pd.read_csv(run_dir / "directional_predictions.csv")
    frame = frame[frame["method"].isin(methods)].copy()
    if frame["anchor_budget"].nunique() != 1:
        raise ValueError("Plot one inference anchor budget at a time")
    frame = frame[frame["victim_id"] != frame["aggressor_id"]].copy()
    if frame.empty or set(frame["method"]) != set(methods):
        raise ValueError("Missing non-self comparison predictions")
    keys = ["pair_cluster_id", "direction"]
    reference = frame[frame["method"] == methods[0]].set_index(keys)["true_slowdown"].sort_index()
    for method in methods:
        observed = frame[frame["method"] == method].set_index(keys)["true_slowdown"].sort_index()
        pd.testing.assert_series_equal(observed, reference)
    frame["signed_log_error"] = np.log(frame["predicted_slowdown"] / frame["true_slowdown"])
    frame["absolute_log_error"] = frame["signed_log_error"].abs()
    frame["absolute_percentage_error"] = 100 * (frame["predicted_slowdown"] / frame["true_slowdown"] - 1).abs()

    rows = []
    for stratum, subset in [("all_non_self", frame),
                            ("floor", frame[frame["true_slowdown"] == 1]),
                            ("above_floor", frame[frame["true_slowdown"] > 1])]:
        for method in methods:
            group = subset[subset["method"] == method]
            if not group.empty:
                rows.append({"stratum": stratum, "method": method, "n": len(group),
                             **compute_metrics(group),
                             "signed_log_bias": group["signed_log_error"].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "comparison_metrics.csv", index=False)
    frame.to_csv(output_dir / "comparison_predictions.csv", index=False)
    overall = summary[summary["stratum"] == "all_non_self"].set_index("method").loc[methods]
    panel = json.loads((run_dir / "selected_inhibitors" / "selection_report.json").read_text())
    title = f"App-App prediction errors | {panel['selected_count']} selected inhibitor profiles | non-self pairs"

    def save(fig, name):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        for extension in ["png", "svg"]:
            fig.savefig(output_dir / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ylabel in zip(axes, ["log_mae", "raw_mape", "signed_log_bias"],
                                  ["Mean absolute log error", "Mean absolute percentage error", "Signed log bias (+ = overprediction)"]):
        values = overall[metric].to_numpy()
        ax.bar(labels, values, color=colors)
        ax.margins(y=.2)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_ylabel(ylabel)
        for i, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 4 if value >= 0 else -14), textcoords="offset points", ha="center")
    save(fig, "overall_errors")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, label, color in zip(methods, labels, colors):
        group = frame[frame["method"] == method]
        errors = np.sort(group["absolute_log_error"])
        axes[0].step(errors, np.arange(1, len(errors) + 1) / len(errors), label=label.replace("\n", " "), color=color)
    axes[0].set(xlabel="Absolute log prediction error", ylabel="Cumulative fraction")
    axes[0].legend(fontsize=8)
    axes[1].boxplot([frame.loc[frame["method"] == m, "signed_log_error"] for m in methods])
    axes[1].set_xticks(np.arange(1, len(methods) + 1), labels)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Signed log error (+ = overprediction)")
    save(fig, "error_distributions")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    upper = frame[["true_slowdown", "predicted_slowdown"]].to_numpy().max() * 1.05
    for ax, method, label, color in zip(axes, methods, labels, colors):
        group = frame[frame["method"] == method]
        ax.scatter(group["true_slowdown"], group["predicted_slowdown"], color=color, alpha=.7, s=20)
        ax.plot([1, upper], [1, upper], "k--", linewidth=.8)
        ax.set(xscale="log", yscale="log", xlabel="Measured slowdown", title=label)
    axes[0].set_ylabel("Predicted slowdown")
    save(fig, "prediction_calibration")

    per_victim = frame.groupby(["victim_id", "method"])[["absolute_log_error", "signed_log_error"]].mean()
    per_victim.to_csv(output_dir / "per_victim_errors.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, metric, ylabel in zip(axes, ["absolute_log_error", "signed_log_error"], ["Mean absolute log error", "Signed log bias"]):
        table = per_victim[metric].unstack("method")[methods]
        table.columns = [label.replace("\n", " ") for label in labels]
        table.plot.bar(ax=ax, color=colors, rot=35)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=.7)
    save(fig, "per_application_errors")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, stratum in zip(axes, ["floor", "above_floor"]):
        subset = summary[summary["stratum"] == stratum].set_index("method").reindex(methods)
        ax.bar(labels, subset["log_mae"], color=colors)
        count = int(subset["n"].iloc[0]) if subset["n"].notna().any() else 0
        ax.set(title=f"{stratum.replace('_', ' ')}: {count} directional outcomes", ylabel="Mean absolute log error")
    save(fig, "floor_case_errors")
    (output_dir / "comparison_summary.json").write_text(json.dumps({
        "run_dir": str(run_dir.resolve()), "delta_method": delta,
        "selected_inhibitor_count": panel["selected_count"],
        "selected_ids": panel["selected_ids"], "non_self_pair_clusters": int(frame["pair_cluster_id"].nunique()),
        "baseline": "isolated speed: constant slowdown 1", "self_pairs_excluded": True,
        "evaluation_note": "Historical App-App dataset; not a pristine holdout. Outcomes excluded from fitting and selection.",
        "metrics": summary.to_dict("records"),
    }, indent=2, allow_nan=False) + "\n")
    print(overall[["log_mae", "raw_mae", "raw_mape", "signed_log_bias"]].to_string())
    print(f"Plots written to {output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "filtered_fit")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plot_run(args.run_dir, args.output_dir)
