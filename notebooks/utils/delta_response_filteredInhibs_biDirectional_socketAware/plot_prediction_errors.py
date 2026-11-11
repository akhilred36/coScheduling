"""Plot directional, pair-level, auxiliary, ablation, and support diagnostics for one run.

All App-App comparisons use the pair clusters that every method can evaluate
(``method_coverage.csv`` records any exclusions). Positive signed log error
means overprediction.
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

from metrics import common_pair_rows


HERE = Path(__file__).resolve().parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FOCUS = [
    "constant_1", "legacy_absolute", "joint_profile_calibrated",
    "bidirectional_profile_calibrated", "bidirectional_context_direct", "bidirectional_context_calibrated",
]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def save(figure: plt.Figure, directory: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(directory / f"{name}.png", bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def directional_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    error = predictions["predicted_log_slowdown"] - np.log(predictions["true_slowdown"])
    frame = predictions.assign(abs_error=error.abs(), signed=error)
    return frame.groupby("method", as_index=False).agg(
        n=("abs_error", "size"), log_mae=("abs_error", "mean"), signed_log_bias=("signed", "mean")
    ).sort_values("log_mae")


def plot_directional(summary: pd.DataFrame, directory: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(summary) + 1.4), sharey=True)
    positions = np.arange(len(summary))
    axes[0].barh(positions, summary["log_mae"], color=SERIES[0], height=0.6)
    axes[0].set_xlabel("Directional log MAE (non-self, common pairs)")
    axes[1].barh(positions, summary["signed_log_bias"], color=SERIES[1], height=0.6)
    axes[1].axvline(0, color=MUTED, linewidth=0.8)
    axes[1].set_xlabel("Signed log bias (positive = overprediction)")
    axes[0].set_yticks(positions, summary["method"])
    axes[0].invert_yaxis()
    for position, value in zip(positions, summary["log_mae"]):
        axes[0].text(value, position, f" {value:.3f}", va="center", fontsize=7, color=INK)
    save(figure, directory, "directional_error_bias")


def plot_censoring(predictions: pd.DataFrame, directory: Path) -> pd.DataFrame:
    frame = predictions.assign(
        stratum=np.where(predictions["true_slowdown"] <= 1.0, "floor", "above floor"),
        abs_error=(predictions["predicted_log_slowdown"] - np.log(predictions["true_slowdown"])).abs(),
    )
    table = frame.pivot_table(index="method", columns="stratum", values="abs_error", aggfunc="mean")
    table = table.loc[table.mean(axis=1).sort_values().index]
    figure, axis = plt.subplots(figsize=(10, 0.32 * len(table) + 1.4))
    positions = np.arange(len(table))
    for offset, (stratum, color) in enumerate(zip(["floor", "above floor"], SERIES)):
        if stratum in table:
            axis.barh(positions + (offset - 0.5) * 0.38, table[stratum], height=0.36, color=color, label=stratum)
    axis.set_yticks(positions, table.index)
    axis.invert_yaxis()
    axis.set_xlabel("Directional log MAE by App-App censoring stratum")
    axis.legend(loc="lower right")
    save(figure, directory, "floor_case_errors")
    return table.reset_index()


def plot_pairs(pairs: pd.DataFrame, directory: Path) -> None:
    methods = [method for method in FOCUS if method in set(pairs["method"])]
    if not methods:
        return
    figure, axes = plt.subplots(2, len(methods), figsize=(2.6 * len(methods), 5.2), squeeze=False)
    for column, method in enumerate(methods):
        frame = pairs[pairs["method"] == method]
        for row, (truth, predicted, label) in enumerate([
            ("true_severity", "predicted_severity", "severity"),
            ("true_imbalance", "predicted_imbalance", "imbalance"),
        ]):
            axis = axes[row, column]
            values = np.concatenate([frame[truth], frame[predicted]])
            low, high = float(values.min()), float(values.max())
            axis.plot([low, high], [low, high], color=MUTED, linewidth=0.8, linestyle="--")
            axis.scatter(frame[truth], frame[predicted], s=16, color=SERIES[row], edgecolor="white", linewidth=0.6)
            axis.set_title(method if row == 0 else "", fontsize=8)
            axis.set_xlabel(f"true {label}")
            if column == 0:
                axis.set_ylabel(f"predicted {label}")
    save(figure, directory, "pair_severity_imbalance")


def plot_auxiliary(auxiliary: pd.DataFrame, directory: Path) -> None:
    valid = auxiliary[auxiliary["valid"].astype(bool)]
    if valid.empty:
        return
    methods = sorted(valid["method"].unique())[:3]
    tasks = sorted(valid["task"].unique())
    figure, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.2), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        frame = valid[valid["task"] == task]
        values = np.concatenate([frame["true"], frame["predicted"]])
        axis.plot([values.min(), values.max()], [values.min(), values.max()], color=MUTED, linewidth=0.8, linestyle="--")
        for color, method in zip(SERIES, methods):
            group = frame[frame["method"] == method]
            axis.scatter(group["true"], group["predicted"], s=14, color=color, edgecolor="white", linewidth=0.5,
                         label=method)
        axis.set_title(task, fontsize=9)
        axis.set_xlabel("true log degradation")
    axes[0, 0].set_ylabel("predicted log degradation")
    axes[0, -1].legend(fontsize=7, loc="upper left")
    save(figure, directory, "auxiliary_calibration")


def plot_ablation(ablation: pd.DataFrame, directory: Path) -> None:
    if ablation.empty:
        return
    scopes = list(dict.fromkeys(ablation["scope"]))
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 0.36 * ablation["contrast"].nunique() + 1.6),
                                squeeze=False, sharey=True)
    contrasts = list(dict.fromkeys(ablation["contrast"]))
    for axis, scope in zip(axes[0], scopes):
        frame = ablation[ablation["scope"] == scope].set_index("contrast").reindex(contrasts)
        positions = np.arange(len(contrasts))
        estimate = frame["estimate_difference"].to_numpy(float)
        low = frame["ci_lower"].to_numpy(float)
        high = frame["ci_upper"].to_numpy(float)
        standard_error = frame["standard_error"].to_numpy(float)
        low = np.where(np.isfinite(low), low, estimate - 1.96 * standard_error)
        high = np.where(np.isfinite(high), high, estimate + 1.96 * standard_error)
        axis.errorbar(estimate, positions, xerr=[estimate - low, high - estimate], fmt="o", color=SERIES[0],
                      ecolor=MUTED, elinewidth=1, capsize=2, markersize=5)
        axis.axvline(0, color=MUTED, linewidth=0.8)
        axis.set_title(scope.replace("_", " "), fontsize=8)
        axis.set_xlabel("method − comparison log MAE\n(negative = improvement)")
        axis.set_yticks(positions, contrasts)
    axes[0, 0].invert_yaxis()
    save(figure, directory, "ablation_contrasts")


def plot_support(calibration: pd.DataFrame, directory: Path) -> None:
    if calibration.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    for axis, column, label in [
        (axes[0], "context_inhibitor_count", "fingerprint inhibitors per view"),
        (axes[1], "inhibitor_count", "calibration inhibitors per view"),
        (axes[2], "floor_count", "floor calibration anchors per view"),
    ]:
        if column in calibration:
            values = calibration[column].dropna().astype(int)
            bins = np.arange(values.min(), values.max() + 2) - 0.5
            axis.hist(values, bins=bins, color=SERIES[2], edgecolor="white", linewidth=1)
            axis.set_xlabel(label)
    axes[0].set_ylabel("calibration views")
    save(figure, directory, "context_calibration_support")


def plot_run(run_dir: Path, output_dir: Path | None = None) -> pd.DataFrame:
    style()
    directory = output_dir or run_dir / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(run_dir / "directional_predictions.csv")
    common, _ = common_pair_rows(predictions)
    non_self = common[common["victim_id"] != common["aggressor_id"]]
    summary = directional_summary(non_self)
    summary.to_csv(directory / "directional_summary.csv", index=False)
    plot_directional(summary, directory)
    plot_censoring(non_self, directory).to_csv(directory / "floor_case_errors.csv", index=False)
    pairs_path = run_dir / "pair_predictions.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        plot_pairs(pairs[pairs["pair_scope"] == "non_self"], directory)
    if (run_dir / "auxiliary_predictions.csv").exists():
        plot_auxiliary(pd.read_csv(run_dir / "auxiliary_predictions.csv"), directory)
    if (run_dir / "ablation_summary.csv").exists():
        plot_ablation(pd.read_csv(run_dir / "ablation_summary.csv"), directory)
    if (run_dir / "calibration_diagnostics.csv").exists():
        calibration = pd.read_csv(run_dir / "calibration_diagnostics.csv")
        plot_support(calibration, directory)
    best = summary.iloc[0]
    (directory / "comparison_summary.json").write_text(json.dumps({
        "pair_scope": "non_self common pair clusters",
        "directional_outcomes": int(best["n"]),
        "lowest_log_mae_method": str(best["method"]),
        "lowest_log_mae": float(best["log_mae"]),
    }, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "outputs" / "filtered_fit")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = plot_run(args.run_dir, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
