#!/usr/bin/env python3
"""Generate paper-oriented plots from a focused experiment runner output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "constant_1",
    "victim_inhibitor_median",
    "nearest_anchor",
    "absolute_response",
    "delta_single_anchor",
    "delta_uniform",
    "delta_median",
    "delta_kernel",
    "delta_ood_kernel",
    "generic_potential",
]

METHOD_LABELS = {
    "constant_1": "No-degradation baseline",
    "victim_inhibitor_median": "Victim proxy median",
    "nearest_anchor": "Nearest proxy response",
    "absolute_response": "Absolute-response NN",
    "delta_single_anchor": "Delta: nearest proxy",
    "delta_uniform": "Delta: uniform",
    "delta_median": "Delta: median",
    "delta_kernel": "Delta: kernel",
    "delta_ood_kernel": "Delta: OOD kernel",
    "generic_potential": "Generic potential",
}

METHOD_COLORS = {
    "constant_1": "#7A7A7A",
    "victim_inhibitor_median": "#A6761D",
    "nearest_anchor": "#CC79A7",
    "absolute_response": "#D55E00",
    "delta_single_anchor": "#56B4E9",
    "delta_uniform": "#E69F00",
    "delta_median": "#F0E442",
    "delta_kernel": "#6A3D9A",
    "delta_ood_kernel": "#0072B2",
    "generic_potential": "#009E73",
}

TIER_LABELS = {
    "random_split": "All endpoints used for fitting",
    "one_known": "One known endpoint",
    "zero_shot": "Zero-shot endpoints",
}

STAGE_LABELS = {
    "tune_architecture": "Feature set and rank",
    "tune_optimizer": "Optimizer",
    "tune_capacity": "Network capacity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze consolidated delta-response experiments and create "
            "publication-oriented figures and summary tables."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments"),
        help="Experiment root containing consolidated/ (default: experiments).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Analysis output directory (default: ROOT/analysis).",
    )
    parser.add_argument(
        "--comparison-size",
        type=int,
        help=(
            "Training-application count used in endpoint-comparison plots. "
            "By default, use the largest size not flagged as low-support zero-shot."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Figure formats to write (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster output DPI.")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def load_csv(consolidated: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = consolidated / name
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing {path}. Run run_experiments.py consolidate first."
            )
        return pd.DataFrame()
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def method_label(method: str, primary: str | None = None) -> str:
    label = METHOD_LABELS.get(method, method.replace("_", " ").title())
    if method == primary:
        return f"{label} (selected)"
    return label


def ordered_methods(values: Iterable[str]) -> list[str]:
    unique = set(values)
    return [method for method in METHOD_ORDER if method in unique] + sorted(
        unique - set(METHOD_ORDER)
    )


def primary_method(selection: dict[str, Any], accuracy: pd.DataFrame) -> str:
    marked = accuracy.loc[
        accuracy["is_global_primary_low_rank"].astype(str).str.lower().eq("true"),
        "method",
    ].dropna()
    if not marked.empty:
        values = marked.unique()
        if len(values) != 1:
            raise ValueError(f"Multiple methods marked as global primary: {values}")
        return str(values[0])

    aggregation = selection["aggregations"]["low_rank"]["primary_method"]
    return "delta_single_anchor" if aggregation == "single" else f"delta_{aggregation}"


def choose_comparison_size(
    learning: pd.DataFrame, primary: str, requested: int | None
) -> int:
    available = learning[
        learning["method"].eq(primary)
        & learning["metric"].eq("log_mae")
        & learning["evaluation_method"].eq("zero_shot")
    ].copy()
    if available.empty:
        raise ValueError(f"No zero-shot learning-curve rows found for {primary}")
    sizes = sorted(available["training_app_count"].astype(int).unique())
    if requested is not None:
        if requested not in sizes:
            raise ValueError(
                f"comparison size {requested} is unavailable; choose from {sizes}"
            )
        return requested
    if "low_support_zero_shot" in available:
        supported = available[
            ~available["low_support_zero_shot"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        if not supported.empty:
            return int(supported["training_app_count"].max())
    return int(max(sizes))


def save_figure(
    fig: plt.Figure,
    plots_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    paths = []
    for suffix in formats:
        path = plots_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def architecture_candidate_label(stage: str, row: pd.Series) -> str:
    if stage == "tune_architecture":
        return f"{row['feature_set']}\nr={int(row['rank'])}"
    if stage == "tune_optimizer":
        return f"lr={row['learning_rate']:g}\nwd={row['weight_decay']:g}"
    return f"hidden={int(row['hidden_dim'])}\nembed={int(row['embedding_dim'])}"


def plot_architecture_search(search: pd.DataFrame, primary: str) -> plt.Figure:
    require_columns(
        search,
        "architecture_search.csv",
        ["stage", "selected", "mean_cv_log_mae", "config_id"],
    )
    stages = [stage for stage in STAGE_LABELS if stage in set(search["stage"])]
    fig, axes = plt.subplots(1, len(stages), figsize=(5.0 * len(stages), 4.3))
    axes = np.atleast_1d(axes)
    for ax, stage in zip(axes, stages):
        frame = search[search["stage"].eq(stage)].sort_values("config_id").copy()
        selected = frame["selected"].astype(str).str.lower().eq("true")
        x = np.arange(len(frame))
        colors = np.where(selected, METHOD_COLORS[primary], "#B8C2CC")
        ax.bar(x, frame["mean_cv_log_mae"], color=colors, edgecolor="#333333")
        for position, (_, row) in zip(x, frame.iterrows()):
            if str(row["selected"]).lower() == "true":
                ax.scatter(
                    position,
                    row["mean_cv_log_mae"],
                    marker="*",
                    s=130,
                    color="#111111",
                    zorder=3,
                )
        ax.set_xticks(
            x,
            [architecture_candidate_label(stage, row) for _, row in frame.iterrows()],
            rotation=45 if stage == "tune_optimizer" else 0,
            ha="right" if stage == "tune_optimizer" else "center",
        )
        ax.set_title(STAGE_LABELS[stage])
        ax.set_ylabel("Crossed-validation log MAE")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Sequential App-Inhibitor Model Selection", fontsize=14, weight="bold")
    fig.text(
        0.5,
        -0.015,
        "Star and blue bar denote the candidate carried into the next stage.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout()
    return fig


def aggregate_estimates(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return frame.groupby(groups, as_index=False).agg(
        estimate=("estimate", "mean"),
        between_split_sd=("estimate", "std"),
        split_count=("estimate", "count"),
    )


def plot_endpoint_comparison(
    accuracy: pd.DataFrame, primary: str, comparison_size: int
) -> tuple[plt.Figure, pd.DataFrame]:
    frame = accuracy[
        accuracy["metric"].eq("log_mae")
        & accuracy["scope"].eq("overall_non_self")
        & accuracy["statistic"].eq("estimate")
    ].copy()
    random = frame[
        frame["evaluation_method"].eq("random_split")
        & frame["comparison_scope"].eq("protocol_native")
    ]
    restricted = frame[
        frame["evaluation_method"].isin(["one_known", "zero_shot"])
        & frame["training_app_count"].eq(comparison_size)
    ]
    aggregate = aggregate_estimates(
        pd.concat([random, restricted], ignore_index=True),
        ["evaluation_method", "comparison_scope", "method"],
    )
    methods = ordered_methods(aggregate["method"])
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.3), sharey=True)
    for ax, tier in zip(axes, ["random_split", "one_known", "zero_shot"]):
        native = aggregate[
            aggregate["evaluation_method"].eq(tier)
            & aggregate["comparison_scope"].eq("protocol_native")
        ].set_index("method")
        estimates = np.array(
            [native.loc[m, "estimate"] if m in native.index else np.nan for m in methods]
        )
        errors = np.array(
            [
                native.loc[m, "between_split_sd"] if m in native.index else np.nan
                for m in methods
            ]
        )
        ax.barh(
            y,
            estimates,
            xerr=np.nan_to_num(errors, nan=0.0),
            color=[METHOD_COLORS.get(m, "#777777") for m in methods],
            alpha=0.88,
            capsize=2,
        )
        if tier != "random_split":
            matched = aggregate[
                aggregate["evaluation_method"].eq(tier)
                & aggregate["comparison_scope"].eq("matched_random_split")
            ].set_index("method")
            matched_values = np.array(
                [
                    matched.loc[m, "estimate"] if m in matched.index else np.nan
                    for m in methods
                ]
            )
            ax.scatter(
                matched_values,
                y,
                marker="D",
                s=27,
                facecolor="white",
                edgecolor="#111111",
                linewidth=1.1,
                label="Matched full-data fit",
                zorder=3,
            )
            ax.legend(loc="lower right")
        ax.set_title(TIER_LABELS[tier])
        ax.set_xlabel("Directional log MAE (lower is better)")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [method_label(m, primary) for m in methods])
    axes[0].invert_yaxis()
    fig.suptitle(
        f"Endpoint Familiarity and Method Accuracy (restricted fits: {comparison_size} training apps)",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Error bars are descriptive SD across overlapping balanced rotations; the full-data fit is a single run.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig, aggregate


def learning_focus_methods(primary: str, available: Iterable[str]) -> list[str]:
    wanted = [
        primary,
        "absolute_response",
        "generic_potential",
        "nearest_anchor",
        "constant_1",
    ]
    unique = set(available)
    return list(dict.fromkeys(method for method in wanted if method in unique))


def plot_learning_curves(
    training: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    raw = training[
        training["metric"].eq("log_mae")
        & training["scope"].eq("overall_non_self")
        & training["statistic"].eq("estimate")
        & training["comparison_scope"].eq("protocol_native")
    ].copy()
    summary = aggregate_estimates(
        raw, ["training_app_count", "evaluation_method", "method"]
    )
    methods = learning_focus_methods(primary, raw["method"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)
    for ax, tier in zip(axes, ["one_known", "zero_shot"]):
        for method in methods:
            rows = summary[
                summary["evaluation_method"].eq(tier)
                & summary["method"].eq(method)
            ].sort_values("training_app_count")
            raw_rows = raw[
                raw["evaluation_method"].eq(tier) & raw["method"].eq(method)
            ]
            color = METHOD_COLORS[method]
            ax.scatter(
                raw_rows["training_app_count"],
                raw_rows["estimate"],
                color=color,
                alpha=0.18,
                s=14,
                zorder=1,
            )
            ax.errorbar(
                rows["training_app_count"],
                rows["estimate"],
                yerr=rows["between_split_sd"].fillna(0.0),
                color=color,
                marker="o",
                linewidth=2.0 if method == primary else 1.4,
                markersize=5,
                capsize=2,
                label=method_label(method, primary),
                zorder=2,
            )
        if tier == "zero_shot" and 8 in set(raw["training_app_count"]):
            ax.axvspan(7.75, 8.25, color="#999999", alpha=0.12)
            ax.text(
                8,
                ax.get_ylim()[1],
                "one pair",
                ha="center",
                va="top",
                fontsize=8,
                color="#555555",
            )
        ax.set_title(TIER_LABELS[tier])
        ax.set_xlabel("Applications used for fitting")
        ax.set_xticks(sorted(raw["training_app_count"].astype(int).unique()))
    axes[0].set_ylabel("Directional log MAE (lower is better)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training-Application Learning Curves", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.005,
        "Points show individual rotations; error bars show descriptive between-rotation SD.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig, summary


def plot_native_vs_matched(accuracy: pd.DataFrame, primary: str) -> plt.Figure:
    frame = accuracy[
        accuracy["metric"].eq("log_mae")
        & accuracy["scope"].eq("overall_non_self")
        & accuracy["statistic"].eq("estimate")
        & accuracy["method"].eq(primary)
        & accuracy["evaluation_method"].isin(["one_known", "zero_shot"])
    ].copy()
    summary = aggregate_estimates(
        frame,
        ["training_app_count", "evaluation_method", "comparison_scope", "method"],
    )
    scope_styles = {
        "protocol_native": ("Restricted fit", "o", "-", METHOD_COLORS[primary]),
        "matched_random_split": ("Matched full-data fit", "D", "--", "#222222"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    for ax, tier in zip(axes, ["one_known", "zero_shot"]):
        for scope, (label, marker, linestyle, color) in scope_styles.items():
            rows = summary[
                summary["evaluation_method"].eq(tier)
                & summary["comparison_scope"].eq(scope)
            ].sort_values("training_app_count")
            ax.errorbar(
                rows["training_app_count"],
                rows["estimate"],
                yerr=rows["between_split_sd"].fillna(0.0),
                label=label,
                marker=marker,
                linestyle=linestyle,
                color=color,
                capsize=2,
                linewidth=1.8,
            )
        if tier == "zero_shot":
            ax.axvspan(7.75, 8.25, color="#999999", alpha=0.12)
        ax.set_title(TIER_LABELS[tier])
        ax.set_xlabel("Applications used for fitting")
        ax.legend(loc="best")
    axes[0].set_ylabel("Directional log MAE (lower is better)")
    fig.suptitle(
        f"Training Restriction vs Pair Difficulty: {method_label(primary)}",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def random_non_self_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[
        predictions["stage"].eq("evaluate_random_split")
        & predictions["victim_id"].ne(predictions["aggressor_id"])
    ].copy()


def plot_prediction_calibration(
    predictions: pd.DataFrame, primary: str
) -> plt.Figure:
    random = random_non_self_predictions(predictions)
    methods = learning_focus_methods(primary, random["method"])
    methods = methods[:4]
    true_log = np.log(random["true_slowdown"].to_numpy(dtype=float))
    limit = max(float(np.nanmax(true_log)), float(random["predicted_log_slowdown"].max()))
    limit = max(limit * 1.04, 0.05)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        rows = random[random["method"].eq(method)]
        x = np.log(rows["true_slowdown"].to_numpy(dtype=float))
        y = rows["predicted_log_slowdown"].to_numpy(dtype=float)
        floor = np.isclose(x, 0.0)
        ax.scatter(
            x[~floor],
            y[~floor],
            s=26,
            alpha=0.65,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.25,
        )
        ax.scatter(
            x[floor],
            y[floor],
            s=28,
            alpha=0.8,
            marker="x",
            color="#222222",
            linewidth=0.8,
            label="Observed at floor",
        )
        ax.plot([0, limit], [0, limit], linestyle="--", color="#444444", linewidth=1)
        mae = float(np.mean(np.abs(y - x)))
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(
            0.04,
            0.94,
            f"MAE = {mae:.3f}\nSpearman = {rho:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_title(method_label(method, primary))
        ax.set_xlim(-0.01, limit)
        ax.set_ylim(-0.01, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed log slowdown")
    axes[0].set_ylabel("Predicted log slowdown")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Prediction Calibration on Non-Self App-App Outcomes", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def annotated_heatmap(
    ax: plt.Axes,
    values: pd.DataFrame,
    xlabels: list[str],
    ylabels: list[str],
    label_format: str = ".2f",
) -> Any:
    array = values.to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    vmax = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
    image = ax.imshow(
        np.ma.masked_invalid(array),
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=max(vmax, 1e-12)),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.grid(False)
    threshold = vmax * 0.58
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    format(value, label_format),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if value > threshold else "#222222",
                )
    return image


def plot_per_victim_heatmap(
    predictions: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    random = random_non_self_predictions(predictions)
    grouped = (
        random.groupby(["method", "victim_id"], as_index=False)["absolute_log_error"]
        .mean()
        .rename(columns={"absolute_log_error": "log_mae"})
    )
    overall = grouped.groupby("method")["log_mae"].mean().sort_values()
    methods = list(overall.index)
    victims = sorted(grouped["victim_id"].unique())
    matrix = grouped.pivot(index="method", columns="victim_id", values="log_mae").reindex(
        index=methods, columns=victims
    )
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    image = annotated_heatmap(
        ax,
        matrix,
        victims,
        [method_label(method, primary) for method in methods],
    )
    fig.colorbar(image, ax=ax, label="Directional log MAE")
    ax.set_xlabel("Victim application")
    ax.set_ylabel("")
    ax.set_title("Per-Victim Error Heterogeneity (full-data fit, non-self pairs)")
    fig.tight_layout()
    return fig, grouped


def plot_pairwise_error_heatmap(
    predictions: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    rows = random_non_self_predictions(predictions)
    rows = rows[rows["method"].eq(primary)]
    grouped = (
        rows.groupby(["victim_id", "aggressor_id"], as_index=False)[
            "absolute_log_error"
        ]
        .mean()
        .rename(columns={"absolute_log_error": "log_mae"})
    )
    applications = sorted(set(grouped["victim_id"]) | set(grouped["aggressor_id"]))
    matrix = grouped.pivot(
        index="victim_id", columns="aggressor_id", values="log_mae"
    ).reindex(index=applications, columns=applications)
    fig, ax = plt.subplots(figsize=(8.5, 7.4))
    image = annotated_heatmap(ax, matrix, applications, applications)
    fig.colorbar(image, ax=ax, label="Absolute log error")
    ax.set_xlabel("Co-scheduled aggressor application")
    ax.set_ylabel("Victim application")
    ax.set_title(f"Directional Pairwise Error: {method_label(primary)}")
    fig.tight_layout()
    return fig, grouped


def diagnostic_correlations(
    predictions: pd.DataFrame, primary: str
) -> pd.DataFrame:
    rows = random_non_self_predictions(predictions)
    rows = rows[rows["method"].eq(primary)]
    diagnostics = [
        "nearest_anchor_distance",
        "ood_score",
        "model_seed_log_std",
        "unweighted_residual_std",
        "effective_anchor_count",
    ]
    records = []
    for diagnostic in diagnostics:
        if diagnostic not in rows:
            continue
        finite = rows[[diagnostic, "absolute_log_error"]].dropna()
        if len(finite) < 2 or finite[diagnostic].nunique() < 2:
            rho = np.nan
        else:
            rho = finite[diagnostic].corr(
                finite["absolute_log_error"], method="spearman"
            )
        records.append(
            {
                "method": primary,
                "diagnostic": diagnostic,
                "n": len(finite),
                "spearman_with_absolute_log_error": rho,
            }
        )
    return pd.DataFrame(records)


def plot_error_diagnostics(
    predictions: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    rows = random_non_self_predictions(predictions)
    rows = rows[rows["method"].eq(primary)]
    diagnostics = [
        ("nearest_anchor_distance", "Nearest proxy distance"),
        ("ood_score", "OOD score"),
        ("model_seed_log_std", "Model-seed log SD"),
    ]
    correlations = diagnostic_correlations(predictions, primary)
    fig, axes = plt.subplots(1, len(diagnostics), figsize=(13.0, 4.1), sharey=True)
    for ax, (column, label) in zip(axes, diagnostics):
        finite = rows[[column, "absolute_log_error"]].dropna().copy()
        ax.scatter(
            finite[column],
            finite["absolute_log_error"],
            s=22,
            alpha=0.35,
            color=METHOD_COLORS[primary],
        )
        if finite[column].nunique() >= 4:
            finite["bin"] = pd.qcut(finite[column], q=5, duplicates="drop")
            binned = finite.groupby("bin", observed=True).agg(
                x=(column, "median"),
                error=("absolute_log_error", "mean"),
            )
            ax.plot(
                binned["x"],
                binned["error"],
                color="#111111",
                marker="o",
                linewidth=1.8,
                label="Equal-count bin mean",
            )
        match = correlations[correlations["diagnostic"].eq(column)]
        rho = float(match["spearman_with_absolute_log_error"].iloc[0])
        ax.text(
            0.04,
            0.94,
            f"Spearman = {rho:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_xlabel(label)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Absolute log error")
    fig.suptitle(
        f"Proxy-Support and Uncertainty Diagnostics: {method_label(primary)}",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, correlations


def primary_paired_differences(
    metrics: pd.DataFrame, primary: str
) -> pd.DataFrame:
    rows = metrics[
        metrics["stage"].eq("evaluate_random_split")
        & metrics["scope"].eq("non_self_paired_difference")
        & metrics["statistic"].eq("method_minus_comparison")
        & metrics["metric"].eq("log_mae")
        & (
            metrics["method"].eq(primary)
            | metrics["comparison_method"].eq(primary)
        )
    ]
    records = []
    for _, row in rows.iterrows():
        if row["method"] == primary:
            comparator = row["comparison_method"]
            estimate = row["estimate"]
            lower = row["ci_lower"]
            upper = row["ci_upper"]
        else:
            comparator = row["method"]
            estimate = -row["estimate"]
            lower = -row["ci_upper"]
            upper = -row["ci_lower"]
        records.append(
            {
                "primary_method": primary,
                "comparison_method": comparator,
                "primary_minus_comparison": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(records).sort_values("primary_minus_comparison")


def plot_bootstrap_differences(
    metrics: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    differences = primary_paired_differences(metrics, primary)
    if differences.empty:
        raise ValueError(f"No non-self paired bootstrap comparisons found for {primary}")
    y = np.arange(len(differences))
    estimates = differences["primary_minus_comparison"].to_numpy(dtype=float)
    lower = differences["ci_lower"].to_numpy(dtype=float)
    upper = differences["ci_upper"].to_numpy(dtype=float)
    significant = (lower > 0) | (upper < 0)
    colors = np.where(significant, METHOD_COLORS[primary], "#999999")
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        ecolor="#555555",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(estimates, y, c=colors, s=48, zorder=2)
    ax.axvline(0, color="#111111", linestyle="--", linewidth=1)
    ax.set_yticks(
        y,
        [method_label(method) for method in differences["comparison_method"]],
    )
    ax.set_xlabel("Log MAE difference: selected low-rank minus comparator")
    ax.set_title("Paired App-App Bootstrap Comparisons (95% CI, non-self pairs)")
    ax.text(
        0.01,
        -0.12,
        "Negative favors selected low-rank; positive favors comparator.",
        transform=ax.transAxes,
        color="#555555",
    )
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, differences


def plot_floor_strata(
    predictions: pd.DataFrame, primary: str
) -> tuple[plt.Figure, pd.DataFrame]:
    rows = random_non_self_predictions(predictions)
    rows["target_stratum"] = np.where(
        np.isclose(rows["true_slowdown"], 1.0), "Observed at floor", "Slowdown > 1"
    )
    summary = rows.groupby(["method", "target_stratum"], as_index=False).agg(
        log_mae=("absolute_log_error", "mean"), n=("absolute_log_error", "size")
    )
    methods = ordered_methods(summary["method"])
    y = np.arange(len(methods))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    styles = [
        ("Observed at floor", "#8DA0CB", -height / 2),
        ("Slowdown > 1", "#FC8D62", height / 2),
    ]
    for stratum, color, offset in styles:
        indexed = summary[summary["target_stratum"].eq(stratum)].set_index("method")
        values = [indexed.loc[m, "log_mae"] for m in methods]
        count = int(indexed["n"].iloc[0]) if not indexed.empty else 0
        ax.barh(y + offset, values, height=height, color=color, label=f"{stratum} (n={count})")
    ax.set_yticks(y, [method_label(method, primary) for method in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Directional log MAE (lower is better)")
    ax.set_title("Accuracy by Censored-Target Stratum (full-data fit, non-self pairs)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, summary


def method_ranking(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = random_non_self_predictions(predictions)
    ranking = rows.groupby("method", as_index=False).agg(
        n=("absolute_log_error", "size"),
        log_mae=("absolute_log_error", "mean"),
        log_rmse=("absolute_log_error", lambda value: float(np.sqrt(np.mean(value**2)))),
        raw_mae=("absolute_error", "mean"),
        median_absolute_log_error=("absolute_log_error", "median"),
    )
    return ranking.sort_values("log_mae").reset_index(drop=True)


def completion_summary(root: Path) -> dict[str, Any]:
    manifest_path = root / "experiment_manifest.csv"
    if not manifest_path.exists():
        return {"manifest_available": False}
    manifest = pd.read_csv(manifest_path)
    counts = manifest["status"].fillna("unknown").value_counts().to_dict()
    return {
        "manifest_available": True,
        "planned_tasks": int(len(manifest)),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "complete": bool(len(manifest) and (manifest["status"] == "succeeded").all()),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    consolidated = root / "consolidated"
    output_dir = (args.output_dir or root / "analysis").resolve()
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    accuracy = load_csv(consolidated, "accuracy_by_eval_method.csv")
    training = load_csv(consolidated, "accuracy_by_training_size.csv")
    learning = load_csv(consolidated, "learning_curve_summary.csv")
    search = load_csv(consolidated, "architecture_search.csv")
    metrics = load_csv(consolidated, "metrics_long.csv")
    predictions = load_csv(consolidated, "predictions.csv", required=False)
    selection_path = consolidated / "global_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing {selection_path}. Complete tuning and consolidate the run first."
        )
    selection = json.loads(selection_path.read_text())

    require_columns(
        accuracy,
        "accuracy_by_eval_method.csv",
        [
            "method",
            "metric",
            "estimate",
            "evaluation_method",
            "comparison_scope",
            "scope",
            "statistic",
            "is_global_primary_low_rank",
        ],
    )
    require_columns(
        training,
        "accuracy_by_training_size.csv",
        ["training_app_count", "evaluation_method", "method", "metric", "estimate"],
    )
    if not predictions.empty:
        require_columns(
            predictions,
            "predictions.csv",
            [
                "stage",
                "method",
                "victim_id",
                "aggressor_id",
                "true_slowdown",
                "predicted_log_slowdown",
                "absolute_log_error",
            ],
        )

    configure_style()
    primary = primary_method(selection, accuracy)
    comparison_size = choose_comparison_size(learning, primary, args.comparison_size)
    figures: list[dict[str, Any]] = []

    def emit(stem: str, description: str, fig: plt.Figure) -> None:
        paths = save_figure(fig, plots_dir, stem, args.formats, args.dpi)
        figures.append(
            {
                "figure": stem,
                "description": description,
                "files": ";".join(
                    str(Path(path).relative_to(output_dir)) for path in paths
                ),
            }
        )

    emit(
        "01_model_selection",
        "Sequential App-Inhibitor crossed-validation search and selected candidates.",
        plot_architecture_search(search, primary),
    )
    endpoint_fig, endpoint_table = plot_endpoint_comparison(
        accuracy, primary, comparison_size
    )
    emit(
        "02_endpoint_comparison",
        "Non-self log MAE by endpoint familiarity, including matched full-data fits.",
        endpoint_fig,
    )
    endpoint_table.to_csv(tables_dir / "endpoint_comparison.csv", index=False)

    learning_fig, learning_table = plot_learning_curves(training, primary)
    emit(
        "03_learning_curves",
        "Accuracy versus fitting-application count with descriptive split variation.",
        learning_fig,
    )
    learning_table.to_csv(tables_dir / "learning_curve_summary.csv", index=False)
    emit(
        "04_native_vs_matched",
        "Selected-model restricted-fit performance versus matched full-data predictions.",
        plot_native_vs_matched(accuracy, primary),
    )

    if not predictions.empty:
        emit(
            "05_prediction_calibration",
            "Observed versus predicted log slowdown for the full-data fit.",
            plot_prediction_calibration(predictions, primary),
        )
        victim_fig, victim_table = plot_per_victim_heatmap(predictions, primary)
        emit(
            "06_per_victim_error",
            "Per-victim non-self log MAE heatmap for all methods.",
            victim_fig,
        )
        victim_table.to_csv(tables_dir / "per_victim_log_mae.csv", index=False)
        pair_fig, pair_table = plot_pairwise_error_heatmap(predictions, primary)
        emit(
            "07_pairwise_error",
            "Victim-aggressor absolute log error heatmap for the selected model.",
            pair_fig,
        )
        pair_table.to_csv(tables_dir / "pairwise_log_error.csv", index=False)
        diagnostic_fig, correlations = plot_error_diagnostics(predictions, primary)
        emit(
            "08_error_diagnostics",
            "Associations between proxy-support diagnostics and selected-model error.",
            diagnostic_fig,
        )
        correlations.to_csv(tables_dir / "diagnostic_correlations.csv", index=False)
        floor_fig, floor_table = plot_floor_strata(predictions, primary)
        emit(
            "10_floor_stratification",
            "Method accuracy separated by censored-floor and positive-slowdown outcomes.",
            floor_fig,
        )
        floor_table.to_csv(tables_dir / "floor_stratification.csv", index=False)
        method_ranking(predictions).to_csv(
            tables_dir / "random_split_method_ranking.csv", index=False
        )

    bootstrap_fig, differences = plot_bootstrap_differences(metrics, primary)
    emit(
        "09_paired_bootstrap",
        "Paired cluster-bootstrap log-MAE differences for the selected model.",
        bootstrap_fig,
    )
    differences.to_csv(tables_dir / "paired_bootstrap_differences.csv", index=False)

    figure_index = pd.DataFrame(figures).sort_values("figure")
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)
    completion = completion_summary(root)
    observed_splits = sorted(training["split_id"].dropna().astype(str).unique())
    summary = {
        "experiment_root": str(root),
        "primary_metric": "non-self directional log MAE",
        "primary_method": primary,
        "comparison_training_app_count": comparison_size,
        "figure_count": len(figures),
        "formats": list(args.formats),
        "predictions_available": not predictions.empty,
        "observed_training_split_count": len(observed_splits),
        "observed_training_splits": observed_splits,
        "experiment_completion": completion,
        "interpretation_notes": [
            "Between-split SD is descriptive because balanced cyclic subsets overlap.",
            "Model seeds are repeated fits, not independent samples.",
            "One-known and zero-shot model selection is transductive with respect to App-Inhibitor hyperparameters.",
            "The historical App-App data are not a pristine untouched holdout.",
            "The training-size-eight zero-shot condition has one non-self pair per split.",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    status = "complete" if completion.get("complete") else "partial"
    print(f"Generated {len(figures)} figures in {plots_dir}")
    print(f"Primary method: {primary}; endpoint comparison size: {comparison_size}")
    print(f"Experiment manifest status: {status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
