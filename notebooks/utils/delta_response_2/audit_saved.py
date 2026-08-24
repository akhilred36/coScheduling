"""Recompute App-Inhibitor-only handoff diagnostics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .data import load_training_data


def _describe(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def audit(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = load_training_data(args.jobs_csv, args.inhibitors_csv, args.job_inh_csv)
    targets = data.responses["log_slowdown"].to_numpy(dtype=float)
    target_summary = {
        **_describe(targets),
        "floor_count": int((targets == 0).sum()),
        "floor_fraction": float(np.mean(targets == 0)),
        "uncensored_mean": float(np.mean(targets[targets > 0])),
    }
    (output / "target_distribution.json").write_text(
        json.dumps(target_summary, indent=2, sort_keys=True) + "\n"
    )
    victim_rows = []
    for victim, frame in data.responses.groupby("job_id", sort=False):
        values = frame["log_slowdown"].to_numpy(dtype=float)
        victim_rows.append(
            {
                "victim_id": victim,
                "count": len(values),
                "floor_count": int((values == 0).sum()),
                "floor_fraction": float(np.mean(values == 0)),
                "mean_log_slowdown": float(np.mean(values)),
            }
        )
    pd.DataFrame(victim_rows).to_csv(output / "target_distribution_per_victim.csv", index=False)

    if args.saved_predictions is None:
        return
    selected_chunks = []
    for chunk in pd.read_csv(args.saved_predictions, chunksize=100_000):
        selected = chunk[
            (chunk["config_id"] == args.config_id)
            & (chunk["model_kind"] == "low_rank")
            & (chunk["method"] == "ood_kernel")
            & np.isclose(chunk["temperature"], args.temperature)
            & np.isclose(chunk["ood_quantile"], args.ood_quantile)
            & (chunk["ood_fallback"] == args.ood_fallback)
        ]
        if len(selected):
            selected_chunks.append(selected)
    if not selected_chunks:
        raise RuntimeError("The selected saved-prediction recipe was not found")
    selected = pd.concat(selected_chunks, ignore_index=True)
    selected.to_csv(output / "selected_saved_predictions.csv", index=False)
    keys = ["victim_id", "heldout_block", "query_inhib_id", "true_log_slowdown"]
    ensemble = selected.groupby(keys, as_index=False).agg(
        latent_predicted_log_slowdown=("latent_predicted_log_slowdown", "mean"),
        nearest_anchor_distance=("nearest_anchor_distance", "mean"),
        seed_log_std=("latent_predicted_log_slowdown", lambda values: float(np.std(values))),
    )
    ensemble["predicted_log_slowdown"] = np.maximum(
        0.0, ensemble["latent_predicted_log_slowdown"]
    )
    error = ensemble["predicted_log_slowdown"] - ensemble["true_log_slowdown"]
    grouped_seed_mae = (
        selected.assign(
            recomputed_absolute_log_error=(
                selected["predicted_log_slowdown"] - selected["true_log_slowdown"]
            ).abs()
        )
        .groupby(["victim_id", "heldout_block", "seed"])[
            "recomputed_absolute_log_error"
        ]
        .mean()
        .groupby(["victim_id", "heldout_block"])
        .mean()
    )
    prediction_summary = {
        "count": len(ensemble),
        "ensemble_mean_prediction": float(ensemble["predicted_log_slowdown"].mean()),
        "mean_target": float(ensemble["true_log_slowdown"].mean()),
        "ensemble_signed_log_bias": float(error.mean()),
        "ensemble_query_weighted_log_mae": float(np.abs(error).mean()),
        "ensemble_pearson": float(
            pearsonr(ensemble["predicted_log_slowdown"], ensemble["true_log_slowdown"])[0]
        ),
        "ensemble_spearman": float(
            spearmanr(ensemble["predicted_log_slowdown"], ensemble["true_log_slowdown"]).correlation
        ),
        "pooled_seed_query_weighted_log_mae": float(
            np.mean(
                np.abs(
                    selected["predicted_log_slowdown"]
                    - selected["true_log_slowdown"]
                )
            )
        ),
        "grouped_seed_averaged_log_mae": float(grouped_seed_mae.mean()),
        "pooled_seed_pearson": float(
            pearsonr(
                selected["predicted_log_slowdown"], selected["true_log_slowdown"]
            )[0]
        ),
        "pooled_seed_spearman": float(
            spearmanr(
                selected["predicted_log_slowdown"], selected["true_log_slowdown"]
            ).correlation
        ),
        "floor_mae": float(
            np.abs(error[ensemble["true_log_slowdown"] == 0]).mean()
        ),
        "positive_mae": float(
            np.abs(error[ensemble["true_log_slowdown"] > 0]).mean()
        ),
    }
    (output / "selected_prediction_summary.json").write_text(
        json.dumps(prediction_summary, indent=2, sort_keys=True) + "\n"
    )
    seed_rows = []
    for seed, frame in selected.groupby("seed"):
        seed_error = (
            frame["predicted_log_slowdown"].to_numpy(dtype=float)
            - frame["true_log_slowdown"].to_numpy(dtype=float)
        )
        seed_rows.append(
            {
                "seed": int(seed),
                "log_mae": float(np.abs(seed_error).mean()),
                "signed_log_bias": float(seed_error.mean()),
            }
        )
    pd.DataFrame(seed_rows).to_csv(output / "seed_stability.csv", index=False)
    ensemble.to_csv(output / "ensemble_saved_predictions.csv", index=False)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    data = package.parent / "few_shot" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, default=data / "jobs.csv")
    parser.add_argument("--inhibitors-csv", type=Path, default=data / "inhibitors.csv")
    parser.add_argument("--job-inh-csv", type=Path, default=data / "job_inh.csv")
    parser.add_argument("--saved-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--ood-quantile", type=float, default=0.90)
    parser.add_argument("--ood-fallback", default="uniform")
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
