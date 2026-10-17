"""Select a small, application-targeted inhibitor panel without App-App outcomes.

Distances are RMS standardized differences in isolated communication intensity.
The scaler sees only the requested target applications, never the inhibitor grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import BASE_FEATURES, load_training_data


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "app_time"
FEATURES = ["comm_frac", "log1p_message_rate", "log1p_byte_rate", "log1p_mean_message_size"]


def communication_features(frame: pd.DataFrame) -> np.ndarray:
    """Use rates over estimated *total* runtime, not just MPI-active time.

    Undefined runtime or contradictory zero-message profiles fail explicitly;
    division by an epsilon would turn them into artificial extreme candidates.
    """
    values = frame[BASE_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Communication profiles must be finite and non-negative")
    mpi = frame["mpi_time"].to_numpy(dtype=float)
    frac = frame["comm_frac"].to_numpy(dtype=float)
    msgs = frame["total_msgs"].to_numpy(dtype=float)
    size = frame["total_bytes"].to_numpy(dtype=float)
    if (mpi <= 0).any() or (frac <= 0).any() or (frac > 1).any():
        raise ValueError("Rate geometry requires mpi_time > 0 and 0 < comm_frac <= 1")
    if ((msgs == 0) & (size > 0)).any():
        raise ValueError("Positive total_bytes with zero total_msgs is inconsistent")
    runtime = mpi / frac
    mean_size = np.divide(size, msgs, out=np.zeros_like(size), where=msgs > 0)
    result = np.column_stack(
        [frac, np.log1p(msgs / runtime), np.log1p(size / runtime), np.log1p(mean_size)]
    )
    if not np.isfinite(result).all():
        raise ValueError("Derived communication features must be finite")
    return result


def coverage_score(nearest_squared: np.ndarray, tail_weight: float) -> float:
    return float(
        (1 - tail_weight) * nearest_squared.mean()
        + tail_weight * nearest_squared.max()
    )


def select_panel(
    distances_squared: np.ndarray, budget: int = 10, tail_weight: float = 0.5
) -> tuple[list[int], list[dict], str]:
    """Facility-location coverage with deterministic forward selection + swaps.

    When the distinct per-application nearest candidates fit the budget, their
    union achieves the pointwise lower bound and is globally optimal. Otherwise
    this is a heuristic, refined to a one-swap local optimum. Input candidates
    must be sorted by ID to make distance ties independent of CSV row order.
    """
    d = np.asarray(distances_squared, dtype=float)
    if d.ndim != 2 or min(d.shape) == 0 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Expected a nonempty finite non-negative distance matrix")
    if not isinstance(budget, int) or not 1 <= budget <= 10:
        raise ValueError("budget must be an integer in [1, 10]")
    if not np.isfinite(tail_weight) or not 0 <= tail_weight <= 1:
        raise ValueError("tail_weight must lie in [0, 1]")
    budget = min(budget, d.shape[1])
    trace = []
    nearest_union = sorted(set(d.argmin(axis=1).tolist()))
    if len(nearest_union) <= budget:
        selected = nearest_union
        status = "global_optimum_pointwise_lower_bound"
    else:
        selected = []
        nearest = np.full(d.shape[0], np.inf)
        for step in range(budget):
            scores = np.array([
                coverage_score(np.minimum(nearest, d[:, j]), tail_weight)
                if j not in selected else np.inf
                for j in range(d.shape[1])
            ])
            candidate = int(scores.argmin())
            if selected and scores[candidate] >= coverage_score(nearest, tail_weight) - 1e-12:
                break
            selected.append(candidate)
            nearest = np.minimum(nearest, d[:, candidate])
            trace.append({"operation": "add", "candidate_index": candidate,
                          "objective": float(scores[candidate])})
        while True:
            score = coverage_score(d[:, selected].min(axis=1), tail_weight)
            best_score, best_swap = score, None
            for old in sorted(selected):
                retained = [j for j in selected if j != old]
                nearest = (d[:, retained].min(axis=1) if retained
                           else np.full(d.shape[0], np.inf))
                for new in range(d.shape[1]):
                    if new in selected:
                        continue
                    trial = coverage_score(np.minimum(nearest, d[:, new]), tail_weight)
                    if trial < best_score - 1e-12:
                        best_score, best_swap = trial, (old, new)
            if best_swap is None:
                break
            old, new = best_swap
            selected.remove(old)
            selected.append(new)
            trace.append({"operation": "swap", "removed_index": old,
                          "candidate_index": new, "objective": best_score})
        status = "heuristic_one_swap_local_optimum"
    # Remove objectively redundant anchors; the budget is a cap, not a quota.
    for old in sorted(selected, reverse=True):
        if len(selected) <= 1:
            break
        trial = [j for j in selected if j != old]
        if np.allclose(d[:, trial].min(axis=1), d[:, selected].min(axis=1),
                       rtol=0, atol=1e-12):
            selected = trial
    return sorted(selected), trace, status


def coverage_summary(distances_squared: np.ndarray, tail_weight: float) -> dict:
    nearest = distances_squared.min(axis=1)
    return {
        "objective": coverage_score(nearest, tail_weight),
        "mean_nearest_distance": float(np.sqrt(nearest).mean()),
        "worst_nearest_distance": float(np.sqrt(nearest).max()),
    }


def run(args: argparse.Namespace) -> dict:
    data = load_training_data(args.jobs_csv, args.inhibitors_csv, args.job_inh_csv)
    jobs = data.jobs.sort_values("job_id").reset_index(drop=True)
    if args.target_apps is not None:
        ids = {value.strip() for value in args.target_apps.split(",") if value.strip()}
        missing = ids - set(jobs["job_id"])
        if missing or not ids:
            raise ValueError(f"Invalid --target-apps; unknown IDs: {sorted(missing)}")
        jobs = jobs[jobs["job_id"].isin(ids)].reset_index(drop=True)
    # Availability is a quality constraint. Response magnitude is never a score.
    responses = data.responses[data.responses["job_id"].isin(jobs["job_id"])]
    counts = responses.groupby("inhib_id")["job_id"].nunique()
    inhibitors = data.inhibitors.sort_values("inhib_id").reset_index(drop=True)
    eligible = inhibitors["inhib_id"].map(counts).fillna(0).eq(len(jobs))
    candidates = inhibitors[eligible].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No inhibitor has a valid response for every target application")
    x, z = communication_features(jobs), communication_features(candidates)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    # A feature constant among apps has no target range to standardize. Exclude
    # it explicitly rather than giving tiny noise an arbitrarily large weight.
    active = scale > 1e-8
    if not active.any():
        raise ValueError("Target profiles have no varying communication features")
    scale[~active] = 1.0
    distance = (((x[:, None, active] - z[None, :, active]) / scale[active]) ** 2).mean(axis=2)
    selected, trace, status = select_panel(distance, args.max_profiles, args.tail_weight)
    panel = candidates.iloc[selected].copy()
    chosen_ids = panel["inhib_id"].tolist()
    nearest = distance[:, selected].argmin(axis=1)
    all_nearest = distance.argmin(axis=1)
    assignments = pd.DataFrame({
        "job_id": jobs["job_id"],
        "selected_inhib_id": [chosen_ids[j] for j in nearest],
        "selected_distance": np.sqrt(distance[:, selected].min(axis=1)),
        "best_available_inhib_id": candidates.iloc[all_nearest]["inhib_id"].to_numpy(),
        "best_available_distance": np.sqrt(distance.min(axis=1)),
    })
    assignments["squared_distance_regret"] = distance[:, selected].min(axis=1) - distance.min(axis=1)
    # Descriptive one-target-SD flag, not a validated transfer/OOD threshold.
    assignments["distance_exceeds_one_target_sd"] = assignments["selected_distance"] > 1
    panel_responses = data.responses[data.responses["inhib_id"].isin(chosen_ids)].copy()
    anchor_quality = []
    for job_id in jobs["job_id"]:
        rows = panel_responses[panel_responses["job_id"] == job_id]
        anchor_quality.append({
            "job_id": job_id, "valid_inhibitor_count": int(rows["inhib_id"].nunique()),
            "uncensored_inhibitor_count": int(rows.loc[rows["slowdown"] > 1, "inhib_id"].nunique()),
            "floor_row_fraction": float(rows["slowdown"].eq(1).mean()),
        })
    ranges = []
    for j, feature in enumerate(FEATURES):
        ranges.append({
            "feature": feature, "active_in_distance": bool(active[j]),
            "target_min": float(x[:, j].min()), "target_max": float(x[:, j].max()),
            "candidate_min": float(z[:, j].min()), "candidate_max": float(z[:, j].max()),
            "selected_min": float(z[selected, j].min()), "selected_max": float(z[selected, j].max()),
        })
    rng = np.random.default_rng(1701)
    random_scores = [coverage_summary(distance[:, rng.choice(len(candidates), len(selected), replace=False)],
                                      args.tail_weight) for _ in range(200)]
    report = {
        "method": "application_targeted_facility_location",
        "optimization_status": status,
        "max_profiles": args.max_profiles, "selected_count": len(selected),
        "selected_ids": chosen_ids, "target_apps": jobs["job_id"].tolist(),
        "profile_selection_uses_response_magnitude": False,
        "app_app_outcomes_loaded": False,
        "target_profile_use": "transductive; use --target-apps to restrict the selection population",
        "input_inhibitor_count": len(inhibitors), "eligible_inhibitor_count": len(candidates),
        "excluded_missing_target_responses": inhibitors.loc[~eligible, "inhib_id"].tolist(),
        "rejected_response_rows": len(data.rejected_responses),
        "features": FEATURES, "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
        "active_features": active.tolist(), "tail_weight": args.tail_weight,
        "objective_definition": "(1-tail_weight)*mean(nearest_squared_distance) + tail_weight*max(nearest_squared_distance)",
        "selected_coverage": coverage_summary(distance[:, selected], args.tail_weight),
        "all_eligible_coverage_lower_bound": coverage_summary(distance, args.tail_weight),
        "random_same_size_baseline": {
            "seed": 1701, "draws": 200,
            **{key + "_median": float(np.median([row[key] for row in random_scores]))
               for key in random_scores[0]},
        },
        "anchor_availability": anchor_quality,
        "selection_trace": trace,
        "input_sha256": {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in [args.jobs_csv, args.inhibitors_csv, args.job_inh_csv]},
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "inhibitors.csv", index=False)
    data.jobs.to_csv(output / "jobs.csv", index=False)
    # Keep raw valid measurements/replicates in the pipeline's input schema.
    panel_responses.drop(columns=["log_slowdown", "is_censored", "replicate_id"]).to_csv(
        output / "job_inh.csv", index=False)
    assignments.to_csv(output / "application_coverage.csv", index=False)
    pd.DataFrame(ranges).to_csv(output / "feature_ranges.csv", index=False)
    pd.DataFrame(anchor_quality).to_csv(output / "anchor_availability.csv", index=False)
    pd.DataFrame(np.sqrt(distance), index=jobs["job_id"], columns=candidates["inhib_id"]).to_csv(
        output / "candidate_distances.csv")
    (output / "selection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, default=DEFAULT_DATA / "jobs.csv")
    parser.add_argument("--inhibitors-csv", type=Path, default=DEFAULT_DATA / "inhibitors.csv")
    parser.add_argument("--job-inh-csv", type=Path, default=DEFAULT_DATA / "job_inh.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "selected_inhibitors")
    parser.add_argument("--max-profiles", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--tail-weight", type=float, default=0.5,
                        help="Worst-app coverage weight in [0,1]; remaining weight is mean-app coverage")
    parser.add_argument("--target-apps", help="Comma-separated selection population; default: all isolated apps")
    args = parser.parse_args()
    if not np.isfinite(args.tail_weight) or not 0 <= args.tail_weight <= 1:
        parser.error("--tail-weight must lie in [0,1]")
    report = run(args)
    print(f"Selected {report['selected_count']} of {report['eligible_inhibitor_count']} eligible inhibitors")
    print("\n".join(report["selected_ids"]))
    print(json.dumps(report["selected_coverage"], indent=2))
    print(f"Optimization: {report['optimization_status']}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
