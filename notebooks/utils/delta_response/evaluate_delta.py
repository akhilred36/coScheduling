"""
Evaluation script for ResponseScoreModel - transitive App-App evaluation.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from data import load_and_prepare_data, expand_pair_df, ProfileNormalizer
from model import ResponseScoreModel


warnings.filterwarnings("ignore")


def aggregate_predictions(anchor_log_preds, z_a_anchors, z_a_target, temperature=1.0):
    """
    Similarity-weighted average of anchor predictions.
    
    anchor_log_preds: (num_anchors,) predicted log-slowdown from each anchor
    z_a_anchors: (num_anchors, role_dim) aggressor embeddings of each anchor inhibitor
    z_a_target: (role_dim,) aggressor embedding of the target (App B)
    """
    dists_sq = ((z_a_anchors - z_a_target[None, :]) ** 2).sum(dim=-1)
    weights = torch.softmax(-dists_sq / temperature, dim=0)
    return (weights * anchor_log_preds).sum()


def aggregate_unweighted(anchor_log_preds):
    """Simple mean aggregation."""
    return anchor_log_preds.mean()


def aggregate_median(anchor_log_preds):
    """Simple median aggregation."""
    return torch.median(anchor_log_preds)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = ResponseScoreModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def compute_metrics(predictions, targets):
    log_pred = np.log(predictions)
    log_true = np.log(targets)

    mae_log = np.mean(np.abs(log_pred - log_true))
    rmse_log = np.sqrt(np.mean((log_pred - log_true) ** 2))
    mae_raw = np.mean(np.abs(predictions - targets))
    rmse_raw = np.sqrt(np.mean((predictions - targets) ** 2))
    spearman_corr, _ = spearmanr(predictions, targets)

    return {
        "log_mae": mae_log,
        "log_rmse": rmse_log,
        "raw_mae": mae_raw,
        "raw_rmse": rmse_raw,
        "spearman": spearman_corr,
    }


def evaluate_transitive(model, data, device, temperature=1.0):
    jobs_df = data["jobs_df"]
    inhib_df = data["inhibitors_df"]
    job_inh_df = data["job_inh_df"]
    pair_df = data["pair_df"]
    normalizer = data["normalizer"]

    jobs_index = {jid: i for i, jid in enumerate(jobs_df["job_id"])}
    inhib_index = {iid: i for i, iid in enumerate(inhib_df["inhib_id"])}

    directional_df = expand_pair_df(pair_df)
    directional_df = directional_df.rename(columns={"aggressor_job_id": "target_id"})

    job_ids = jobs_df["job_id"].values
    jobs_feat = normalizer.transform(jobs_df.set_index("job_id"))
    inhib_feat = normalizer.transform(inhib_df.set_index("inhib_id"))

    model.eval()
    all_predictions = []
    all_targets = []
    all_std_dev = []
    prediction_rows = []

    with torch.no_grad():
        for _, row in directional_df.iterrows():
            victim_id = row["job_id"]
            target_id = row["target_id"]
            true_slowdown = row["slowdown"]

            victim_idx = jobs_index[victim_id]
            victim_profile = torch.tensor(jobs_feat[victim_idx], dtype=torch.float32).unsqueeze(0).to(device)

            target_idx = jobs_index[target_id]
            target_profile = torch.tensor(jobs_feat[target_idx], dtype=torch.float32).unsqueeze(0).to(device)

            anchor_log_preds = []
            z_a_anchors_list = []

            anchor_rows = job_inh_df[job_inh_df["job_id"] == victim_id]
            if len(anchor_rows) == 0:
                continue

            for _, anchor_row in anchor_rows.iterrows():
                inhib_id = anchor_row["inhib_id"]
                anchor_slowdown = anchor_row["slowdown"]
                inhib_idx = inhib_index[inhib_id]

                aggressor_profile = torch.tensor(inhib_feat[inhib_idx], dtype=torch.float32).unsqueeze(0).to(device)
                target_aggressor_profile = torch.tensor(inhib_feat[inhib_idx], dtype=torch.float32).unsqueeze(0).to(device)

                r_anchor = model.response_score(victim_profile, aggressor_profile)
                r_target = model.response_score(victim_profile, target_profile)

                delta = r_target - r_anchor
                pred_log_slowdown = np.log(anchor_slowdown) + delta.item()
                anchor_log_preds.append(pred_log_slowdown)

                z_a_anchor = model.aggressor_head(model.encoder(aggressor_profile)).squeeze(0).cpu()
                z_a_anchors_list.append(z_a_anchor)

            anchor_log_preds = torch.tensor(np.array(anchor_log_preds), dtype=torch.float32)
            z_a_anchors = torch.stack(z_a_anchors_list)
            z_a_target = model.aggressor_head(model.encoder(target_profile)).squeeze(0).cpu()

            pred_log = aggregate_predictions(anchor_log_preds, z_a_anchors, z_a_target, temperature)
            pred_slowdown = np.exp(pred_log.item())

            anchor_std = float(torch.std(anchor_log_preds).item())
            all_predictions.append(pred_slowdown)
            all_targets.append(true_slowdown)
            all_std_dev.append(anchor_std)

            prediction_rows.append({
                "victim_id": victim_id,
                "target_id": target_id,
                "true_slowdown": true_slowdown,
                "predicted_slowdown": pred_slowdown,
                "abs_error": abs(pred_slowdown - true_slowdown),
                "anchor_std": anchor_std,
            })

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    metrics = compute_metrics(predictions, targets)
    metrics["anchor_std_mean"] = np.mean(all_std_dev)
    metrics["anchor_std_std"] = np.std(all_std_dev)

    predictions_df = pd.DataFrame(prediction_rows)

    return predictions_df, metrics


def evaluate_unweighted(model, data, device):
    jobs_df = data["jobs_df"]
    job_inh_df = data["job_inh_df"]
    pair_df = data["pair_df"]
    normalizer = data["normalizer"]

    jobs_index = {jid: i for i, jid in enumerate(jobs_df["job_id"])}
    jobs_feat = normalizer.transform(jobs_df.set_index("job_id"))

    model.eval()
    all_predictions = []
    all_targets = []
    prediction_rows = []

    with torch.no_grad():
        directional_df = expand_pair_df(pair_df)

        for _, row in directional_df.iterrows():
            victim_id = row["job_id"]
            target_id = row["target_id"]
            true_slowdown = row["slowdown"]

            victim_idx = jobs_index[victim_id]
            victim_profile = torch.tensor(jobs_feat[victim_idx], dtype=torch.float32).unsqueeze(0).to(device)

            target_idx = jobs_index[target_id]
            target_profile = torch.tensor(jobs_feat[target_idx], dtype=torch.float32).unsqueeze(0).to(device)

            anchor_rows = job_inh_df[job_inh_df["job_id"] == victim_id]

            anchor_log_preds = []
            for _, anchor_row in anchor_rows.iterrows():
                anchor_slowdown = anchor_row["slowdown"]
                inhib_idx = jobs_df[jobs_df["job_id"] == anchor_row["inhib_id"]]
                if len(inhib_idx) > 0:
                    inhib_profile = torch.tensor(jobs_feat[jobs_df[jobs_df["job_id"] == anchor_row["inhib_id"]].index[0]], dtype=torch.float32).unsqueeze(0).to(device)
                    r_anchor = model.response_score(victim_profile, inhib_profile)
                    r_target = model.response_score(victim_profile, target_profile)
                    delta = r_target - r_anchor
                    pred_log = np.log(anchor_slowdown) + delta.item()
                    anchor_log_preds.append(pred_log)

            if len(anchor_log_preds) > 0:
                anchor_log_preds = torch.tensor(np.array(anchor_log_preds), dtype=torch.float32)
                pred_log = aggregate_unweighted(anchor_log_preds)
                pred_slowdown = np.exp(pred_log.item())
            else:
                pred_slowdown = true_slowdown

            all_predictions.append(pred_slowdown)
            all_targets.append(true_slowdown)

            prediction_rows.append({
                "victim_id": victim_id,
                "target_id": target_id,
                "true_slowdown": true_slowdown,
                "predicted_slowdown": pred_slowdown,
                "abs_error": abs(pred_slowdown - true_slowdown),
                "anchor_std": 0.0,
            })

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    metrics = compute_metrics(predictions, targets)
    predictions_df = pd.DataFrame(prediction_rows)

    return predictions_df, metrics


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate Delta Response Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--jobs_csv", type=str, required=True)
    parser.add_argument("--inhibitors_csv", type=str, required=True)
    parser.add_argument("--job_inh_csv", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--temperature", type=float, default=1.0)

    if args is None:
        args = parser.parse_args()

    data = load_and_prepare_data(args.jobs_csv, args.inhibitors_csv, args.job_inh_csv, args.pair_csv)
    model, model_config = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running transitive evaluation with similarity-weighted aggregation...")
    pred_df, metrics = evaluate_transitive(model, data, device, temperature=args.temperature)

    print(f"\nTransitive Evaluation Results (temperature={args.temperature}):")
    print(f"  Log MAE: {metrics['log_mae']:.4f}")
    print(f"  Log RMSE: {metrics['log_rmse']:.4f}")
    print(f"  Raw MAE: {metrics['raw_mae']:.4f}")
    print(f"  Raw RMSE: {metrics['raw_rmse']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  Anchor Std Mean: {metrics['anchor_std_mean']:.4f}")
    print(f"  Anchor Std Std: {metrics['anchor_std_std']:.4f}")

    checkpoint_name = Path(args.checkpoint).stem
    pred_df_path = output_dir / f"{checkpoint_name}_predictions.csv"
    pred_df.to_csv(pred_df_path, index=False)
    print(f"\nPredictions saved to {pred_df_path}")

    metrics_path = output_dir / f"{checkpoint_name}_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump({
            "log_mae": metrics['log_mae'],
            "log_rmse": metrics['log_rmse'],
            "raw_mae": metrics['raw_mae'],
            "raw_rmse": metrics['raw_rmse'],
            "spearman": metrics['spearman'],
            "anchor_std_mean": metrics['anchor_std_mean'],
            "anchor_std_std": metrics['anchor_std_std'],
            "temperature": args.temperature,
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
