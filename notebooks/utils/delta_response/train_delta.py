"""
Training script for ResponseScoreModel with LOAO CV and final training.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from data import load_and_prepare_data, expand_pair_df, ProfileNormalizer
from delta_data import DeltaPairDataset
from model import ResponseScoreModel


warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for victim, aggr_j, aggr_k, target_delta in loader:
            victim = victim.to(device)
            aggr_j = aggr_j.to(device)
            aggr_k = aggr_k.to(device)

            pred_delta = model(victim, aggr_j, aggr_k)

            all_preds.extend(pred_delta.cpu().numpy())
            all_targets.extend(target_delta.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    spearman_corr, _ = spearmanr(all_preds, all_targets)

    return {
        "delta_mae": mae,
        "delta_rmse": rmse,
        "delta_spearman": spearman_corr,
    }


def reconstruct_metrics(model, loader, job_inh_df, normalizer, jobs_df, inhib_df, jobs_index, inhib_index, device):
    model.eval()
    all_recon_preds = []
    all_true_log_slowdown_k = []

    by_job = {
        job_id: g.reset_index(drop=True)
        for job_id, g in job_inh_df.groupby("job_id")
    }

    with torch.no_grad():
        for victim, aggr_j, aggr_k, target_delta in loader:
            victim = victim.to(device)
            aggr_j = aggr_j.to(device)
            aggr_k = aggr_k.to(device)

            pred_delta = model(victim, aggr_j, aggr_k)

            log_slowdown_j = torch.log(torch.tensor(job_inh_df["slowdown"].values.astype(np.float32)))
            reconstructed = log_slowdown_j + pred_delta

            all_recon_preds.extend(reconstructed.cpu().numpy())
            all_true_log_slowdown_k.extend(np.log(job_inh_df["slowdown"].values.astype(np.float32)))

    all_recon_preds = np.array(all_recon_preds)
    all_true_log_slowdown_k = np.array(all_true_log_slowdown_k)

    mae = np.mean(np.abs(all_recon_preds - all_true_log_slowdown_k))
    rmse = np.sqrt(np.mean((all_recon_preds - all_true_log_slowdown_k) ** 2))

    return {"recon_mae": mae, "recon_rmse": rmse}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for victim, aggr_j, aggr_k, target_delta in loader:
        victim = victim.to(device)
        aggr_j = aggr_j.to(device)
        aggr_k = aggr_k.to(device)
        target_delta = target_delta.to(device)

        optimizer.zero_grad()
        pred_delta = model(victim, aggr_j, aggr_k)
        loss = criterion(pred_delta, target_delta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for victim, aggr_j, aggr_k, target_delta in loader:
            victim = victim.to(device)
            aggr_j = aggr_j.to(device)
            aggr_k = aggr_k.to(device)
            target_delta = target_delta.to(device)

            pred_delta = model(victim, aggr_j, aggr_k)
            loss = criterion(pred_delta, target_delta)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def run_loao_cv(data, model_config, num_seeds=3, lr=1e-3, weight_decay=1e-5, max_epochs=300, patience=20, batch_size=256, samples_per_epoch=20000):
    jobs_df = data["jobs_df"]
    inhib_df = data["inhibitors_df"]
    job_inh_df = data["job_inh_df"]
    normalizer = data["normalizer"]
    jobs_index = {jid: i for i, jid in enumerate(jobs_df["job_id"])}
    inhib_index = {iid: i for i, iid in enumerate(inhib_df["inhib_id"])}

    job_ids = jobs_df["job_id"].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_results = []

    for seed in range(num_seeds):
        seed_results = {}
        set_seed(seed)

        for app_id in job_ids:
            train_mask = job_inh_df["job_id"] != app_id
            val_mask = job_inh_df["job_id"] == app_id

            train_df = job_inh_df[train_mask].reset_index(drop=True)
            val_df = job_inh_df[val_mask].reset_index(drop=True)

            fold_normalizer = ProfileNormalizer()
            train_jobs = jobs_df[jobs_df["job_id"] != app_id]
            fold_normalizer.fit(pd.concat([train_jobs, inhib_df], ignore_index=True))

            train_dataset = DeltaPairDataset(
                train_df, train_jobs, inhib_df, fold_normalizer,
                samples_per_epoch=samples_per_epoch, seed=seed
            )
            val_dataset = DeltaPairDataset(
                val_df, jobs_df, inhib_df, fold_normalizer,
                samples_per_epoch=2000, seed=seed + 1000
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = ResponseScoreModel(**model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.HuberLoss(delta=1.0)

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0
            best_metrics = None

            for epoch in range(max_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss = validate(model, val_loader, criterion, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                    with torch.no_grad():
                        metrics = compute_metrics(model, val_loader, device)
                        best_metrics = metrics
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            fold_results = {
                "seed": seed,
                "app_id": app_id,
                "val_mae": best_metrics["delta_mae"] if best_metrics else None,
                "val_rmse": best_metrics["delta_rmse"] if best_metrics else None,
                "val_spearman": best_metrics["delta_spearman"] if best_metrics else None,
            }
            seed_results[app_id] = fold_results

        all_fold_results.append(seed_results)

    return all_fold_results


def train_final_model(data, model_config, seed, lr=1e-3, weight_decay=1e-5, max_epochs=300, patience=20, batch_size=256, val_split=0.1, samples_per_epoch=20000):
    jobs_df = data["jobs_df"]
    inhib_df = data["inhibitors_df"]
    job_inh_df = data["job_inh_df"]
    normalizer = data["normalizer"]
    jobs_index = {jid: i for i, jid in enumerate(jobs_df["job_id"])}
    inhib_index = {iid: i for i, iid in enumerate(inhib_df["inhib_id"])}

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices = np.random.RandomState(seed).permutation(len(job_inh_df))
    val_size = int(len(job_inh_df) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_df = job_inh_df.iloc[train_indices].reset_index(drop=True)
    val_df = job_inh_df.iloc[val_indices].reset_index(drop=True)

    train_dataset = DeltaPairDataset(
        train_df, jobs_df, inhib_df, normalizer,
        samples_per_epoch=samples_per_epoch, seed=seed
    )
    val_dataset = DeltaPairDataset(
        val_df, jobs_df, inhib_df, normalizer,
        samples_per_epoch=2000, seed=seed + 1000
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ResponseScoreModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_metrics = None

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
            with torch.no_grad():
                metrics = compute_metrics(model, val_loader, device)
                best_metrics = metrics
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def main(args=None):
    parser = argparse.ArgumentParser(description="Train Delta Response Model")
    parser.add_argument("--jobs_csv", type=str, required=True)
    parser.add_argument("--inhibitors_csv", type=str, required=True)
    parser.add_argument("--job_inh_csv", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--init_from_baseline", action="store_true")
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--role_dim", type=int, default=16)
    parser.add_argument("--score_hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)

    if args is None:
        args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_and_prepare_data(args.jobs_csv, args.inhibitors_csv, args.job_inh_csv, args.pair_csv)

    model_config = {
        "in_dim": 4,
        "hidden": args.hidden,
        "emb_dim": args.emb_dim,
        "role_dim": args.role_dim,
        "score_hidden": args.score_hidden,
    }

    print("Running LOAO CV...")
    loao_results = run_loao_cv(
        data, model_config, num_seeds=args.num_seeds, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch
    )

    print("\nTraining final model on all data...")
    final_model, final_metrics = train_final_model(
        data, model_config, seed=0, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch
    )

    mean_mae = np.mean([r["val_mae"] for seed_res in loao_results for r in seed_res.values() if r["val_mae"]])
    mean_rmse = np.mean([r["val_rmse"] for seed_res in loao_results for r in seed_res.values() if r["val_mae"]])
    mean_spearman = np.mean([r["val_spearman"] for seed_res in loao_results for r in seed_res.values() if r["val_mae"]])

    print(f"LOAO CV Results (mean over {args.num_seeds} seeds):")
    print(f"  Delta MAE: {mean_mae:.4f}")
    print(f"  Delta RMSE: {mean_rmse:.4f}")
    print(f"  Delta Spearman: {mean_spearman:.4f}")

    print("\nTraining final model on all data...")
    final_model, final_metrics = train_final_model(
        data, model_config, seed=0,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size
    )

    print(f"Final model training metrics:")
    print(f"  Delta MAE: {final_metrics['delta_mae']:.4f}")
    print(f"  Delta RMSE: {final_metrics['delta_rmse']:.4f}")
    print(f"  Delta Spearman: {final_metrics['delta_spearman']:.4f}")

    checkpoint_path = os.path.join(args.output_dir, "delta_model_seed0.pt")
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_config": model_config,
        "normalizer_mean": data["normalizer"].mean_,
        "normalizer_std": data["normalizer"].std_,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
