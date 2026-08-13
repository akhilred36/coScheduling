#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_csvs, ProfileNormalizer, VictimAggressorDataset, expand_pair_df, FEATURE_COLS
from model import TwoTowerSlowdownModel
from train import train_final_model, loao_cross_validation
from utils import set_seed, compute_metrics
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import HuberLoss
from torch.optim import Adam


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for victim, aggressor, y in loader:
        victim, aggressor, y = victim.to(device), aggressor.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(victim, aggressor)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for victim, aggressor, y in loader:
            victim, aggressor, y = victim.to(device), aggressor.to(device), y.to(device)
            pred = model(victim, aggressor)
            loss = criterion(pred, y)
            total_loss += loss.item()
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(all_pred), np.array(all_true)


def train_with_early_stop(model, train_loader, val_loader, criterion, optimizer, 
                          device, patience=20, max_epochs=300):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Two-Tower Baseline for MPI Co-Scheduling Slowdown Prediction")
    parser.add_argument("--jobs_csv", type=str, required=True, help="Path to jobs.csv")
    parser.add_argument("--inhibitors_csv", type=str, required=True, help="Path to inhibitors.csv")
    parser.add_argument("--job_inh_csv", type=str, required=True, help="Path to job_inh.csv")
    parser.add_argument("--pair_csv", type=str, required=True, help="Path to pair.csv")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--role_dim", type=int, default=16, help="Role projection dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--max_epochs", type=int, default=300, help="Max epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds for CV")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Two-Tower Baseline: MPI Co-Scheduling Slowdown Prediction")
    print("=" * 60)
    
    print("\n[1/4] Loading data...")
    jobs_df, inhibitors_df, job_inh_df, pair_df = load_csvs(
        args.jobs_csv, args.inhibitors_csv, args.job_inh_csv, args.pair_csv
    )
    
    print(f"  - Jobs (apps): {len(jobs_df)}")
    print(f"  - Inhibitors: {len(inhibitors_df)}")
    print(f"  - Training pairs (job × inhibitor): {len(job_inh_df)}")
    print(f"  - Evaluation pairs (jobA × jobB): {len(pair_df)}")
    
    print("\n[2/4] Running LOAO cross-validation...")
    app_ids = jobs_df["job_id"].unique()
    print(f"  - Apps for LOAO: {len(app_ids)}")
    
    all_cv_metrics = {"mae_log": [], "rmse_log": [], "mae_raw": [], "rmse_raw": [], "spearman_corr": []}
    
    for seed in args.seeds:
        print(f"\n  Seed {seed}:")
        set_seed(seed)
        
        for held_out_app in app_ids:
            train_mask = job_inh_df["job_id"] != held_out_app
            val_mask = job_inh_df["job_id"] == held_out_app
            
            train_pairs = job_inh_df[train_mask].reset_index(drop=True)
            val_pairs = job_inh_df[val_mask].reset_index(drop=True)
            
            train_apps = jobs_df[~jobs_df["job_id"].isin([held_out_app])].reset_index(drop=True)
            all_apps_for_norm = pd.concat([train_apps, jobs_df[jobs_df["job_id"] == held_out_app]], axis=0)
            
            normalizer = ProfileNormalizer()
            normalizer.fit(all_apps_for_norm)
            
            train_dataset = VictimAggressorDataset(train_pairs, jobs_df, inhibitors_df, normalizer)
            val_dataset = VictimAggressorDataset(val_pairs, jobs_df, inhibitors_df, normalizer)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TwoTowerSlowdownModel(
                in_dim=len(FEATURE_COLS), 
                hidden=args.hidden, 
                emb_dim=args.emb_dim, 
                role_dim=args.role_dim
            ).to(device)
            
            criterion = HuberLoss(delta=1.0)
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            
            model = train_with_early_stop(
                model, train_loader, val_loader, criterion, optimizer,
                device, patience=args.patience, max_epochs=args.max_epochs
            )
            
            _, val_pred_log, val_true_log = evaluate_model(model, val_loader, criterion, device)
            _, val_pred_raw, val_true_raw = evaluate_model(model, val_loader, criterion, device)
            
            metrics = compute_metrics(val_pred_log, val_true_log, val_pred_raw, np.exp(val_true_log))
            
            for k in all_cv_metrics:
                all_cv_metrics[k].append(metrics[k])
            
            print(f"    {held_out_app}: MAE_log={metrics['mae_log']:.4f}, RMSE_log={metrics['rmse_log']:.4f}, Spearman={metrics['spearman_corr']:.4f}")
    
    print("\n  LOAO CV AVERAGE:")
    for k in all_cv_metrics:
        if all_cv_metrics[k]:
            print(f"    {k}: {np.mean(all_cv_metrics[k]):.4f} (+/- {np.std(all_cv_metrics[k]):.4f})")
    
    print("\n[3/4] Training final model on all job_inh data...")
    set_seed(42)
    
    final_normalizer = ProfileNormalizer()
    final_normalizer.fit(pd.concat([jobs_df, inhibitors_df]))
    
    final_dataset = VictimAggressorDataset(job_inh_df, jobs_df, inhibitors_df, final_normalizer)
    
    indices = np.random.permutation(len(final_dataset))
    split_idx = int(0.9 * len(final_dataset))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_subset = torch.utils.data.Subset(final_dataset, train_indices)
    val_subset = torch.utils.data.Subset(final_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    final_model = TwoTowerSlowdownModel(
        in_dim=len(FEATURE_COLS),
        hidden=args.hidden,
        emb_dim=args.emb_dim,
        role_dim=args.role_dim
    ).to(device)
    
    final_criterion = HuberLoss(delta=1.0)
    final_optimizer = Adam(final_model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    final_model = train_with_early_stop(
        final_model, train_loader, val_loader, final_criterion, final_optimizer,
        device, patience=args.patience, max_epochs=args.max_epochs
    )
    
    print("  Final model trained and best checkpoint restored.")
    
    print("\n[4/4] Evaluating on pair_df holdout...")
    directional = expand_pair_df(pair_df)
    
    job_list = directional["job_id"].tolist() + directional["aggressor_job_id"].tolist()
    all_job_feats = final_normalizer.transform(jobs_df.set_index("job_id").loc[list(set(job_list))])
    
    job_to_feats = dict(zip(jobs_df["job_id"].str.strip().tolist(), all_job_feats))
    
    victim_list = directional["job_id"].tolist()
    aggressor_list = directional["aggressor_job_id"].tolist()
    
    victim_feats = torch.tensor(
        np.array([job_to_feats[jid.strip()] for jid in victim_list]), 
        dtype=torch.float32
    ).to(device)
    aggressor_feats = torch.tensor(
        np.array([job_to_feats[jid.strip()] for jid in aggressor_list]), 
        dtype=torch.float32
    ).to(device)
    
    final_model.eval()
    with torch.no_grad():
        log_slowdown_pred = final_model(victim_feats, aggressor_feats).cpu().numpy()
    
    true_slowdown = directional["slowdown"].values.astype(np.float32)
    pred_slowdown = np.exp(log_slowdown_pred)
    true_log = np.log(true_slowdown)
    
    mae_log = np.mean(np.abs(log_slowdown_pred - true_log))
    rmse_log = np.sqrt(np.mean((log_slowdown_pred - true_log) ** 2))
    mae_raw = np.mean(np.abs(pred_slowdown - true_slowdown))
    rmse_raw = np.sqrt(np.mean((pred_slowdown - true_slowdown) ** 2))
    spearman_corr, _ = spearmanr(pred_slowdown, true_slowdown)
    
    metrics = {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "mae_raw": mae_raw,
        "rmse_raw": rmse_raw,
        "spearman_corr": spearman_corr
    }
    
    results_df = directional.copy()
    results_df["log_pred"] = log_slowdown_pred
    results_df["pred_slowdown"] = pred_slowdown
    results_df["abs_error"] = np.abs(pred_slowdown - true_slowdown)
    
    print("\n  FINAL METRICS ON pair_df HOLDOUT:")
    print(f"    MAE (log space):  {mae_log:.4f}")
    print(f"    RMSE (log space): {rmse_log:.4f}")
    print(f"    MAE (raw space):  {mae_raw:.4f}")
    print(f"    RMSE (raw space): {rmse_raw:.4f}")
    print(f"    Spearman Corr:    {spearman_corr:.4f}")
    
    print("\n  Individual predictions (first 20 rows):")
    print(results_df[["job_id", "aggressor_job_id", "slowdown", "pred_slowdown", "abs_error"]].head(20).to_string(index=False))
    
    save_path = args.pair_csv.replace(".csv", "_predictions.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n  Saved full predictions to: {save_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
