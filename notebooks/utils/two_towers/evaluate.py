import torch
import pandas as pd
import numpy as np
import argparse
import os

from data import load_csvs, evaluate_on_pair_df, PROFILE_COLS
from model import TwoTowerSlowdownModel
from train import train_final_model, loao_cross_validation


def print_metrics_table(metrics_dict, title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"{title}")
    print(f"{'Metric':<30} {'Value':<20}")
    print(f"{'-' * 60}")
    for key, value in metrics_dict.items():
        if key != "held_out_app" and key != "seed":
            print(f"{key:<30} {value:<20.6f}")


def main():
    parser = argparse.ArgumentParser(description="Two-Tower Baseline for MPI Co-Scheduling")
    parser.add_argument("--jobs_csv", type=str, required=True)
    parser.add_argument("--inhibitors_csv", type=str, required=True)
    parser.add_argument("--job_inh_csv", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--role_dim", type=int, default=16, help="Role projection dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--max_epochs", type=int, default=300, help="Max epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds for CV")
    args = parser.parse_args()
    
    print("Loading data...")
    jobs_df, inhibitors_df, job_inh_df, pair_df = load_csvs(
        args.jobs_csv, args.inhibitors_csv, args.job_inh_csv, args.pair_csv
    )
    
    print(f"Jobs: {len(jobs_df)}, Inhibitors: {len(inhibitors_df)}, Train pairs: {len(job_inh_df)}, Eval pairs: {len(pair_df)}")
    
    print("\nRunning LOAO cross-validation...")
    loao_results = loao_cross_validation(
        jobs_df, inhibitors_df, job_inh_df,
        test_seeds=args.seeds,
        hidden=args.hidden, emb_dim=args.emb_dim, role_dim=args.role_dim,
        lr=args.lr, batch_size=args.batch_size
    )
    
    all_cv_metrics = {k: [] for k in ["mae_log", "rmse_log", "mae_raw", "rmse_raw", "spearman_corr"]}
    for fold_results in loao_results:
        for m in fold_results:
            for k in all_cv_metrics:
                all_cv_metrics[k].append(m[k])
    
    print("\n=== LOAO Cross-Validation Results ===")
    print_metrics_table({k: np.mean(v) for k, v in all_cv_metrics.items()}, "AVERAGE METRICS")
    print("\nPer-FOLD METRICS:")
    for seed_idx, fold_results in enumerate(loao_results):
        for m in fold_results:
            print(f"Seed {m['seed']}, App {m['held_out_app']}: MAE_log={m['mae_log']:.4f}, Spearman={m['spearman_corr']:.4f}")
    
    print("\nTraining final model on all data...")
    final_model, final_normalizer = train_final_model(
        jobs_df, inhibitors_df, job_inh_df,
        seed=42, hidden=args.hidden, emb_dim=args.emb_dim, role_dim=args.role_dim,
        lr=args.lr, batch_size=args.batch_size, patience=args.patience, max_epochs=args.max_epochs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)
    
    print("\nEvaluating on pair_df holdout...")
    results_df, metrics = evaluate_on_pair_df(final_model, jobs_df, pair_df, final_normalizer, device)
    
    print_metrics_table(metrics, "FINAL EVALUATION ON pair_df")
    
    print("\n=== Individual Predictions ===")
    print(results_df[["job_id", "aggressor_job_id", "slowdown", "pred_slowdown", "abs_error"]].to_string(index=False))
    
    save_dir = os.path.dirname(args.pair_csv) or "."
    results_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    
    print(f"\nSaved predictions to {os.path.join(save_dir, 'predictions.csv')}")
    
    return metrics


if __name__ == "__main__":
    main()
