#!/usr/bin/env python3
"""
Evaluation Script for Slowdown Predictor
"""

import argparse
import torch
from torch.utils.data import DataLoader

from dataset import SlowdownDataset
from model import SlowdownPredictor

# Training apps (for model training)
TRAIN_APPS = ['amg', 'beatnik', 'fiesta', 'laghos', 
              'lammps', 'minife', 'minivite']

DATA_PATH = "data/processed_data.npz"
MODEL_PATH = "best_model.pt"
VAL_SPLIT = 0.2
EVAL_METHOD = 'zero_shot'


def evaluate(model, dataloader, criterion, device, y_mean=None, y_std=None, dataset=None):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    all_set_A = []
    all_set_B = []
    all_b_A = []
    all_b_B = []
    all_b_A_raw = []
    all_b_B_raw = []
    all_app_A = []
    all_app_B = []
    
    with torch.no_grad():
        for batch in dataloader:
            set_A = batch['set_A'].to(device)
            set_B = batch['set_B'].to(device)
            b_A = batch['b_A'].to(device)
            b_B = batch['b_B'].to(device)
            y = batch['y'].to(device)
            
            pred = model(set_A, set_B, b_A, b_B)
            
            # If normalization parameters provided, inverse transform
            if y_mean is not None and y_std is not None:
                pred = pred * y_std + y_mean
                y = y * y_std + y_mean
            
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            all_set_A.extend(set_A.cpu().numpy().tolist())
            all_set_B.extend(set_B.cpu().numpy().tolist())
            all_b_A.extend(b_A.cpu().numpy().tolist())
            all_b_B.extend(b_B.cpu().numpy().tolist())
            all_b_A_raw.extend(batch['b_A_raw'].cpu().numpy().tolist())
            all_b_B_raw.extend(batch['b_B_raw'].cpu().numpy().tolist())
            all_app_A.extend(batch['app_A'])
            all_app_B.extend(batch['app_B'])
        
        mse = criterion(torch.tensor(all_preds, device=device), torch.tensor(all_targets, device=device))
        mae = torch.mean(torch.abs(torch.tensor(all_preds, device=device) - torch.tensor(all_targets, device=device)))
        
        return mse.item(), mae.item(), all_preds, all_targets, all_set_A, all_set_B, all_b_A, all_b_B, all_b_A_raw, all_b_B_raw, all_app_A, all_app_B


def main():
    parser = argparse.ArgumentParser(description='Evaluation script for slowdown predictor')
    parser.add_argument('--eval_method', type=str, default=EVAL_METHOD,
                        choices=['random_split', 'zero_shot', 'one_known'],
                        help='Evaluation method to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save train.csv and test.csv files')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("Loading evaluation datasets...")
    
    # Training dataset (held-out pairs from training apps)
    train_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='train', val_split=VAL_SPLIT,
                                    eval_method=EVAL_METHOD, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    print(f"Train samples: {len(train_dataset)}")
    
    # Test dataset
    test_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='test', val_split=0.0,
                                   eval_method=EVAL_METHOD, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    print(f"Normalization: mean={train_dataset.y_mean:.4f}, std={train_dataset.y_std:.4f}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SlowdownPredictor().to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    (test_mse, test_mae, test_preds, test_true, test_set_A, test_set_B, test_b_A, test_b_B,
     test_b_A_raw, test_b_B_raw, test_app_A, test_app_B) = evaluate(
        model, test_loader, criterion, device, test_dataset.y_mean, test_dataset.y_std, test_dataset)
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    (train_mse, train_mae, train_preds, train_true, train_set_A, train_set_B, train_b_A, train_b_B,
     train_b_A_raw, train_b_B_raw, train_app_A, train_app_B) = evaluate(
        model, train_loader, criterion, device, train_dataset.y_mean, train_dataset.y_std, train_dataset)
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    
    # Save CSV files if requested
    if args.save_csv:
        import csv
        import os
        
        os.makedirs('output', exist_ok=True)
        
        # For CSV output, train dataset uses train_pairs_only to get actual training data
        csv_train_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='train', val_split=VAL_SPLIT,
                                            eval_method=args.eval_method, seed=args.seed,
                                            train_pairs_only=True)
        csv_train_loader = DataLoader(csv_train_dataset, batch_size=len(csv_train_dataset), shuffle=False)
        (_, _, csv_train_preds, csv_train_true, csv_train_set_A, csv_train_set_B,
         csv_train_b_A, csv_train_b_B, csv_train_b_A_raw, csv_train_b_B_raw,
         csv_train_app_A, csv_train_app_B) = evaluate(
            model, csv_train_loader, criterion, device,
            csv_train_dataset.y_mean, csv_train_dataset.y_std, csv_train_dataset)
        
        # Test dataset uses the specified eval method
        csv_test_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='test', val_split=0.0,
                                           eval_method=args.eval_method, seed=args.seed)
        csv_test_loader = DataLoader(csv_test_dataset, batch_size=len(csv_test_dataset), shuffle=False)
        (_, _, csv_test_preds, csv_test_true, csv_test_set_A, csv_test_set_B,
         csv_test_b_A, csv_test_b_B, csv_test_b_A_raw, csv_test_b_B_raw,
         csv_test_app_A, csv_test_app_B) = evaluate(
            model, csv_test_loader, criterion, device,
            csv_test_dataset.y_mean, csv_test_dataset.y_std, csv_test_dataset)
        
        # Write train.csv
        with open('output/train.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['app_A', 'app_B', 'set_A', 'set_B', 'b_A', 'b_B', 'b_A_raw', 'b_B_raw', 'y_true', 'y_pred'])
            for i in range(len(csv_train_true)):
                writer.writerow([csv_train_app_A[i], csv_train_app_B[i],
                               csv_train_set_A[i], csv_train_set_B[i], csv_train_b_A[i], csv_train_b_B[i],
                               csv_train_b_A_raw[i], csv_train_b_B_raw[i],
                               csv_train_true[i], csv_train_preds[i]])
        print(f"Saved train.csv with {len(csv_train_true)} samples")
        
        # Write test.csv
        with open('output/test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['app_A', 'app_B', 'set_A', 'set_B', 'b_A', 'b_B', 'b_A_raw', 'b_B_raw', 'y_true', 'y_pred'])
            for i in range(len(csv_test_true)):
                writer.writerow([csv_test_app_A[i], csv_test_app_B[i],
                               csv_test_set_A[i], csv_test_set_B[i], csv_test_b_A[i], csv_test_b_B[i],
                               csv_test_b_A_raw[i], csv_test_b_B_raw[i],
                               csv_test_true[i], csv_test_preds[i]])
        print(f"Saved test.csv with {len(csv_test_true)} samples")
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
