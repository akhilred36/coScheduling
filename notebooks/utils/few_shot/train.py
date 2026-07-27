#!/usr/bin/env python3
"""
Training Script for Slowdown Predictor
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SlowdownDataset
from model import SlowdownPredictor

DATA_PATH = "data/processed_data.npz"
MODEL_PATH = "best_model.pt"
SEED = 42
EPOCHS = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 5e-3
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2
EVAL_METHOD = 'random_split'


def main():
    parser = argparse.ArgumentParser(description='Training script for slowdown predictor')
    parser.add_argument('--eval_method', type=str, default=EVAL_METHOD,
                        choices=['random_split', 'zero_shot', 'one_known'],
                        help='Evaluation method to use')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/test split fraction for random_split evaluation (default: 0.8)')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_csv', type=str, nargs='?', const='output', default=None,
                        help='Save train.csv and test.csv files to specified directory (default: output/)')
    parser.add_argument('--training_apps', type=str, default='amg,beatnik,fiesta,laghos,lammps,minife,minivite',
                        help='Comma-separated list of training applications (default: amg,beatnik,fiesta,laghos,lammps,minife,minivite)')
    args = parser.parse_args()
    
    if args.eval_method != 'random_split' and args.train_split != 0.8:
        print("Error: --train_split can only be used with --eval_method random_split")
        sys.exit(1)
    
    # Parse training apps from command line
    custom_train_apps = [app.strip() for app in args.training_apps.split(',')]
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    print("Loading training dataset...")
    if args.eval_method == 'random_split':
        train_val_split = 1 - args.train_split
    else:
        train_val_split = VAL_SPLIT
    train_dataset = SlowdownDataset(DATA_PATH, train_apps=custom_train_apps, mode='train', val_split=train_val_split,
                                    eval_method=args.eval_method, seed=args.seed, train_pairs_only=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    print(f"Training samples (with val split): {len(train_dataset)}")
    print(f"Normalization: mean={train_dataset.y_mean:.4f}, std={train_dataset.y_std:.4f}")
    
    # Create validation dataset with same normalization
    val_dataset = SlowdownDataset(DATA_PATH, train_apps=custom_train_apps, mode='val', test_split=0.0,
                                  eval_method=args.eval_method, seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SlowdownPredictor().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_train_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Train on all batches
        train_loss = 0.0
        for batch in train_loader:
            set_A = batch['set_A'].to(device)
            set_B = batch['set_B'].to(device)
            b_A = batch['b_A'].to(device)
            b_B = batch['b_B'].to(device)
            y = batch['y'].to(device)
            
            # Forward pass
            pred = model(set_A, set_B, b_A, b_B)
            
            # Compute loss
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_set_A = val_batch['set_A'].to(device)
            val_set_B = val_batch['set_B'].to(device)
            val_b_A = val_batch['b_A'].to(device)
            val_b_B = val_batch['b_B'].to(device)
            val_y = val_batch['y'].to(device)
            
            val_pred = model(val_set_A, val_set_B, val_b_A, val_b_B)
            val_loss = criterion(val_pred, val_y).item()
        
        lr_current = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr_current:.2e}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  New best model saved: Val Loss: {best_val_loss:.6f}, Train Loss: {best_train_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Learning rate decay
        if epoch % 1000 == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
  # Load best model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"\nTraining complete. Best model loaded from {MODEL_PATH}")
    print(f"Best Val Loss: {best_val_loss:.6f}, Train Loss: {best_train_loss:.6f}")
    
    # Save CSV files if requested
    if args.save_csv:
        import csv
        import os
        
        csv_dir = args.save_csv
        os.makedirs(csv_dir, exist_ok=True)
        
        # Generate predictions on training dataset
        model.eval()
        train_preds = []
        train_true = []
        train_set_A = []
        train_set_B = []
        train_b_A = []
        train_b_B = []
        train_b_A_raw = []
        train_b_B_raw = []
        train_app_A = []
        train_app_B = []
        
        with torch.no_grad():
            train_loader_eval = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            for batch in train_loader_eval:
                set_A = batch['set_A'].to(device)
                set_B = batch['set_B'].to(device)
                b_A = batch['b_A'].to(device)
                b_B = batch['b_B'].to(device)
                y = batch['y'].to(device)
                
                pred = model(set_A, set_B, b_A, b_B)
                
                # Denormalize
                pred_orig = pred * train_dataset.y_std + train_dataset.y_mean
                y_orig = y * train_dataset.y_std + train_dataset.y_mean
                
                train_preds.extend(pred_orig.cpu().numpy().tolist())
                train_true.extend(y_orig.cpu().numpy().tolist())
                train_set_A.extend(set_A.cpu().numpy().tolist())
                train_set_B.extend(set_B.cpu().numpy().tolist())
                train_b_A.extend(b_A.cpu().numpy().tolist())
                train_b_B.extend(b_B.cpu().numpy().tolist())
                train_b_A_raw.extend(batch['b_A_raw'].cpu().numpy().tolist())
                train_b_B_raw.extend(batch['b_B_raw'].cpu().numpy().tolist())
                train_app_A.extend(batch['app_A'])
                train_app_B.extend(batch['app_B'])
        
        # Write train.csv
        with open(os.path.join(csv_dir, 'train.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['app_A', 'app_B', 'set_A', 'set_B', 'b_A', 'b_B', 'b_A_raw', 'b_B_raw', 'y_true', 'y_pred'])
            for i in range(len(train_true)):
                writer.writerow([train_app_A[i], train_app_B[i],
                               train_set_A[i], train_set_B[i], train_b_A[i], train_b_B[i],
                               train_b_A_raw[i], train_b_B_raw[i],
                               train_true[i], train_preds[i]])
        
        print(f"Saved train.csv with {len(train_true)} samples")
        
# Generate predictions on test dataset
        test_dataset = SlowdownDataset(DATA_PATH, train_apps=custom_train_apps, mode='test', val_split=0.0,
                                         eval_method=args.eval_method, seed=args.seed, train_split=args.train_split)
        
        model.eval()
        test_preds = []
        test_true = []
        test_set_A = []
        test_set_B = []
        test_b_A = []
        test_b_B = []
        test_b_A_raw = []
        test_b_B_raw = []
        test_app_A = []
        test_app_B = []
        
        with torch.no_grad():
            test_loader_eval = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            for batch in test_loader_eval:
                set_A = batch['set_A'].to(device)
                set_B = batch['set_B'].to(device)
                b_A = batch['b_A'].to(device)
                b_B = batch['b_B'].to(device)
                y = batch['y'].to(device)
                
                pred = model(set_A, set_B, b_A, b_B)
                
                # Denormalize
                pred_orig = pred * test_dataset.y_std + test_dataset.y_mean
                y_orig = y * test_dataset.y_std + test_dataset.y_mean
                
                test_preds.extend(pred_orig.cpu().numpy().tolist())
                test_true.extend(y_orig.cpu().numpy().tolist())
                test_set_A.extend(set_A.cpu().numpy().tolist())
                test_set_B.extend(set_B.cpu().numpy().tolist())
                test_b_A.extend(b_A.cpu().numpy().tolist())
                test_b_B.extend(b_B.cpu().numpy().tolist())
                test_b_A_raw.extend(batch['b_A_raw'].cpu().numpy().tolist())
                test_b_B_raw.extend(batch['b_B_raw'].cpu().numpy().tolist())
                test_app_A.extend(batch['app_A'])
                test_app_B.extend(batch['app_B'])
        
        # Write test.csv
        with open(os.path.join(csv_dir, 'test.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['app_A', 'app_B', 'set_A', 'set_B', 'b_A', 'b_B', 'b_A_raw', 'b_B_raw', 'y_true', 'y_pred'])
            for i in range(len(test_true)):
                writer.writerow([test_app_A[i], test_app_B[i],
                               test_set_A[i], test_set_B[i], test_b_A[i], test_b_B[i],
                               test_b_A_raw[i], test_b_B_raw[i],
                               test_true[i], test_preds[i]])
        
        print(f"Saved test.csv with {len(test_true)} samples")


if __name__ == "__main__":
    main()
