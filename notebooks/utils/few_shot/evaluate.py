#!/usr/bin/env python3
"""
Evaluation Script for Slowdown Predictor
"""

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


def evaluate(model, dataloader, criterion, device, y_mean=None, y_std=None):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        batch = next(iter(dataloader))
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
        
        mse = criterion(pred, y)
        mae = torch.mean(torch.abs(pred - y))
        
        return mse.item(), mae.item()


def main():
    # Set random seed
    torch.manual_seed(42)
    
    print("Loading evaluation datasets...")
    
    # Training dataset (held-out pairs from training apps)
    train_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='train', val_split=VAL_SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    print(f"Train samples (80% of training app pairs): {len(train_dataset)}")
    
    # Test dataset (unseen apps: kripke, quicksilver, tricount)
    test_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='test', val_split=0.0)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    print(f"Test samples (unseen apps): {len(test_dataset)}")
    
    print(f"Normalization: mean={train_dataset.y_mean:.4f}, std={train_dataset.y_std:.4f}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SlowdownPredictor().to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Evaluate on test set (unseen apps - true OOD generalization)
    print("\nEvaluating on test set (unseen apps - kripke, quicksilver, tricount)...")
    test_mse, test_mae = evaluate(model, test_loader, criterion, device, train_dataset.y_mean, train_dataset.y_std)
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")
    
    # Evaluate on training set (held-out pairs)
    print("\nEvaluating on training set (held-out pairs from training apps)...")
    train_mse, train_mae = evaluate(model, train_loader, criterion, device, train_dataset.y_mean, train_dataset.y_std)
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
