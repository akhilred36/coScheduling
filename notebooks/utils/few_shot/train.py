#!/usr/bin/env python3
"""
Training Script for Slowdown Predictor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SlowdownDataset
from model import SlowdownPredictor

# Training configuration
TRAIN_APPS = ['amg', 'beatnik', 'fiesta', 'laghos', 
              'lammps', 'minife', 'minivite']

DATA_PATH = "data/processed_data.npz"
MODEL_PATH = "best_model.pt"
SEED = 42
EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 5e-3
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2


def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    
    print("Loading training dataset...")
    train_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='train', val_split=VAL_SPLIT)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    print(f"Training samples (with val split): {len(train_dataset)}")
    print(f"Normalization: mean={train_dataset.y_mean:.4f}, std={train_dataset.y_std:.4f}")
    
    # Create validation dataset with same normalization
    val_dataset = SlowdownDataset(DATA_PATH, train_apps=TRAIN_APPS, mode='val', test_split=0.0)
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


if __name__ == "__main__":
    main()
