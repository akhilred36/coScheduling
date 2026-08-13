import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import HuberLoss
import numpy as np
from tqdm import tqdm

from data import load_csvs, ProfileNormalizer, VictimAggressorDataset, FEATURE_COLS
from model import TwoTowerSlowdownModel
from utils import set_seed, compute_metrics


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


def loao_cross_validation(jobs_df, inhibitors_df, job_inh_df, test_seeds=[42, 123, 456],
                          hidden=64, emb_dim=32, role_dim=16, lr=1e-3, batch_size=128):
    """Run Leave-One-App-Out cross-validation."""
    app_ids = jobs_df["job_id"].unique()
    all_folds_results = []
    
    for seed in test_seeds:
        set_seed(seed)
        fold_results = []
        
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
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TwoTowerSlowdownModel(in_dim=len(FEATURE_COLS), hidden=hidden, 
                                          emb_dim=emb_dim, role_dim=role_dim).to(device)
            
            criterion = HuberLoss(delta=1.0)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            model = train_with_early_stop(model, train_loader, val_loader, criterion, optimizer, 
                                          device, patience=20, max_epochs=300)
            
            _, val_pred_log, val_true_log = evaluate_model(model, val_loader, criterion, device)
            _, val_pred_raw, val_true_raw = evaluate_model(model, val_loader, criterion, device)
            
            metrics = compute_metrics(val_pred_log, val_true_log, val_pred_raw, np.exp(val_true_log))
            metrics["held_out_app"] = held_out_app
            metrics["seed"] = seed
            fold_results.append(metrics)
        
        all_folds_results.append(fold_results)
    
    return all_folds_results


def train_final_model(jobs_df, inhibitors_df, job_inh_df, seed=42, hidden=64, emb_dim=32, 
                      role_dim=16, lr=1e-3, batch_size=128, patience=20, max_epochs=300):
    """Train final model on all data."""
    set_seed(seed)
    
    normalizer = ProfileNormalizer()
    normalizer.fit(pd.concat([jobs_df, inhibitors_df]))
    
    dataset = VictimAggressorDataset(job_inh_df, jobs_df, inhibitors_df, normalizer)
    
    indices = np.random.permutation(len(dataset))
    split_idx = int(0.9 * len(dataset))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerSlowdownModel(in_dim=len(FEATURE_COLS), hidden=hidden, 
                                  emb_dim=emb_dim, role_dim=role_dim).to(device)
    
    criterion = HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model = train_with_early_stop(model, train_loader, val_loader, criterion, optimizer, 
                                  device, patience=patience, max_epochs=max_epochs)
    
    return model, normalizer
