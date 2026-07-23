#!/usr/bin/env python3
"""
PyTorch Dataset Definition for Slowdown Prediction
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SlowdownDataset(Dataset):
    def __init__(self, data_path, train_apps, mode='train', val_split=0.0, test_split=0.0):
        """
        Args:
            data_path: Path to processed_data.npz
            train_apps: List of job names to use for training
            mode: 'train', 'val', or 'test' - determines which pairs to include
            val_split: Fraction of training data to use for validation (0.0 to disable)
            test_split: Fraction of data to use for test (held-out pairs)
        """
        self.mode = mode
        self.val_split = val_split
        self.test_split = test_split
        
        # Load data
        data = np.load(data_path, allow_pickle=True)
        self.job_ids = data['job_ids'].tolist()
        self.set_features = torch.from_numpy(data['set_features']).float()
        self.isolated_profiles = torch.from_numpy(data['isolated_profiles']).float()
        self.pairs = torch.from_numpy(data['pairs']).int()
        
        # Build mapping from job_id to index
        self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(self.job_ids)}
        
        # Determine which job indices belong to train vs test
        self.train_indices = set(self.job_id_to_idx[job] for job in train_apps)
        self.test_indices = set(self.job_id_to_idx[job] for job in self.job_ids if job not in train_apps)
        
        # Filter pairs based on mode
        if mode == 'train':
            # Keep rows where both jobs are in training indices
            mask = torch.tensor([
                idx_a in self.train_indices and idx_b in self.train_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
        elif mode == 'val':
            # Keep rows where both jobs are in training indices (for validation)
            mask = torch.tensor([
                idx_a in self.train_indices and idx_b in self.train_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
        else:  # mode == 'test'
            # Keep rows where both jobs are in test indices
            mask = torch.tensor([
                idx_a in self.test_indices and idx_b in self.test_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
        
        self.filtered_pairs = self.pairs[mask]
        
        # Build samples list: each row yields two samples (both directions)
        self.samples = []
        for idx_a, idx_b, slowdown_a, slowdown_b in self.filtered_pairs.tolist():
            # Add sample for A predicting slowdown of A when paired with B
            self.samples.append((idx_a, idx_b, float(slowdown_a)))
            # Add sample for B predicting slowdown of B when paired with A
            self.samples.append((idx_b, idx_a, float(slowdown_b)))
        
        # Split data into train/val/test if requested
        import random
        random.seed(42)
        
        if mode == 'train' and val_split > 0:
            random.shuffle(self.samples)
            split_idx = int(len(self.samples) * (1 - val_split))
            self.samples = self.samples[:split_idx]
        
        # Compute normalization statistics
        all_y = [y for _, _, y in self.samples]
        self.y_mean = torch.tensor(all_y, dtype=torch.float32).mean().item()
        self.y_std = torch.tensor(all_y, dtype=torch.float32).std().item()
        if self.y_std == 0:
            self.y_std = 1.0
        
        # Normalize targets
        self.samples = [(idx_a, idx_b, (y - self.y_mean) / self.y_std) 
                        for idx_a, idx_b, y in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        idx_a, idx_b, y = self.samples[index]
        
        set_a = self.set_features[idx_a]  # (N, 8)
        set_b = self.set_features[idx_b]  # (N, 8)
        b_a = self.isolated_profiles[idx_a]  # (4,)
        b_b = self.isolated_profiles[idx_b]  # (4,)
        
        return {
            'set_A': set_a,
            'set_B': set_b,
            'b_A': b_a,
            'b_B': b_b,
            'y': torch.tensor(y, dtype=torch.float32)
        }
