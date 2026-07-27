#!/usr/bin/env python3
"""
PyTorch Dataset Definition for Slowdown Prediction
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SlowdownDataset(Dataset):
    def __init__(self, data_path, train_apps, mode='train', val_split=0.0, test_split=0.0,
                 eval_method='zero_shot', seed=42, known_apps=None, train_pairs_only=False, train_split=0.8):
        """
        Args:
            data_path: Path to processed_data.npz
            train_apps: List of job names to use for training
            mode: 'train', 'val', 'test', or 'eval' - determines which pairs to include
            val_split: Fraction of training data to use for validation (0.0 to disable)
            test_split: Fraction of data to use for test (held-out pairs)
            eval_method: 'random_split', 'zero_shot', or 'one_known'
            seed: Random seed for reproducibility
            known_apps: List of known apps for one_known evaluation (optional)
            train_pairs_only: If True, bypass eval_method filter and include only training-app pairs
            train_split: Train/test split fraction for random_split evaluation (default: 0.8)
        """
        self.mode = mode
        self.val_split = val_split
        self.test_split = test_split
        self.eval_method = eval_method
        self.seed = seed
        self.known_apps = known_apps
        self.train_pairs_only = train_pairs_only
        
      # Load data
        data = np.load(data_path, allow_pickle=True)
        self.job_ids = data['job_ids'].tolist()
        self.set_features = torch.from_numpy(data['set_features']).float()
        self.isolated_profiles = torch.from_numpy(data['isolated_profiles']).float()
        self.isolated_profiles_raw = torch.from_numpy(data['isolated_profiles_raw']).float()
        self.pairs = torch.from_numpy(data['pairs']).float()
        
        # Build mapping from job_id to index
        self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(self.job_ids)}
        
        # Determine which job indices belong to train vs test
        self.train_indices = set(self.job_id_to_idx[job] for job in train_apps)
        self.test_indices = set(self.job_id_to_idx[job] for job in self.job_ids if job not in train_apps)
        
        if self.known_apps is not None:
            self.known_indices = set(self.job_id_to_idx[job] for job in self.known_apps)
        else:
            self.known_indices = self.train_indices
        
        # Filter pairs based on mode and evaluation method
        # If train_pairs_only is True, always use training-app pairs regardless of eval_method
        if self.train_pairs_only:
            mask = torch.tensor([
                idx_a in self.train_indices and idx_b in self.train_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
            self.filtered_pairs = self.pairs[mask]
        elif self.eval_method == 'random_split':
            mask = torch.tensor([
                idx_a in self.train_indices and idx_b in self.train_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
            self.filtered_pairs = self.pairs[mask]
        elif self.eval_method == 'zero_shot':
            mask = torch.tensor([
                idx_a in self.test_indices and idx_b in self.test_indices
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
            self.filtered_pairs = self.pairs[mask]
        elif self.eval_method == 'one_known':
            mask = torch.tensor([
                (idx_a in self.train_indices and idx_b in self.known_indices) or
                (idx_a in self.known_indices and idx_b in self.train_indices) or
                (idx_a in self.test_indices and idx_b in self.known_indices) or
                (idx_a in self.known_indices and idx_b in self.test_indices) or
                (idx_a in self.known_indices and idx_b in self.known_indices)
                for idx_a, idx_b, _, _ in self.pairs.tolist()
            ])
            self.filtered_pairs = self.pairs[mask]
        else:
            raise ValueError(f"Unknown eval_method: {self.eval_method}")
        
        # Build samples list: each row yields two samples (both directions)
        self.samples = []
        for pair in self.filtered_pairs.tolist():
            idx_a = int(pair[0])
            idx_b = int(pair[1])
            slowdown_a = float(pair[2])
            slowdown_b = float(pair[3])
            # Add sample for A predicting slowdown of A when paired with B
            self.samples.append((idx_a, idx_b, slowdown_a))
            # Add sample for B predicting slowdown of B when paired with A
            self.samples.append((idx_b, idx_a, slowdown_b))
        
        # Split data into train/val/test if requested
        import random
        random.seed(self.seed)
        
        if mode == 'train' and val_split > 0:
            random.shuffle(self.samples)
            if self.eval_method == 'random_split' and train_split < 1.0:
                split_idx = int(len(self.samples) * train_split)
                self.samples = self.samples[:split_idx]
            else:
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
        b_a_raw = self.isolated_profiles_raw[idx_a]  # (4,)
        b_b_raw = self.isolated_profiles_raw[idx_b]  # (4,)
        app_a = self.job_ids[idx_a]
        app_b = self.job_ids[idx_b]
        
        return {
            'set_A': set_a,
            'set_B': set_b,
            'b_A': b_a,
            'b_B': b_b,
            'b_A_raw': b_a_raw,
            'b_B_raw': b_b_raw,
            'app_A': app_a,
            'app_B': app_b,
            'y': torch.tensor(y, dtype=torch.float32)
        }
