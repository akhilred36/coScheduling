#!/usr/bin/env python3
"""
Model Architecture: Deep Sets + Relation Network for Slowdown Prediction
"""

import torch
import torch.nn as nn

class SlowdownPredictor(nn.Module):
    def __init__(self, set_dim=8, embed_dim=16, relation_input_dim=40, relation_hidden_dim=8):
        """
        Args:
            set_dim: Input dimension for each element in the set (8)
            embed_dim: Dimension of the embedding (reduced to 16 for small dataset)
            relation_input_dim: Input dimension for relation module (32 = 16+16+4+4)
            relation_hidden_dim: Hidden dimension (reduced to 8)
        """
        super(SlowdownPredictor, self).__init__()
        
        self.dropout = nn.Dropout(0.4)
        
        # Set encoder phi: MLP mapping 8-dim vector to 16-dim embedding
        self.phi = nn.Sequential(
            nn.Linear(set_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Prototype mapper rho: MLP mapping mean embedding to prototype (16-dim)
        self.rho = nn.Linear(embed_dim, embed_dim)
        
        # Relation module: simplified MLP
        self.relation = nn.Sequential(
            nn.Linear(relation_input_dim, relation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(relation_hidden_dim, 1)
        )
        
        # Initialize weights with Xavier uniform
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, set_A, set_B, b_A, b_B):
        """
        Args:
            set_A: (batch, N, 8) - set features for job A
            set_B: (batch, N, 8) - set features for job B
            b_A: (batch, 4) - isolated profile for job A
            b_B: (batch, 4) - isolated profile for job B
        
        Returns:
            pred: (batch,) - predicted slowdown scalar for each pair
        """
        # Compute prototypes
        # Apply phi to each element: (batch, N, 8) -> (batch, N, 16)
        phi_A = self.phi(set_A)  # (batch, N, 16)
        phi_B = self.phi(set_B)  # (batch, N, 16)
        
        # Mean pooling over N: (batch, N, 16) -> (batch, 16)
        mean_A = phi_A.mean(dim=1)
        mean_B = phi_B.mean(dim=1)
        
        # Apply rho to get prototypes
        c_A = self.rho(mean_A)  # (batch, 16)
        c_B = self.rho(mean_B)  # (batch, 16)
        
        # Concatenate prototypes and isolated profiles
        # Total dim: 16 + 16 + 4 + 4 = 40
        cat = torch.cat([c_A, c_B, b_A, b_B], dim=-1)
        
        # Pass through relation module and squeeze last dimension
        pred = self.relation(cat).squeeze(-1)  # (batch,)
        
        # Apply dropout
        pred = self.dropout(pred)
        
        return pred
    
    def get_prototype(self, set_features):
        """
        Helper method to compute prototype for a single app.
        
        Args:
            set_features: (N, 8) - set features for one app
        
        Returns:
            prototype: (16,) - prototype vector
        """
        # Add batch dimension
        set_features = set_features.unsqueeze(0)  # (1, N, 8)
        
        # Apply phi and mean pooling
        phi_out = self.phi(set_features)  # (1, N, 16)
        mean_out = phi_out.mean(dim=1)  # (1, 16)
        
        # Apply rho
        prototype = self.rho(mean_out)  # (1, 16)
        
        return prototype.squeeze(0)  # (16,)
