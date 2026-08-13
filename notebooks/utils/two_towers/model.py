import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden=64, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwoTowerSlowdownModel(nn.Module):
    def __init__(self, in_dim=4, hidden=64, emb_dim=32, role_dim=16, interaction_hidden=32):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, hidden, emb_dim)
        self.victim_head = nn.Linear(emb_dim, role_dim)
        self.aggressor_head = nn.Linear(emb_dim, role_dim)
        self.interaction = nn.Sequential(
            nn.Linear(role_dim * 3, interaction_hidden),
            nn.ReLU(),
            nn.Linear(interaction_hidden, 1),
        )

    def forward(self, victim_profile, aggressor_profile):
        z_victim_full = self.encoder(victim_profile)
        z_aggr_full = self.encoder(aggressor_profile)
        z_v = self.victim_head(z_victim_full)
        z_a = self.aggressor_head(z_aggr_full)
        interaction_input = torch.cat([z_v, z_a, z_v * z_a], dim=-1)
        log_slowdown_pred = self.interaction(interaction_input).squeeze(-1)
        return log_slowdown_pred
