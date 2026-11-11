"""Response-potential and absolute-response models."""

from __future__ import annotations

import torch
from torch import nn


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8, embedding_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        return self.network(profiles)


class LowRankPotential(nn.Module):
    """R(A, X) = h(E(X)) + <u(E(A)), v(E(X))>."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        embedding_dim: int = 4,
        rank: int = 2,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, embedding_dim)
        self.global_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.victim_head = nn.Linear(embedding_dim, rank)
        self.aggressor_head = nn.Linear(embedding_dim, rank)

    def response(self, victim: torch.Tensor, aggressor: torch.Tensor) -> torch.Tensor:
        victim_embedding = self.encoder(victim)
        aggressor_embedding = self.encoder(aggressor)
        global_pressure = self.global_head(aggressor_embedding).squeeze(-1)
        interaction = (
            self.victim_head(victim_embedding)
            * self.aggressor_head(aggressor_embedding)
        ).sum(dim=-1)
        return global_pressure + interaction

    def forward(
        self, victim: torch.Tensor, anchor: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        return self.response(victim, query) - self.response(victim, anchor)


class GenericPotential(nn.Module):
    """Higher-capacity response-potential ablation with the same delta structure."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 16,
        embedding_dim: int = 8,
        rank: int = 2,
    ):
        super().__init__()
        del rank
        self.encoder = SharedEncoder(input_dim, hidden_dim, embedding_dim)
        self.response_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def response(self, victim: torch.Tensor, aggressor: torch.Tensor) -> torch.Tensor:
        victim_embedding = self.encoder(victim)
        aggressor_embedding = self.encoder(aggressor)
        features = torch.cat(
            [victim_embedding, aggressor_embedding, victim_embedding * aggressor_embedding],
            dim=-1,
        )
        return self.response_head(features).squeeze(-1)

    def forward(
        self, victim: torch.Tensor, anchor: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        return self.response(victim, query) - self.response(victim, anchor)


class AbsoluteResponseModel(LowRankPotential):
    """Architecture-matched baseline trained to predict absolute log slowdown."""

    def forward(self, victim: torch.Tensor, aggressor: torch.Tensor) -> torch.Tensor:
        return self.response(victim, aggressor)
