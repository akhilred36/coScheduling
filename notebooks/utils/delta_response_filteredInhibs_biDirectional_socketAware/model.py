"""Legacy response-potential/absolute models and the joint bidirectional model."""

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


class SupportSetEncoder(nn.Module):
    """Permutation-invariant masked-mean encoder over fingerprint tokens."""

    def __init__(self, token_dim: int, hidden_dim: int = 8, context_dim: int = 4):
        super().__init__()
        self.context_dim = context_dim
        self.token_network = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = mask.to(tokens.dtype).sum(dim=1)
        if tokens.shape[1] == 0:
            return tokens.new_zeros(tokens.shape[0], self.context_dim), count
        embedded = self.token_network(tokens) * mask.unsqueeze(-1).to(tokens.dtype)
        pooled = embedded.sum(dim=1) / count.clamp_min(1.0).unsqueeze(-1)
        return pooled, count


class JointResponseModel(nn.Module):
    """R(V, X) = b(e_V) + h(e_X) + <u(e_V), v(e_X)>, with an optional
    directional bandwidth head and optional measured-support fingerprints.

    Deltas exist only as differences of responses, so identity, antisymmetry,
    and path consistency hold exactly for fixed endpoint contexts.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        embedding_dim: int = 4,
        rank: int = 2,
        aux_tasks: tuple[str, ...] | list[str] = (),
        aux_hidden_dim: int = 8,
        token_dim: int | None = None,
        context_hidden_dim: int = 8,
        context_dim: int = 4,
    ):
        super().__init__()
        self.aux_tasks = tuple(aux_tasks)
        self.encoder = SharedEncoder(input_dim, hidden_dim, embedding_dim)
        self.victim_level = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.global_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.victim_head = nn.Linear(embedding_dim, rank)
        self.aggressor_head = nn.Linear(embedding_dim, rank)
        self.context_encoder = None
        self.context_fusion = None
        if token_dim is not None:
            self.context_encoder = SupportSetEncoder(token_dim, context_hidden_dim, context_dim)
            self.context_fusion = nn.Sequential(
                nn.Linear(context_dim + 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim)
            )
        self.bandwidth_head = None
        if self.aux_tasks:
            self.bandwidth_head = nn.Sequential(
                nn.Linear(2 * embedding_dim + 3 * rank, aux_hidden_dim),
                nn.ReLU(),
                nn.Linear(aux_hidden_dim, len(self.aux_tasks)),
            )

    @property
    def uses_context(self) -> bool:
        return self.context_encoder is not None

    def embed(
        self,
        profiles: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder(profiles)
        if self.context_encoder is None or context is None:
            return embedding
        tokens, mask = context
        pooled, count = self.context_encoder(tokens, mask)
        present = (count > 0).to(embedding.dtype).unsqueeze(-1)
        fused = self.context_fusion(
            torch.cat([pooled, present, torch.log1p(count).unsqueeze(-1)], dim=-1)
        )
        # Exactly zero additive contribution for an empty support set.
        return embedding + present * fused

    def response_from_embeddings(
        self, victim_embedding: torch.Tensor, aggressor_embedding: torch.Tensor
    ) -> torch.Tensor:
        level = self.victim_level(victim_embedding).squeeze(-1)
        pressure = self.global_head(aggressor_embedding).squeeze(-1)
        interaction = (
            self.victim_head(victim_embedding) * self.aggressor_head(aggressor_embedding)
        ).sum(dim=-1)
        return level + pressure + interaction

    def bandwidth_from_embeddings(
        self, victim_embedding: torch.Tensor, aggressor_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.bandwidth_head is None:
            raise RuntimeError("This model has no auxiliary bandwidth head")
        u = self.victim_head(victim_embedding)
        v = self.aggressor_head(aggressor_embedding)
        features = torch.cat([victim_embedding, aggressor_embedding, u, v, u * v], dim=-1)
        return self.bandwidth_head(features)

    def response(
        self,
        victim: torch.Tensor,
        aggressor: torch.Tensor,
        victim_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        aggressor_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.response_from_embeddings(
            self.embed(victim, victim_context), self.embed(aggressor, aggressor_context)
        )

    def bandwidth_response(
        self,
        victim: torch.Tensor,
        aggressor: torch.Tensor,
        victim_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        aggressor_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.bandwidth_from_embeddings(
            self.embed(victim, victim_context), self.embed(aggressor, aggressor_context)
        )

    def delta(
        self,
        victim: torch.Tensor,
        anchor: torch.Tensor,
        query: torch.Tensor,
        victim_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        anchor_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        query_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        victim_embedding = self.embed(victim, victim_context)
        return self.response_from_embeddings(
            victim_embedding, self.embed(query, query_context)
        ) - self.response_from_embeddings(victim_embedding, self.embed(anchor, anchor_context))

    def forward(
        self,
        victim: torch.Tensor,
        aggressor: torch.Tensor,
        victim_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        aggressor_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.response(victim, aggressor, victim_context, aggressor_context)
