"""Single-anchor episodic training."""

from __future__ import annotations

import copy
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import SOCKET_COLUMN, ProfileScaler, socket_pair_inputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def profile_maps(
    jobs: pd.DataFrame, inhibitors: pd.DataFrame, scaler: ProfileScaler
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    job_values = scaler.transform(jobs)
    inhibitor_values = scaler.transform(inhibitors)
    return (
        dict(zip(jobs["job_id"].astype(str), job_values)),
        dict(zip(inhibitors["inhib_id"].astype(str), inhibitor_values)),
    )


class EpisodeSampler:
    """Samples victims uniformly and two distinct observed response rows."""

    def __init__(
        self,
        responses: pd.DataFrame,
        job_profiles: dict[str, np.ndarray],
        inhibitor_profiles: dict[str, np.ndarray],
        seed: int,
    ):
        self.groups = {}
        for job_id, group in responses.groupby("job_id", sort=False):
            if len(group) < 2 or group["inhib_id"].nunique() < 2:
                continue
            inhibitor_ids = group["inhib_id"].astype(str).to_numpy()
            self.groups[str(job_id)] = (
                inhibitor_ids,
                np.stack([inhibitor_profiles[value] for value in inhibitor_ids]),
                group["log_slowdown"].to_numpy(dtype=float),
            )
        if not self.groups:
            raise ValueError("No victim has two distinct observed inhibitors")
        self.job_ids = np.asarray(list(self.groups), dtype=object)
        self.job_profiles = job_profiles
        self.inhibitor_profiles = inhibitor_profiles
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        victims: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        queries: list[np.ndarray] = []
        target_logs: list[tuple[float, float]] = []
        for job_id in self.rng.choice(self.job_ids, size=batch_size, replace=True):
            inhibitor_ids, profiles, log_slowdowns = self.groups[str(job_id)]
            while True:
                left, right = self.rng.choice(len(inhibitor_ids), size=2, replace=False)
                if inhibitor_ids[left] != inhibitor_ids[right]:
                    break
            victims.append(self.job_profiles[str(job_id)])
            anchors.append(profiles[left])
            queries.append(profiles[right])
            target_logs.append(
                (float(log_slowdowns[left]), float(log_slowdowns[right]))
            )
        return (
            torch.as_tensor(np.stack(victims), dtype=torch.float32),
            torch.as_tensor(np.stack(anchors), dtype=torch.float32),
            torch.as_tensor(np.stack(queries), dtype=torch.float32),
            torch.as_tensor(target_logs, dtype=torch.float32),
        )


class SocketEpisodeSampler:
    """Socket-aware episodes: a uniform victim, two distinct inhibitor configs,
    then one socket-resolved row per config, each scored at its own socket."""

    def __init__(
        self,
        responses: pd.DataFrame,
        job_profiles: dict[str, np.ndarray],
        inhibitor_profiles: dict[str, np.ndarray],
        seed: int,
    ):
        self.groups = {}
        for job_id, group in responses.groupby("job_id", sort=False):
            configs = [
                (
                    inhibitor_profiles[str(inhib_id)],
                    rows[SOCKET_COLUMN].to_numpy(dtype=int),
                    rows["log_slowdown"].to_numpy(dtype=float),
                )
                for inhib_id, rows in group.groupby("inhib_id", sort=False)
            ]
            if len(configs) >= 2:
                self.groups[str(job_id)] = configs
        if not self.groups:
            raise ValueError("No victim has two distinct observed inhibitors")
        self.job_ids = np.asarray(list(self.groups), dtype=object)
        self.job_profiles = job_profiles
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        profiles: list[np.ndarray] = []
        sockets: list[int] = []
        target_logs: list[float] = []
        victims: list[np.ndarray] = []
        for job_id in self.rng.choice(self.job_ids, size=batch_size, replace=True):
            configs = self.groups[str(job_id)]
            for index in self.rng.choice(len(configs), size=2, replace=False):
                profile, config_sockets, log_slowdowns = configs[index]
                row = int(self.rng.integers(len(config_sockets)))
                victims.append(self.job_profiles[str(job_id)])
                profiles.append(profile)
                sockets.append(int(config_sockets[row]))
                target_logs.append(float(log_slowdowns[row]))
        victim_values, aggressor_values = socket_pair_inputs(
            np.stack(victims), np.stack(profiles), sockets
        )
        as_tensor = lambda values: torch.as_tensor(values, dtype=torch.float32)
        return (
            as_tensor(victim_values[0::2]),
            as_tensor(aggressor_values[0::2]),
            as_tensor(victim_values[1::2]),
            as_tensor(aggressor_values[1::2]),
            as_tensor(np.reshape(target_logs, (-1, 2))),
        )


def socket_delta(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """R(V|s_q, Q|1-s_q) - R(V|s_a, A|1-s_a) for a SocketEpisodeSampler batch."""
    anchor_victim, anchor, query_victim, query, _ = batch
    return model.response(query_victim, query) - model.response(anchor_victim, anchor)


def censored_delta_huber_loss(
    predicted_delta: torch.Tensor, target_logs: torch.Tensor
) -> torch.Tensor:
    """Huber loss for exact and one-sided pairwise censored observations."""
    anchor = target_logs[:, 0]
    query = target_logs[:, 1]
    anchor_exact = anchor > 0
    query_exact = query > 0
    exact = anchor_exact & query_exact
    query_censored = anchor_exact & ~query_exact
    anchor_censored = ~anchor_exact & query_exact
    residual = torch.zeros_like(predicted_delta)
    residual = torch.where(exact, predicted_delta - (query - anchor), residual)
    residual = torch.where(
        query_censored, torch.relu(predicted_delta + anchor), residual
    )
    residual = torch.where(
        anchor_censored, torch.relu(query - predicted_delta), residual
    )
    return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)


def censored_absolute_huber_loss(
    latent_prediction: torch.Tensor,
    observed_log: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact Huber loss above the floor and a one-sided loss at the floor.

    Optional per-row weights give a normalized weighted mean (socket-aware mode).
    """
    residual = torch.where(
        observed_log > 0,
        latent_prediction - observed_log,
        torch.relu(latent_prediction),
    )
    if weights is None:
        return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)
    losses = F.huber_loss(
        residual, torch.zeros_like(residual), delta=1.0, reduction="none"
    )
    return (weights * losses).sum() / weights.sum()


def train_potential(
    model: nn.Module,
    responses: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    validation_fn: Callable[[nn.Module], float] | None = None,
    patience: int | None = None,
    socket_feature: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    model.to(device)
    sampler_class = SocketEpisodeSampler if socket_feature else EpisodeSampler
    sampler = sampler_class(responses, job_profiles, inhibitor_profiles, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_score = float("inf")
    best_epoch = epochs
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            if socket_feature:
                batch = tuple(tensor.to(device) for tensor in sampler.sample(batch_size))
                optimizer.zero_grad(set_to_none=True)
                loss = censored_delta_huber_loss(socket_delta(model, batch), batch[-1])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                continue
            victim, anchor, query, target_logs = (
                tensor.to(device) for tensor in sampler.sample(batch_size)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = censored_delta_huber_loss(
                model(victim, anchor, query), target_logs
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = float(validation_fn(model)) if validation_fn is not None else np.nan
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "validation": score})
        if validation_fn is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience is not None and stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "epochs_run": len(history),
        "history": history,
    }


def train_absolute(
    model: nn.Module,
    responses: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    validation_fn: Callable[[nn.Module], float] | None = None,
    patience: int | None = None,
    socket_feature: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    model.to(device)
    if socket_feature:
        # 1/n_rows(job, inhib), rescaled to mean 1 within each victim: victims and then rows are sampled
        # uniformly, so every (victim, configuration) cell keeps its blind-mode expected weight.
        cell_rows = responses.groupby(["job_id", "inhib_id"])["inhib_id"].transform("size")
        victim_rows = responses.groupby("job_id")["inhib_id"].transform("size")
        victim_cells = responses.groupby("job_id")["inhib_id"].transform("nunique")
        responses = responses.assign(_cell_weight=victim_rows / (victim_cells * cell_rows))
    groups = {
        str(job_id): group.reset_index(drop=True)
        for job_id, group in responses.groupby("job_id", sort=False)
    }
    job_ids = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_score = float("inf")
    best_epoch = epochs
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            victim_values = []
            aggressor_values = []
            targets = []
            sockets = []
            weights = []
            for job_id in rng.choice(job_ids, size=batch_size, replace=True):
                group = groups[str(job_id)]
                row = group.iloc[int(rng.integers(len(group)))]
                victim_values.append(job_profiles[str(job_id)])
                aggressor_values.append(inhibitor_profiles[str(row["inhib_id"])])
                targets.append(float(row["log_slowdown"]))
                if socket_feature:
                    sockets.append(int(row[SOCKET_COLUMN]))
                    weights.append(float(row["_cell_weight"]))
            if socket_feature:
                victim_values, aggressor_values = socket_pair_inputs(
                    np.stack(victim_values), np.stack(aggressor_values), sockets
                )
            victim = torch.as_tensor(np.stack(victim_values), dtype=torch.float32, device=device)
            aggressor = torch.as_tensor(
                np.stack(aggressor_values), dtype=torch.float32, device=device
            )
            target = torch.as_tensor(targets, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = censored_absolute_huber_loss(
                model(victim, aggressor),
                target,
                torch.as_tensor(weights, dtype=torch.float32, device=device)
                if socket_feature
                else None,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = float(validation_fn(model)) if validation_fn is not None else np.nan
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "validation": score})
        if validation_fn is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience is not None and stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "epochs_run": len(history),
        "history": history,
    }


def response_values(
    model: nn.Module,
    victim: np.ndarray,
    aggressors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    victim_batch = (
        victim if victim.ndim == 2 else np.repeat(victim[None, :], len(aggressors), axis=0)
    )
    with torch.no_grad():
        values = model.response(
            torch.as_tensor(victim_batch, dtype=torch.float32, device=device),
            torch.as_tensor(aggressors, dtype=torch.float32, device=device),
        )
    return values.detach().cpu().numpy()
