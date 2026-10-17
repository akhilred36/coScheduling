"""Legacy single-anchor training and joint absolute/delta/auxiliary training."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import RESPONSE_ENDPOINTS, ProfileScaler, aux_mask_column, aux_target_column


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
    latent_prediction: torch.Tensor, observed_log: torch.Tensor
) -> torch.Tensor:
    """Exact Huber loss above the floor and a one-sided loss at the floor."""
    residual = torch.where(
        observed_log > 0,
        latent_prediction - observed_log,
        torch.relu(latent_prediction),
    )
    return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)


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
) -> dict[str, object]:
    set_seed(seed)
    model.to(device)
    sampler = EpisodeSampler(responses, job_profiles, inhibitor_profiles, seed)
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
) -> dict[str, object]:
    set_seed(seed)
    model.to(device)
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
            for job_id in rng.choice(job_ids, size=batch_size, replace=True):
                group = groups[str(job_id)]
                row = group.iloc[int(rng.integers(len(group)))]
                victim_values.append(job_profiles[str(job_id)])
                aggressor_values.append(inhibitor_profiles[str(row["inhib_id"])])
                targets.append(float(row["log_slowdown"]))
            victim = torch.as_tensor(np.stack(victim_values), dtype=torch.float32, device=device)
            aggressor = torch.as_tensor(
                np.stack(aggressor_values), dtype=torch.float32, device=device
            )
            target = torch.as_tensor(targets, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = censored_absolute_huber_loss(model(victim, aggressor), target)
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
    victim_batch = np.repeat(victim[None, :], len(aggressors), axis=0)
    with torch.no_grad():
        values = model.response(
            torch.as_tensor(victim_batch, dtype=torch.float32, device=device),
            torch.as_tensor(aggressors, dtype=torch.float32, device=device),
        )
    return values.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Joint absolute/delta/auxiliary training for the new variants.
# ---------------------------------------------------------------------------


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    """A differentiable zero attached to the graph of ``reference``."""
    return reference.sum() * 0.0


def censored_absolute_residuals(
    latent_prediction: torch.Tensor, observed_log: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        observed_log > 0, latent_prediction - observed_log, torch.relu(latent_prediction)
    )


def masked_absolute_huber_loss(
    latent_prediction: torch.Tensor, observed_log: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    residual = censored_absolute_residuals(latent_prediction, observed_log)
    losses = F.huber_loss(residual, torch.zeros_like(residual), delta=1.0, reduction="none")
    total = weight.sum()
    if float(total.detach()) <= 0:
        return _zero_like(latent_prediction)
    return (losses * weight).sum() / total


def identified_delta_huber_loss(
    predicted_delta: torch.Tensor,
    anchor_log: torch.Tensor,
    query_log: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Censored delta Huber averaged over identified constraints only.

    Both observations at the floor identify no delta constraint and contribute
    neither loss nor denominator.
    """
    anchor_exact = anchor_log > 0
    query_exact = query_log > 0
    exact = anchor_exact & query_exact
    query_censored = anchor_exact & ~query_exact
    anchor_censored = ~anchor_exact & query_exact
    identified = valid & (exact | query_censored | anchor_censored)
    residual = torch.zeros_like(predicted_delta)
    residual = torch.where(exact, predicted_delta - (query_log - anchor_log), residual)
    residual = torch.where(query_censored, torch.relu(predicted_delta + anchor_log), residual)
    residual = torch.where(anchor_censored, torch.relu(query_log - predicted_delta), residual)
    count = int(identified.sum())
    if count == 0:
        return _zero_like(predicted_delta), 0
    losses = F.huber_loss(residual, torch.zeros_like(residual), delta=1.0, reduction="none")
    return losses[identified].mean(), count


def masked_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Scale-normalized Huber: mean within each task, then across available tasks."""
    if prediction.shape[-1] == 0:
        return _zero_like(prediction), []
    safe_target = torch.where(mask, target, prediction.detach())
    residual = (prediction - safe_target) / scales
    losses = F.huber_loss(residual, torch.zeros_like(residual), delta=1.0, reduction="none")
    task_losses = []
    counts = []
    for index in range(prediction.shape[-1]):
        selected = mask[..., index]
        count = int(selected.sum())
        counts.append(count)
        if count:
            task_losses.append(losses[..., index][selected].mean())
    if not task_losses:
        return _zero_like(prediction), counts
    return torch.stack(task_losses).mean(), counts


class JointEpisodeSampler:
    """Uniform applications, then distinct inhibitor configurations, then replicates.

    Each episode holds one victim application, a support set S of inhibitor IDs
    (used only for fingerprints), and two distinct query IDs (anchor, query)
    disjoint from S. All replicates of a query ID are excluded from S because
    tokens are built per configuration.
    """

    def __init__(
        self,
        responses: pd.DataFrame,
        job_profiles: dict[str, np.ndarray],
        inhibitor_profiles: dict[str, np.ndarray],
        *,
        aux_tasks: tuple[str, ...],
        seed: int,
        token_matrices: dict[str, tuple[list[str], np.ndarray]] | None = None,
        max_support: int = 0,
        empty_probability: float = 0.25,
    ):
        frame = responses.reset_index(drop=True)
        self.aux_tasks = tuple(aux_tasks)
        self.log = frame["log_slowdown"].to_numpy(dtype=np.float32)
        self.inhib_profile = np.stack(
            [inhibitor_profiles[str(value)] for value in frame["inhib_id"]]
        ).astype(np.float32)
        self.aux_target = {}
        self.aux_mask = {}
        for endpoint in RESPONSE_ENDPOINTS:
            targets = []
            masks = []
            for task in self.aux_tasks:
                targets.append(frame[aux_target_column(endpoint, task)].fillna(0.0).to_numpy(np.float32))
                masks.append(frame[aux_mask_column(endpoint, task)].astype(bool).to_numpy())
            width = len(self.aux_tasks)
            self.aux_target[endpoint] = (
                np.stack(targets, axis=1) if width else np.zeros((len(frame), 0), np.float32)
            )
            self.aux_mask[endpoint] = (
                np.stack(masks, axis=1) if width else np.zeros((len(frame), 0), bool)
            )
        self.groups: dict[str, dict[str, Any]] = {}
        for job_id, group in frame.groupby("job_id", sort=True):
            ids = sorted(group["inhib_id"].astype(str).unique())
            rows = [group.index[group["inhib_id"].astype(str) == value].to_numpy() for value in ids]
            entry = {"ids": ids, "rows": rows}
            if token_matrices is not None:
                token_ids, matrix = token_matrices[str(job_id)]
                if token_ids != ids:
                    raise ValueError("Token matrices must align with training inhibitor IDs")
                entry["tokens"] = matrix
            self.groups[str(job_id)] = entry
        if not self.groups:
            raise ValueError("Joint training requires at least one application")
        self.job_ids = np.asarray(sorted(self.groups), dtype=object)
        self.job_profiles = job_profiles
        self.token_matrices = token_matrices
        self.token_width = (
            next(iter(token_matrices.values()))[1].shape[1] if token_matrices else 0
        )
        self.max_support = int(max_support) if token_matrices is not None else 0
        self.empty_probability = float(empty_probability)
        self.rng = np.random.default_rng(seed)

    def support_size(self, available: int) -> int:
        limit = min(available - 2, self.max_support)
        if limit <= 0 or self.rng.random() < self.empty_probability:
            return 0
        return int(self.rng.integers(1, limit + 1))

    def sample_episode(self, job_id: str) -> dict[str, Any]:
        entry = self.groups[job_id]
        count = len(entry["ids"])
        if count >= 2:
            size = self.support_size(count)
            order = self.rng.permutation(count)
            support, anchor, query = order[:size], int(order[size]), int(order[size + 1])
            distinct = True
        else:
            support, anchor, query, distinct = np.empty(0, dtype=int), 0, 0, False
        return {
            "job_id": job_id,
            "support": support,
            "support_ids": [entry["ids"][index] for index in support],
            "anchor_row": int(self.rng.choice(entry["rows"][anchor])),
            "query_row": int(self.rng.choice(entry["rows"][query])),
            "anchor_id": entry["ids"][anchor],
            "query_id": entry["ids"][query],
            "distinct": distinct,
        }

    def sample(self, batch_size: int) -> dict[str, Any]:
        episodes = [
            self.sample_episode(str(job_id))
            for job_id in self.rng.choice(self.job_ids, size=batch_size, replace=True)
        ]
        anchor_rows = np.asarray([episode["anchor_row"] for episode in episodes])
        query_rows = np.asarray([episode["query_row"] for episode in episodes])
        distinct = np.asarray([episode["distinct"] for episode in episodes])
        batch: dict[str, Any] = {
            "victim": np.stack([self.job_profiles[episode["job_id"]] for episode in episodes]),
            "anchor": self.inhib_profile[anchor_rows],
            "query": self.inhib_profile[query_rows],
            "anchor_log": self.log[anchor_rows],
            "query_log": self.log[query_rows],
            "distinct": distinct,
            "episodes": episodes,
        }
        for endpoint in RESPONSE_ENDPOINTS:
            batch[f"{endpoint}_aux_target"] = np.stack(
                [self.aux_target[endpoint][anchor_rows], self.aux_target[endpoint][query_rows]], axis=1
            )
            query_mask = self.aux_mask[endpoint][query_rows] & distinct[:, None]
            batch[f"{endpoint}_aux_mask"] = np.stack(
                [self.aux_mask[endpoint][anchor_rows], query_mask], axis=1
            )
        if self.token_matrices is not None:
            width = max([len(episode["support"]) for episode in episodes] + [0])
            tokens = np.zeros((batch_size, width, self.token_width), dtype=np.float32)
            mask = np.zeros((batch_size, width), dtype=bool)
            for index, episode in enumerate(episodes):
                size = len(episode["support"])
                if size:
                    tokens[index, :size] = self.groups[episode["job_id"]]["tokens"][episode["support"]]
                    mask[index, :size] = True
            batch["support_tokens"] = tokens
            batch["support_mask"] = mask
        return batch


def sample_joint_episode(sampler: JointEpisodeSampler, job_id: str) -> dict[str, Any]:
    return sampler.sample_episode(str(job_id))


def batch_to_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        if key == "episodes":
            continue
        dtype = torch.bool if np.asarray(value).dtype == bool else torch.float32
        result[key] = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    return result


def joint_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    lambda_abs: float,
    lambda_delta: float,
    lambda_aux: float,
    aux_scales: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    context = None
    if "support_tokens" in batch and model.uses_context:
        context = (batch["support_tokens"], batch["support_mask"])
    victim_embedding = model.embed(batch["victim"], context)
    anchor_embedding = model.embed(batch["anchor"])
    query_embedding = model.embed(batch["query"])
    anchor_response = model.response_from_embeddings(victim_embedding, anchor_embedding)
    query_response = model.response_from_embeddings(victim_embedding, query_embedding)
    distinct = batch["distinct"]
    weights = torch.cat([torch.ones_like(anchor_response), distinct.to(query_response.dtype)])
    absolute = masked_absolute_huber_loss(
        torch.cat([anchor_response, query_response]),
        torch.cat([batch["anchor_log"], batch["query_log"]]),
        weights,
    )
    delta, delta_count = identified_delta_huber_loss(
        query_response - anchor_response, batch["anchor_log"], batch["query_log"], distinct
    )
    total = lambda_abs * absolute + lambda_delta * delta
    auxiliary = _zero_like(total)
    counts: list[int] = []
    if lambda_aux > 0 and getattr(model, "bandwidth_head", None) is not None:
        forward = torch.stack(
            [
                model.bandwidth_from_embeddings(victim_embedding, anchor_embedding),
                model.bandwidth_from_embeddings(victim_embedding, query_embedding),
            ],
            dim=1,
        )
        # Reverse direction: the inhibitor is the victim endpoint, the app aggresses.
        reverse = torch.stack(
            [
                model.bandwidth_from_embeddings(anchor_embedding, victim_embedding),
                model.bandwidth_from_embeddings(query_embedding, victim_embedding),
            ],
            dim=1,
        )
        auxiliary, counts = masked_auxiliary_loss(
            torch.cat([forward, reverse], dim=1),
            torch.cat([batch["app_aux_target"], batch["inhib_aux_target"]], dim=1),
            torch.cat([batch["app_aux_mask"], batch["inhib_aux_mask"]], dim=1),
            aux_scales,
        )
        total = total + lambda_aux * auxiliary
    return total, {
        "absolute_loss": float(absolute.detach()),
        "delta_loss": float(delta.detach()),
        "auxiliary_loss": float(auxiliary.detach()),
        "delta_constraints": delta_count,
        "auxiliary_valid_counts": counts,
    }


def train_joint_response(
    model: nn.Module,
    responses: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    aux_tasks: tuple[str, ...],
    aux_scales: np.ndarray | None,
    lambda_abs: float,
    lambda_delta: float,
    lambda_aux: float,
    seed: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    token_matrices: dict[str, tuple[list[str], np.ndarray]] | None = None,
    max_support: int = 0,
    empty_probability: float = 0.25,
    validation_fn: Callable[[nn.Module], float] | None = None,
    patience: int | None = None,
) -> dict[str, object]:
    set_seed(seed)
    model.to(device)
    sampler = JointEpisodeSampler(
        responses,
        job_profiles,
        inhibitor_profiles,
        aux_tasks=aux_tasks,
        seed=seed,
        token_matrices=token_matrices,
        max_support=max_support,
        empty_probability=empty_probability,
    )
    scales = (
        torch.as_tensor(aux_scales, dtype=torch.float32, device=device)
        if aux_scales is not None and len(aux_scales)
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_score = float("inf")
    best_epoch = epochs
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        components: dict[str, list[float]] = {
            "train_loss": [], "absolute_loss": [], "delta_loss": [], "auxiliary_loss": []
        }
        delta_constraints = 0
        aux_counts = np.zeros(len(aux_tasks), dtype=int)
        support_sizes = []
        for _ in range(batches_per_epoch):
            raw = sampler.sample(batch_size)
            support_sizes.extend(len(episode["support"]) for episode in raw["episodes"])
            batch = batch_to_tensors(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = joint_loss(
                model, batch, lambda_abs=lambda_abs, lambda_delta=lambda_delta,
                lambda_aux=lambda_aux, aux_scales=scales,
            )
            loss.backward()
            optimizer.step()
            components["train_loss"].append(float(loss.detach().cpu()))
            for key in ["absolute_loss", "delta_loss", "auxiliary_loss"]:
                components[key].append(details[key])
            delta_constraints += details["delta_constraints"]
            if details["auxiliary_valid_counts"]:
                aux_counts += np.asarray(details["auxiliary_valid_counts"], dtype=int)
        score = float(validation_fn(model)) if validation_fn is not None else np.nan
        record: dict[str, Any] = {
            "epoch": epoch,
            **{key: float(np.mean(values)) for key, values in components.items()},
            "delta_constraints": delta_constraints,
            "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
            "validation": score,
        }
        for task, count in zip(aux_tasks, aux_counts):
            record[f"auxiliary_valid_{task}"] = int(count)
        history.append(record)
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
