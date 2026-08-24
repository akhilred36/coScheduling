"""Fixed-optimizer-step neural and convex absolute-response training."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_finite_model(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"Model parameter became non-finite: {name}")


def censored_delta_huber_loss(
    predicted_delta: torch.Tensor, target_logs: torch.Tensor
) -> tuple[torch.Tensor, int]:
    anchor = target_logs[:, 0]
    query = target_logs[:, 1]
    anchor_exact = anchor > 0
    query_exact = query > 0
    exact = anchor_exact & query_exact
    query_floor = anchor_exact & ~query_exact
    anchor_floor = ~anchor_exact & query_exact
    residual = torch.zeros_like(predicted_delta)
    residual = torch.where(exact, predicted_delta - (query - anchor), residual)
    residual = torch.where(query_floor, torch.relu(predicted_delta + anchor), residual)
    residual = torch.where(anchor_floor, torch.relu(query - predicted_delta), residual)
    informative = exact | query_floor | anchor_floor
    informative_count = int(informative.sum().item())
    if informative_count == 0:
        loss = predicted_delta.sum() * 0.0
    else:
        selected = residual[informative]
        loss = F.huber_loss(selected, torch.zeros_like(selected), delta=1.0)
    return loss, informative_count


def censored_absolute_huber_loss(
    latent_prediction: torch.Tensor, observed_log: torch.Tensor
) -> torch.Tensor:
    residual = torch.where(
        observed_log > 0,
        latent_prediction - observed_log,
        torch.relu(latent_prediction),
    )
    return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)


class DeltaSampler:
    def __init__(
        self,
        responses: pd.DataFrame,
        job_profiles: dict[str, np.ndarray],
        inhibitor_profiles: dict[str, np.ndarray],
        seed: int,
    ):
        self.groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for job_id, group in responses.groupby("job_id", sort=False):
            if group["inhib_id"].nunique() < 2 or not (group["log_slowdown"] > 0).any():
                continue
            inhibitor_ids = group["inhib_id"].astype(str).to_numpy()
            self.groups.append(
                (
                job_profiles[str(job_id)],
                inhibitor_ids,
                np.stack([inhibitor_profiles[value] for value in inhibitor_ids]),
                group["log_slowdown"].to_numpy(dtype=float),
                )
            )
        if not self.groups:
            raise ValueError("No victim has responses for two distinct inhibitors")
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        profile_dim = self.groups[0][0].shape[0]
        victims = np.empty((batch_size, profile_dim), dtype=np.float32)
        anchors = np.empty_like(victims)
        queries = np.empty_like(victims)
        targets = np.empty((batch_size, 2), dtype=np.float32)
        chosen = self.rng.integers(len(self.groups), size=batch_size)
        for group_index in np.unique(chosen):
            positions = np.flatnonzero(chosen == group_index)
            victim, inhibitor_ids, profiles, logs = self.groups[int(group_index)]
            left = self.rng.integers(len(inhibitor_ids), size=len(positions))
            right = self.rng.integers(len(inhibitor_ids), size=len(positions))
            invalid = (inhibitor_ids[left] == inhibitor_ids[right]) | (
                (logs[left] == 0) & (logs[right] == 0)
            )
            while invalid.any():
                right[invalid] = self.rng.integers(
                    len(inhibitor_ids), size=int(invalid.sum())
                )
                invalid = (inhibitor_ids[left] == inhibitor_ids[right]) | (
                    (logs[left] == 0) & (logs[right] == 0)
                )
            victims[positions] = victim
            anchors[positions] = profiles[left]
            queries[positions] = profiles[right]
            targets[positions, 0] = logs[left]
            targets[positions, 1] = logs[right]
        return (
            torch.as_tensor(victims, dtype=torch.float32),
            torch.as_tensor(anchors, dtype=torch.float32),
            torch.as_tensor(queries, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
        )


class AbsoluteSampler:
    def __init__(
        self,
        responses: pd.DataFrame,
        job_profiles: dict[str, np.ndarray],
        inhibitor_profiles: dict[str, np.ndarray],
        seed: int,
    ):
        self.groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for job_id, group in responses.groupby("job_id", sort=False):
            identifiers = group["inhib_id"].astype(str).to_numpy()
            self.groups.append(
                (
                    job_profiles[str(job_id)],
                    np.stack([inhibitor_profiles[value] for value in identifiers]),
                    group["log_slowdown"].to_numpy(dtype=float),
                )
            )
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        profile_dim = self.groups[0][0].shape[0]
        victims = np.empty((batch_size, profile_dim), dtype=np.float32)
        aggressors = np.empty_like(victims)
        targets = np.empty(batch_size, dtype=np.float32)
        chosen = self.rng.integers(len(self.groups), size=batch_size)
        for group_index in np.unique(chosen):
            positions = np.flatnonzero(chosen == group_index)
            victim, profiles, logs = self.groups[int(group_index)]
            rows = self.rng.integers(len(logs), size=len(positions))
            victims[positions] = victim
            aggressors[positions] = profiles[rows]
            targets[positions] = logs[rows]
        return (
            torch.as_tensor(victims, dtype=torch.float32),
            torch.as_tensor(aggressors, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
        )


def _validation_record(
    validation_fn: Callable[[nn.Module], dict[str, float]] | None,
    model: nn.Module,
) -> dict[str, float]:
    if validation_fn is None:
        return {}
    model.eval()
    values = {key: float(value) for key, value in validation_fn(model).items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("Validation produced a non-finite metric")
    return values


def train_potential_fixed(
    model: nn.Module,
    responses: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    seed: int,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_interval: int,
    device: torch.device,
    validation_fn: Callable[[nn.Module], dict[str, float]] | None = None,
) -> dict[str, object]:
    if optimizer_steps <= 0 or batch_size <= 0 or checkpoint_interval <= 0:
        raise ValueError("Training budget fields must be positive")
    set_seed(seed)
    model.to(device)
    sampler = DeltaSampler(responses, job_profiles, inhibitor_profiles, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = [{"step": 0, "train_loss": np.nan, **_validation_record(validation_fn, model)}]
    interval_losses: list[float] = []
    informative_examples = 0
    for step in range(1, optimizer_steps + 1):
        model.train()
        victim, anchor, query, targets = (
            tensor.to(device) for tensor in sampler.sample(batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        loss, informative = censored_delta_huber_loss(
            model(victim, anchor, query), targets
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Potential training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        assert_finite_model(model)
        interval_losses.append(float(loss.detach().cpu()))
        informative_examples += informative
        if step % checkpoint_interval == 0 or step == optimizer_steps:
            history.append(
                {
                    "step": step,
                    "train_loss": float(np.mean(interval_losses)),
                    **_validation_record(validation_fn, model),
                }
            )
            interval_losses = []
    return {
        "optimizer_steps": optimizer_steps,
        "sampled_examples": optimizer_steps * batch_size,
        "informative_examples": informative_examples,
        "history": history,
    }


def train_absolute_fixed(
    model: nn.Module,
    responses: pd.DataFrame,
    job_profiles: dict[str, np.ndarray],
    inhibitor_profiles: dict[str, np.ndarray],
    *,
    seed: int,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_interval: int,
    device: torch.device,
    validation_fn: Callable[[nn.Module], dict[str, float]] | None = None,
) -> dict[str, object]:
    if optimizer_steps <= 0 or batch_size <= 0 or checkpoint_interval <= 0:
        raise ValueError("Training budget fields must be positive")
    set_seed(seed)
    model.to(device)
    sampler = AbsoluteSampler(responses, job_profiles, inhibitor_profiles, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = [{"step": 0, "train_loss": np.nan, **_validation_record(validation_fn, model)}]
    interval_losses: list[float] = []
    for step in range(1, optimizer_steps + 1):
        model.train()
        victim, aggressor, target = (
            tensor.to(device) for tensor in sampler.sample(batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        loss = censored_absolute_huber_loss(model(victim, aggressor), target)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Absolute training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        assert_finite_model(model)
        interval_losses.append(float(loss.detach().cpu()))
        if step % checkpoint_interval == 0 or step == optimizer_steps:
            history.append(
                {
                    "step": step,
                    "train_loss": float(np.mean(interval_losses)),
                    **_validation_record(validation_fn, model),
                }
            )
            interval_losses = []
    return {
        "optimizer_steps": optimizer_steps,
        "sampled_examples": optimizer_steps * batch_size,
        "informative_examples": optimizer_steps * batch_size,
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
    result = values.detach().cpu().numpy().astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("Model response produced non-finite values")
    return result


def absolute_values(
    model: nn.Module,
    victims: np.ndarray,
    aggressors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(
            torch.as_tensor(victims, dtype=torch.float32, device=device),
            torch.as_tensor(aggressors, dtype=torch.float32, device=device),
        )
    result = values.detach().cpu().numpy().astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("Absolute model produced non-finite values")
    return result


def linear_design(victims: np.ndarray, aggressors: np.ndarray) -> np.ndarray:
    victim_values = np.asarray(victims, dtype=float)
    aggressor_values = np.asarray(aggressors, dtype=float)
    if victim_values.shape != aggressor_values.shape or victim_values.ndim != 2:
        raise ValueError("Victim and aggressor matrices must have matching 2-D shapes")
    return np.column_stack(
        [
            np.ones(len(victim_values)),
            victim_values,
            aggressor_values,
            victim_values * aggressor_values,
        ]
    )


def fit_censored_linear(
    design: np.ndarray,
    observed_log: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a convex censored-Huber linear baseline with an unpenalized intercept."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(observed_log, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Invalid linear-regression inputs")
    if ridge < 0 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid ridge or observed targets")
    exact = y > 0
    penalty = np.ones(x.shape[1], dtype=float)
    penalty[0] = 0.0

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        latent = x @ coefficients
        residual = np.where(exact, latent - y, np.maximum(latent, 0.0))
        absolute = np.abs(residual)
        losses = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
        derivative = np.clip(residual, -1.0, 1.0)
        derivative[~exact & (latent <= 0)] = 0.0
        value = float(losses.mean() + 0.5 * ridge * np.sum((penalty * coefficients) ** 2))
        gradient = x.T @ derivative / len(x) + ridge * penalty * coefficients
        return value, gradient

    result = minimize(
        lambda coefficients: objective(coefficients),
        np.zeros(x.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    coefficients = np.asarray(result.x, dtype=float)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Censored linear fit produced non-finite coefficients")
    return coefficients, {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }
