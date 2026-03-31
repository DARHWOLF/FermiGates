from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Reduction = Literal["none", "mean", "sum"]
NormKind = Literal["l1", "l2"]


@dataclass(frozen=True)
class FermiLossBreakdown:
    """Named components for easier training logs and ablations."""

    total: torch.Tensor
    task: torch.Tensor
    free_energy: torch.Tensor
    sparsity: torch.Tensor
    budget: torch.Tensor
    consistency: torch.Tensor


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduction '{reduction}'.")


def _as_temperature(
    temperature: float | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    temp = torch.as_tensor(temperature, dtype=dtype, device=device)
    if torch.any(temp <= 0):
        raise ValueError("temperature must be strictly positive.")
    return temp.mean()


def binary_entropy_loss(
    probabilities: torch.Tensor,
    reduction: Reduction = "sum",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary entropy over occupancy probabilities in ``[0, 1]``."""

    p = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return _reduce(entropy, reduction)


def fermi_free_energy_loss(
    probabilities: torch.Tensor,
    energies: torch.Tensor,
    interaction: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
    reduction: Reduction = "sum",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Free-energy inspired term: ``P*eps + 0.5*P*J(P) - T*H(P)``."""

    if probabilities.shape != energies.shape:
        raise ValueError(
            "probabilities and energies must have same shape, "
            f"got {probabilities.shape} and {energies.shape}."
        )

    interaction_term = torch.zeros_like(probabilities)
    if interaction is not None:
        if interaction.shape != probabilities.shape:
            raise ValueError(
                "interaction must match probabilities shape, "
                f"got {interaction.shape} and {probabilities.shape}."
            )
        interaction_term = 0.5 * probabilities * interaction

    temperature_scalar = _as_temperature(
        temperature,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    entropy = binary_entropy_loss(probabilities, reduction="none", eps=eps)
    free_energy = (probabilities * energies) + interaction_term - (temperature_scalar * entropy)
    return _reduce(free_energy, reduction)


def sparsity_l1_loss(probabilities: torch.Tensor, reduction: Reduction = "sum") -> torch.Tensor:
    """L1-style occupancy penalty to encourage sparse gate activations."""

    return _reduce(probabilities.abs(), reduction)


def budget_penalty_loss(
    probabilities: torch.Tensor,
    target: float | torch.Tensor,
    target_is_fraction: bool = True,
) -> torch.Tensor:
    """Quadratic occupancy budget penalty ``(sum(P) - target)^2``."""

    target_tensor = torch.as_tensor(target, dtype=probabilities.dtype, device=probabilities.device)
    if target_is_fraction:
        if torch.any((target_tensor < 0.0) | (target_tensor > 1.0)):
            raise ValueError("Fractional budget target must be in [0, 1].")
        target_tensor = target_tensor * probabilities.numel()
    elif torch.any(target_tensor < 0.0):
        raise ValueError("Absolute budget target must be non-negative.")

    return (probabilities.sum() - target_tensor).pow(2)


def consistency_loss(
    probabilities: torch.Tensor,
    previous_probabilities: torch.Tensor,
    norm: NormKind = "l2",
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Temporal consistency penalty against a previous occupancy estimate."""

    if probabilities.shape != previous_probabilities.shape:
        raise ValueError(
            "probabilities and previous_probabilities must have the same shape, "
            f"got {probabilities.shape} and {previous_probabilities.shape}."
        )

    delta = probabilities - previous_probabilities
    if norm == "l1":
        return _reduce(delta.abs(), reduction)
    if norm == "l2":
        return _reduce(delta.square(), reduction)
    raise ValueError(f"Unsupported norm '{norm}'.")


def fermi_informed_loss(
    task_loss: torch.Tensor,
    probabilities: torch.Tensor,
    energies: torch.Tensor,
    *,
    interaction: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
    lambda_free_energy: float = 1.0,
    lambda_sparsity: float = 0.0,
    lambda_budget: float = 0.0,
    budget_target: float | torch.Tensor | None = None,
    budget_target_is_fraction: bool = True,
    lambda_consistency: float = 0.0,
    previous_probabilities: torch.Tensor | None = None,
    consistency_norm: NormKind = "l2",
) -> FermiLossBreakdown:
    """Compose task and regularization losses used in Fermi-informed training."""

    free_energy = fermi_free_energy_loss(
        probabilities=probabilities,
        energies=energies,
        interaction=interaction,
        temperature=temperature,
        reduction="sum",
    )
    sparsity = sparsity_l1_loss(probabilities, reduction="sum")

    budget = torch.zeros_like(task_loss)
    if lambda_budget != 0.0:
        if budget_target is None:
            raise ValueError("budget_target is required when lambda_budget is non-zero.")
        budget = budget_penalty_loss(
            probabilities=probabilities,
            target=budget_target,
            target_is_fraction=budget_target_is_fraction,
        )

    consistency = torch.zeros_like(task_loss)
    if lambda_consistency != 0.0:
        if previous_probabilities is None:
            raise ValueError(
                "previous_probabilities is required when "
                "lambda_consistency is non-zero."
            )
        consistency = consistency_loss(
            probabilities=probabilities,
            previous_probabilities=previous_probabilities,
            norm=consistency_norm,
            reduction="mean",
        )

    total = (
        task_loss
        + (lambda_free_energy * free_energy)
        + (lambda_sparsity * sparsity)
        + (lambda_budget * budget)
        + (lambda_consistency * consistency)
    )
    return FermiLossBreakdown(
        total=total,
        task=task_loss,
        free_energy=free_energy,
        sparsity=sparsity,
        budget=budget,
        consistency=consistency,
    )
