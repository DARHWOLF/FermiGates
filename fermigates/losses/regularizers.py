from __future__ import annotations

import math
from typing import Literal

import torch

Reduction = Literal["none", "mean", "sum"]


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduction '{reduction}'.")


def kl_to_bernoulli_prior_loss(
    probabilities: torch.Tensor,
    prior_prob: float | torch.Tensor = 0.5,
    reduction: Reduction = "sum",
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(Bernoulli(p) || Bernoulli(q)) regularizer."""

    q = torch.as_tensor(prior_prob, dtype=probabilities.dtype, device=probabilities.device)
    if torch.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("prior_prob must be strictly between 0 and 1.")

    p = probabilities.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)

    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    return _reduce(kl, reduction)


def group_sparsity_l21_loss(
    values: torch.Tensor,
    group_dim: int = 0,
    reduction: Reduction = "sum",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Group sparsity loss: sum/mean of L2 norms across groups."""

    moved = values.movedim(group_dim, 0)
    flat = moved.reshape(moved.shape[0], -1)
    norms = torch.sqrt(flat.square().sum(dim=1) + eps)
    return _reduce(norms, reduction)


def hoyer_sparsity_score(
    values: torch.Tensor,
    dim: int | None = None,
    reduction: Reduction = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Hoyer sparsity index in [0, 1], where larger means sparser."""

    if dim is None:
        flat = values.reshape(1, -1)
    else:
        moved = values.movedim(dim, 0)
        flat = moved.reshape(moved.shape[0], -1)

    l1 = flat.abs().sum(dim=1)
    l2 = torch.sqrt(flat.square().sum(dim=1) + eps)
    ratio = l1 / (l2 + eps)

    n = flat.shape[1]
    if n <= 1:
        score = torch.ones_like(ratio)
    else:
        denom = math.sqrt(float(n)) - 1.0
        score = (math.sqrt(float(n)) - ratio) / max(denom, eps)
    return _reduce(score.clamp(0.0, 1.0), reduction)


def hoyer_sparsity_loss(
    values: torch.Tensor,
    dim: int | None = None,
    reduction: Reduction = "mean",
    normalized: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Hoyer-based sparsity regularizer.

    If ``normalized=False`` this returns ``L1/L2`` (smaller is sparser).
    If ``normalized=True`` this returns ``1 - hoyer_sparsity_score``.
    """

    if normalized:
        return 1.0 - hoyer_sparsity_score(values, dim=dim, reduction=reduction, eps=eps)

    if dim is None:
        flat = values.reshape(1, -1)
    else:
        moved = values.movedim(dim, 0)
        flat = moved.reshape(moved.shape[0], -1)
    ratio = flat.abs().sum(dim=1) / torch.sqrt(flat.square().sum(dim=1) + eps)
    return _reduce(ratio, reduction)
