from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .base import BaseGate


class MagnitudeGate(BaseGate):
    """Deterministic gate that normalizes importance scores to [0, 1].

    If scores are not provided, ``w.abs()`` from the forward input is used.
    """

    def __init__(
        self,
        shape: Sequence[int] | torch.Size,
        scores: Optional[torch.Tensor] = None,
        inverted: bool = False,
    ) -> None:
        super().__init__(shape=shape, init_T=1.0)
        if scores is not None and scores.shape != torch.Size(shape):
            raise ValueError("scores shape must match gate shape.")
        if scores is None:
            scores = torch.zeros(shape, dtype=torch.float32)
        self.register_buffer("scores", scores.detach().clone().to(dtype=torch.float32))
        self.inverted = bool(inverted)

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        score = self.scores
        if w is not None:
            score = w.detach().abs().to(dtype=torch.float32)
        if self.inverted:
            score = -score

        smin = score.amin()
        smax = score.amax()
        denom = (smax - smin).clamp_min(1e-12)
        return (score - smin) / denom


class GroupLassoGate(BaseGate):
    """Group-wise sigmoid gates repeated by ``group_size``."""

    def __init__(self, groups: int, group_size: int = 1, init: float = 0.0) -> None:
        if groups <= 0 or group_size <= 0:
            raise ValueError("groups and group_size must be positive integers.")
        self.groups = int(groups)
        self.group_size = int(group_size)
        super().__init__(shape=(self.groups * self.group_size,), init_T=1.0)
        self.gate = nn.Parameter(torch.full((self.groups,), float(init), dtype=torch.float32))

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        del w
        p = torch.sigmoid(self.gate)
        return p.repeat_interleave(self.group_size)


class GompertzGate(BaseGate):
    """Deterministic Gompertz gate ``exp(-alpha * exp(-beta * s))``."""

    def __init__(
        self,
        size: int,
        alpha: float = 2.0,
        beta: float = 1.0,
        learn_alpha: bool = False,
        init_score: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive.")
        super().__init__(shape=(size,), init_T=1.0)
        self.s = nn.Parameter(torch.full((size,), float(init_score), dtype=torch.float32))

        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        self.beta = float(beta)

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        del w
        return torch.exp(-self.alpha * torch.exp(-self.beta * self.s))
