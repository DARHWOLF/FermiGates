from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .base import BaseGate


class BinaryConcreteGate(BaseGate):
    """Binary Concrete (Gumbel-Sigmoid) gate over learnable logits."""

    def __init__(
        self,
        shape: Sequence[int] | torch.Size,
        init_log_alpha: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(shape=shape, init_T=init_T)
        self.log_alpha = nn.Parameter(
            torch.full(self.shape, float(init_log_alpha), dtype=torch.float32)
        )

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        del w
        return torch.sigmoid(self.log_alpha / (self.T + 1e-12))

    def sample(self) -> torch.Tensor:
        eps = 1e-6
        u = torch.rand_like(self.log_alpha).clamp_(eps, 1 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.log_alpha + logistic_noise) / (self.T + 1e-12))

    def forward(self, w: Optional[torch.Tensor] = None, sample: Optional[bool] = None) -> torch.Tensor:
        del w
        sample = self.training if sample is None else sample
        return self.sample() if sample else self.probabilities()


class HardConcreteGate(BaseGate):
    """Hard-Concrete gate for near-binary stochastic sparsity control."""

    def __init__(
        self,
        shape: Sequence[int] | torch.Size,
        init_log_alpha: float = 0.0,
        init_T: float = 2.0 / 3.0,
        lower: float = -0.1,
        upper: float = 1.1,
    ) -> None:
        super().__init__(shape=shape, init_T=init_T)
        if lower >= 0 or upper <= 1 or lower >= upper:
            raise ValueError("HardConcrete bounds must satisfy lower < 0 < 1 < upper.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.log_alpha = nn.Parameter(
            torch.full(self.shape, float(init_log_alpha), dtype=torch.float32)
        )

    def _stretch(self, s: torch.Tensor) -> torch.Tensor:
        return s * (self.upper - self.lower) + self.lower

    def _hard_sigmoid(self, s: torch.Tensor) -> torch.Tensor:
        return s.clamp(0.0, 1.0)

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        del w
        # Approximate expected gate-open probability from Hard-Concrete literature.
        shift = self.T * math.log(-self.lower / self.upper)
        return torch.sigmoid((self.log_alpha - shift) / (self.T + 1e-12)).clamp(0.0, 1.0)

    def sample(self) -> torch.Tensor:
        eps = 1e-6
        u = torch.rand_like(self.log_alpha).clamp_(eps, 1 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        s = torch.sigmoid((self.log_alpha + logistic_noise) / (self.T + 1e-12))
        s_bar = self._stretch(s)
        return self._hard_sigmoid(s_bar)

    def forward(self, w: Optional[torch.Tensor] = None, sample: Optional[bool] = None) -> torch.Tensor:
        del w
        sample = self.training if sample is None else sample
        return self.sample() if sample else self.probabilities()
