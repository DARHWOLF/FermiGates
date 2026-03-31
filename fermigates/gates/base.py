from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
import torch.nn as nn


class BaseGate(nn.Module, ABC):
    """Base interface for differentiable gates used by Fermi layers.

    A gate maps either a reference tensor (typically a weight tensor) or an
    internal score tensor to probabilities in ``[0, 1]``.
    """

    def __init__(self, shape: Sequence[int] | torch.Size, init_T: float = 1.0) -> None:
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer("T", torch.tensor(float(init_T), dtype=torch.float32))

    @property
    def temperature(self) -> float:
        return float(self.T.item())

    def set_temperature(self, T_new: float) -> None:
        if T_new <= 0:
            raise ValueError("Temperature must be positive.")
        self.T.fill_(float(T_new))

    @abstractmethod
    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return soft gate probabilities in [0, 1]."""

    def forward(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.probabilities(w)

    def hard_mask(self, w: Optional[torch.Tensor] = None, threshold: float = 0.5) -> torch.Tensor:
        probs = self.probabilities(w)
        return (probs >= threshold).to(dtype=probs.dtype)

    def l0_penalty(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.probabilities(w).sum()
