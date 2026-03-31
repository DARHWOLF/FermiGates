from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .base import BaseGate


class FermiGate(BaseGate):
    """Fermi-Dirac gate over a weight tensor.

    Probability function:
        p(w) = 1 / (exp((|w| - mu) / T) + 1)
    """

    def __init__(
        self,
        shape: Sequence[int] | torch.Size,
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(shape=shape, init_T=init_T)
        self.mu = nn.Parameter(torch.full(self.shape, float(init_mu), dtype=torch.float32))

    def probabilities(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if w is None:
            raise ValueError("FermiGate requires a reference tensor `w` for probabilities().")
        z = (w.abs() - self.mu) / (self.T + 1e-12)
        return torch.sigmoid(-z)

    def initialize_mu_from_reference(
        self,
        reference: torch.Tensor,
        percentile: float = 0.5,
        per_output: bool = False,
    ) -> None:
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be in [0, 1].")

        with torch.no_grad():
            ref = reference.detach().abs()
            if per_output:
                first_dim = ref.shape[0]
                flat = ref.view(first_dim, -1)
                # quantile is clearer than top-k sorting for readability.
                q = torch.quantile(flat, percentile, dim=1)
                expanded = q.view(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
                self.mu.copy_(expanded)
            else:
                q = torch.quantile(ref.view(-1), percentile)
                self.mu.fill_(float(q.item()))
