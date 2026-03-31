from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fermigates.base import BaseFermiLayer
from fermigates.gates import BaseGate, FermiGate


class FermiGatedLinear(BaseFermiLayer):
    """Linear layer with a differentiable gate on weights.

    By default this uses :class:`FermiGate`, but any ``BaseGate`` with matching
    shape can be injected.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: Optional[BaseGate] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gate = gate or FermiGate(self.linear.weight.shape, init_mu=init_mu, init_T=init_T)
        if self.gate.shape != self.linear.weight.shape:
            raise ValueError(
                f"Gate shape {tuple(self.gate.shape)} must match weight shape {tuple(self.linear.weight.shape)}."
            )

        # Compatibility with earlier API naming.
        self.mask = self.gate

    def gate_probabilities(self) -> torch.Tensor:
        return self.gate(self.linear.weight)

    def effective_weight(self) -> torch.Tensor:
        return self.gate_probabilities() * self.linear.weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.gate_probabilities()
        out = F.linear(x, p * self.linear.weight, self.linear.bias)
        return out, p

    def set_temperature(self, T_new: float) -> None:
        self.gate.set_temperature(T_new)

    def initialize_mu_from_weight_percentile(self, percentile: float = 0.5, per_neuron: bool = False) -> None:
        init_fn = getattr(self.gate, "initialize_mu_from_reference", None)
        if callable(init_fn):
            init_fn(self.linear.weight, percentile=percentile, per_output=per_neuron)

    @torch.no_grad()
    def hard_pruned_linear(self, threshold: float = 0.5) -> nn.Linear:
        """Export a vanilla ``nn.Linear`` with hard-thresholded gate mask."""

        hard = self.gate.hard_mask(self.linear.weight, threshold=threshold)
        dense = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        dense.weight.copy_(self.linear.weight * hard)
        if self.linear.bias is not None:
            dense.bias.copy_(self.linear.bias)
        return dense
