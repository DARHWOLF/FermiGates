from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fermigates.base import BaseFermiLayer
from fermigates.gates import BaseGate, FermiGate


class FermiGatedConv2d(BaseFermiLayer):
    """2D convolution with differentiable weight gating."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: Optional[BaseGate] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.gate = gate or FermiGate(self.conv.weight.shape, init_mu=init_mu, init_T=init_T)
        if self.gate.shape != self.conv.weight.shape:
            raise ValueError(
                f"Gate shape {tuple(self.gate.shape)} must match weight shape {tuple(self.conv.weight.shape)}."
            )

        self.mask = self.gate

    def gate_probabilities(self) -> torch.Tensor:
        return self.gate(self.conv.weight)

    def effective_weight(self) -> torch.Tensor:
        return self.gate_probabilities() * self.conv.weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.gate_probabilities()
        out = F.conv2d(
            x,
            p * self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out, p

    def set_temperature(self, T_new: float) -> None:
        self.gate.set_temperature(T_new)

    def initialize_mu_from_weight_percentile(self, percentile: float = 0.5, per_neuron: bool = False) -> None:
        init_fn = getattr(self.gate, "initialize_mu_from_reference", None)
        if callable(init_fn):
            init_fn(self.conv.weight, percentile=percentile, per_output=per_neuron)

    @torch.no_grad()
    def hard_pruned_conv2d(self, threshold: float = 0.5) -> nn.Conv2d:
        """Export a vanilla ``nn.Conv2d`` with hard-thresholded gate mask."""

        hard = self.gate.hard_mask(self.conv.weight, threshold=threshold)
        dense = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=self.conv.bias is not None,
            padding_mode=self.conv.padding_mode,
            device=self.conv.weight.device,
            dtype=self.conv.weight.dtype,
        )
        dense.weight.copy_(self.conv.weight * hard)
        if self.conv.bias is not None:
            dense.bias.copy_(self.conv.bias)
        return dense
