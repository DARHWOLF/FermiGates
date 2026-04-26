from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fermigates.base import BaseFermiLayer
from fermigates.layers._base import BaseLayer


class Linear(BaseLayer, BaseFermiLayer):
    """Modular linear layer with optional input/weight/output gates.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default=True
        Whether to include an additive bias.
    weight_gate : nn.Module or None, optional
        Optional gate applied to the weight tensor.
    input_gate : nn.Module or None, optional
        Optional gate applied to the input tensor.
    output_gate : nn.Module or None, optional
        Optional gate applied to the output tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_gate: nn.Module | None = None,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        super().__init__(
            weight_gate=weight_gate,
            input_gate=input_gate,
            output_gate=output_gate,
        )

        # Step 1: Define learnable parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Step 2: Initialize layer parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight and bias parameters."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_weight(self) -> torch.Tensor:
        return self.weight

    def _core_forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        return F.linear(x, weight if weight is not None else self.weight, self.bias)

    def gate_probabilities(self) -> torch.Tensor:
        """Return weight-gate probabilities for sparsity utilities."""
        if self.weight_gate is None:
            return torch.ones_like(self.weight)
        return self.weight_gate(self.weight)


class FermiGatedLinear(Linear):
    """Compatibility wrapper around ``Linear`` with legacy constructor fields."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: nn.Module | None = None,
        weight_gate: nn.Module | None = None,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        # Step 1: Resolve compatibility gate precedence explicitly
        resolved_weight_gate = weight_gate if weight_gate is not None else gate

        # Step 2: Build generic linear layer without implicit default gating
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            weight_gate=resolved_weight_gate,
            input_gate=input_gate,
            output_gate=output_gate,
        )

        # Step 3: Keep legacy attribute aliases
        self.linear = SimpleNamespace(
            weight=self.weight,
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
        )
        self.gate = resolved_weight_gate
        self.mask = resolved_weight_gate
        self._init_mu = float(init_mu)
        self._init_T = float(init_T)

    def set_temperature(self, T_new: float) -> None:
        """Set temperature when the attached gate supports it."""
        if self.gate is None:
            return
        set_temperature_fn = getattr(self.gate, "set_temperature", None)
        if callable(set_temperature_fn):
            set_temperature_fn(T_new)

    def initialize_mu_from_weight_percentile(
        self,
        percentile: float = 0.5,
        per_neuron: bool = False,
    ) -> None:
        """Initialize gate thresholds from current weight statistics."""
        if self.gate is None:
            return
        init_fn = getattr(self.gate, "initialize_mu_from_reference", None)
        if callable(init_fn):
            parameters = inspect.signature(init_fn).parameters
            if "per_output" in parameters:
                init_fn(self.weight.abs(), percentile=percentile, per_output=per_neuron)
            else:
                init_fn(self.weight.abs(), percentile=percentile)

    @torch.no_grad()
    def hard_pruned_linear(self, threshold: float = 0.5) -> nn.Linear:
        """Export a dense ``nn.Linear`` with hard-thresholded gate mask."""

        # Step 1: Resolve hard mask explicitly based on gate availability
        if self.gate is None:
            hard = torch.ones_like(self.weight)
        else:
            hard_mask_fn = getattr(self.gate, "hard_mask", None)
            if callable(hard_mask_fn):
                hard = hard_mask_fn(self.weight, threshold=threshold).to(dtype=self.weight.dtype)
            else:
                hard = torch.ones_like(self.weight)

        # Step 2: Create dense linear module with masked weights
        dense = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        dense.weight.copy_(self.weight * hard)
        if self.bias is not None:
            dense.bias.copy_(self.bias)
        return dense
