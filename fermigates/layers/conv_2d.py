from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fermigates.base import BaseFermiLayer
from fermigates.layers._base import BaseLayer


class Conv2d(BaseLayer, BaseFermiLayer):
    """Modular 2D convolution layer with optional gates.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Convolution kernel size.
    stride : int or tuple[int, int], default=1
        Convolution stride.
    padding : int or tuple[int, int], default=0
        Zero-padding size.
    dilation : int or tuple[int, int], default=1
        Dilation factor.
    groups : int, default=1
        Grouped convolution factor.
    bias : bool, default=True
        Whether to include an additive bias.
    padding_mode : str, default="zeros"
        Padding mode for ``nn.Conv2d``.
    weight_gate : nn.Module or None, optional
        Optional gate applied to convolution weights.
    input_gate : nn.Module or None, optional
        Optional gate applied to the input tensor.
    output_gate : nn.Module or None, optional
        Optional gate applied to the output tensor.
    """

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
        weight_gate: nn.Module | None = None,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        super().__init__(
            weight_gate=weight_gate,
            input_gate=input_gate,
            output_gate=output_gate,
        )

        # Step 1: Define underlying convolution module
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

    def _get_weight(self) -> torch.Tensor:
        return self.conv.weight

    def _core_forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        return F.conv2d(
            x,
            weight if weight is not None else self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def gate_probabilities(self) -> torch.Tensor:
        """Return weight-gate probabilities for sparsity utilities."""
        if self.weight_gate is None:
            return torch.ones_like(self.conv.weight)
        return self.weight_gate(self.conv.weight)


class FermiGatedConv2d(Conv2d):
    """Compatibility wrapper around ``Conv2d`` with legacy constructor fields."""

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
        gate: nn.Module | None = None,
        weight_gate: nn.Module | None = None,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        # Step 1: Resolve compatibility gate precedence explicitly
        resolved_weight_gate = weight_gate if weight_gate is not None else gate

        # Step 2: Build generic convolution layer without implicit default gating
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_gate=resolved_weight_gate,
            input_gate=input_gate,
            output_gate=output_gate,
        )

        # Step 3: Keep legacy attribute aliases
        self.gate = resolved_weight_gate
        self.mask = resolved_weight_gate
        self._init_mu = float(init_mu)
        self._init_T = float(init_T)
        self._build_conv_view()

    def _build_conv_view(self) -> None:
        """Create lightweight compatibility view for old attribute access."""
        self.conv_view = SimpleNamespace(
            weight=self.conv.weight,
            bias=self.conv.bias,
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            padding_mode=self.conv.padding_mode,
        )

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
                init_fn(self.conv.weight.abs(), percentile=percentile, per_output=per_neuron)
            else:
                init_fn(self.conv.weight.abs(), percentile=percentile)

    @torch.no_grad()
    def hard_pruned_conv2d(self, threshold: float = 0.5) -> nn.Conv2d:
        """Export a dense ``nn.Conv2d`` with hard-thresholded gate mask."""

        # Step 1: Resolve hard mask explicitly based on gate availability
        if self.gate is None:
            hard = torch.ones_like(self.conv.weight)
        else:
            hard_mask_fn = getattr(self.gate, "hard_mask", None)
            if callable(hard_mask_fn):
                hard = hard_mask_fn(self.conv.weight, threshold=threshold).to(
                    dtype=self.conv.weight.dtype
                )
            else:
                hard = torch.ones_like(self.conv.weight)

        # Step 2: Create dense convolution module with masked weights
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
