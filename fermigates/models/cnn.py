from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from fermigates.base import BaseFermiClassifier
from fermigates.gates import BaseGate
from fermigates.layers.conv_2d import FermiGatedConv2d
from fermigates.layers.linear import FermiGatedLinear


class FermiConvClassifier(BaseFermiClassifier):
    """Compact CNN classifier with Fermi-gated convolution and linear head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (32, 64, 96),
        dropout: float = 0.1,
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: Callable[..., BaseGate] | BaseGate | None = None,
    ) -> None:
        super().__init__(num_classes=num_classes)
        c1, c2, c3 = channels

        # Step 1: resolve optional user gate into per-layer factory
        gate_factory = self._normalize_gate_factory(gate)

        # Step 2: build gated convolution stack
        self.conv1 = FermiGatedConv2d(
            in_channels,
            c1,
            kernel_size=3,
            padding=1,
            init_mu=init_mu,
            init_T=init_T,
            gate=self._build_gate(gate_factory, (c1, in_channels, 3, 3)),
        )
        self.conv2 = FermiGatedConv2d(
            c1,
            c2,
            kernel_size=3,
            padding=1,
            init_mu=init_mu,
            init_T=init_T,
            gate=self._build_gate(gate_factory, (c2, c1, 3, 3)),
        )
        self.conv3 = FermiGatedConv2d(
            c2,
            c3,
            kernel_size=3,
            padding=1,
            init_mu=init_mu,
            init_T=init_T,
            gate=self._build_gate(gate_factory, (c3, c2, 3, 3)),
        )

        # Step 3: build normalization and classifier head
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        self.norm3 = nn.BatchNorm2d(c3)
        self.dropout = nn.Dropout(dropout)
        self.classifier = FermiGatedLinear(
            c3,
            num_classes,
            init_mu=init_mu,
            init_T=init_T,
            gate=self._build_gate(gate_factory, (num_classes, c3)),
        )

    def _normalize_gate_factory(
        self,
        gate: Callable[..., BaseGate] | BaseGate | None,
    ) -> Callable[[tuple[int, ...]], BaseGate] | None:
        """Normalize gate input into a per-layer factory."""

        # Step 1: no custom gate path
        if gate is None:
            return None

        # Step 2: callable factory path
        if callable(gate) and not isinstance(gate, BaseGate):
            code_object = getattr(gate, "__code__", None)
            if code_object is not None and code_object.co_argcount == 0:
                return lambda _shape: gate()
            return lambda shape: gate(shape)

        # Step 3: gate instance path (FermiGate-compatible reconstruction)
        mode = getattr(gate, "mode", "elementwise")
        rank = getattr(gate, "rank", None)
        init_mu = getattr(gate, "_init_mu", 0.0)
        init_temperature = getattr(gate, "temperature", 1.0)
        annealer = getattr(gate, "annealer", "linear")
        gate_type = type(gate)
        return lambda shape: gate_type(
            shape=shape,
            mode=mode,
            rank=rank,
            init_mu=init_mu,
            init_temperature=init_temperature,
            annealer=annealer,
        )

    def _build_gate(
        self,
        gate_factory: Callable[[tuple[int, ...]], BaseGate] | None,
        shape: tuple[int, ...],
    ) -> BaseGate | None:
        """Build one gate instance per layer."""
        if gate_factory is None:
            return None
        return gate_factory(shape)

    def logits(self, x: torch.Tensor, return_masks: bool = False):
        masks: dict[str, torch.Tensor] = {}

        x, p = self.conv1(x)
        masks["conv1"] = p
        x = self.pool(torch.relu(self.norm1(x)))

        x, p = self.conv2(x)
        masks["conv2"] = p
        x = self.pool(torch.relu(self.norm2(x)))

        x, p = self.conv3(x)
        masks["conv3"] = p
        x = torch.relu(self.norm3(x))

        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        x = self.dropout(x)

        logits, p = self.classifier(x)
        masks["classifier"] = p

        if return_masks:
            return logits, masks
        return logits


class CNN(FermiConvClassifier):
    """User-facing CNN API compatible with ``input_channels`` naming."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        gate: Callable[..., BaseGate] | BaseGate | None = None,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        calibration: nn.Module | None = None,
        channels: tuple[int, int, int] = (32, 64, 96),
        dropout: float = 0.1,
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
            init_mu=init_mu,
            init_T=init_T,
            gate=gate,
        )

        # Step 1: compatibility fields consumed by user-level Experiment
        self.loss_fn = loss
        self.calibration = calibration
