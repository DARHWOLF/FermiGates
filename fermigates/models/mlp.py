from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from fermigates.base import BaseFermiClassifier
from fermigates.layers.linear_layers import FermiGatedLinear


def _activation(name: str):
    key = name.lower()
    if key == "relu":
        return torch.relu
    if key == "tanh":
        return torch.tanh
    if key == "silu":
        return torch.nn.functional.silu
    if key == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unsupported activation '{name}'.")


class FermiMLPClassifier(BaseFermiClassifier):
    """Fully-connected classifier built from stacked ``FermiGatedLinear`` layers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
        activation: str = "gelu",
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        dims = [int(input_dim), *[int(v) for v in hidden_dims], int(num_classes)]
        if any(v <= 0 for v in dims):
            raise ValueError("All dimensions must be positive.")

        self.layers = nn.ModuleList(
            [
                FermiGatedLinear(
                    dims[i],
                    dims[i + 1],
                    init_mu=init_mu,
                    init_T=init_T,
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.act = _activation(activation)

    def logits(self, x: torch.Tensor, return_masks: bool = False):
        if x.ndim > 2:
            x = x.flatten(1)

        masks: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            x, p = layer(x)
            masks.append(p)
            if idx < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        if return_masks:
            return x, masks
        return x
