from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from fermigates.layers.linear import Linear


def _resolve_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
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


class FermiMLPBlock(nn.Module):
    """Two-layer feed-forward block with Fermi-gated linears."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__()
        del init_mu
        del init_T

        # Step 1: Build stacked linear layers without implicit gates
        self.fc1 = Linear(d_in, d_hidden)
        self.fc2 = Linear(d_hidden, d_out)
        self.activation = _resolve_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, p1 = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x, p2 = self.fc2(x)
        return x, {"fc1": p1, "fc2": p2}


class FermiResidualBlock(nn.Module):
    """Residual wrapper around ``FermiMLPBlock`` for stable deep stacks."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.block = FermiMLPBlock(
            d_in=dim,
            d_hidden=hidden,
            d_out=dim,
            dropout=dropout,
            activation=activation,
            init_mu=init_mu,
            init_T=init_T,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, masks = self.block(self.norm(x))
        return x + y, masks
