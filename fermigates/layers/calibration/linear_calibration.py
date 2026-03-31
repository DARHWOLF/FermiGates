from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCalibration(nn.Module):
    """Linear correction module: ``y = X @ W + b``.

    ``W`` is stored with shape ``(d_in, d_out)``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        learnable: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        W = torch.zeros(d_in, d_out, device=device, dtype=dtype or torch.float32)
        b = torch.zeros(d_out, device=device, dtype=dtype or torch.float32)

        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

    def load_calibration(
        self,
        W_hat: torch.Tensor,
        b_hat: Optional[torch.Tensor] = None,
        to_param: bool = False,
    ) -> None:
        W_hat = W_hat.detach().to(dtype=self.W.dtype, device=self.W.device)
        if b_hat is None:
            b_hat = torch.zeros(W_hat.shape[1], dtype=self.b.dtype, device=self.b.device)
        else:
            b_hat = b_hat.detach().to(dtype=self.b.dtype, device=self.b.device)

        if W_hat.shape != self.W.shape:
            raise ValueError(f"Expected W_hat shape {tuple(self.W.shape)}, got {tuple(W_hat.shape)}")
        if b_hat.shape != self.b.shape:
            raise ValueError(f"Expected b_hat shape {tuple(self.b.shape)}, got {tuple(b_hat.shape)}")

        with torch.no_grad():
            self.W.copy_(W_hat)
            self.b.copy_(b_hat)

        if to_param:
            if not isinstance(self.W, nn.Parameter):
                self.W = nn.Parameter(self.W)
            if not isinstance(self.b, nn.Parameter):
                self.b = nn.Parameter(self.b)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.linear(X, self.W.t(), self.b)
