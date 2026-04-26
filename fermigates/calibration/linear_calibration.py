from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCalibration(nn.Module):
    """Linear correction module: ``y = X @ W + b``.

    ``W`` is stored with shape ``(d_in, d_out)``.
    """

    def __init__(
        self,
        d_in: int | None = None,
        d_out: int | None = None,
        learnable: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.learnable = bool(learnable)
        self._dtype = dtype or torch.float32
        self._device = device
        self._initialized = False
        self._d_in = d_in
        self._d_out = d_out

        # Step 1: initialize immediately when dimensions are provided
        if d_in is not None and d_out is not None:
            self._initialize_parameters(int(d_in), int(d_out), self._device, self._dtype)

    def _initialize_parameters(
        self,
        d_in: int,
        d_out: int,
        device,
        dtype,
    ) -> None:
        """Initialize calibration weights and bias tensors."""

        # Step 1: create default affine transform parameters
        W = torch.zeros(d_in, d_out, device=device, dtype=dtype)
        if d_in == d_out:
            W += torch.eye(d_in, device=device, dtype=dtype)
        b = torch.zeros(d_out, device=device, dtype=dtype)

        # Step 2: register learnable parameters or buffers
        if self.learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

        # Step 3: mark initialization state
        self._initialized = True
        self._d_in = d_in
        self._d_out = d_out

    def load_calibration(
        self,
        W_hat: torch.Tensor,
        b_hat: torch.Tensor | None = None,
        to_param: bool = False,
    ) -> None:
        # Step 1: lazily initialize if constructor dimensions were omitted
        if not self._initialized:
            self._initialize_parameters(
                int(W_hat.shape[0]),
                int(W_hat.shape[1]),
                W_hat.device,
                W_hat.dtype,
            )

        # Step 2: normalize provided tensors to internal dtype/device
        W_hat = W_hat.detach().to(dtype=self.W.dtype, device=self.W.device)
        if b_hat is None:
            b_hat = torch.zeros(W_hat.shape[1], dtype=self.b.dtype, device=self.b.device)
        else:
            b_hat = b_hat.detach().to(dtype=self.b.dtype, device=self.b.device)

        # Step 3: validate tensor shapes
        if W_hat.shape != self.W.shape:
            raise ValueError(
                f"Expected W_hat shape {tuple(self.W.shape)}, got {tuple(W_hat.shape)}"
            )
        if b_hat.shape != self.b.shape:
            raise ValueError(
                f"Expected b_hat shape {tuple(self.b.shape)}, got {tuple(b_hat.shape)}"
            )

        # Step 4: copy values into calibration tensors
        with torch.no_grad():
            self.W.copy_(W_hat)
            self.b.copy_(b_hat)

        # Step 5: optional conversion to learnable parameters
        if to_param:
            if not isinstance(self.W, nn.Parameter):
                self.W = nn.Parameter(self.W)
            if not isinstance(self.b, nn.Parameter):
                self.b = nn.Parameter(self.b)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            inferred_dim = int(X.shape[-1])
            self._initialize_parameters(inferred_dim, inferred_dim, X.device, X.dtype)
        return F.linear(X, self.W.t(), self.b)
