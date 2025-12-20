import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LinearCalibration(nn.Module):
    """
    Linear calibration module: output = X @ W + b
    By default registers W and b as buffers (frozen). Pass learnable=True to make them Parameters.
    W shape: (d_in, d_out)
    b shape: (d_out,)
    """
    def __init__(self, d_in: int, d_out: int, learnable: bool = False, device=None, dtype=None):
        super().__init__()
        W = torch.zeros(d_in, d_out, device=device, dtype=dtype)
        b = torch.zeros(d_out, device=device, dtype=dtype)
        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer('W', W)
            self.register_buffer('b', b)

    def load_calibration(self, W_hat: torch.Tensor, b_hat: Optional[torch.Tensor] = None, to_param: bool = False):
        """
        Load computed calibration parameters (W_hat (d_in,d_out), optional b_hat (d_out,)).
        If to_param True, convert them into nn.Parameter (trainable).
        """
        W_hat = W_hat.detach().clone()
        if b_hat is None:
            b_hat = torch.zeros(W_hat.shape[1], device=W_hat.device, dtype=W_hat.dtype)
        b_hat = b_hat.detach().clone()

        # ensure shapes
        if hasattr(self, 'W') and isinstance(self.W, torch.Tensor):
            # replace buffer
            self.W = W_hat.to(self.W.device)
        else:
            self.W.data.copy_(W_hat.to(self.W.device))

        if hasattr(self, 'b') and isinstance(self.b, torch.Tensor):
            self.b = b_hat.to(self.b.device)
        else:
            self.b.data.copy_(b_hat.to(self.b.device))

        if to_param:
            if not isinstance(self.W, nn.Parameter):
                self.W = nn.Parameter(self.W)
            if not isinstance(self.b, nn.Parameter):
                self.b = nn.Parameter(self.b)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, d_in); returns (batch, d_out)
        # Use F.linear which expects weight shape (out_features, in_features)
        return F.linear(X, self.W.t(), self.b)
