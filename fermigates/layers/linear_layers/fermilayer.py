import torch
import torch.nn as nn  
import torch.nn.functional as F
from typing import Tuple
from fermigates.masks.fermimask import FermiMask

class FermiGatedLinear(nn.Module):
    """
    Linear layer with a Fermi mask over its weights.
    - in_features, out_features: same as nn.Linear
    - init_mu/init_T forwarded to the FermiMask (mask shape = weight.shape)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_mu: float = 0.0, init_T: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mask = FermiMask(self.linear.weight.shape, init_mu=init_mu, init_T=init_T)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - output: Linear(x) using masked weights
          - P: mask probabilities (same shape as weight)
        """
        P = self.mask(self.linear.weight)
        w_tilde = P * self.linear.weight
        out = F.linear(x, w_tilde, self.linear.bias)
        return out, P

    def set_temperature(self, T_new: float):
        self.mask.set_temperature(T_new)

    def initialize_mu_from_weight_percentile(self, percentile: float = 0.5, per_neuron: bool = False):
        """
        Initialize mu from percentiles of |w|. If per_neuron True, compute percentile per output neuron (dim=1 of weight).
        """
        with torch.no_grad():
            W = self.linear.weight.data.abs()
            if per_neuron:
                # W shape [out, in, ...] -> flatten trailing dims and compute percentile along in-dim
                out_dim = W.shape[0]
                flat = W.view(out_dim, -1)
                kth = int(max(0, min(flat.shape[1] - 1, round(percentile * flat.shape[1]))))
                # approximate percentile using topk
                vals, _ = flat.sort(dim=1)
                mu_vals = vals[:, kth]
                # mu has shape like weight; broadcast per-row
                mu_full = mu_vals.view(-1, 1).expand_as(W)
                self.mask.mu.data.copy_(mu_full)
            else:
                flat = W.view(-1)
                kth = int(max(0, min(flat.numel() - 1, round(percentile * flat.numel()))))
                vals, _ = flat.sort()
                mu_val = vals[kth].item()
                self.mask.mu.data.fill_(mu_val)

