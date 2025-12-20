import torch
import torch.nn as nn
import torch.nn.functional as F

class FermiMask(nn.Module):
    """
    Differentiable Fermi mask.
    - shape: shape of weight tensor (e.g. (out, in))
    - init_mu: initial pivot; can be scalar or a tensor broadcastable to `shape`
    - init_T: initial temperature (buffer)
    """
    def __init__(self, shape: torch.Size, init_mu: float = 0.0, init_T: float = 1.0):
        super().__init__()
        # mu has same shape as weights by default (per-weight). For per-neuron pivot,
        # user can reshape or use mean/aggregate.
        self.mu = nn.Parameter(torch.full(shape, float(init_mu)))
        self.register_buffer('T', torch.tensor(float(init_T)))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # returns retention probability P(w) in (0,1), same shape as w
        return 1.0 / (torch.exp((w.abs() - self.mu) / (self.T + 1e-12)) + 1.0)

    def set_temperature(self, T_new: float):
        self.T.fill_(float(T_new))
