import torch
import torch.nn as nn

class MagnitudeMask(nn.Mask):
    """Deterministic magnitude mask derived from external `scores` or weights.


    Args:
    scores: a 1-D tensor-like object containing magnitudes or importance scores.
    inverted: if True, smaller scores correspond to keep (rare). Default False.
    """


    def __init__(self, scores: torch.Tensor, inverted: bool = False):
        super().__init__()
        assert scores.ndim == 1, "scores must be 1-D"
        self.register_buffer("scores", scores.clone().detach())
        self.inverted = inverted


    def forward(self, sample: bool = False) -> torch.Tensor:
        # deterministic mask in [0,1] (1 for kept entries, 0 for pruned) — but we keep the raw scores
        if self.inverted:
            s = -self.scores
        else:
            s = self.scores
        # map scores to [0,1] via min-max (keeps relative ordering). Useful for soft usage.
        smin, smax = s.min(), s.max()
        if smax - smin < 1e-12:
            return torch.ones_like(s)
        return (s - smin) / (smax - smin)


    def l0_penalty(self) -> torch.Tensor:
        return self.forward().sum()