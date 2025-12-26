import torch
import torch.nn as nn

class GroupLassoMask(nn.Module):
    """Group Lasso style mask: groups share a single gate. Useful for channel or head pruning.


    Args:
    groups: number of groups
    group_size: size of each group (if uniform)
    init: initial gate logit
    """


    def __init__(self, groups: int, group_size: int = 1, init: float = 0.0):
        super().__init__()
        self.groups = groups
        self.group_size = group_size
        # gate for each group
        self.gate = nn.Parameter(torch.full((groups,), float(init)))


    def forward(self, sample: bool = False) -> torch.Tensor:
        # returns a repeated mask for each element in a group
        p = torch.sigmoid(self.gate)
        out = p.repeat_interleave(self.group_size)
        if sample:
            return torch.bernoulli(out)
        return out

    def l0_penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.gate).sum()