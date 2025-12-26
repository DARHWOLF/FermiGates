import torch
import torch.nn as nn


class GompertzMask(nn.Module):
    """
    Deterministic Gompertz-based mask for pruning and gating.

    Mask definition:
        m(s) = exp(-alpha * exp(-beta * s))

    Parameters
    ----------
    size : int or tuple
        Number of mask elements (e.g., channels, neurons).
    alpha : float, default=2.0
        Controls pruning aggressiveness.
    beta : float, default=1.0
        Controls transition sharpness.
    learn_alpha : bool, default=False
        Whether alpha should be learnable.
    init_score : float, default=0.0
        Initial value of the score parameter.
    """

    def __init__(
        self,
        size,
        alpha: float = 2.0,
        beta: float = 1.0,
        learn_alpha: bool = False,
        init_score: float = 0.0,
    ):
        super().__init__()

        self.s = nn.Parameter(torch.full((size,), init_score))

        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

        self.beta = beta

    def forward(self):
        """
        Returns
        -------
        mask : torch.Tensor
            Values in (0, 1], same shape as `s`
        """
        return torch.exp(-self.alpha * torch.exp(-self.beta * self.s))

    def l0_penalty(self):
        """
        Expected L0-style penalty (sum of survival probabilities).
        """
        return self.forward().sum()
