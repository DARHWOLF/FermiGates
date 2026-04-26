import torch
import torch.nn.functional as F

from .fermi import (
    FermiLossBreakdown,
    binary_entropy_loss,
    budget_penalty_loss,
    consistency_loss,
    fermi_free_energy_loss,
    fermi_informed_loss,
    sparsity_l1_loss,
)
from .regularizers import (
    group_sparsity_l21_loss,
    hoyer_sparsity_loss,
    hoyer_sparsity_score,
    kl_to_bernoulli_prior_loss,
)


def fermiloss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compatibility loss alias for user-facing API tests.

    Parameters
    ----------
    logits : torch.Tensor
        Model logits.
    targets : torch.Tensor
        Integer class targets.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """

    return F.cross_entropy(logits, targets)


__all__ = [
    "FermiLossBreakdown",
    "fermi_informed_loss",
    "fermi_free_energy_loss",
    "binary_entropy_loss",
    "sparsity_l1_loss",
    "budget_penalty_loss",
    "consistency_loss",
    "kl_to_bernoulli_prior_loss",
    "group_sparsity_l21_loss",
    "hoyer_sparsity_loss",
    "hoyer_sparsity_score",
    "fermiloss",
]
