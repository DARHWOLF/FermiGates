from __future__ import annotations

from typing import Optional

import torch

from fermigates.gates.structured import MagnitudeGate


class MagnitudeMask(MagnitudeGate):
    """Backward-compatible alias for :class:`fermigates.gates.MagnitudeGate`."""

    def __init__(self, scores: torch.Tensor, inverted: bool = False) -> None:
        if scores.ndim != 1:
            raise ValueError("scores must be 1-D for MagnitudeMask.")
        super().__init__(shape=scores.shape, scores=scores, inverted=inverted)

    def forward(self, sample: Optional[bool] = None) -> torch.Tensor:
        del sample
        return self.probabilities()
