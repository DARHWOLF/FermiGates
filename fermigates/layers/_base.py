from __future__ import annotations

import torch
import torch.nn as nn


class BaseLayer(nn.Module):
    """Base layer with explicit pluggable gating injection points.

    Parameters
    ----------
    weight_gate : nn.Module or None, optional
        Optional gate applied to the layer weight tensor.
    input_gate : nn.Module or None, optional
        Optional gate applied to the layer input tensor.
    output_gate : nn.Module or None, optional
        Optional gate applied to the layer output tensor.
    """

    def __init__(
        self,
        weight_gate: nn.Module | None = None,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Step 1: Store optional gate modules exactly as provided
        self.weight_gate = weight_gate
        self.input_gate = input_gate
        self.output_gate = output_gate

    def _core_forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """Core layer computation implemented by subclasses."""
        raise NotImplementedError

    def _get_weight(self) -> torch.Tensor | None:
        """Return layer weight tensor when available."""
        return None

    def _apply_input_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply input gate and return gated tensor with gate probabilities."""

        # Step 1: Preserve input when gate is absent
        if self.input_gate is None:
            return x, None

        # Step 2: Compute input gate probabilities and apply gating
        probs = self.input_gate(x)
        gated = x * probs
        return gated, probs

    def _apply_weight_gate(
        self,
        weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply weight gate and return gated weight with gate probabilities."""

        # Step 1: Preserve weight when no gating is applicable
        if weight is None or self.weight_gate is None:
            return weight, None

        # Step 2: Compute weight gate probabilities and apply gating
        probs = self.weight_gate(weight)
        gated_weight = weight * probs
        return gated_weight, probs

    def _apply_output_gate(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply output gate and return gated output with gate probabilities."""

        # Step 1: Preserve output when gate is absent
        if self.output_gate is None:
            return out, None

        # Step 2: Compute output gate probabilities and apply gating
        probs = self.output_gate(out)
        gated = out * probs
        return gated, probs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the unified pipeline `input -> core_op -> output`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Layer output and the latest gate probabilities applied in the
            pipeline. Returns ``None`` when no gate is used.
        """

        # Step 1: Optionally gate the input
        x, input_probs = self._apply_input_gate(x)

        # Step 2: Optionally gate the layer weights
        weight = self._get_weight()
        weight, weight_probs = self._apply_weight_gate(weight)

        # Step 3: Run core computation
        out = self._core_forward(x, weight)

        # Step 4: Optionally gate the output
        out, output_probs = self._apply_output_gate(out)

        # Step 5: Return output with the most recently applied gate probabilities
        if output_probs is not None:
            return out, output_probs
        if weight_probs is not None:
            return out, weight_probs
        if input_probs is not None:
            return out, input_probs
        return out, None
