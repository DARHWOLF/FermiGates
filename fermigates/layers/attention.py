from __future__ import annotations

import torch
import torch.nn as nn

from fermigates.layers._base import BaseLayer


class Attention(BaseLayer):
    """Minimal multi-head attention layer with pluggable input/output gates.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float, default=0.0
        Attention dropout probability.
    input_gate : nn.Module or None, optional
        Optional token-level input gate.
    output_gate : nn.Module or None, optional
        Optional output gate. Supports head-level gating when gate mode is
        configured for attention heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        input_gate: nn.Module | None = None,
        output_gate: nn.Module | None = None,
    ) -> None:
        super().__init__(
            weight_gate=None,
            input_gate=input_gate,
            output_gate=output_gate,
        )

        # Step 1: Build underlying multi-head attention module
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

    def _core_forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        del weight
        out, _ = self.attn(x, x, x)
        return out

    def _apply_output_gate(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply output gate with head-aware behavior when configured."""

        # Step 1: Keep output untouched when no gate is supplied
        if self.output_gate is None:
            return out, None

        # Step 2: Handle head-level gating on explicit head axis
        gate_mode = getattr(self.output_gate, "mode", None)
        if gate_mode == "head" and out.ndim == 3 and out.shape[-1] == self.embed_dim:
            batch_size, seq_len, _ = out.shape
            reshaped = out.view(batch_size, seq_len, self.num_heads, self.head_dim)
            head_first = reshaped.permute(0, 2, 1, 3).contiguous()
            probs = self.output_gate(head_first)
            gated_heads = head_first * probs
            restored = gated_heads.permute(0, 2, 1, 3).contiguous()
            return restored.view(batch_size, seq_len, self.embed_dim), probs

        # Step 3: Fallback to generic output gating behavior
        probs = self.output_gate(out)
        return out * probs, probs
