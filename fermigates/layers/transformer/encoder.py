from __future__ import annotations

import copy
import inspect

import torch
import torch.nn as nn

from fermigates.layers.attention import Attention
from fermigates.layers.linear import Linear


class FermiTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer built from modular Attention and Linear blocks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate=None,
    ) -> None:
        super().__init__()
        del batch_first
        del init_mu
        del init_T

        # Step 1: Build self-attention and feed-forward layers
        attn_gate = self._build_gate(gate, (1, nhead, 1, 1))
        ffn1_gate = self._build_gate(gate, (dim_feedforward, d_model))
        ffn2_gate = self._build_gate(gate, (d_model, dim_feedforward))

        self.self_attn = Attention(
            embed_dim=d_model,
            num_heads=nhead,
            input_gate=self._attention_input_gate(attn_gate),
            output_gate=self._attention_output_gate(attn_gate),
        )
        self.linear1 = Linear(
            d_model,
            dim_feedforward,
            **self._linear_gate_kwargs(ffn1_gate),
        )
        self.linear2 = Linear(
            dim_feedforward,
            d_model,
            **self._linear_gate_kwargs(ffn2_gate),
        )

        # Step 2: Build normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Step 3: Resolve activation function
        act = activation.lower()
        if act == "relu":
            self.activation = torch.relu
        elif act == "gelu":
            self.activation = torch.nn.functional.gelu
        elif act == "silu":
            self.activation = torch.nn.functional.silu
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

    def _build_gate(self, gate, shape: tuple[int, ...]):
        """Build one gate instance for a target layer shape."""

        # Step 1: keep gating disabled when no gate is configured
        if gate is None:
            return None

        # Step 2: clone module instances to avoid shared states
        if isinstance(gate, nn.Module):
            return copy.deepcopy(gate)

        # Step 3: call gate factories with shape-aware fallback
        signature = inspect.signature(gate)
        if len(signature.parameters) == 0:
            return gate()
        return gate(shape)

    def _linear_gate_kwargs(self, gate_module):
        """Map gate mode to linear-layer injection points."""

        # Step 1: disable all linear gates when gate module is missing
        if gate_module is None:
            return {
                "weight_gate": None,
                "input_gate": None,
                "output_gate": None,
            }

        # Step 2: map gate mode to one explicit injection point
        mode = getattr(gate_module, "mode", "").lower()
        if mode == "weight":
            return {"weight_gate": gate_module, "input_gate": None, "output_gate": None}
        if mode in {"feature", "token", "input"}:
            return {"weight_gate": None, "input_gate": gate_module, "output_gate": None}
        if mode == "head":
            return {"weight_gate": None, "input_gate": None, "output_gate": None}
        return {"weight_gate": None, "input_gate": None, "output_gate": gate_module}

    def _attention_input_gate(self, gate_module):
        """Return attention input gate when mode indicates token/feature gating."""
        if gate_module is None:
            return None
        mode = getattr(gate_module, "mode", "").lower()
        if mode in {"feature", "token", "input"}:
            return gate_module
        return None

    def _attention_output_gate(self, gate_module):
        """Return attention output gate for head/output gating modes."""
        if gate_module is None:
            return None
        mode = getattr(gate_module, "mode", "").lower()
        if mode in {"head", "neuron", "channel", "output"}:
            return gate_module
        return None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        del src_mask
        del src_key_padding_mask
        del need_weights

        # Step 1: Self-attention sub-layer
        attn_out, _ = self.self_attn(src)
        src = self.norm1(src + self.dropout(attn_out))

        # Step 2: Feed-forward sub-layer with optional gate captures
        ff, p1 = self.linear1(src)
        ff = self.activation(ff)
        ff = self.dropout(ff)
        ff, p2 = self.linear2(ff)
        out = self.norm2(src + self.dropout(ff))

        # Step 3: Return output with FFN gate probabilities
        return out, {"ffn1": p1, "ffn2": p2}
