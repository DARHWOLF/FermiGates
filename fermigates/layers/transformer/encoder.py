from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from fermigates.layers.linear_layers import FermiGatedLinear


class FermiTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with Fermi-gated feed-forward network."""

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
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear1 = FermiGatedLinear(d_model, dim_feedforward, init_mu=init_mu, init_T=init_T)
        self.linear2 = FermiGatedLinear(dim_feedforward, d_model, init_mu=init_mu, init_T=init_T)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        act = activation.lower()
        if act == "relu":
            self.activation = torch.relu
        elif act == "gelu":
            self.activation = torch.nn.functional.gelu
        elif act == "silu":
            self.activation = torch.nn.functional.silu
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
        )
        src = self.norm1(src + self.dropout(attn_out))

        ff, p1 = self.linear1(src)
        ff = self.activation(ff)
        ff = self.dropout(ff)
        ff, p2 = self.linear2(ff)
        out = self.norm2(src + self.dropout(ff))

        masks = {"ffn1": p1, "ffn2": p2}
        if need_weights:
            masks["attn_weights"] = attn_weights
        return out, masks
