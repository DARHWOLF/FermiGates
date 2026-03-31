from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from fermigates.base import BaseFermiClassifier
from fermigates.layers.linear_layers import FermiGatedLinear
from fermigates.layers.transformer import FermiTransformerEncoderLayer


class FermiTransformerClassifier(BaseFermiClassifier):
    """Token classifier with Transformer encoder + Fermi-gated MLP sublayers."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_seq_len: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.max_seq_len = int(max_seq_len)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                FermiTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    init_mu=init_mu,
                    init_T=init_T,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = FermiGatedLinear(d_model, num_classes, init_mu=init_mu, init_T=init_T)

    def logits(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_masks: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must be of shape (batch, seq_len).")

        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={self.max_seq_len}."
            )

        x = self.token_embed(tokens) + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        layer_masks: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            x, masks = layer(x, src_key_padding_mask=src_key_padding_mask)
            layer_masks.append(masks)

        x = self.norm(x)

        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.to(dtype=x.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * weights.unsqueeze(-1)).sum(dim=1) / denom

        logits, head_mask = self.classifier(pooled)

        if return_masks:
            return logits, {"encoder": layer_masks, "head": head_mask}
        return logits
