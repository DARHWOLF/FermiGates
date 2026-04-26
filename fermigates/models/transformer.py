from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from fermigates.base import BaseFermiClassifier
from fermigates.gates import BaseGate
from fermigates.layers.linear import FermiGatedLinear
from fermigates.layers.transformer import FermiTransformerEncoderLayer


class FermiTransformerClassifier(BaseFermiClassifier):
    """Transformer classifier with gated FFN blocks and gated output head."""

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
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: Callable[..., BaseGate] | BaseGate | None = None,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        # Step 1: store high-level dimensions
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)

        # Step 2: build token + positional embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(dropout)

        # Step 3: build encoder stack with optional custom gates
        gate_factory = self._normalize_gate_factory(gate)
        self.encoder_layers = nn.ModuleList(
            [
                FermiTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    init_mu=init_mu,
                    init_T=init_T,
                    gate=gate_factory,
                )
                for _ in range(num_layers)
            ]
        )

        # Step 4: build gated classifier head
        self.head = FermiGatedLinear(
            self.d_model,
            num_classes,
            init_mu=init_mu,
            init_T=init_T,
            gate=self._build_gate(gate_factory, (num_classes, self.d_model)),
        )

    def _normalize_gate_factory(
        self,
        gate: Callable[..., BaseGate] | BaseGate | None,
    ) -> Callable[[tuple[int, ...]], BaseGate] | None:
        """Normalize gate input into a per-layer factory."""

        # Step 1: no custom gate path
        if gate is None:
            return None

        # Step 2: callable factory path
        if callable(gate) and not isinstance(gate, BaseGate):
            code_object = getattr(gate, "__code__", None)
            if code_object is not None and code_object.co_argcount == 0:
                return lambda _shape: gate()
            return lambda shape: gate(shape)

        # Step 3: gate instance path (FermiGate-compatible reconstruction)
        mode = getattr(gate, "mode", "elementwise")
        rank = getattr(gate, "rank", None)
        init_mu = getattr(gate, "_init_mu", 0.0)
        init_temperature = getattr(gate, "temperature", 1.0)
        annealer = getattr(gate, "annealer", "linear")
        gate_type = type(gate)
        return lambda shape: gate_type(
            shape=shape,
            mode=mode,
            rank=rank,
            init_mu=init_mu,
            init_temperature=init_temperature,
            annealer=annealer,
        )

    def _build_gate(
        self,
        gate_factory: Callable[[tuple[int, ...]], BaseGate] | None,
        shape: tuple[int, ...],
    ) -> BaseGate | None:
        """Build one gate instance from a normalized factory."""
        if gate_factory is None:
            return None
        return gate_factory(shape)

    def logits(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_masks: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, seq_len).")
        if tokens.size(1) > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

        # Step 1: embed tokens and positions
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x)

        # Step 2: run encoder layers and collect masks
        encoder_masks: list[dict[str, torch.Tensor]] = []
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for layer in self.encoder_layers:
            x, masks = layer(x, src_key_padding_mask=key_padding_mask)
            encoder_masks.append(masks)

        # Step 3: pool sequence and compute classifier logits
        pooled = x.mean(dim=1)
        logits, head_mask = self.head(pooled)

        if return_masks:
            return logits, {"encoder": encoder_masks, "head": head_mask}
        return logits


class Transformer(FermiTransformerClassifier):
    """User-facing Transformer API compatible with legacy argument names."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        gate: Callable[..., BaseGate] | BaseGate | None = None,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        calibration: nn.Module | None = None,
        num_classes: int = 2,
        max_seq_len: int = 64,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_seq_len=max_seq_len,
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            init_mu=init_mu,
            init_T=init_T,
            gate=gate,
        )

        # Step 1: compatibility fields consumed by user-level Experiment
        self.loss_fn = loss
        self.calibration = calibration
