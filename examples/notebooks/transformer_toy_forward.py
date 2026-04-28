"""Toy forward pass for FermiTransformerClassifier."""

from pathlib import Path
import sys

import torch

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.models import FermiTransformerClassifier


def main() -> None:
    torch.manual_seed(11)

    model = FermiTransformerClassifier(
        vocab_size=500,
        num_classes=3,
        max_seq_len=64,
        d_model=48,
        nhead=4,
        num_layers=2,
        dim_feedforward=96,
    )

    tokens = torch.randint(0, 500, (8, 20))
    attention_mask = torch.ones_like(tokens)

    logits, masks = model.logits(tokens, attention_mask=attention_mask, return_masks=True)
    kept, total, frac = model.compute_sparsity(threshold=0.3)

    print("logits shape:", tuple(logits.shape))
    print("encoder layers:", len(masks["encoder"]))
    print(f"sparsity kept={kept}/{total} ({frac:.2%})")


if __name__ == "__main__":
    main()
