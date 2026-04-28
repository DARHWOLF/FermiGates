"""Minimal end-to-end training demo for FermiMLPClassifier."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.models import FermiMLPClassifier


def make_dataset(n: int = 512, d_in: int = 32, num_classes: int = 4):
    x = torch.randn(n, d_in)
    w = torch.randn(d_in, num_classes)
    logits = x @ w
    y = logits.argmax(dim=1)
    return x, y


def main() -> None:
    torch.manual_seed(7)
    x, y = make_dataset()

    model = FermiMLPClassifier(input_dim=32, hidden_dims=(64, 32), num_classes=4, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        model.train()
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        model.set_temperature(max(0.2, 1.0 - 0.08 * epoch))
        kept, total, frac = model.compute_sparsity(threshold=0.3)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean().item()

        print(
            f"epoch={epoch:02d} loss={loss.item():.4f} acc={acc:.3f} "
            f"kept={kept}/{total} ({frac:.2%})"
        )


if __name__ == "__main__":
    main()
