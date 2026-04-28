"""Synthetic image demo for FermiConvClassifier."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.models import FermiConvClassifier


def main() -> None:
    torch.manual_seed(3)
    batch_size = 64
    num_classes = 10

    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, num_classes, (batch_size,))

    model = FermiConvClassifier(in_channels=3, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step in range(1, 6):
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        model.set_temperature(max(0.3, 1.0 - 0.12 * step))
        kept, total, frac = model.compute_sparsity(threshold=0.3)

        print(f"step={step} loss={loss.item():.4f} kept={kept}/{total} ({frac:.2%})")


if __name__ == "__main__":
    main()
