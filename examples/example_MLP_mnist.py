"""Train and compare vanilla vs gated MLP on torchvision MNIST.

This example uses the `Experiment` API and the dataset registry.
MNIST is loaded through torchvision and downloaded to `./data` when missing.
"""

# Step 0: Imports
from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.calibration.linear_calibration import LinearCalibration
from fermigates.datasets import get_dataloader
from fermigates.experiments import Experiment
from fermigates.gates import FermiGate
from fermigates.losses import fermiloss
from fermigates.models import MLP


def build_model(use_gate: bool) -> MLP:
    """Build an MNIST MLP model.

    Parameters
    ----------
    use_gate : bool
        Whether to attach Fermi neuron gates.

    Returns
    -------
    MLP
        Configured model instance.
    """

    # Step 1: Resolve optional gate factory.
    gate = None
    if use_gate:
        gate = lambda: FermiGate(
            mode="neuron",
            annealer="linear",
            init_mu=-2.0,
            init_temperature=2.0,
        )

    # Step 2: Build and return model.
    return MLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        gate=gate,
        loss=fermiloss,
        calibration=LinearCalibration(),
    )


def activation_gate_sparsity(model: MLP, device: torch.device) -> float:
    """Estimate activation-gate sparsity on MNIST test split.

    Parameters
    ----------
    model : MLP
        Trained MLP model.
    device : torch.device
        Device used for forward passes.

    Returns
    -------
    float
        Overall sparsity value in ``[0.0, 1.0]``.
    """

    # Step 1: Build test loader and initialize counters.
    test_loader = get_dataloader(
        name="mnist",
        split="test",
        batch_size=256,
        shuffle=False,
    )
    active = 0
    total = 0

    # Step 2: Collect gate activations from real batches.
    model.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            _logits, gate_outputs = model(x_batch, return_gate_outputs=True)
            for gate_probs in gate_outputs:
                if gate_probs is None:
                    continue
                active += int((gate_probs > 0.5).sum().item())
                total += int(gate_probs.numel())

    # Step 3: Convert active fraction to sparsity.
    if total == 0:
        return 0.0
    return 1.0 - (float(active) / float(total))


def main() -> None:
    """Run vanilla vs gated MLP training on MNIST."""

    # Step 1: Fix seed and build models.
    torch.manual_seed(7)
    vanilla_model = build_model(use_gate=False)
    torch.manual_seed(7)
    gated_model = build_model(use_gate=True)

    # Step 2: Create experiments.
    vanilla_exp = Experiment(
        model=vanilla_model,
        dataset="mnist",
        epochs=3,
        learning_rate=1e-3,
    )
    gated_exp = Experiment(
        model=gated_model,
        dataset="mnist",
        epochs=8,
        learning_rate=1e-3,
    )

    # Step 3: Train models.
    print("\n===== Training Vanilla MLP =====")
    vanilla_exp.train()
    print("\n===== Training Gated MLP =====")
    gated_exp.train()

    # Step 4: Evaluate and print accuracy.
    vanilla_acc = vanilla_exp.accuracy()
    gated_acc = gated_exp.accuracy()
    print("\n===== Evaluation =====")
    print(f"\nVanilla Accuracy: {vanilla_acc:.4f}")
    print(f"Gated Accuracy:   {gated_acc:.4f}")

    # Step 5: Report gate sparsity metrics.
    vanilla_weight_sparsity = 0.0
    gated_weight_sparsity = 0.0
    if hasattr(vanilla_model, "compute_sparsity"):
        kept, total, _ = vanilla_model.compute_sparsity(threshold=0.5)
        if total > 0:
            vanilla_weight_sparsity = 1.0 - (float(kept) / float(total))
    if hasattr(gated_model, "compute_sparsity"):
        kept, total, _ = gated_model.compute_sparsity(threshold=0.5)
        if total > 0:
            gated_weight_sparsity = 1.0 - (float(kept) / float(total))

    vanilla_activation_sparsity = activation_gate_sparsity(vanilla_model, vanilla_exp.device)
    gated_activation_sparsity = activation_gate_sparsity(gated_model, gated_exp.device)

    print(f"\nVanilla Weight-Gate Sparsity: {vanilla_weight_sparsity:.4f}")
    print(f"Gated   Weight-Gate Sparsity: {gated_weight_sparsity:.4f}")
    print(f"Vanilla Activation-Gate Sparsity: {vanilla_activation_sparsity:.4f}")
    print(f"Gated   Activation-Gate Sparsity: {gated_activation_sparsity:.4f}")


if __name__ == "__main__":
    main()
