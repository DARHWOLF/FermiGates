"""Train and compare vanilla vs gated MLP on torchvision MNIST.

This example uses the `Experiment` API and the dataset registry.
MNIST is loaded through torchvision and downloaded to `./data` when missing.
"""

# Step 0: Imports
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.calibration.linear_calibration import LinearCalibration
from fermigates.experiments import Experiment
from fermigates.gates import FermiGate
from fermigates.losses import fermiloss
from fermigates.models import MLP


def build_model(variant: Literal["vanilla", "neuron", "weight"]) -> MLP:
    """Build an MNIST MLP model.

    Parameters
    ----------
    variant : {"vanilla", "neuron", "weight"}
        Model variant that controls gate placement.

    Returns
    -------
    MLP
        Configured model instance.
    """

    # Step 1: Validate and resolve optional gate factory.
    if variant not in {"vanilla", "neuron", "weight"}:
        raise ValueError("variant must be one of {'vanilla', 'neuron', 'weight'}.")

    gate = None
    if variant == "neuron":
        def gate() -> FermiGate:
            return FermiGate(
                mode="neuron",
                annealer="linear",
                init_mu=0.0,
                init_temperature=0.1,
            )
    if variant == "weight":
        def gate() -> FermiGate:
            return FermiGate(
                mode="weight",
                annealer="linear",
                init_mu=0.5,
                init_temperature=1,
            )

    # Step 2: Build and return model.
    return MLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        gate=gate,
        loss=fermiloss,
        calibration=LinearCalibration(d_in=10, d_out=10, learnable=False),
    )


def main() -> None:
    """Run vanilla, neuron-gated, and weight-gated MLP training on MNIST."""

    # Step 1: Define model variants and build deterministic model instances.
    model_variants: list[tuple[str, Literal["vanilla", "neuron", "weight"]]] = [
        ("Vanilla", "vanilla"),
        ("Neuron-Gated", "neuron"),
        ("Weight-Gated", "weight"),
    ]
    models: dict[str, MLP] = {}
    for label, variant in model_variants:
        torch.manual_seed(7)
        models[label] = build_model(variant=variant)

    # Step 2: Build experiments with shared training configuration.
    experiments: dict[str, Experiment] = {}
    for label, _ in model_variants:
        experiments[label] = Experiment(
            model=models[label],
            dataset="mnist",
            epochs=10,
            learning_rate=1e-3,
        )

    # Step 3: Train all models sequentially with explicit calibration policy.
    print("\n===== Training Vanilla MLP =====")
    experiments["Vanilla"].train()
    print("\n===== Training Neuron-Gated MLP =====")
    experiments["Neuron-Gated"].train(calibrate_after_first_epoch=True, ridge_lambda=1e-3)
    print("\n===== Training Weight-Gated MLP =====")
    experiments["Weight-Gated"].train(calibrate_after_first_epoch=True, ridge_lambda=1e-3)

    # Step 4: Evaluate and print accuracy for each model.
    accuracies: dict[str, float] = {}
    for label, _ in model_variants:
        accuracies[label] = experiments[label].accuracy()
    print("\n===== Evaluation =====")
    print(f"\nVanilla Accuracy:       {accuracies['Vanilla']:.4f}")
    print(f"Neuron-Gated Accuracy:  {accuracies['Neuron-Gated']:.4f}")
    print(f"Weight-Gated Accuracy:  {accuracies['Weight-Gated']:.4f}")

    # Step 5: Collect weight and activation sparsity metrics from Experiment APIs.
    weight_sparsity: dict[str, dict[str, float | int] | None] = {}
    activation_sparsity: dict[str, dict[str, float | int | str] | None] = {}
    for label, _ in model_variants:
        weight_sparsity[label] = experiments[label].weight_sparsity_metrics(threshold=0.5)
        activation_sparsity[label] = experiments[label].activation_sparsity_metrics(
            split="test",
            threshold=0.5,
        )

    # Step 6: Normalize sparse-fraction values used for final comparable summaries.
    weight_fraction_sparse: dict[str, float] = {}
    activation_fraction_sparse: dict[str, float] = {}
    for label, _ in model_variants:
        weight_fraction_sparse[label] = 0.0
        if weight_sparsity[label] is not None:
            weight_fraction_sparse[label] = float(weight_sparsity[label]["fraction_sparse"])
        activation_fraction_sparse[label] = weight_fraction_sparse[label]
        if activation_sparsity[label] is not None:
            activation_fraction_sparse[label] = float(activation_sparsity[label]["fraction_sparse"])

    # Step 7: Build normalized activation display strings for readability.
    activation_display: dict[str, str] = {}
    for label, _ in model_variants:
        activation_display[label] = "N/A (no activation gates)"
        if activation_sparsity[label] is not None:
            activation_display[label] = str(activation_sparsity[label])

    # Step 8: Print explicit sparsity interpretation for neuron and weight gating modes.
    print("\n===== Sparsity Report =====")
    print(
        "For mode='neuron' output gating, activation-gate sparsity is the primary "
        "signal and weight-gate sparsity is not expected to drop."
    )
    print(
        "For mode='weight' gating, weight-gate sparsity is the primary structural "
        "signal."
    )
    print(f"\nVanilla Weight-Gate Sparsity: {weight_sparsity['Vanilla']}")
    print(f"Neuron-Gated Weight-Gate Sparsity: {weight_sparsity['Neuron-Gated']}")
    print(f"Weight-Gated Weight-Gate Sparsity: {weight_sparsity['Weight-Gated']}")
    print(f"Vanilla Activation-Gate Sparsity: {activation_display['Vanilla']}")
    print(f"Neuron-Gated Activation-Gate Sparsity: {activation_display['Neuron-Gated']}")
    print(f"Weight-Gated Activation-Gate Sparsity: {activation_display['Weight-Gated']}")

    # Step 9: Print final comparable sparsity lines with per-model expected signal.
    vanilla_final = weight_fraction_sparse["Vanilla"]
    neuron_final = activation_fraction_sparse["Neuron-Gated"]
    weight_final = weight_fraction_sparse["Weight-Gated"]
    if weight_sparsity["Weight-Gated"] is None:
        weight_final = activation_fraction_sparse["Weight-Gated"]
    print(
        f"\nVanilla Final Comparable Sparsity: {vanilla_final:.4f} "
        "(from weight-gate sparsity)"
    )
    print(
        f"Neuron-Gated Final Comparable Sparsity: {neuron_final:.4f} "
        "(activation-gate preferred; weight-gate fallback)"
    )
    print(
        f"Weight-Gated Final Comparable Sparsity: {weight_final:.4f} "
        "(weight-gate preferred; activation-gate fallback)"
    )


if __name__ == "__main__":
    main()
