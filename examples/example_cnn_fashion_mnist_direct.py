"""Train and compare vanilla vs gated CNN on torchvision FashionMNIST.

This example uses the `Experiment` API and the dataset registry.
FashionMNIST is loaded through torchvision and downloaded to `./data` when
missing.
"""

# Step 0: Imports
from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.calibration.linear_calibration import LinearCalibration
from fermigates.experiments import Experiment
from fermigates.gates import FermiGate
from fermigates.losses import fermiloss
from fermigates.models import CNN

# Step 1: Fix seed and build models directly.
torch.manual_seed(7)
vanilla_model = CNN(
    input_channels=1,
    num_classes=10,
    gate=None,
    loss=fermiloss,
    calibration=LinearCalibration(d_in=10, d_out=10, learnable=False),
)
torch.manual_seed(7)
gated_model = CNN(
    input_channels=1,
    num_classes=10,
    gate=lambda: FermiGate(
        mode="channel",
        annealer="linear",
        init_mu=0.0,
        init_temperature=0.0,
    ),
    loss=fermiloss,
    calibration=LinearCalibration(d_in=10, d_out=10, learnable=False),
)

# Step 2: Create experiments.
vanilla_exp = Experiment(
    model=vanilla_model,
    dataset="fashion_mnist",
    epochs=2,
    learning_rate=1e-3,
)
gated_exp = Experiment(
    model=gated_model,
    dataset="fashion_mnist",
    epochs=2,
    learning_rate=1e-3,
)

# Step 3: Train models.
print("\n===== Training Vanilla CNN =====")
vanilla_exp.train()
print("\n===== Training Gated CNN =====")
gated_exp.train(calibrate_after_first_epoch=True, ridge_lambda=1e-3)

# Step 4: Evaluate and print accuracy.
vanilla_acc = vanilla_exp.accuracy()
gated_acc = gated_exp.accuracy()
print("\n===== Evaluation =====")
print(f"\nVanilla Accuracy: {vanilla_acc:.4f}")
print(f"Gated Accuracy:   {gated_acc:.4f}")

# Step 5: Report sparsity metrics from Experiment APIs.
vanilla_weight_sparsity = vanilla_exp.weight_sparsity_metrics(threshold=0.5)
gated_weight_sparsity = gated_exp.weight_sparsity_metrics(threshold=0.5)
vanilla_activation_sparsity = vanilla_exp.activation_sparsity_metrics(
    split="test",
    threshold=0.5,
)
gated_activation_sparsity = gated_exp.activation_sparsity_metrics(
    split="test",
    threshold=0.5,
)

print(f"\nVanilla Weight-Gate Sparsity: {vanilla_weight_sparsity}")
print(f"Gated   Weight-Gate Sparsity: {gated_weight_sparsity}")
print(f"Vanilla Activation-Gate Sparsity: {vanilla_activation_sparsity}")
print(f"Gated   Activation-Gate Sparsity: {gated_activation_sparsity}")
