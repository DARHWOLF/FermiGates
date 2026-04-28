from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fermigates.calibration import LinearCalibration
from fermigates.experiments import Experiment
from fermigates.gates import FermiGate
from fermigates.models import MLP


def _patched_get_dataloader(
    name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    data_dir,
    download: bool,
) -> DataLoader:
    """Return deterministic synthetic classification loaders for experiment tests."""
    del name
    del shuffle
    del data_dir
    del download

    # Step 1: Build deterministic synthetic features and labels.
    generator = torch.Generator().manual_seed(7 if split == "train" else 11)
    n_samples = 96 if split == "train" else 48
    x_value = torch.randn(n_samples, 4, generator=generator)
    y_value = torch.randint(low=0, high=3, size=(n_samples,), generator=generator)

    # Step 2: Build and return strict DataLoader.
    dataset = TensorDataset(x_value, y_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_other_metrics_before_and_after_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate ``other_metrics`` stability before and after training."""

    # Step 1: Patch dataset loading with deterministic synthetic data.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_get_dataloader,
    )

    # Step 2: Build gated model and experiment harness.
    model = MLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=3,
        gate=lambda: FermiGate(mode="neuron", init_temperature=1.0),
        calibration=LinearCalibration(d_in=3, d_out=3, learnable=False),
    )
    exp = Experiment(model=model, dataset="mnist", epochs=2, batch_size=16, device="cpu")

    # Step 3: Validate pre-training metrics payload.
    pre_metrics = exp.other_metrics()
    assert pre_metrics["final_train_loss"] is None
    assert pre_metrics["final_train_accuracy"] is None
    assert pre_metrics["final_test_accuracy"] is None
    assert pre_metrics["weight_sparsity"] is not None
    assert "fraction_sparse" in pre_metrics["weight_sparsity"]

    # Step 4: Train and validate post-training metrics payload.
    exp.train()
    post_metrics = exp.other_metrics()
    assert post_metrics["final_train_loss"] is not None
    assert post_metrics["final_train_accuracy"] is not None
    assert post_metrics["final_test_accuracy"] is not None
    assert post_metrics["weight_sparsity"] is not None
    assert "fraction_sparse" in post_metrics["weight_sparsity"]


def test_activation_sparsity_metrics_for_gated_and_vanilla_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate activation sparsity helper for supported and unsupported gate outputs."""

    # Step 1: Patch dataset loading with deterministic synthetic data.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_get_dataloader,
    )

    # Step 2: Build gated and vanilla experiments.
    gated_model = MLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=3,
        gate=lambda: FermiGate(mode="neuron", init_temperature=1.0),
        calibration=LinearCalibration(d_in=3, d_out=3, learnable=False),
    )
    vanilla_model = MLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=3,
        gate=None,
        calibration=LinearCalibration(d_in=3, d_out=3, learnable=False),
    )
    gated_exp = Experiment(
        model=gated_model,
        dataset="mnist",
        epochs=1,
        batch_size=16,
        device="cpu",
    )
    vanilla_exp = Experiment(
        model=vanilla_model,
        dataset="mnist",
        epochs=1,
        batch_size=16,
        device="cpu",
    )

    # Step 3: Validate activation sparsity payload behavior.
    gated_metrics = gated_exp.activation_sparsity_metrics(split="test", threshold=0.5)
    vanilla_metrics = vanilla_exp.activation_sparsity_metrics(split="test", threshold=0.5)
    assert gated_metrics is not None
    assert "fraction_sparse" in gated_metrics
    assert vanilla_metrics is None


def test_first_epoch_ridge_calibration_reduces_label_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate first-epoch ridge calibration fitting and residual reduction."""

    # Step 1: Patch dataset loading with deterministic synthetic data.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_get_dataloader,
    )

    # Step 2: Build model and collect first-epoch logits/targets.
    model = MLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=3,
        gate=None,
        calibration=LinearCalibration(d_in=3, d_out=3, learnable=False),
    )
    exp = Experiment(model=model, dataset="mnist", epochs=1, batch_size=16, device="cpu")
    train_loader, _ = exp._build_loaders()
    logits_history: list[torch.Tensor] = []
    target_history: list[torch.Tensor] = []
    for x_batch, y_batch in train_loader:
        logits_batch = exp._forward_logits(x_batch.to(exp.device))
        logits_history.append(logits_batch.detach().cpu())
        target_history.append(y_batch.detach().cpu())

    # Step 3: Compute residual norms before fitting calibration.
    logits_all = torch.cat(logits_history, dim=0)
    targets_all = torch.cat(target_history, dim=0)
    one_hot = torch.nn.functional.one_hot(targets_all, num_classes=3).to(dtype=logits_all.dtype)
    residual_before = torch.norm(one_hot - logits_all)

    # Step 4: Fit calibration and compute post-fit residual norm.
    fitted = exp._fit_first_epoch_ridge_calibration(
        logits_history=logits_history,
        target_history=target_history,
        ridge_lambda=1e-6,
    )
    assert fitted
    correction = model.calibration(logits_all)
    residual_after = torch.norm(one_hot - (logits_all + correction))
    assert residual_after < residual_before


def test_compute_loss_prefers_model_loss_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate custom model loss path is used by experiment loss computation."""

    # Step 1: Patch dataset loading with deterministic synthetic data.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_get_dataloader,
    )

    # Step 2: Build model with explicit custom loss.
    custom_loss_called = {"value": 0}

    def custom_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        custom_loss_called["value"] += 1
        return logits.square().mean()

    model = MLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=3,
        gate=None,
        loss=custom_loss,
        calibration=LinearCalibration(d_in=3, d_out=3, learnable=False),
    )
    exp = Experiment(model=model, dataset="mnist", epochs=1, batch_size=16, device="cpu")
    train_loader, _ = exp._build_loaders()
    x_batch, y_batch = next(iter(train_loader))
    logits = exp._forward_logits(x_batch.to(exp.device))

    # Step 3: Validate custom loss function path.
    loss_value = exp._compute_loss(logits, y_batch.to(exp.device))
    assert custom_loss_called["value"] == 1
    assert torch.is_tensor(loss_value)
