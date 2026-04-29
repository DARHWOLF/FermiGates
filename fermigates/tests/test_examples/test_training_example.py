from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _patched_token_get_dataloader(
    name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    data_dir,
    download: bool,
) -> DataLoader:
    """Return deterministic token-classification loaders for example smoke tests."""
    del name
    del shuffle
    del data_dir
    del download

    # Step 1: Build deterministic token sequences and labels.
    generator = torch.Generator().manual_seed(17 if split == "train" else 19)
    n_samples = 48 if split == "train" else 24
    x_value = torch.randint(0, 4096, (n_samples, 64), generator=generator, dtype=torch.long)
    y_value = torch.randint(0, 10, (n_samples,), generator=generator, dtype=torch.long)

    # Step 2: Build and return strict DataLoader.
    dataset = TensorDataset(x_value, y_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_example_transformer_cifar10_main_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test Transformer CIFAR10 example with patched in-memory loaders."""

    # Step 1: Patch dataset loading path used by Experiment.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_token_get_dataloader,
    )

    # Step 2: Import example module from file path.
    path = Path(__file__).resolve().parents[3] / "examples" / "example_transformer_cifar10.py"
    spec = spec_from_file_location("example_transformer_cifar10", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Step 3: Execute main function and validate it completes.
    module.main()


def _patched_image_get_dataloader(
    name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    data_dir,
    download: bool,
) -> DataLoader:
    """Return deterministic image-classification loaders for example smoke tests."""
    del name
    del shuffle
    del data_dir
    del download

    # Step 1: Build deterministic image tensors and labels.
    generator = torch.Generator().manual_seed(29 if split == "train" else 31)
    n_samples = 16 if split == "train" else 8
    x_value = torch.randn(n_samples, 1, 28, 28, generator=generator)
    y_value = torch.randint(0, 10, (n_samples,), generator=generator, dtype=torch.long)

    # Step 2: Build and return strict DataLoader.
    dataset = TensorDataset(x_value, y_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_example_cnn_fashion_mnist_direct_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test direct CNN FashionMNIST example with patched in-memory loaders."""

    # Step 1: Patch dataset loading path used by Experiment.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_image_get_dataloader,
    )

    # Step 2: Import top-level example module and validate execution.
    path = Path(__file__).resolve().parents[3] / "examples" / "example_cnn_fashion_mnist_direct.py"
    spec = spec_from_file_location("example_cnn_fashion_mnist_direct", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)


def test_example_mlp_mnist_sparsity_reporting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate MNIST MLP example prints explicit gated sparsity semantics."""

    # Step 1: Patch dataset loading path used by Experiment.
    monkeypatch.setattr(
        "fermigates.experiments.experiment.get_dataloader",
        _patched_image_get_dataloader,
    )

    # Step 2: Import example module from file path.
    path = Path(__file__).resolve().parents[3] / "examples" / "example_MLP_mnist.py"
    spec = spec_from_file_location("example_MLP_mnist", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Step 3: Execute main function and capture stdout.
    module.main()
    captured = capsys.readouterr().out

    # Step 4: Validate explicit sparsity-report sections and normalized output.
    assert "===== Training Neuron-Gated MLP =====" in captured
    assert "===== Training Weight-Gated MLP =====" in captured
    assert "Neuron-Gated Accuracy:" in captured
    assert "Weight-Gated Accuracy:" in captured
    assert "===== Sparsity Report =====" in captured
    assert "For mode='neuron' output gating" in captured
    assert "For mode='weight' gating" in captured
    assert "Vanilla Activation-Gate Sparsity: N/A (no activation gates)" in captured
    assert "Neuron-Gated Weight-Gate Sparsity:" in captured
    assert "Weight-Gated Weight-Gate Sparsity:" in captured
    assert "Neuron-Gated Activation-Gate Sparsity:" in captured
    assert "Weight-Gated Activation-Gate Sparsity:" in captured
    assert "Vanilla Final Comparable Sparsity:" in captured
    assert "Neuron-Gated Final Comparable Sparsity:" in captured
    assert "Weight-Gated Final Comparable Sparsity:" in captured
