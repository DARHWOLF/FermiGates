from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from fermigates.datasets import BaseDataset, MNISTDataset, get_dataset


class _FakeTorchvisionMNIST(Dataset):
    """Tiny in-memory MNIST stub for deterministic dataset tests."""

    def __init__(
        self,
        root: str,
        train: bool,
        transform,
        download: bool,
    ) -> None:
        # Step 1: Store constructor arguments for optional inspection.
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        # Step 2: Build one deterministic sample.
        self._x_value = torch.zeros(1, 28, 28, dtype=torch.float32)
        self._y_value = 3

    def __len__(self) -> int:
        # Step 1: Return deterministic dataset length.
        return 1

    def __getitem__(self, idx: int):
        # Step 1: Return deterministic sample.
        return self._x_value, self._y_value


def test_mnist_load_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate torchvision MNIST loading and tensor shapes."""

    # Step 1: Patch torchvision MNIST with deterministic in-memory dataset.
    monkeypatch.setattr("fermigates.datasets.mnist.datasets.MNIST", _FakeTorchvisionMNIST)

    # Step 2: Load MNIST train dataset.
    dataset = MNISTDataset(split="train", root="./tmp-data", download=True)

    # Step 3: Fetch one sample for shape and dtype checks.
    x_value, y_value = dataset[0]

    # Step 4: Validate strict tensor outputs.
    assert len(dataset) > 0
    assert x_value.shape == (1, 28, 28)
    assert x_value.dtype == torch.float32
    assert y_value.dtype == torch.long


def test_missing_file_raises() -> None:
    """Validate that missing CSV files fail loudly."""

    # Step 1: Construct non-existent CSV path.
    missing_path = Path(__file__).resolve().parent / "does_not_exist.csv"

    # Step 2: Assert strict missing-file error.
    with pytest.raises(FileNotFoundError):
        BaseDataset(file_path=str(missing_path))


def test_invalid_dataset_name() -> None:
    """Validate registry behavior for unknown dataset names."""

    # Step 1: Call dataset registry with unsupported name.
    # Step 2: Assert strict unsupported-dataset error.
    with pytest.raises(ValueError):
        get_dataset(name="unknown_dataset", split="train")
