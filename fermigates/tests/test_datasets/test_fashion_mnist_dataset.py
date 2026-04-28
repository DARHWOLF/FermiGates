from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from fermigates.datasets import FashionMNISTDataset, get_dataset


class _FakeTorchvisionFashionMNIST(Dataset):
    """Tiny in-memory FashionMNIST stub for deterministic dataset tests."""

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
        self._y_value = 5

    def __len__(self) -> int:
        # Step 1: Return deterministic dataset length.
        return 1

    def __getitem__(self, idx: int):
        # Step 1: Return deterministic sample.
        return self._x_value, self._y_value


def test_fashion_mnist_load_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate torchvision FashionMNIST loading and tensor shapes."""

    # Step 1: Patch torchvision FashionMNIST with deterministic in-memory dataset.
    monkeypatch.setattr(
        "fermigates.datasets.fashion_mnist.datasets.FashionMNIST",
        _FakeTorchvisionFashionMNIST,
    )

    # Step 2: Load FashionMNIST train dataset.
    dataset = FashionMNISTDataset(split="train", root="./tmp-data", download=True)

    # Step 3: Fetch one sample for shape and dtype checks.
    x_value, y_value = dataset[0]

    # Step 4: Validate strict tensor outputs.
    assert len(dataset) > 0
    assert x_value.shape == (1, 28, 28)
    assert x_value.dtype == torch.float32
    assert y_value.dtype == torch.long


def test_fashion_mnist_registry_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate dataset registry resolves ``fashion_mnist``."""

    # Step 1: Patch torchvision FashionMNIST with deterministic in-memory dataset.
    monkeypatch.setattr(
        "fermigates.datasets.fashion_mnist.datasets.FashionMNIST",
        _FakeTorchvisionFashionMNIST,
    )

    # Step 2: Resolve dataset via public registry.
    dataset = get_dataset(name="fashion_mnist", split="test", data_dir="./tmp-data", download=False)

    # Step 3: Validate resolved dataset type and output contract.
    assert isinstance(dataset, FashionMNISTDataset)
    x_value, y_value = dataset[0]
    assert x_value.shape == (1, 28, 28)
    assert y_value.dtype == torch.long
