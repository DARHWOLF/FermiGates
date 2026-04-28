from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from fermigates.datasets import CIFAR10TokenDataset, get_dataset


class _FakeTorchvisionCIFAR10(Dataset):
    """Tiny in-memory CIFAR10 stub for deterministic dataset tests."""

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
        self._x_value = torch.ones(3, 32, 32, dtype=torch.float32) * 0.5
        self._y_value = 4

    def __len__(self) -> int:
        # Step 1: Return deterministic dataset length.
        return 1

    def __getitem__(self, idx: int):
        # Step 1: Return deterministic sample.
        return self._x_value, self._y_value


def test_cifar10_tokens_load_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate CIFAR10 token dataset conversion and output dtypes."""

    # Step 1: Patch torchvision CIFAR10 with deterministic in-memory dataset.
    monkeypatch.setattr(
        "fermigates.datasets.cifar10_tokens.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    # Step 2: Load tokenized CIFAR10 train dataset.
    dataset = CIFAR10TokenDataset(split="train", root="./tmp-data", download=True)

    # Step 3: Fetch one sample for shape/range/dtype checks.
    token_seq, y_value = dataset[0]

    # Step 4: Validate strict output contracts.
    assert len(dataset) > 0
    assert token_seq.shape == (64,)
    assert token_seq.dtype == torch.long
    assert int(token_seq.min().item()) >= 0
    assert int(token_seq.max().item()) <= 4095
    assert y_value.dtype == torch.long


def test_cifar10_tokens_registry_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate dataset registry resolves ``cifar10_tokens``."""

    # Step 1: Patch torchvision CIFAR10 with deterministic in-memory dataset.
    monkeypatch.setattr(
        "fermigates.datasets.cifar10_tokens.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    # Step 2: Resolve dataset via public registry.
    dataset = get_dataset(name="cifar10_tokens", split="test", data_dir="./tmp-data", download=False)

    # Step 3: Validate resolved dataset type and one sample contract.
    assert isinstance(dataset, CIFAR10TokenDataset)
    token_seq, y_value = dataset[0]
    assert token_seq.shape == (64,)
    assert y_value.dtype == torch.long
