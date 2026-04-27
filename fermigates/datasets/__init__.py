from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from fermigates.datasets._base import BaseDataset
from fermigates.datasets.mnist import MNISTDataset

_DATASETS = {
    "mnist": MNISTDataset,
}


def get_dataset(
    name: str,
    split: str,
    data_dir: str | Path = "./data",
    download: bool = True,
):
    """Return a dataset instance from the registry.

    Parameters
    ----------
    name : str
        Dataset registry name.
    split : str
        Dataset split value.
    data_dir : str or Path, default="./data"
        Dataset root directory passed to dataset implementations.
    download : bool, default=True
        Whether dataset implementations may download missing files.

    Returns
    -------
    torch.utils.data.Dataset
        Resolved dataset instance.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    # Step 1: Normalize dataset registry key.
    dataset_name = name.lower()

    # Step 2: Validate registry entry.
    if dataset_name not in _DATASETS:
        raise ValueError(f"Dataset '{name}' is not supported.")

    # Step 3: Instantiate dataset for the requested split.
    dataset_class = _DATASETS[dataset_name]
    return dataset_class(split=split, root=data_dir, download=download)


def get_dataloader(
    name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    data_dir: str | Path = "./data",
    download: bool = True,
) -> DataLoader:
    """Build a dataloader from a registered dataset.

    Parameters
    ----------
    name : str
        Dataset registry name.
    split : str
        Dataset split value.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        DataLoader shuffle flag.
    data_dir : str or Path, default="./data"
        Dataset root directory passed to dataset implementations.
    download : bool, default=True
        Whether dataset implementations may download missing files.

    Returns
    -------
    torch.utils.data.DataLoader
        Data loader for the requested dataset split.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    """
    # Step 1: Validate batch size.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Step 2: Build dataset from strict registry.
    dataset = get_dataset(name=name, split=split, data_dir=data_dir, download=download)

    # Step 3: Build and return dataloader.
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)


__all__ = ["BaseDataset", "MNISTDataset", "get_dataset", "get_dataloader"]
