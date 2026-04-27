from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Torchvision-backed MNIST dataset.

    Parameters
    ----------
    split : {"train", "test"}, default="train"
        Dataset split name.
    root : str or Path, default="./data"
        Root directory used by torchvision for dataset storage.
    download : bool, default=True
        Whether torchvision should download files when not present locally.

    Raises
    ------
    ValueError
        If ``split`` is invalid.
    """

    def __init__(
        self,
        split: str = "train",
        root: str | Path = "./data",
        download: bool = True,
    ) -> None:
        # Step 1: Validate split value.
        split_name = split.lower()
        if split_name not in {"train", "test"}:
            raise ValueError(f"Invalid MNIST split '{split}'. Expected 'train' or 'test'.")

        # Step 2: Resolve torchvision split and transform.
        is_train_split = split_name == "train"
        transform_pipeline = transforms.Compose([transforms.ToTensor()])

        # Step 3: Build torchvision MNIST dataset.
        self.dataset = datasets.MNIST(
            root=str(root),
            train=is_train_split,
            transform=transform_pipeline,
            download=download,
        )

    def __len__(self) -> int:
        """Return number of examples in the split.

        Returns
        -------
        int
            Dataset length.
        """
        return int(len(self.dataset))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image-label pair.

        Parameters
        ----------
        idx : int
            Example index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image tensor in ``(1, 28, 28)`` and label tensor as ``torch.long``.
        """
        # Step 1: Fetch raw example from torchvision dataset.
        x_value, y_value = self.dataset[idx]

        # Step 2: Ensure output dtypes match library expectations.
        x_value = x_value.to(dtype=torch.float32)
        y_tensor = torch.tensor(y_value, dtype=torch.long)

        # Step 3: Return converted tensors.
        return x_value, y_tensor
