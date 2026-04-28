from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10TokenDataset(Dataset):
    """Torchvision-backed CIFAR10 dataset converted to discrete token sequences.

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
            raise ValueError(f"Invalid CIFAR10 split '{split}'. Expected 'train' or 'test'.")

        # Step 2: Resolve torchvision split and transform.
        is_train_split = split_name == "train"
        transform_pipeline = transforms.Compose([transforms.ToTensor()])

        # Step 3: Build torchvision CIFAR10 dataset.
        self.dataset = datasets.CIFAR10(
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
        """Return one token-sequence and label pair.

        Parameters
        ----------
        idx : int
            Example index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Sequence tensor in ``(64,)`` with values in ``[0, 4095]`` and label
            tensor as ``torch.long``.
        """
        # Step 1: Fetch one normalized image-label pair.
        x_value, y_value = self.dataset[idx]
        x_value = x_value.to(dtype=torch.float32)

        # Step 2: Patchify with 4x4 average pooling to 8x8 per channel.
        pooled = F.avg_pool2d(x_value.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)

        # Step 3: Quantize channels to 16 bins and pack RGB bins into one token id.
        quantized = torch.clamp((pooled * 16.0).to(dtype=torch.long), min=0, max=15)
        token_grid = (quantized[0] * 256) + (quantized[1] * 16) + quantized[2]
        tokens = token_grid.reshape(-1).to(dtype=torch.long)

        # Step 4: Return packed token sequence and class label.
        y_tensor = torch.tensor(y_value, dtype=torch.long)
        return tokens, y_tensor
