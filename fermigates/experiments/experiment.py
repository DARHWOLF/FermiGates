from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fermigates.datasets import get_dataloader


class Experiment:
    """Simple training harness for user-facing examples.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train and evaluate.
    dataset : str, default="mnist"
        Dataset registry name.
    epochs : int, default=5
        Number of training epochs.
    batch_size : int, default=128
        Mini-batch size.
    learning_rate : float, default=1e-3
        Optimizer learning rate.
    seed : int, default=7
        Compatibility seed field retained for API stability.
    device : str or torch.device or None, optional
        Compute device. If ``None``, auto-selects CUDA when available.
    data_dir : str or Path, default="./data"
        Dataset root directory passed to registered dataset loaders.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: str = "mnist",
        epochs: int = 5,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 7,
        device: str | torch.device | None = None,
        data_dir: str | Path = "./data",
    ) -> None:
        # Step 1: Validate user configuration.
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        # Step 2: Store run configuration.
        self.model = model
        self.dataset = dataset.lower()
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Step 3: Initialize tracking containers.
        self.model.to(self.device)
        self._train_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._dataset_used = self.dataset
        self._training_time_seconds = 0.0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Resolve dataloaders from the configured dataset name.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            Training and test DataLoader objects.
        """
        # Step 1: Build train split loader from dataset registry.
        self._train_loader = get_dataloader(
            name=self.dataset,
            split="train",
            batch_size=self.batch_size,
            shuffle=True,
            data_dir=self.data_dir,
            download=True,
        )

        # Step 2: Build strict test split loader from dataset registry.
        self._test_loader = get_dataloader(
            name=self.dataset,
            split="test",
            batch_size=self.batch_size,
            shuffle=False,
            data_dir=self.data_dir,
            download=True,
        )

        # Step 3: Return cached loaders.
        return self._train_loader, self._test_loader

    def _compute_loss(self, logits: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """Compute training loss, preferring model-provided loss function.

        Parameters
        ----------
        logits : torch.Tensor
            Model outputs.
        y_batch : torch.Tensor
            Ground-truth labels.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """

        # Step 1: Use model-provided loss when available.
        model_loss = getattr(self.model, "loss_fn", None)
        if callable(model_loss):
            return model_loss(logits, y_batch)

        # Step 2: Fallback to cross-entropy.
        return F.cross_entropy(logits, y_batch)

    def train(self) -> None:
        """Train the configured model for the requested epochs."""

        # Step 1: Prepare data and optimizer.
        train_loader, _ = self._build_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Step 2: Run training epochs.
        started_at = time.perf_counter()
        for _ in range(self.epochs):
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self._compute_loss(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.item()) * int(y_batch.size(0))
                preds = logits.argmax(dim=1)
                epoch_correct += int((preds == y_batch).sum().item())
                epoch_total += int(y_batch.size(0))

            epoch_loss = epoch_loss_sum / float(epoch_total)
            epoch_acc = float(epoch_correct) / float(epoch_total)
            self.history["train_loss"].append(epoch_loss)
            self.history["train_accuracy"].append(epoch_acc)
            self.history["test_accuracy"].append(self.accuracy())

        # Step 3: Store elapsed runtime.
        ended_at = time.perf_counter()
        self._training_time_seconds = ended_at - started_at

    @torch.no_grad()
    def accuracy(self) -> float:
        """Return classification accuracy on the test split.

        Returns
        -------
        float
            Test-set accuracy in ``[0.0, 1.0]``.
        """

        # Step 1: Build/load test data.
        _, test_loader = self._build_loaders()

        # Step 2: Run evaluation loop.
        self.model.eval()
        correct = 0
        total = 0
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            logits = self.model(x_batch)
            preds = logits.argmax(dim=1)
            correct += int((preds == y_batch).sum().item())
            total += int(y_batch.size(0))

        # Step 3: Return scalar accuracy.
        return float(correct) / float(total)

    def training_time(self) -> float:
        """Return accumulated training time in seconds."""
        return float(self._training_time_seconds)

    def other_metrics(self) -> dict[str, Any]:
        """Return a compact metrics summary for downstream logging.

        Returns
        -------
        dict[str, Any]
            Dictionary with dataset, loss, accuracy, runtime, and sparsity keys.
        """

        # Step 1: Gather latest scalar metrics.
        final_train_loss = self.history["train_loss"][-1] if self.history["train_loss"] else None
        final_train_accuracy = (
            self.history["train_accuracy"][-1] if self.history["train_accuracy"] else None
        )
        final_test_accuracy = self.accuracy()

        # Step 2: Include model sparsity when supported.
        sparsity = None
        compute_sparsity = getattr(self.model, "compute_sparsity", None)
        if callable(compute_sparsity):
            kept, total, frac = compute_sparsity(threshold=0.5)
            sparsity = {
                "kept": int(kept),
                "total": int(total),
                "fraction_kept": float(frac),
            }

        # Step 3: Return combined metrics payload.
        return {
            "dataset": self._dataset_used,
            "epochs": self.epochs,
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "training_time_seconds": self.training_time(),
            "sparsity": sparsity,
        }
