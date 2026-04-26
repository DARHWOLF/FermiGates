from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Experiment:
    """Simple training harness for user-facing examples.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train and evaluate.
    dataset : str, default="mnist"
        Dataset name. Currently supports ``"mnist"`` with automatic synthetic
        fallback when local MNIST files are unavailable.
    epochs : int, default=5
        Number of training epochs.
    batch_size : int, default=128
        Mini-batch size.
    learning_rate : float, default=1e-3
        Optimizer learning rate.
    seed : int, default=7
        Random seed for synthetic data generation.
    device : str or torch.device or None, optional
        Compute device. If ``None``, auto-selects CUDA when available.
    data_dir : str or Path, default="./data"
        Local data directory for MNIST checks.
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
        # Step 1: validate user configuration
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        # Step 2: store run configuration
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
        self.num_classes = getattr(self.model, "num_classes", 10)

        # Step 3: initialize tracking containers
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

    def _model_input_dim(self) -> int:
        """Infer flattened input dimension from common model structures."""

        # Step 1: use explicit model attribute when available
        value = getattr(self.model, "input_dim", None)
        if isinstance(value, int) and value > 0:
            return value

        # Step 2: infer from first linear layer in MLP-style models
        layers = getattr(self.model, "layers", None)
        if layers is not None and len(layers) > 0:
            first_layer = layers[0]
            linear = getattr(first_layer, "linear", None)
            in_features = getattr(linear, "in_features", None)
            if isinstance(in_features, int) and in_features > 0:
                return in_features

        # Step 3: explicit fallback
        return 784

    def _mnist_files_present(self) -> bool:
        """Check whether standard MNIST files exist locally."""

        # Step 1: define required MNIST raw files
        raw_dir = self.data_dir / "MNIST" / "raw"
        required_files = (
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        )

        # Step 2: verify every required file
        return all((raw_dir / filename).exists() for filename in required_files)

    def _synthetic_mnist_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Build deterministic synthetic MNIST-like loaders."""

        # Step 1: set deterministic synthetic data generation
        generator = torch.Generator().manual_seed(self.seed)
        input_dim = self._model_input_dim()
        num_classes = int(self.num_classes)
        n_train = 2048
        n_test = 512

        # Step 2: generate synthetic image-like tensors and pseudo-labels
        x_train = torch.rand(n_train, input_dim, generator=generator)
        x_test = torch.rand(n_test, input_dim, generator=generator)
        projection = torch.randn(input_dim, num_classes, generator=generator)
        y_train = (x_train @ projection).argmax(dim=1)
        y_test = (x_test @ projection).argmax(dim=1)

        # Step 3: reshape to MNIST image tensor form when possible
        if input_dim == 784:
            x_train = x_train.view(n_train, 1, 28, 28)
            x_test = x_test.view(n_test, 1, 28, 28)

        # Step 4: materialize dataloaders
        train_set = TensorDataset(x_train, y_train)
        test_set = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def _synthetic_text_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Build deterministic synthetic text classification loaders."""

        # Step 1: set deterministic token generation and dimensions
        generator = torch.Generator().manual_seed(self.seed)
        vocab_size = int(getattr(self.model, "vocab_size", 1000))
        seq_len = int(getattr(self.model, "max_seq_len", 32))
        num_classes = int(self.num_classes)
        n_train = 2048
        n_test = 512

        # Step 2: synthesize token tensors and deterministic labels
        x_train = torch.randint(0, vocab_size, (n_train, seq_len), generator=generator)
        x_test = torch.randint(0, vocab_size, (n_test, seq_len), generator=generator)
        y_train = torch.remainder(x_train.sum(dim=1), num_classes).to(dtype=torch.long)
        y_test = torch.remainder(x_test.sum(dim=1), num_classes).to(dtype=torch.long)

        # Step 3: materialize dataloaders
        train_set = TensorDataset(x_train, y_train)
        test_set = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Resolve dataloaders from configured dataset name."""

        # Step 1: return existing loaders when already built
        if self._train_loader is not None and self._test_loader is not None:
            return self._train_loader, self._test_loader

        # Step 2: use MNIST when torchvision + local files are available
        if self.dataset == "mnist":
            has_torchvision = importlib.util.find_spec("torchvision") is not None
            has_local_mnist = self._mnist_files_present()
            if has_torchvision and has_local_mnist:
                from torchvision import datasets, transforms

                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )
                train_set = datasets.MNIST(
                    root=str(self.data_dir),
                    train=True,
                    transform=transform,
                    download=False,
                )
                test_set = datasets.MNIST(
                    root=str(self.data_dir),
                    train=False,
                    transform=transform,
                    download=False,
                )
                self._dataset_used = "mnist"
                self._train_loader = DataLoader(
                    train_set, batch_size=self.batch_size, shuffle=True
                )
                self._test_loader = DataLoader(
                    test_set, batch_size=self.batch_size, shuffle=False
                )
                return self._train_loader, self._test_loader

            self._dataset_used = "synthetic-mnist"
            self._train_loader, self._test_loader = self._synthetic_mnist_loaders()
            return self._train_loader, self._test_loader

        # Step 3: synthetic text dataset for transformer API tests
        if self.dataset == "synthetic_text":
            self._dataset_used = "synthetic_text"
            self._train_loader, self._test_loader = self._synthetic_text_loaders()
            return self._train_loader, self._test_loader

        # Step 4: fail fast for unsupported dataset names
        raise ValueError(f"Unsupported dataset '{self.dataset}'.")

    def _compute_loss(self, logits: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """Compute training loss, preferring model-provided loss function."""

        # Step 1: use model-provided loss when available
        model_loss = getattr(self.model, "loss_fn", None)
        if callable(model_loss):
            return model_loss(logits, y_batch)

        # Step 2: fallback to cross-entropy
        return F.cross_entropy(logits, y_batch)

    def train(self) -> None:
        """Train the configured model for the requested epochs."""

        # Step 1: prepare data and optimizer
        train_loader, _ = self._build_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Step 2: run training epochs
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

        # Step 3: store elapsed runtime
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

        # Step 1: build/load test data
        _, test_loader = self._build_loaders()

        # Step 2: run evaluation loop
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

        # Step 3: return scalar accuracy
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

        # Step 1: gather latest scalar metrics
        final_train_loss = self.history["train_loss"][-1] if self.history["train_loss"] else None
        final_train_accuracy = (
            self.history["train_accuracy"][-1] if self.history["train_accuracy"] else None
        )
        final_test_accuracy = self.accuracy()

        # Step 2: include model sparsity when supported
        sparsity = None
        compute_sparsity = getattr(self.model, "compute_sparsity", None)
        if callable(compute_sparsity):
            kept, total, frac = compute_sparsity(threshold=0.5)
            sparsity = {
                "kept": int(kept),
                "total": int(total),
                "fraction_kept": float(frac),
            }

        # Step 3: return combined metrics payload
        return {
            "dataset": self._dataset_used,
            "epochs": self.epochs,
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "training_time_seconds": self.training_time(),
            "sparsity": sparsity,
        }
