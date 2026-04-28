from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fermigates.base import BaseFermiModel
from fermigates.calibration import LinearCalibration
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

    def _supports_return_gate_outputs(self) -> bool:
        """Return whether model forward accepts ``return_gate_outputs`` keyword.

        Returns
        -------
        bool
            ``True`` when the model forward signature supports gate traces.
        """

        # Step 1: Inspect forward signature.
        forward_signature = inspect.signature(self.model.forward)
        parameters = forward_signature.parameters

        # Step 2: Accept explicit keyword or var-keyword signatures.
        if "return_gate_outputs" in parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    def _forward_logits(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Run model forward pass and return logits tensor.

        Parameters
        ----------
        x_batch : torch.Tensor
            Input mini-batch.

        Returns
        -------
        torch.Tensor
            Logits tensor.
        """

        # Step 1: Execute forward pass.
        output = self.model(x_batch)

        # Step 2: Normalize tuple outputs to logits-only tensor.
        if isinstance(output, tuple):
            return output[0]
        return output

    def _fit_first_epoch_ridge_calibration(
        self,
        logits_history: list[torch.Tensor],
        target_history: list[torch.Tensor],
        ridge_lambda: float,
    ) -> bool:
        """Fit and load logits-space ridge calibration from first-epoch residuals.

        Parameters
        ----------
        logits_history : list[torch.Tensor]
            Collected first-epoch logits batches.
        target_history : list[torch.Tensor]
            Collected first-epoch integer targets.
        ridge_lambda : float
            Ridge regularization coefficient.

        Returns
        -------
        bool
            ``True`` when calibration was fitted and loaded.
        """

        # Step 1: Validate collected first-epoch tensors.
        if not logits_history or not target_history:
            return False

        # Step 2: Build regression design and residual targets.
        X = torch.cat(logits_history, dim=0)
        y = torch.cat(target_history, dim=0).to(dtype=torch.long)
        if X.ndim != 2:
            return False
        if y.ndim != 1:
            return False
        num_classes = int(X.shape[1])
        if num_classes <= 0:
            return False
        one_hot = F.one_hot(y, num_classes=num_classes).to(dtype=X.dtype)
        residual = one_hot - X

        # Step 3: Solve closed-form ridge calibration parameters.
        W_hat, b_hat = BaseFermiModel.solve_ridge_cpu(
            X,
            residual,
            lam=float(ridge_lambda),
            add_bias=True,
        )
        if b_hat is None:
            return False

        # Step 4: Resolve calibration module instance.
        calibration_module = getattr(self.model, "calibration", None)
        if calibration_module is None:
            calibration_module = LinearCalibration(
                d_in=num_classes,
                d_out=num_classes,
                learnable=False,
                device=self.device,
                dtype=X.dtype,
            )
            self.model.calibration = calibration_module
        if not isinstance(calibration_module, LinearCalibration):
            return False

        # Step 5: Load solved parameters into model calibration module.
        calibration_module.load_calibration(
            W_hat.to(device=self.device),
            b_hat.to(device=self.device),
        )
        return True

    def weight_sparsity_metrics(self, threshold: float = 0.5) -> dict[str, float | int] | None:
        """Return weight-gate sparsity metrics when supported by the model.

        Parameters
        ----------
        threshold : float, default=0.5
            Threshold used by model sparsity utilities.

        Returns
        -------
        dict[str, float | int] or None
            Weight sparsity payload or ``None`` when unsupported.
        """

        # Step 1: Resolve model sparsity utility.
        compute_sparsity = getattr(self.model, "compute_sparsity", None)
        if not callable(compute_sparsity):
            return None

        # Step 2: Build normalized sparsity payload.
        kept, total, fraction_kept = compute_sparsity(threshold=threshold)
        total_count = int(total)
        kept_count = int(kept)
        fraction_kept_value = float(fraction_kept)
        fraction_sparse_value = 1.0 - fraction_kept_value if total_count > 0 else 0.0
        return {
            "kept": kept_count,
            "total": total_count,
            "fraction_kept": fraction_kept_value,
            "fraction_sparse": fraction_sparse_value,
        }

    @torch.no_grad()
    def activation_sparsity_metrics(
        self,
        split: str = "test",
        threshold: float = 0.5,
    ) -> dict[str, float | int | str] | None:
        """Return activation-gate sparsity metrics when gate outputs are exposed.

        Parameters
        ----------
        split : {"train", "test"}, default="test"
            Dataset split used for gate activation measurement.
        threshold : float, default=0.5
            Active/inactive threshold for gate probabilities.

        Returns
        -------
        dict[str, float | int | str] or None
            Activation sparsity payload or ``None`` when unsupported.
        """

        # Step 1: Validate split argument and gate-output support.
        split_name = split.lower()
        if split_name not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        if not self._supports_return_gate_outputs():
            return None

        # Step 2: Resolve requested split loader.
        train_loader, test_loader = self._build_loaders()
        target_loader = train_loader if split_name == "train" else test_loader

        # Step 3: Aggregate active/total gate activation counts.
        self.model.eval()
        active = 0
        total = 0
        for x_batch, _ in target_loader:
            x_batch = x_batch.to(self.device)
            output = self.model(x_batch, return_gate_outputs=True)
            if not isinstance(output, tuple):
                return None
            gate_outputs = output[1]
            if not isinstance(gate_outputs, list):
                return None
            for gate_probs in gate_outputs:
                if gate_probs is None:
                    continue
                active += int((gate_probs > threshold).sum().item())
                total += int(gate_probs.numel())

        # Step 4: Return None when no gate activations were exposed.
        if total == 0:
            return None

        # Step 5: Return normalized activation sparsity payload.
        fraction_active = float(active) / float(total)
        fraction_sparse = 1.0 - fraction_active
        return {
            "split": split_name,
            "active": int(active),
            "total": int(total),
            "fraction_active": float(fraction_active),
            "fraction_sparse": float(fraction_sparse),
        }

    def train(
        self,
        calibrate_after_first_epoch: bool = False,
        ridge_lambda: float = 1e-3,
    ) -> None:
        """Train the configured model for the requested epochs."""

        # Step 1: Prepare data and optimizer.
        train_loader, _ = self._build_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Step 2: Run training epochs.
        started_at = time.perf_counter()
        for epoch_idx in range(self.epochs):
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_logits: list[torch.Tensor] = []
            epoch_targets: list[torch.Tensor] = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self._forward_logits(x_batch)
                loss = self._compute_loss(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.item()) * int(y_batch.size(0))
                preds = logits.argmax(dim=1)
                epoch_correct += int((preds == y_batch).sum().item())
                epoch_total += int(y_batch.size(0))
                if calibrate_after_first_epoch and epoch_idx == 0:
                    epoch_logits.append(logits.detach().cpu())
                    epoch_targets.append(y_batch.detach().cpu())

            epoch_loss = epoch_loss_sum / float(epoch_total)
            epoch_acc = float(epoch_correct) / float(epoch_total)
            self.history["train_loss"].append(epoch_loss)
            self.history["train_accuracy"].append(epoch_acc)
            self.history["test_accuracy"].append(self.accuracy())
            if calibrate_after_first_epoch and epoch_idx == 0:
                self._fit_first_epoch_ridge_calibration(
                    logits_history=epoch_logits,
                    target_history=epoch_targets,
                    ridge_lambda=ridge_lambda,
                )

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

        # Step 1: Gather latest scalar metrics from recorded history only.
        final_train_loss = self.history["train_loss"][-1] if self.history["train_loss"] else None
        final_train_accuracy = (
            self.history["train_accuracy"][-1] if self.history["train_accuracy"] else None
        )
        final_test_accuracy = (
            self.history["test_accuracy"][-1] if self.history["test_accuracy"] else None
        )

        # Step 2: Include lightweight weight sparsity summary.
        weight_sparsity = self.weight_sparsity_metrics(threshold=0.5)

        # Step 3: Return combined metrics payload.
        return {
            "dataset": self._dataset_used,
            "epochs": self.epochs,
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "training_time_seconds": self.training_time(),
            "weight_sparsity": weight_sparsity,
        }
