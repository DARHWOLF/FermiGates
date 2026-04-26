from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn

from fermigates.gates import BaseGate

if TYPE_CHECKING:
    from fermigates.export.pruning import ModelPruningReport
    from fermigates.calibration.linear_calibration import LinearCalibration
    from fermigates.metrics.tracking import ModelOccupancyMetrics


@dataclass(frozen=True)
class SparsitySummary:
    kept: int
    total: int
    fraction_kept: float


class BaseFermiLayer(nn.Module, ABC):
    """Abstract base class for any layer exposing differentiable gate probabilities."""

    @abstractmethod
    def gate_probabilities(self) -> torch.Tensor:
        """Return soft gate values associated with this layer."""

    def count_nonzero(self, threshold: float = 0.5) -> tuple[int, int]:
        probs = self.gate_probabilities()
        total = probs.numel()
        kept = int((probs >= threshold).sum().item())
        return kept, total


class BaseFermiModel(nn.Module):
    """Unified base model for gating, sparsity accounting, and calibration utilities."""

    def __init__(self) -> None:
        super().__init__()
        self.calibrations = nn.ModuleDict()

    def iter_gates(self):
        for module in self.modules():
            if isinstance(module, BaseGate):
                yield module

    def set_temperature(self, T_new: float) -> None:
        for gate in self.iter_gates():
            gate.set_temperature(T_new)

    def init_mu_from_weights(self, percentile: float = 0.5, per_layer_neuron: bool = False) -> None:
        """Initialize ``mu`` for all compatible layers.

        Any module that implements ``initialize_mu_from_weight_percentile`` is supported.
        """

        for module in self.modules():
            init_fn = getattr(module, "initialize_mu_from_weight_percentile", None)
            if callable(init_fn):
                init_fn(percentile=percentile, per_neuron=per_layer_neuron)

    def compute_sparsity_summary(self, threshold: float = 0.5) -> SparsitySummary:
        kept = 0
        total = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, BaseFermiLayer):
                    k, t = module.count_nonzero(threshold=threshold)
                    kept += k
                    total += t
        frac = float(kept) / float(total) if total > 0 else 0.0
        return SparsitySummary(kept=kept, total=total, fraction_kept=frac)

    def compute_sparsity(self, threshold: float = 0.5) -> Tuple[int, int, float]:
        summary = self.compute_sparsity_summary(threshold=threshold)
        return summary.kept, summary.total, summary.fraction_kept

    def collect_gate_metrics(self, threshold: float = 0.5) -> "ModelOccupancyMetrics":
        from fermigates.metrics import collect_gate_metrics

        return collect_gate_metrics(self, threshold=threshold)

    def hard_masked_state_dict(self, threshold: float = 0.5) -> dict[str, torch.Tensor]:
        from fermigates.export import hard_masked_state_dict

        return hard_masked_state_dict(self, threshold=threshold)

    def to_hard_masked_model(self, threshold: float = 0.5) -> "BaseFermiModel":
        from fermigates.export import to_hard_masked_model

        return to_hard_masked_model(self, threshold=threshold)

    def pruning_report(
        self,
        threshold: float = 0.5,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> "ModelPruningReport":
        from fermigates.export import pruning_report

        return pruning_report(self, threshold=threshold, example_inputs=example_inputs)

    @staticmethod
    def solve_ridge_cpu(
        X: torch.Tensor,
        E: torch.Tensor,
        lam: float = 1e-3,
        add_bias: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Closed-form ridge solver on CPU in float64 for numerical stability."""

        Xc = X.detach().cpu().to(dtype=torch.float64)
        Ec = E.detach().cpu().to(dtype=torch.float64)
        n, d_in = Xc.shape

        if add_bias:
            ones = torch.ones((n, 1), dtype=Xc.dtype, device=Xc.device)
            X_aug = torch.cat([Xc, ones], dim=1)
            d_aug = d_in + 1
            XtX = X_aug.T @ X_aug
            XtX += lam * torch.eye(d_aug, dtype=XtX.dtype, device=XtX.device)
            XTE = X_aug.T @ Ec
            try:
                W_aug = torch.linalg.solve(XtX, XTE)
            except RuntimeError:
                W_aug = torch.pinverse(XtX) @ XTE
            W_hat = W_aug[:d_in, :].to(dtype=X.dtype)
            b_hat = W_aug[d_in, :].to(dtype=X.dtype)
            return W_hat, b_hat

        XtX = Xc.T @ Xc
        XtX += lam * torch.eye(d_in, dtype=XtX.dtype, device=XtX.device)
        XTE = Xc.T @ Ec
        try:
            W_hat = torch.linalg.solve(XtX, XTE)
        except RuntimeError:
            W_hat = torch.pinverse(XtX) @ XTE
        return W_hat.to(dtype=X.dtype), None

    def _model_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def calibrate_with_loader(
        self,
        layer_input_fn: Callable[[torch.Tensor], torch.Tensor],
        original_layer_fn: Callable[[torch.Tensor], torch.Tensor],
        pruned_layer_fn: Callable[[torch.Tensor], torch.Tensor],
        calibration_loader: torch.utils.data.DataLoader,
        lam: float = 1e-3,
        add_bias: bool = True,
        name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "LinearCalibration":
        from fermigates.calibration.linear_calibration import LinearCalibration

        device = device or self._model_device()
        X_list = []
        E_list = []

        self.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                if not isinstance(inputs, torch.Tensor):
                    raise TypeError("calibration_loader must yield Tensor inputs or (Tensor, target).")

                inputs = inputs.to(device)
                X_flat = layer_input_fn(inputs)
                f_orig = original_layer_fn(inputs)
                f_pruned = pruned_layer_fn(inputs)

                E = (f_orig - f_pruned).detach().cpu()
                X_list.append(X_flat.detach().cpu())
                E_list.append(E)

        if not X_list:
            raise ValueError("Calibration loader produced no data.")

        X = torch.cat(X_list, dim=0)
        E = torch.cat(E_list, dim=0)
        W_hat, b_hat = self.solve_ridge_cpu(X, E, lam=lam, add_bias=add_bias)

        calib = LinearCalibration(d_in=X.shape[1], d_out=E.shape[1], learnable=False, device=device)
        calib.load_calibration(W_hat.to(device), b_hat.to(device) if b_hat is not None else None)

        if name:
            self.calibrations[name] = calib
        return calib

    def attach_calibration(self, name: str, calib: "LinearCalibration") -> None:
        self.calibrations[name] = calib

    def apply_calibration(self, name: str, X_flat: torch.Tensor, y_pruned: torch.Tensor) -> torch.Tensor:
        if name not in self.calibrations:
            raise KeyError(f"Calibration '{name}' not found.")
        return y_pruned + self.calibrations[name](X_flat)

    def clear_calibrations(self) -> None:
        self.calibrations.clear()


class BaseFermiBackbone(BaseFermiModel, ABC):
    """Base class for reusable Fermi feature extractors."""

    @abstractmethod
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Produce backbone representations."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.encode(x, *args, **kwargs)


class BaseFermiClassifier(BaseFermiBackbone, ABC):
    """Base class for supervised classifiers built on top of Fermi components."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)

    @abstractmethod
    def logits(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return class logits."""

    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.logits(x, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.logits(x, *args, **kwargs)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x, *args, **kwargs).argmax(dim=-1)
