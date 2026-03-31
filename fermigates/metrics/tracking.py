from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from fermigates.base import BaseFermiLayer
from fermigates.losses import binary_entropy_loss


@dataclass(frozen=True)
class LayerOccupancyMetrics:
    name: str
    mean_probability: float
    std_probability: float
    entropy: float
    kept: int
    total: int
    fraction_kept: float


@dataclass(frozen=True)
class ModelOccupancyMetrics:
    layers: list[LayerOccupancyMetrics]
    kept: int
    total: int
    fraction_kept: float
    mean_entropy: float


@dataclass(frozen=True)
class FreeEnergyComponents:
    energy: float
    interaction: float
    entropy: float
    temperature: float
    free_energy: float


@torch.no_grad()
def collect_gate_metrics(
    model: nn.Module,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> ModelOccupancyMetrics:
    """Collect per-layer occupancy and entropy metrics from all Fermi layers."""

    layers: list[LayerOccupancyMetrics] = []
    kept_total = 0
    count_total = 0
    entropy_weighted = 0.0

    for name, module in model.named_modules():
        if not isinstance(module, BaseFermiLayer):
            continue

        probs = module.gate_probabilities().detach()
        total = probs.numel()
        kept = int((probs >= threshold).sum().item())
        frac = float(kept) / float(total) if total > 0 else 0.0
        entropy = float(binary_entropy_loss(probs, reduction="mean", eps=eps).item())
        layer = LayerOccupancyMetrics(
            name=name,
            mean_probability=float(probs.mean().item()),
            std_probability=float(probs.std(unbiased=False).item()),
            entropy=entropy,
            kept=kept,
            total=total,
            fraction_kept=frac,
        )
        layers.append(layer)
        kept_total += kept
        count_total += total
        entropy_weighted += entropy * total

    fraction_kept = float(kept_total) / float(count_total) if count_total > 0 else 0.0
    mean_entropy = entropy_weighted / float(count_total) if count_total > 0 else 0.0
    return ModelOccupancyMetrics(
        layers=layers,
        kept=kept_total,
        total=count_total,
        fraction_kept=fraction_kept,
        mean_entropy=mean_entropy,
    )


def free_energy_components(
    probabilities: torch.Tensor,
    energies: torch.Tensor,
    interaction: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
    eps: float = 1e-8,
) -> FreeEnergyComponents:
    """Compute scalar free-energy terms for logging and debugging."""

    if probabilities.shape != energies.shape:
        raise ValueError(
            "probabilities and energies must share shape, "
            f"got {probabilities.shape} and {energies.shape}."
        )
    if interaction is not None and interaction.shape != probabilities.shape:
        raise ValueError(
            "interaction must share shape with probabilities, "
            f"got {interaction.shape} and {probabilities.shape}."
        )

    p = probabilities.clamp(eps, 1.0 - eps)
    temp = torch.as_tensor(
        temperature,
        dtype=probabilities.dtype,
        device=probabilities.device,
    ).mean()
    energy = float((probabilities * energies).sum().item())
    interaction_term = 0.0
    if interaction is not None:
        interaction_term = float((0.5 * probabilities * interaction).sum().item())
    entropy = float((-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))).sum().item())
    free_energy = energy + interaction_term - (float(temp.item()) * entropy)
    return FreeEnergyComponents(
        energy=energy,
        interaction=interaction_term,
        entropy=entropy,
        temperature=float(temp.item()),
        free_energy=free_energy,
    )


class MetricsTracker:
    """Simple in-memory metric timeline for training loops."""

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def log(
        self,
        step: int,
        metrics: Mapping[str, float] | None = None,
        **kwargs: float,
    ) -> None:
        payload: dict[str, float] = {"step": float(step)}
        if metrics is not None:
            payload.update({k: float(v) for k, v in metrics.items()})
        payload.update({k: float(v) for k, v in kwargs.items()})
        self.records.append(payload)

    def log_gate_metrics(
        self,
        step: int,
        snapshot: ModelOccupancyMetrics,
        prefix: str = "gates",
    ) -> None:
        flat: dict[str, float] = {
            f"{prefix}.fraction_kept": snapshot.fraction_kept,
            f"{prefix}.mean_entropy": snapshot.mean_entropy,
            f"{prefix}.kept": float(snapshot.kept),
            f"{prefix}.total": float(snapshot.total),
        }
        for layer in snapshot.layers:
            key = layer.name.replace(".", "_")
            flat[f"{prefix}.{key}.fraction_kept"] = layer.fraction_kept
            flat[f"{prefix}.{key}.entropy"] = layer.entropy
            flat[f"{prefix}.{key}.mean_probability"] = layer.mean_probability
        self.log(step=step, metrics=flat)

    def latest(self) -> dict[str, float]:
        if not self.records:
            return {}
        return dict(self.records[-1])

    def history(self, metric_name: str) -> list[float]:
        return [record[metric_name] for record in self.records if metric_name in record]

    def as_dict(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        for record in self.records:
            for key, value in record.items():
                series.setdefault(key, []).append(value)
        return series
