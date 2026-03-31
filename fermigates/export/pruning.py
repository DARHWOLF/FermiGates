from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from fermigates.layers.conv_layers import FermiGatedConv2d
from fermigates.layers.linear_layers import FermiGatedLinear


@dataclass(frozen=True)
class LayerPruningReport:
    name: str
    layer_type: str
    total_weights: int
    kept_weights: int
    pruned_weights: int
    fraction_kept: float
    dense_macs: float
    kept_macs: float
    saved_macs: float
    saved_macs_fraction: float


@dataclass(frozen=True)
class ModelPruningReport:
    layers: list[LayerPruningReport]
    total_weights: int
    kept_weights: int
    pruned_weights: int
    fraction_kept: float
    dense_macs: float
    kept_macs: float
    saved_macs: float
    saved_macs_fraction: float


def _iter_prunable_layers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (FermiGatedLinear, FermiGatedConv2d)):
            yield name, module


def _weight_tensor(layer: FermiGatedLinear | FermiGatedConv2d) -> torch.Tensor:
    if isinstance(layer, FermiGatedLinear):
        return layer.linear.weight
    return layer.conv.weight


def _weight_key(name: str, layer: FermiGatedLinear | FermiGatedConv2d) -> str:
    suffix = "linear.weight" if isinstance(layer, FermiGatedLinear) else "conv.weight"
    return f"{name}.{suffix}" if name else suffix


def _hard_mask(layer: FermiGatedLinear | FermiGatedConv2d, threshold: float) -> torch.Tensor:
    w = _weight_tensor(layer)
    return layer.gate.hard_mask(w, threshold=threshold).to(dtype=w.dtype)


def _estimate_dense_macs(
    layer: FermiGatedLinear | FermiGatedConv2d,
    shape_info: tuple[tuple[int, ...], tuple[int, ...]] | None,
    fallback: int,
) -> float:
    if shape_info is None:
        return float(fallback)

    in_shape, out_shape = shape_info
    if isinstance(layer, FermiGatedLinear):
        in_features = layer.linear.in_features
        if in_features <= 0:
            return float(fallback)
        num_vectors = int(torch.tensor(in_shape).prod().item() // in_features)
        return float(num_vectors * layer.linear.in_features * layer.linear.out_features)

    if len(out_shape) != 4:
        return float(fallback)
    n, c_out, h_out, w_out = out_shape
    k_h, k_w = layer.conv.kernel_size
    c_in = layer.conv.in_channels
    per_output = (c_in // layer.conv.groups) * k_h * k_w
    return float(n * c_out * h_out * w_out * per_output)


def _normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if example_inputs is None:
        return ()
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


@torch.no_grad()
def hard_mask_module_weights_(model: nn.Module, threshold: float = 0.5) -> None:
    """In-place hard-masking of all prunable layer weights."""

    for _, layer in _iter_prunable_layers(model):
        w = _weight_tensor(layer)
        w.mul_(_hard_mask(layer, threshold=threshold))


@torch.no_grad()
def hard_masked_state_dict(model: nn.Module, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """Return a state-dict where gated weights are hard-masked."""

    state = copy.deepcopy(model.state_dict())
    for name, layer in _iter_prunable_layers(model):
        key = _weight_key(name, layer)
        w = _weight_tensor(layer)
        state[key] = (w * _hard_mask(layer, threshold=threshold)).detach().clone()
    return state


@torch.no_grad()
def to_hard_masked_model(model: nn.Module, threshold: float = 0.5) -> nn.Module:
    """Return a deep-copied model with hard masks applied to gated weights."""

    cloned = copy.deepcopy(model)
    hard_mask_module_weights_(cloned, threshold=threshold)
    return cloned


@torch.no_grad()
def pruning_report(
    model: nn.Module,
    threshold: float = 0.5,
    example_inputs: Any = None,
    example_kwargs: dict[str, Any] | None = None,
) -> ModelPruningReport:
    """Create parameter and MAC-estimate pruning report for gated layers."""

    example_kwargs = example_kwargs or {}
    shape_map: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    handles = []
    if example_inputs is not None:
        args = _normalize_example_inputs(example_inputs)

        def make_hook(layer_name: str):
            def hook(_module: nn.Module, inp: tuple[Any, ...], out: Any):
                if not inp:
                    return
                inp0 = inp[0]
                out0 = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(inp0, torch.Tensor) and isinstance(out0, torch.Tensor):
                    shape_map[layer_name] = (tuple(inp0.shape), tuple(out0.shape))

            return hook

        was_training = model.training
        for name, layer in _iter_prunable_layers(model):
            handles.append(layer.register_forward_hook(make_hook(name)))
        try:
            model.eval()
            model(*args, **example_kwargs)
        finally:
            if was_training:
                model.train()
            for handle in handles:
                handle.remove()

    layer_reports: list[LayerPruningReport] = []
    total_weights = 0
    kept_weights = 0
    dense_macs = 0.0
    kept_macs = 0.0

    for name, layer in _iter_prunable_layers(model):
        mask = _hard_mask(layer, threshold=threshold)
        total = mask.numel()
        kept = int(mask.sum().item())
        pruned = total - kept
        frac_kept = float(kept) / float(total) if total > 0 else 0.0

        dense = _estimate_dense_macs(layer, shape_map.get(name), fallback=total)
        kept_est = dense * frac_kept
        saved = dense - kept_est
        saved_frac = saved / dense if dense > 0 else 0.0

        layer_reports.append(
            LayerPruningReport(
                name=name,
                layer_type=layer.__class__.__name__,
                total_weights=total,
                kept_weights=kept,
                pruned_weights=pruned,
                fraction_kept=frac_kept,
                dense_macs=dense,
                kept_macs=kept_est,
                saved_macs=saved,
                saved_macs_fraction=saved_frac,
            )
        )

        total_weights += total
        kept_weights += kept
        dense_macs += dense
        kept_macs += kept_est

    pruned_weights = total_weights - kept_weights
    frac_kept = float(kept_weights) / float(total_weights) if total_weights > 0 else 0.0
    saved_macs = dense_macs - kept_macs
    saved_macs_frac = saved_macs / dense_macs if dense_macs > 0 else 0.0
    return ModelPruningReport(
        layers=layer_reports,
        total_weights=total_weights,
        kept_weights=kept_weights,
        pruned_weights=pruned_weights,
        fraction_kept=frac_kept,
        dense_macs=dense_macs,
        kept_macs=kept_macs,
        saved_macs=saved_macs,
        saved_macs_fraction=saved_macs_frac,
    )
