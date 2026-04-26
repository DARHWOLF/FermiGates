from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from fermigates.layers.linear import Linear


def _activation(name: str):
    key = name.lower()
    if key == "relu":
        return torch.relu
    if key == "tanh":
        return torch.tanh
    if key == "silu":
        return torch.nn.functional.silu
    if key == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unsupported activation '{name}'.")


class MLP(nn.Module):
    """Modular MLP architecture with optional pluggable gates.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : Sequence[int]
        Hidden layer dimensions.
    output_dim : int
        Output feature dimension.
    gate : callable or nn.Module or None, optional
        Gate factory or gate module template. When callable, it is invoked once
        per layer to build independent gate instances.
    loss : callable or None, optional
        Optional loss function stored as ``loss_fn``.
    calibration : nn.Module or None, optional
        Optional calibration module applied only on final logits.
    dropout : float, default=0.0
        Dropout probability between hidden layers.
    activation : str, default="gelu"
        Hidden activation function name.
    bias : bool, default=True
        Whether to include linear biases.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        gate: Callable[[], nn.Module] | nn.Module | None = None,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        calibration: nn.Module | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Step 1: Validate and store high-level configuration
        dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
        if any(v <= 0 for v in dims):
            raise ValueError("All dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_classes = int(output_dim)
        self.loss_fn = loss
        self.calibration = calibration
        self.act = _activation(activation)
        self.dropout = nn.Dropout(float(dropout))

        # Step 2: Resolve gate factory and build linear stack
        gate_factory = self._resolve_gate_factory(gate)
        self.layers = nn.ModuleList()
        for idx in range(len(dims) - 1):
            layer_shape = (dims[idx + 1], dims[idx])
            layer_gate = self._instantiate_gate(gate_factory, layer_shape)
            gate_kwargs = self._gate_kwargs(layer_gate)
            layer = Linear(
                in_features=dims[idx],
                out_features=dims[idx + 1],
                bias=bias,
                **gate_kwargs,
            )
            self.layers.append(layer)

    def _resolve_gate_factory(
        self,
        gate: Callable[[], nn.Module] | nn.Module | None,
    ) -> Callable[[], nn.Module] | None:
        """Resolve gate input into a layer-local factory."""

        # Step 1: disable gating when the user provides no gate
        if gate is None:
            return None

        # Step 2: clone module templates to avoid shared gate state across layers
        if isinstance(gate, nn.Module):
            return lambda: copy.deepcopy(gate)

        # Step 3: use callable gate factory directly
        return gate

    def _instantiate_gate(
        self,
        gate_factory: Callable[[], nn.Module] | None,
        shape: tuple[int, ...],
    ) -> nn.Module | None:
        """Instantiate one independent gate for a target layer."""

        # Step 1: keep gate disabled when factory is absent
        if gate_factory is None:
            return None

        # Step 2: support zero-arg and shape-aware factories explicitly
        signature = inspect.signature(gate_factory)
        if len(signature.parameters) == 0:
            return gate_factory()
        return gate_factory(shape)

    def _gate_kwargs(self, gate_module: nn.Module | None) -> dict[str, nn.Module | None]:
        """Map a gate module to the correct layer injection point."""

        # Step 1: keep all gate injection points empty when no gate exists
        if gate_module is None:
            return {
                "weight_gate": None,
                "input_gate": None,
                "output_gate": None,
            }

        # Step 2: choose injection point explicitly from gate mode
        mode = getattr(gate_module, "mode", "").lower()
        if mode == "weight":
            return {"weight_gate": gate_module, "input_gate": None, "output_gate": None}
        if mode in {"feature", "token", "input"}:
            return {"weight_gate": None, "input_gate": gate_module, "output_gate": None}
        return {"weight_gate": None, "input_gate": None, "output_gate": gate_module}

    def logits(
        self,
        x: torch.Tensor,
        return_gate_outputs: bool = False,
        return_masks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Run forward logits computation with optional gate output collection."""

        # Step 1: Flatten non-vector inputs for MLP processing
        if x.ndim > 2:
            x = x.flatten(1)

        # Step 2: Apply stacked linear layers with hidden activation and dropout
        gate_outputs: list[torch.Tensor | None] = []
        for idx, layer in enumerate(self.layers):
            x, gate_probs = layer(x)
            gate_outputs.append(gate_probs)
            if idx < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        # Step 3: Apply optional calibration only to final logits
        logits = self.calibration(x) if self.calibration is not None else x

        # Step 4: Return logits with optional gate traces
        if return_gate_outputs or return_masks:
            return logits, gate_outputs
        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_gate_outputs: bool = False,
        return_masks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass alias with compatibility flag ``return_masks``."""
        return self.logits(
            x,
            return_gate_outputs=return_gate_outputs,
            return_masks=return_masks,
        )

    def iter_gates(self):
        """Yield all gates attached to internal layers."""
        for layer in self.layers:
            for gate in (layer.weight_gate, layer.input_gate, layer.output_gate):
                if gate is not None:
                    yield gate

    def set_temperature(self, T_new: float) -> None:
        """Set temperature on attached gates when supported."""

        # Step 1: propagate temperature updates to every attached gate
        for gate in self.iter_gates():
            set_temperature_fn = getattr(gate, "set_temperature", None)
            if callable(set_temperature_fn):
                set_temperature_fn(T_new)

    def compute_sparsity(self, threshold: float = 0.5) -> tuple[int, int, float]:
        """Compute kept/total/fraction using weight-gate hard masks."""

        # Step 1: aggregate sparsity over weight-gated layers
        kept = 0
        total = 0
        for layer in self.layers:
            weight = layer.weight
            layer_total = int(weight.numel())
            gate = layer.weight_gate
            if gate is None:
                layer_kept = layer_total
            else:
                hard_mask_fn = getattr(gate, "hard_mask", None)
                if callable(hard_mask_fn):
                    hard = hard_mask_fn(weight, threshold=threshold)
                    layer_kept = int((hard >= threshold).sum().item())
                else:
                    layer_kept = layer_total
            kept += layer_kept
            total += layer_total

        # Step 2: return explicit kept/total/fraction tuple
        fraction = float(kept) / float(total) if total > 0 else 0.0
        return kept, total, fraction


class FermiMLPClassifier(MLP):
    """Compatibility wrapper exposing ``num_classes`` naming."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
        activation: str = "gelu",
        init_mu: float = 0.0,
        init_T: float = 1.0,
        gate: Callable[[], nn.Module] | nn.Module | None = None,
    ) -> None:
        del init_mu
        del init_T
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            gate=gate,
            dropout=dropout,
            activation=activation,
        )
