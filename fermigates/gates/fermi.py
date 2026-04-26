import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ._base import BaseGate


class FermiGate(BaseGate):
    """Fermi-style gate supporting multiple masking granularities.

    Parameters
    ----------
    shape : tuple[int, ...] or None, optional
        Reference tensor shape used to initialize gate parameters. If ``None``,
        parameters are initialized lazily from the first forward input.
    mode : {"elementwise", "neuron", "head", "channel", "block", "lowrank"}, default="elementwise"
        Granularity of the learnable ``mu`` parameters.
    rank : int or None, optional
        Rank used only when ``mode="lowrank"``.
    init_mu : float, default=0.0
        Initial value for ``mu`` parameters.
    init_temperature : float, default=1.0
        Initial gate temperature.
    annealer : str or None, default="linear"
        Annealing schedule name.
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        mode: str = "elementwise",
        rank: int | None = None,
        init_mu: float = 0.0,
        init_temperature: float = 1.0,
        annealer: str | None = "linear",
    ) -> None:
        super().__init__(annealer=annealer)

        # Step 1: store static configuration
        self.shape = tuple(shape) if shape is not None else None
        self.mode = mode
        self.rank = rank
        self._init_mu = float(init_mu)
        self.temperature = float(init_temperature)
        self._init_temperature = float(init_temperature)
        self._initialized = False

        # Step 2: initialize gate parameters when shape is available
        if self.shape is not None:
            self._initialize_parameters(self.shape)

    def _initialize_parameters(self, shape: Sequence[int]) -> None:
        """Create learnable parameters from the resolved shape.

        Parameters
        ----------
        shape : Sequence[int]
            Resolved input shape for the gate.
        """

        # Step 1: normalize shape and validate
        resolved_shape = tuple(int(v) for v in shape)
        if len(resolved_shape) == 0:
            raise ValueError("shape must contain at least one dimension.")

        # Step 2: create mode-specific parameters
        if self.mode == "elementwise":
            self.mu = nn.Parameter(torch.full(resolved_shape, self._init_mu))
        elif self.mode == "neuron":
            self.mu = nn.Parameter(torch.full((resolved_shape[-1],), self._init_mu))
        elif self.mode == "head":
            if len(resolved_shape) < 2:
                raise ValueError("head mode requires at least 2 dimensions.")
            self.mu = nn.Parameter(torch.full((resolved_shape[1],), self._init_mu))
        elif self.mode == "channel":
            if len(resolved_shape) < 2:
                raise ValueError("channel mode requires at least 2 dimensions.")
            self.mu = nn.Parameter(torch.full((resolved_shape[1],), self._init_mu))
        elif self.mode == "block":
            self.mu = nn.Parameter(torch.tensor(self._init_mu))
        elif self.mode == "lowrank":
            if len(resolved_shape) != 2:
                raise ValueError("lowrank mode requires exactly 2 dimensions.")
            if self.rank is None:
                raise ValueError("rank must be specified for lowrank mode.")
            d1, d2 = resolved_shape
            self.U = nn.Parameter(torch.randn(d1, self.rank) * 0.01)
            self.V = nn.Parameter(torch.randn(d2, self.rank) * 0.01)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Step 3: persist resolution state
        self.shape = resolved_shape
        self._initialized = True

    def _maybe_initialize_from_input(self, x: torch.Tensor) -> None:
        """Initialize parameters lazily from a runtime input tensor."""

        # Step 1: initialize exactly once
        if self._initialized:
            return

        # Step 2: infer shape and initialize
        self._initialize_parameters(tuple(x.shape))

    # ---------------------------------------------------------
    # μ computation
    # ---------------------------------------------------------
    def _compute_mu(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_initialize_from_input(x)

        # Step 1: elementwise
        if self.mode == "elementwise":
            return self.mu

        # Step 2: neuron (last dim)
        elif self.mode == "neuron":
            return self.mu.view(*([1] * (x.dim() - 1)), -1)

        # Step 3: head (dim=1)
        elif self.mode == "head" or self.mode == "channel":
            return self.mu.view(1, -1, *([1] * (x.dim() - 2)))

        # Step 5: block
        elif self.mode == "block":
            return self.mu

        # Step 6: low-rank
        elif self.mode == "lowrank":
            return self.U @ self.V.T

        else:
            raise RuntimeError("Invalid mode")

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self._compute_mu(x)

        z = (x - mu) / self.temperature
        z = torch.clamp(z, -20, 20)  # numerical stability

        return torch.sigmoid(z)

    # ---------------------------------------------------------
    # Annealing
    # ---------------------------------------------------------
    def _initialize_annealing(self) -> None:
        config = self.annealing_config

        if "T0" not in config:
            config["T0"] = self._init_temperature

        if "T_min" not in config:
            config["T_min"] = 0.01 * config["T0"]

        if "total_steps" not in config:
            config["total_steps"] = 10000

        self.annealing_config = config

    def _apply_annealing(self, step: int) -> None:
        config = self.annealing_config

        if self.annealer == "linear":
            T0 = config["T0"]
            T_min = config["T_min"]
            total_steps = config["total_steps"]

            value = max(
                T_min,
                T0 - (T0 - T_min) * (step / total_steps)
            )

        elif self.annealer == "exponential":
            T0 = config.get("T0", self._init_temperature)
            decay = config.get("decay", 0.99)
            value = T0 * (decay ** step)

        elif self.annealer == "cosine":
            T0 = config["T0"]
            T_min = config["T_min"]
            total_steps = config["total_steps"]

            value = T_min + 0.5 * (T0 - T_min) * (
                1 + math.cos(math.pi * step / total_steps)
            )

        else:
            raise ValueError(f"Unknown annealer type: {self.annealer}")

        self._update_annealed_parameters(value)

    def _update_annealed_parameters(self, value: float) -> None:
        self.temperature = max(float(value), 1e-6)

    def set_temperature(self, value: float) -> None:
        """Set temperature explicitly.

        Parameters
        ----------
        value : float
            New positive temperature value.
        """
        if value <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = float(value)

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    @torch.no_grad()
    def initialize_mu_from_reference(
        self,
        reference: torch.Tensor,
        percentile: float = 0.5
    ) -> None:

        self._maybe_initialize_from_input(reference)

        if self.mode == "elementwise":
            val = torch.quantile(reference, percentile)
            self.mu.copy_(torch.full_like(reference, val))

        elif self.mode in ["neuron", "channel"]:
            dims = tuple(range(reference.dim() - 1))
            vals = torch.quantile(reference, percentile, dim=dims)
            self.mu.copy_(vals)

        elif self.mode == "head":
            dims = tuple(i for i in range(reference.dim()) if i != 1)
            vals = torch.quantile(reference, percentile, dim=dims)
            self.mu.copy_(vals)

        elif self.mode == "block":
            val = torch.quantile(reference, percentile)
            self.mu.copy_(torch.tensor(val))

        elif self.mode == "lowrank":
            # initialize U,V to approximate percentile baseline
            val = torch.quantile(reference, percentile)
            nn.init.constant_(self.U, val / self.rank)
            nn.init.constant_(self.V, 1.0)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, temperature={self.temperature}"
