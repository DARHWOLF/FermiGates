from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptiveBudgetController:
    """Online controller that adapts ``lambda_budget`` to hit target occupancy."""

    target_fraction_kept: float
    lambda_budget: float = 1e-3
    gain: float = 0.1
    ema_beta: float = 0.9
    tolerance: float = 0.0
    min_lambda: float = 0.0
    max_lambda: float = 10.0
    _ema_error: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_fraction_kept <= 1.0:
            raise ValueError("target_fraction_kept must be in [0, 1].")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1).")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative.")
        if self.min_lambda < 0.0:
            raise ValueError("min_lambda must be non-negative.")
        if self.max_lambda < self.min_lambda:
            raise ValueError("max_lambda must be >= min_lambda.")

    def set_target(self, target_fraction_kept: float) -> None:
        if not 0.0 <= target_fraction_kept <= 1.0:
            raise ValueError("target_fraction_kept must be in [0, 1].")
        self.target_fraction_kept = float(target_fraction_kept)

    def update(self, fraction_kept: float) -> float:
        """Update and return ``lambda_budget`` based on current sparsity."""

        if not 0.0 <= fraction_kept <= 1.0:
            raise ValueError("fraction_kept must be in [0, 1].")

        error = float(fraction_kept) - float(self.target_fraction_kept)
        if abs(error) <= self.tolerance:
            return float(self.lambda_budget)

        self._ema_error = (self.ema_beta * self._ema_error) + ((1.0 - self.ema_beta) * error)
        updated = float(self.lambda_budget) + (self.gain * self._ema_error)
        self.lambda_budget = min(self.max_lambda, max(self.min_lambda, updated))
        return float(self.lambda_budget)

    def state_dict(self) -> dict[str, float]:
        return {
            "target_fraction_kept": float(self.target_fraction_kept),
            "lambda_budget": float(self.lambda_budget),
            "ema_error": float(self._ema_error),
        }

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self.target_fraction_kept = float(state_dict["target_fraction_kept"])
        self.lambda_budget = float(state_dict["lambda_budget"])
        self._ema_error = float(state_dict.get("ema_error", 0.0))
