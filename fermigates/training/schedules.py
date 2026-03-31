from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ScheduleMode = Literal["linear", "cosine", "exponential"]


@dataclass(frozen=True)
class AnnealingSchedule:
    """Scalar annealing schedule with multiple interpolation modes."""

    start: float
    end: float
    total_steps: int
    mode: ScheduleMode = "linear"
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if self.mode == "exponential" and (self.start <= 0.0 or self.end <= 0.0):
            raise ValueError("exponential schedules require start and end to be positive.")

    def value(self, step: int) -> float:
        """Return scheduled value at global ``step``."""

        if step <= self.start_step:
            return float(self.start)
        if step >= self.start_step + self.total_steps:
            return float(self.end)

        progress = (step - self.start_step) / float(self.total_steps)
        if self.mode == "linear":
            return float(self.start + progress * (self.end - self.start))
        if self.mode == "cosine":
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
            return float(self.start + scale * (self.end - self.start))
        if self.mode == "exponential":
            log_start = math.log(self.start)
            log_end = math.log(self.end)
            return float(math.exp(log_start + progress * (log_end - log_start)))
        raise ValueError(f"Unsupported mode '{self.mode}'.")


@dataclass(frozen=True)
class AnnealingState:
    """Resolved schedule values for a given training step."""

    temperature: float
    lambda_free_energy: float
    budget_target: float


@dataclass
class FermiAnnealingPlan:
    """Bundle schedules for core Fermi training controls."""

    temperature: AnnealingSchedule | None = None
    lambda_free_energy: AnnealingSchedule | None = None
    budget_target: AnnealingSchedule | None = None

    def value(
        self,
        step: int,
        *,
        default_temperature: float = 1.0,
        default_lambda_free_energy: float = 0.0,
        default_budget_target: float = 1.0,
    ) -> AnnealingState:
        return AnnealingState(
            temperature=(
                self.temperature.value(step)
                if self.temperature is not None
                else float(default_temperature)
            ),
            lambda_free_energy=(
                self.lambda_free_energy.value(step)
                if self.lambda_free_energy is not None
                else float(default_lambda_free_energy)
            ),
            budget_target=(
                self.budget_target.value(step)
                if self.budget_target is not None
                else float(default_budget_target)
            ),
        )
