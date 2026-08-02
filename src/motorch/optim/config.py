"""Typed configuration for acquisition-function optimization."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class OptimizeAcqfOptions:
    """Numerical controls for local acquisition optimization."""

    optimizer: Literal["adam", "lbfgs"] = "adam"
    max_steps: int = 200
    learning_rate: float = 0.05
    tolerance: float = 1e-8
    patience: int = 25
    gradient_tolerance: float = 1e-7
    lbfgs_history_size: int = 20
    clamp_epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("OptimizeAcqfOptions.max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("OptimizeAcqfOptions.learning_rate must be positive.")
        if self.tolerance < 0:
            raise ValueError("OptimizeAcqfOptions.tolerance must be non-negative.")
        if self.patience < 1:
            raise ValueError("OptimizeAcqfOptions.patience must be positive.")
        if self.gradient_tolerance < 0:
            raise ValueError(
                "OptimizeAcqfOptions.gradient_tolerance must be non-negative."
            )
        if self.lbfgs_history_size < 1:
            raise ValueError(
                "OptimizeAcqfOptions.lbfgs_history_size must be positive."
            )
        if not 0 < self.clamp_epsilon < 0.5:
            raise ValueError(
                "OptimizeAcqfOptions.clamp_epsilon must lie strictly between 0 and 0.5."
            )
