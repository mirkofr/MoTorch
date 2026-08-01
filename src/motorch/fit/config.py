"""Typed configuration for model fitting."""

from dataclasses import dataclass
from typing import Literal

OptimizerName = Literal["adam", "lbfgs"]


@dataclass(frozen=True, slots=True)
class FitOptions:
    """Configure exact-GP marginal-likelihood fitting.

    Parameters are explicit so numerical behavior can be reproduced and
    reviewed. ``max_retries`` counts additional attempts after the first.
    """

    optimizer: OptimizerName = "adam"
    learning_rate: float = 0.05
    max_steps: int = 100
    tolerance_grad: float = 1e-5
    tolerance_loss: float = 1e-6
    patience: int = 10
    max_retries: int = 2
    retry_learning_rate_factor: float = 0.5
    initial_jitter: float = 1e-6
    jitter_multiplier: float = 10.0
    max_jitter: float = 1e-2
    seed: int = 0
    deterministic: bool = False
    warn_on_retry: bool = True
    warn_on_failure: bool = True

    def __post_init__(self) -> None:
        """Validate configuration eagerly with actionable errors."""
        if self.optimizer not in {"adam", "lbfgs"}:
            raise ValueError(
                "FitOptions.optimizer must be 'adam' or 'lbfgs', "
                f"but received {self.optimizer!r}."
            )
        positive = {
            "learning_rate": self.learning_rate,
            "tolerance_grad": self.tolerance_grad,
            "tolerance_loss": self.tolerance_loss,
            "retry_learning_rate_factor": self.retry_learning_rate_factor,
            "initial_jitter": self.initial_jitter,
            "jitter_multiplier": self.jitter_multiplier,
            "max_jitter": self.max_jitter,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(
                    f"FitOptions.{name} must be positive, received {value}."
                )
        integer_positive = {
            "max_steps": self.max_steps,
            "patience": self.patience,
        }
        for name, value in integer_positive.items():
            if value < 1:
                raise ValueError(
                    f"FitOptions.{name} must be at least 1, received {value}."
                )
        if self.max_retries < 0:
            raise ValueError(
                "FitOptions.max_retries must be non-negative, "
                f"received {self.max_retries}."
            )
        if self.seed < 0:
            raise ValueError(
                f"FitOptions.seed must be non-negative, received {self.seed}."
            )
        if self.retry_learning_rate_factor > 1:
            raise ValueError(
                "FitOptions.retry_learning_rate_factor must not exceed 1, "
                f"received {self.retry_learning_rate_factor}."
            )
        if self.jitter_multiplier <= 1:
            raise ValueError(
                "FitOptions.jitter_multiplier must exceed 1, "
                f"received {self.jitter_multiplier}."
            )
        if self.max_jitter < self.initial_jitter:
            raise ValueError(
                "FitOptions.max_jitter must be greater than or equal to "
                f"initial_jitter, received {self.max_jitter} < {self.initial_jitter}."
            )
