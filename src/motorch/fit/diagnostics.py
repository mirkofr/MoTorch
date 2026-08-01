"""Structured diagnostics returned by fitting utilities."""

from dataclasses import dataclass
from enum import Enum


class FitTermination(str, Enum):
    """Reason a fitting attempt or complete fit terminated."""

    GRADIENT_TOLERANCE = "gradient_tolerance"
    LOSS_TOLERANCE = "loss_tolerance"
    MAX_STEPS = "max_steps"
    NONFINITE_LOSS = "nonfinite_loss"
    NONFINITE_GRADIENT = "nonfinite_gradient"
    NUMERICAL_ERROR = "numerical_error"


@dataclass(frozen=True, slots=True)
class FitAttempt:
    """Diagnostics for one optimizer attempt."""

    attempt: int
    optimizer: str
    learning_rate: float
    jitter: float
    steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    max_gradient: float
    converged: bool
    termination: FitTermination
    message: str


@dataclass(frozen=True, slots=True)
class FitResult:
    """Complete fitting result with per-attempt diagnostics."""

    converged: bool
    termination: FitTermination
    initial_loss: float
    final_loss: float
    best_loss: float
    total_steps: int
    attempts: tuple[FitAttempt, ...]

    @property
    def retries(self) -> int:
        """Return the number of attempts after the initial attempt."""
        return max(0, len(self.attempts) - 1)

    @property
    def message(self) -> str:
        """Return the final actionable diagnostic message."""
        return self.attempts[-1].message
