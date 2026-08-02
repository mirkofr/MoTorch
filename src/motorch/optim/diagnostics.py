"""Structured diagnostics for acquisition-function optimization."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RestartResult:
    """Result from one local optimization restart."""

    index: int
    success: bool
    candidate: torch.Tensor | None
    value: torch.Tensor | None
    steps: int
    message: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Best candidates plus diagnostics for all attempted restarts."""

    candidates: torch.Tensor
    values: torch.Tensor
    restarts: tuple[RestartResult, ...]

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.candidates
        yield self.values
