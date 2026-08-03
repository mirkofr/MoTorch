"""Base contracts for sampled Monte Carlo objectives."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class MCAcquisitionObjective(nn.Module, ABC):
    """Transform posterior samples into scalar utilities per candidate.

    Implementations receive samples with shape
    ``sample_shape x batch_shape x q x m`` and must return utilities with shape
    ``sample_shape x batch_shape x q``.
    """

    @abstractmethod
    def forward(
        self,
        samples: torch.Tensor,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scalar sampled utilities for every candidate point."""
