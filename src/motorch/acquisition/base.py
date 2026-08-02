"""Base contracts for acquisition functions."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from motorch.models import Model


class AcquisitionFunction(nn.Module, ABC):
    """Base class for utilities evaluated on candidate tensors.

    Candidate inputs follow ``batch_shape x q x d`` and returned values follow
    ``batch_shape`` for the analytic acquisition functions introduced in Phase 6.
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        if not isinstance(model, Model):
            raise TypeError(
                "AcquisitionFunction: model must be an instance of motorch.models.Model."
            )
        self.model = model

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate acquisition utility at candidate tensor ``X``."""
