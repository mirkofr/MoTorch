"""Base model contract for probabilistic MoTorch models."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from motorch.posteriors import Posterior


class Model(nn.Module, ABC):
    """Base class for models that map candidate tensors to posteriors.

    Candidate inputs follow ``batch_shape x q x d``. Concrete models must
    document their output count, batching semantics, and observation-noise
    behavior.
    """

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Return the number of modeled outputs."""

    @abstractmethod
    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        """Construct a posterior for candidate inputs ``X``."""

    def forward(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        """Delegate module calls to :meth:`posterior`."""
        return self.posterior(X, observation_noise=observation_noise)
