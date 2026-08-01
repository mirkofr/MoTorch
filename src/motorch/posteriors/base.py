"""Core posterior protocol used by models, samplers, and objectives."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Posterior(Protocol):
    """Structural contract for probabilistic predictions.

    Posterior tensors use ``batch_shape x q x m`` for analytic moments and
    ``sample_shape x batch_shape x q x m`` for samples.
    """

    @property
    def mean(self) -> torch.Tensor:
        """Return the analytic posterior mean."""
        ...

    @property
    def variance(self) -> torch.Tensor:
        """Return the analytic marginal posterior variance."""
        ...

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
        *,
        base_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw differentiable reparameterized samples."""
        ...
