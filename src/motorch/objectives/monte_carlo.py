"""Standard objectives for Monte Carlo acquisition functions."""

import torch

from motorch.objectives.base import MCAcquisitionObjective


class IdentityMCObjective(MCAcquisitionObjective):
    """Select the only posterior output as scalar sampled utility."""

    def forward(
        self,
        samples: torch.Tensor,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del X
        if not isinstance(samples, torch.Tensor):
            raise TypeError("IdentityMCObjective.forward: samples must be a torch.Tensor.")
        if samples.ndim < 2:
            raise ValueError(
                "IdentityMCObjective.forward: samples must have shape "
                "sample_shape x batch_shape x q x m."
            )
        if samples.shape[-1] != 1:
            raise ValueError(
                "IdentityMCObjective.forward: expected one posterior output, "
                f"received m={samples.shape[-1]}."
            )
        if not samples.is_floating_point():
            raise TypeError(
                "IdentityMCObjective.forward: samples must use a floating-point dtype."
            )
        return samples.squeeze(-1)
