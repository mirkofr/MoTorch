"""Base contract for posterior samplers."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class SampleablePosterior(Protocol):
    """Structural metadata required by generic posterior samplers."""

    @property
    def base_sample_shape(self) -> torch.Size:
        """Return non-sample dimensions required by base samples."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the posterior sample dtype."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the posterior sample device."""
        ...

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
        *,
        base_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw differentiable reparameterized samples."""
        ...


class PosteriorSampler(nn.Module, ABC):
    """Generate cached base samples for differentiable posterior sampling."""

    _base_samples: torch.Tensor

    def __init__(self, sample_shape: torch.Size, *, seed: int = 0) -> None:
        super().__init__()
        resolved_shape = torch.Size(sample_shape)
        if any(dimension < 1 for dimension in resolved_shape):
            raise ValueError(
                "PosteriorSampler: every sample_shape dimension must be positive, "
                f"but received {tuple(resolved_shape)}."
            )
        if seed < 0:
            raise ValueError(
                f"PosteriorSampler: seed must be non-negative, received {seed}."
            )
        self.sample_shape = resolved_shape
        self.seed = seed
        self.register_buffer("_base_samples", torch.empty(0), persistent=False)

    @property
    def base_samples(self) -> torch.Tensor | None:
        """Return currently cached base samples, or ``None`` before first use."""
        if self._base_samples.numel() == 0:
            return None
        return self._base_samples

    def reset_base_samples(self) -> None:
        """Clear cached base samples without changing the configured seed."""
        self._base_samples = torch.empty(
            0,
            dtype=self._base_samples.dtype,
            device=self._base_samples.device,
        )

    def forward(self, posterior: SampleablePosterior) -> torch.Tensor:
        """Draw posterior samples using cached standard-normal base samples."""
        if not isinstance(posterior, SampleablePosterior):
            raise TypeError(
                "PosteriorSampler.forward: posterior must provide "
                "base_sample_shape, dtype, device, and rsample()."
            )
        expected_shape = self.sample_shape + posterior.base_sample_shape
        if self._requires_new_base_samples(
            expected_shape,
            dtype=posterior.dtype,
            device=posterior.device,
        ):
            generated = self._construct_base_samples(
                expected_shape,
                dtype=posterior.dtype,
                device=posterior.device,
            )
            if generated.shape != expected_shape:
                raise RuntimeError(
                    f"{type(self).__name__}: generated base samples with shape "
                    f"{tuple(generated.shape)}, expected {tuple(expected_shape)}."
                )
            self._base_samples = generated
        return posterior.rsample(self.sample_shape, base_samples=self._base_samples)

    def _requires_new_base_samples(
        self,
        expected_shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            self._base_samples.shape != expected_shape
            or self._base_samples.dtype != dtype
            or self._base_samples.device != device
        )

    @abstractmethod
    def _construct_base_samples(
        self,
        shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct standard-normal base samples with the requested metadata."""
