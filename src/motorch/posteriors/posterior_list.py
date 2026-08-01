"""Posterior composition for independent output groups."""

from collections.abc import Sequence

import torch

from motorch.exceptions import ShapeError
from motorch.posteriors.base import Posterior
from motorch.utils.validation import validate_same_dtype_device, validate_tensor


class PosteriorList:
    """Combine posteriors with matching batches and candidate dimensions.

    Components are treated as independent output groups and concatenated along
    the final output dimension. Their individual covariance structures are
    preserved within each component during sampling.
    """

    def __init__(self, *posteriors: Posterior) -> None:
        if not posteriors:
            raise ValueError("PosteriorList requires at least one posterior.")
        self._posteriors = tuple(posteriors)
        self._validate_components()

    @classmethod
    def from_sequence(cls, posteriors: Sequence[Posterior]) -> "PosteriorList":
        """Construct a posterior list from a sequence."""
        return cls(*posteriors)

    @property
    def posteriors(self) -> tuple[Posterior, ...]:
        """Return component posteriors in output order."""
        return self._posteriors

    @property
    def mean(self) -> torch.Tensor:
        """Concatenate component means along the output dimension."""
        return torch.cat(
            [posterior.mean for posterior in self._posteriors],
            dim=-1,
        )

    @property
    def variance(self) -> torch.Tensor:
        """Concatenate component marginal variances."""
        return torch.cat(
            [posterior.variance for posterior in self._posteriors],
            dim=-1,
        )

    @property
    def batch_shape(self) -> torch.Size:
        """Return shared leading batch dimensions."""
        return self._posteriors[0].mean.shape[:-2]

    @property
    def event_shape(self) -> torch.Size:
        """Return combined ``q x m`` event shape."""
        return self.mean.shape[-2:]

    @property
    def base_sample_shape(self) -> torch.Size:
        """Return required non-sample dimensions for combined base samples."""
        return self.mean.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the shared component dtype."""
        return self._posteriors[0].mean.dtype

    @property
    def device(self) -> torch.device:
        """Return the shared component device."""
        return self._posteriors[0].mean.device

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
        *,
        base_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample every component and concatenate outputs."""
        sample_shape = torch.Size(sample_shape)
        if base_samples is None:
            samples = [
                posterior.rsample(sample_shape) for posterior in self._posteriors
            ]
        else:
            expected_shape = sample_shape + self.base_sample_shape
            module = f"{type(self).__name__}.rsample"
            validate_tensor(base_samples, name="base_samples", module=module)
            validate_same_dtype_device(
                {"mean": self.mean, "base_samples": base_samples},
                module=module,
            )
            if base_samples.shape != expected_shape:
                raise ShapeError(
                    f"{module}: expected base_samples to have shape "
                    f"{tuple(expected_shape)}, but received "
                    f"{tuple(base_samples.shape)}."
                )
            output_sizes = [posterior.mean.shape[-1] for posterior in self._posteriors]
            component_base_samples = base_samples.split(output_sizes, dim=-1)
            samples = [
                posterior.rsample(sample_shape, base_samples=component_base)
                for posterior, component_base in zip(
                    self._posteriors,
                    component_base_samples,
                    strict=True,
                )
            ]
        return torch.cat(samples, dim=-1)

    def _validate_components(self) -> None:
        reference = self._posteriors[0].mean
        if reference.ndim < 2:
            raise ShapeError(
                "PosteriorList: expected every posterior mean to have shape "
                "batch_shape x q x m."
            )
        tensors: dict[str, torch.Tensor] = {}
        for index, posterior in enumerate(self._posteriors):
            mean = posterior.mean
            variance = posterior.variance
            validate_tensor(
                mean,
                name=f"posterior[{index}].mean",
                module="PosteriorList",
            )
            validate_tensor(
                variance,
                name=f"posterior[{index}].variance",
                module="PosteriorList",
            )
            if variance.shape != mean.shape:
                raise ShapeError(
                    "PosteriorList: expected each variance shape to match its mean "
                    f"shape, but posterior[{index}] has mean {tuple(mean.shape)} "
                    f"and variance {tuple(variance.shape)}."
                )
            if mean.shape[:-1] != reference.shape[:-1]:
                raise ShapeError(
                    "PosteriorList: expected all posterior means to share batch and "
                    f"candidate dimensions {tuple(reference.shape[:-1])}, but "
                    f"posterior[{index}] has {tuple(mean.shape[:-1])}."
                )
            tensors[f"posterior[{index}].mean"] = mean
            tensors[f"posterior[{index}].variance"] = variance
        validate_same_dtype_device(tensors, module="PosteriorList")
