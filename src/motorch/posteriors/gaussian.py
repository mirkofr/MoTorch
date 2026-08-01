"""Tensor-native multivariate Gaussian posterior implementation."""

import torch

from motorch.exceptions import ShapeError, TensorValidationError
from motorch.utils.validation import validate_same_dtype_device, validate_tensor


class GaussianPosterior:
    """A batched multivariate Gaussian posterior over ``q x m`` events.

    Parameters
    ----------
    mean:
        Tensor with shape ``batch_shape x q x m``.
    covariance_matrix:
        Full event covariance with shape
        ``batch_shape x (q * m) x (q * m)``. Positive-semidefinite matrices,
        including singular covariances, are supported.

    Notes
    -----
    Event dimensions are flattened in row-major ``q x m`` order. The class
    never casts, moves, clones, or detaches user tensors.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        covariance_matrix: torch.Tensor,
    ) -> None:
        module = type(self).__name__
        validate_tensor(mean, name="mean", module=module)
        validate_tensor(
            covariance_matrix,
            name="covariance_matrix",
            module=module,
        )
        validate_same_dtype_device(
            {"mean": mean, "covariance_matrix": covariance_matrix},
            module=module,
        )
        if mean.ndim < 2:
            raise ShapeError(
                f"{module}: expected mean to have shape batch_shape x q x m, "
                f"but received {tuple(mean.shape)}."
            )
        event_size = mean.shape[-2] * mean.shape[-1]
        expected_covariance_shape = (*mean.shape[:-2], event_size, event_size)
        if covariance_matrix.shape != expected_covariance_shape:
            raise ShapeError(
                f"{module}: expected covariance_matrix to have shape "
                f"{expected_covariance_shape} for mean shape {tuple(mean.shape)}, "
                f"but received {tuple(covariance_matrix.shape)}."
            )
        if not torch.allclose(
            covariance_matrix,
            covariance_matrix.transpose(-1, -2),
            rtol=self._symmetry_rtol(mean.dtype),
            atol=self._symmetry_atol(mean.dtype),
        ):
            raise TensorValidationError(
                f"{module}: expected covariance_matrix to be symmetric."
            )
        minimum_eigenvalue = torch.linalg.eigvalsh(covariance_matrix).amin()
        tolerance = self._psd_tolerance(covariance_matrix)
        if bool(minimum_eigenvalue < -tolerance):
            raise TensorValidationError(
                f"{module}: expected covariance_matrix to be positive "
                "semidefinite, but its minimum eigenvalue is "
                f"{minimum_eigenvalue.detach().item():.6g}."
            )
        self._mean = mean
        self._covariance_matrix = covariance_matrix

    @property
    def mean(self) -> torch.Tensor:
        """Return the original mean tensor."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Return marginal variances with the same shape as ``mean``."""
        return self._covariance_matrix.diagonal(dim1=-2, dim2=-1).reshape(
            self._mean.shape
        )

    @property
    def covariance_matrix(self) -> torch.Tensor:
        """Return the full flattened-event covariance matrix."""
        return self._covariance_matrix

    @property
    def batch_shape(self) -> torch.Size:
        """Return leading posterior batch dimensions."""
        return self._mean.shape[:-2]

    @property
    def event_shape(self) -> torch.Size:
        """Return the ``q x m`` event shape."""
        return self._mean.shape[-2:]

    @property
    def base_sample_shape(self) -> torch.Size:
        """Return the required non-sample dimensions for base samples."""
        return self._mean.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the posterior tensor dtype."""
        return self._mean.dtype

    @property
    def device(self) -> torch.device:
        """Return the posterior tensor device."""
        return self._mean.device

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
        *,
        base_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw differentiable samples, optionally from supplied base samples."""
        sample_shape = torch.Size(sample_shape)
        expected_shape = sample_shape + self.base_sample_shape
        if base_samples is None:
            base_samples = torch.randn(
                expected_shape,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._validate_base_samples(base_samples, expected_shape)

        flat_base_samples = base_samples.reshape(
            *sample_shape,
            *self.batch_shape,
            -1,
        )
        root = self._covariance_root()
        transformed = torch.matmul(root, flat_base_samples.unsqueeze(-1)).squeeze(-1)
        flat_mean = self._mean.reshape(*self.batch_shape, -1)
        return (flat_mean + transformed).reshape(expected_shape)

    def _validate_base_samples(
        self,
        base_samples: torch.Tensor,
        expected_shape: torch.Size,
    ) -> None:
        module = f"{type(self).__name__}.rsample"
        validate_tensor(base_samples, name="base_samples", module=module)
        validate_same_dtype_device(
            {"mean": self._mean, "base_samples": base_samples},
            module=module,
        )
        if base_samples.shape != expected_shape:
            raise ShapeError(
                f"{module}: expected base_samples to have shape "
                f"{tuple(expected_shape)}, but received {tuple(base_samples.shape)}."
            )

    def _covariance_root(self) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(self._covariance_matrix)
        tolerance = self._psd_tolerance(self._covariance_matrix)
        nonnegative_eigenvalues = eigenvalues.clamp_min(0.0)
        nonnegative_eigenvalues = torch.where(
            eigenvalues >= -tolerance,
            nonnegative_eigenvalues,
            eigenvalues,
        )
        return eigenvectors * nonnegative_eigenvalues.sqrt().unsqueeze(-2)

    @staticmethod
    def _symmetry_rtol(dtype: torch.dtype) -> float:
        return 1e-5 if dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-10

    @staticmethod
    def _symmetry_atol(dtype: torch.dtype) -> float:
        return 1e-6 if dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-12

    @staticmethod
    def _psd_tolerance(matrix: torch.Tensor) -> torch.Tensor:
        event_size = matrix.shape[-1]
        scale = matrix.abs().amax().clamp_min(1.0)
        return scale * torch.finfo(matrix.dtype).eps * event_size * 10
