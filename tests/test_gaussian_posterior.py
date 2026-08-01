import pytest
import torch

from motorch.exceptions import DTypeError, ShapeError, TensorValidationError
from motorch.posteriors import GaussianPosterior, Posterior


def make_covariance(
    batch_shape: torch.Size,
    event_size: int,
    *,
    dtype: torch.dtype = torch.double,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(
        *batch_shape,
        event_size,
        event_size,
        generator=generator,
        dtype=dtype,
    )
    identity = torch.eye(event_size, dtype=dtype)
    return matrix @ matrix.transpose(-1, -2) + 0.4 * identity


def test_gaussian_posterior_satisfies_protocol_and_exposes_shapes() -> None:
    mean = torch.zeros(2, 3, 4, dtype=torch.double)
    covariance = make_covariance(torch.Size([2]), 12)
    posterior = GaussianPosterior(mean, covariance)

    assert isinstance(posterior, Posterior)
    assert posterior.mean is mean
    assert posterior.covariance_matrix is covariance
    assert posterior.batch_shape == torch.Size([2])
    assert posterior.event_shape == torch.Size([3, 4])
    assert posterior.base_sample_shape == mean.shape
    assert posterior.dtype == torch.double
    assert posterior.device.type == "cpu"
    assert posterior.variance.shape == mean.shape
    assert torch.equal(
        posterior.variance.reshape(2, 12),
        covariance.diagonal(dim1=-2, dim2=-1),
    )


def test_gaussian_posterior_rsample_preserves_sample_and_batch_shapes() -> None:
    mean = torch.zeros(2, 1, 3, 2, dtype=torch.double)
    covariance = make_covariance(torch.Size([2, 1]), 6)
    posterior = GaussianPosterior(mean, covariance)

    samples = posterior.rsample(torch.Size([5, 7]))

    assert samples.shape == torch.Size([5, 7, 2, 1, 3, 2])
    assert samples.dtype == torch.double
    assert samples.device.type == "cpu"


def test_gaussian_posterior_base_samples_are_exactly_reproducible() -> None:
    mean = torch.randn(2, 3, dtype=torch.double)
    covariance = make_covariance(torch.Size(), 6)
    posterior = GaussianPosterior(mean, covariance)
    base_samples = torch.randn(13, 2, 3, dtype=torch.double)

    first = posterior.rsample(torch.Size([13]), base_samples=base_samples)
    second = posterior.rsample(torch.Size([13]), base_samples=base_samples)

    assert torch.equal(first, second)


def test_gaussian_posterior_supports_singular_covariance() -> None:
    mean = torch.tensor([[[1.0], [2.0]]], dtype=torch.double)
    covariance = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.double)
    posterior = GaussianPosterior(mean, covariance)
    base_samples = torch.tensor([[[[0.5], [-0.5]]]], dtype=torch.double)

    samples = posterior.rsample(torch.Size([1]), base_samples=base_samples)

    assert samples.shape == torch.Size([1, 1, 2, 1])
    centered = samples - mean
    assert torch.allclose(centered[..., 0, 0], centered[..., 1, 0])


def test_gaussian_posterior_accepts_float32() -> None:
    mean = torch.zeros(2, 1, dtype=torch.float32)
    covariance = torch.eye(2, dtype=torch.float32)

    posterior = GaussianPosterior(mean, covariance)

    assert posterior.rsample(torch.Size([3])).dtype == torch.float32


def test_gaussian_posterior_rejects_invalid_mean_rank() -> None:
    with pytest.raises(ShapeError, match=r"batch_shape x q x m.*\(3,\)"):
        GaussianPosterior(torch.zeros(3), torch.eye(3))


def test_gaussian_posterior_rejects_zero_sized_event_dimensions() -> None:
    with pytest.raises(ShapeError, match="q and m to be positive"):
        GaussianPosterior(
            torch.zeros(0, 1, dtype=torch.double),
            torch.empty(0, 0, dtype=torch.double),
        )


def test_gaussian_posterior_rejects_covariance_shape_mismatch() -> None:
    mean = torch.zeros(2, 3, dtype=torch.double)
    with pytest.raises(ShapeError, match=r"\(6, 6\).*(5, 5)"):
        GaussianPosterior(mean, torch.eye(5, dtype=torch.double))


def test_gaussian_posterior_rejects_dtype_mismatch() -> None:
    mean = torch.zeros(2, 1, dtype=torch.float32)
    covariance = torch.eye(2, dtype=torch.float64)
    with pytest.raises(DTypeError, match="covariance_matrix.*torch.float32.*mean"):
        GaussianPosterior(mean, covariance)


def test_gaussian_posterior_rejects_unsupported_low_precision_dtype() -> None:
    mean = torch.zeros(2, 1, dtype=torch.float16)
    covariance = torch.eye(2, dtype=torch.float16)
    with pytest.raises(DTypeError, match="torch.float32 or torch.float64"):
        GaussianPosterior(mean, covariance)


def test_gaussian_posterior_rejects_non_symmetric_covariance() -> None:
    mean = torch.zeros(2, 1, dtype=torch.double)
    covariance = torch.tensor([[1.0, 0.2], [0.3, 1.0]], dtype=torch.double)
    with pytest.raises(TensorValidationError, match="symmetric"):
        GaussianPosterior(mean, covariance)


def test_gaussian_posterior_rejects_indefinite_covariance() -> None:
    mean = torch.zeros(2, 1, dtype=torch.double)
    covariance = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    with pytest.raises(TensorValidationError, match="positive semidefinite"):
        GaussianPosterior(mean, covariance)


def test_gaussian_posterior_rejects_invalid_base_sample_shape() -> None:
    posterior = GaussianPosterior(
        torch.zeros(2, 1, dtype=torch.double),
        torch.eye(2, dtype=torch.double),
    )
    with pytest.raises(ShapeError, match=r"\(4, 2, 1\).*(4, 2)"):
        posterior.rsample(
            torch.Size([4]),
            base_samples=torch.zeros(4, 2, dtype=torch.double),
        )
