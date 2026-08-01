import pytest
import torch

from motorch.exceptions import DTypeError, TensorValidationError
from motorch.posteriors import GaussianPosterior, Posterior, PosteriorList


class DeterministicPosterior:
    """Small structural posterior used to verify protocol interoperability."""

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    @property
    def mean(self) -> torch.Tensor:
        return self._value

    @property
    def variance(self) -> torch.Tensor:
        return torch.zeros_like(self._value)

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
        *,
        base_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_shape = sample_shape + self._value.shape
        if base_samples is not None and base_samples.shape != expected_shape:
            raise ValueError("invalid base-sample shape")
        return self._value.expand(expected_shape)


def test_posterior_protocol_accepts_structural_implementation() -> None:
    posterior = DeterministicPosterior(torch.ones(2, 1, dtype=torch.double))

    assert isinstance(posterior, Posterior)
    assert posterior.rsample(torch.Size([3])).shape == torch.Size([3, 2, 1])


def test_gaussian_sample_gradient_matches_finite_difference() -> None:
    base_sample = torch.tensor([[[0.75]]], dtype=torch.double)
    scale = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    covariance = scale.square().reshape(1, 1)
    posterior = GaussianPosterior(
        torch.zeros(1, 1, dtype=torch.double),
        covariance,
    )
    value = posterior.rsample(
        torch.Size([1]),
        base_samples=base_sample,
    ).sum()
    value.backward()
    assert scale.grad is not None

    step = 1e-6

    def evaluate(candidate_scale: float) -> float:
        candidate = torch.tensor(candidate_scale, dtype=torch.double)
        candidate_posterior = GaussianPosterior(
            torch.zeros(1, 1, dtype=torch.double),
            candidate.square().reshape(1, 1),
        )
        return float(
            candidate_posterior.rsample(
                torch.Size([1]),
                base_samples=base_sample,
            ).sum()
        )

    finite_difference = (evaluate(1.3 + step) - evaluate(1.3 - step)) / (2 * step)

    assert float(scale.grad) == pytest.approx(finite_difference, rel=1e-6, abs=1e-7)


def test_gaussian_posterior_rejects_non_finite_parameters() -> None:
    with pytest.raises(TensorValidationError, match="finite values"):
        GaussianPosterior(
            torch.tensor([[float("nan")]], dtype=torch.double),
            torch.ones(1, 1, dtype=torch.double),
        )

    with pytest.raises(TensorValidationError, match="finite values"):
        GaussianPosterior(
            torch.zeros(1, 1, dtype=torch.double),
            torch.tensor([[float("inf")]], dtype=torch.double),
        )


def test_gaussian_posterior_rejects_invalid_base_sample_values_and_dtype() -> None:
    posterior = GaussianPosterior(
        torch.zeros(1, 1, dtype=torch.double),
        torch.ones(1, 1, dtype=torch.double),
    )

    with pytest.raises(TensorValidationError, match="finite values"):
        posterior.rsample(
            torch.Size([1]),
            base_samples=torch.tensor([[[float("nan")]]], dtype=torch.double),
        )

    with pytest.raises(DTypeError, match="base_samples"):
        posterior.rsample(
            torch.Size([1]),
            base_samples=torch.zeros(1, 1, 1, dtype=torch.float32),
        )


def test_posterior_list_preserves_component_gradients() -> None:
    first_mean = torch.tensor([[0.2]], dtype=torch.double, requires_grad=True)
    second_mean = torch.tensor([[-0.4, 0.7]], dtype=torch.double, requires_grad=True)
    first = GaussianPosterior(
        first_mean,
        torch.ones(1, 1, dtype=torch.double),
    )
    second = GaussianPosterior(
        second_mean,
        torch.eye(2, dtype=torch.double),
    )
    posterior = PosteriorList(first, second)
    base_samples = torch.tensor(
        [[[0.1, -0.3, 0.8]], [[0.5, 0.2, -0.6]]],
        dtype=torch.double,
    )

    posterior.rsample(
        torch.Size([2]),
        base_samples=base_samples,
    ).square().sum().backward()

    assert first_mean.grad is not None
    assert second_mean.grad is not None
    assert torch.isfinite(first_mean.grad).all()
    assert torch.isfinite(second_mean.grad).all()
