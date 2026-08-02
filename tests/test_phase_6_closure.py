"""Independent Phase 6 acceptance and integration tests."""

import pytest
import torch

from motorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from motorch.models import Model, SingleTaskGP
from motorch.posteriors import GaussianPosterior, Posterior


class ConstantGaussianModel(Model):
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        super().__init__()
        self._mean = mean
        self._variance = variance

    @property
    def num_outputs(self) -> int:
        return self._mean.shape[-1]

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        del observation_noise
        batch_shape = X.shape[:-2]
        mean = self._mean.expand(*batch_shape, *self._mean.shape)
        flat_variance = self._variance.expand(*batch_shape, *self._variance.shape)
        covariance = torch.diag_embed(flat_variance.reshape(*batch_shape, -1))
        return GaussianPosterior(mean, covariance)


def test_phase_6_expected_improvement_gradient_matches_finite_difference() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)
    candidate = torch.tensor([[[0.37]]], dtype=torch.double, requires_grad=True)
    acquisition = ExpectedImprovement(model, best_f=train_Y.max())

    acquisition(candidate).sum().backward()

    assert candidate.grad is not None
    autograd_value = candidate.grad.item()
    step = 1e-6

    def evaluate(value: float) -> float:
        X = torch.tensor([[[value]]], dtype=torch.double)
        return acquisition(X).item()

    finite_difference = (evaluate(0.37 + step) - evaluate(0.37 - step)) / (2 * step)
    assert autograd_value == pytest.approx(finite_difference, rel=2e-4, abs=2e-5)


def test_phase_6_zero_variance_limits_are_explicit() -> None:
    model = ConstantGaussianModel(
        torch.tensor([[0.5]], dtype=torch.double),
        torch.tensor([[0.0]], dtype=torch.double),
    )
    X = torch.zeros(1, 1, dtype=torch.double)

    assert ProbabilityOfImprovement(model, best_f=0.4)(X).item() == 1.0
    assert ProbabilityOfImprovement(model, best_f=0.5)(X).item() == 0.0
    assert ExpectedImprovement(model, best_f=0.4)(X).item() == pytest.approx(0.1)
    assert ExpectedImprovement(model, best_f=0.6)(X).item() == 0.0
    assert UpperConfidenceBound(model, beta=9.0)(X).item() == pytest.approx(0.5)


def test_phase_6_multioutput_posterior_is_rejected() -> None:
    model = ConstantGaussianModel(
        torch.tensor([[0.2, 0.4]], dtype=torch.double),
        torch.tensor([[0.1, 0.2]], dtype=torch.double),
    )

    with pytest.raises(ValueError, match="single-output"):
        PosteriorMean(model)(torch.zeros(1, 1, dtype=torch.double))


def test_phase_6_scalar_buffers_follow_module_dtype() -> None:
    model = ConstantGaussianModel(
        torch.tensor([[0.2]], dtype=torch.double),
        torch.tensor([[0.1]], dtype=torch.double),
    )
    acquisition = ExpectedImprovement(model, best_f=0.0).to(dtype=torch.double)

    assert acquisition.best_f.dtype is torch.double
    assert acquisition(torch.zeros(1, 1, dtype=torch.double)).dtype is torch.double


def test_phase_6_actual_gp_all_analytic_functions_return_finite_batch_values() -> None:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cos(train_X * 3.0)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.double)
    acquisitions = [
        PosteriorMean(model),
        ProbabilityOfImprovement(model, best_f=train_Y.max()),
        ExpectedImprovement(model, best_f=train_Y.max()),
        UpperConfidenceBound(model, beta=0.2),
    ]

    for acquisition in acquisitions:
        values = acquisition(X)
        assert values.shape == torch.Size([3])
        assert torch.isfinite(values).all()
