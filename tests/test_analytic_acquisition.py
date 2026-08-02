import math

import pytest
import torch

from motorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from motorch.models import Model
from motorch.posteriors import GaussianPosterior, Posterior


class AffineGaussianModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        del observation_noise
        mean = X.sum(dim=-1, keepdim=True)
        variance = 0.25 + 0.1 * X.square().sum(dim=-1, keepdim=True)
        covariance = torch.diag_embed(variance.squeeze(-1))
        return GaussianPosterior(mean, covariance)


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-value / math.sqrt(2.0))


def _normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def test_analytic_values_match_independent_formulas() -> None:
    model = AffineGaussianModel()
    X = torch.tensor([[[0.2, 0.3]], [[-0.4, 0.1]]], dtype=torch.double)
    mean = X.sum(dim=-1).squeeze(-1)
    variance = 0.25 + 0.1 * X.square().sum(dim=-1).squeeze(-1)
    sigma = variance.sqrt()
    best_f = torch.tensor(0.1, dtype=torch.double)
    z = (mean - best_f) / sigma

    torch.testing.assert_close(PosteriorMean(model)(X), mean)
    torch.testing.assert_close(
        ProbabilityOfImprovement(model, best_f)(X), _normal_cdf(z)
    )
    torch.testing.assert_close(
        ExpectedImprovement(model, best_f)(X),
        (mean - best_f) * _normal_cdf(z) + sigma * _normal_pdf(z),
    )
    torch.testing.assert_close(
        UpperConfidenceBound(model, beta=2.0)(X),
        mean + math.sqrt(2.0) * sigma,
    )


def test_analytic_acquisitions_preserve_batch_shape_dtype_and_gradients() -> None:
    model = AffineGaussianModel()
    X = torch.tensor(
        [[[[0.1, 0.2]]], [[[0.3, -0.4]]]],
        dtype=torch.double,
        requires_grad=True,
    )
    acquisition = ExpectedImprovement(model, best_f=0.0)

    values = acquisition(X)
    values.sum().backward()

    assert values.shape == torch.Size([2, 1])
    assert values.dtype is torch.double
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_minimization_convention_negates_mean_and_reverses_improvement() -> None:
    model = AffineGaussianModel()
    X = torch.tensor([[[0.2, 0.3]]], dtype=torch.double)

    torch.testing.assert_close(
        PosteriorMean(model, maximize=False)(X),
        torch.tensor([-0.5], dtype=torch.double),
    )
    assert (
        ExpectedImprovement(model, best_f=0.0, maximize=False)(X).item()
        < ExpectedImprovement(model, best_f=0.0)(X).item()
    )
    assert UpperConfidenceBound(model, beta=0.0, maximize=False)(
        X
    ).item() == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda model: ProbabilityOfImprovement(model, best_f=0.0),
        lambda model: ExpectedImprovement(model, best_f=0.0),
        lambda model: UpperConfidenceBound(model, beta=1.0),
        lambda model: PosteriorMean(model),
    ],
)
def test_analytic_acquisitions_reject_q_greater_than_one(factory: object) -> None:
    model = AffineGaussianModel()
    acquisition = factory(model)  # type: ignore[operator]
    X = torch.zeros(2, 3, dtype=torch.double)

    with pytest.raises(ValueError, match="require q=1"):
        acquisition(X)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), torch.tensor([1.0, 2.0])]
)
def test_improvement_acquisitions_reject_invalid_best_f(value: object) -> None:
    model = AffineGaussianModel()

    with pytest.raises(ValueError, match="best_f"):
        ExpectedImprovement(model, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="best_f"):
        ProbabilityOfImprovement(model, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("beta", [-1.0, float("nan"), torch.tensor([1.0, 2.0])])
def test_ucb_rejects_invalid_beta(beta: object) -> None:
    with pytest.raises(ValueError, match="beta"):
        UpperConfidenceBound(AffineGaussianModel(), beta)  # type: ignore[arg-type]
