import math

import pytest
import torch

from motorch.acquisition import (
    ExpectedImprovement,
    qConstrainedExpectedImprovement,
    qExpectedImprovement,
)
from motorch.models import Model
from motorch.objectives import IdentityMCObjective, MCAcquisitionObjective
from motorch.optim import OptimizeAcqfOptions, optimize_acqf
from motorch.posteriors import GaussianPosterior, Posterior
from motorch.sampling import SobolQMCNormalSampler


class AffineGaussianModel(Model):
    def __init__(self, *, variance: float = 0.16) -> None:
        super().__init__()
        self.variance = variance
        self.last_X: torch.Tensor | None = None

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
        self.last_X = X
        mean = X.sum(dim=-1, keepdim=True)
        variance = torch.full_like(mean, self.variance)
        covariance = torch.diag_embed(variance.squeeze(-1))
        return GaussianPosterior(mean, covariance)


class QuadraticGaussianModel(Model):
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
        mean = -(X - 0.72).square().sum(dim=-1, keepdim=True)
        variance = torch.full_like(mean, 1e-6)
        covariance = torch.diag_embed(variance.squeeze(-1))
        return GaussianPosterior(mean, covariance)


class DoubledObjective(MCAcquisitionObjective):
    def forward(
        self,
        samples: torch.Tensor,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del X
        return 2.0 * samples.squeeze(-1)


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-value / math.sqrt(2.0))


def _normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def test_q_expected_improvement_q1_matches_independent_reference() -> None:
    model = AffineGaussianModel()
    X = torch.tensor([[[0.3]], [[-0.2]]], dtype=torch.double)
    best_f = torch.tensor(0.0, dtype=torch.double)
    sampler = SobolQMCNormalSampler(torch.Size([8192]), seed=7)

    actual = qExpectedImprovement(model, best_f, sampler)(X)
    mean = X.squeeze(-1).squeeze(-1)
    sigma = torch.full_like(mean, 0.4)
    z = (mean - best_f) / sigma
    expected = mean * _normal_cdf(z) + sigma * _normal_pdf(z)

    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(
        ExpectedImprovement(model, best_f)(X), expected, atol=1e-12, rtol=1e-12
    )


def test_q_expected_improvement_preserves_batch_shape_dtype_and_gradients() -> None:
    model = AffineGaussianModel()
    sampler = SobolQMCNormalSampler(torch.Size([256]), seed=11)
    acquisition = qExpectedImprovement(model, best_f=0.0, sampler=sampler)
    X = torch.tensor(
        [[[[0.1], [0.3]]], [[[0.2], [-0.4]]]],
        dtype=torch.double,
        requires_grad=True,
    )

    values = acquisition(X)
    values.sum().backward()

    assert values.shape == torch.Size([2, 1])
    assert values.dtype is torch.double
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_q_expected_improvement_uses_custom_sampled_objective() -> None:
    model = AffineGaussianModel(variance=1e-8)
    X = torch.tensor([[[0.4]]], dtype=torch.double)
    sampler = SobolQMCNormalSampler(torch.Size([512]), seed=2)

    ordinary = qExpectedImprovement(model, 0.0, sampler)(X)
    sampler.reset_base_samples()
    doubled = qExpectedImprovement(
        model,
        0.0,
        sampler,
        objective=DoubledObjective(),
    )(X)

    torch.testing.assert_close(doubled, 2.0 * ordinary, atol=2e-4, rtol=2e-4)


def test_pending_points_are_included_and_can_be_cleared() -> None:
    model = AffineGaussianModel()
    acquisition = qExpectedImprovement(
        model,
        best_f=0.0,
        sampler=SobolQMCNormalSampler(torch.Size([64]), seed=3),
    )
    X = torch.tensor([[0.2]], dtype=torch.double)
    pending = torch.tensor([[0.6], [0.8]], dtype=torch.double)

    acquisition.set_pending_points(pending)
    acquisition(X)

    assert model.last_X is not None
    torch.testing.assert_close(model.last_X, torch.cat((X, pending), dim=-2))
    acquisition.set_pending_points(None)
    assert acquisition.X_pending is None


def test_constrained_expected_improvement_penalizes_infeasible_samples() -> None:
    model = AffineGaussianModel(variance=1e-6)
    sampler = SobolQMCNormalSampler(torch.Size([2048]), seed=19)

    def upper_limit(samples: torch.Tensor) -> torch.Tensor:
        return samples.squeeze(-1) - 0.5

    acquisition = qConstrainedExpectedImprovement(
        model,
        best_f=0.0,
        sampler=sampler,
        constraints=[upper_limit],
        eta=0.01,
    )
    X = torch.tensor([[[0.4]], [[0.8]]], dtype=torch.double, requires_grad=True)

    values = acquisition(X)
    values.sum().backward()

    assert values[0] > values[1]
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_identity_objective_rejects_multiple_outputs() -> None:
    samples = torch.zeros(8, 2, 3, 2, dtype=torch.double)

    with pytest.raises(ValueError, match="one posterior output"):
        IdentityMCObjective()(samples)


def test_monte_carlo_acquisition_rejects_invalid_inputs_and_constraints() -> None:
    model = AffineGaussianModel()
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=0)
    acquisition = qExpectedImprovement(model, 0.0, sampler)

    with pytest.raises(ValueError, match="positive q and d"):
        acquisition(torch.empty(0, 1, dtype=torch.double))
    with pytest.raises(TypeError, match="floating-point"):
        acquisition(torch.ones(1, 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="best_f"):
        qExpectedImprovement(model, float("nan"), sampler)
    with pytest.raises(ValueError, match="non-empty"):
        qConstrainedExpectedImprovement(model, 0.0, sampler, constraints=[])
    with pytest.raises(ValueError, match="eta"):
        qConstrainedExpectedImprovement(
            model,
            0.0,
            sampler,
            constraints=[lambda samples: samples.squeeze(-1)],
            eta=0.0,
        )


def test_phase_8_optimized_qei_beats_equal_budget_random_search() -> None:
    model = QuadraticGaussianModel()
    acquisition = qExpectedImprovement(
        model,
        best_f=-1.0,
        sampler=SobolQMCNormalSampler(torch.Size([256]), seed=23),
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, _ = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=6,
        raw_samples=64,
        options=OptimizeAcqfOptions(max_steps=100, learning_rate=0.08),
        seed=31,
    )
    generator = torch.Generator().manual_seed(31)
    random_candidates = torch.rand(4, 1, 1, dtype=torch.double, generator=generator)
    optimized_value = model.posterior(candidate).mean.squeeze()
    random_value = model.posterior(random_candidates).mean.max()

    assert optimized_value > random_value
    assert candidate.item() == pytest.approx(0.72, abs=0.02)
