"""Independent Phase 8 Monte Carlo acquisition acceptance tests."""

import torch

from motorch.acquisition import (
    qConstrainedExpectedImprovement,
    qExpectedImprovement,
)
from motorch.models import Model, SingleTaskGP
from motorch.objectives import MCAcquisitionObjective
from motorch.posteriors import GaussianPosterior, Posterior
from motorch.sampling import IIDNormalSampler, SobolQMCNormalSampler


class TwoOutputAffineModel(Model):
    """Small differentiable model used only for acceptance testing."""

    @property
    def num_outputs(self) -> int:
        return 2

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        del observation_noise
        first = X.sum(dim=-1)
        second = X[..., 0] - X[..., -1]
        mean = torch.stack((first, second), dim=-1)
        variance = torch.full_like(mean, 0.04)
        covariance = torch.diag_embed(variance.flatten(start_dim=-2))
        return GaussianPosterior(mean, covariance)


class WeightedObjective(MCAcquisitionObjective):
    """Collapse two posterior outputs into one sampled utility."""

    def forward(
        self,
        samples: torch.Tensor,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del X
        return samples[..., 0] + 0.5 * samples[..., 1]


def test_qei_supports_multioutput_model_through_explicit_objective() -> None:
    X = torch.tensor(
        [[[0.2, 0.4], [0.7, 0.1]]],
        dtype=torch.double,
        requires_grad=True,
    )
    acquisition = qExpectedImprovement(
        TwoOutputAffineModel(),
        best_f=0.0,
        sampler=SobolQMCNormalSampler(torch.Size([256]), seed=17),
        objective=WeightedObjective(),
    )

    value = acquisition(X)
    value.sum().backward()

    assert value.shape == torch.Size([1])
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_qei_candidate_gradient_matches_central_finite_difference() -> None:
    model = TwoOutputAffineModel()
    sampler = SobolQMCNormalSampler(torch.Size([512]), seed=23)
    acquisition = qExpectedImprovement(
        model,
        best_f=0.1,
        sampler=sampler,
        objective=WeightedObjective(),
    )
    X = torch.tensor([[[0.35, 0.15]]], dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    value.backward()
    assert X.grad is not None
    analytic = X.grad[0, 0, 0].detach()

    step = 1e-5
    plus = X.detach().clone()
    minus = X.detach().clone()
    plus[0, 0, 0] += step
    minus[0, 0, 0] -= step
    finite_difference = (acquisition(plus) - acquisition(minus)) / (2.0 * step)

    torch.testing.assert_close(
        analytic, finite_difference.squeeze(), rtol=2e-3, atol=2e-3
    )


def test_pending_points_resize_sampler_cache_and_clear_cleanly() -> None:
    sampler = IIDNormalSampler(torch.Size([32]), seed=5)
    acquisition = qExpectedImprovement(
        TwoOutputAffineModel(),
        best_f=0.0,
        sampler=sampler,
        objective=WeightedObjective(),
    )
    X = torch.tensor([[[0.2, 0.1]]], dtype=torch.double)
    pending = torch.tensor([[[0.8, 0.3], [0.5, 0.4]]], dtype=torch.double)

    acquisition.set_pending_points(pending)
    pending_value = acquisition(X)
    pending_base_shape = (
        sampler.base_samples.shape if sampler.base_samples is not None else None
    )

    acquisition.set_pending_points(None)
    clear_value = acquisition(X)
    clear_base_shape = (
        sampler.base_samples.shape if sampler.base_samples is not None else None
    )

    assert pending_base_shape == torch.Size([32, 1, 3, 2])
    assert clear_base_shape == torch.Size([32, 1, 1, 2])
    assert acquisition.X_pending is None
    assert torch.isfinite(pending_value).all()
    assert torch.isfinite(clear_value).all()


def test_constrained_qei_combines_multiple_feasibility_terms() -> None:
    sampler = SobolQMCNormalSampler(torch.Size([256]), seed=11)
    X = torch.tensor([[[0.6, 0.2]]], dtype=torch.double)
    feasible = qConstrainedExpectedImprovement(
        TwoOutputAffineModel(),
        best_f=0.0,
        sampler=sampler,
        objective=WeightedObjective(),
        constraints=(
            lambda samples: samples[..., 0] - 10.0,
            lambda samples: samples[..., 1] - 10.0,
        ),
        eta=0.1,
    )
    infeasible = qConstrainedExpectedImprovement(
        TwoOutputAffineModel(),
        best_f=0.0,
        sampler=SobolQMCNormalSampler(torch.Size([256]), seed=11),
        objective=WeightedObjective(),
        constraints=(
            lambda samples: samples[..., 0] + 10.0,
            lambda samples: samples[..., 1] + 10.0,
        ),
        eta=0.1,
    )

    assert feasible(X).item() > infeasible(X).item()
    assert infeasible(X).item() >= 0.0


def test_mc_runtime_caches_are_not_serialized() -> None:
    acquisition = qExpectedImprovement(
        TwoOutputAffineModel(),
        best_f=0.0,
        sampler=IIDNormalSampler(torch.Size([16]), seed=3),
        objective=WeightedObjective(),
    )
    acquisition.set_pending_points(torch.tensor([[[0.1, 0.2]]], dtype=torch.double))
    acquisition(torch.tensor([[[0.3, 0.4]]], dtype=torch.double))

    keys = set(acquisition.state_dict())

    assert "best_f" in keys
    assert all("_X_pending" not in key for key in keys)
    assert all("_base_samples" not in key for key in keys)


def test_qei_integrates_with_single_task_gp_and_candidate_gradients() -> None:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = -((train_X - 0.65) ** 2)
    model = SingleTaskGP(train_X, train_Y)
    acquisition = qExpectedImprovement(
        model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(torch.Size([128]), seed=29),
    )
    X = torch.tensor([[[0.45], [0.75]]], dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert value.shape == torch.Size([1])
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
