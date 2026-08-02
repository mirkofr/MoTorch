"""Independent acceptance checks for Phase 7 acquisition optimization."""

import pytest
import torch

from motorch.acquisition import AcquisitionFunction, PosteriorMean
from motorch.models import Model, SingleTaskGP
from motorch.optim import OptimizeAcqfOptions, generate_raw_candidates, optimize_acqf
from motorch.posteriors import Posterior
from motorch.warnings import OptimizationWarning


class _UnusedModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        del X, observation_noise
        raise NotImplementedError


class LinearAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(_UnusedModel())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=(-1, -2))


class OneRestartFailureAcquisition(AcquisitionFunction):
    def __init__(self, target: float) -> None:
        super().__init__(_UnusedModel())
        self.target = target
        self.fail_next_local_call = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = -(X - self.target).square().sum(dim=(-1, -2))
        if X.shape[0] == 1 and self.fail_next_local_call:
            self.fail_next_local_call = False
            return torch.full_like(values, torch.nan)
        return values


class PendingAwareAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(_UnusedModel())
        self.pending_points: torch.Tensor | None = None

    def set_pending_points(self, X: torch.Tensor | None) -> None:
        self.pending_points = X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        target = 0.25 if self.pending_points is None else 0.75
        return -(X - target).square().sum(dim=(-1, -2))


def test_phase_7_optimizer_approaches_boundary_optimum_and_respects_bounds() -> None:
    bounds = torch.tensor([[-2.0, 1.0], [3.0, 4.0]], dtype=torch.double)

    candidate, value = optimize_acqf(
        LinearAcquisition(),
        bounds,
        q=1,
        num_restarts=8,
        raw_samples=128,
        options=OptimizeAcqfOptions(max_steps=160, learning_rate=0.1),
        seed=5,
    )

    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.all(candidate > bounds[1] - 0.03)
    assert torch.isfinite(value)


def test_phase_7_failed_restart_does_not_poison_successful_result() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with pytest.warns(OptimizationWarning, match="restart 0 failed"):
        candidate, value = optimize_acqf(
            OneRestartFailureAcquisition(0.63),
            bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
            options=OptimizeAcqfOptions(max_steps=100),
            seed=9,
        )

    assert candidate.item() == pytest.approx(0.63, abs=4e-3)
    assert value.item() > -2e-5


def test_phase_7_seeded_sobol_initialization_preserves_global_rng_state() -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double)
    torch.manual_seed(431)
    state_before = torch.random.get_rng_state().clone()

    generate_raw_candidates(bounds, q=3, raw_samples=16, seed=17)

    torch.testing.assert_close(torch.random.get_rng_state(), state_before)


def test_phase_7_sequential_generation_updates_and_clears_pending_points() -> None:
    acquisition = PendingAwareAcquisition()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, values = optimize_acqf(
        acquisition,
        bounds,
        q=3,
        num_restarts=5,
        raw_samples=64,
        options=OptimizeAcqfOptions(max_steps=100),
        seed=21,
        sequential=True,
    )

    assert candidates.shape == torch.Size([3, 1])
    assert values.shape == torch.Size([3])
    assert candidates[0, 0].item() == pytest.approx(0.25, abs=4e-3)
    assert torch.allclose(
        candidates[1:, 0],
        torch.full((2,), 0.75, dtype=torch.double),
        atol=4e-3,
        rtol=0,
    )
    assert acquisition.pending_points is None


def test_phase_7_integrates_with_actual_gp_posterior_mean() -> None:
    train_X = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=6,
        raw_samples=64,
        options=OptimizeAcqfOptions(max_steps=80),
        seed=4,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert candidate.dtype is torch.double
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.isfinite(value)
