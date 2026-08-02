"""Tests for bounded acquisition-function optimization."""

import pytest
import torch

from motorch.acquisition import AcquisitionFunction
from motorch.models import Model
from motorch.optim import (
    OptimizationResult,
    OptimizeAcqfOptions,
    generate_raw_candidates,
    optimize_acqf,
    select_restart_candidates,
)
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


class QuadraticAcquisition(AcquisitionFunction):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__(_UnusedModel())
        self.register_buffer("target", target)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X - self.target).square().sum(dim=(-1, -2))


class PartlyFiniteAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(_UnusedModel())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = -X.square().sum(dim=(-1, -2))
        return torch.where(X[..., 0, 0] < 0.5, values, torch.nan)


def test_sobol_raw_candidates_are_seeded_bounded_and_dtype_preserving() -> None:
    bounds = torch.tensor([[-2.0, 1.0], [3.0, 4.0]], dtype=torch.double)

    first = generate_raw_candidates(bounds, q=2, raw_samples=16, seed=7)
    second = generate_raw_candidates(bounds, q=2, raw_samples=16, seed=7)

    torch.testing.assert_close(first, second)
    assert first.shape == torch.Size([16, 2, 2])
    assert first.dtype is torch.double
    assert torch.all(first >= bounds[0])
    assert torch.all(first <= bounds[1])


def test_restart_selection_excludes_nonfinite_values() -> None:
    raw = torch.tensor([[[0.1]], [[0.8]], [[0.2]]], dtype=torch.double)

    with pytest.warns(OptimizationWarning, match="Non-finite"):
        starts, values = select_restart_candidates(
            PartlyFiniteAcquisition(), raw, num_restarts=2
        )

    assert starts.shape == torch.Size([2, 1, 1])
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_optimizer_finds_known_quadratic_maximum(optimizer: str) -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double)
    target = torch.tensor([[[0.7, 0.4]]], dtype=torch.double)
    acquisition = QuadraticAcquisition(target)
    options = OptimizeAcqfOptions(
        optimizer=optimizer,  # type: ignore[arg-type]
        max_steps=120,
        learning_rate=0.08,
    )

    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=6,
        raw_samples=64,
        options=options,
        seed=11,
    )

    torch.testing.assert_close(candidates, target.squeeze(0), atol=2e-3, rtol=0)
    assert value.item() >= -1e-5
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])


def test_joint_q_optimization_and_fixed_features() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    target = torch.tensor([[[0.2, 0.3], [0.8, 0.3]]], dtype=torch.double)

    result = optimize_acqf(
        QuadraticAcquisition(target),
        bounds,
        q=2,
        num_restarts=8,
        raw_samples=128,
        fixed_features={1: 0.3},
        seed=3,
        return_diagnostics=True,
    )

    assert isinstance(result, OptimizationResult)
    torch.testing.assert_close(result.candidates, target.squeeze(0), atol=3e-3, rtol=0)
    assert torch.all(result.candidates[..., 1] == 0.3)
    assert any(restart.success for restart in result.restarts)


def test_optimizer_is_deterministic_for_seeded_initialization() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acquisition = QuadraticAcquisition(torch.tensor([[[0.61]]], dtype=torch.double))
    options = OptimizeAcqfOptions(max_steps=40)

    first = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=4,
        raw_samples=32,
        options=options,
        seed=19,
    )
    second = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=4,
        raw_samples=32,
        options=options,
        seed=19,
    )

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_optimizer_rejects_invalid_configuration() -> None:
    acquisition = QuadraticAcquisition(torch.zeros(1, 1, 1, dtype=torch.double))
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="raw_samples"):
        optimize_acqf(
            acquisition,
            bounds,
            q=1,
            num_restarts=4,
            raw_samples=3,
        )
    with pytest.raises(ValueError, match="outside its bounds"):
        optimize_acqf(
            acquisition,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=4,
            fixed_features={0: 2.0},
        )
