import copy

import pytest
import torch
from torch import nn

from motorch.fit import FitOptions, FitTermination, fit_gp
from motorch.models import SingleTaskGP
from motorch.warnings import FittingWarning, NumericalWarning


class AlwaysFailingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0, dtype=torch.double))

    def training_loss(self) -> torch.Tensor:
        raise RuntimeError("synthetic numerical failure")


class ImprovingThenFailingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(2.0, dtype=torch.double))
        self.calls = 0

    def training_loss(self) -> torch.Tensor:
        self.calls += 1
        if self.calls > 4:
            raise RuntimeError("failure after finite progress")
        return (self.value - 0.25).square()


class ParameterlessModel(nn.Module):
    def training_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, dtype=torch.double, requires_grad=True)


def test_phase_4_retry_jitter_is_capped_and_attempts_are_indexed() -> None:
    with pytest.warns((FittingWarning, NumericalWarning)):
        result = fit_gp(
            AlwaysFailingModel(),
            options=FitOptions(
                max_steps=2,
                max_retries=3,
                initial_jitter=1e-4,
                jitter_multiplier=10.0,
                max_jitter=1e-3,
            ),
        )

    assert [attempt.attempt for attempt in result.attempts] == [0, 1, 2, 3]
    assert [attempt.jitter for attempt in result.attempts] == pytest.approx(
        [1e-4, 1e-3, 1e-3, 1e-3]
    )
    assert result.retries == 3
    assert result.total_steps == 0
    assert result.termination is FitTermination.NUMERICAL_ERROR


def test_phase_4_failed_retry_restores_best_finite_state() -> None:
    model = ImprovingThenFailingModel()
    initial_distance = abs(model.value.item() - 0.25)

    with pytest.warns((FittingWarning, NumericalWarning)):
        result = fit_gp(
            model,
            options=FitOptions(
                learning_rate=0.2,
                max_steps=3,
                max_retries=1,
            ),
        )

    assert result.best_loss < float("inf")
    assert abs(model.value.item() - 0.25) < initial_distance
    assert torch.isfinite(model.value)


def test_phase_4_deterministic_mode_restores_algorithm_setting() -> None:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()

    fit_gp(
        model,
        options=FitOptions(
            max_steps=5,
            max_retries=0,
            deterministic=True,
            seed=19,
            warn_on_failure=False,
        ),
    )

    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before


def test_phase_4_float32_fit_preserves_dtype_and_cpu_device() -> None:
    train_X = torch.linspace(0.0, 1.0, 7, dtype=torch.float32).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)

    result = fit_gp(
        model,
        options=FitOptions(
            max_steps=10,
            max_retries=0,
            warn_on_failure=False,
        ),
    )
    posterior = model.posterior(train_X[:2])

    assert torch.isfinite(torch.tensor(result.best_loss))
    assert posterior.mean.dtype is torch.float32
    assert posterior.mean.device.type == "cpu"
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())


def test_phase_4_nearly_duplicated_points_fit_with_explicit_jitter() -> None:
    train_X = torch.tensor(
        [[0.0], [0.2], [0.2 + 1e-10], [0.7], [1.0]], dtype=torch.double
    )
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)

    result = fit_gp(
        model,
        options=FitOptions(
            max_steps=20,
            max_retries=1,
            initial_jitter=1e-6,
            warn_on_retry=False,
            warn_on_failure=False,
        ),
    )

    assert torch.isfinite(torch.tensor(result.best_loss))
    assert torch.isfinite(model.posterior(train_X).mean).all()


def test_phase_4_parameterless_model_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="no trainable parameters"):
        fit_gp(
            ParameterlessModel(),
            options=FitOptions(max_retries=0, warn_on_failure=False),
        )


def test_phase_4_seeded_runs_reproduce_full_diagnostics() -> None:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cos(train_X * 3.0)
    first = SingleTaskGP(train_X, train_Y)
    second = SingleTaskGP(train_X, train_Y)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    options = FitOptions(
        max_steps=12,
        max_retries=0,
        deterministic=True,
        seed=7,
        warn_on_failure=False,
    )

    first_result = fit_gp(first, options=options)
    second_result = fit_gp(second, options=options)

    assert first_result == second_result
