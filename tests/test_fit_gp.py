import copy

import pytest
import torch

from motorch.fit import FitOptions, FitTermination, fit_gp
from motorch.models import FixedNoiseGP, SingleTaskGP


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    return train_X, train_Y


def test_fit_gp_reduces_marginal_likelihood_loss() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    initial_loss = float(model.training_loss().detach())

    result = fit_gp(
        model,
        options=FitOptions(
            max_steps=80,
            patience=8,
            tolerance_loss=1e-5,
            max_retries=0,
            warn_on_failure=False,
        ),
    )

    final_loss = float(model.training_loss().detach())
    assert result.best_loss < initial_loss
    assert final_loss <= initial_loss
    assert result.total_steps > 0
    assert len(result.attempts) == 1


def test_fit_gp_improves_reference_posterior_mean() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    test_X = torch.linspace(0.05, 0.95, 13, dtype=torch.double).unsqueeze(-1)
    truth = torch.sin(test_X * 6.0)
    initial_error = torch.mean((model.posterior(test_X).mean - truth).square())

    fit_gp(
        model,
        options=FitOptions(
            max_steps=100,
            patience=10,
            tolerance_loss=1e-5,
            max_retries=0,
            warn_on_failure=False,
        ),
    )

    final_error = torch.mean((model.posterior(test_X).mean - truth).square())
    assert final_error < initial_error
    assert final_error < 0.08


def test_fit_gp_supports_fixed_noise_model() -> None:
    train_X, train_Y = make_training_data()
    model = FixedNoiseGP(train_X, train_Y, torch.full_like(train_Y, 0.01))

    result = fit_gp(
        model,
        options=FitOptions(
            max_steps=50,
            max_retries=0,
            warn_on_failure=False,
        ),
    )

    assert torch.isfinite(torch.tensor(result.best_loss))
    assert model.posterior(train_X[:2]).mean.shape == torch.Size([2, 1])


def test_fit_gp_supports_lbfgs() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    initial_loss = float(model.training_loss().detach())

    result = fit_gp(
        model,
        options=FitOptions(
            optimizer="lbfgs",
            learning_rate=0.2,
            max_steps=20,
            patience=5,
            max_retries=0,
            warn_on_failure=False,
        ),
    )

    assert result.best_loss < initial_loss
    assert result.attempts[0].optimizer == "lbfgs"


def test_fit_gp_deterministic_mode_reproduces_state_and_restores_rng() -> None:
    train_X, train_Y = make_training_data()
    first = SingleTaskGP(train_X, train_Y)
    second = SingleTaskGP(train_X, train_Y)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    options = FitOptions(
        max_steps=25,
        max_retries=0,
        deterministic=True,
        seed=17,
        warn_on_failure=False,
    )
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()

    first_result = fit_gp(first, options=options)
    rng_after = torch.random.get_rng_state()
    second_result = fit_gp(second, options=options)

    assert torch.equal(rng_before, rng_after)
    assert first_result.best_loss == pytest.approx(second_result.best_loss)
    for first_value, second_value in zip(
        first.state_dict().values(),
        second.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(first_value, second_value)


def test_fit_result_exposes_actionable_diagnostics() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)

    result = fit_gp(
        model,
        options=FitOptions(
            max_steps=1,
            max_retries=0,
            warn_on_failure=False,
        ),
    )

    assert not result.converged
    assert result.termination is FitTermination.MAX_STEPS
    assert "max_steps" in result.message
    assert result.retries == 0
