import pytest
import torch

from motorch.exceptions import ShapeError, TensorValidationError
from motorch.models import FixedNoiseGP


def make_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 3.0)
    train_Yvar = torch.full_like(train_Y, 0.02)
    return train_X, train_Y, train_Yvar


def test_fixed_noise_gp_preserves_supplied_noise() -> None:
    train_X, train_Y, train_Yvar = make_data()
    model = FixedNoiseGP(train_X, train_Y, train_Yvar)

    assert torch.equal(model.train_X, train_X)
    assert torch.equal(model.train_Y, train_Y)
    assert torch.equal(model.train_Yvar, train_Yvar)
    assert model.posterior(train_X).mean.shape == train_Y.shape


def test_fixed_noise_gp_training_loss_is_finite_and_differentiable() -> None:
    train_X, train_Y, train_Yvar = make_data()
    model = FixedNoiseGP(train_X, train_Y, train_Yvar)

    loss = model.training_loss()
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_fixed_noise_gp_rejects_invalid_noise() -> None:
    train_X, train_Y, train_Yvar = make_data()

    with pytest.raises(ShapeError, match="match train_Y shape"):
        FixedNoiseGP(train_X, train_Y, train_Yvar[:-1])
    with pytest.raises(TensorValidationError, match="strictly positive"):
        invalid_noise = train_Yvar.clone()
        invalid_noise[0] = 0.0
        FixedNoiseGP(train_X, train_Y, invalid_noise)


def test_fixed_noise_gp_rejects_unspecified_test_observation_noise() -> None:
    train_X, train_Y, train_Yvar = make_data()
    model = FixedNoiseGP(train_X, train_Y, train_Yvar)

    with pytest.raises(ValueError, match="candidate-specific noise"):
        model.posterior(train_X, observation_noise=True)
