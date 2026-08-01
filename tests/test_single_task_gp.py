import pytest
import torch

from motorch.exceptions import DTypeError, ShapeError, TensorValidationError
from motorch.models import SingleTaskGP
from motorch.posteriors import GaussianPosterior


def make_training_data(
    *,
    dtype: torch.dtype = torch.double,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=dtype).unsqueeze(-1)
    train_Y = (2.0 * train_X - 1.0).sin()
    return train_X, train_Y


def test_single_task_gp_constructs_posterior_with_expected_shapes() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    X = torch.linspace(0.1, 0.9, 5, dtype=torch.double).unsqueeze(-1)

    posterior = model.posterior(X)

    assert isinstance(posterior, GaussianPosterior)
    assert posterior.mean.shape == torch.Size([5, 1])
    assert posterior.variance.shape == torch.Size([5, 1])
    assert posterior.covariance_matrix.shape == torch.Size([5, 5])
    assert model.input_dim == 1
    assert model.num_outputs == 1
    assert model.batch_shape == torch.Size()


def test_single_task_gp_supports_batches_and_multiple_outputs() -> None:
    base_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_X = base_X.expand(2, 6, 1)
    first = torch.sin(train_X)
    second = torch.cos(train_X)
    train_Y = torch.cat([first, second], dim=-1)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.rand(2, 4, 1, dtype=torch.double)

    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([2, 4, 2])
    assert posterior.covariance_matrix.shape == torch.Size([2, 8, 8])
    cross_output = posterior.covariance_matrix[..., 0::2, 1::2]
    assert torch.count_nonzero(cross_output) == 0


def test_single_task_gp_observation_noise_increases_variance() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)

    latent = model.posterior(train_X)
    observed = model.posterior(train_X, observation_noise=True)

    assert torch.all(observed.variance > latent.variance)


def test_single_task_gp_training_loss_is_differentiable() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)

    loss = model.training_loss()
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_single_task_gp_posterior_gradients_reach_candidates() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double, requires_grad=True)

    posterior = model.posterior(X)
    posterior.mean.sum().backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert X.grad.abs().sum() > 0


def test_single_task_gp_rejects_invalid_training_data() -> None:
    train_X, train_Y = make_training_data()
    with pytest.raises(DTypeError, match="train_Y"):
        SingleTaskGP(train_X.float(), train_Y)
    with pytest.raises(TensorValidationError, match="finite"):
        invalid_Y = train_Y.clone()
        invalid_Y[0] = torch.nan
        SingleTaskGP(train_X, invalid_Y)
    with pytest.raises(ShapeError, match="same number"):
        SingleTaskGP(train_X, train_Y[:-1])


def test_single_task_gp_rejects_invalid_candidate_shape() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)

    with pytest.raises(ShapeError, match="input dimension"):
        model.posterior(torch.zeros(2, 2, dtype=torch.double))
    with pytest.raises(DTypeError, match="X"):
        model.posterior(torch.zeros(2, 1, dtype=torch.float32))
