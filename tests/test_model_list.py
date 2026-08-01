import pytest
import torch

from motorch.models import ModelList, SingleTaskGP
from motorch.posteriors import PosteriorList


def make_model(scale: float) -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * scale)
    return SingleTaskGP(train_X, train_Y)


def test_model_list_combines_component_posteriors() -> None:
    first = make_model(2.0)
    second = make_model(4.0)
    model = ModelList(first, second)
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)

    posterior = model.posterior(X)

    assert isinstance(posterior, PosteriorList)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert model.num_outputs == 2
    assert torch.allclose(posterior.mean[..., :1], first.posterior(X).mean)
    assert torch.allclose(posterior.mean[..., 1:], second.posterior(X).mean)


def test_model_list_registers_component_parameters_and_moves_dtype() -> None:
    model = ModelList(make_model(2.0), make_model(3.0))

    assert list(model.parameters())
    model = model.to(dtype=torch.float32)
    X = torch.tensor([[0.4]], dtype=torch.float32)

    assert model.posterior(X).dtype == torch.float32


def test_model_list_requires_a_component() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelList()
