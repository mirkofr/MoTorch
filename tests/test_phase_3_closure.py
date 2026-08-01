import copy

import pytest
import torch

from motorch.exceptions import ShapeError, TensorValidationError
from motorch.models import FixedNoiseGP, ModelList, SingleTaskGP


def make_data(*, outputs: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)
    columns = [torch.sin((index + 1) * train_X) for index in range(outputs)]
    return train_X, torch.cat(columns, dim=-1)


def test_phase_3_state_dict_round_trip_preserves_posterior() -> None:
    train_X, train_Y = make_data(outputs=2)
    source = SingleTaskGP(train_X, train_Y)
    candidate_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    expected = source.posterior(candidate_X)

    restored = SingleTaskGP(train_X, train_Y)
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    actual = restored.posterior(candidate_X)

    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)


def test_phase_3_candidate_and_model_batches_broadcast() -> None:
    base_X, base_Y = make_data()
    train_X = base_X.expand(2, *base_X.shape)
    train_Y = base_Y.expand(2, *base_Y.shape)
    model = SingleTaskGP(train_X, train_Y)
    candidate_X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)

    posterior = model.posterior(candidate_X)

    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.covariance_matrix.shape == torch.Size([2, 2, 2])


def test_phase_3_multi_output_covariance_uses_q_major_ordering() -> None:
    train_X, train_Y = make_data(outputs=2)
    model = SingleTaskGP(train_X, train_Y)
    candidate_X = torch.tensor([[0.2], [0.7]], dtype=torch.double)

    posterior = model.posterior(candidate_X)
    covariance = posterior.covariance_matrix

    # Flattened event order is (q0, m0), (q0, m1), (q1, m0), (q1, m1).
    assert covariance.shape == torch.Size([4, 4])
    torch.testing.assert_close(covariance[0, 2], covariance[2, 0])
    torch.testing.assert_close(covariance[1, 3], covariance[3, 1])
    assert covariance[0, 1] == 0
    assert covariance[0, 3] == 0
    assert covariance[1, 2] == 0
    assert covariance[2, 3] == 0


def test_phase_3_candidate_gradient_matches_finite_difference() -> None:
    train_X, train_Y = make_data()
    model = SingleTaskGP(train_X, train_Y)
    candidate_X = torch.tensor([[0.43]], dtype=torch.double, requires_grad=True)

    value = model.posterior(candidate_X).mean.squeeze()
    value.backward()
    assert candidate_X.grad is not None
    analytic = candidate_X.grad.item()

    step = 1e-5
    upper = torch.tensor([[0.43 + step]], dtype=torch.double)
    lower = torch.tensor([[0.43 - step]], dtype=torch.double)
    finite_difference = (
        model.posterior(upper).mean.item() - model.posterior(lower).mean.item()
    ) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, rel=2e-3, abs=2e-4)


def test_phase_3_model_list_preserves_candidate_gradients() -> None:
    train_X, train_Y = make_data()
    models = ModelList(
        SingleTaskGP(train_X, train_Y),
        FixedNoiseGP(train_X, train_Y, torch.full_like(train_Y, 0.01)),
    )
    candidate_X = torch.tensor([[0.35], [0.65]], dtype=torch.double, requires_grad=True)

    models.posterior(candidate_X).mean.sum().backward()

    assert candidate_X.grad is not None
    assert torch.isfinite(candidate_X.grad).all()
    assert candidate_X.grad.abs().sum() > 0


def test_phase_3_rejects_nonfinite_and_empty_candidates() -> None:
    train_X, train_Y = make_data()
    model = SingleTaskGP(train_X, train_Y)

    with pytest.raises(TensorValidationError, match="finite"):
        model.posterior(torch.tensor([[torch.nan]], dtype=torch.double))
    with pytest.raises(ShapeError, match="q to be positive"):
        model.posterior(torch.empty(0, 1, dtype=torch.double))


def test_phase_3_cuda_smoke_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment.")

    train_X, train_Y = make_data()
    model = SingleTaskGP(train_X.cuda(), train_Y.cuda())
    candidate_X = torch.tensor([[0.4]], dtype=torch.double, device="cuda", requires_grad=True)

    posterior = model.posterior(candidate_X)
    posterior.mean.sum().backward()

    assert posterior.mean.device.type == "cuda"
    assert posterior.covariance_matrix.device.type == "cuda"
    assert candidate_X.grad is not None
    assert torch.isfinite(candidate_X.grad).all()
