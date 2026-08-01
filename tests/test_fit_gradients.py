import torch

from motorch.fit import FitOptions, fit_gp
from motorch.models import SingleTaskGP


def test_candidate_gradients_remain_finite_after_fitting() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)
    fit_gp(
        model,
        options=FitOptions(
            max_steps=30,
            max_retries=0,
            warn_on_failure=False,
        ),
    )
    candidate_X = torch.tensor([[0.4]], dtype=torch.double, requires_grad=True)

    model.posterior(candidate_X).mean.sum().backward()

    assert candidate_X.grad is not None
    assert torch.isfinite(candidate_X.grad).all()
