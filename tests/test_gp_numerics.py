import torch

from motorch.models import SingleTaskGP


def test_single_task_gp_fitting_improves_training_posterior_mean() -> None:
    torch.manual_seed(0)
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    model = SingleTaskGP(train_X, train_Y)

    before = model.posterior(train_X).mean.detach()
    before_error = (before - train_Y).square().mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    for _ in range(80):
        optimizer.zero_grad()
        loss = model.training_loss()
        loss.backward()
        optimizer.step()

    after = model.posterior(train_X).mean.detach()
    after_error = (after - train_Y).square().mean()

    assert torch.isfinite(after).all()
    assert after_error < before_error
    assert after_error < 0.03


def test_single_task_gp_posterior_sample_gradients_are_finite() -> None:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cos(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[0.35], [0.65]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(X)
    base_samples = torch.tensor(
        [[[0.25], [-0.75]], [[1.0], [0.5]]],
        dtype=torch.double,
    )

    samples = posterior.rsample(torch.Size([2]), base_samples=base_samples)
    samples.square().sum().backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert X.grad.abs().sum() > 0
