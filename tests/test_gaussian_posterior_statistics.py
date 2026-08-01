import torch

from motorch.posteriors import GaussianPosterior


def test_gaussian_posterior_sample_moments_match_analytic_moments() -> None:
    mean = torch.tensor([[0.7], [-1.2]], dtype=torch.double)
    covariance = torch.tensor([[1.4, 0.35], [0.35, 0.8]], dtype=torch.double)
    posterior = GaussianPosterior(mean, covariance)
    generator = torch.Generator().manual_seed(2026)
    base_samples = torch.randn(
        80_000,
        2,
        1,
        generator=generator,
        dtype=torch.double,
    )

    samples = posterior.rsample(
        torch.Size([80_000]),
        base_samples=base_samples,
    ).reshape(80_000, 2)
    empirical_mean = samples.mean(dim=0)
    centered = samples - empirical_mean
    empirical_covariance = centered.transpose(-1, -2) @ centered / (
        samples.shape[0] - 1
    )

    assert torch.allclose(empirical_mean, mean.reshape(-1), atol=0.015, rtol=0.0)
    assert torch.allclose(empirical_covariance, covariance, atol=0.025, rtol=0.03)


def test_gaussian_posterior_rsample_propagates_mean_and_covariance_gradients() -> None:
    mean = torch.tensor([[0.2], [-0.4]], dtype=torch.double, requires_grad=True)
    raw_root = torch.tensor(
        [[1.1, 0.0], [0.25, 0.8]],
        dtype=torch.double,
        requires_grad=True,
    )
    covariance = raw_root @ raw_root.transpose(-1, -2)
    posterior = GaussianPosterior(mean, covariance)
    base_samples = torch.tensor(
        [[[0.3], [-1.2]], [[1.1], [0.5]]],
        dtype=torch.double,
    )

    loss = posterior.rsample(
        torch.Size([2]),
        base_samples=base_samples,
    ).square().sum()
    loss.backward()

    assert mean.grad is not None
    assert raw_root.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(raw_root.grad).all()
    assert raw_root.grad.abs().sum() > 0
