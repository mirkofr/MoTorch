import torch

from motorch.posteriors import GaussianPosterior
from motorch.sampling import IIDNormalSampler, SobolQMCNormalSampler


def _standard_normal_posterior(dimension: int) -> GaussianPosterior:
    mean = torch.zeros(dimension, 1, dtype=torch.double)
    covariance = torch.eye(dimension, dtype=torch.double)
    return GaussianPosterior(mean, covariance)


def test_iid_base_samples_have_expected_standard_normal_statistics() -> None:
    sampler = IIDNormalSampler(torch.Size([8192]), seed=41)
    sampler(_standard_normal_posterior(3))
    base_samples = sampler.base_samples

    assert base_samples is not None
    flattened = base_samples.reshape(-1, 3)
    torch.testing.assert_close(
        flattened.mean(dim=0),
        torch.zeros(3, dtype=torch.double),
        atol=0.04,
        rtol=0.0,
    )
    torch.testing.assert_close(
        flattened.var(dim=0, unbiased=True),
        torch.ones(3, dtype=torch.double),
        atol=0.06,
        rtol=0.0,
    )


def test_sobol_qmc_base_samples_have_expected_standard_normal_statistics() -> None:
    sampler = SobolQMCNormalSampler(torch.Size([4096]), seed=29)
    sampler(_standard_normal_posterior(4))
    base_samples = sampler.base_samples

    assert base_samples is not None
    flattened = base_samples.reshape(-1, 4)
    torch.testing.assert_close(
        flattened.mean(dim=0),
        torch.zeros(4, dtype=torch.double),
        atol=0.01,
        rtol=0.0,
    )
    torch.testing.assert_close(
        flattened.var(dim=0, unbiased=True),
        torch.ones(4, dtype=torch.double),
        atol=0.02,
        rtol=0.0,
    )


def test_sobol_qmc_posterior_samples_match_analytic_moments() -> None:
    mean = torch.tensor([[1.0], [-0.5]], dtype=torch.double)
    covariance = torch.tensor(
        [[1.5, 0.4], [0.4, 0.7]],
        dtype=torch.double,
    )
    posterior = GaussianPosterior(mean, covariance)
    sampler = SobolQMCNormalSampler(torch.Size([8192]), seed=7)

    samples = sampler(posterior).reshape(8192, 2)
    empirical_mean = samples.mean(dim=0)
    empirical_covariance = torch.cov(samples.transpose(0, 1))

    torch.testing.assert_close(empirical_mean, mean.squeeze(-1), atol=0.015, rtol=0.0)
    torch.testing.assert_close(empirical_covariance, covariance, atol=0.025, rtol=0.0)
