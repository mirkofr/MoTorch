import torch

from motorch.models import SingleTaskGP
from motorch.posteriors import GaussianPosterior, PosteriorList
from motorch.sampling import IIDNormalSampler, SobolQMCNormalSampler


def _unit_posterior(*, dtype: torch.dtype = torch.double) -> GaussianPosterior:
    return GaussianPosterior(
        torch.zeros(2, 1, dtype=dtype),
        torch.eye(2, dtype=dtype),
    )


def test_phase_5_empty_sample_shape_returns_one_unprefixed_sample() -> None:
    posterior = _unit_posterior()
    sampler = IIDNormalSampler(torch.Size(), seed=3)

    samples = sampler(posterior)

    assert samples.shape == posterior.mean.shape
    assert sampler.base_samples is not None
    assert sampler.base_samples.shape == posterior.base_sample_shape


def test_phase_5_iid_sampler_does_not_consume_global_rng_state() -> None:
    posterior = _unit_posterior()
    torch.manual_seed(41)
    state_before = torch.random.get_rng_state().clone()

    IIDNormalSampler(torch.Size([16]), seed=9)(posterior)

    torch.testing.assert_close(torch.random.get_rng_state(), state_before)


def test_phase_5_sampler_integrates_with_posterior_list() -> None:
    first = GaussianPosterior(
        torch.zeros(2, 1, dtype=torch.double),
        torch.eye(2, dtype=torch.double),
    )
    second = GaussianPosterior(
        torch.ones(2, 2, dtype=torch.double),
        torch.eye(4, dtype=torch.double),
    )
    posterior = PosteriorList(first, second)
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=5)

    samples = sampler(posterior)

    assert samples.shape == torch.Size([32, 2, 3])
    assert torch.isfinite(samples).all()


def test_phase_5_qmc_samples_preserve_actual_gp_candidate_gradients() -> None:
    train_X = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double, requires_grad=True)
    sampler = SobolQMCNormalSampler(torch.Size([64]), seed=17)

    sampler(model.posterior(candidate)).mean().backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_phase_5_float32_qmc_base_samples_are_finite() -> None:
    sampler = SobolQMCNormalSampler(torch.Size([1024]), seed=13)

    samples = sampler(_unit_posterior(dtype=torch.float32))

    assert samples.dtype is torch.float32
    assert sampler.base_samples is not None
    assert sampler.base_samples.dtype is torch.float32
    assert torch.isfinite(sampler.base_samples).all()


def test_phase_5_module_to_moves_cache_and_reuses_compatible_metadata() -> None:
    sampler = IIDNormalSampler(torch.Size([8]), seed=7)
    sampler(_unit_posterior(dtype=torch.float32))

    moved = sampler.to(dtype=torch.double)
    cached = moved.base_samples
    samples = moved(_unit_posterior(dtype=torch.double))

    assert cached is not None
    assert cached.dtype is torch.double
    assert moved.base_samples is cached
    assert samples.dtype is torch.double
