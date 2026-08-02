import copy

import pytest
import torch

from motorch.posteriors import GaussianPosterior
from motorch.sampling import IIDNormalSampler, SobolQMCNormalSampler


def _posterior(
    *,
    batch_shape: torch.Size = torch.Size(),
    q: int = 2,
    m: int = 1,
    dtype: torch.dtype = torch.double,
    device: torch.device | str = "cpu",
) -> GaussianPosterior:
    mean = torch.zeros(*batch_shape, q, m, dtype=dtype, device=device)
    event_size = q * m
    covariance = torch.eye(event_size, dtype=dtype, device=device).expand(
        *batch_shape, event_size, event_size
    )
    return GaussianPosterior(mean, covariance)


@pytest.mark.parametrize("sampler_type", [IIDNormalSampler, SobolQMCNormalSampler])
def test_sampler_returns_documented_shape_and_caches_base_samples(
    sampler_type: type[IIDNormalSampler] | type[SobolQMCNormalSampler],
) -> None:
    posterior = _posterior(batch_shape=torch.Size([2, 3]), q=4, m=2)
    sampler = sampler_type(torch.Size([5, 7]), seed=11)

    first = sampler(posterior)
    cached = sampler.base_samples
    second = sampler(posterior)

    assert first.shape == torch.Size([5, 7, 2, 3, 4, 2])
    assert cached is not None
    assert cached.shape == first.shape
    assert sampler.base_samples is cached
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("sampler_type", [IIDNormalSampler, SobolQMCNormalSampler])
def test_sampler_same_seed_reproduces_base_samples(
    sampler_type: type[IIDNormalSampler] | type[SobolQMCNormalSampler],
) -> None:
    posterior = _posterior(q=3, m=2)
    first = sampler_type(torch.Size([16]), seed=23)
    second = sampler_type(torch.Size([16]), seed=23)

    first(posterior)
    second(posterior)

    assert first.base_samples is not None
    assert second.base_samples is not None
    torch.testing.assert_close(first.base_samples, second.base_samples)


@pytest.mark.parametrize("sampler_type", [IIDNormalSampler, SobolQMCNormalSampler])
def test_sampler_reset_regenerates_configured_seed(
    sampler_type: type[IIDNormalSampler] | type[SobolQMCNormalSampler],
) -> None:
    posterior = _posterior()
    sampler = sampler_type(torch.Size([32]), seed=5)
    first = sampler(posterior)

    sampler.reset_base_samples()
    assert sampler.base_samples is None
    second = sampler(posterior)

    torch.testing.assert_close(first, second)


def test_iid_sampler_different_seed_changes_base_samples() -> None:
    posterior = _posterior()
    first = IIDNormalSampler(torch.Size([64]), seed=1)
    second = IIDNormalSampler(torch.Size([64]), seed=2)

    first(posterior)
    second(posterior)

    assert first.base_samples is not None
    assert second.base_samples is not None
    assert not torch.equal(first.base_samples, second.base_samples)


@pytest.mark.parametrize("sampler_type", [IIDNormalSampler, SobolQMCNormalSampler])
def test_sampler_rebuilds_cache_for_shape_and_dtype_changes(
    sampler_type: type[IIDNormalSampler] | type[SobolQMCNormalSampler],
) -> None:
    sampler = sampler_type(torch.Size([8]), seed=3)
    sampler(_posterior(q=2, dtype=torch.float32))
    first = sampler.base_samples
    sampler(_posterior(q=3, dtype=torch.float64))
    second = sampler.base_samples

    assert first is not None
    assert second is not None
    assert first.shape == torch.Size([8, 2, 1])
    assert first.dtype is torch.float32
    assert second.shape == torch.Size([8, 3, 1])
    assert second.dtype is torch.float64


def test_sampler_preserves_gradients_through_reparameterized_samples() -> None:
    mean = torch.tensor([[0.2], [0.7]], dtype=torch.double, requires_grad=True)
    raw_scale = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    covariance = torch.diag(
        torch.stack((raw_scale.square(), (raw_scale + 0.4).square()))
    )
    posterior = GaussianPosterior(mean, covariance)
    sampler = SobolQMCNormalSampler(torch.Size([64]), seed=13)

    loss = sampler(posterior).square().mean()
    loss.backward()

    assert mean.grad is not None
    assert raw_scale.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(raw_scale.grad)


def test_sampler_cache_is_not_persisted_in_state_dict() -> None:
    posterior = _posterior()
    sampler = IIDNormalSampler(torch.Size([8]), seed=17)
    sampler(posterior)

    state = copy.deepcopy(sampler.state_dict())
    restored = IIDNormalSampler(torch.Size([8]), seed=17)
    restored.load_state_dict(state)

    assert state == {}
    assert restored.base_samples is None
    torch.testing.assert_close(sampler(posterior), restored(posterior))


@pytest.mark.parametrize("sample_shape", [torch.Size([-1]), torch.Size([2, 0])])
def test_sampler_rejects_nonpositive_sample_shape(sample_shape: torch.Size) -> None:
    with pytest.raises(ValueError, match="sample_shape dimension must be positive"):
        IIDNormalSampler(sample_shape)


def test_sampler_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        SobolQMCNormalSampler(torch.Size([4]), seed=-1)


def test_sampler_rejects_incomplete_posterior_contract() -> None:
    sampler = IIDNormalSampler(torch.Size([4]))

    with pytest.raises(TypeError, match="posterior must provide"):
        sampler(object())  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_iid_sampler_preserves_cuda_device() -> None:
    posterior = _posterior(device="cuda")
    sampler = IIDNormalSampler(torch.Size([8]), seed=2).cuda()

    samples = sampler(posterior)

    assert samples.device.type == "cuda"
    assert sampler.base_samples is not None
    assert sampler.base_samples.device.type == "cuda"
