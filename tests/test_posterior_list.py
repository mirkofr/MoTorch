import pytest
import torch

from motorch.exceptions import DTypeError, ShapeError
from motorch.posteriors import GaussianPosterior, Posterior, PosteriorList


def diagonal_posterior(
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> GaussianPosterior:
    covariance = torch.diag_embed(variance.reshape(*variance.shape[:-2], -1))
    return GaussianPosterior(mean, covariance)


def test_posterior_list_combines_moments_and_satisfies_protocol() -> None:
    first_mean = torch.tensor([[[1.0], [2.0]]], dtype=torch.double)
    first_variance = torch.tensor([[[0.5], [0.7]]], dtype=torch.double)
    second_mean = torch.tensor(
        [[[3.0, 4.0], [5.0, 6.0]]],
        dtype=torch.double,
    )
    second_variance = torch.tensor(
        [[[0.8, 0.9], [1.0, 1.1]]],
        dtype=torch.double,
    )
    posterior = PosteriorList(
        diagonal_posterior(first_mean, first_variance),
        diagonal_posterior(second_mean, second_variance),
    )

    assert isinstance(posterior, Posterior)
    assert posterior.mean.shape == torch.Size([1, 2, 3])
    assert posterior.variance.shape == torch.Size([1, 2, 3])
    assert torch.equal(posterior.mean, torch.cat([first_mean, second_mean], dim=-1))
    assert torch.equal(
        posterior.variance,
        torch.cat([first_variance, second_variance], dim=-1),
    )
    assert posterior.batch_shape == torch.Size([1])
    assert posterior.event_shape == torch.Size([2, 3])
    assert posterior.base_sample_shape == torch.Size([1, 2, 3])


def test_posterior_list_rsample_splits_base_samples_by_output() -> None:
    first = diagonal_posterior(
        torch.zeros(2, 1, dtype=torch.double),
        torch.ones(2, 1, dtype=torch.double),
    )
    second = diagonal_posterior(
        torch.zeros(2, 2, dtype=torch.double),
        torch.ones(2, 2, dtype=torch.double),
    )
    posterior = PosteriorList.from_sequence([first, second])
    base_samples = torch.arange(24, dtype=torch.double).reshape(4, 2, 3) / 10

    combined = posterior.rsample(torch.Size([4]), base_samples=base_samples)
    expected = torch.cat(
        [
            first.rsample(
                torch.Size([4]),
                base_samples=base_samples[..., :1],
            ),
            second.rsample(
                torch.Size([4]),
                base_samples=base_samples[..., 1:],
            ),
        ],
        dim=-1,
    )

    assert torch.equal(combined, expected)
    assert combined.shape == torch.Size([4, 2, 3])


def test_posterior_list_rejects_empty_components() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PosteriorList()


def test_posterior_list_rejects_incompatible_candidate_dimensions() -> None:
    first = diagonal_posterior(
        torch.zeros(2, 1, dtype=torch.double),
        torch.ones(2, 1, dtype=torch.double),
    )
    second = diagonal_posterior(
        torch.zeros(3, 1, dtype=torch.double),
        torch.ones(3, 1, dtype=torch.double),
    )
    with pytest.raises(ShapeError, match="candidate dimensions"):
        PosteriorList(first, second)


def test_posterior_list_rejects_dtype_mismatch() -> None:
    first = diagonal_posterior(
        torch.zeros(2, 1, dtype=torch.float32),
        torch.ones(2, 1, dtype=torch.float32),
    )
    second = diagonal_posterior(
        torch.zeros(2, 1, dtype=torch.float64),
        torch.ones(2, 1, dtype=torch.float64),
    )
    with pytest.raises(DTypeError, match=r"posterior\[1\]"):
        PosteriorList(first, second)


def test_posterior_list_rejects_invalid_combined_base_sample_shape() -> None:
    component = diagonal_posterior(
        torch.zeros(2, 1, dtype=torch.double),
        torch.ones(2, 1, dtype=torch.double),
    )
    posterior = PosteriorList(component, component)
    with pytest.raises(ShapeError, match=r"\(3, 2, 2\).*(3, 2, 1)"):
        posterior.rsample(
            torch.Size([3]),
            base_samples=torch.zeros(3, 2, 1, dtype=torch.double),
        )
