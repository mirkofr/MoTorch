# Posterior sampling

MoTorch posterior samplers generate standard-normal base samples and pass them to a posterior's differentiable `rsample` method.

```python
import torch

from motorch.posteriors import GaussianPosterior
from motorch.sampling import SobolQMCNormalSampler

mean = torch.zeros(2, 1, dtype=torch.double)
covariance = torch.eye(2, dtype=torch.double)
posterior = GaussianPosterior(mean, covariance)

sampler = SobolQMCNormalSampler(
    sample_shape=torch.Size([256]),
    seed=0,
)
samples = sampler(posterior)
```

## Public API

The sampling package exposes:

- `PosteriorSampler`, the abstract stateful sampler contract;
- `IIDNormalSampler`, using independent pseudorandom standard normals;
- `SobolQMCNormalSampler`, using scrambled Sobol points transformed through the inverse standard-normal CDF.

All samplers inherit from `torch.nn.Module` and implement computation in `forward`.

## Shape contract

For a posterior with:

```text
base_sample_shape = posterior_batch_shape × q × m
```

and sampler `sample_shape`, generated base samples and returned posterior samples use:

```text
sample_shape × posterior_batch_shape × q × m
```

Every explicit sample dimension must be positive. An empty `sample_shape` is supported and represents one sample without an additional leading sample dimension.

## Base-sample caching

The first sampler call creates base samples. Repeated calls reuse the same tensor while posterior shape, dtype, and device are unchanged. This common-random-number behavior is important for stable differentiable Monte Carlo acquisition optimization.

`reset_base_samples()` clears the cache. The next call regenerates samples from the configured seed, so reset-and-redraw is reproducible.

Cached base samples are runtime state and are intentionally excluded from `state_dict`. The sampler seed and sample shape are constructor configuration and should be recreated explicitly when loading application configuration.

## IID normal sampling

`IIDNormalSampler` uses a local `torch.Generator` initialized from `seed`. It does not consume or mutate the process-wide PyTorch random state.

```python
sampler = IIDNormalSampler(torch.Size([128]), seed=17)
```

A different seed produces a different deterministic IID base-sample tensor.

## Sobol QMC normal sampling

`SobolQMCNormalSampler` creates scrambled Sobol points in the unit hypercube using `torch.quasirandom.SobolEngine`. Uniform values are clamped away from exactly zero and one before applying `torch.special.ndtri`, preventing infinite normal values.

```python
sampler = SobolQMCNormalSampler(torch.Size([512]), seed=17)
```

The Sobol dimension equals the flattened posterior base-sample event and batch dimensions. QMC construction is correctness-oriented and may become expensive for very large flattened dimensions.

PyTorch's Sobol engine generates points on CPU. MoTorch transforms them in the requested dtype and transfers the completed base-sample tensor to the posterior device. No posterior tensor is silently moved or cast.

## Dtype, device, and gradients

Base samples match `posterior.dtype` and `posterior.device`. Samplers support `torch.float32` and `torch.float64` whenever the posterior supports them.

Base samples are constants. Gradients pass through the posterior's reparameterization to posterior means, covariance parameters, model parameters, and candidate tensors where the posterior implementation supports those derivatives.

## Reproducibility

Reproducibility is controlled by:

- the sampler class;
- `sample_shape`;
- `seed`;
- posterior base-sample shape;
- dtype and device;
- the installed PyTorch version and supported backend behavior.

Identical sampler configuration and posterior metadata reproduce identical cached base samples within the same supported environment. Bitwise equality across different PyTorch versions or hardware backends is not guaranteed.

## Failure modes

Samplers raise actionable errors for:

- non-positive sample dimensions;
- negative seeds;
- objects that do not satisfy the posterior sampling contract;
- internal base-sample shape mismatches.

Posterior-specific shape, dtype, device, and numerical validation remains the posterior's responsibility.

## Current limitations

- Samplers currently target normal reparameterized posteriors.
- Antithetic sampling is not implemented.
- Nested or collapsed batch-dimension policies are not implemented.
- Cached base samples are not persisted in module state dictionaries.
- Dedicated CUDA CI is not yet available.
