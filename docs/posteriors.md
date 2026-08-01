# Posterior abstractions

MoTorch posteriors provide the probabilistic contract between models and downstream samplers, objectives, and acquisition functions.

## Tensor conventions

Analytic moments use:

```text
batch_shape × q × m
```

where `q` is the number of jointly evaluated candidate points and `m` is the number of outputs. Reparameterized samples use:

```text
sample_shape × batch_shape × q × m
```

A Gaussian covariance matrix flattens the `q × m` event dimensions in row-major order and has shape:

```text
batch_shape × (q * m) × (q * m)
```

## Posterior protocol

`motorch.posteriors.Posterior` is a runtime-checkable structural protocol exposing:

- `mean`;
- `variance`;
- differentiable `rsample`;
- optional caller-supplied base samples.

Implementations do not need to inherit from a common concrete base class when they satisfy this contract.

## GaussianPosterior

`GaussianPosterior` stores an explicit mean and full covariance matrix. It validates:

- finite tensors using `torch.float32` or `torch.float64`;
- matching dtype and device;
- positive candidate and output dimensions;
- mean and covariance shapes;
- covariance symmetry;
- positive-semidefinite covariance.

It supports arbitrary leading batch dimensions, multiple candidates, multiple outputs, and singular positive-semidefinite covariance matrices. Marginal variance is exposed with the same shape as the mean.

Sampling uses a differentiable eigendecomposition-based covariance root. Supplying the same base samples produces exactly the same result. Base samples must match the requested sample shape, posterior batch shape, candidate count, output count, dtype, and device.

## PosteriorList

`PosteriorList` combines independent posterior output groups that share batch and candidate dimensions. Means, variances, and samples are concatenated along the final output dimension. Combined base samples are split by component output size, which preserves deterministic base-sample reuse.

`PosteriorList` does not introduce cross-component covariance. Correlation within each component is retained by that component's own sampling implementation.

## Current limitations

- No model constructs these posteriors yet; model integration belongs to the next development phase.
- No separate sampler classes are included yet.
- CUDA behavior is supported by tensor-native operations but is not covered by a dedicated CI runner.
- Dense covariance storage and eigendecomposition are intended for correctness-first foundations, not large-scale structured linear algebra.
