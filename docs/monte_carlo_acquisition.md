# Monte Carlo acquisition functions

Phase 8 adds differentiable sample-based acquisition functions for joint Bayesian optimization batches.

## Public API

```python
from motorch.acquisition import (
    MCAcquisitionFunction,
    qConstrainedExpectedImprovement,
    qExpectedImprovement,
)
from motorch.objectives import IdentityMCObjective, MCAcquisitionObjective
```

## Tensor shapes

Candidate inputs use `batch_shape x q x d`. Posterior samples use `sample_shape x batch_shape x q x m`. A Monte Carlo objective maps those samples to `sample_shape x batch_shape x q`, and an acquisition returns `batch_shape` after reducing the candidate and sample dimensions.

## Expected improvement

For sampled scalar utilities `Y`, `qExpectedImprovement` computes the sample average of the largest non-negative improvement among the jointly proposed points. `best_f` must be one finite scalar. Maximization is the default; set `maximize=False` for an explicit minimization convention.

```python
sampler = SobolQMCNormalSampler(torch.Size([256]), seed=0)
acquisition = qExpectedImprovement(
    model,
    best_f=train_Y.max(),
    sampler=sampler,
)
```

The default `IdentityMCObjective` requires one posterior output. Multi-output or transformed utilities must use an explicit `MCAcquisitionObjective` implementation.

## Pending points

Call `set_pending_points(X)` to include fixed pending candidates in joint sample utility. Pending points must match candidate batch shape, feature dimension, dtype, and device. Call `set_pending_points(None)` to clear them. This interface supports Phase 7 sequential candidate generation.

## Constraints

`qConstrainedExpectedImprovement` accepts a non-empty sequence of sample constraints. Each callable receives posterior samples and returns one value per sampled candidate. Values less than or equal to zero are feasible. A positive `eta` controls the sigmoid relaxation around the feasibility boundary, preserving gradients while approximating an indicator.

## Randomness and gradients

Randomness is controlled by the supplied posterior sampler. Cached base samples make repeated evaluations deterministic and preserve differentiability with respect to candidates. Reset the sampler base samples when an independent Monte Carlo draw is required.

## Limitations

- The default objective supports one posterior output.
- Constraint callables operate on posterior samples and must return matching dtype, device, and shape.
- Smooth feasibility is an approximation controlled by `eta`.
- General candidate-space constraints remain part of Phase 9.
- Dedicated CUDA CI is not yet available.
