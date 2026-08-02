# Acquisition optimization

Phase 7 provides bounded continuous maximization of MoTorch acquisition functions.

## Public API

```python
from motorch.optim import OptimizeAcqfOptions, optimize_acqf
```

`optimize_acqf` follows the candidate convention `q x d` for the returned candidate tensor. It maximizes the acquisition value; minimization objectives must already be represented by a sign-adjusted acquisition function.

## Algorithm

1. Validate finite `2 x d` box bounds.
2. Generate `raw_samples x q x d` scrambled Sobol points.
3. Evaluate the acquisition function and retain the best finite raw starts.
4. Map each start to an unconstrained logistic parameterization.
5. Refine every restart independently using Adam or L-BFGS.
6. Exclude failed restarts with `OptimizationWarning` and return the best successful result.

The logistic parameterization keeps every local iterate inside the supplied box. Fixed features are reapplied at every evaluation, so they remain exact during optimization.

## Example

```python
import torch

from motorch.acquisition import ExpectedImprovement
from motorch.optim import OptimizeAcqfOptions, optimize_acqf

acquisition = ExpectedImprovement(model, best_f=train_Y.max())
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

candidate, value = optimize_acqf(
    acquisition,
    bounds,
    q=1,
    num_restarts=8,
    raw_samples=128,
    options=OptimizeAcqfOptions(max_steps=200),
    seed=0,
)
```

## Joint and sequential interfaces

Joint mode directly optimizes a `q x d` tensor and therefore requires an acquisition function that supports that `q` value.

Sequential `q > 1` generation is available for acquisition functions implementing `set_pending_points(X | None)`. The optimizer updates the accumulated pending set before each one-point optimization and clears it afterward. Phase 6 analytic acquisitions support only `q=1`, so they should use ordinary joint mode with `q=1`.

## Diagnostics

Set `return_diagnostics=True` to receive `OptimizationResult`, including every `RestartResult`, success state, step count, and termination message. Failed restarts never replace a successful result.

## Reproducibility

A non-negative seed controls Sobol scrambling. The same seed, bounds, options, and deterministic acquisition function reproduce the same raw starts and optimization result without consuming PyTorch's global random state.

## Dtype and device

Bounds determine generated-candidate dtype and device. The optimizer does not silently cast or move user tensors. PyTorch's Sobol engine generates on CPU and the points are then transferred to the bounds device before acquisition evaluation.

## Limitations

- Only box constraints and fixed features are supported.
- General linear and nonlinear constraints are reserved for Phase 9.
- The optimizer assumes each local restart evaluates to one scalar utility.
- Sequential generation requires pending-point support from the acquisition function.
- Dedicated CUDA CI is not yet available.
- SciPy is not a dependency; the implementation remains tensor-native and differentiable.
