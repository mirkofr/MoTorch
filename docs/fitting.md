# Model fitting

MoTorch fits exact Gaussian-process models by minimizing their negative exact marginal log likelihood.

```python
import torch

from motorch.fit import FitOptions, fit_gp
from motorch.models import SingleTaskGP

train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X * 6.0)
model = SingleTaskGP(train_X, train_Y)

result = fit_gp(
    model,
    options=FitOptions(
        optimizer="adam",
        learning_rate=0.05,
        max_steps=100,
        deterministic=True,
        seed=0,
    ),
)

print(result.converged, result.best_loss)
```

## Public API

`fit_gp(model, *, options=None)` mutates a model in place and returns a `FitResult`. The model must expose a differentiable scalar `training_loss()` and standard PyTorch parameter and state-dictionary methods.

`FitOptions` controls:

- optimizer choice: Adam or L-BFGS;
- learning rate and maximum steps;
- gradient and loss-change convergence tolerances;
- patience for loss-based convergence;
- retry count and retry learning-rate reduction;
- initial, multiplied, and maximum Cholesky jitter;
- deterministic test mode and seed;
- retry and terminal warning behavior.

Configuration is validated when constructed. Invalid numerical limits fail before model mutation.

## Convergence

An attempt is considered converged when either:

1. the largest absolute trainable-parameter gradient is no greater than `tolerance_grad`; or
2. the relative loss change remains within `tolerance_loss` for `patience` consecutive steps.

Reaching `max_steps` is reported as non-convergence even when the best state improves substantially. Callers should inspect `FitResult.termination`, `FitResult.message`, and every `FitAttempt` rather than assuming that a finite final loss proves convergence.

## Retries and state restoration

Each retry:

- starts from the best finite model state found so far;
- lowers the learning rate by `retry_learning_rate_factor`;
- raises Cholesky jitter by `jitter_multiplier`, capped at `max_jitter`;
- records the effective optimizer settings and numerical outcome.

Recoverable retries emit `FittingWarning` by default. Exhausted non-finite or factorization failures emit `NumericalWarning`. Warnings can be disabled for controlled testing, but the result object still records the failure.

## Jitter policy

Jitter is applied explicitly through the GPyTorch Cholesky setting during each attempt. MoTorch does not silently choose an unlimited fallback value. Every attempt records the jitter used, allowing numerical behavior to be reproduced and reviewed.

Jitter can stabilize nearly singular covariance matrices, but excessive jitter changes the effective numerical problem. A successful fit with large jitter should therefore be interpreted carefully.

## Deterministic mode

`deterministic=True` temporarily:

- seeds CPU and available CUDA random generators;
- enables deterministic PyTorch algorithms;
- restores the caller's CPU/CUDA random states and deterministic-algorithm setting afterward.

Deterministic mode improves reproducibility for supported operations. It does not guarantee bitwise equality across different devices, PyTorch versions, BLAS implementations, or unsupported nondeterministic kernels.

## Dtype, device, and gradients

Fitting preserves model dtype and device. It does not cast or move model parameters. Exact GP models currently support `torch.float32` and `torch.float64`; double precision is recommended for numerical tests and difficult covariance systems.

The optimizer consumes gradients from `training_loss()`. Posterior gradients with respect to candidates remain available after fitting.

## Failure modes

Actionable diagnostics distinguish:

- maximum-step termination;
- non-finite loss;
- non-finite gradients;
- numerical runtime failures;
- loss-tolerance convergence;
- gradient-tolerance convergence.

A model with no trainable parameters, an incompatible fitting contract, or a non-scalar training loss raises immediately.

## Current limitations

- The fitting utility targets exact GP-like models exposing `training_loss()`.
- It does not fit `ModelList` as one joint objective.
- It does not standardize outcomes or normalize inputs automatically.
- It does not claim global hyperparameter optimality.
- Long-running benchmark orchestration and scheduler integration are outside the core fitting layer.
