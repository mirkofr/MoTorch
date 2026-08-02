# Analytic acquisition functions

Phase 6 provides closed-form acquisition functions for Gaussian single-output posteriors.

```python
import torch

from motorch.acquisition import ExpectedImprovement
from motorch.models import SingleTaskGP

train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X * 6.0)
model = SingleTaskGP(train_X, train_Y)

acquisition = ExpectedImprovement(model, best_f=train_Y.max())
X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
values = acquisition(X)
```

## Public API

- `AcquisitionFunction`: abstract PyTorch module mapping candidates to utilities.
- `PosteriorMean`: posterior mean, or its negative for minimization.
- `ProbabilityOfImprovement`: Gaussian probability of improving over `best_f`.
- `ExpectedImprovement`: Gaussian expected improvement over `best_f`.
- `UpperConfidenceBound`: signed posterior mean plus `sqrt(beta)` times posterior standard deviation.

## Shape contract

Candidate inputs use `batch_shape x q x d`. Phase 6 analytic functions require `q=1` and a single-output posterior. Returned acquisition values use `batch_shape`.

This restriction is explicit because closed-form single-point formulas do not represent joint utility for `q>1`. Batch acquisition functions are planned separately.

## Maximization and minimization

Acquisition optimization is conventionally expressed as maximization. With `maximize=False`:

- posterior mean returns the negative posterior mean;
- improvement is measured as `best_f - mean`;
- UCB uses the negative mean plus the exploration bonus.

## Zero variance

At deterministic posterior points:

- probability of improvement is one only for strict positive improvement;
- expected improvement is the positive part of deterministic improvement;
- UCB reduces to the signed mean.

## Validation

Analytic functions reject:

- non-tensor candidate inputs;
- candidate tensors without `q` and feature dimensions;
- `q != 1`;
- multi-output posterior moments;
- mismatched or non-finite posterior moments;
- negative posterior variance;
- non-scalar or non-finite `best_f`;
- negative or non-finite `beta`.

## Dtype, device, and gradients

Scalar configuration buffers are converted to the posterior moment dtype and device during evaluation. Acquisition values remain differentiable with respect to candidate inputs and model parameters wherever the model posterior provides those gradients.

## Current limitations

- Analytic functions assume Gaussian posterior moments.
- Only single-output, single-candidate formulas are included.
- Observation-noise selection and posterior transforms are not exposed yet.
- Multi-objective and constrained acquisitions are future phases.
