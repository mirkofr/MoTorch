# Model abstractions

MoTorch models are `torch.nn.Module` objects that map candidate tensors to posterior distributions.

## Tensor conventions

Training data use:

```text
train_X: batch_shape × n × d
train_Y: batch_shape × n × m
```

Candidate inputs use:

```text
X: candidate_batch_shape × q × d
```

The candidate batch shape must broadcast with the model training batch shape. Returned posterior moments use:

```text
posterior_batch_shape × q × m
```

where `posterior_batch_shape` is the broadcasted batch shape.

## Model

`motorch.models.Model` is the abstract base class for probabilistic models. It exposes:

- `num_outputs`;
- `posterior(X, observation_noise=False)`;
- `forward`, which delegates to `posterior`.

Model implementations preserve PyTorch module behavior, including parameter registration, `.train()`, `.eval()`, state dictionaries, autograd, and `.to(device=..., dtype=...)`.

## SingleTaskGP

`SingleTaskGP` is an exact Gaussian process with:

- a constant mean;
- an ARD radial-basis-function kernel;
- an output-scale parameter;
- learned homoskedastic Gaussian observation noise.

Multiple outputs are represented as conditionally independent batched GPs. Each output has separate mean, kernel, output-scale, and likelihood parameters. Cross-output posterior covariance is zero.

```python
import torch

from motorch.models import SingleTaskGP

train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X * 6.0)
model = SingleTaskGP(train_X, train_Y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(25):
    optimizer.zero_grad()
    loss = model.training_loss()
    loss.backward()
    optimizer.step()

X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(X)
```

`training_loss()` returns the summed negative exact marginal log likelihood. It is a differentiable scalar intended for direct PyTorch optimization and for the fitting utilities introduced in a later phase.

Setting `observation_noise=True` applies the learned Gaussian likelihood to the latent posterior.

## FixedNoiseGP

`FixedNoiseGP` accepts known observation variances:

```text
train_Yvar: batch_shape × n × m
```

Every variance must be finite, strictly positive, and match `train_Y` in shape, dtype, and device.

```python
train_Yvar = torch.full_like(train_Y, 0.01)
model = FixedNoiseGP(train_X, train_Y, train_Yvar)
```

Candidate-specific observation variances cannot be inferred from training variances. Therefore `FixedNoiseGP.posterior(..., observation_noise=True)` raises an actionable error instead of guessing a noise value.

## ModelList

`ModelList` combines independent models that accept the same candidate tensor. Component posteriors are joined through `PosteriorList`, and outputs are concatenated in model order.

```python
from motorch.models import ModelList

combined = ModelList(first_model, second_model)
posterior = combined.posterior(X)
```

The combined model does not introduce cross-component covariance.

## Dtype, device, and gradients

Exact GP models support `torch.float32` and `torch.float64`. Training inputs, outcomes, fixed noise, and candidate tensors must have matching dtype and device. No tensor is silently cast or moved.

Posterior means, covariance matrices, and reparameterized samples remain differentiable with respect to candidate inputs and model parameters where supported by PyTorch and GPyTorch.

## Numerical assumptions and limitations

- Models are exact GPs and scale cubically with the number of observations.
- Dense MoTorch posterior covariance construction is correctness-first and is not intended for large candidate sets.
- Inputs and outcomes are not normalized or standardized automatically.
- Hyperparameters are not fitted during construction.
- General fitting retries, convergence diagnostics, and jitter policy belong to the fitting phase.
- Dedicated CUDA CI is not yet available.
