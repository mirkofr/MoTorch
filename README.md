# MoTorch

MoTorch is an early-stage, low-level, modular, tensor-native optimization research library intended for PyTorch-based Bayesian optimization and scientific optimization research.

> **Development status:** Pre-alpha. The repository contains tensor foundations, posterior abstractions, exact Gaussian-process models, and reusable model-fitting utilities. Samplers, acquisition functions, and candidate optimization are not implemented yet.

## Scope

MoTorch is intended to provide:

- modular optimization research components;
- PyTorch-native tensor operations;
- differentiable optimization components where mathematically appropriate;
- Bayesian optimization abstractions developed in independently tested layers.

MoTorch is explicitly **not**:

- a web application;
- a SaaS platform;
- an experiment database;
- an LLM assistant.

## Current architecture

The implemented foundation currently includes:

- shared tensor shape, dtype, device, finite-value, gradient, and randomness conventions;
- a structural `Posterior` protocol;
- a dense, batched `GaussianPosterior` with differentiable reparameterized sampling;
- deterministic caller-supplied base samples;
- `PosteriorList` composition for independent output groups;
- an abstract `Model` contract;
- exact `SingleTaskGP` and `FixedNoiseGP` models;
- `ModelList` composition for independent model groups;
- typed exact-GP fitting configuration, diagnostics, retries, jitter policy, and deterministic test mode.

Future, separately implemented and tested layers are expected to include samplers, objectives, acquisition functions, and candidate optimization.

## Installation from source

```bash
git clone https://github.com/mirkofr/MoTorch.git
cd MoTorch
python -m pip install .
```

## Current usage

```python
import torch

from motorch.fit import FitOptions, fit_gp
from motorch.models import SingleTaskGP

train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X * 6.0)
model = SingleTaskGP(train_X, train_Y)

result = fit_gp(
    model,
    options=FitOptions(max_steps=100, deterministic=True, seed=0),
)

X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(X)
print(result.converged, result.best_loss)
```

See [the tensor conventions](docs/tensor_conventions.md), [posterior documentation](docs/posteriors.md), [model documentation](docs/models.md), and [fitting documentation](docs/fitting.md) for the current public contracts.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Testing and quality checks

```bash
python -m pytest
ruff format --check .
ruff check .
mypy src
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations and [SECURITY.md](SECURITY.md) for vulnerability reporting guidance.

## License

MoTorch is available under the [MIT License](LICENSE).

## Independence

MoTorch is an independent open-source project. It is not affiliated with, endorsed by, or maintained by the PyTorch, GPyTorch, or BoTorch project teams.
