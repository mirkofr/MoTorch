# MoTorch

MoTorch is an early-stage, low-level, modular, tensor-native optimization research library intended for PyTorch-based Bayesian optimization and scientific optimization research.

> **Development status:** Pre-alpha. The repository contains tensor-validation foundations and the first posterior abstractions. No Gaussian-process models, samplers, acquisition functions, or candidate-optimization algorithms are implemented yet.

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
- `PosteriorList` composition for independent output groups.

Future, separately implemented and tested layers are expected to include models, samplers, objectives, acquisition functions, and candidate optimization.

## Installation from source

```bash
git clone https://github.com/mirkofr/MoTorch.git
cd MoTorch
python -m pip install .
```

## Current usage

```python
import torch

from motorch.posteriors import GaussianPosterior

mean = torch.zeros(2, 1, dtype=torch.double)
covariance = torch.eye(2, dtype=torch.double)
posterior = GaussianPosterior(mean, covariance)

base_samples = torch.randn(8, 2, 1, dtype=torch.double)
samples = posterior.rsample(
    torch.Size([8]),
    base_samples=base_samples,
)
```

See [the tensor conventions](docs/tensor_conventions.md) and [posterior documentation](docs/posteriors.md) for the current public contracts.

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

MoTorch is an independent open-source project. It is not affiliated with, endorsed by, or maintained by the PyTorch or BoTorch project teams.
