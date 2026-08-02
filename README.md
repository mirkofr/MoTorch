# MoTorch

MoTorch is an early-stage, low-level, modular, tensor-native optimization research library intended for PyTorch-based Bayesian optimization and scientific optimization research.

> **Development status:** Pre-alpha. The repository contains tensor and posterior foundations, exact Gaussian-process models, reusable fitting utilities, differentiable posterior samplers, analytic and Monte Carlo acquisition functions, and bounded acquisition optimization.

## Scope

MoTorch provides composable PyTorch-native mathematical components for Bayesian optimization research. It is explicitly not a web application, SaaS platform, experiment database, or LLM assistant.

## Current architecture

Implemented layers include:

- shared tensor shape, dtype, device, finite-value, gradient, and randomness conventions;
- dense Gaussian posteriors with differentiable reparameterized sampling;
- exact `SingleTaskGP`, `FixedNoiseGP`, and independent model-list composition;
- typed exact-GP fitting with diagnostics, retries, jitter handling, Adam, and L-BFGS;
- cached IID and scrambled Sobol QMC posterior samplers;
- analytic posterior mean, probability of improvement, expected improvement, and upper confidence bound;
- sampled objective contracts, `qExpectedImprovement`, pending points, and constrained MC improvement;
- Sobol-initialized bounded multistart acquisition optimization with fixed features and restart diagnostics.

## Installation from source

```bash
git clone https://github.com/mirkofr/MoTorch.git
cd MoTorch
python -m pip install .
```

## Minimal Bayesian optimization components

```python
import torch

from motorch.acquisition import qExpectedImprovement
from motorch.fit import FitOptions, fit_gp
from motorch.models import SingleTaskGP
from motorch.optim import optimize_acqf
from motorch.sampling import SobolQMCNormalSampler

train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X * 6.0)
model = SingleTaskGP(train_X, train_Y)
fit_gp(model, options=FitOptions(max_steps=100, deterministic=True, seed=0))

sampler = SobolQMCNormalSampler(torch.Size([256]), seed=0)
acquisition = qExpectedImprovement(model, train_Y.max(), sampler)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
candidates, values = optimize_acqf(
    acquisition,
    bounds,
    q=2,
    num_restarts=8,
    raw_samples=128,
    seed=0,
)
```

See the [documentation index](docs/index.md) for tensor shapes, probabilistic contracts, fitting, sampling, acquisitions, and optimization behavior.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

MoTorch is available under the [MIT License](LICENSE).

## Independence

MoTorch is an independent open-source project. It is not affiliated with, endorsed by, or maintained by the PyTorch, GPyTorch, or BoTorch project teams.
