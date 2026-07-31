# MoTorch

MoTorch is an early-stage, low-level, modular, tensor-native optimization research library intended for PyTorch-based Bayesian optimization and scientific optimization research.

> **Development status:** Pre-alpha. The repository contains tensor and validation foundations only. No Gaussian-process models, posteriors, acquisition functions, samplers, or candidate-optimization algorithms are implemented yet.

## Scope

MoTorch is intended to provide:

- modular optimization research components;
- PyTorch-native tensor operations;
- differentiable optimization components where mathematically appropriate;
- future Bayesian optimization abstractions.

MoTorch is explicitly **not**:

- a web application;
- a SaaS platform;
- an experiment database;
- an LLM assistant.

## Planned architecture

Future, separately implemented and tested layers are expected to include models, posteriors, samplers, objectives, acquisition functions, and candidate optimization. These names describe the planned architecture, not currently available functionality.

## Installation from source

```bash
git clone https://github.com/mirkofr/MoTorch.git
cd MoTorch
python -m pip install .
```

## Current usage

```python
import torch

from motorch.utils import validate_shape

x = torch.rand(4, 2, dtype=torch.double)
validate_shape(x, name="x", module="example", trailing_shape=(2,))
```

See [the tensor conventions](docs/tensor_conventions.md) for the current public contracts.

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
