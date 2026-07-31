# Contributing to MoTorch

MoTorch is pre-alpha. Contributions should remain small, focused, mathematically explicit, and consistent with PyTorch tensor conventions.

## Workflow

1. Create a feature or fix branch from `main`.
2. Make one coherent change without unrelated reformatting.
3. Add or update tests and documentation.
4. Run all relevant quality checks.
5. Open a pull request describing API, numerical, compatibility, and dependency impact.

## Development checks

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy src
python -m pytest
python -m build
```

## Expectations

- Add tests for expected behavior and relevant edge cases.
- Preserve typing quality and document public interfaces.
- For numerical work, include independent reference calculations, gradient checks, stability cases, or benchmarks as appropriate.
- Do not make unverified performance, accuracy, stability, or compatibility claims.
- Do not copy third-party source code without completing a license review and preserving required attribution.
- Keep private information, credentials, internal instructions, and unrelated material out of the repository.

By participating, contributors agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
