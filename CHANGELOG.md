# Changelog

All notable changes to MoTorch will be documented in this file.

## Unreleased

- Added the Phase 3 model contract, exact `SingleTaskGP`, `FixedNoiseGP`, and model-list abstraction.
- Added differentiable exact marginal-likelihood training objectives and posterior construction.
- Added GPyTorch and LinearOperator as GP runtime dependencies.
- Added model shape, validation, likelihood, batching, fitting, and gradient tests.
- Added public model documentation.
- Added the Phase 2 posterior contract, dense Gaussian posterior, and posterior-list abstraction.
- Added differentiable reparameterized sampling with deterministic base-sample support.
- Added statistical, shape, gradient, covariance, batching, and validation tests for posteriors.
- Added public posterior documentation.
- Added Phase 1 tensor-shape, dtype, device, finite-value, gradient, and reproducible-randomness foundations.
- Added PyTorch as the first runtime dependency.

## 0.0.1 — Initial repository foundation

- Added the minimal importable `motorch` package.
- Added project metadata, tests, quality tooling, CI, and public governance documentation.
- No optimization functionality is included in this release foundation.
