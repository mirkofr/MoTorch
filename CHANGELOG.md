# Changelog

All notable changes to MoTorch will be documented in this file.

## Unreleased

- Added the Phase 6 analytic acquisition-function contract, posterior mean, probability of improvement, expected improvement, and upper confidence bound.
- Added independent formula, shape, dtype, gradient, minimization, and validation tests for analytic acquisitions.
- Added public analytic-acquisition documentation.
- Added the Phase 5 posterior sampler contract, IID normal sampler, and scrambled Sobol QMC normal sampler.
- Added cached reproducible base samples with explicit seeds, reset behavior, dtype/device preservation, and differentiable posterior sampling.
- Added sampler shape, reproducibility, statistical, gradient, validation, serialization, and conditional CUDA tests.
- Added public posterior-sampling documentation.
- Added typed exact-GP fitting configuration and structured convergence diagnostics.
- Added Adam and L-BFGS marginal-likelihood fitting with best-state restoration.
- Added bounded retry, learning-rate reduction, and Cholesky-jitter escalation behavior.
- Added recoverable fitting and numerical warnings plus deterministic fitting test mode.
- Added reference fitting, failure-path, reproducibility, and configuration tests.
- Added public model-fitting documentation.
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
