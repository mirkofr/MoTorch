# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. Phases 1–7 provide tensor foundations, posterior and model abstractions, model fitting, posterior samplers, analytic acquisitions, and bounded acquisition optimization. Phase 8 adds sampled objectives and differentiable Monte Carlo acquisition functions.

## Intended architecture

The architecture separates models, posteriors, fitting, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

## Current capabilities

- Typed exception and warning hierarchies.
- Shape, dtype, device, and finite-value validation.
- Explicit seeded randomness and reusable posterior base samples.
- Dense batched Gaussian posteriors and posterior-list composition.
- Exact single-task and fixed-noise Gaussian-process models.
- Typed model fitting with diagnostics, retries, and jitter handling.
- IID and scrambled Sobol QMC normal posterior sampling.
- Analytic posterior mean, probability of improvement, expected improvement, and upper confidence bound.
- Sobol-initialized bounded multistart acquisition optimization.
- Fixed-feature, failed-restart, joint-batch, and pending-aware sequential interfaces.
- Sampled Monte Carlo objective contract and single-output identity objective.
- Joint `qExpectedImprovement` with explicit maximization or minimization direction.
- Pending-point handling and smoothly constrained Monte Carlo expected improvement.

## Current limitations

- The default Monte Carlo objective supports one posterior output.
- Acquisition optimization currently supports box bounds and fixed features only.
- Candidate-space linear and nonlinear constraints are reserved for Phase 9.
- Exact GP training scales cubically with observation count.
- Dense posterior covariance storage is not intended for large structured problems.
- Joint `ModelList` fitting is not implemented.
- CUDA behavior is not yet exercised by dedicated CI jobs.
- Compatibility with other optimization libraries is not claimed.

## Foundation documentation

- [Tensor conventions](tensor_conventions.md)
- [Posterior abstractions](posteriors.md)
- [Model abstractions](models.md)
- [Model fitting](fitting.md)
- [Posterior sampling](sampling.md)
- [Analytic acquisition functions](acquisition.md)
- [Acquisition optimization](optimization.md)
- [Monte Carlo acquisition functions](monte_carlo_acquisition.md)

## Next planned work

The next phase is reusable input and outcome transforms plus candidate-space constraints, including normalization, standardization, chaining, inverse transforms, linear constraints, nonlinear hooks, and feasibility utilities.
