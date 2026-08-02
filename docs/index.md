# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. Phase 1 provides shared tensor foundations, Phase 2 adds posterior abstractions, Phase 3 introduces trainable exact Gaussian-process models, Phase 4 adds reliable model-fitting orchestration, Phase 5 adds differentiable posterior samplers, and Phase 6 adds closed-form analytic acquisition functions.

## Intended architecture

The architecture separates models, posteriors, fitting, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

## Current capabilities

- Typed exception and warning hierarchies.
- Shape, dtype, device, and finite-value validation.
- Explicit seeded `torch.Generator` creation.
- Central tensor-shape and autograd conventions.
- Runtime-checkable posterior protocol.
- Dense batched Gaussian posterior with full covariance support.
- Differentiable reparameterized sampling with deterministic base samples.
- Posterior-list composition for independent output groups.
- Abstract PyTorch model contract.
- Exact single-task and fixed-noise Gaussian-process models.
- Independent model-list composition.
- Differentiable exact marginal-likelihood training objective.
- Typed fitting configuration and structured convergence diagnostics.
- Adam and L-BFGS fitting with state restoration, retries, and bounded jitter escalation.
- Recoverable fitting and numerical warnings.
- Deterministic fitting test mode that restores caller state.
- Stateful posterior sampler contract with reusable cached base samples.
- IID standard-normal posterior sampling with local seeded generators.
- Scrambled Sobol QMC normal posterior sampling.
- Analytic posterior mean, probability of improvement, expected improvement, and upper confidence bound.

## Current limitations

- There are no objective or candidate-optimization APIs.
- Analytic acquisition functions currently require `q=1` and one posterior output.
- Exact GP training scales cubically with observation count.
- Dense posterior covariance storage is not intended for large structured problems.
- Joint `ModelList` fitting is not implemented.
- Antithetic and specialized batch-collapsing samplers are not implemented.
- CUDA behavior is not yet exercised by dedicated CI jobs.
- Compatibility with other optimization libraries is not claimed.

## Foundation documentation

- [Tensor conventions](tensor_conventions.md)
- [Posterior abstractions](posteriors.md)
- [Model abstractions](models.md)
- [Model fitting](fitting.md)
- [Posterior sampling](sampling.md)
- [Analytic acquisition functions](acquisition.md)

## Next planned work

The next phase is acquisition optimization under box constraints, including Sobol initialization, restart selection, gradient-based local optimization, fixed features, failed-restart handling, and sequential and joint interfaces.
