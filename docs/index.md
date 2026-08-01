# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. Phase 1 provides shared tensor foundations, Phase 2 adds posterior abstractions, and Phase 3 introduces the first trainable exact Gaussian-process models. General fitting orchestration and optimization algorithms are not implemented yet.

## Intended architecture

The architecture separates models, posteriors, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

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

## Current limitations

- There are no fitting utility, sampler, objective, acquisition, or candidate-optimization APIs.
- Exact GP training scales cubically with observation count.
- Dense posterior covariance storage is not intended for large structured problems.
- CUDA behavior is not yet exercised by dedicated CI jobs.
- Compatibility with other optimization libraries is not claimed.

## Foundation documentation

- [Tensor conventions](tensor_conventions.md)
- [Posterior abstractions](posteriors.md)
- [Model abstractions](models.md)

## Next planned work

The next phase is reliable model-fitting orchestration, including optimizer configuration, convergence diagnostics, retry behavior, jitter handling, and fitting warnings.
