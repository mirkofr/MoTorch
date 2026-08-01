# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. Phase 1 provides shared tensor foundations, and Phase 2 adds the first probabilistic posterior contract and implementations. No trainable models or optimization algorithms are implemented yet.

## Intended architecture

The architecture separates models, posteriors, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

## Current capabilities

- Typed exception and warning hierarchies.
- Shape, dtype, device, and finite-value validation.
- Explicit seeded `torch.Generator` creation.
- Test helpers for deterministic tensors and finite gradients.
- Central tensor-shape and autograd conventions.
- Runtime-checkable posterior protocol.
- Dense batched Gaussian posterior with full covariance support.
- Differentiable reparameterized sampling with deterministic base samples.
- Posterior-list composition for independent output groups.

## Current limitations

- There are no trainable model, sampler, objective, acquisition, or candidate-optimization APIs.
- Dense covariance storage is correctness-oriented and not intended for large structured problems.
- CUDA behavior is not yet exercised by dedicated CI jobs.
- Compatibility with other optimization libraries is not claimed.

## Foundation documentation

- [Tensor conventions](tensor_conventions.md)
- [Posterior abstractions](posteriors.md)

## Next planned work

The next phase is the model abstraction and first Gaussian-process models. It should begin only after the Phase 2 posterior contracts and acceptance tests are reviewed.
