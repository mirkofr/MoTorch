# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. Phase 1 provides shared tensor validation and reproducible-randomness foundations. No probabilistic models or optimization algorithms are implemented.

## Intended architecture

The planned architecture separates models, posteriors, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

## Current capabilities

- Typed exception and warning hierarchies.
- Shape, dtype, device, and finite-value validation.
- Explicit seeded `torch.Generator` creation.
- Test helpers for deterministic tensors and finite gradients.
- Central tensor-shape and autograd conventions.

## Current limitations

- There are no model, posterior, sampler, objective, acquisition, or candidate-optimization APIs.
- CUDA behavior is not yet exercised by dedicated CI jobs.
- Compatibility with other optimization libraries is not claimed.

## Tensor foundations

Phase 1 tensor shape, dtype, device, gradient, and randomness conventions are documented in [Tensor conventions](tensor_conventions.md).

## Next planned work

The next phase is the posterior abstraction. It should begin only after the Phase 1 public utilities and conventions are reviewed.
