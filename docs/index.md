# MoTorch documentation

## Mission

MoTorch aims to provide low-level, modular, tensor-native building blocks for Bayesian optimization and scientific optimization research using PyTorch conventions.

## Status

MoTorch is pre-alpha. The current repository contains only a reproducible Python package foundation. No probabilistic models or optimization algorithms are implemented.

## Intended architecture

The planned architecture separates models, posteriors, samplers, objectives, acquisition functions, and candidate optimization so that each mathematical layer can be tested and replaced independently.

## Current limitations

- The package currently exposes only `motorch.__version__`.
- There are no runtime numerical dependencies.
- There are no public optimization APIs or executable optimization examples.
- Compatibility with other optimization libraries is not claimed.

## Next foundation work

The next planned task is to define tensor-shape, dtype, device, validation, warning, exception, and reproducible-randomness conventions before probabilistic abstractions are introduced.
