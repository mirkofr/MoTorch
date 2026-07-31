# Tensor conventions

MoTorch uses PyTorch tensors and preserves user-provided dtype, device, and autograd history. Public validation functions do not silently cast tensors, move them between devices, or detach them from their computation graph.

## Standard shapes

| Quantity | Convention | Meaning |
| --- | --- | --- |
| Training inputs | `batch_shape × n × d` | `n` observations with `d` input features |
| Training outcomes | `batch_shape × n × m` | `m` outcomes for the same `n` observations |
| Candidate inputs | `batch_shape × q × d` | `q` jointly evaluated candidate points |
| Posterior mean | `posterior_batch_shape × q × m` | Mean for each candidate and output |
| Posterior samples | `sample_shape × posterior_batch_shape × q × m` | Reparameterized samples |
| Acquisition values | `batch_shape` | One utility value per acquisition batch |

The foundation utilities accept arbitrary leading batch dimensions. Validation occurs at public boundaries and error messages identify the module, tensor name, expected contract, and received shape.

## Dtype and device

Scientific examples and numerical tests default to `torch.double`. APIs do not silently cast dtype or move tensors. Operations combining tensors should call `validate_same_dtype_device` before computation.

## Gradients

Validation functions return the original tensor object. They do not clone, detach, or use in-place mutation, so existing autograd graphs and `requires_grad` state are preserved.

## Randomness

Stochastic APIs should accept an explicit `torch.Generator`, seed, or base samples. `make_generator` creates a seeded device-specific generator without changing global PyTorch random state.
