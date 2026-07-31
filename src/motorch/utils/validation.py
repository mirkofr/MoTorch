"""Dtype, device, and finite-value validation for tensors."""

from collections.abc import Mapping

import torch

from motorch.exceptions import DTypeError, DeviceError, TensorValidationError


def validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    module: str,
    floating: bool = True,
    finite: bool = True,
) -> torch.Tensor:
    """Validate one tensor without casting, moving, detaching, or copying it."""
    if floating and not tensor.is_floating_point():
        raise DTypeError(
            f"{module}: expected {name} to use a floating-point dtype, "
            f"but received {tensor.dtype}."
        )
    if finite and not bool(torch.isfinite(tensor).all()):
        raise TensorValidationError(
            f"{module}: expected {name} to contain only finite values, "
            f"but received at least one NaN or infinity."
        )
    return tensor


def validate_same_dtype_device(
    tensors: Mapping[str, torch.Tensor],
    *,
    module: str,
) -> None:
    """Require all named tensors to have exactly matching dtype and device."""
    items = list(tensors.items())
    if not items:
        raise ValueError("tensors must contain at least one named tensor.")
    reference_name, reference = items[0]
    for name, tensor in items[1:]:
        if tensor.dtype != reference.dtype:
            raise DTypeError(
                f"{module}: expected {name} to have dtype {reference.dtype} "
                f"to match {reference_name}, but received {tensor.dtype}."
            )
        if tensor.device != reference.device:
            raise DeviceError(
                f"{module}: expected {name} to be on device {reference.device} "
                f"to match {reference_name}, but received {tensor.device}."
            )
