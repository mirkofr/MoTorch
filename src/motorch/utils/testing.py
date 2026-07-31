"""Small tensor helpers used by MoTorch's numerical tests."""

import torch


def make_test_tensor(
    *shape: int,
    dtype: torch.dtype = torch.double,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a deterministic, finite tensor for shape and gradient tests."""
    if any(size < 0 for size in shape):
        raise ValueError(f"shape entries must be non-negative, but received {shape}.")
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(count, dtype=dtype, device=device)
    return values.reshape(shape).requires_grad_(requires_grad)


def assert_finite_gradients(tensor: torch.Tensor) -> None:
    """Raise ``AssertionError`` unless a tensor has finite populated gradients."""
    if tensor.grad is None:
        raise AssertionError("Expected tensor.grad to be populated, but it is None.")
    if not bool(torch.isfinite(tensor.grad).all()):
        raise AssertionError("Expected all gradients to be finite.")
