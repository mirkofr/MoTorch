import pytest
import torch

from motorch.exceptions import DTypeError, TensorValidationError
from motorch.utils.validation import validate_same_dtype_device, validate_tensor


def test_validate_tensor_accepts_cpu_double() -> None:
    tensor = torch.ones(3, dtype=torch.double)
    assert validate_tensor(tensor, name="x", module="test") is tensor


def test_validate_tensor_rejects_integer_dtype() -> None:
    with pytest.raises(DTypeError, match="floating-point dtype"):
        validate_tensor(torch.ones(3, dtype=torch.long), name="x", module="test")


def test_validate_tensor_rejects_non_finite_values() -> None:
    with pytest.raises(TensorValidationError, match="finite values"):
        validate_tensor(torch.tensor([1.0, float("nan")]), name="x", module="test")


def test_validate_same_dtype_device_rejects_dtype_mismatch() -> None:
    tensors = {
        "x": torch.ones(2, dtype=torch.float32),
        "y": torch.ones(2, dtype=torch.float64),
    }
    with pytest.raises(DTypeError, match=r"y.*torch.float32.*x.*torch.float64"):
        validate_same_dtype_device(tensors, module="test")


def test_validate_same_dtype_device_rejects_device_mismatch() -> None:
    tensors = {
        "x": torch.ones(2, device="cpu"),
        "y": torch.empty(2, device="meta"),
    }
    from motorch.exceptions import DeviceError

    with pytest.raises(DeviceError, match=r"y.*cpu.*x.*meta"):
        validate_same_dtype_device(tensors, module="test")
