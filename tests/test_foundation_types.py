from motorch.exceptions import (
    DTypeError,
    DeviceError,
    MoTorchError,
    ShapeError,
    TensorValidationError,
)
from motorch.warnings import MoTorchWarning, NumericalWarning, OptimizationWarning


def test_validation_errors_share_public_base_types() -> None:
    assert issubclass(TensorValidationError, MoTorchError)
    assert issubclass(TensorValidationError, ValueError)
    assert issubclass(ShapeError, TensorValidationError)
    assert issubclass(DTypeError, TensorValidationError)
    assert issubclass(DeviceError, TensorValidationError)


def test_warning_types_share_public_base_type() -> None:
    assert issubclass(NumericalWarning, MoTorchWarning)
    assert issubclass(OptimizationWarning, MoTorchWarning)
    assert issubclass(MoTorchWarning, UserWarning)
