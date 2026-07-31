"""Exception hierarchy for MoTorch public validation failures."""


class MoTorchError(Exception):
    """Base exception for MoTorch-specific errors."""


class TensorValidationError(MoTorchError, ValueError):
    """Base error raised when a tensor violates a public input contract."""


class ShapeError(TensorValidationError):
    """Raised when a tensor shape does not satisfy an expected contract."""


class DTypeError(TensorValidationError):
    """Raised when tensor dtypes are invalid or inconsistent."""


class DeviceError(TensorValidationError):
    """Raised when tensor devices are invalid or inconsistent."""
