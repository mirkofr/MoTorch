"""Foundational tensor utilities for MoTorch."""

from motorch.utils.random import make_generator
from motorch.utils.shapes import ShapeSpec, validate_shape, validate_training_shapes
from motorch.utils.validation import validate_same_dtype_device, validate_tensor

__all__ = [
    "ShapeSpec",
    "make_generator",
    "validate_same_dtype_device",
    "validate_shape",
    "validate_tensor",
    "validate_training_shapes",
]
