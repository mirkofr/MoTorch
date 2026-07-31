"""Reusable tensor-shape contracts and validation helpers."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from motorch.exceptions import ShapeError


@dataclass(frozen=True)
class ShapeSpec:
    """Describe a tensor rank and the semantic names of its dimensions.

    Parameters
    ----------
    ndim:
        Required number of dimensions.
    dimension_names:
        Names ordered from the first to the last dimension.
    """

    ndim: int
    dimension_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ndim < 0:
            raise ValueError("ndim must be non-negative.")
        if len(self.dimension_names) != self.ndim:
            raise ValueError(
                "dimension_names must contain exactly ndim entries; "
                f"received ndim={self.ndim} and {len(self.dimension_names)} names."
            )

    @property
    def description(self) -> str:
        """Return a compact human-readable shape description."""
        return " x ".join(self.dimension_names) if self.dimension_names else "scalar"


def validate_shape(
    tensor: torch.Tensor,
    *,
    name: str,
    module: str,
    ndim: int | None = None,
    min_ndim: int | None = None,
    trailing_shape: Sequence[int | None] | None = None,
) -> torch.Tensor:
    """Validate tensor rank and optional trailing dimensions without copying it.

    ``None`` entries in ``trailing_shape`` act as wildcards. The original tensor
    is returned unchanged, preserving dtype, device, and autograd history.
    """
    if ndim is not None and min_ndim is not None:
        raise ValueError("Specify only one of ndim and min_ndim.")
    if ndim is not None and tensor.ndim != ndim:
        raise ShapeError(
            f"{module}: expected {name} to have {ndim} dimensions, "
            f"but received shape {tuple(tensor.shape)} ({tensor.ndim} dimensions)."
        )
    if min_ndim is not None and tensor.ndim < min_ndim:
        raise ShapeError(
            f"{module}: expected {name} to have at least {min_ndim} dimensions, "
            f"but received shape {tuple(tensor.shape)} ({tensor.ndim} dimensions)."
        )
    if trailing_shape is not None:
        expected = tuple(trailing_shape)
        if len(expected) > tensor.ndim:
            raise ShapeError(
                f"{module}: expected {name} to end with shape {expected}, "
                f"but received shape {tuple(tensor.shape)}."
            )
        received = tuple(tensor.shape[-len(expected) :]) if expected else ()
        matches = all(exp is None or exp == got for exp, got in zip(expected, received))
        if not matches:
            raise ShapeError(
                f"{module}: expected {name} to end with shape {expected}, "
                f"but received shape {tuple(tensor.shape)}."
            )
    return tensor


def validate_training_shapes(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    module: str = "validate_training_shapes",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate ``batch_shape x n x d`` and ``batch_shape x n x m`` data."""
    validate_shape(train_x, name="train_x", module=module, min_ndim=2)
    validate_shape(train_y, name="train_y", module=module, min_ndim=2)
    if train_x.shape[:-2] != train_y.shape[:-2]:
        raise ShapeError(
            f"{module}: expected train_x and train_y to share batch dimensions, "
            f"but received {tuple(train_x.shape[:-2])} and {tuple(train_y.shape[:-2])}."
        )
    if train_x.shape[-2] != train_y.shape[-2]:
        raise ShapeError(
            f"{module}: expected train_x and train_y to contain the same number "
            f"of observations n, but received {train_x.shape[-2]} and {train_y.shape[-2]}."
        )
    return train_x, train_y
