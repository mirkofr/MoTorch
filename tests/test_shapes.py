import pytest
import torch

from motorch.exceptions import ShapeError
from motorch.utils.shapes import ShapeSpec, validate_shape, validate_training_shapes


def test_shape_spec_describes_dimensions() -> None:
    spec = ShapeSpec(3, ("batch", "q", "d"))
    assert spec.description == "batch x q x d"


def test_validate_shape_returns_original_tensor() -> None:
    tensor = torch.zeros(2, 3, 4)
    result = validate_shape(
        tensor, name="candidates", module="test", ndim=3, trailing_shape=(3, 4)
    )
    assert result is tensor


def test_validate_shape_error_is_informative() -> None:
    tensor = torch.zeros(2, 3)
    with pytest.raises(ShapeError, match=r"test_module.*train_x.*3 dimensions.*\(2, 3\)"):
        validate_shape(tensor, name="train_x", module="test_module", ndim=3)


def test_validate_training_shapes_accepts_multiple_batch_dimensions() -> None:
    train_x = torch.zeros(2, 4, 7, 3)
    train_y = torch.zeros(2, 4, 7, 1)
    result_x, result_y = validate_training_shapes(train_x, train_y)
    assert result_x is train_x
    assert result_y is train_y


def test_validate_training_shapes_rejects_observation_mismatch() -> None:
    with pytest.raises(ShapeError, match="same number of observations"):
        validate_training_shapes(torch.zeros(5, 2), torch.zeros(4, 1))
