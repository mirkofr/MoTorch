import pytest
import torch

from motorch.utils.shapes import ShapeSpec, validate_shape
from motorch.utils.testing import assert_finite_gradients, make_test_tensor
from motorch.utils.validation import validate_same_dtype_device


def test_shape_spec_describes_named_dimensions() -> None:
    spec = ShapeSpec(ndim=3, dimension_names=("batch", "n", "d"))

    assert spec.description == "batch x n x d"


def test_shape_spec_rejects_inconsistent_dimension_names() -> None:
    with pytest.raises(ValueError, match="exactly ndim entries"):
        ShapeSpec(ndim=2, dimension_names=("n",))


def test_validate_shape_accepts_wildcard_trailing_dimension() -> None:
    tensor = torch.zeros(2, 5, 3, dtype=torch.double)

    validated = validate_shape(
        tensor,
        name="candidates",
        module="test",
        trailing_shape=(None, 3),
    )

    assert validated is tensor


def test_validate_shape_rejects_conflicting_rank_contracts() -> None:
    tensor = torch.zeros(2, 3, dtype=torch.double)

    with pytest.raises(ValueError, match="Specify only one"):
        validate_shape(tensor, name="x", module="test", ndim=2, min_ndim=1)


def test_make_test_tensor_defaults_are_deterministic_cpu_double() -> None:
    first = make_test_tensor(2, 3)
    second = make_test_tensor(2, 3)

    assert first.dtype == torch.double
    assert first.device.type == "cpu"
    assert torch.equal(first, second)


def test_make_test_tensor_rejects_negative_shape_entries() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_test_tensor(2, -1)


def test_assert_finite_gradients_rejects_missing_gradient() -> None:
    tensor = make_test_tensor(2, requires_grad=True)

    with pytest.raises(AssertionError, match="populated"):
        assert_finite_gradients(tensor)


def test_validate_same_dtype_device_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_same_dtype_device({}, module="test")
