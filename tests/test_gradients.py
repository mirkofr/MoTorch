from motorch.utils.shapes import validate_shape
from motorch.utils.testing import assert_finite_gradients, make_test_tensor
from motorch.utils.validation import validate_tensor


def test_validation_preserves_gradient_history() -> None:
    tensor = make_test_tensor(2, 3, requires_grad=True)
    validated = validate_shape(tensor, name="x", module="test", ndim=2)
    validated = validate_tensor(validated, name="x", module="test")
    assert validated is tensor
    assert validated.requires_grad
    validated.square().sum().backward()
    assert_finite_gradients(tensor)
