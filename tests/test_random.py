import torch

from motorch.utils.random import make_generator


def test_make_generator_is_reproducible() -> None:
    first = torch.rand(5, generator=make_generator(17))
    second = torch.rand(5, generator=make_generator(17))
    assert torch.equal(first, second)


def test_make_generator_does_not_change_global_state() -> None:
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    _ = make_generator(99)
    actual = torch.rand(3)
    assert torch.equal(actual, expected)
