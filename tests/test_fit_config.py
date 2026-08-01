import pytest

from motorch.fit import FitOptions


def test_fit_options_defaults_are_valid() -> None:
    options = FitOptions()

    assert options.optimizer == "adam"
    assert options.max_steps > 0
    assert options.max_retries >= 0
    assert options.initial_jitter <= options.max_jitter


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("learning_rate", 0.0, "learning_rate"),
        ("max_steps", 0, "max_steps"),
        ("patience", 0, "patience"),
        ("max_retries", -1, "max_retries"),
        ("seed", -1, "seed"),
        ("retry_learning_rate_factor", 1.1, "retry_learning_rate_factor"),
        ("jitter_multiplier", 1.0, "jitter_multiplier"),
        ("max_jitter", 1e-8, "max_jitter"),
    ],
)
def test_fit_options_reject_invalid_values(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FitOptions(**{keyword: value})  # type: ignore[arg-type]
