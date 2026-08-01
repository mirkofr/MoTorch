import pytest
import torch
from torch import nn

from motorch.fit import FitOptions, FitTermination, fit_gp
from motorch.warnings import FittingWarning, NumericalWarning


class FlakyQuadratic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(2.0, dtype=torch.double))
        self.calls = 0

    def training_loss(self) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic factorization failure")
        return (self.value - 0.25).square()


class NonfiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0, dtype=torch.double))

    def training_loss(self) -> torch.Tensor:
        return self.value * torch.tensor(torch.nan, dtype=self.value.dtype)


class VectorLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.ones(2, dtype=torch.double))

    def training_loss(self) -> torch.Tensor:
        return self.value.square()


def test_fit_gp_retries_after_numerical_error_with_escalated_jitter() -> None:
    model = FlakyQuadratic()

    with pytest.warns(FittingWarning, match="retrying"):
        result = fit_gp(
            model,
            options=FitOptions(
                learning_rate=0.1,
                max_steps=80,
                max_retries=1,
                initial_jitter=1e-6,
                jitter_multiplier=10.0,
                warn_on_failure=False,
            ),
        )

    assert len(result.attempts) == 2
    assert result.attempts[0].termination is FitTermination.NUMERICAL_ERROR
    assert result.attempts[1].jitter == pytest.approx(1e-5)
    assert result.attempts[1].learning_rate == pytest.approx(0.05)
    assert result.best_loss < 0.1


def test_fit_gp_warns_with_numerical_diagnostic_after_exhausted_retries() -> None:
    model = NonfiniteModel()

    with pytest.warns(NumericalWarning, match="non-finite"):
        result = fit_gp(
            model,
            options=FitOptions(
                max_steps=2,
                max_retries=0,
                warn_on_retry=False,
            ),
        )

    assert not result.converged
    assert result.termination is FitTermination.NONFINITE_LOSS
    assert "non-finite loss" in result.message


def test_fit_gp_rejects_non_scalar_training_loss() -> None:
    with pytest.raises(ValueError, match="scalar tensor"):
        fit_gp(
            VectorLossModel(),
            options=FitOptions(max_retries=0, warn_on_failure=False),
        )


def test_fit_gp_rejects_object_without_fitting_contract() -> None:
    with pytest.raises(TypeError, match="training_loss"):
        fit_gp(object())  # type: ignore[arg-type]
