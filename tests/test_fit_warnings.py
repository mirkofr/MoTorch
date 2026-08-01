from motorch.warnings import FittingWarning, MoTorchWarning


def test_fitting_warning_inherits_motorch_warning() -> None:
    assert issubclass(FittingWarning, MoTorchWarning)
