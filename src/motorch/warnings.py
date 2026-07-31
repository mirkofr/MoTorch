"""Warning hierarchy for recoverable MoTorch conditions."""


class MoTorchWarning(UserWarning):
    """Base warning for recoverable MoTorch conditions."""


class NumericalWarning(MoTorchWarning):
    """Warn about recoverable numerical instability or degraded behavior."""


class OptimizationWarning(MoTorchWarning):
    """Warn about recoverable candidate-optimization failures."""
