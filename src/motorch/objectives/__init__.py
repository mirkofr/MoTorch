"""Objectives for transforming posterior samples into utilities."""

from motorch.objectives.base import MCAcquisitionObjective
from motorch.objectives.monte_carlo import IdentityMCObjective

__all__ = ["IdentityMCObjective", "MCAcquisitionObjective"]
