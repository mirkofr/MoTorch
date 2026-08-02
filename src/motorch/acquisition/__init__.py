"""Acquisition functions for Bayesian optimization."""

from motorch.acquisition.analytic import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from motorch.acquisition.base import AcquisitionFunction

__all__ = [
    "AcquisitionFunction",
    "ExpectedImprovement",
    "PosteriorMean",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
]
