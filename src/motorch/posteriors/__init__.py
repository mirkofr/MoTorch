"""Posterior contracts and tensor-native implementations."""

from motorch.posteriors.base import Posterior
from motorch.posteriors.gaussian import GaussianPosterior
from motorch.posteriors.posterior_list import PosteriorList

__all__ = ["GaussianPosterior", "Posterior", "PosteriorList"]
