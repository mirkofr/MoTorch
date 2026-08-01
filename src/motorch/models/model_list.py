"""Composition of independent probabilistic models."""

from collections.abc import Sequence

import torch
from torch import nn

from motorch.models.base import Model
from motorch.posteriors import PosteriorList


class ModelList(Model):
    """Combine independent models into one multi-output model.

    Every component receives the same candidate tensor. Component posterior
    outputs are concatenated in model order through :class:`PosteriorList`.
    """

    def __init__(self, *models: Model) -> None:
        super().__init__()
        if not models:
            raise ValueError("ModelList requires at least one model.")
        self.models = nn.ModuleList(models)

    @classmethod
    def from_sequence(cls, models: Sequence[Model]) -> "ModelList":
        """Construct a model list from a sequence."""
        return cls(*models)

    @property
    def num_outputs(self) -> int:
        """Return the sum of component output counts."""
        return sum(model.num_outputs for model in self.models)

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> PosteriorList:
        """Construct and combine every component posterior."""
        return PosteriorList.from_sequence(
            [
                model.posterior(X, observation_noise=observation_noise)
                for model in self.models
            ]
        )
