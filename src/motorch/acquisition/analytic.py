"""Closed-form single-point acquisition functions."""

import math

import torch

from motorch.acquisition.base import AcquisitionFunction
from motorch.models import Model


def _standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-value / math.sqrt(2.0))


def _standard_normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def _analytic_moments(
    acquisition: AcquisitionFunction,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"{type(acquisition).__name__}.forward: X must be a torch.Tensor."
        )
    if X.ndim < 2:
        raise ValueError(
            f"{type(acquisition).__name__}.forward: X must have shape "
            "batch_shape x q x d."
        )
    if X.shape[-2] != 1:
        raise ValueError(
            f"{type(acquisition).__name__}.forward: analytic acquisition functions "
            f"require q=1, received q={X.shape[-2]}."
        )
    posterior = acquisition.model.posterior(X)
    mean = posterior.mean
    variance = posterior.variance
    if mean.shape != variance.shape:
        raise RuntimeError(
            f"{type(acquisition).__name__}: posterior mean and variance shapes differ: "
            f"{tuple(mean.shape)} and {tuple(variance.shape)}."
        )
    if mean.ndim < 2 or mean.shape[-2:] != torch.Size([1, 1]):
        raise ValueError(
            f"{type(acquisition).__name__}: analytic acquisition functions require "
            "a single-output posterior with q=1."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError(
            f"{type(acquisition).__name__}: posterior moments must be finite."
        )
    if (variance < 0).any():
        raise ValueError(
            f"{type(acquisition).__name__}: posterior variance must be non-negative."
        )
    return mean.squeeze(-1).squeeze(-1), variance.squeeze(-1).squeeze(-1)


class PosteriorMean(AcquisitionFunction):
    """Return posterior mean utility for a single candidate."""

    def __init__(self, model: Model, *, maximize: bool = True) -> None:
        super().__init__(model)
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, _ = _analytic_moments(self, X)
        return mean if self.maximize else -mean


class ProbabilityOfImprovement(AcquisitionFunction):
    """Closed-form probability of improvement for a Gaussian posterior."""

    def __init__(
        self,
        model: Model,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> None:
        super().__init__(model)
        best = torch.as_tensor(best_f)
        if best.numel() != 1 or not torch.isfinite(best).all():
            raise ValueError("ProbabilityOfImprovement: best_f must be one finite scalar.")
        self.register_buffer("best_f", best.reshape(()))
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, variance = _analytic_moments(self, X)
        best_f = self.best_f.to(dtype=mean.dtype, device=mean.device)
        improvement = mean - best_f if self.maximize else best_f - mean
        sigma = variance.sqrt()
        positive_sigma = sigma > 0
        safe_sigma = torch.where(positive_sigma, sigma, torch.ones_like(sigma))
        probability = _standard_normal_cdf(improvement / safe_sigma)
        deterministic = (improvement > 0).to(dtype=mean.dtype)
        return torch.where(positive_sigma, probability, deterministic)


class ExpectedImprovement(AcquisitionFunction):
    """Closed-form expected improvement for a Gaussian posterior."""

    def __init__(
        self,
        model: Model,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> None:
        super().__init__(model)
        best = torch.as_tensor(best_f)
        if best.numel() != 1 or not torch.isfinite(best).all():
            raise ValueError("ExpectedImprovement: best_f must be one finite scalar.")
        self.register_buffer("best_f", best.reshape(()))
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, variance = _analytic_moments(self, X)
        best_f = self.best_f.to(dtype=mean.dtype, device=mean.device)
        improvement = mean - best_f if self.maximize else best_f - mean
        sigma = variance.sqrt()
        positive_sigma = sigma > 0
        safe_sigma = torch.where(positive_sigma, sigma, torch.ones_like(sigma))
        standardized = improvement / safe_sigma
        expected = improvement * _standard_normal_cdf(standardized)
        expected = expected + sigma * _standard_normal_pdf(standardized)
        deterministic = improvement.clamp_min(0)
        return torch.where(positive_sigma, expected, deterministic)


class UpperConfidenceBound(AcquisitionFunction):
    """Gaussian upper confidence bound with non-negative exploration weight."""

    def __init__(
        self,
        model: Model,
        beta: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> None:
        super().__init__(model)
        resolved_beta = torch.as_tensor(beta)
        if (
            resolved_beta.numel() != 1
            or not torch.isfinite(resolved_beta).all()
            or bool((resolved_beta < 0).any())
        ):
            raise ValueError("UpperConfidenceBound: beta must be one finite non-negative scalar.")
        self.register_buffer("beta", resolved_beta.reshape(()))
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, variance = _analytic_moments(self, X)
        beta = self.beta.to(dtype=mean.dtype, device=mean.device)
        signed_mean = mean if self.maximize else -mean
        return signed_mean + beta.sqrt() * variance.sqrt()
