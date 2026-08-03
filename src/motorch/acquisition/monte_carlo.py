"""Differentiable Monte Carlo acquisition functions."""

from collections.abc import Callable, Sequence

import torch

from motorch.acquisition.base import AcquisitionFunction
from motorch.models import Model
from motorch.objectives import IdentityMCObjective, MCAcquisitionObjective
from motorch.sampling import PosteriorSampler

SampleConstraint = Callable[[torch.Tensor], torch.Tensor]


class MCAcquisitionFunction(AcquisitionFunction):
    """Base class for sampler-driven acquisition functions.

    Candidate inputs follow ``batch_shape x q x d``. Posterior samples follow
    ``sample_shape x batch_shape x q x m``, sampled objectives remove the final
    output dimension, and concrete acquisitions return ``batch_shape``.
    """

    _X_pending: torch.Tensor

    def __init__(
        self,
        model: Model,
        sampler: PosteriorSampler,
        *,
        objective: MCAcquisitionObjective | None = None,
    ) -> None:
        super().__init__(model)
        if not isinstance(sampler, PosteriorSampler):
            raise TypeError(
                "MCAcquisitionFunction: sampler must be a PosteriorSampler instance."
            )
        if len(sampler.sample_shape) == 0:
            raise ValueError(
                "MCAcquisitionFunction: sampler.sample_shape must contain at least "
                "one Monte Carlo sample dimension."
            )
        resolved_objective = objective or IdentityMCObjective()
        if not isinstance(resolved_objective, MCAcquisitionObjective):
            raise TypeError(
                "MCAcquisitionFunction: objective must be an "
                "MCAcquisitionObjective instance."
            )
        self.sampler = sampler
        self.objective = resolved_objective
        self.register_buffer("_X_pending", torch.empty(0), persistent=False)

    @property
    def X_pending(self) -> torch.Tensor | None:
        """Return fixed pending candidates included in joint sample utility."""
        if self._X_pending.numel() == 0:
            return None
        return self._X_pending

    def set_pending_points(self, X: torch.Tensor | None) -> None:
        """Set fixed pending candidates, or clear them with ``None``."""
        if X is None:
            self._X_pending = torch.empty(
                0,
                dtype=self._X_pending.dtype,
                device=self._X_pending.device,
            )
            return
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                "MCAcquisitionFunction.set_pending_points: X must be a torch.Tensor "
                "or None."
            )
        if X.ndim < 2 or X.shape[-2] < 1 or X.shape[-1] < 1:
            raise ValueError(
                "MCAcquisitionFunction.set_pending_points: X must have shape "
                "batch_shape x q_pending x d with positive q_pending and d."
            )
        if not X.is_floating_point():
            raise TypeError(
                "MCAcquisitionFunction.set_pending_points: X must use a "
                "floating-point dtype."
            )
        if not torch.isfinite(X).all():
            raise ValueError(
                "MCAcquisitionFunction.set_pending_points: X must contain only "
                "finite values."
            )
        self._X_pending = X.detach().clone()

    def _with_pending_points(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"{type(self).__name__}.forward: X must be a torch.Tensor.")
        if X.ndim < 2 or X.shape[-2] < 1 or X.shape[-1] < 1:
            raise ValueError(
                f"{type(self).__name__}.forward: X must have shape "
                "batch_shape x q x d with positive q and d."
            )
        if not X.is_floating_point():
            raise TypeError(
                f"{type(self).__name__}.forward: X must use a floating-point dtype."
            )
        if not torch.isfinite(X).all():
            raise ValueError(
                f"{type(self).__name__}.forward: X must contain only finite values."
            )
        pending = self.X_pending
        if pending is None:
            return X
        if pending.dtype != X.dtype or pending.device != X.device:
            raise ValueError(
                f"{type(self).__name__}.forward: pending points and X must share "
                "dtype and device."
            )
        if pending.shape[:-2] != X.shape[:-2] or pending.shape[-1] != X.shape[-1]:
            raise ValueError(
                f"{type(self).__name__}.forward: pending points must match X batch "
                "shape and feature dimension."
            )
        return torch.cat((X, pending), dim=-2)

    def _sample_objective(
        self,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_X = self._with_pending_points(X)
        posterior = self.model.posterior(joint_X)
        samples = self.sampler(posterior)
        if samples.ndim < len(self.sampler.sample_shape) + 2:
            raise RuntimeError(
                f"{type(self).__name__}: posterior samples have too few dimensions: "
                f"received {tuple(samples.shape)}."
            )
        if samples.shape[-2] != joint_X.shape[-2]:
            raise RuntimeError(
                f"{type(self).__name__}: posterior sample q dimension "
                f"{samples.shape[-2]} does not match candidate q={joint_X.shape[-2]}."
            )
        values = self.objective(samples, joint_X)
        expected = samples.shape[:-1]
        if values.shape != expected:
            raise ValueError(
                f"{type(self.objective).__name__}.forward: expected sampled objective "
                f"shape {tuple(expected)}, received {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise ValueError(
                f"{type(self).__name__}: sampled objective values must be finite."
            )
        return samples, values

    def _mean_over_samples(self, values: torch.Tensor) -> torch.Tensor:
        sample_dims = tuple(range(len(self.sampler.sample_shape)))
        return values.mean(dim=sample_dims)


class qExpectedImprovement(MCAcquisitionFunction):
    """Monte Carlo expected improvement over the best observed scalar utility."""

    best_f: torch.Tensor

    def __init__(
        self,
        model: Model,
        best_f: float | torch.Tensor,
        sampler: PosteriorSampler,
        *,
        objective: MCAcquisitionObjective | None = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model, sampler, objective=objective)
        best = torch.as_tensor(best_f)
        if best.numel() != 1 or not torch.isfinite(best).all():
            raise ValueError("qExpectedImprovement: best_f must be one finite scalar.")
        self.register_buffer("best_f", best.reshape(()))
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, values = self._sample_objective(X)
        best_f = self.best_f.to(dtype=values.dtype, device=values.device)
        improvement = values - best_f if self.maximize else best_f - values
        batch_improvement = improvement.clamp_min(0).amax(dim=-1)
        return self._mean_over_samples(batch_improvement)


class qConstrainedExpectedImprovement(qExpectedImprovement):
    """Expected improvement weighted by smooth sampled feasibility.

    Every constraint callable receives posterior samples and must return one value
    per sampled candidate with shape ``sample_shape x batch_shape x q``. Values
    less than or equal to zero represent feasibility. ``eta`` controls the sigmoid
    relaxation around the feasibility boundary.
    """

    eta: torch.Tensor

    def __init__(
        self,
        model: Model,
        best_f: float | torch.Tensor,
        sampler: PosteriorSampler,
        constraints: Sequence[SampleConstraint],
        *,
        objective: MCAcquisitionObjective | None = None,
        maximize: bool = True,
        eta: float | torch.Tensor = 1e-3,
    ) -> None:
        super().__init__(
            model,
            best_f,
            sampler,
            objective=objective,
            maximize=maximize,
        )
        if not isinstance(constraints, Sequence) or len(constraints) == 0:
            raise ValueError(
                "qConstrainedExpectedImprovement: constraints must be a non-empty "
                "sequence of callables."
            )
        if not all(callable(constraint) for constraint in constraints):
            raise TypeError(
                "qConstrainedExpectedImprovement: every constraint must be callable."
            )
        resolved_eta = torch.as_tensor(eta)
        if (
            resolved_eta.numel() != 1
            or not torch.isfinite(resolved_eta).all()
            or bool((resolved_eta <= 0).any())
        ):
            raise ValueError(
                "qConstrainedExpectedImprovement: eta must be one finite positive "
                "scalar."
            )
        self.constraints = tuple(constraints)
        self.register_buffer("eta", resolved_eta.reshape(()))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        samples, values = self._sample_objective(X)
        best_f = self.best_f.to(dtype=values.dtype, device=values.device)
        improvement = values - best_f if self.maximize else best_f - values
        feasibility = torch.ones_like(values)
        eta = self.eta.to(dtype=values.dtype, device=values.device)
        for index, constraint in enumerate(self.constraints):
            constraint_values = constraint(samples)
            if not isinstance(constraint_values, torch.Tensor):
                raise TypeError(
                    "qConstrainedExpectedImprovement: constraint "
                    f"{index} must return a torch.Tensor."
                )
            if constraint_values.shape != values.shape:
                raise ValueError(
                    "qConstrainedExpectedImprovement: constraint "
                    f"{index} must return shape {tuple(values.shape)}, received "
                    f"{tuple(constraint_values.shape)}."
                )
            if (
                constraint_values.dtype != values.dtype
                or constraint_values.device != values.device
            ):
                raise ValueError(
                    "qConstrainedExpectedImprovement: constraint outputs must share "
                    "sampled-objective dtype and device."
                )
            if not torch.isfinite(constraint_values).all():
                raise ValueError(
                    "qConstrainedExpectedImprovement: constraint outputs must be finite."
                )
            feasibility = feasibility * torch.sigmoid(-constraint_values / eta)
        constrained = improvement.clamp_min(0) * feasibility
        return self._mean_over_samples(constrained.amax(dim=-1))
