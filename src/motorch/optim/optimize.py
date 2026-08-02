"""Bounded multistart acquisition-function optimization."""

import warnings
from collections.abc import Mapping
from typing import Protocol, cast

import torch

from motorch.acquisition import AcquisitionFunction
from motorch.optim.config import OptimizeAcqfOptions
from motorch.optim.diagnostics import OptimizationResult, RestartResult
from motorch.optim.initializers import (
    apply_fixed_features,
    generate_raw_candidates,
    select_restart_candidates,
    validate_bounds,
    validate_fixed_features,
)
from motorch.warnings import OptimizationWarning


class _PendingPointAcquisition(Protocol):
    def set_pending_points(self, X: torch.Tensor | None) -> None: ...


def _to_unconstrained(
    X: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    unit = ((X - lower) / (upper - lower)).clamp(epsilon, 1.0 - epsilon)
    return torch.logit(unit)


def _from_unconstrained(
    parameter: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    fixed_features: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    candidate = lower + (upper - lower) * parameter.sigmoid()
    return apply_fixed_features(candidate, fixed_features)


def _evaluate_scalar(
    acq_function: AcquisitionFunction,
    candidate: torch.Tensor,
) -> torch.Tensor:
    value = acq_function(candidate.unsqueeze(0))
    if value.numel() != 1:
        raise ValueError(
            "optimize_acqf: local restart evaluation must return one scalar value; "
            f"received shape {tuple(value.shape)}."
        )
    return value.reshape(())


def _optimize_restart(
    acq_function: AcquisitionFunction,
    initial: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    fixed_features: Mapping[int, torch.Tensor],
    options: OptimizeAcqfOptions,
    index: int,
) -> RestartResult:
    parameter = torch.nn.Parameter(
        _to_unconstrained(initial, lower, upper, options.clamp_epsilon)
    )
    if options.optimizer == "adam":
        optimizer: torch.optim.Adam | torch.optim.LBFGS = torch.optim.Adam(
            [parameter], lr=options.learning_rate
        )
    else:
        optimizer = torch.optim.LBFGS(
            [parameter],
            lr=options.learning_rate,
            max_iter=1,
            history_size=options.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

    best_candidate: torch.Tensor | None = None
    best_value: torch.Tensor | None = None
    stale_steps = 0
    previous_best: float | None = None

    try:
        for step in range(1, options.max_steps + 1):

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                candidate = _from_unconstrained(parameter, lower, upper, fixed_features)
                value = _evaluate_scalar(acq_function, candidate)
                if not torch.isfinite(value):
                    raise FloatingPointError("acquisition value became non-finite")
                loss = -value
                loss.backward()
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("candidate gradient became non-finite")
                return loss

            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()

            with torch.no_grad():
                candidate = _from_unconstrained(parameter, lower, upper, fixed_features)
                value = _evaluate_scalar(acq_function, candidate)
                if not torch.isfinite(value):
                    raise FloatingPointError("acquisition value became non-finite")
                if best_value is None or bool(value > best_value):
                    best_value = value.detach().clone()
                    best_candidate = candidate.detach().clone()

                current_best = float(best_value)
                improvement = (
                    float("inf")
                    if previous_best is None
                    else current_best - previous_best
                )
                stale_steps = stale_steps + 1 if improvement <= options.tolerance else 0
                previous_best = current_best
                grad_norm = (
                    float(parameter.grad.detach().abs().max())
                    if parameter.grad is not None
                    else float("inf")
                )
                if grad_norm <= options.gradient_tolerance:
                    return RestartResult(
                        index,
                        True,
                        best_candidate,
                        best_value,
                        step,
                        "gradient tolerance reached",
                    )
                if stale_steps >= options.patience:
                    return RestartResult(
                        index,
                        True,
                        best_candidate,
                        best_value,
                        step,
                        "patience tolerance reached",
                    )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        completed_steps = step if "step" in locals() else 0
        return RestartResult(index, False, None, None, completed_steps, str(error))

    if best_candidate is None or best_value is None:
        return RestartResult(
            index, False, None, None, options.max_steps, "no finite iterate"
        )
    return RestartResult(
        index,
        True,
        best_candidate,
        best_value,
        options.max_steps,
        "maximum steps reached",
    )


def _optimize_joint(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    *,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Mapping[int, float | torch.Tensor] | None,
    options: OptimizeAcqfOptions,
    seed: int | None,
) -> OptimizationResult:
    lower, upper = validate_bounds(bounds)
    resolved_fixed = validate_fixed_features(fixed_features, bounds=bounds)
    raw = generate_raw_candidates(bounds, q=q, raw_samples=raw_samples, seed=seed)
    raw = apply_fixed_features(raw, resolved_fixed)
    starts, _ = select_restart_candidates(acq_function, raw, num_restarts=num_restarts)
    results = tuple(
        _optimize_restart(
            acq_function,
            start,
            lower=lower,
            upper=upper,
            fixed_features=resolved_fixed,
            options=options,
            index=index,
        )
        for index, start in enumerate(starts)
    )
    successful = [
        result
        for result in results
        if result.success and result.candidate is not None and result.value is not None
    ]
    for result in results:
        if not result.success:
            warnings.warn(
                f"Acquisition optimization restart {result.index} failed: "
                f"{result.message}.",
                OptimizationWarning,
                stacklevel=2,
            )
    if not successful:
        raise RuntimeError("optimize_acqf: every local optimization restart failed.")
    best = max(successful, key=lambda item: float(cast(torch.Tensor, item.value)))
    return OptimizationResult(
        candidates=cast(torch.Tensor, best.candidate),
        values=cast(torch.Tensor, best.value),
        restarts=results,
    )


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    *,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Mapping[int, float | torch.Tensor] | None = None,
    options: OptimizeAcqfOptions | None = None,
    seed: int | None = None,
    sequential: bool = False,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | OptimizationResult:
    """Maximize an acquisition function under box constraints.

    Joint mode optimizes a ``q x d`` candidate tensor directly. Sequential mode
    requires an acquisition function exposing ``set_pending_points`` and optimizes
    one point at a time while updating the accumulated pending set.
    """
    if not isinstance(acq_function, AcquisitionFunction):
        raise TypeError(
            "optimize_acqf: acq_function must be an AcquisitionFunction instance."
        )
    if q < 1:
        raise ValueError("optimize_acqf: q must be positive.")
    if num_restarts < 1:
        raise ValueError("optimize_acqf: num_restarts must be positive.")
    if raw_samples < num_restarts:
        raise ValueError(
            "optimize_acqf: raw_samples must be greater than or equal to num_restarts."
        )
    if seed is not None and seed < 0:
        raise ValueError("optimize_acqf: seed must be non-negative.")
    resolved_options = options or OptimizeAcqfOptions()

    if not sequential or q == 1:
        result = _optimize_joint(
            acq_function,
            bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=resolved_options,
            seed=seed,
        )
    else:
        if not hasattr(acq_function, "set_pending_points"):
            raise TypeError(
                "optimize_acqf: sequential q>1 generation requires the acquisition "
                "function to implement set_pending_points(X | None)."
            )
        pending_acquisition = cast(_PendingPointAcquisition, acq_function)
        selected: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        diagnostics: list[RestartResult] = []
        try:
            for index in range(q):
                pending = torch.cat(selected, dim=-2) if selected else None
                pending_acquisition.set_pending_points(pending)
                one = _optimize_joint(
                    acq_function,
                    bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features=fixed_features,
                    options=resolved_options,
                    seed=None if seed is None else seed + index,
                )
                selected.append(one.candidates)
                values.append(one.values)
                diagnostics.extend(one.restarts)
        finally:
            pending_acquisition.set_pending_points(None)
        result = OptimizationResult(
            candidates=torch.cat(selected, dim=-2),
            values=torch.stack(values),
            restarts=tuple(diagnostics),
        )

    if return_diagnostics:
        return result
    return result.candidates, result.values
