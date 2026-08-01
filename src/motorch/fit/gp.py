"""Reliable marginal-likelihood fitting for exact Gaussian-process models."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast, runtime_checkable

import torch
from gpytorch.settings import cholesky_jitter
from torch import nn

from motorch.fit.config import FitOptions
from motorch.fit.diagnostics import FitAttempt, FitResult, FitTermination
from motorch.warnings import FittingWarning, NumericalWarning


@runtime_checkable
class FittableExactGP(Protocol):
    """Structural contract required by :func:`fit_gp`."""

    def training_loss(self) -> torch.Tensor:
        """Return a differentiable scalar negative marginal log likelihood."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return trainable parameters."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return serializable model state."""

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> object:
        """Restore model state."""

    def train(self, mode: bool = True) -> object:
        """Set training mode."""


@contextmanager
def _deterministic_context(enabled: bool, seed: int) -> Iterator[None]:
    """Temporarily configure deterministic execution and restore caller state."""
    if not enabled:
        yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(
            deterministic_before,
            warn_only=warn_only_before,
        )


def _trainable_parameters(model: FittableExactGP) -> list[nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("fit_gp: model has no trainable parameters.")
    return parameters


def _maximum_gradient(parameters: list[nn.Parameter]) -> float:
    values = [
        float(parameter.grad.detach().abs().max().item())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return max(values, default=0.0)


def _make_optimizer(
    parameters: list[nn.Parameter],
    options: FitOptions,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if options.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    return torch.optim.LBFGS(
        parameters,
        lr=learning_rate,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )


def _evaluate_loss(model: FittableExactGP) -> torch.Tensor:
    loss = model.training_loss()
    if loss.ndim != 0:
        raise ValueError(
            "fit_gp: training_loss() must return a scalar tensor, "
            f"but received shape {tuple(loss.shape)}."
        )
    return loss


def _attempt_fit(
    model: FittableExactGP,
    *,
    attempt_index: int,
    learning_rate: float,
    jitter: float,
    options: FitOptions,
) -> tuple[FitAttempt, dict[str, torch.Tensor]]:
    parameters = _trainable_parameters(model)
    optimizer = _make_optimizer(parameters, options, learning_rate)
    model.train(True)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    initial_loss = math.inf
    final_loss = math.inf
    max_gradient = math.inf
    stable_steps = 0
    previous_loss: float | None = None
    termination = FitTermination.MAX_STEPS
    message = f"Reached max_steps={options.max_steps} without convergence."
    completed_steps = 0

    with cholesky_jitter(
        float_value=jitter,
        double_value=jitter,
        half_value=jitter,
    ):
        for step in range(1, options.max_steps + 1):
            completed_steps = step
            optimizer.zero_grad(set_to_none=True)

            if options.optimizer == "lbfgs":

                def closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    closure_loss = _evaluate_loss(model)
                    closure_loss.backward()
                    return closure_loss

                loss = cast(torch.Tensor, optimizer.step(closure))
                optimizer.zero_grad(set_to_none=True)
                loss = _evaluate_loss(model)
                loss.backward()
            else:
                loss = _evaluate_loss(model)
                loss.backward()

            loss_value = float(loss.detach().item())
            if step == 1:
                initial_loss = loss_value
            if not math.isfinite(loss_value):
                termination = FitTermination.NONFINITE_LOSS
                message = (
                    f"Attempt {attempt_index} produced a non-finite loss at step {step}; "
                    "retry with larger jitter or rescale the training data."
                )
                break

            max_gradient = _maximum_gradient(parameters)
            if not math.isfinite(max_gradient):
                termination = FitTermination.NONFINITE_GRADIENT
                message = (
                    f"Attempt {attempt_index} produced non-finite gradients at step {step}; "
                    "retry with larger jitter or inspect model initialization."
                )
                break

            if loss_value < best_loss:
                best_loss = loss_value
                best_state = copy.deepcopy(model.state_dict())

            if max_gradient <= options.tolerance_grad:
                termination = FitTermination.GRADIENT_TOLERANCE
                message = (
                    f"Converged at step {step}: maximum gradient {max_gradient:.3e} "
                    f"is at or below tolerance {options.tolerance_grad:.3e}."
                )
                final_loss = loss_value
                break

            if previous_loss is not None:
                threshold = options.tolerance_loss * max(1.0, abs(previous_loss))
                stable_steps = (
                    stable_steps + 1
                    if abs(previous_loss - loss_value) <= threshold
                    else 0
                )
                if stable_steps >= options.patience:
                    termination = FitTermination.LOSS_TOLERANCE
                    message = (
                        f"Converged at step {step}: loss changed by at most the configured "
                        f"relative tolerance for {options.patience} consecutive steps."
                    )
                    final_loss = loss_value
                    break
            previous_loss = loss_value

            if options.optimizer == "adam":
                optimizer.step()
            final_loss = loss_value
        else:
            final_loss = float(_evaluate_loss(model).detach().item())
            if math.isfinite(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best_state = copy.deepcopy(model.state_dict())

    converged = termination in {
        FitTermination.GRADIENT_TOLERANCE,
        FitTermination.LOSS_TOLERANCE,
    }
    attempt = FitAttempt(
        attempt=attempt_index,
        optimizer=options.optimizer,
        learning_rate=learning_rate,
        jitter=jitter,
        steps=completed_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        max_gradient=max_gradient,
        converged=converged,
        termination=termination,
        message=message,
    )
    return attempt, best_state


def fit_gp(
    model: FittableExactGP,
    *,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit an exact GP by minimizing its negative marginal log likelihood.

    The function mutates ``model`` in place and returns structured diagnostics.
    Failed attempts restore the best finite state before retrying. Jitter and
    learning-rate changes are explicit in each :class:`FitAttempt`.
    """
    if not isinstance(model, FittableExactGP):
        raise TypeError(
            "fit_gp: model must provide training_loss(), parameters(), state_dict(), "
            "load_state_dict(), and train()."
        )
    resolved = FitOptions() if options is None else options
    attempts: list[FitAttempt] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf

    with _deterministic_context(resolved.deterministic, resolved.seed):
        for attempt_index in range(resolved.max_retries + 1):
            learning_rate = resolved.learning_rate * (
                resolved.retry_learning_rate_factor**attempt_index
            )
            jitter = min(
                resolved.initial_jitter * (resolved.jitter_multiplier**attempt_index),
                resolved.max_jitter,
            )
            try:
                attempt, attempt_state = _attempt_fit(
                    model,
                    attempt_index=attempt_index,
                    learning_rate=learning_rate,
                    jitter=jitter,
                    options=resolved,
                )
            except (RuntimeError, ArithmeticError) as error:
                attempt = FitAttempt(
                    attempt=attempt_index,
                    optimizer=resolved.optimizer,
                    learning_rate=learning_rate,
                    jitter=jitter,
                    steps=0,
                    initial_loss=math.inf,
                    final_loss=math.inf,
                    best_loss=math.inf,
                    max_gradient=math.inf,
                    converged=False,
                    termination=FitTermination.NUMERICAL_ERROR,
                    message=(
                        f"Attempt {attempt_index} failed with {type(error).__name__}: {error}. "
                        "The next retry will use larger jitter and a lower learning rate."
                    ),
                )
                attempt_state = best_state

            attempts.append(attempt)
            if attempt.best_loss < best_loss:
                best_loss = attempt.best_loss
                best_state = attempt_state
            model.load_state_dict(copy.deepcopy(best_state))

            if attempt.converged:
                break
            if attempt_index < resolved.max_retries and resolved.warn_on_retry:
                warnings.warn(
                    f"fit_gp retrying after attempt {attempt_index}: {attempt.message}",
                    FittingWarning,
                    stacklevel=2,
                )

    final_attempt = attempts[-1]
    converged = final_attempt.converged
    if not converged and resolved.warn_on_failure:
        warning_type = (
            NumericalWarning
            if final_attempt.termination
            in {
                FitTermination.NONFINITE_LOSS,
                FitTermination.NONFINITE_GRADIENT,
                FitTermination.NUMERICAL_ERROR,
            }
            else FittingWarning
        )
        warnings.warn(
            f"fit_gp did not converge after {len(attempts)} attempt(s): "
            f"{final_attempt.message}",
            warning_type,
            stacklevel=2,
        )

    return FitResult(
        converged=converged,
        termination=final_attempt.termination,
        initial_loss=attempts[0].initial_loss,
        final_loss=final_attempt.final_loss,
        best_loss=best_loss,
        total_steps=sum(attempt.steps for attempt in attempts),
        attempts=tuple(attempts),
    )
