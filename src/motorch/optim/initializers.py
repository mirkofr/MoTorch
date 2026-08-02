"""Raw candidate generation and restart selection."""

import warnings
from collections.abc import Mapping

import torch

from motorch.warnings import OptimizationWarning


def validate_bounds(bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate ``2 x d`` finite box bounds and return lower and upper rows."""
    if not isinstance(bounds, torch.Tensor):
        raise TypeError("validate_bounds: bounds must be a torch.Tensor.")
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError(
            "validate_bounds: bounds must have shape 2 x d with d >= 1; "
            f"received {tuple(bounds.shape)}."
        )
    if not bounds.is_floating_point():
        raise TypeError("validate_bounds: bounds must use a floating-point dtype.")
    if not torch.isfinite(bounds).all():
        raise ValueError("validate_bounds: bounds must contain only finite values.")
    lower, upper = bounds[0], bounds[1]
    if not torch.all(lower < upper):
        raise ValueError(
            "validate_bounds: every lower bound must be strictly smaller than its upper bound."
        )
    return lower, upper


def validate_fixed_features(
    fixed_features: Mapping[int, float | torch.Tensor] | None,
    *,
    bounds: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Validate fixed feature indices and values against bounds."""
    if fixed_features is None:
        return {}
    lower, upper = validate_bounds(bounds)
    resolved: dict[int, torch.Tensor] = {}
    for index, value in fixed_features.items():
        if not isinstance(index, int):
            raise TypeError("fixed feature indices must be integers.")
        if index < 0 or index >= bounds.shape[-1]:
            raise ValueError(
                f"fixed feature index {index} is outside [0, {bounds.shape[-1] - 1}]."
            )
        tensor = torch.as_tensor(value, dtype=bounds.dtype, device=bounds.device)
        if tensor.numel() != 1 or not torch.isfinite(tensor).all():
            raise ValueError(f"fixed feature {index} must be one finite scalar.")
        scalar = tensor.reshape(())
        if bool(scalar < lower[index]) or bool(scalar > upper[index]):
            raise ValueError(
                f"fixed feature {index}={scalar.item()} lies outside its bounds "
                f"[{lower[index].item()}, {upper[index].item()}]."
            )
        resolved[index] = scalar
    return resolved


def apply_fixed_features(
    X: torch.Tensor,
    fixed_features: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Return ``X`` with selected final-dimension features replaced."""
    if not fixed_features:
        return X
    result = X.clone()
    for index, value in fixed_features.items():
        result[..., index] = value
    return result


def generate_raw_candidates(
    bounds: torch.Tensor,
    *,
    q: int,
    raw_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate Sobol candidates with shape ``raw_samples x q x d``."""
    lower, upper = validate_bounds(bounds)
    if q < 1:
        raise ValueError("generate_raw_candidates: q must be positive.")
    if raw_samples < 1:
        raise ValueError("generate_raw_candidates: raw_samples must be positive.")
    if seed is not None and seed < 0:
        raise ValueError("generate_raw_candidates: seed must be non-negative.")
    dimension = q * bounds.shape[-1]
    engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=dimension,
        scramble=True,
        seed=seed,
    )
    unit = engine.draw(raw_samples, dtype=bounds.dtype)
    unit = unit.to(device=bounds.device).reshape(raw_samples, q, bounds.shape[-1])
    return lower + (upper - lower) * unit


def select_restart_candidates(
    acq_function: torch.nn.Module,
    raw_candidates: torch.Tensor,
    *,
    num_restarts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the highest-valued finite raw candidates for local refinement."""
    if num_restarts < 1:
        raise ValueError("select_restart_candidates: num_restarts must be positive.")
    if raw_candidates.ndim < 3:
        raise ValueError(
            "select_restart_candidates: raw_candidates must have shape "
            "raw_samples x q x d."
        )
    with torch.no_grad():
        values = acq_function(raw_candidates)
    if values.shape != raw_candidates.shape[:-2]:
        raise ValueError(
            "select_restart_candidates: acquisition values must match candidate batch "
            f"shape {tuple(raw_candidates.shape[:-2])}; received {tuple(values.shape)}."
        )
    if values.ndim != 1:
        raise ValueError(
            "select_restart_candidates: restart selection requires one raw-sample "
            "dimension and no additional acquisition batch dimensions."
        )
    finite = torch.isfinite(values)
    if not finite.all():
        warnings.warn(
            "Non-finite acquisition values were excluded during restart selection.",
            OptimizationWarning,
            stacklevel=2,
        )
    finite_indices = finite.nonzero(as_tuple=False).squeeze(-1)
    if finite_indices.numel() == 0:
        raise RuntimeError(
            "select_restart_candidates: no finite acquisition values were available."
        )
    count = min(num_restarts, finite_indices.numel())
    finite_values = values[finite_indices]
    selected_local = torch.topk(finite_values, k=count).indices
    selected = finite_indices[selected_local]
    return raw_candidates[selected], values[selected]
