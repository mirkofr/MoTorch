"""Explicit and reproducible PyTorch random-number utilities."""

import torch


def make_generator(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Create an explicitly seeded generator for the requested device.

    This function does not mutate PyTorch's global random state.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, but received {seed}.")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
