"""Sobol quasi-Monte Carlo standard-normal posterior sampler."""

import math

import torch

from motorch.sampling.base import PosteriorSampler


class SobolQMCNormalSampler(PosteriorSampler):
    """Sample posteriors from scrambled Sobol standard-normal base samples.

    Sobol points are generated on CPU by PyTorch, transformed through the
    inverse standard-normal CDF, and then moved to the posterior device without
    changing dtype.
    """

    def _construct_base_samples(
        self,
        shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        sample_count = math.prod(self.sample_shape) if self.sample_shape else 1
        dimension = math.prod(shape[len(self.sample_shape) :])
        engine = torch.quasirandom.SobolEngine(
            dimension=dimension,
            scramble=True,
            seed=self.seed,
        )
        uniforms = engine.draw(sample_count, dtype=dtype)
        epsilon = torch.finfo(dtype).eps
        uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
        normals = torch.special.ndtri(uniforms)
        return normals.reshape(shape).to(device=device)
