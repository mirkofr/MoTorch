"""Independent standard-normal posterior sampler."""

import torch

from motorch.sampling.base import PosteriorSampler


class IIDNormalSampler(PosteriorSampler):
    """Sample posteriors from cached IID standard-normal base samples."""

    def _construct_base_samples(
        self,
        shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
