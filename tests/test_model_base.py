import torch

from motorch.models import Model
from motorch.posteriors import GaussianPosterior


class ConstantModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> GaussianPosterior:
        del observation_noise
        mean = torch.zeros(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)
        event_size = X.shape[-2]
        covariance = torch.eye(
            event_size,
            dtype=X.dtype,
            device=X.device,
        ).expand(*X.shape[:-2], event_size, event_size)
        return GaussianPosterior(mean, covariance)


def test_model_forward_delegates_to_posterior() -> None:
    model = ConstantModel()
    X = torch.zeros(3, 2, dtype=torch.double)

    posterior = model(X)

    assert posterior.mean.shape == torch.Size([3, 1])
    assert model.num_outputs == 1
