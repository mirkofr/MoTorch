import torch

from motorch.fit import FittableExactGP
from motorch.models import SingleTaskGP


def test_single_task_gp_satisfies_fitting_contract() -> None:
    train_X = torch.linspace(0.0, 1.0, 4, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X)

    assert isinstance(SingleTaskGP(train_X, train_Y), FittableExactGP)
