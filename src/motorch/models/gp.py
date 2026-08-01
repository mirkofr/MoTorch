"""Exact Gaussian-process model implementations."""

from abc import abstractmethod
from typing import cast

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from motorch.exceptions import DTypeError, ShapeError, TensorValidationError
from motorch.models.base import Model
from motorch.posteriors import GaussianPosterior
from motorch.utils import (
    validate_same_dtype_device,
    validate_shape,
    validate_tensor,
    validate_training_shapes,
)

Likelihood = GaussianLikelihood | FixedNoiseGaussianLikelihood


class _ExactGPBackend(ExactGP):
    """Internal batched exact GP with a constant mean and ARD RBF kernel."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        likelihood: Likelihood,
    ) -> None:
        super().__init__(train_X, train_Y, likelihood)
        batch_shape = torch.Size(train_X.shape[:-2])
        input_dim = train_X.shape[-1]
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=input_dim, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, X: torch.Tensor) -> MultivariateNormal:
        """Return the latent multivariate normal at ``X``."""
        mean = self.mean_module(X)
        covariance = self.covar_module(X)
        return MultivariateNormal(mean, covariance)


class _BaseExactGP(Model):
    """Shared exact-GP wrapper producing MoTorch Gaussian posteriors."""

    _SUPPORTED_DTYPES = {torch.float32, torch.float64}

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        *,
        noise: torch.Tensor | None,
    ) -> None:
        super().__init__()
        module = type(self).__name__
        train_X, train_Y = self._validate_training_data(train_X, train_Y, module=module)
        if noise is not None:
            noise = self._validate_noise(noise, train_Y, module=module)

        self._num_outputs = train_Y.shape[-1]
        self._input_dim = train_X.shape[-1]
        self._training_batch_shape = torch.Size(train_X.shape[:-2])

        backend_train_X = self._expand_inputs_for_outputs(train_X, self._num_outputs)
        backend_train_Y = train_Y.movedim(-1, -2)
        backend_noise = None if noise is None else noise.movedim(-1, -2)
        likelihood = self._make_likelihood(
            backend_train_X,
            backend_train_Y,
            backend_noise,
        )
        self._gp = _ExactGPBackend(backend_train_X, backend_train_Y, likelihood)

    @property
    def num_outputs(self) -> int:
        """Return the number of independent modeled outputs."""
        return self._num_outputs

    @property
    def input_dim(self) -> int:
        """Return the input feature dimension ``d``."""
        return self._input_dim

    @property
    def batch_shape(self) -> torch.Size:
        """Return the training-data batch shape."""
        return self._training_batch_shape

    @property
    def likelihood(self) -> Likelihood:
        """Return the GPyTorch likelihood used by the exact GP."""
        return cast(Likelihood, self._gp.likelihood)

    @property
    def train_X(self) -> torch.Tensor:
        """Return training inputs with shape ``batch_shape x n x d``."""
        train_inputs = cast(tuple[torch.Tensor, ...], self._gp.train_inputs)
        return train_inputs[0].select(dim=-3, index=0)

    @property
    def train_Y(self) -> torch.Tensor:
        """Return training outcomes with shape ``batch_shape x n x m``."""
        targets = cast(torch.Tensor, self._gp.train_targets)
        return targets.movedim(-2, -1)

    def training_loss(self) -> torch.Tensor:
        """Return the summed negative exact marginal log likelihood.

        This differentiable scalar is suitable for direct optimization with a
        PyTorch optimizer. General fitting orchestration is intentionally left
        to MoTorch's later fitting phase.
        """
        self.train()
        train_inputs = cast(tuple[torch.Tensor, ...], self._gp.train_inputs)
        train_targets = cast(torch.Tensor, self._gp.train_targets)
        output = self._gp(train_inputs[0])
        marginal_log_likelihood = ExactMarginalLogLikelihood(
            self.likelihood,
            self._gp,
        )
        loss = -marginal_log_likelihood(output, train_targets)
        return cast(torch.Tensor, loss.sum())

    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
    ) -> GaussianPosterior:
        """Construct a Gaussian posterior at ``batch_shape x q x d`` inputs."""
        module = f"{type(self).__name__}.posterior"
        self._validate_candidates(X, module=module)
        if observation_noise and isinstance(
            self.likelihood,
            FixedNoiseGaussianLikelihood,
        ):
            raise ValueError(
                f"{module}: observation_noise=True is undefined for FixedNoiseGP "
                "at new points because no candidate-specific noise was supplied."
            )

        posterior_batch_shape = self._posterior_batch_shape(X, module=module)
        expanded_X = X.expand(*posterior_batch_shape, *X.shape[-2:])
        backend_X = self._expand_inputs_for_outputs(expanded_X, self.num_outputs)

        was_training = self.training
        self.eval()
        try:
            latent = self._gp(backend_X)
            predictive = self.likelihood(latent) if observation_noise else latent
            mean = predictive.mean.movedim(-2, -1)
            covariance_by_output = predictive.covariance_matrix
        finally:
            self.train(was_training)

        covariance = self._independent_output_covariance(covariance_by_output)
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        return GaussianPosterior(mean, covariance)

    @abstractmethod
    def _make_likelihood(
        self,
        backend_train_X: torch.Tensor,
        backend_train_Y: torch.Tensor,
        backend_noise: torch.Tensor | None,
    ) -> Likelihood:
        """Construct the model likelihood."""

    @classmethod
    def _validate_training_data(
        cls,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        *,
        module: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_tensor(train_X, name="train_X", module=module)
        validate_tensor(train_Y, name="train_Y", module=module)
        validate_same_dtype_device(
            {"train_X": train_X, "train_Y": train_Y},
            module=module,
        )
        validate_training_shapes(train_X, train_Y, module=module)
        if train_X.dtype not in cls._SUPPORTED_DTYPES:
            raise DTypeError(
                f"{module}: expected train_X and train_Y to use torch.float32 "
                f"or torch.float64, but received {train_X.dtype}."
            )
        if train_X.shape[-2] < 1 or train_X.shape[-1] < 1:
            raise ShapeError(
                f"{module}: expected n and d to be positive, but received "
                f"train_X shape {tuple(train_X.shape)}."
            )
        if train_Y.shape[-1] < 1:
            raise ShapeError(
                f"{module}: expected m to be positive, but received "
                f"train_Y shape {tuple(train_Y.shape)}."
            )
        return train_X, train_Y

    @staticmethod
    def _validate_noise(
        noise: torch.Tensor,
        train_Y: torch.Tensor,
        *,
        module: str,
    ) -> torch.Tensor:
        validate_tensor(noise, name="train_Yvar", module=module)
        validate_same_dtype_device(
            {"train_Y": train_Y, "train_Yvar": noise},
            module=module,
        )
        if noise.shape != train_Y.shape:
            raise ShapeError(
                f"{module}: expected train_Yvar to match train_Y shape "
                f"{tuple(train_Y.shape)}, but received {tuple(noise.shape)}."
            )
        if bool((noise <= 0).any()):
            raise TensorValidationError(
                f"{module}: expected train_Yvar to contain strictly positive "
                "observation variances."
            )
        return noise

    def _validate_candidates(self, X: torch.Tensor, *, module: str) -> None:
        validate_tensor(X, name="X", module=module)
        validate_shape(X, name="X", module=module, min_ndim=2)
        validate_same_dtype_device(
            {"train_X": self.train_X, "X": X},
            module=module,
        )
        if X.shape[-2] < 1:
            raise ShapeError(
                f"{module}: expected q to be positive, but received "
                f"X shape {tuple(X.shape)}."
            )
        if X.shape[-1] != self.input_dim:
            raise ShapeError(
                f"{module}: expected X to have input dimension d={self.input_dim}, "
                f"but received shape {tuple(X.shape)}."
            )

    def _posterior_batch_shape(
        self,
        X: torch.Tensor,
        *,
        module: str,
    ) -> torch.Size:
        try:
            return torch.broadcast_shapes(self.batch_shape, X.shape[:-2])
        except RuntimeError as error:
            raise ShapeError(
                f"{module}: expected candidate batch shape {tuple(X.shape[:-2])} "
                f"to broadcast with model batch shape {tuple(self.batch_shape)}."
            ) from error

    @staticmethod
    def _expand_inputs_for_outputs(
        X: torch.Tensor,
        num_outputs: int,
    ) -> torch.Tensor:
        return X.unsqueeze(-3).expand(
            *X.shape[:-2],
            num_outputs,
            *X.shape[-2:],
        )

    @staticmethod
    def _independent_output_covariance(
        covariance_by_output: torch.Tensor,
    ) -> torch.Tensor:
        num_outputs = covariance_by_output.shape[-3]
        q = covariance_by_output.shape[-1]
        identity = torch.eye(
            num_outputs,
            dtype=covariance_by_output.dtype,
            device=covariance_by_output.device,
        )
        covariance = torch.einsum(
            "...aij,ab->...iajb",
            covariance_by_output,
            identity,
        )
        return cast(
            torch.Tensor,
            covariance.reshape(*covariance_by_output.shape[:-3], q * num_outputs, -1),
        )


class SingleTaskGP(_BaseExactGP):
    """Exact GP with learned homoskedastic Gaussian observation noise.

    Multiple outputs are represented as independent batched exact GPs with
    shared input locations and separately learned kernel and likelihood state.
    """

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        super().__init__(train_X, train_Y, noise=None)

    def _make_likelihood(
        self,
        backend_train_X: torch.Tensor,
        backend_train_Y: torch.Tensor,
        backend_noise: torch.Tensor | None,
    ) -> Likelihood:
        del backend_train_Y, backend_noise
        return GaussianLikelihood(batch_shape=torch.Size(backend_train_X.shape[:-2]))


class FixedNoiseGP(_BaseExactGP):
    """Exact GP with known per-observation Gaussian noise variances."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor,
    ) -> None:
        super().__init__(train_X, train_Y, noise=train_Yvar)

    @property
    def train_Yvar(self) -> torch.Tensor:
        """Return fixed training variances with shape ``batch_shape x n x m``."""
        noise = cast(torch.Tensor, self.likelihood.noise)
        return noise.movedim(-2, -1)

    def _make_likelihood(
        self,
        backend_train_X: torch.Tensor,
        backend_train_Y: torch.Tensor,
        backend_noise: torch.Tensor | None,
    ) -> Likelihood:
        del backend_train_X, backend_train_Y
        if backend_noise is None:
            raise RuntimeError("FixedNoiseGP requires backend observation noise.")
        return FixedNoiseGaussianLikelihood(
            noise=backend_noise,
            learn_additional_noise=False,
        )
