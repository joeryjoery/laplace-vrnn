from __future__ import annotations
from typing import Sequence, TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

import flax.linen as nn

from jaxtyping import PRNGKeyArray

from axme.core import Scope

from lvrnn.distributions import SerializeTree, mvn
from lvrnn.manifold import orthogonal

from .interface import RecurrentLatentVariableModel, RLVMState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class VRNNState:
    scale_factor: jax.Array
    basis: jax.Array | None
    step: int = 0


class VariationalRNN(RecurrentLatentVariableModel):
    """Extend an RNN core for a Latent-Variable Variational Posterior."""
    prior_stddev: float = 1.0
    diagonal: bool = False

    log_scale_bounds: tuple[float, float] = (-10.0, 2.0)
    manifold: Literal['cayley', 'exp'] = 'cayley'  # Parameterization of SO(n)

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState[mvn.MultivariateNormalExpOrthogonal]:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, _ = self.adapter.split_transformable(cell)

        # Initialize with an N(0_n, I_n * scale) Gaussian prior.
        mean = jnp.zeros(self.features)
        log_scale = jnp.full(self.features, jnp.log(self.prior_stddev))
        basis = None if self.diagonal else jnp.eye(self.features)

        exp_mvn = self._parameters_to_mvn(mean, log_scale, basis)
        init_state = (cell, exp_mvn)

        return RLVMState(
            cell=init_state,
            state=VRNNState(scale_factor=log_scale, basis=basis, step=0)
        )

    def _parameters_to_mvn(
            self,
            mean: jax.Array,
            log_scale: jax.Array,
            basis: jax.Array | None
    ) -> SerializeTree[mvn.MultivariateNormalExpOrthogonal]:
        if self.diagonal:
            return SerializeTree(
                mvn.MultivariateNormalExpOrthogonal, mean, log_scale
            )
        return SerializeTree(
            mvn.MultivariateNormalExpOrthogonal, mean, log_scale, basis
        )

    @staticmethod
    def _compute_metrics(
            posterior: mvn.MultivariateNormal,
            prior: mvn.MultivariateNormal,
            step: int
    ) -> dict[str, dict[str, jax.Array] | jax.Array]:

        # Differential Entropy for a diagonal covariance
        marginal_entropy = 0.5 * (
                1 + jnp.log(2 * jnp.pi) + jnp.log(posterior.variance()).sum()
        )
        kl_div, kl_metrics = posterior.kl_divergence(prior)

        return {
            'posterior_entropy': {
                'marginal': marginal_entropy,
                'differential': posterior.entropy()
            },
            'kl_divergence': {'value': kl_div, **kl_metrics},
            'step': step
        }

    @nn.compact
    def __call__(
            self,
            prev_state: RLVMState[mvn.MultivariateNormal],
            inputs: jax.Array
    ) -> tuple[
        RLVMState[mvn.MultivariateNormal],
        SerializeTree[mvn.MultivariateNormal]
    ]:
        # 1) State-management
        cell, prev_out = prev_state.cell

        # 2) Computation of posterior parameters and core-transition.
        new_cell, phi = self.core(cell, inputs)

        mean = nn.Dense(self.features, name='var_mean')(phi)
        if self.diagonal:
            log_scale = nn.Dense(self.features, name='var_scale')(phi)
            basis = None
        else:
            scale = nn.Dense(
                self.features * (self.features + 1) // 2, name='var_scale'
            )(phi)
            log_scale, phi = jnp.split(scale, (self.features, ))

            # TODO: consider low-rank approximations? -> Very common in Laplace
            basis = orthogonal.real_to_orthogonal(phi, method=self.manifold)

        # Transform parametrized log_scale for gradient-stability.
        log_scale = jnp.clip(
            log_scale, *self.log_scale_bounds
        )

        # 3) Parametrize Gaussian Posterior given Parameters
        dist = self._parameters_to_mvn(mean, log_scale, basis)

        # 4) Optional Logging
        if self.is_mutable_collection(Scope.Intermediates):
            metrics = self._compute_metrics(
                dist.get, prev_out.get, prev_state.state.step + 1
            )
            self.sow(
                Scope.Intermediates,
                type(self).__name__ + '/metrics',
                metrics
            )

        new_vrnn_state = VRNNState(
            scale_factor=log_scale, basis=basis, step=prev_state.state.step + 1
        )
        return RLVMState(cell=(new_cell, dist), state=new_vrnn_state), dist


class ConstantVarianceVariationalRNN(RecurrentLatentVariableModel):
    """Extend an RNN core to have a constant Diagonal Gaussian Posterior."""
    prior_stddev: float = 1.0

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState[mvn.MultivariateNormalExpOrthogonal]:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, _ = self.adapter.split_transformable(cell)

        # Initialize with an N(0_n, I_n * scale) Gaussian prior.
        mean = jnp.zeros(self.features)
        scale = jnp.full(self.features, self.prior_stddev)

        normal = self._parameters_to_mvn(mean, scale)
        init_state = (cell, normal)

        return RLVMState(
            cell=init_state,
            state=VRNNState(scale_factor=scale, basis=None, step=0)
        )

    def _parameters_to_mvn(
            self,
            mean: jax.Array,
            scale: jax.Array
    ) -> SerializeTree[mvn.MultivariateNormalDiagonalCovariance]:
        return SerializeTree(
            mvn.MultivariateNormalDiagonalCovariance, mean, scale
        )

    @staticmethod
    def _compute_metrics(
            posterior: mvn.MultivariateNormal,
            prior: mvn.MultivariateNormal,
            step: int
    ) -> dict[str, int]:
        # Differential Entropy for a diagonal covariance
        kl_div, kl_metrics = posterior.kl_divergence(prior)

        return {
            'posterior_entropy': {
                'differential': posterior.entropy()  # == Constant
            },
            'kl_divergence': {'value': kl_div, **kl_metrics},
            'step': step
        }

    @nn.compact
    def __call__(
            self,
            prev_state: RLVMState[mvn.MultivariateNormal],
            inputs: jax.Array
    ) -> tuple[
        RLVMState[mvn.MultivariateNormal],
        SerializeTree[mvn.MultivariateNormal]
    ]:
        # 1) State-management
        cell, prev_out = prev_state.cell

        # 2) Computation of posterior parameters and core-transition.
        new_cell, phi = self.core(cell, inputs)

        mean = nn.Dense(self.features, name='var_mean')(phi)

        # 3) Parametrize Gaussian Posterior given Parameters
        dist = self._parameters_to_mvn(mean, prev_state.state.scale_factor)

        # 4) Optional Logging
        if self.is_mutable_collection(Scope.Intermediates):
            metrics = self._compute_metrics(
                dist.get, prev_out.get, prev_state.state.step + 1
            )
            self.sow(
                Scope.Intermediates,
                type(self).__name__ + '/metrics',
                metrics
            )

        new_vrnn_state = VRNNState(
            scale_factor=prev_state.state.scale_factor,
            basis=None, step=prev_state.state.step + 1
        )
        return RLVMState(cell=(new_cell, dist), state=new_vrnn_state), dist
