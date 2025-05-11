from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

import flax.linen as nn

from axme.core import Scope

from lvrnn.distributions import SerializeTree, Deterministic, Distribution

from .interface import RecurrentLatentVariableModel, RLVMState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class DetRNNState:
    step: int = 0


class DeterministicRNN(RecurrentLatentVariableModel):
    """Extend an RNN core for RLVM-interface compatibility.

    We treat the mean-output of the given RNN-core as a Dirac distribution.
    """

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, _ = self.adapter.split_transformable(cell)

        init_state = (cell, self._dist_from_param(jnp.zeros(self.features)))

        return RLVMState(cell=init_state, state=DetRNNState(step=0))

    @staticmethod
    def _compute_metrics(
            posterior: Deterministic,
            prior: Deterministic,
            step: int
    ) -> dict[str, jax.Array]:
        # Measure Standard Gaussian Mahalanobis distance (i.e., L2 distance).
        delta_mean = posterior.mean() - prior.mean()
        return {
            'delta_norm': jnp.square(delta_mean).sum(),
            'posterior_norm': jnp.square(posterior.mean()).sum(),
            'posterior_max': posterior.mean().max(),
            'posterior_min': posterior.mean().min(),
            'step': step
        }

    def _dist_from_param(
            self,
            features: jax.Array
    ) -> SerializeTree[Deterministic]:
        return SerializeTree(Deterministic, features)

    @nn.compact
    def __call__(
            self,
            prev_state: RLVMState,
            inputs: jax.Array
    ) -> tuple[RLVMState, SerializeTree[Distribution]]:
        # 1) State-management
        cell, prev_out = prev_state.cell

        # 2) Computation of posterior parameters == core-transition.
        new_cell, phi = self.core(cell, inputs)
        mean = nn.Dense(self.features, name='var_mean')(phi)

        # 3) Parametrize Gaussian Posterior given Parameters
        out = self._dist_from_param(mean)

        # 4) Logging
        if self.is_mutable_collection(Scope.Intermediates):
            metrics = self._compute_metrics(
                out.get, prev_out.get, prev_state.state.step + 1
            )
            self.sow(Scope.Intermediates, type(self).__name__, metrics)

        return RLVMState(
            cell=(new_cell, out),
            state=DetRNNState(step=prev_state.state.step + 1)
        ), out
