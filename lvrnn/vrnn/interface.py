from __future__ import annotations
from typing import Literal, Any, Generic, Sequence, TypeVar, TYPE_CHECKING
from abc import ABC

from jaxtyping import PyTree, PRNGKeyArray

import flax.linen as nn

import jax
import jax.numpy as jnp

from axme.data import Modality

from lvrnn.distributions import SerializeTree
from lvrnn.distributions.interface import DistT
from lvrnn.agent_model import (
    StateAdapter, FlaxRNNCoreAdapter, FlaxRNNStateAdapter,
    State, Embedding, AfterState, DefaultTransition
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class ModelModality(Modality):
    """Config to specify how a model output should be formatted."""

    likelihood: Literal['gaussian', 'categorical', 'dirac'] = 'gaussian'
    likelihood_kwargs: dict[str, Any] | None = None

    ensemble_method: Literal['mixture', 'aggregate'] | None = None
    uniform_ensemble: bool | None = True  # Weight members to their log-probs.


VarState = TypeVar("VarState")


@dataclass
class RLVMState(Generic[DistT, VarState]):
    cell: tuple[PyTree[jax.Array], SerializeTree[DistT]]
    state: VarState


class RLVMAdapter(StateAdapter, Generic[DistT, VarState]):  # StateAdapterProtocol
    """Adapter for Recurrent-Latent-Variable Model states compatibility."""

    @staticmethod
    def split_transformable(
            state: RLVMState[DistT, VarState]
    ) -> tuple[RLVMState[DistT, VarState], DistT]:
        cell, out = FlaxRNNStateAdapter.split_transformable(state.cell)
        return state, out

    @staticmethod
    def combine_transformed(
            sub_state: RLVMState[DistT, VarState],
            transformed: DistT
    ) -> RLVMState[DistT, VarState]:
        cell, _ = FlaxRNNStateAdapter.split_transformable(sub_state.cell)
        new_cell = FlaxRNNStateAdapter.combine_transformed(
            cell, transformed
        )
        return RLVMState(cell=new_cell, state=sub_state.state)


class RLVMTransition(DefaultTransition):

    # Choose between :
    #  1. Stochastic State-Space Model: Output = RVs
    #  2. Recurrent SSM: Output = RVs and Deterministic State
    #  3. Bayes-Adaptive Model: Output = RVs and Last Observation
    posterior_type: Literal['ssm', 'rssm', 'bamdp'] = 'ssm'

    action_conditional: bool = True  # TODO Stupid Hack variable because the code is shit

    @nn.compact
    def __call__(
            self,
            e_t: Embedding,
            as_prev: AfterState
    ) -> State:
        # Unpack carried state into transformable elements.
        s_prev, a_embed_t = self.adapter.split_transformable(as_prev)
        cell_prev, _ = self.state_adapter.split_transformable(s_prev)

        ins = e_t
        if a_embed_t is not None:
            ins = jnp.concatenate([e_t, a_embed_t], -1)

        carry, out = self.core(cell_prev, ins)

        if self.posterior_type == 'ssm':
            return self.state_adapter.combine_transformed(carry, (None, out))

        elif self.posterior_type == 'rssm':
            cell, _ = FlaxRNNStateAdapter.split_transformable(carry.cell)
            _, hidden = FlaxRNNCoreAdapter.split_transformable(cell)

            return self.state_adapter.combine_transformed(carry, (hidden, out))

        elif self.posterior_type == 'bamdp':
            return self.state_adapter.combine_transformed(carry, (e_t, out))

        raise ValueError(
            f"Posterior Type must be either `ssm`, `rssm`, or"
            f"`bamdp`, invalid option: {self.posterior_type}!"
        )

    def initialize_carry(
      self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> State:
        # Note: method out-signature *must* match __call__ out-signature!
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        carry, out = self.core_adapter.split_transformable(cell)

        if self.posterior_type == 'ssm':
            return self.state_adapter.combine_transformed(carry, (None, out))

        elif self.posterior_type == 'rssm':
            cell, _ = FlaxRNNStateAdapter.split_transformable(carry.cell)
            _, hidden = FlaxRNNCoreAdapter.split_transformable(cell)

            return self.state_adapter.combine_transformed(carry, (hidden, out))

        elif self.posterior_type == 'bamdp':
            if self.action_conditional:
                # TODO: Stupid hack to half the size of the input_shape.
                return self.state_adapter.combine_transformed(
                    carry, (jnp.zeros(input_shape)[::2], out)
                )
            return self.state_adapter.combine_transformed(
                carry, (jnp.zeros(input_shape), out)
            )

        raise ValueError(
            f"Posterior Type must be either `ssm`, `rssm`, or"
            f"`bamdp`, invalid option: {self.posterior_type}!"
        )


class RecurrentLatentVariableModel(nn.RNNCellBase, ABC):
    adapter: StateAdapter = FlaxRNNCoreAdapter

    features: int | None = None
    core: nn.RNNCellBase = nn.OptimizedLSTMCell(128)
