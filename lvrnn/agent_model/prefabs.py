from __future__ import annotations
from typing import Sequence, Callable, Generic, Any

from jaxtyping import PyTree, PRNGKeyArray

import flax.linen as nn

import jax
import jax.numpy as jnp

import numpy as np

from .interface import (
    Embedding, Observation, State, Action, AfterState,
    PosteriorPredictive, StateAdapter,
    Embedder, ApplyAction, PosteriorTransition, PriorTransition,
    SignalPrediction, TV
)


class MLP(nn.Module):
    layer_features: Sequence[int]
    dense_kwargs: dict[str, Any] | None = None

    activation: Callable[[jax.Array, ...], jax.Array] = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: Implement with nn.scan
        layers = [
            l for size in self.layer_features for l in (
                nn.Dense(size, **(self.dense_kwargs or {})), self.activation
            )
        ]

        if not self.activate_final:
            layers.pop()

        return nn.Sequential(layers)(x)


class TransitionAdapter(StateAdapter):
    """Adapter for State-Action and AfterState comspatibility."""

    @staticmethod
    def split_transformable(
            state: AfterState
    ) -> tuple[Embedding, State]:
        return state[0], state[1]

    @staticmethod
    def combine_transformed(
            sub_state: State,
            transformed: Embedding
    ) -> AfterState:
        return sub_state, transformed


class FlaxRNNCoreAdapter(StateAdapter, Generic[TV]):
    """Adapter to handle flax RNN Carry datastructures.

    For example, how to deal with LSTM state Vs. GRU state.
    """

    @staticmethod
    def split_transformable(
            state: tuple[PyTree[jax.Array], TV]
    ) -> tuple[PyTree[jax.Array], TV]:
        # Split the core state structure into (output, hidden-state).
        if isinstance(state, tuple):
            return state, state[1]  # LSTM: ((cell, hidden), hidden)
        return state, state  # RNN/ GRU: (z, z)

    @staticmethod
    def combine_transformed(
            sub_state: PyTree[jax.Array],
            transformed: TV
    ) -> tuple[PyTree[jax.Array], TV]:
        # Combine (output, hidden-state) into a core state structure.
        if isinstance(sub_state, tuple):
            return sub_state[0], transformed  # LSTM: (cell, hidden)
        return transformed  # RNN/ GRU: z


class FlaxRNNStateAdapter(StateAdapter):
    """Adapter to handle flax RNN __call__ output: (Carry, Out).

    This adapter is an API hook, and only wraps a 2-tuple unpacking.
    """

    @staticmethod
    def split_transformable(
            state: tuple[PyTree[jax.Array], PyTree[jax.Array]]
    ) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
        return state

    @staticmethod
    def combine_transformed(
            sub_state: PyTree[jax.Array],
            transformed: PyTree[jax.Array]
    ) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
        return sub_state, transformed


class DefaultEmbedder(Embedder):
    layer_sizes: Sequence[int] = (64, 64)
    activation: Callable[[jax.Array, ...], jax.Array] = nn.leaky_relu
    normalize: bool = True

    @nn.compact
    def __call__(
            self,
            ins: Observation | Action
    ) -> Embedding:
        flat = jax.tree_map(jnp.ravel, ins)
        ins = jnp.concatenate(jax.tree_util.tree_leaves(flat), -1)

        net = MLP(
            self.layer_sizes,
            activation=self.activation,
            activate_final=True
        )
        e = net(ins)

        if self.normalize:
            return nn.LayerNorm()(e)
        return e


class DefaultAfterState(ApplyAction):
    adapter: StateAdapter = TransitionAdapter

    @nn.compact
    def __call__(
            self,
            s_t: State,
            a_embed_t: Embedding
    ) -> AfterState:
        return self.adapter.combine_transformed(s_t, a_embed_t)


class DefaultPredicter(PosteriorPredictive):
    layer_sizes: Sequence[int] = (64,)
    output_shape: Sequence[int] = (1,)  # TODO: Output_tree
    activation: Callable = jax.nn.leaky_relu

    adapter: StateAdapter = FlaxRNNStateAdapter

    @nn.compact
    def __call__(self, s_t: State):
        net = MLP(
            (*self.layer_sizes, np.prod(self.output_shape)),
            activation=self.activation, activate_final=False
        )

        _, ins = self.adapter.split_transformable(s_t)

        return net(ins).reshape(self.output_shape)


class DefaultTransition(PosteriorTransition):
    adapter: StateAdapter = TransitionAdapter
    state_adapter: StateAdapter = FlaxRNNStateAdapter
    core_adapter: StateAdapter = FlaxRNNCoreAdapter

    core: nn.RNNCellBase = nn.OptimizedLSTMCell(128)

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

        return self.state_adapter.combine_transformed(carry, out)

    def initialize_carry(
      self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> State:
        # Note: method out-signature *must* match __call__ out-signature!
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        _, out = self.core_adapter.split_transformable(cell)

        return self.state_adapter.combine_transformed(cell, out)


class DefaultSimulater(PriorTransition):
    adapter: StateAdapter = TransitionAdapter
    state_adapter: StateAdapter = FlaxRNNStateAdapter

    layer_sizes: Sequence[int] = (128,)
    output_shape: Sequence[int] = (1,)  # TODO: output_tree
    activation: Callable = jax.nn.leaky_relu

    @nn.compact
    def __call__(
            self,
            as_t: AfterState
    ) -> tuple[SignalPrediction, State]:

        state, a_embed_t = self.adapter.split_transformable(as_t)
        carry, h = self.state_adapter.split_transformable(state)

        normalizer = nn.LayerNorm()
        signal_net = MLP(
            (*self.layer_sizes, h.size + np.prod(self.output_shape)),
            activation=self.activation, activate_final=False
        )

        ins = h.ravel()
        if a_embed_t is not None:
            ins = jnp.concatenate([a_embed_t.ravel(), ins], -1)

        outputs = signal_net(normalizer(ins))
        h_out, y_hat = jnp.split(outputs, indices_or_sections=(h.size,))
        h_out = h_out.reshape(h.shape)

        if not y_hat.size:
            y_hat = None

        return y_hat, self.state_adapter.combine_transformed(carry, h_out)
