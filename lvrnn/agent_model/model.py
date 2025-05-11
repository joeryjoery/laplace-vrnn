from __future__ import annotations
from typing import Generic, Any

from dataclasses import fields

import chex

import jax
import jax.numpy as jnp

import flax.linen as nn

from jaxtyping import PRNGKeyArray

from axme.core import Scope

from .interface import (
    Batched,
    Observation, Action, Embedding, State,
    AfterState, SignalPrediction, StatePrediction, AfterStatePrediction,
    Embedder, ApplyAction, PosteriorTransition, PriorTransition,
    PosteriorPredictive, PriorPredictive, StateInitializer
)


def get_sub_module_kwargs(submodule, kwargs) -> dict[str, Any]:
    if submodule.name:
        return kwargs.get(submodule.name, {})
    return kwargs.get(type(submodule).__name__, {})


class ModelComponents(nn.Module, Generic[
    Observation,
    Action,
    Embedding,
    State,
    AfterState,
    SignalPrediction,
    StatePrediction,
    AfterStatePrediction
]):
    """Partially specified AgentModel composed of atomic transformations."""

    # Mappings of data modalities to an Agent's modality.
    observation_embedder: Embedder[Observation, Embedding]
    action_embedder: Embedder[Action, Embedding]

    # Prior and Posterior state transition method.
    transition: PosteriorTransition[Embedding, AfterState, State]
    apply_action: ApplyAction[State, Action, AfterState]
    simulate: PriorTransition[AfterState, SignalPrediction, State] | None

    # Prior and Posterior state prediction method.
    posterior_predictive: PosteriorPredictive[State, StatePrediction] | None
    prior_predictive: PriorPredictive[AfterState, AfterStatePrediction] | None

    # Generate a starting state for the transition and simulate method.
    initial_state: StateInitializer

    def get_rngs(
            self,
            rng: PRNGKeyArray
    ) -> dict[str, PRNGKeyArray]:
        # Helper method to get individual RNG-branches for all submodules
        attributes = [getattr(self, f.name) for f in fields(self)]
        sub_modules = [s for s in attributes if isinstance(s, nn.Module)]

        keys = jax.random.split(rng, len(sub_modules))

        return {type(s).__name__: k for s, k in zip(sub_modules, keys)}

    def __call__(self, observation: Observation, action: Action) -> None:
        """Individually call all atomic ModelComponents for param-init. """

        # Embed input modalities
        o_embed_t = self.observation_embedder(observation)
        a_embed_t = self.action_embedder(action)

        # Initialize after-state
        s_init = self.initial_state(observation, action)
        as_init = self.apply_action(s_init, a_embed_t)

        # Apply state-update method
        s_t = self.transition(o_embed_t, as_init)
        as_t = self.apply_action(s_t, a_embed_t)

        if self.simulate:
            _ = self.simulate(as_t)

        # Apply state-statistic prediction method
        if self.posterior_predictive:
            _ = self.posterior_predictive(s_t)

        if self.prior_predictive:
            _ = self.prior_predictive(as_t)


class DefaultModelState(
    StateInitializer[State, Observation, Action],
    Generic[State, Observation, Action]
):
    observation_embedder: Embedder
    action_embedder: Embedder
    transition: PosteriorTransition

    @nn.nowrap
    def _forward_input(self, obs: Observation, a: Action) -> jax.Array:
        ins = self.observation_embedder(obs)

        if a is not None:
            ins = jnp.concatenate([ins, self.action_embedder(a)], -1)
        return ins

    def __call__(
            self,
            observation: Observation,
            action: Action,
            *args, **kwargs
    ) -> State:
        # TODO: trace for shape only!
        ins = self._forward_input(observation, action)

        return self.transition.initialize_carry(
            self.make_rng(type(self).__name__),
            ins.shape
        )


class InitialInference(nn.Module):
    observation_embedder: Embedder
    action_embedder: Embedder
    transition: PosteriorTransition
    apply_action: ApplyAction

    prior_predictive: PriorPredictive | None
    posterior_predictive: PosteriorPredictive | None

    initial_state: StateInitializer

    def step(
            self,
            o_t: Observation,
            a_prev: Action,
            s_prev: State | None,
            **kwargs
    ) -> tuple[
        tuple[StatePrediction | None, AfterStatePrediction | None],
        tuple[AfterState, State]
    ]:
        """Utility for updating the model Posterior given IO.

        Performs the joint mapping:
            (o_t, a_prev, s_prev) -> (f_t, q_prev, (signal_t, s_t))
        """

        # State Prior
        if s_prev is None:
            s_prev = self.initial_state(
                o_t, a_prev,
                **get_sub_module_kwargs(self.initial_state, kwargs)
            )

        a_embed_prev = self.action_embedder(
            a_prev,
            **get_sub_module_kwargs(self.action_embedder, kwargs)
        )
        as_prev = self.apply_action(
            s_prev, a_embed_prev,
            **get_sub_module_kwargs(self.apply_action, kwargs)
        )

        # State Posterior
        o_embed_t = self.observation_embedder(
            o_t,
            **get_sub_module_kwargs(self.observation_embedder, kwargs)
        )
        s_t = self.transition(
            o_embed_t, as_prev,
            **get_sub_module_kwargs(self.transition, kwargs)
        )

        q_prev = f_t = None
        if self.prior_predictive:
            q_prev = self.prior_predictive(
                as_prev,
                **get_sub_module_kwargs(self.prior_predictive, kwargs)
            )

        if self.posterior_predictive:
            f_t = self.posterior_predictive(
                s_t,
                **get_sub_module_kwargs(self.posterior_predictive, kwargs)
            )

        return (f_t, q_prev), (as_prev, s_t)

    def batch(
            self,
            actions: Batched[Action],
            s_t: State,
            **kwargs
    ):
        raise NotImplementedError()  # TODO: Transformers

    def unroll(
            self,
            observations: Batched[Observation],
            actions: Batched[Action],
            s_prev: State | None,
            **kwargs
    ) -> tuple[
        tuple[
            Batched[StatePrediction] | None,
            Batched[AfterStatePrediction] | None
        ],
        tuple[Batched[AfterState], Batched[State]]
    ]:
        chex.assert_trees_all_equal_comparator(
            lambda a, b: jnp.shape(a)[0] == jnp.shape(b)[0],
            lambda a, b: "Observations and Actions must have equal lengths! "
                         f"Received leaf with Obs: {jnp.shape(a)[0]}, "
                         f"Act: {jnp.shape(b)[0]}",
            observations, actions
        )

        if s_prev is None:
            o_0, a_0 = jax.tree_map(
                lambda x: x.at[0].get(),
                (observations, actions)
            )
            s_prev = self.initial_state(o_0, a_0)

        def body(
                module: InitialInference,
                carry: State,
                x: tuple[Observation, Action]
        ) -> tuple[
            State, tuple[
                tuple[StatePrediction, AfterStatePrediction | None],
                tuple[AfterState, State]
            ]
        ]:
            (f, q), (as_t, new_state) = module.step(*x, carry, **kwargs)
            return new_state, ((f, q), (as_t, new_state))

        scanner = nn.scan(
            body,
            variable_axes={Scope.Intermediates: 0},
            variable_broadcast=Scope.Params,
            split_rngs={Scope.Params: False}
        )

        return scanner(self, s_prev, (observations, actions))[1]


class RecurrentInference(nn.Module):
    action_embedder: Embedder
    simulate: PriorTransition
    apply_action: ApplyAction

    prior_predictive: PriorPredictive | None
    posterior_predictive: PosteriorPredictive | None

    def step(
            self,
            a_t: Action,
            s_t: State,
            **kwargs
    ) -> tuple[
        tuple[StatePrediction | None, AfterStatePrediction | None],
        tuple[SignalPrediction, tuple[AfterState, State]]
    ]:
        """Utility for simulating a transition with the model Prior.

        Performs the joint mapping:
            (s_t, a_t) -> (f_next, q_t, (signal_next, s_next))
        """
        a_embed_t = self.action_embedder(
            a_t,
            **get_sub_module_kwargs(self.action_embedder, kwargs)
        )

        as_t = self.apply_action(
            s_t, a_embed_t,
            **get_sub_module_kwargs(self.apply_action, kwargs)
        )
        signal_next, s_next = self.simulate(
            as_t,
            **get_sub_module_kwargs(self.simulate, kwargs)
        )

        q_t = f_next = None
        if self.prior_predictive:
            q_t = self.prior_predictive(
                as_t,
                **get_sub_module_kwargs(self.prior_predictive, kwargs)
            )

        if self.posterior_predictive:
            f_next = self.posterior_predictive(
                s_next,
                **get_sub_module_kwargs(self.posterior_predictive, kwargs)
            )

        return (f_next, q_t), (signal_next, (as_t, s_next))

    def batch(
            self,
            actions: Batched[Action],
            s_t: State,
            **kwargs
    ):
        raise NotImplementedError()  # TODO: Transformers

    def unroll(
            self,
            actions: Batched[Action],
            s_t: State,
            **kwargs
    ) -> tuple[
        tuple[
            Batched[StatePrediction] | None,
            Batched[AfterStatePrediction] | None
        ],
        tuple[
            Batched[SignalPrediction],
            tuple[Batched[AfterState], Batched[State]]
        ]
    ]:
        def body(
                module: RecurrentInference,
                carry: State,
                x: Action
        ) -> tuple[
            State, tuple[
                tuple[StatePrediction, AfterStatePrediction | None],
                tuple[SignalPrediction, tuple[AfterState, State]]
            ]
        ]:
            (f, q), (out, (as_t, new_state)) = module.step(x, carry, **kwargs)
            return new_state, ((f, q), (out, (as_t, new_state)))

        scanner = nn.scan(
            body,
            variable_axes={Scope.Intermediates: 0},
            variable_broadcast=Scope.Params,
            split_rngs={Scope.Params: False}
        )

        return scanner(self, s_t, actions)[1]


class AgentModel(ModelComponents):
    joint_transition: InitialInference
    joint_simulate: RecurrentInference | None
