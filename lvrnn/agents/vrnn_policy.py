"""Implements a simple control policy based on AgentModel.

"""
from __future__ import annotations
from typing import Generic, Any, TYPE_CHECKING

from jaxtyping import PRNGKeyArray

import jax
import jax.numpy as jnp

from jit_env import Observation, Action, State

from axme.actor import Policy, P
from axme.core import Variables

from lvrnn.agent_model import AgentModel
from lvrnn.vrnn import DeterministicRNN

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


class Scope:
    """Define annotations for Agent output modalities.

    I.e., what does the agent need to predict in order to derive a policy
    from the model. This can be captured in a dictionary field for compatible
    Model-Policy IO.
    """
    Value: str = 'Value'
    Policy: str = 'Policy'
    Reward: str = 'Reward'

    Signal: str = 'Signal'


class AgentModelPolicy(
    Policy[P, Observation, Action], Generic[P, Observation, Action]
):

    def __init__(self, model: AgentModel, *args, **kwargs):
        self.model = model


@dataclass
class MemoryState(Generic[State, Action]):
    key: PRNGKeyArray
    s_prev: State
    a_prev: Action
    extras: dict[str, Any] | None


class VRNNPolicy(
    AgentModelPolicy[MemoryState, Observation, Action],
    Generic[Observation, Action]
):
    """An implementation for Policy based on AgentModel and a Variational RNN

    """

    def __init__(
            self,
            model: AgentModel,
            test: bool,
            num_ensemble_members: int
    ):
        super().__init__(model)

        if isinstance(model.transition, DeterministicRNN):
            num_ensemble_members = 0

        self.test = test
        self.predictive_kwargs = dict(n=num_ensemble_members)

        modality = [
            m for m in self.model.posterior_predictive.modalities
            if Scope.Policy in m.name
        ]
        self.out = modality[0].spec

    def select_action(
            self,
            variables: Variables,
            state: MemoryState,
            observation: Observation
    ) -> tuple[MemoryState, Action]:
        key_new, key_model, key_sample = jax.random.split(state.key, 3)

        # TODO: flax-State compat through mutable kwarg?
        (predictions, _), (_, model_state) = self.model.apply(
            variables,
            observation, state.a_prev, state.s_prev,
            method=lambda m, *a: m.joint_transition.step(
                *a, **{
                    type(self.model.posterior_predictive).__name__:
                        self.predictive_kwargs
                }
            ),
            rngs=self.model.get_rngs(key_model)
        )

        # TODO: Allow greedification
        action_dist = predictions.get(Scope.Policy).get
        action, log_prob = action_dist.sample_and_log_prob(seed=key_sample)

        return MemoryState(
            key_new,
            model_state,
            action,
            extras={'log_prob': log_prob}
        ), action

    def reset(
            self,
            variables: Variables,
            rng: PRNGKeyArray,
            observation: Observation
    ) -> MemoryState:
        carry, key_init = jax.random.split(rng)

        state = self.model.apply(
            variables,
            observation, jnp.zeros(self.out.shape, self.out.dtype),
            method=lambda m, *a: m.initial_state(*a),
            rngs=self.model.get_rngs(key_init)
        )

        _k = jax.random.key(0)  # dummy key as output doesn't matter
        out = self.model.apply(
            variables,
            state,
            method=lambda m, *a: m.posterior_predictive(*a),
            rngs=self.model.get_rngs(_k)
        )

        action_dist = out.get(Scope.Policy).get
        action, log_prob = jax.tree_map(
            jnp.zeros_like, action_dist.sample_and_log_prob(seed=_k)
        )

        return MemoryState(
            carry,
            state,
            action,
            extras={'log_prob': log_prob}
        )

    def get_extras(
            self,
            policy_state: MemoryState | None
    ) -> dict[str, Any]:
        if policy_state is None:
            return {'log_prob': jnp.zeros(())}
        return policy_state.extras


class VRNNValueInspectionPolicy(
    VRNNPolicy[Observation, Action],
    Generic[Observation, Action]
):
    """An implementation for Policy based on AgentModel and a Variational RNN

    Instead of sampling actions directly from the predicted policy, take
    the predicted policy values into account and use the policy prediction
    only to bias the action-selection proportionally to the values.

    This class implements an MPO-like policy improvement.
    """

    def __init__(
            self,
            model: AgentModel,
            test: bool,
            num_ensemble_members: int,
            num_samples: int,
            epsilon: float = 1e-2
    ):
        super().__init__(model, test, num_ensemble_members)

        self.num_samples = num_samples  # Number of MC-samples for PI.
        self.epsilon = epsilon  # KL-bound for normalizing Q-values

    def select_action(
            self,
            variables: Variables,
            state: MemoryState,
            observation: Observation
    ) -> tuple[MemoryState, Action]:
        key_new, key_model, key_sample = jax.random.split(state.key, 3)

        # TODO: flax-State compat through mutable kwarg?
        (predictions, _), (_, model_state) = self.model.apply(
            variables,
            observation, state.a_prev, state.s_prev,
            method=lambda m, *a: m.joint_transition.step(
                *a, **{
                    type(self.model.posterior_predictive).__name__:
                        self.predictive_kwargs
                }
            ),
            rngs=self.model.get_rngs(key_model)
        )

        # TODO: Allow greedification
        action_dist = predictions.get(Scope.Policy).get
        action, log_prob = action_dist.sample_and_log_prob(seed=key_sample)

        # TODO: Pass self.num_samples of `action` through
        #  model.prior_predictive to get Q-values. Transform the Q-values
        #  and sample a true action from (aka Gumbel MuZero):
        #  softmax(log_prob_action + f(Q_action) + noise)

        return MemoryState(
            key_new,
            model_state,
            action,
            extras={'log_prob': log_prob}
        ), action

    def reset(
            self,
            variables: Variables,
            rng: PRNGKeyArray,
            observation: Observation
    ) -> MemoryState:
        carry, key_init = jax.random.split(rng)

        state = self.model.apply(
            variables,
            observation, jnp.zeros(self.out.shape, self.out.dtype),
            method=lambda m, *a: m.initial_state(*a),
            rngs=self.model.get_rngs(key_init)
        )

        _k = jax.random.key(0)  # dummy key as output doesn't matter
        out = self.model.apply(
            variables,
            state,
            method=lambda m, *a: m.posterior_predictive(*a),
            rngs=self.model.get_rngs(_k)
        )

        # TODO: Modify action selection: see 'step' comment
        action_dist = out.get(Scope.Policy).get
        action, log_prob = jax.tree_map(
            jnp.zeros_like, action_dist.sample_and_log_prob(seed=_k)
        )

        return MemoryState(
            carry,
            state,
            action,
            extras={'log_prob': log_prob}
        )
