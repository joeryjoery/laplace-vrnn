from __future__ import annotations
from typing import Sequence, TypeVar, Generic, Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxtyping import PyTree, PRNGKeyArray

import jit_env

from axme.actor import Policy
from axme.core import Variables
from axme.data import environment

from lvrnn.agent_model import AgentModel, State, Action
from lvrnn.agent_model.model import Batched

if TYPE_CHECKING:
    from dataclasses import dataclass
    container = dataclass
else:
    from flax.struct import dataclass
    from chex import dataclass as _c
    container = _c(frozen=False)


def nest_vmap(f: Callable, n: int, *args, **kwargs):
    if n == 0:
        return f
    else:
        return nest_vmap(
            jax.vmap(f, *args, **kwargs),
            n - 1,
            *args, **kwargs
        )


E = TypeVar("E", bound=jit_env.Environment)


@container
class EnvEvaluator(Generic[E]):
    # Define a base utility to sample from env-policy space.
    model: AgentModel
    policy: Policy
    task: environment.Environment[E]
    fixed_seed: PRNGKeyArray

    def _trajectory_fun(
            self,
            key: PRNGKeyArray,
            variables: Variables,
            options: jit_env.EnvOptions,
            length: int
    ) -> tuple[jit_env.State, jit_env.TimeStep, jit_env.Action]:

        key_env, key_policy = jax.random.split(key)

        init_state, init_step = self.task.env.reset(key_env, options=options)
        policy_init = self.policy.reset(
            variables, key_policy, init_step.observation
        )

        def body(carry, x: None):
            state, step, policy_state = carry

            policy_state, action = self.policy.select_action(
                variables, policy_state, step.observation
            )
            state, step = self.task.env.step(state, action)

            return (state, step, policy_state), (state, step, action)

        _, (states, steps, actions) = jax.lax.scan(
            body, (init_state, init_step, policy_init),
            None, length=length
        )

        # Prepend zeroth state, remove last state
        states = jax.tree_map(
            lambda a, b: jnp.concatenate([
                jnp.expand_dims(a, 0), b.at[:-1].get()
            ]),
            init_state, states
        )

        return states, steps, actions


@dataclass
class AgentModelUtility:
    """Utility Class to get evaluation statistics from AgentModel."""
    model: AgentModel
    sub_samples: Sequence[int] | None = None

    def get_states(
            self,
            params: Variables,
            key: PRNGKeyArray,
            xs: PyTree[Batched[jax.Array]],
            ys: PyTree[Batched[jax.Array]],
            return_metrics: bool = False,
            initial_state: bool = True
    ) -> Batched[State]:
        init_key, unroll_key = jax.random.split(key)

        x_0, y_0 = jax.tree_map(lambda x: x.at[0].get(), (xs, ys))

        s_0 = self.model.apply(
            params,
            y_0, x_0,
            method=lambda m, *a: m.initial_state(*a),
            rngs=self.model.get_rngs(init_key)
        )

        metrics = None
        if return_metrics:
            (_, (_, s_seq)), metrics = self.model.apply(
                params, ys, xs, s_0,
                method=lambda m, *a: m.joint_transition.unroll(*a),
                rngs=self.model.get_rngs(unroll_key),
                capture_intermediates=lambda m, name: 'metric' in name
            )
        else:
            _, (_, s_seq) = self.model.apply(
                params, ys, xs, s_0,
                method=lambda m, *a: m.joint_transition.unroll(*a),
                rngs=self.model.get_rngs(unroll_key)
            )

        s_full_seq = s_seq
        if initial_state:
            s_full_seq = jax.tree_map(
                lambda a, b: jnp.concatenate(
                    [jnp.expand_dims(a, 0), b], axis=0
                ),
                s_0, s_seq
            )

        if self.sub_samples is None:
            return s_full_seq, metrics

        return jax.tree_map(
            lambda x: x.at[jnp.asarray(self.sub_samples)].get(),
            (s_full_seq, metrics)
        )

    def get_predictives(
            self,
            params: Variables,
            key: PRNGKeyArray,
            state: State,
            n: int
    ):
        return self.model.apply(
            params,
            state, n,
            method=lambda m, *a: m.posterior_predictive(*a),
            rngs=self.model.get_rngs(key)
        )

    def get_simulations(
            self,
            params: Variables,
            key: PRNGKeyArray,
            state: State,
            action: Action,
            n: int
    ):
        key_embed, key_apply, key_sim = jax.random.split(key, 3)

        action_embedding = self.model.apply(
            params,
            action,
            method=lambda m, *a: m.action_embedder(*a),
            rngs=self.model.get_rngs(key_embed)
        )
        afterstates = self.model.apply(
            params,
            state, action_embedding,
            method=lambda m, *a: m.apply_action(*a),
            rngs=self.model.get_rngs(key_apply)
        )
        signal_prediction, _ = self.model.apply(
            params,
            afterstates, n,
            method=lambda m, *a: m.simulate(*a),
            rngs=self.model.get_rngs(key_sim)
        )

        return signal_prediction
