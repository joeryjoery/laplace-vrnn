"""TODO: BBOx integration."""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

import jit_env

from jit_env import specs, State, Action, TimeStep

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class BanditState:
    key: PRNGKeyArray
    ps: jax.Array
    step: int


class Multinoulli(jit_env.Environment):
    # TODO: Remove reward from observation field, bandit has no observations.
    #  Or ignore now for simplicity...
    _REGRET_KEY: str = 'regret'
    _REGRET_MINMAX_KEY: str = 'regret min-max'

    def __init__(self, n: int, alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.alpha = alpha

    def reset(
            self,
            key: PRNGKeyArray, 
            *, 
            options: dict | None = None
    ) -> tuple[State, TimeStep]:
        key_carry, key_prior = jax.random.split(key)

        ps = jax.random.dirichlet(
            key_prior, self.alpha * jnp.ones(self.n), ()
        )

        step = jit_env.restart(
            jnp.zeros(()),
            extras={
                self._REGRET_KEY: jnp.zeros(()),
                self._REGRET_MINMAX_KEY: jnp.zeros(())
            }
        )

        return BanditState(
            key=key_carry,
            ps=ps,
            step=0
        ), step

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        key_carry, key_sample = jax.random.split(state.key)

        p = state.ps.at[action].get()
        r = jax.random.bernoulli(key_sample, p).astype(jnp.float32)

        # Min-Max scale regret between 0 (optimal) and 1 (random).
        best = state.ps.max()
        uniform = best / state.ps.size

        regret = best - p  # Immediate Regret in expectation
        regret_minmax = regret / (best - uniform)

        step = jit_env.transition(
            r, r, discount=jnp.zeros_like(r),
            extras={
                self._REGRET_KEY: regret,
                self._REGRET_MINMAX_KEY: regret_minmax
            }
        )

        return BanditState(
            key=key_carry, ps=state.ps, step=state.step+1
        ), step

    def reward_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32, 0.0, 1.0)

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32,0.0, 1.0)

    def observation_spec(self) -> specs.Spec:
        return self.reward_spec()

    def action_spec(self) -> specs.Spec:
        return specs.DiscreteArray(self.n)


def rbf(
        x: jax.Array,
        y: jax.Array,
        scale: float = 1.0,
        variance: float = 1.0
) -> jax.Array:
    # TODO: BBOx integration
    return scale * jnp.exp(-jnp.square(x - y) / (variance * 2))


class GaussianProcessPrior(jit_env.Environment):

    def __init__(
            self,
            n: int,
            bounds: tuple[jax.Array, jax.Array],
            resolution: int,
            similarity_fun: Callable[[jax.Array, jax.Array], jax.Array] = rbf,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n = n

        self.resolution = resolution
        self.bounds = bounds

        self.kernel = jax.vmap(
            jax.vmap(
                similarity_fun, in_axes=(None, 0)
            ), in_axes=(0, None)
        )

    def reset(
            self,
            key: PRNGKeyArray,
            *,
            options: dict | None = None
    ) -> tuple[State, TimeStep]:
        pass

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        pass

    def reward_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32, 0.0, 1.0)

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32,0.0, 1.0)

    def observation_spec(self) -> specs.Spec:
        return self.reward_spec()

    def action_spec(self) -> specs.Spec:
        return specs.BoundedArray((self.n,), jnp.float32, *self.bounds)
