from __future__ import annotations
from typing import Generic, Sequence, Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

from lvrnn.distributions.interface import (
    Distribution, EventT, DistT, MonteCarloEntropyMixin
)
from lvrnn.distributions.serialize import SerializeTree


class Mixture(MonteCarloEntropyMixin, Distribution, Generic[DistT]):
    """Forms an ensemble as a convex combination of models.

    See: https://en.wikipedia.org/wiki/Mixture_distribution
    """

    def __init__(
            self,
            dist_stream: SerializeTree[DistT],  # Batched Distribution
            weights: jax.Array
    ):
        super().__init__()
        self.dist = dist_stream.cls_type
        self.dist_stream = dist_stream

        # Ensure linear combination is convex
        bound = jnp.clip(weights.min(), a_max=0)
        self.weights = (weights + bound) / (weights + bound).sum()

    def _to_batch_fun(
            self,
            method: Callable[[DistT, ...], ...],
            *static_args, **static_kwargs
    ) -> Callable:
        inner_fun = jax.vmap(
            lambda dist, *a, **k: method(
                dist.get, *a, *static_args, **k, **static_kwargs
            )
        )
        return partial(inner_fun, self.dist_stream)

    def _sample_n(self, key: PRNGKeyArray, n: int) -> EventT:
        key_choice, key_component = jax.random.split(key)

        idx = jax.random.choice(
            key_choice, len(self.weights), (n,),
            replace=True, p=self.weights
        )

        samples_per_component = self._to_batch_fun(
            self.dist.sample, sample_shape=(n,)
        )(seed=jax.random.split(key_component, num=len(self.weights)))

        return samples_per_component.at[idx, jnp.arange(n)].get()

    @property
    def event_shape(self) -> Sequence[int]:
        return jax.vmap(lambda d: d.get.event_shape)(self.dist_stream)[0]

    def mode(self) -> jax.Array:
        # Mixture can be multimodal or the modes are difficult to compute.
        raise NotImplementedError(
            "The `mode` of a Generic Mixture is ambiguous."
        )

    def median(self) -> jax.Array:
        # Difficult to compute
        raise NotImplementedError()

    def mean(self) -> jax.Array:
        means = self._to_batch_fun(self.dist.mean)()
        return jax.vmap(jnp.multiply)(self.weights, means).sum(axis=0)

    def variance(self) -> jax.Array:
        # Var(X) = E[Var(X | model)] + Var(E[X | model])
        variances = self._to_batch_fun(self.dist.variance)()
        means = self._to_batch_fun(self.dist.mean)()

        mean = jax.vmap(jnp.multiply)(self.weights, means).sum(axis=0)
        mean_var = jax.vmap(jnp.multiply)(self.weights, variances).sum(axis=0)

        errors = jax.vmap(jnp.multiply)(
            self.weights, jnp.square(means - mean[None, ...])
        )
        var_mean = errors.sum(axis=0)

        return mean_var + var_mean

    def log_prob(self, value: EventT) -> jax.Array:
        log_probs = self._to_batch_fun(self.dist.log_prob, value)()
        return jax.vmap(jnp.multiply)(self.weights, log_probs).sum(axis=0)

    def log_cdf(self, value: EventT) -> jax.Array:
        log_cdfs = self._to_batch_fun(self.dist.log_cdf, value)()
        return jax.vmap(jnp.multiply)(self.weights, log_cdfs).sum(axis=0)
