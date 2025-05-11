from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp

from .interface import Distribution, EventT


class Deterministic(Distribution):

    def __init__(self, loc):
        self.loc = loc

    def _sample_n(self, key: jax.random.KeyArray, n: int) -> EventT:
        return jnp.broadcast_to(self.loc, (n, *self.loc.shape))

    @property
    def event_shape(self) -> Sequence[int]:
        return self.loc.shape

    def mean(self) -> jax.Array:
        return self.loc

    def mode(self) -> jax.Array:
        return self.loc

    def median(self) -> jax.Array:
        return self.loc

    def variance(self) -> jax.Array:
        return jnp.zeros_like(self.loc)

    def log_prob(self, value: EventT) -> jax.Array:
        """See `Distribution.log_prob`."""
        return jnp.log(self.prob(value))

    def prob(self, value: EventT) -> jax.Array:
        """See `Distribution.prob`."""
        return jnp.where(
            jnp.isclose(jnp.abs(value - self.loc), 0.0),
            1., 0.
        )

    def log_cdf(self, value: EventT) -> jax.Array:
        return jnp.log(self.cdf(value))

    def cdf(self, value: EventT) -> jax.Array:
        return jnp.where(value >= self.loc, 1., 0.)

    def kl_divergence(
            self,
            other_dist: Deterministic,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return -self.log_prob(other_dist.loc), {}

    def entropy(self):
        return jnp.zeros_like(self.loc)
