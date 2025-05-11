from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp

from .interface import Distribution, EventT


class Categorical(Distribution):
    """
    Equivalent to the Boltzmann distribution over a discrete topology.
    See: https://en.wikipedia.org/wiki/Categorical_distribution
    """

    def __init__(
            self,
            logits: jax.Array,  # Untested for anything other than 1D
            temperature: jax.typing.ArrayLike = 1.0
    ):
        self.logits = logits
        self.temperature = jnp.asarray(temperature, dtype=logits.dtype)

    def _sample_n(self, key: jax.random.KeyArray, n: int) -> jax.Array:

        greedy = jnp.arange(self.logits.size) == self.logits.argmax()
        ps = jax.nn.softmax(
            self.logits / jnp.clip(self.temperature, a_min=1e-8)
        )
        ps = jax.lax.select(
            jnp.isclose(self.temperature, 0.0),
            greedy.astype(ps.dtype),
            ps
        )

        sample = jax.random.choice(
            key, self.logits.size, (n, ), replace=True, p=ps
        )

        return jnp.broadcast_to(sample, (n, *self.event_shape))

    @property
    def event_shape(self) -> Sequence[int]:
        return self.logits.shape[1:]

    def mean(self) -> jax.Array:
        raise NotImplementedError(
            f"{type(self).__name__} has no mean implemented!"
        )

    def mode(self) -> jax.Array:
        return jnp.argmax(self.logits, axis=-1)

    def median(self) -> jax.Array:
        raise NotImplementedError(
            f"{type(self).__name__} has no mean implemented!"
        )

    def variance(self) -> jax.Array:
        raise NotImplementedError(
            f"{type(self).__name__} has no variance implemented!"
        )

    def log_prob(self, value: EventT) -> jax.Array:
        """See `Distribution.log_prob`."""
        value = jnp.asarray(value, jnp.int32)

        log_probs = jax.nn.log_softmax(
            self.logits / jnp.clip(self.temperature, a_min=1e-8)
        )
        ll = log_probs.at[value.astype(jnp.int32)].get()

        is_inf = jnp.isclose(self.temperature, 0.0) & \
            (value != self.logits.argmax())

        return jax.lax.select(
            (0 <= value) & (value < self.logits.size) & ~is_inf,
            ll, -jnp.inf
        )

    def prob(self, value: EventT) -> jax.Array:
        """See `Distribution.prob`."""
        return jnp.exp(self.log_prob(value))

    def log_cdf(self, value: EventT) -> jax.Array:
        return jnp.log(self.cdf(value))

    def cdf(self, value: EventT) -> jax.Array:
        raise NotImplementedError(
            f"{type(self).__name__} has no CDF implemented as the elements "
            f"have no ordering!"
        )

    def kl_divergence(
            self,
            other: Categorical,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if self.logits.shape != other.logits.shape:
            raise ValueError(
                f"KL-Divergence is undefined for {type(self)} with "
                f"different support!"
            )

        # Compute sum_i p(i) log(p(i) / q(i))
        log_probs = jax.nn.log_softmax(
            self.logits / jnp.clip(self.temperature, 1e-8)
        )
        log_probs_other = jax.nn.log_softmax(
            other.logits / jnp.clip(other.temperature, 1e-8)
        )
        probs = jnp.exp(log_probs)

        # Pre-compute KL-Divergences for all special cases.
        greedy_kl = jax.lax.select(
            self.logits.argmax() == other.logits.argmax(),
            0.0, -jnp.inf
        )  # self.T == other.T == 0
        uniform_kl = jnp.zeros(())  # self.T == other.T == inf
        uniform_to_normal = log_probs_other.mean()  # self.T == inf
        normal_to_uniform = probs.mean()  # other.T == inf
        greedy_to_normal = -log_probs_other.at[log_probs.argmax()].get()

        safe_delta = jnp.where(probs == 0, 0.0, log_probs - log_probs_other)
        normal_to_normal = (safe_delta * probs).sum()

        greedy_self = jnp.isclose(self.temperature, 0.0)
        greedy_other = jnp.isclose(other.temperature, 0.0)
        uniform_self = jnp.isinf(self.temperature)
        uniform_other = jnp.isinf(other.temperature)
        normal_self = ~greedy_self & ~uniform_self
        normal_other = ~greedy_other & ~uniform_other

        # Form all combinations to make a lookup-table
        cases = jnp.asarray([
            greedy_self & greedy_other,
            greedy_self & normal_other,
            greedy_self & uniform_other,

            normal_self & greedy_other,
            normal_self & normal_other,
            normal_self & uniform_other,

            uniform_self & greedy_other,
            uniform_self & normal_other,
            uniform_self & uniform_other
        ])
        divergences = jnp.asarray([
            greedy_kl,
            greedy_to_normal,
            greedy_to_normal,

            -jnp.inf,
            normal_to_normal,
            normal_to_uniform,

            -jnp.inf,
            uniform_to_normal,
            uniform_kl
        ])

        return divergences.at[cases.argmax()].get(), {}

    def entropy(self):

        # Prevent NaNs in gradients by clipping Temperature.
        log_probs = jax.nn.log_softmax(
            self.logits / jnp.clip(self.temperature, a_min=1e-8)
        )
        probs = jnp.exp(log_probs)

        # Filter out negative infinities in log_probs.
        return -(jnp.where(probs == 0, 0.0, log_probs) * probs).sum()
