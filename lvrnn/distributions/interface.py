from __future__ import annotations
from typing import TypeVar, Generic, Sequence, Callable
import abc

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

import numpy as np

# Distrax inspired API.
EventT = TypeVar("EventT")

DistT = TypeVar("DistT", bound="Distribution")


class Distribution(abc.ABC, Generic[EventT]):  # TODO: Make a Protocol

    @property
    def get(self) -> Distribution:
        return self

    @abc.abstractmethod
    def _sample_n(self, key: PRNGKeyArray, n: int) -> EventT:
        pass

    @property
    def name(self) -> str:
        """Distribution name."""
        return type(self).__name__

    # @property
    # @abc.abstractmethod
    # def batch_shape(self) -> Sequence[int]:
    #     pass

    @property
    @abc.abstractmethod
    def event_shape(self) -> Sequence[int]:
        pass

    @abc.abstractmethod
    def mean(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def mode(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def median(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def variance(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def log_prob(self, value: EventT) -> jax.Array:
        pass

    @abc.abstractmethod
    def log_cdf(self, value: EventT) -> jax.Array:
        pass

    @abc.abstractmethod
    def kl_divergence(
            self,
            other_dist: Distribution,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Enforce the function to return the KL + logging statistics.

        The KL-divergence can be a numerically unstable statistic, it is
        important that implementations log different intermediate points!
        """
        pass

    @abc.abstractmethod
    def entropy(self) -> jax.Array:
        pass

    def cross_entropy(
            self,
            other_dist: Distribution,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        kl, metrics = self.kl_divergence(other_dist, **kwargs)
        return kl + self.entropy(), metrics

    def prob(self, value: EventT) -> jax.Array:
        return jnp.exp(self.log_prob(value))

    def cdf(self, value: EventT) -> jax.Array:
        return jnp.exp(self.log_cdf(value))

    def stddev(self) -> EventT:
        return jnp.sqrt(self.variance())

    def sample(
            self,
            *,
            seed: PRNGKeyArray,
            sample_shape: Sequence[int] = ()
    ) -> EventT:
        samples = self._sample_n(seed, int(np.prod(sample_shape)))
        return samples.reshape((*sample_shape, *samples.shape[1:]))

    def sample_and_log_prob(
            self,
            seed: PRNGKeyArray,
            sample_shape: Sequence[int] = ()
    ):
        samples = self.sample(seed=seed, sample_shape=sample_shape)

        if len(sample_shape):
            f = self.log_prob
            for _ in range(len(sample_shape)):
                f = jax.vmap(f)
            log_probs = f(samples)
        else:
            log_probs = self.log_prob(samples)
        return samples, log_probs


class MonteCarloEntropyMixin:
    """Mixin to allow MC-estimation of Entropy or KL-divergences"""

    # Type Alias
    sample_and_log_prob: Callable[[PRNGKeyArray, Sequence[int]], jax.Array]

    def kl_divergence(
            self,
            other: Distribution,
            *,
            seed: PRNGKeyArray,
            num_samples: int | None = None,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if not num_samples:
            raise NotImplementedError(
                "KL-Div has no closed form. Approximate it through sampling."
            )

        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        samples, log_probs = self.sample_and_log_prob(seed, (num_samples, ))
        log_probs_other = jax.vmap(other.log_prob)(samples)
        return jnp.mean(log_probs - log_probs_other).mean(axis=0), {}

    def entropy(
            self,
            *,
            seed: PRNGKeyArray,
            num_samples: int | None = None
    ) -> jax.Array:
        if not num_samples:
            raise NotImplementedError(
                "Entropy has no closed form. Approximate it through sampling."
            )
        # https://en.wikipedia.org/wiki/Entropy_(information_theory)
        _, log_probs = self.sample_and_log_prob(seed, (num_samples, ))
        return -log_probs.mean(axis=0)

