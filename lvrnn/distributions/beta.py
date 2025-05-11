"""Re-implementations of parameterizing MultiVariate Normal distributions

Code is inspired by Distrax, but implemented differently due to missing
features, compatibility issues, or numerical optimizations.
"""
from __future__ import annotations
from typing import Sequence
import abc

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

import numpy as np

from .interface import Distribution, EventT


class Beta(Distribution):
    """Base distribution for MultivariateNormal extensions"""

    def __init__(
            self,
            alpha: jax.Array,
            beta: jax.Array,
            jitter: float = 1e-4
    ):
        self.alpha = alpha
        self.beta = beta
        self.jitter = jitter

    @property
    def event_shape(self) -> Sequence[int]:
        return self.alpha.shape

    def mean(self) -> jax.Array:
        return self.alpha / (self.alpha + self.beta)

    def mode(self) -> jax.Array:
        # Does not cover the bimodal case of alpha, beta < 1 (becomes zero)
        gt1 = (self.alpha - 1) / (self.alpha + self.beta - 2)

        out = (
            ((self.alpha > 1.0) & (self.beta > 1.0)) * gt1 +
            (jnp.isclose(self.alpha, 1.) & jnp.isclose(self.beta, 1.)) * 0.5 +
            ((self.alpha > 1.0) & (self.beta <= 1.0)) * 1.0
            # beta > 1 & alpha <= 1.0 -> = 0 default. As is alpha, beta < 0
        )

        return out

    def median(self) -> jax.Array:
        return self.mean()  # TODO

    def variance(self) -> jax.Array:
        num = self.alpha * self.beta
        den = jnp.square(self.alpha + self.beta) * (self.alpha + self.beta + 1)
        return num / den

    def _sample_n(self, key: PRNGKeyArray, n: int) -> EventT:
        return jax.random.beta(key, self.alpha, self.beta, (n, ))

    def log_prob(self, value: EventT) -> jax.Array:
        # loc/scale are min-max bounded to [0 - C, 1 + C] to prevent infs
        return jax.scipy.stats.beta.logpdf(
            value, self.alpha, self.beta,
            scale=1.0 + self.jitter, loc=-self.jitter/2.0
        )

    def log_cdf(self, value: EventT) -> jax.Array:
        # loc/scale are min-max bounded to [0 - C, 1 + C] to prevent infs
        return jax.scipy.stats.beta.logcdf(
            value, self.alpha, self.beta,
            scale=1.0 + self.jitter, loc=-self.jitter/2.0
        )

    def kl_divergence(
            self,
            other: Beta,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        # https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
        digamma = jax.scipy.special.digamma

        lnB = jax.scipy.special.betaln(self.alpha, self.beta)
        lnBother = jax.scipy.special.betaln(other.alpha, other.beta)

        diga = (self.alpha - other.alpha) * digamma(self.alpha)
        digb = (self.beta - other.beta) * digamma(self.beta)
        digab = (other.alpha - self.alpha + other.beta - self.beta) * digamma(
            self.alpha + self.beta
        )

        return lnB -lnBother + diga + digb + digab

    def entropy(self) -> jax.Array:
        # https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
        digamma = jax.scipy.special.digamma

        lnB = jax.scipy.special.betaln(self.alpha, self.beta)
        diga = (self.alpha - 1) * digamma(self.alpha)
        digb = (self.beta - 1) * digamma(self.beta)
        digab = (self.alpha + self.beta - 2) * digamma(self.alpha + self.beta)

        return lnB - diga - digb + digab
