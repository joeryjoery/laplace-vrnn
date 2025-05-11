from __future__ import annotations
from typing import Generic, Sequence, Callable
from functools import partial

import jax
import jax.numpy as jnp

import numpy as np

from jaxtyping import PRNGKeyArray

from ..interface import Distribution, DistT, EventT, MonteCarloEntropyMixin
from .. import mvn, Deterministic, Categorical, Beta
from .. import SerializeTree


class ParameterAggregation:
    """Function namespace to transform batches of distributions into singular
    distributions."""

    @staticmethod
    def mvn_exp_orthogonal(
            stream: SerializeTree[mvn.MultivariateNormalExpOrthogonal],
            weights: jax.Array
    ) -> mvn.MultivariateNormalExpOrthogonal:
        raise NotImplementedError()

    @staticmethod
    def delta_to_mvn(
            stream: SerializeTree[Deterministic],
            weights: jax.Array
    ) -> mvn.MultivariateNormalTriangular:

        bound = jnp.clip(weights.min(), a_max=0)
        weights = (weights + bound) / (weights + bound).sum()

        def extract_params(
                serialized: SerializeTree[Deterministic]
        ) -> jax.Array:
            return serialized.get.loc

        means = jax.vmap(extract_params)(stream)
        new_mean = jax.vmap(jnp.multiply)(weights, means).sum(axis=0)

        errors = jax.vmap(jnp.multiply)(
            weights, jnp.square(means - new_mean[None, ...])
        )
        scale = jnp.sqrt(jnp.sum(errors, axis=0))

        return mvn.MultivariateNormalTriangular(
            new_mean, scale, inverse=False, diagonal=True
        )

    @staticmethod
    def mvn_triangular(
            stream: SerializeTree[mvn.MultivariateNormalTriangular],
            weights: jax.Array,
            jitter: float = 1e-6
    ) -> mvn.MultivariateNormalTriangular:
        # Result is equivalent to SampleAggregate due to linearity of Gaussian

        bound = jnp.clip(weights.min(), a_max=0)
        weights = (weights + bound) / (weights + bound).sum()

        def extract_params(
                serialized: SerializeTree[mvn.MultivariateNormalTriangular]
        ) -> tuple[jax.Array, jax.Array, bool, bool]:
            dist = serialized.get

            # Prevent unnecessary inversion
            if dist.is_diag:
                _scale = dist.variance()
            else:
                _scale = dist.precision() if dist.inverted else dist.covariance()

            return dist.mean(), _scale, dist.inverted, dist.is_diag

        means, scales, is_inverted, is_diag = jax.vmap(
            extract_params,
            out_axes=(0, 0, None, None)
        )(stream)

        mean = jax.vmap(jnp.multiply)(weights, means).sum(axis=0)
        scale = jax.vmap(jnp.multiply)(weights, scales).sum(axis=0)

        if is_diag:
            # Automatically converts precisions to standard deviations.
            return mvn.MultivariateNormalTriangular(
                mean, scale, diagonal=True, inverse=False
            )

        scale = jnp.linalg.cholesky(scale + jitter * jnp.eye(len(scale)))

        return mvn.MultivariateNormalTriangular(
            mean, scale, diagonal=False, inverse=is_inverted
        )

    @staticmethod
    def categorical(
            stream: SerializeTree[Categorical],
            weights: jax.Array
    ) -> Categorical:

        bound = jnp.clip(weights.min(), a_max=0)
        weights = (weights + bound) / (weights + bound).sum()

        def extract_params(
                serialized: SerializeTree[Categorical]
        ) -> jax.Array:
            dist = serialized.get

            if np.isclose(dist.temperature, 0.0):
                return jnp.clip(
                    jnp.log(dist.logits == dist.logits.max()),
                    a_min=-1e8
                )
            if np.isinf(dist.temperature):
                return dist.logits * 0

            return jax.nn.log_softmax(dist.logits / dist.temperature)

        logit_batch = jax.vmap(extract_params)(stream)
        logits = jax.vmap(jnp.multiply)(weights, logit_batch).sum(axis=0)

        return Categorical(logits, **stream.statics[1])

    @staticmethod
    def beta(
            stream: SerializeTree[Beta],
            weights: jax.Array
    ) -> Beta:

        bound = jnp.clip(weights.min(), a_max=0)
        weights = (weights + bound) / (weights + bound).sum()

        def extract_params(
                serialized: SerializeTree[Beta]
        ) -> tuple[jax.Array, jax.Array]:
            dist = serialized.get
            return dist.alpha, dist.beta

        a, b = jax.vmap(extract_params)(stream)

        a = jax.vmap(jnp.multiply)(weights, a).sum(axis=0)
        b = jax.vmap(jnp.multiply)(weights, b).sum(axis=0)

        return Beta(a, b)


class SampleAggregation(MonteCarloEntropyMixin, Distribution, Generic[DistT]):
    # TODO: Not done; Don't use
    """Forms an ensemble by linearly aggregating all member-predictions.

    Note that this class does not check validity of aggregation for the
    member methods (like mean, variance, mode, and median). For a Gaussian,
    this implementation is fine to use, but for more complex distributions
    there may be errors.
    """

    def __init__(
            self,
            dist_stream: SerializeTree[DistT],  # Batched Distribution
            weights: jax.Array  # Must be batch-compatible
    ):
        super().__init__()
        self.dist = dist_stream.cls_type
        self.dist_stream = dist_stream

        # L1 normalize weights to get valid probabilities.
        bound = jnp.clip(weights.min(), a_max=0)
        self.weights = (weights + bound) / (weights + bound).sum()

        # TODO: Develop further?
        raise NotImplementedError(
            "Don't use this class. Opt for Mixture or Param-Aggregate instead")

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

    def _sample_n(self, key: PRNGKeyArray, n: int) -> jax.Array:
        key_per_component = jax.random.split(key, self.weights.size)

        sampling_fun = self._to_batch_fun(self.dist.sample, sample_shape=(n,))
        samples = sampling_fun(key_per_component)

        return jax.vmap(jnp.multiply)(self.weights, samples).sum(axis=0)

    @property
    def event_shape(self) -> Sequence[int]:
        return jax.vmap(lambda d: d.get.event_shape)(self.dist_stream)[0]

    def mean(self) -> jax.Array:
        # E[aX + bY] = aE[X] + bE[Y]
        means = self._to_batch_fun(self.dist.mean)()
        return jax.vmap(jnp.multiply)(self.weights, means).sum(axis=0)

    def mode(self) -> jax.Array:
        # Probably Wrong
        modes = self._to_batch_fun(self.dist.mode)()
        return jax.vmap(jnp.multiply)(self.weights, modes).sum(axis=0)

    def median(self) -> jax.Array:
        # Probably Wrong
        medians = self._to_batch_fun(self.dist.median)()
        return jax.vmap(jnp.multiply)(self.weights, medians).sum(axis=0)

    def variance(self) -> jax.Array:
        # Var(aX + bY) = aVar(X) + bVar(Y)
        variances = self._to_batch_fun(self.dist.variance)()
        return jax.vmap(jnp.multiply)(self.weights, variances).sum(axis=0)

    def log_prob(self, value: EventT) -> jax.Array:
        log_probs = self._to_batch_fun(self.dist.log_prob, value)()
        return jax.vmap(jnp.multiply)(self.weights, log_probs).sum(axis=0)

    def log_cdf(self, value: EventT) -> jax.Array:
        log_cdfs = self._to_batch_fun(self.dist.log_cdf, value)()
        return jax.vmap(jnp.multiply)(self.weights, log_cdfs).sum(axis=0)
