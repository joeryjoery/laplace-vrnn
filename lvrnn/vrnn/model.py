from __future__ import annotations
from typing import Callable, Sequence, Literal, Type
from functools import partial

import flax.linen as nn

import jax
import jax.numpy as jnp

import numpy as np

from axme.core import Scope

import lvrnn.distributions as dist
from lvrnn.distributions import Distribution

from lvrnn.agent_model import (
    State, AfterState,
    StateAdapter, TransitionAdapter, PriorTransition,
    PosteriorPredictive, prefabs, FlaxRNNStateAdapter
)

from .interface import ModelModality


# Forward declaration for naming reference in partial
Predictive: type


class Predictive(nn.Module):

    layer_sizes: Sequence[int] = (128,)
    output_shape: Sequence[int] = (1,)
    activation: Callable = jax.nn.leaky_relu

    use_layer_norm: bool = True

    @partial(
        nn.vmap,
        variable_axes={Scope.Params: None},
        split_rngs={Scope.Params: False},
        in_axes=(0, None)
    )
    def decode(
            self,
            inputs: jax.Array,
            out_size: int | None = None
    ) -> jax.Array:
        """Deterministic mapping from inputs to outputs"""

        if self.use_layer_norm:
            inputs = nn.LayerNorm()(inputs)

        if out_size is None:
            out_size = np.prod(self.output_shape).astype(int)

        decoder = prefabs.MLP(
            tuple(self.layer_sizes) + (out_size,),
            activation=self.activation,
            activate_final=False
        )

        return decoder(inputs)


class GaussianPredictive(Predictive):

    predict_gaussian_variance: Literal[
        'model', 'bias', 'constant', 'standard'
    ] = 'model'
    log_variance_bounds: tuple[float, float] = (0.001, 5.0)

    # Example values:
    # -1.0: std = 0.3132617..., std = 1.0: jnp.log(jnp.e - 1) = 0.54132485
    log_variance_constant: float = -1.0

    @partial(
        nn.vmap,
        variable_axes={Scope.Params: None},
        split_rngs={Scope.Params: False},
        in_axes=(0, None)
    )
    def _bias_param(
            self,
            inputs: jax.Array,
            out_size: int | None = None
    ) -> jax.Array:
        return nn.Dense(
            out_size, name='predictive_variance'
        )(jnp.ones(1))

    @nn.compact
    def __call__(self, inputs: jax.Array) -> dist.SerializeTree[Distribution]:

        out_size = np.prod(self.output_shape).astype(int)
        if self.predict_gaussian_variance == 'model':
            out_size *= 2

        out = self.decode(inputs, out_size)

        if self.predict_gaussian_variance == 'model':
            # Input dependent variance
            mu, log_sigma = jnp.array_split(out, 2)

        elif self.predict_gaussian_variance == 'bias':
            # Input independent variance
            mu = out
            log_sigma = self._bias_param(inputs, out_size)

        elif self.predict_gaussian_variance == 'constant':
            # Constant iid variance
            mu, log_sigma = out, jnp.zeros_like(out)
        elif self.predict_gaussian_variance == 'standard':
            mu, sigma = out, jnp.ones_like(out)

            return dist.SerializeTree(
                dist.MultivariateNormalTriangular, mu, sigma,
                static_kwargs=dict(inverse=False, diagonal=True)
            )
        else:
            raise NotImplementedError(
                f"{self.predict_gaussian_variance} is not supported!"
            )

        # Constrain/ Transform standard-deviations for stability
        log_sigma = jnp.clip(
            log_sigma + self.log_variance_constant,
            *self.log_variance_bounds
        )
        sigma = jax.nn.softplus(log_sigma)

        return dist.SerializeTree(
            dist.MultivariateNormalTriangular, mu, sigma,
            static_kwargs=dict(inverse=False, diagonal=True)
        )


class BetaPredictive(Predictive):

    log_bounds: tuple[float, float] = (0.001, 5.0)
    log_constant: float = -1.0  # Constant to shift initial output

    @nn.compact
    def __call__(self, inputs: jax.Array) -> dist.SerializeTree[Distribution]:

        out_size = np.prod(self.output_shape).astype(int)
        out = self.decode(inputs, out_size * 2)

        out = jnp.clip(out + self.log_constant, *self.log_bounds)

        alpha, beta = jnp.array_split(out, 2)

        return dist.SerializeTree(
            dist.Beta, alpha, beta
        )


class CategoricalPredictive(Predictive):
    num_classes: int | None = None
    temperature: float = 1.0

    @nn.compact
    def __call__(self, inputs: jax.Array) -> dist.SerializeTree[Distribution]:
        if self.num_classes is None:
            raise RuntimeError("Number of output classes is unspecified!")

        out = self.decode(inputs, self.num_classes)

        return dist.SerializeTree(
            dist.Categorical, out,
            static_kwargs=dict(temperature=self.temperature)
        )


class DiracPredictive(Predictive):

    @nn.compact
    def __call__(self, inputs: jax.Array) -> dist.SerializeTree[Distribution]:
        return dist.SerializeTree(dist.Deterministic, self.decode(inputs))


def get_predictive_model(
        option: str,
        ensemble: str | None
) -> tuple[Type[Predictive], Callable]:
    """Pattern match input-specification to get the desired Class Types."""

    match option:
        case 'gaussian':
            model = GaussianPredictive
            aggregator = dist.ensemble.ParameterAggregation.mvn_triangular
        case 'categorical' | 'boltzmann':
            model = CategoricalPredictive
            aggregator = dist.ensemble.ParameterAggregation.categorical
        case 'dirac':
            model = DiracPredictive
            aggregator = dist.ensemble.ParameterAggregation.delta_to_mvn
        case 'beta':
            model = BetaPredictive
            aggregator = dist.ensemble.ParameterAggregation.beta
        case 'dirichlet':
            raise NotImplementedError("TODO Dirichlet Modality")  # TODO
            # aggregator = dist.ensemble.ParameterAggregation.dirichlet
        case 'truncnorm':
            raise NotImplementedError("TODO TruncNorm Modality")  # TODO
            # aggregator = dist.ensemble.ParameterAggregation.truncnorm
        case _:
            raise NotImplementedError(f'Wrong option: {option}')

    match ensemble:
        case "mixture": ensemble_method = dist.ensemble.Mixture
        case "aggregate": ensemble_method = aggregator
        case None: ensemble_method = None  # Only works for num_components = 1
        case _:
            raise NotImplementedError(f"Wrong ensemble: {ensemble}")

    return model, ensemble_method


class VariationalModel(nn.Module):
    layer_sizes: Sequence[int] = (128,)
    activation: Callable = jax.nn.leaky_relu

    modalities: list[ModelModality] | None = None

    def sample_hypothesis(self, distribution: Distribution, n: int):
        if n == 0:
            return jnp.expand_dims(distribution.mean(), axis=0), jnp.zeros(1)
        else:
            # Create Ensemble
            key = self.make_rng(type(self).__name__)
            return distribution.sample_and_log_prob(
                seed=key, sample_shape=(n, )
            )

    def compute_outputs(self, samples: jax.Array, log_probs: jax.Array):

        output_modalities = {}
        for modality in self.modalities:
            model_type, ensemble_type = get_predictive_model(
                modality.likelihood, modality.ensemble_method
            )

            model = model_type(
                layer_sizes=self.layer_sizes,
                activation=self.activation,
                output_shape=modality.spec.shape,
                **modality.likelihood_kwargs,
            )

            output_dist_stream: dist.SerializeTree = model(samples)

            if len(log_probs) > 1:
                # Combine multiple output-distributions into ensemble.

                weights = log_probs
                if modality.uniform_ensemble:
                    weights = jnp.ones(len(log_probs)) / len(log_probs)

                output_dist_stream = dist.SerializeTree(
                    ensemble_type, output_dist_stream, jnp.atleast_1d(weights)
                )
            else:
                # Remove 1D prefix.
                output_dist_stream = jax.tree_map(
                    lambda x: x.at[0].get(),
                    output_dist_stream
                )

            output_modalities[modality.name] = output_dist_stream

        return output_modalities


class VariationalPredicter(VariationalModel, PosteriorPredictive):
    core_adapter: StateAdapter = FlaxRNNStateAdapter

    log_probs_name: str = "model_log_probs"

    @nn.compact
    def __call__(
            self,
            s_t: State,
            n: int = 0
    ) -> dict[str, dist.Distribution | dist.SerializeTree[dist.Distribution]]:
        # dev-note: state = tuple[RLVMState, Serialized-dist]
        carry, out = self.core_adapter.split_transformable(s_t)

        s_hat, z_posterior = out
        samples, log_probs = self.sample_hypothesis(z_posterior.get, n)

        ins = samples
        if s_hat is not None:
            if n > 0:
                s_hat = jnp.broadcast_to(s_hat, (n, *s_hat.shape))
                ins = jnp.concatenate([samples, s_hat], -1)
            else:  # n == 0
                ins = jnp.concatenate([samples.ravel(), s_hat.ravel()], -1)
                ins = jnp.expand_dims(ins, 0)

        output_modalities = self.compute_outputs(ins, log_probs)

        return output_modalities


class VariationalSimulater(VariationalModel, PriorTransition):
    adapter: StateAdapter = TransitionAdapter
    core_adapter: StateAdapter = FlaxRNNStateAdapter

    log_probs_name: str = "model_log_probs"

    @nn.compact
    def __call__(
            self,
            as_t: AfterState,
            n: int = 0
    ) -> tuple[
        dict[str, dist.Distribution | dist.SerializeTree[dist.Distribution]],
        State
    ]:
        # dev-note: state = tuple[RLVMState, Serialized-dist]
        state, a_embed_t = self.adapter.split_transformable(as_t)
        carry, out = self.core_adapter.split_transformable(state)

        s_hat, z_prior = out
        samples, log_probs = self.sample_hypothesis(z_prior.get, n)

        ins = []
        if s_hat is not None:
            ins.append(s_hat)
        if a_embed_t is not None:
            ins.append(a_embed_t)

        if n == 0:
            ins = jnp.concatenate([samples.ravel()] + [x.ravel() for x in ins], -1)
            ins = jnp.expand_dims(ins, 0)
        else:
            ins = [jnp.broadcast_to(x, (n, *x.shape)) for x in ins]
            ins = jnp.concatenate([samples] + ins, -1)

        output_modalities = self.compute_outputs(ins, log_probs)

        # TODO: State update
        new_state = state

        return output_modalities, new_state
