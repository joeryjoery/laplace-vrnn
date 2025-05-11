from __future__ import annotations
from typing import Sequence, Callable, Any, TYPE_CHECKING
from functools import partial

import jit_env
import wandb

import jax
import jax.numpy as jnp

import flax

from jaxtyping import PRNGKeyArray

from axme.core import Variables, Scope

from lvrnn.agent_model import AgentModel, State
from lvrnn.agents import Scope as AgentScope

from . import bandits
from .plotting import plot_discrete_bandit

from .._utils import AgentModelUtility, EnvEvaluator

if TYPE_CHECKING:
    from dataclasses import dataclass as container
else:
    from chex import dataclass as _c
    container = _c(frozen=False)


@container
class ComputeMetrics:
    model: AgentModel
    utility: AgentModelUtility

    resolution: int

    def sample_evaluate(
            self,
            model_params: Variables,
            key: PRNGKeyArray,
            states: jit_env.State,
            steps: jit_env.TimeStep,
            actions: jit_env.Action,
            num_ensemble: tuple[int],
            return_model_metrics: bool
    ):
        key_state, key_sample, key_sim = jax.random.split(key, 3)

        model_states, metrics = self.utility.get_states(
            model_params, key_state, actions, steps.observation,
            return_metrics=return_model_metrics, initial_state=False
        )

        def compute_predictive_metrics(
                _key: PRNGKeyArray,
                _state: State,
                _action: jit_env.Action,
                _n: int
        ):
            _key_out, _key_sample = jax.random.split(_key)
            predictive = self.utility.get_predictives(
                model_params, _key_out, _state, _n
            )

            action_distribution = predictive.get(AgentScope.Policy).get

            log_prob_actual = action_distribution.log_prob(_action)

            _, log_probs = action_distribution.sample_and_log_prob(
                seed=_key_sample, sample_shape=(self.resolution,)
            )
            entropy = -log_probs.mean()

            return {
                'log_prob_action': log_prob_actual,
                'entropy': entropy,
                'log_prob_mode': log_probs.max()
            }

        actions = jax.tree_map(lambda x: x[1:], actions)
        if self.utility.sub_samples:
            actions = jax.tree_map(
                lambda x: x.at[jnp.asarray(self.utility.sub_samples)].get(),
                actions
            )

        rng_batch_shape = jax.tree_leaves(model_states)[0].shape[0]
        predict_rngs = jax.random.split(key_sample, rng_batch_shape)

        # Compute Predictive Metrics over batched states/ keys
        predictive_metrics = dict()
        for n in num_ensemble:
            member_metrics = jax.vmap(
                compute_predictive_metrics, in_axes=(0, 0, 0, None)
            )(predict_rngs, model_states, actions, n)

            predictive_metrics |= {
                f'ensemble-{n} {k}': v for k, v in member_metrics.items()
            }

        simulations = None
        if self.model.simulate:

            rng_batch_shape = jax.tree_leaves(model_states)[0].shape[0]
            predict_rngs = jax.random.split(key_sim, rng_batch_shape)

            # Compute Predictions: Batch States + Keys, then Batch Test-Inputs.
            # Keep keys fixed over Actions! -> Draw 1 hypothesis, evaluate inputs.
            simulate_fun = partial(
                self.utility.get_simulations,
                model_params, n=max(num_ensemble)
            )
            simulations = jax.vmap(
                jax.vmap(simulate_fun, in_axes=(None, None, 0)),
                in_axes=(0, 0, None)
            )(predict_rngs, model_states, jnp.arange(0, states.ps.shape[-1]))

        if return_model_metrics:
            return metrics | predictive_metrics, simulations
        return predictive_metrics, simulations

    def evaluate(
            self,
            model_params: Variables,
            key: PRNGKeyArray,
            states: jit_env.State,
            steps: jit_env.TimeStep,
            actions: jit_env.Action,
            num_ensemble: tuple[int],
            return_model_metrics: bool
    ):
        rng_batch = jax.random.split(key, len(steps.step_type))

        # Use lax.map over vmap to reduce memory requirements!!
        return jax.lax.map(
            lambda args: self.sample_evaluate(
                model_params, *args,
                num_ensemble=num_ensemble,
                return_model_metrics=return_model_metrics
            ), (rng_batch, states, steps, actions)
        )


@container
class MultinoulliEvaluator(EnvEvaluator[bandits.Multinoulli]):
    sample_lengths: Sequence[int]  # Episode Lengths for metrics
    plot_sample_lengths: Sequence[int]

    num_samples: int  # Number of Episodes/ Datasets
    num_predictions: list[int]  # Number of ensemble-members per TimeStep.
    resolution: int  # Number of samples to estimate entropies/ modes.

    num_predictions_plot: int  # Number of ensemble-members for plots only.
    num_plots: int  # Number of Trajectories to plot

    metric_fun: Callable[..., ...] | None = None
    plot_metric_fun: Callable[..., ...] | None = None

    sampling_fun: Callable[..., ...] | None = None
    sampling_batch_fun: Callable[..., ...] | None = None

    def __post_init__(self):
        self.sampling_fun = jax.jit(self._trajectory_fun, static_argnums=3)
        self.sampling_batch_fun = jax.jit(
            jax.vmap(
                self._trajectory_fun, in_axes=(0, None, None, None)
            ), static_argnums=3
        )

        metrics_util = ComputeMetrics(
            model=self.model,
            utility=AgentModelUtility(self.model, self.sample_lengths),
            resolution=self.resolution
        )

        self.metric_fun = jax.jit(
            partial(metrics_util.evaluate, return_model_metrics=True),
            static_argnums=5
        )

        if self.model.simulate:
            visualize_util = ComputeMetrics(
                model=self.model,
                utility=AgentModelUtility(
                    self.model, self.plot_sample_lengths
                ),
                resolution=self.resolution
            )

            self.plot_metric_fun = jax.jit(
                partial(
                    visualize_util.sample_evaluate,
                    return_model_metrics=False
                ),
                static_argnums=5
            )

    @property
    def name(self) -> str:
        return '.'.join([self.__class__.__name__,  self.task.name])

    def _generate_metrics(  # do not jit
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, jax.Array]:

        # Freeze randomness to get a deterministic function of the variables.
        states, steps, actions = self.sampling_batch_fun(
            jax.random.split(self.fixed_seed, num=self.num_samples),
            variables, None, max(self.sample_lengths)
        )

        # Trees of dimensionality: (batch, samples)
        metrics, predictions = self.metric_fun(
            variables, key, states, steps, actions,
            tuple(self.num_predictions)
        )

        simulate_metrics = {}
        if predictions is not None:
            # Calculate cross-entropy
            def log_prob(true, predict):
                return predict.get.log_prob(true)

            if self.sample_lengths:
                states = jax.tree_map(
                    lambda x: x.at[:, jnp.asarray(self.sample_lengths)].get(),
                    states
                )

            log_probs = jax.vmap(jax.vmap(jax.vmap(log_prob)))(
                states.ps, predictions.get(AgentScope.Reward)
            )
            cross_entropy = -log_probs.mean(axis=-1)
            simulate_metrics = {
                'cross_entropy': cross_entropy,
                'perplexity': jnp.exp(cross_entropy)
            }

        # Formatting of flax.linen.sow values.
        intermediates = metrics.pop(Scope.Intermediates)
        flattened = flax.traverse_util.flatten_dict(intermediates, sep=' ')
        formatted = {
            k.split(' ')[-1].split('/')[0]: v[0]
            for k, v in flattened.items()
        }
        model_metrics_formatted = flax.traverse_util.flatten_dict(
            formatted, sep='_'
        )

        # Extract test-rewards/ regrets
        regret = steps.extras.get(bandits.Multinoulli._REGRET_KEY)
        min_max = steps.extras.get(bandits.Multinoulli._REGRET_MINMAX_KEY)

        cum_rewards = jnp.cumsum(steps.reward, axis=-1)
        cum_regret = jnp.cumsum(regret, axis=-1)
        cum_minmax = jnp.cumsum(min_max, axis=-1)

        best_reward = jax.lax.associative_scan(
            jnp.maximum, steps.reward, axis=-1
        )
        simple_regret = jax.lax.associative_scan(jnp.minimum, regret, axis=-1)
        simple_minmax = jax.lax.associative_scan(jnp.minimum, min_max, axis=-1)

        reward_metrics = {
            'cumulative regret': cum_regret,
            'cumulative regret min-max': cum_minmax,
            'cumulative reward': cum_rewards,
            'simple regret': simple_regret,
            'simple regret min-max': simple_minmax,
            'best reward': best_reward
        }
        reward_metrics = jax.tree_map(
            lambda x: x.at[:, jnp.asarray(self.sample_lengths)].get(),
            reward_metrics
        )

        combined = (
                metrics | model_metrics_formatted |
                reward_metrics | simulate_metrics
        )

        # Procedurally aggregate over all test-cases (i.e., batch-wise).
        statistics = [
            (jnp.mean, 'E {}'), (partial(jnp.var, ddof=1), 'Var {}'),
            (jnp.median, 'Med {}'),
            (partial(jnp.percentile, q=25), 'LQ {}'),
            (partial(jnp.percentile, q=75), 'UQ {}'),
            (jnp.min, 'Min {}'), (jnp.max, 'Max {}'),
        ]
        data = {
            name.format(k): jax.tree_map(partial(fun, axis=0), v)
            for k, v in combined.items() for fun, name in statistics
        }

        nB, nS = self.num_samples, len(self.sample_lengths)
        data_sizes = {
            'batch_size': jnp.asarray([nB] * nS),
            'time': jnp.asarray(self.sample_lengths)
        }

        # Return a large dictionary of aggregated performance metrics
        return data | data_sizes

    def agent_metrics(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, wandb.Table]:

        # Get a dataset of distributional statistics of the performance
        # and model-metrics for the given `train_state`.
        # Statistics describe the distribution over the test-cases, the
        # quantities per test-case are averaged over all predictions.
        data = jax.jit(self._generate_metrics)(key, variables)

        keys, values = list(zip(*data.items()))
        table_values = [v.tolist() for v in values]
        transposed = [list(i) for i in zip(*table_values)]

        return {'table': wandb.Table(columns=list(keys), data=transposed)}

    def sample_plots(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, dict[str, Any]]:

        if not self.model.simulate:
            return {}

        # Freeze environments for fair comparison across parameters.
        data_keys = jax.random.split(self.fixed_seed, num=self.num_plots)
        metrics_keys = jax.random.split(key, num=self.num_plots)

        data = dict()
        for i, (a, b) in enumerate(zip(data_keys, metrics_keys)):

            states, steps, actions = self.sampling_fun(
                a, variables, None,
                max(self.plot_sample_lengths)
            )

            _, predictions = self.plot_metric_fun(
                variables, b, states, steps, actions,
                (self.num_predictions_plot, )
            )

            figure = plot_discrete_bandit(
                states, steps.reward, actions,
                predictions.get(AgentScope.Reward),
                self.plot_sample_lengths
            )

            data[f'Bandit-{i+1}'] = wandb.Image(figure)

        return {'plots': data}
