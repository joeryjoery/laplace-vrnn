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
from axme.data import environment

from lvrnn.agent_model import AgentModel, State
from lvrnn.agents import Scope as AgentScope

from . import gridworld
from .plotting import plot_agent

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
            steps: jit_env.TimeStep,
            actions: jit_env.Action,
            num_ensemble: tuple[int]
    ):
        key_state, key_sample = jax.random.split(key)

        model_states, metrics = self.utility.get_states(
            model_params, key_state, actions, steps.observation,
            return_metrics=True, initial_state=False
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

        return metrics | predictive_metrics

    def evaluate(
            self,
            model_params: Variables,
            key: PRNGKeyArray,
            steps: jit_env.TimeStep,
            actions: jit_env.Action,
            num_ensemble: tuple[int]
    ):
        rng_batch = jax.random.split(key, len(steps.step_type))

        # Use lax.map over vmap to reduce memory requirements!!
        return jax.lax.map(
            lambda args: self.sample_evaluate(
                model_params, *args, num_ensemble=num_ensemble
            ), (rng_batch, steps, actions)
        )


@container
class GridworldEvaluator(EnvEvaluator[gridworld.SquareGrid]):
    sample_lengths: Sequence[int]  # Episode Lengths for metrics

    num_samples: int  # Number of Episodes/ Datasets
    num_predictions: list[int]  # Number of ensemble-members per TimeStep.
    resolution: int  # Number of samples to estimate entropies/ modes.

    plot_length: int  # Episode Length for plots
    plot_num_episodes: int  # Max number of episodes to plot
    plot_options: list[dict[str, Any]]  # Env Options for episode-plotting

    metric_fun: Callable[..., ...] | None = None
    sampling_fun: Callable[..., ...] | None = None
    sampling_batch_fun: Callable[..., ...] | None = None

    def __post_init__(self):
        metrics_util = ComputeMetrics(
            model=self.model,
            utility=AgentModelUtility(self.model, self.sample_lengths),
            resolution=self.resolution
        )
        self.metric_fun = jax.jit(metrics_util.evaluate, static_argnums=4)

        self.sampling_fun = jax.jit(self._trajectory_fun, static_argnums=3)
        self.sampling_batch_fun = jax.jit(
            jax.vmap(
                self._trajectory_fun, in_axes=(0, None, None, None)
            ), static_argnums=3
        )

    @property
    def name(self) -> str:
        return '.'.join([self.__class__.__name__,  self.task.name])

    def _generate_metrics(
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
        metrics = self.metric_fun(
            variables, key, steps, actions, tuple(self.num_predictions)
        )

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
        regret = steps.extras.get(gridworld.SquareGrid._REGRET_KEY)

        cum_rewards = jnp.cumsum(steps.reward, axis=-1)
        cum_regret = jnp.cumsum(regret, axis=-1)

        best_reward = jax.lax.associative_scan(
            jnp.maximum, steps.reward, axis=-1
        )
        simple_regret = jax.lax.associative_scan(jnp.minimum, regret, axis=-1)

        reward_metrics = {
            'cumulative regret': cum_regret,
            'cumulative reward': cum_rewards,
            'simple regret': simple_regret,
            'best reward': best_reward
        }

        combined = metrics | model_metrics_formatted | reward_metrics

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

        # Freeze environments for fair comparison across parameters.
        keys = jax.random.split(self.fixed_seed, num=len(self.plot_options))

        data = dict()
        for i, (rng, options) in enumerate(zip(keys, self.plot_options)):

            states, steps, _ = self.sampling_fun(
                rng, variables, {k: tuple(v) for k, v in options.items()},
                self.plot_length
            )

            # Regenerate (nicer) observations agnostic to the policy.
            observations = jax.vmap(
                partial(
                    self.task.env.make_observation,
                    observe_goal=True, observe_start=False,
                    one_hot_encoding=False
                )
            )(states)

            figure = plot_agent(
                states, observations, steps.reward,
                max_episodes=self.plot_num_episodes
            )

            data[f'Grid-{i+1}'] = wandb.Image(figure)

        return {'plots': data}
