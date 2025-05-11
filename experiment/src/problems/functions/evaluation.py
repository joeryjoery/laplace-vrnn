from __future__ import annotations
from typing import Sequence, Callable, Any, TYPE_CHECKING
from functools import partial
import io

import wandb
from PIL import Image

import numpy as np

import jax
import jax.numpy as jnp

import flax

from jaxtyping import PRNGKeyArray

from axme.core import Variables, Scope
from axme.data import supervised

from lvrnn.agent_model import AgentModel

from .._utils import AgentModelUtility, nest_vmap
from .plotting import plot_1D_function_predictive, plot_2D_function_predictive

if TYPE_CHECKING:
    from dataclasses import dataclass as container
else:
    from chex import dataclass as _c
    container = _c(frozen=False)


@container
class SupervisedSampler:
    model: AgentModel
    task: supervised.Regression

    utility: AgentModelUtility

    test_domain: Sequence[tuple[jax.Array, jax.Array]]
    data_shape: tuple[int, int, int]
    resolution: int

    randomize: bool = False  # Option: Monte-Carlo or linspace inputs

    @classmethod
    def make(
            cls,
            model: AgentModel,
            task: supervised.Regression,
            data_shape: tuple[int, int, int],
            resolution: int,
            test_domain: Sequence[tuple[jax.Array, jax.Array]],
            sample_lengths: Sequence[int] | None = None,
            randomize: bool = False
    ):
        return cls(
            model=model,
            task=task,
            utility=AgentModelUtility(model, sample_lengths),
            test_domain=test_domain,
            data_shape=data_shape,
            resolution=resolution,
            randomize=randomize
        )

    def _get_test_points(self, key: PRNGKeyArray, testing: bool) -> jax.Array:
        # Generate Test Queries and Labels

        low_iid, high_iid = self.task.bounds  # Project to default domain
        low_ood, high_ood = self.test_domain[0][0], self.test_domain[-1][-1]

        low = jax.lax.select(testing, low_ood, low_iid)
        high = jax.lax.select(testing, high_ood, high_iid)

        low = jnp.broadcast_to(low, self.task.shape)
        high = jnp.broadcast_to(high, self.task.shape)

        if self.randomize:
            test_xs = jax.random.uniform(
                key, (self.resolution, *self.task.shape)
            )
            test_xs = test_xs * (high - low) + low
        else:
            size = np.prod(self.task.shape)
            test_xs = jnp.linspace(low, high, self.resolution)
            if size == 2:
                mesh = jnp.meshgrid(*test_xs.T)
                test_xs = jnp.asarray(mesh).reshape(2, -1).T  # (n, 2)
            else:
                raise RuntimeError("Grid Evaluation is only for <= 2D!")

        return test_xs

    def _get_observation_points(
            self, key: PRNGKeyArray, n: int, testing: bool
    ) -> jax.Array:
        sample, shuffle = jax.random.split(key)

        in_xs = jax.random.uniform(sample, (n,))

        # Project in_xs to default domain
        low, high = self.task.bounds
        result_iid = in_xs * (high - low) + low

        # Evenly project inputs to manually specified domains.
        mask = jnp.linspace(0, len(self.test_domain), n, dtype=jnp.int32)
        bounds = jnp.asarray(self.test_domain).at[mask].get()

        result = in_xs * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        # Shuffle the elements between chunks
        result_ood = jax.random.permutation(
            shuffle, result, -1, independent=True
        )

        return jax.lax.select(testing, result_ood, result_iid)

    def sample_evaluate(
            self,
            model_params: Variables,
            data_rng: PRNGKeyArray,
            model_rng: PRNGKeyArray,
            test_xs: jax.Array,
            num_samples: int = 0,
            testing: bool = True,
            return_metrics: bool = True
    ):
        key_fun, key_test, key_burnin = jax.random.split(data_rng, num=3)
        key_obs, key_sim = jax.random.split(model_rng)

        # Initialize Data-Generator
        repetitions, batch_size, samples = self.data_shape
        params: Variables = self.task.function.init(key_fun, jnp.zeros(()))

        # Test-Data: Batch Test-Inputs over the Function.
        test_ys = jax.vmap(
            self.task.function.apply, in_axes=(None, 0)
        )(params, test_xs)  # TODO: RNGs?

        # Burn-In-Observations: Get random Inputs and Batch over the Function.
        burn_in_xs = self._get_observation_points(key_burnin, samples, testing)

        burn_in_ys = jax.vmap(
            self.task.function.apply, in_axes=(None, 0)
        )(params, burn_in_xs)  # TODO: RNGs

        # Unroll the model on Observations
        model_states, metrics = self.utility.get_states(
            model_params,
            key_obs,
            burn_in_xs,
            burn_in_ys,
            return_metrics=return_metrics, initial_state=True
        )

        # Batch Prediction function for States and Actions.
        predict_fun = partial(
            self.utility.get_simulations,
            model_params, n=num_samples
        )

        rng_batch_shape = jax.tree_leaves(model_states)[0].shape[0]
        predict_rngs = jax.random.split(key_sim, rng_batch_shape)

        # Compute Predictions: Batch States + Keys, then Batch Test-Inputs.
        # Keep keys fixed over Actions! -> Draw 1 hypothesis, evaluate inputs.
        predictions = jax.vmap(
            jax.vmap(predict_fun, in_axes=(None, None, 0)),
            in_axes=(0, 0, None)
        )(
            predict_rngs,
            model_states,
            test_xs
        )

        if return_metrics:
            return (
                (test_xs, test_ys), (burn_in_xs, burn_in_ys), predictions
            ), metrics

        return (test_xs, test_ys), (burn_in_xs, burn_in_ys), predictions

    def evaluate(
            self,
            test_rng: PRNGKeyArray,
            rng: PRNGKeyArray,
            model_params: Variables,
            num_samples: int = 0,
            testing: bool = True,
            return_metrics: bool = True
    ):
        repetitions, batch_size, samples = self.data_shape

        test_input, test_data = jax.random.split(test_rng)
        data_rngs = jax.random.split(test_rng, repetitions * batch_size)
        model_rngs = jax.random.split(rng, repetitions * batch_size)

        # Ensure that the test-input points are the same for each sample
        ood_xs = self._get_test_points(test_input, testing=True)
        iid_xs = self._get_test_points(test_input, testing=False)

        test_xs = jax.lax.select(testing, ood_xs, iid_xs)

        # Use lax.map instead of vmap to reduce memory requirements
        out = jax.lax.map(
            lambda args: self.sample_evaluate(
                model_params, *args,
                num_samples=num_samples,
                test_xs=test_xs,
                testing=testing, return_metrics=return_metrics
            ), (data_rngs, model_rngs)
        )

        metrics = None
        if return_metrics:
            out, metrics = out

        (_, test_ys), burn_ins, model_predictions = out

        if metrics:
            return ((test_xs, test_ys), burn_ins, model_predictions), metrics
        return (test_xs, test_ys), burn_ins, model_predictions


@container
class SupervisedEvaluator:
    model: AgentModel
    task: supervised.Regression
    fixed_seed: PRNGKeyArray  # Static seed for sampling problems

    sample_lengths: Sequence[int]  # Episode Lengths for metrics
    plot_sample_lengths: Sequence[int]  # Episode Lengths for plots

    num_samples: int  # Number of Episodes/ Datasets
    num_predictions: int  # Number of predictions per timestep

    randomize: bool  # False: evaluate on grid, True: evaluate Monte-Carlo
    resolution: int  # Number of inputs to evaluate predictive on.
    test_domain: Sequence[tuple[jax.Array, jax.Array]]

    num_plots: int | None = None  # Number of Trajectories to plot

    _eval_fun: Callable[..., ...] | None = None
    _sampling_fun: Callable[..., ...] | None = None

    def __post_init__(self):
        size = np.prod(self.task.shape)

        test_suite_loss = SupervisedSampler.make(
            self.model,
            self.task,
            (1, self.num_samples, max(self.sample_lengths)),
            resolution=self.resolution,
            test_domain=self.test_domain,
            sample_lengths=self.sample_lengths,
            randomize=self.randomize or (size > 2)
        )
        self._eval_fun = jax.jit(
            partial(test_suite_loss.evaluate, return_metrics=True),
            static_argnums=3
        )

        if size <= 2:
            test_suite_visualize = SupervisedSampler.make(
                self.model,
                self.task,
                (1, self.num_plots, max(self.plot_sample_lengths)),
                resolution=self.resolution,
                test_domain=self.test_domain,
                sample_lengths=self.plot_sample_lengths,
                randomize=self.randomize or (size > 2)
            )
            self._sampling_fun = jax.jit(
                partial(test_suite_visualize.evaluate, return_metrics=False),
                static_argnums=3
            )

    @property
    def name(self) -> str:
        return '.'.join([self.__class__.__name__,  self.task.name])

    def test_functional_metrics(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, wandb.Table]:
        return self._compute_functional_metrics(key, variables, True)

    def train_functional_metrics(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, wandb.Table]:
        return self._compute_functional_metrics(key, variables, False)

    def _compute_metrics(
            self,
            key: PRNGKeyArray,
            variables: Variables,
            testing: bool
    ) -> dict[str, jax.Array]:

        ((true_xs, true_ys), _, y_hat), model_metrics = self._eval_fun(
            self.fixed_seed,
            key,
            variables,
            self.num_predictions,
            testing
        )

        # Helper variables for division
        nB, nS, nT, nP = (
            self.num_samples, len(self.sample_lengths),
            self.resolution, self.num_predictions
        )

        # Extract dimensionality: (batch, samples, targets, predictions)
        name = self.model.simulate.modalities[0].name
        y_hat = y_hat.get(name)

        # Estimate Predictive Cross-Entropy (and Perplexity).
        log_probs, deltas = jax.vmap(
            nest_vmap(
                lambda t, y: (t.get.log_prob(y), (y - t.get.mean()).squeeze()),
                n=2  # batch jointly over (batch, targets)
            ),
            in_axes=(1, None), out_axes=1  # batch over (samples, ) first
        )(y_hat, true_ys)

        l2_errors, l1_errors = jnp.square(deltas), jnp.abs(deltas)

        # Aggregate over targets for the empirical cross-entropy
        cross_entropy = -log_probs.mean(axis=-1)
        perplexity = jnp.exp(cross_entropy)

        # Bound perplexity as it may explode in early stages of training.
        perplexity = jnp.clip(perplexity, a_max=1e6)

        # From flax.linen.sow for capturing intermediates, extract metrics:
        # Assume the form: {'intermediates': {... {'name/metrics': ({...}, )}}}
        # Convert into: {name: {...}}
        intermediates = model_metrics.get(Scope.Intermediates, {})
        flattened = flax.traverse_util.flatten_dict(intermediates, sep=' ')
        formatted = {
            k.split(' ')[-1].split('/')[0]: v[0]
            for k, v in flattened.items()
        }
        model_metrics_formatted = flax.traverse_util.flatten_dict(
            formatted, sep='_'
        )

        # Collect log-data of dimensionality: (batch, samples)
        data = {
            'l2 mean': l2_errors.mean(axis=-1),
            'l2 std': l2_errors.std(axis=-1),
            'l2 max': l2_errors.max(axis=-1),
            'l2 min': l2_errors.min(axis=-1),
            'l1 mean': l1_errors.mean(axis=-1),
            'l1 std': l1_errors.std(axis=-1),
            'l1 max': l1_errors.max(axis=-1),
            'l1 min': l1_errors.min(axis=-1),
            'perplexity': perplexity,
            'cross_entropy': cross_entropy,
            **model_metrics_formatted
        }
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
            for k, v in data.items() for fun, name in statistics
        }

        data_sizes = {
            'batch_size': jnp.asarray([nB] * nS),
            'burn_in': jnp.asarray(self.sample_lengths),
            'num': jnp.asarray([nT] * nS),
            'model_samples': jnp.asarray([nP] * nS)
        }

        # Return a large dictionary of aggregated performance metrics
        return data | data_sizes

    def _compute_functional_metrics(
            self,
            key: PRNGKeyArray,
            variables: Variables,
            testing: bool
    ) -> dict[str, wandb.Table]:
        """Get the test-loss per specified timestep (i.e., non-aggregated)"""

        # Get a dataset of distributional statistics of the performance
        # and model-metrics for the given `train_state`.
        # Statistics describe the distribution over the test-cases, the
        # quantities per test-case are averaged over all predictions.
        data = jax.jit(self._compute_metrics)(key, variables, testing)

        keys, values = list(zip(*data.items()))
        table_values = [v.tolist() for v in values]
        transposed = [list(i) for i in zip(*table_values)]

        return {'table': wandb.Table(columns=list(keys), data=transposed)}

    def test_plots(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, dict[str, jax.Array]]:
        return self._generate_plots(key, variables, True)

    def train_plots(
            self,
            key: PRNGKeyArray,
            variables: Variables
    ) -> dict[str, dict[str, jax.Array]]:
        return self._generate_plots(key, variables, False)

    def _generate_plots(
            self,
            key: PRNGKeyArray,
            variables: Variables,
            testing: bool
    ) -> dict[str, dict[str, Any]]:
        """Get the full 1D-function predictive per specified timestep."""
        if self._sampling_fun is None:
            return {}

        size = np.prod(self.task.shape)
        plotter = plot_1D_function_predictive if size == 1 \
            else plot_2D_function_predictive

        # Evaluate samples and visualize posterior predictive
        (vis_x, vis_y), (obs_x, obs_y), samples_y_hat = self._sampling_fun(
            self.fixed_seed,
            key,
            variables,
            self.num_predictions,
            testing
        )

        name = self.model.simulate.modalities[0].name
        samples_y_hat = samples_y_hat.get(name)

        figures = dict()
        for i, (y_true, x_in, y_in) in enumerate(zip(
            vis_y, obs_x, obs_y
        )):
            task_y_hat = jax.tree_map(lambda x: x[i], samples_y_hat)

            fig = plotter(
                task_y_hat,
                (x_in, y_in),
                (vis_x, y_true),
                self.plot_sample_lengths
            )

            # TODO: 3rd party error: wandb.Image doesn't save plotly png.
            #  Future (desired): wandb.Image(fig)
            image_buffer = io.BytesIO()  # Make new buffer, flush doesn't work
            fig.write_image(image_buffer, format='png')
            im = wandb.Image(Image.open(image_buffer))
            image_buffer.flush()

            # fig.show()  # Only for debugging: opens lots of tabs :)

            figures |= {f'regression_test_{i+1}': im}

        return {'plots': figures}
