from typing import Sequence

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from lvrnn import distributions as dist
from lvrnn.distributions import ensemble

from .bandits import BanditState


DEFAULT_PLOT_HEIGHT: int = 8
DEFAULT_PLOT_WIDTH: int = 8


def plot_discrete_bandit(
        states: BanditState, rewards: jax.Array, actions: jax.Array,
        predictions: dist.SerializeTree[ensemble.Mixture[dist.Categorical]],
        slices: Sequence[int]
) -> plt.Figure:

    mask = jax.vmap(jnp.equal, in_axes=(0, None))(
        actions, jnp.arange(states.ps.shape[-1])
    )

    action_bins = jnp.cumsum(mask, axis=0)
    reward_bins = jnp.cumsum(rewards, axis=0)

    fig, axes = plt.subplots(
        1, len(slices), figsize=(8 * len(slices), 8),
        sharey=True, sharex=True  # type: ignore
    )

    for i, ax in zip(slices, axes):
        mixture_stream = jax.tree_map(lambda x: x[i], predictions)
        plot_history(
            ax, i, states.ps[0], mixture_stream, action_bins[i], reward_bins[i]
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_history(
        canvas: plt.Axes,
        step: int,
        probabilities: jax.Array,
        predictions: dist.SerializeTree[ensemble.Mixture[dist.Categorical]],
        cumulative_actions: jax.Array,
        cumulative_rewards: jax.Array
):
    action_mass = cumulative_actions / cumulative_actions.sum()

    canvas.bar(
        jnp.arange(probabilities.size) + 0.25, probabilities,
        width=0.5, edgecolor='black', hatch='/',
        label='Probabilities'
    )
    canvas.bar(
        jnp.arange(probabilities.size) - 0.25, action_mass,
        width=0.5, edgecolor='black', hatch='.',
        label='Actions'
    )

    for i in range(action_mass.size):
        arm_mixture = jax.tree_map(lambda x: x[i], predictions).get

        component_probs = jax.vmap(lambda stream: stream.get.prob(1))(
            arm_mixture.dist_stream
        )
        low, high = component_probs.min(), component_probs.max()
        median = jnp.median(component_probs)

        bounds = jnp.asarray((abs(low - median), abs(high - median)))

        canvas.errorbar(
            i+0.25, median,
            xerr=0.1,
            yerr=bounds[:, None] + 5e-3,
            ecolor='white', linewidth=4
        )
        canvas.errorbar(
            i+0.25, median,
            xerr=0.1,
            yerr=bounds[:, None],
            ecolor='black'
        )

        canvas.text(
            i-0.25, 0.025 + action_mass[i],  # type: ignore
            f'{cumulative_actions[i]}',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black')
        )

    canvas.errorbar(
        0, 0, xerr=0, yerr=0,
        color='black', label='Model Posterior (Min-Median-Max)'
    )

    canvas.hlines(1.0, -0.5, probabilities.size - 0.5, color='black')

    canvas.set_ylim(0, 1.1)
    canvas.set_xlim(-0.5, 4.5)
    canvas.set_yticks(jnp.linspace(0, 1, 5), jnp.linspace(0, 1, 5))

    canvas.set_title(f"Progression Step: {step+1} - "
                     f"Reward: {cumulative_rewards.sum():.0f}")
    canvas.legend(ncol=3)
