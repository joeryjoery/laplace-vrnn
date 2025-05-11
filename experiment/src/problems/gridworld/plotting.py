import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from .gridworld import GridState


DEFAULT_PLOT_HEIGHT: int = 8
DEFAULT_PLOT_WIDTH: int = 8


def plot_agent(
        states: GridState,
        observations: jax.Array,
        rewards: jax.Array,
        max_episodes: int = 5
) -> plt.Figure:

    # Pre-compute image data
    goal = jax.tree_map(lambda x: x.at[0].get(), states.goal)
    episode_ends = jnp.where(states.step == 0)[0][1:]
    episode_states = jax.tree_map(lambda s: jnp.split(s, episode_ends), states)

    returns = jnp.split(jnp.cumsum(rewards), episode_ends)
    episode_success = [x[-1] for x in returns]

    cumulative_unseen = (jnp.cumsum(observations, axis=0) == 0).astype(int)
    episode_unseen = [
        x[-1] for x in jnp.split(cumulative_unseen, episode_ends, axis=0)
    ]
    plot_masks = [x.at[goal].set(2) for x in episode_unseen]

    # Generate plots
    n = min(1 + len(episode_ends), max_episodes)
    fig, axes = plt.subplots(
        1, n, figsize=(DEFAULT_PLOT_WIDTH * n, DEFAULT_PLOT_HEIGHT)
    )

    c = 0
    for i, (ax, pos, mask, success) in enumerate(
        zip(axes, zip(*episode_states.position), plot_masks, episode_success)
    ):
        plot_episode(ax, pos, mask, success)

        suffix, c = ('Success!', c+1) if success > c else ('Fail!', c)

        ax.set_title(f'Episode {i+1}: {suffix}')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_episode(
        canvas: plt.Axes,
        positions: tuple[jax.Array, jax.Array],
        mask: jax.Array,
        reward: float
):

    canvas.pcolormesh(mask.T, edgecolors='k', linewidth=2, cmap='binary')

    xs, ys = positions
    for i in range(len(xs) - 1):
        canvas.plot(
            [xs[i]+0.5, xs[i+1]+0.5],
            [ys[i]+0.5, ys[i+1]+0.5],
            color='red'
        )

    canvas.scatter(xs[-1]+0.5, ys[-1]+0.5, color='red', s=50)

    canvas.text(
        4.5, 4.5,
        f'{reward:.0f}',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20,
        color='white'
    )
    canvas.set_xticks([])
    canvas.set_yticks([])

    canvas.set_aspect('equal')
    canvas.invert_yaxis()
