from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

import numpy as np

import jit_env

from jit_env import specs, State, Action, TimeStep

from jaxtyping import Shaped, ArrayLike, PRNGKeyArray

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


IntLike = Shaped[ArrayLike, ""]
Coordinate = tuple[IntLike, IntLike]


@dataclass
class GridState:
    key: PRNGKeyArray
    step: IntLike

    start: Coordinate
    goal: Coordinate
    position: Coordinate

    optimal: IntLike


class SquareGrid(jit_env.Environment):
    """Sparse open grid navigation reimplemented from VariBAD.

    This reimplementation poses the Episodic MDP as an infinite horizon
    contextual MDP. This means that returned TimeStep objects will never
    evaluate TimeStep.last() to True. So, this must be wrapped with e.g.,
    a TimeLimit. Internally, the MDP will reset episodically, but this
    resetting is seen as part of the MDP-dynamics (i.e., transition to s_0).

    Note, the agent should not observe the `goal` inside GridState!

    See: Zintgraf L. et al., 2020. https://arxiv.org/abs/1910.08348

    See the reference implementation here,
    https://github.com/lmzintgraf/varibad
    https://github.com/lmzintgraf/varibad/blob/master/environments/navigation/gridworld.py
    """
    _REGRET_KEY: str = 'regret'
    _START_KEY: str = 'start'
    _GOAL_KEY: str = 'goal'

    def __init__(
            self,
            n: int,
            episode_steps: int,
            start_bounds: tuple[jax.Array, jax.Array] | None = None,
            goal_bounds: tuple[jax.Array, jax.Array] | None = None,
            one_hot_encoding: bool = False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n = n
        self.episode_steps = episode_steps
        self.one_hot_encoding = one_hot_encoding

        # Constrain where on the grid the agent and goal are initialized.
        low_0, high_0 = start_bounds or (0, n)
        low_g, high_g = goal_bounds or (0, n)

        # Ensure that goal and start do not always overlap.
        if low_0 == high_0 == low_g == high_g:
            raise RuntimeError("Agent always gets initialized at goal!")

        if low_0 > high_0 or low_g > high_g:
            raise RuntimeError(
                f"Specified lower bound > upper bound: "
                f"start: {low_0} - {high_0}; end: {low_g} - {high_g}."
            )

        # Pre-compute index-ranges to efficiently initialize contexts.
        start_bounds, goal_bounds = jax.tree_map(
            lambda arr: np.broadcast_to(
                np.clip(arr, 0, n).astype(int), (2,)
            ),
            ((low_0, high_0), (low_g, high_g))
        )
        grid_indices = jnp.unravel_index(
            jnp.arange(self.n * self.n), (self.n, self.n)
        )
        grid_indices = jnp.asarray(grid_indices).T

        self.start_mask = (grid_indices <= start_bounds[1]).all(axis=-1) & \
                          (grid_indices >= start_bounds[0]).all(axis=-1)
        self.goal_mask = (grid_indices <= goal_bounds[1]).all(axis=-1) & \
                         (grid_indices >= goal_bounds[0]).all(axis=-1)

    def _get_start_goal_pair(
            self, key: PRNGKeyArray
    ) -> tuple[Coordinate, Coordinate]:
        """Sample a non-overlapping start-goal pair without a while loop.

        For both start and goal sample without replacement 2 options
        regardless of whether this is possible (i.e., only 1 option exists).
        Then, if the goal tile only has 1 option, the second result (=0) is
        overriden with the singular option.

        Finally, enumerate all possible pairs between start and pair (4 in
        total) and return the first pair that is non-overlapping as an
        unraveled Coordinate.

        Start and Goal can never overlap in this function as long as the
        predicate `all(self.start_mask == self.goal_mask)` isn't True.
        But, this is checked in the __init__.

        This method for sampling random indices prevents us from having to
        do awkward size checks or doing while loops. Which is nicer for Jax.
        """
        key_start, key_goal = jax.random.split(key)

        start_idx = jax.random.choice(
            key_start, self.n * self.n, (2,), p=self.start_mask, replace=False
        )
        goal_idx = jax.random.choice(
            key_goal, self.n * self.n, (2,), p=self.goal_mask, replace=False
        )

        single = (self.goal_mask.sum() == 1)
        override = goal_idx.at[1].get() * (1 - single) + goal_idx.at[0].get()
        goal_idx = goal_idx.at[1].set(override)

        pairs = jnp.asarray(jnp.meshgrid(start_idx, goal_idx)).reshape(2, -1)
        i = jnp.argmin(jnp.equal(pairs[0], pairs[1]))

        start_tile = jnp.unravel_index(start_idx.at[i].get(), (self.n, self.n))
        goal_tile = jnp.unravel_index(goal_idx.at[i].get(), (self.n, self.n))

        return start_tile, goal_tile  # type: ignore

    def make_observation(
            self,
            state: GridState,
            observe_goal: bool = False,
            observe_start: bool = False,
            one_hot_encoding: bool = False
    ) -> jax.Array:
        if one_hot_encoding:
            # (n, 2) observation one-hot-encoding of row-column.
            encoding = jnp.zeros((self.n, 2), jnp.float32)
            encoding = encoding.at[state.position, jnp.arange(2)].set(1.0)

            if observe_goal:
                encoding = encoding.at[state.goal, jnp.arange(2)].set(2.0)
            if observe_start:
                encoding = encoding.at[state.start, jnp.arange(2)].set(3.0)

            return encoding

        # (n, n) observation of the full-maze and the agent-position.
        grid = jnp.zeros((self.n, self.n), jnp.float32)
        grid = grid.at[state.position].set(1.0)

        if observe_goal:
            grid = grid.at[state.goal].set(2.0)
        if observe_start:
            grid = grid.at[state.start].set(3.0)

        return grid

    def reset(
            self,
            key: PRNGKeyArray,
            *,
            options: dict[str, Coordinate] | None = None
    ) -> tuple[State, TimeStep]:
        key_carry, key_init = jax.random.split(key)

        if options:
            start = options.get(self._START_KEY, None)
            goal = options.get(self._GOAL_KEY, None)

            if (start is None) or (goal is None):
                raise RuntimeError(
                    f"Empty option encountered: start={start} goal={goal}"
                )

        else:
            # Randomly sample a start-goal pair and compute their L1 distance.
            start, goal = self._get_start_goal_pair(key_init)

        length_optimal_path = sum(jax.tree_map(
            lambda a, b: jnp.abs(a - b), start, goal
        )) - 1  # minus one to remove the starting-position.

        state = GridState(
            None, 0,
            start=start, goal=goal, position=start,
            optimal=length_optimal_path
        )

        obs_init = self.make_observation(
            state, observe_goal=False, observe_start=False,
            one_hot_encoding=self.one_hot_encoding
        )
        step = jit_env.restart(
            obs_init,
            extras={self._REGRET_KEY: jnp.zeros((), jnp.int32)}
        )

        return state, step

    def step(self, state: GridState, action: Action) -> tuple[State, TimeStep]:

        # Decode Action in Int-order: [(-1, 0), (1, 0), (0, -1), (0, 1)]
        is_vertical = action < 2
        step = (action % 2) * 2 - 1
        displacement = is_vertical * step, (1 - is_vertical) * step

        agent_pos = [
            jnp.clip(current + shift, 0, self.n - 1)
            for current, shift in zip(state.position, displacement)
        ]

        # Check for special situations
        goal_achieved = sum(x == y for x, y in zip(agent_pos, state.goal)) == 2
        episode_done = (state.step + 1) % self.episode_steps == 0
        should_restart = goal_achieved | episode_done

        # Transition (including episodic restarts!)
        new_pos: Coordinate = tuple([  # type: ignore
            jax.lax.select(should_restart, start, new)
            for new, start in zip(agent_pos, state.start)
        ])
        new_step = jax.lax.select(should_restart, 0, state.step + 1)

        # Update states
        new_state = GridState(
            None, new_step,
            start=state.start, goal=state.goal,
            position=new_pos,
            optimal=state.optimal
        )

        # Regret can only really be valuated episodically (not step-wise)
        # So every additional step beyond the optimal path is tracked.
        regret_penalty = (new_state.step > state.optimal).astype(jnp.int32)

        obs = self.make_observation(
            new_state, observe_goal=False, observe_start=False,
            one_hot_encoding=self.one_hot_encoding
        )

        step = jit_env.transition(
            reward=jax.lax.select(goal_achieved, 1.0, 0.0),
            discount=jax.lax.select(should_restart, 0.0, 1.0),
            observation=obs,
            extras={self._REGRET_KEY: regret_penalty}
        )

        return new_state, step

    def reward_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray((), jnp.float32, 0.0, 1.0)

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray((), jnp.float32, 0.0, 1.0)

    def observation_spec(self) -> specs.Spec:
        if self.one_hot_encoding:
            return specs.BoundedArray((self.n, 2), jnp.float32, 0.0, 1.0)
        else:
            return specs.BoundedArray((self.n, self.n), jnp.float32, 0.0, 1.0)

    def action_spec(self) -> specs.Spec:
        return specs.DiscreteArray(4)
