"""File containing example implementations for `loss`."""
from __future__ import annotations
from functools import partial
from typing import Generic, TYPE_CHECKING

import jax
import jax.numpy as jnp

import rlax

import jit_env

from jaxtyping import Scalar, PyTree

from axme.consumer import Loss

from lvrnn.distributions import SerializeTree
from lvrnn.distributions.interface import DistT

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class PPO(Loss, Generic[DistT]):
    """Implements the Proximal Policy Optimization Loss

    TODO:
     - Early stopping based on hard KL constraint?

    See: https://arxiv.org/abs/1707.06347
    """
    policy_modality: str
    value_modality: str

    # Loss Aggregation config
    policy_scale: float = 1.0
    value_scale: float = 1.0
    entropy_scale: float = 0.01

    # Policy-Gradient + Value Loss config
    td_lambda: float = 0.9
    discount: float = 0.9
    clip_epsilon: float = 0.2
    normalize_advantage: bool = False
    exact_entropy: bool = True

    # Value-Learning config
    semi_gradient: bool = True

    def eval(
            self,
            target_data: tuple[
                jit_env.TimeStep, jit_env.Action, dict[str, jax.Array]
            ],
            posterior_predictive: dict[str, SerializeTree[DistT]],
            *args, **kwargs
    ) -> tuple[Scalar, dict[str, PyTree[jax.Array]]]:
        """PPO Loss computation on a single episode-trajectory."""

        # Unpack pre-computed statistics
        mean_values = posterior_predictive[self.value_modality].get.mean()
        policies = posterior_predictive[self.policy_modality]

        # Remove trailing policy pi_T since no actions are observed >= T
        policies = jax.tree_map(lambda x: x.at[:-1].get(), policies)

        # Omit zeroth step statistics for model initialization.
        steps, actions, extras = jax.tree_map(
            lambda x: x.at[1:].get(),
            target_data
        )

        def get_policy_stats(
                policy: DistT,
                action: jit_env.Action
        ) -> tuple[jax.Array, jax.Array]:
            dist = policy.get
            # -> pi_t(a_t+1), H[pi_t], t = 0, ..., T - 1
            return dist.log_prob(action), dist.entropy()

        # Compute Policy Loss Statistics
        discounts = steps.discount * self.discount
        log_probs, entropies = jax.vmap(get_policy_stats)(policies, actions)

        ratios = jnp.exp(log_probs - extras['log_prob'])
        advantages = rlax.truncated_generalized_advantage_estimation(
            r_t=steps.reward,  # [1, T]
            discount_t=discounts,  # [1, T],
            lambda_=self.td_lambda,
            values=mean_values.squeeze(),  # [0, T]
            stop_target_gradients=self.semi_gradient
        )

        # Split up advantage graph for Policy Gradient and Value Loss.
        policy_advantages = advantages
        if self.normalize_advantage:
            # Not recommended: variance dependent on batch-size.
            policy_advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-6
            )

        # Compute Equation (9) from the PPO Paper.
        entropy_loss = -entropies.mean() if self.exact_entropy \
            else log_probs.mean()
        policy_loss = rlax.clipped_surrogate_pg_loss(
            ratios,
            adv_t=policy_advantages,
            epsilon=self.clip_epsilon,
            use_stop_gradient=True  # Values shouldn't optimize policy
        )
        value_loss = rlax.l2_loss(advantages).mean()
        total_loss = (
                self.policy_scale * policy_loss +  # Minimize
                self.value_scale * value_loss +  # Minimize
                self.entropy_scale * entropy_loss  # Maximize Entropy
        )

        metrics = {
            'ppo_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
        stats = [
            (jnp.mean, 'E {}'), (partial(jnp.var, ddof=1), 'Var {}'),
            (jnp.median, 'Med {}'),
            (jnp.min, 'Min {}'), (jnp.max, 'Max {}'),
        ]
        loss_data = {
            'ratios': ratios,
            'advantages': advantages,
            'policy_entropy': entropies,
            'approx_policy_entropy': -log_probs
        }
        loss_stats = {
            name.format(k): jax.tree_map(fun, v)
            for k, v in loss_data.items() for fun, name in stats
        }

        return total_loss, metrics | loss_stats
