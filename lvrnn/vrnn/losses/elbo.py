from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

from jaxtyping import PyTree, Scalar

import jax
import jax.numpy as jnp

import flax.linen as nn

from axme.core import Scope
from axme.consumer import Loss

from lvrnn.agent_model import (
    StateAdapter, Observation, Action, State, AfterState,
    TransitionAdapter, FlaxRNNStateAdapter, AgentModel
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class Divergence:

    def __call__(
            self,
            *args, **kwargs
    ) -> tuple[
        Scalar, dict[str, PyTree[jax.Array]]
    ]:
        pass


@dataclass
class VariationalLowerBound(Loss, ABC):
    loss: Loss  # Model likelihood
    divergence: Divergence  # Model complexity


@dataclass
class MAPDivergence(Divergence):
    """KL-divergence between sequential prior and posterior distributions.
    """
    adapter: StateAdapter = TransitionAdapter
    state_adapter: StateAdapter = FlaxRNNStateAdapter
    semi_gradient: bool = True

    def __call__(
            self,
            prior_state: AfterState,
            posterior_state: State,
            *args, **kwargs
    ) -> tuple[
        Scalar, dict[str, PyTree[jax.Array]]
    ]:
        # Extract Distributions (assumes an implementation for KL-Divergence)
        prior_state, _ = self.adapter.split_transformable(prior_state)

        _, (_, prior) = self.state_adapter.split_transformable(prior_state)
        _, (_, post) = self.state_adapter.split_transformable(posterior_state)

        if self.semi_gradient:
            # This option has practical advantages for gradient conditioning:
            # 1) Optimize posterior to match the prior in forward-KL.
            # 2) Disable optimizing the prior to match posterior in reverse-KL.
            prior = jax.tree_map(jax.lax.stop_gradient, prior)

        # Extract distributions in symbolic conventional form.
        q, p = prior.get, post.get

        # Computes KL(q || p)
        divergence, kl_metrics = p.kl_divergence(q)
        post_entropy = p.entropy()

        # Proportional to KL-divergence of isotropic gaussian.
        delta_mu = jnp.square(p.mean() - q.mean()).sum()

        metrics = {
            'kl_divergence': divergence,
            'posterior_entropy': post_entropy,
            'cross_entropy': divergence - post_entropy,
            'euclidean': delta_mu
        } | kl_metrics

        return divergence, metrics


@dataclass
class MahalanobisDivergence(MAPDivergence):
    """Simplification of the KL-Divergence which may aid Optimization.

    This class is only applicable if the covariances are not optimized, i.e.,
    the covariances are treated as constants, and we only optimize the means.
    As a consequence, the KL-divergence might be arbitrarily scaled regardless
    of the goodness of fit for the means, this can make tuning hyperparameters
    more unintuitive. The Mahalanobis distance for the model complexity can be
    more appropriately scaled.
    """

    def __call__(
            self,
            prior_state: AfterState,
            posterior_state: State,
            *args, **kwargs
    ) -> tuple[
        Scalar, dict[str, PyTree[jax.Array]]
    ]:
        loss, metrics = super().__call__(
            prior_state, posterior_state, *args, **kwargs
        )
        return metrics['mahalanobis'], metrics


@dataclass
class IgnoreDivergence(Divergence):
    """Adapter for SequentialELBO compatibility without a KL-implementation.

    This can also be used to eliminate the KL-divergence from the
    differentiation graph, while still logging the relevant metrics.
    """
    base: Divergence | None = None

    def __call__(self, *args, **kwargs) -> tuple[
        Scalar, dict[str, PyTree[jax.Array]]
    ]:
        metrics = {}
        if self.base:
            value, metrics = self.base(*args, **kwargs)

        return jnp.zeros(()), metrics


@dataclass
class SequentialELBO(VariationalLowerBound):
    beta: float = 1e-2
    num_model_samples: int = 1

    def eval(
            self,
            model: AgentModel,
            observations: Observation,
            actions: Action,
            target_data: PyTree[jax.Array],
            s_prev: State | None = None,
            *args, **kwargs
    ):
        if s_prev is None:
            ins_0 = jax.tree_map(
                lambda x: x.at[0].get(), (observations, actions)
            )
            s_prev = model.initial_state(*ins_0)

            # Prevent gradients flowing back for initial state transformation
            s_prev = jax.tree_map(jax.lax.stop_gradient, s_prev)

        # 1) Get model statistics
        (p_seq, q_seq), (as_seq, s_seq) = model.joint_transition.unroll(
            observations, actions, s_prev
        )

        signal_out = None
        if model.simulate:

            signal_out, prior_state = nn.vmap(
                lambda m, *ins: m.simulate(*ins, n=self.num_model_samples),
                variable_axes={Scope.Params: None},  # Tie weights
                split_rngs={
                    Scope.Params: False, type(model.simulate).__name__: True
                }
            )(model, as_seq)

        # 2) Evaluate model statistics given data
        loss_value, loss_metrics = self.loss.eval(
            target_data,
            prior_signal=signal_out,
            prior_predictive=q_seq,
            posterior_predictive=p_seq
        )
        complexity, complexity_metrics = jax.vmap(self.divergence)(
            prior_state=as_seq, posterior_state=s_seq
        )
        penalty = complexity.mean()

        optimizer_loss = loss_value + self.beta * penalty

        # 3) Return optimized scalar and monitoring statistics
        complexity_metrics = jax.tree_map(jnp.mean, complexity_metrics)
        metrics = {
            'loss': optimizer_loss,
            'complexity': complexity_metrics | {
                'penalty': self.beta * penalty
            },
            'target_loss': loss_metrics | {'value': loss_value}
        }

        return optimizer_loss, metrics


@dataclass
class VariBADELBO(VariationalLowerBound):
    """Extends SequentialELBO by computing the loss on a window [t-H:t+H-1].

    At every time-step the reconstruction error is computed not only at `t`,
    but also for all observations within a fixed window.

    Note that for Control problems/ Reinforcement Learning, the original
    paper *stops* gradients for the policy network input. In this way, the
    observation model is only trained on a supervised loss and is not
    confounded with gradients from the policy network. To implement this in
    our framework, you must add a stop_gradient somewhere in the
    `PosteriorPredictive` implementation and *not* here in the loss.

    See, Zintgraf L. et al., 2019. https://arxiv.org/abs/1910.08348
    """
    window_size: int = 10  # TODO: right now, window = full sequence
    beta: float = 1e-2
    num_model_samples: int = 1

    def eval(
            self,
            model: AgentModel,
            observations: Observation,
            actions: Action,
            target_data: PyTree[jax.Array],
            s_prev: State | None = None,
            *args, **kwargs
    ):
        # 1) Compute all model states over the trajectory
        if s_prev is None:
            ins_0 = jax.tree_map(
                lambda x: x.at[0].get(), (observations, actions)
            )

            s_prev = model.initial_state(*ins_0, *args, **kwargs)

            # Prevent gradients flowing back for initial state transformation
            s_prev = jax.tree_map(jax.lax.stop_gradient, s_prev)

        (p_seq, q_seq), (as_seq, s_seq) = model.joint_transition.unroll(
            observations, actions, s_prev
        )
        s_full = jax.tree_map(
            lambda x, y: jnp.concatenate([jnp.expand_dims(x, 0), y], axis=0),
            s_prev, s_seq
        )

        # 2) Embed all actions into the model space.
        action_embeddings = nn.vmap(
            lambda module, a: module.action_embedder(a),
            variable_axes={Scope.Params: None},  # Tie weights
            split_rngs={
                Scope.Params: False,
                type(model.action_embedder).__name__: True
            }
        )(model, actions)

        # 3) Convolve the loss for target-data + actions with all model-states
        def state_loss(
                module: AgentModel,
                s_t: State,
                i: jax.Array
        ):
            afterstates = nn.vmap(
                lambda m, *ins: m.apply_action(*ins),
                in_axes=(None, 0),
                variable_axes={Scope.Params: None},  # Tie weights
                split_rngs={
                    Scope.Params: False,
                    type(module.apply_action).__name__: True
                }
            )(module, s_t, action_embeddings)

            signal_out, prior_state = nn.vmap(
                lambda m, *ins: m.simulate(*ins, n=self.num_model_samples),
                variable_axes={Scope.Params: None},  # Tie weights
                split_rngs={
                    Scope.Params: False,
                    type(module.simulate).__name__: True
                }
            )(module, afterstates)

            # TODO: mask-out a fixed window for self.loss
            ts = jnp.arange(len(jax.tree_util.tree_leaves(actions)[0]))
            mask = ((ts < (i - self.window_size)) +
                    (ts >= (i + self.window_size)))

            return self.loss.eval(
                target_data,
                prior_signal=signal_out,
                prior_predictive=q_seq,
                posterior_predictive=p_seq
            )

        # Convolve timeseries with all model states.
        state_losses, state_metrics = nn.vmap(
            state_loss,
            variable_axes={Scope.Params: None},  # Tie weights
            split_rngs={
                Scope.Params: False,
                type(model.apply_action).__name__: True,
                type(model.simulate).__name__: True
            }
        )(model, s_full, jnp.arange(len(jax.tree_util.tree_leaves(s_full)[0])))

        # TODO: Weight losses based on the current state-timestep? (MuZero)
        # Aggregate losses per state
        loss_value, loss_metrics = jax.tree_map(
            jnp.mean, (state_losses, state_metrics)
        )

        complexity, complexity_metrics = jax.vmap(self.divergence)(
            prior_state=as_seq, posterior_state=s_seq
        )
        penalty = complexity.mean()

        optimizer_loss = loss_value + self.beta * penalty

        # 3) Return optimized scalar and monitoring statistics
        complexity_metrics = jax.tree_map(jnp.mean, complexity_metrics)
        metrics = {
            'loss': optimizer_loss,
            'complexity': complexity_metrics | {
                'penalty': self.beta * penalty
            },
            'target_loss': loss_metrics | {'value': loss_value}
        }

        return optimizer_loss, metrics
