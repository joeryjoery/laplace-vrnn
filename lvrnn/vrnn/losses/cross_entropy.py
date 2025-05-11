"""File containing example implementations for `loss`."""
from __future__ import annotations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

import optax

from jaxtyping import Scalar

from axme.consumer import Loss

from lvrnn.distributions import SerializeTree, Distribution

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class EmpiricalCrossEntropy(Loss):
    modality: str
    from_timestep: bool = False  # TODO: Compatibility hack, refactor

    def eval(
            self,
            targets: jax.Array,
            prior_signal: dict[str, SerializeTree[Distribution]],
            *args, **kwargs
    ) -> tuple[Scalar, dict]:

        @jax.vmap
        def f(target, prior):
            return -prior.get.log_prob(target)

        if self.from_timestep:
            targets = targets[0].reward

            # Remove zeroth observation
            prior_signal, targets = jax.tree_map(
                lambda x: x.at[1:].get(), (prior_signal, targets)
            )

        losses = f(targets, prior_signal.get(self.modality))

        stats = {
            'min': losses.min(),
            'median': jnp.median(losses),
            'mean': losses.mean(),
            'max': losses.max(),
        }

        return losses.mean(), stats


@dataclass
class StandardGaussianEmpiricalCrossEntropy(Loss):
    """Impose a standard gaussian likelihood given the empirical mean.

    In other words, this is the mean-squared-error loss.
    """
    modality: str
    from_timestep: bool = False  # TODO: Compatibility hack, refactor

    def eval(
            self,
            targets: jax.Array,
            prior_signal: dict[str, SerializeTree[Distribution]],
            *args, **kwargs
    ) -> tuple[Scalar, dict]:

        @jax.vmap
        def f(target, prior):
            return optax.l2_loss(
                predictions=prior.get.mean().squeeze(),
                targets=target.squeeze()
            )

        if self.from_timestep:
            targets = targets[0].reward

            # Remove zeroth observation
            prior_signal, targets = jax.tree_map(
                lambda x: x.at[1:].get(), (prior_signal, targets)
            )

        l2 = f(targets, prior_signal.get(self.modality))

        stats = {
            'min': l2.min(),
            'median': jnp.median(l2),
            'mean': l2.mean(),
            'max': l2.max(),
        }

        return l2.mean(), {
            'l2': stats,
            'l1': jax.tree_map(lambda x: jnp.sqrt(x+1e-8), stats)
        }
