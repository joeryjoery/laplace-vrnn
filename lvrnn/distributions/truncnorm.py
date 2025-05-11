"""Re-implementations of parameterizing MultiVariate Normal distributions

Code is inspired by Distrax, but implemented differently due to missing
features, compatibility issues, or numerical optimizations.
"""
from __future__ import annotations
from typing import Sequence
import abc

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

import numpy as np

from .interface import Distribution, EventT


# TODO: Inherit from mvn

class TruncatedNormal(Distribution, abc.ABC):
    """Base distribution for MultivariateNormal extensions"""

    def __init__(
            self,
            loc: jax.Array,
            scale: jax.Array,
            diagonal: bool = False
    ):
        pass
