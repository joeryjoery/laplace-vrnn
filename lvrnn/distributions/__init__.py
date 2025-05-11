"""Module that copies/ reimplements some Distrax utilities.

The distrax API does not interoperate well at the moment due to being
a layer on top of tensorboard_probability. This limits usability of
jax-transforms (especially `vmap`) and makes subclassing more complex.

This module strips this down to the Modules that we need ourselves.
"""

from .interface import Distribution
from .serialize import SerializeTree
from .deterministic import Deterministic
from .mvn import (
    MultivariateNormal,
    MultivariateNormalTriangular,
    MultivariateNormalDiagonalCovariance, MultivariateNormalFullCovariance,
    MultivariateNormalDiagonalPrecision, MultivariateNormalFullPrecision,
    MultivariateNormalExpOrthogonal
)
from .beta import Beta

from .discrete import Categorical

from . import ensemble
