from __future__ import annotations

import jax
import jax.numpy as jnp

import numpy as np


def real_to_unitriangular(x: jax.Array) -> jax.Array:
    """Map a vector of length n(n-1)/2 to an unitriangular matrix.

    This matrix has eigenvalues all equal to 1, s.t., determinant = 1.
    """
    n = int(np.round(1 + np.sqrt(8 * len(x) + 1) // 2))
    m = jnp.zeros((n, n)).at[jnp.tril_indices(n, -1)].set(x)
    return m + jnp.eye(n)
