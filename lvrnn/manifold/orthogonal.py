import jax
import jax.numpy as jnp

import numpy as np


@jax.jit
def cayley_map(matrix: jax.Array) -> jax.Array:
    """Map a nxn Array to SO(n), we transform `matrix` to be skew-symmetric

    The transformation is defined as:
        Q = (A - I)(A + I)^-1,  A: skew-symmetric
          = -(I + A^T)(I + A)^-1 = -M^T M^-1

    See, https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map.
    """
    matrix = jnp.tril(matrix, -1)
    matrix = (matrix - matrix.T) / 2.0 + jnp.eye(len(matrix))

    return -jnp.linalg.solve(
        matrix, matrix.T
    )


def _matrix_exp_body(carry, x):
    matrix, matrix_power, M = carry

    # Associative scan
    matrix_power = matrix_power @ (M * x)
    matrix = matrix + matrix_power

    return (matrix, matrix_power, M), None


@jax.jit
def matrix_exp(matrix: jax.Array) -> jax.Array:
    # Compute the matrix-exp according to the Taylor expansion.
    Q, *_ = jax.lax.scan(
        _matrix_exp_body,
        (jnp.eye(len(matrix)), jnp.eye(len(matrix)), matrix),
        1.0 / jnp.arange(1, len(matrix) // 2)  # Truncate half-way.
    )[0]

    return Q


def real_to_orthogonal(phi: jax.Array, method: str = 'cayley') -> jax.Array:
    """Map a vector of angles of length n(n-1)/2 to a point in SO(n).

    Initial testing shows 'cayley' to be faster than 'exp'.
    We have not implemented the rotation-method.

    Alternatively, see: https://math.stackexchange.com/questions/1364495/haar-measure-on-operatornameson
    """
    options = ('cayley', 'exp')

    n = int(np.round(1 + np.sqrt(8 * len(phi) + 1) // 2))
    m = jnp.zeros((n, n)).at[jnp.tril_indices(n, -1)].set(phi)

    if method.lower() == 'cayley':
        return cayley_map(m)
    elif method.lower() == 'exp':
        return matrix_exp(m - m.T)
    else:
        raise NotImplementedError(
            f"Method `{method}` is not supported. "
            f"Choose `method` from: {options}"
        )
