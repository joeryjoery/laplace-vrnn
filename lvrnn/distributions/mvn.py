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


class MultivariateNormal(Distribution, abc.ABC):
    """Base distribution for MultivariateNormal extensions"""

    def __init__(self, loc, scale, diagonal: bool):
        self.loc = loc
        self.scale = scale
        self._diagonal = diagonal

    @abc.abstractmethod
    def covariance(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def precision(self) -> jax.Array:
        pass

    @abc.abstractmethod
    def _unstandardize(self, value: EventT) -> EventT:
        pass

    @abc.abstractmethod
    def _standardize(self, value: EventT) -> EventT:
        pass

    @abc.abstractmethod
    def mahalanobis_distance(
            self,
            other: MultivariateNormalExpTriangular,
            **kwargs
    ) -> jax.Array:
        # Special case for the KL-divergence isolating the mean-distance.
        pass

    @property
    def event_shape(self) -> Sequence[int]:
        return self.loc.shape

    @property
    def is_diag(self):
        return self._diagonal

    def mean(self) -> jax.Array:
        return self.loc

    def mode(self) -> jax.Array:
        return self.mean()

    def median(self) -> jax.Array:
        return self.mean()

    def variance(self) -> jax.Array:
        return jnp.diag(self.covariance())

    def _sample_n(self, key: PRNGKeyArray, n: int) -> EventT:
        eps = jax.random.normal(key, (n, *self.mean().shape))
        return jax.vmap(self._unstandardize)(eps)

    def log_prob(self, value: EventT) -> jax.Array:
        k = np.prod(self.event_shape)
        log_Z = self.entropy() - k * 0.5

        mahalanobis_factor = self._standardize(value)
        squared_mahalanobis_norm = mahalanobis_factor @ mahalanobis_factor

        return -log_Z - 0.5 * squared_mahalanobis_norm

    def log_cdf(self, value: EventT) -> jax.Array:
        return jax.scipy.special.log_ndtr(self._standardize(value))


class MultivariateNormalTriangular(MultivariateNormal):

    def __init__(
            self,
            loc,
            scale,
            diagonal: bool = False,
            inverse: bool = False
    ):
        super().__init__(loc, scale, diagonal)
        self._inverse = inverse

        self._logdet = jnp.log(scale).sum() if diagonal else \
            jnp.log(jnp.diag(scale)).sum()

    @property
    def inverted(self):
        return self._inverse

    def covariance(self) -> jax.Array:
        if self.inverted:
            return jax.scipy.linalg.solve_triangular(
                self.scale.T, jax.scipy.linalg.solve_triangular(
                    self.scale, jnp.eye(len(self.scale)),
                    lower=True
                ), lower=False
            )
        return self.scale @ self.scale.T

    def precision(self) -> jax.Array:
        if self.inverted:
            return self.scale @ self.scale.T

        return jax.scipy.linalg.solve_triangular(
            self.scale.T, jax.scipy.linalg.solve_triangular(
                self.scale, jnp.eye(len(self.scale)),
                lower=True
            ), lower=False
        )

    def _sample_n(self, key: PRNGKeyArray, n: int) -> EventT:
        eps = jax.random.normal(key, (n, *self.mean().shape))
        return jax.vmap(self._unstandardize)(eps)

    def variance(self) -> jax.Array:
        if self.is_diag:
            scale = 1.0 / self.scale if self.inverted else self.scale
            return jnp.square(scale)

        return jnp.diag(self.covariance())

    def log_prob(self, value: EventT) -> jax.Array:
        k = np.prod(self.event_shape)
        log_Z = self.entropy() - k * 0.5

        mahalanobis_factor = self._standardize(value)
        squared_mahalanobis_norm = jnp.square(mahalanobis_factor).sum()

        return -log_Z - 0.5 * squared_mahalanobis_norm

    def log_cdf(self, value: EventT) -> jax.Array:
        return jax.scipy.special.log_ndtr(self._standardize(value))

    def kl_divergence(
            self,
            other: MultivariateNormalPrecision,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Computes the Kullback-Leibler Divergence of the distribution.

        """
        k = np.prod(self.event_shape)

        if other.is_diag and self.is_diag:
            scale_self = 1.0 / self.scale if self.inverted else self.scale
            scale_other = other.scale if other.inverted else 1.0 / other.scale

            trace = jnp.square(scale_self * scale_other).sum()
        else:
            scale_other = jnp.diag(other.scale) if other.is_diag else other.scale
            scale_self = jnp.diag(self.scale) if self.is_diag else self.scale

            if other.inverted:
                if self.inverted:
                    matmul_root = jax.scipy.linalg.solve_triangular(
                        scale_self, scale_other, lower=True
                    )
                else:
                    matmul_root = scale_other.T @ scale_self
            else:
                if self.inverted:
                    # Not recommended, explicit inverse...
                    matmul_root = jax.scipy.linalg.solve_triangular(
                        scale_other, jnp.linalg.inv(scale_self.T), lower=True
                    )
                else:
                    matmul_root = jax.scipy.linalg.solve_triangular(
                        scale_other, scale_self, lower=True
                    )

            trace = jnp.square(matmul_root).sum()

        # Multiply by two since the log-determinant is computed with Cholesky
        # Adequately flip the log-determinant signs for inversions.
        log_determinant_delta = 2 * (
                jax.lax.select(other.inverted, -1, 1) * other._logdet -
                jax.lax.select(self.inverted, -1, 1) * self._logdet
        ).sum()

        mahalanobis_norm = self.mahalanobis_distance(other)

        kl2 = trace + mahalanobis_norm - k + log_determinant_delta
        return 0.5 * kl2, {
            'trace': 0.5 * trace,
            'mahalanobis': 0.5 * mahalanobis_norm,
            'determinant': 0.5 * log_determinant_delta
        }

    def mahalanobis_distance(
            self,
            other: MultivariateNormalPrecision,
            **kwargs
    ):
        if other.is_diag and self.is_diag:
            scale_other = other.scale if other.inverted else 1.0 / other.scale
            mahalanobis_factor = (other.mean() - self.mean()) * scale_other

        else:
            scale_other = jnp.diag(other.scale) if other.is_diag else other.scale

            if other.inverted:
                mahalanobis_factor = (other.mean() - self.mean()) @ scale_other
            else:
                mahalanobis_factor = jax.scipy.linalg.solve_triangular(
                    scale_other, other.mean() - self.mean(), lower=True
                )

        return mahalanobis_factor @ mahalanobis_factor

    def entropy(self) -> jax.Array:
        """Computes the Differential entropy of the distribution.

        The determinant follows: |A^-1| = 1/|A| --> log(|A^-1| = -log |A|
        """
        k = np.prod(self.event_shape)
        sign = jax.lax.select(self.inverted, -1.0, 1.0)
        return 0.5 * (k + k * jnp.log(2 * jnp.pi) + 2 * sign * self._logdet)

    def _unstandardize(self, value: EventT) -> EventT:
        # Get the square-root of the exponentiated scale:
        #  sqrt(exp(C)) = sqrt(V exp(F) V^T) = V exp(0.5 F)
        if self.is_diag:
            scale = 1.0 / self.scale if self.inverted else self.scale
            return self.mean() + scale * value

        if self.inverted:
            return self.mean() + jax.scipy.linalg.solve_triangular(
                self.scale.T, value, lower=False
            )

        return self.mean() + self.scale @ value

    def _standardize(self, value: EventT) -> EventT:
        # Get the square-root of the inverse-exponentiated scale:
        #  sqrt(inv(exp(C))) = sqrt(V exp(-F) V^T) = V exp(-0.5 F)
        if self.is_diag:
            scale = self.scale if self.inverted else 1.0 / self.scale
            return (value - self.mean()) * scale

        if self.inverted:
            return (value - self.mean()) @ self.scale

        return jax.scipy.linalg.solve_triangular(
            self.scale, value - self.mean(),
            lower=True
        )


class MultivariateNormalPrecision(MultivariateNormalTriangular):

    def __init__(
            self,
            loc,
            precision,
            diagonal: bool = False,
            jitter: float = 1e-6
    ):
        if diagonal:
            inv_scale = jnp.sqrt(precision)
        else:
            inv_scale = jnp.linalg.cholesky(
                precision + jitter * jnp.eye(len(loc))
            )
        super().__init__(loc, inv_scale, diagonal, inverse=True)


class MultivariateNormalFullPrecision(MultivariateNormalPrecision):

    def __init__(
            self,
            loc,
            precision,
            jitter: float = 1e-6
    ):
        super().__init__(loc, precision, diagonal=False, jitter=jitter)


class MultivariateNormalDiagonalPrecision(MultivariateNormalPrecision):

    def __init__(
            self,
            loc,
            precision,
            jitter: float = 1e-6
    ):
        super().__init__(loc, precision, diagonal=True, jitter=jitter)


class MultivariateNormalCovariance(MultivariateNormalTriangular):

    def __init__(
            self,
            loc,
            covariance,
            diagonal: bool = False,
            jitter: float = 1e-6
    ):
        if diagonal:
            scale = jnp.sqrt(covariance)
        else:
            scale = jnp.linalg.cholesky(
                covariance + jitter * jnp.eye(len(loc))
            )
        super().__init__(loc, scale, diagonal, inverse=False)


class MultivariateNormalFullCovariance(MultivariateNormalCovariance):

    def __init__(
            self,
            loc,
            covariance,
            jitter: float = 1e-6
    ):
        super().__init__(loc, covariance, diagonal=False, jitter=jitter)


class MultivariateNormalDiagonalCovariance(MultivariateNormalCovariance):

    def __init__(
            self,
            loc,
            covariance,
            jitter: float = 1e-6
    ):
        super().__init__(loc, covariance, diagonal=True, jitter=jitter)


class MultivariateNormalExpEVs(MultivariateNormal, abc.ABC):
    """Parameterize a MVN where the eigenvalues are given in a log-scale.

    This parameterization can help stabilize gradients of the KL-divergence
    between different distributions.
    """

    def __init__(self, loc, scale, basis: jax.Array | None = None):
        super().__init__(loc, scale, diagonal=basis is None)
        self.basis = basis


class MultivariateNormalExpTriangular(MultivariateNormalExpEVs):
    """Parameterize a MVN according to N(m, L exp(S) L^T).

    The matrix S represents the log-eigenvalues S = diag(s) and L is a
    triangular matrix with only 1s on the diagonal such that its determinant
    is one (i.e., an unitriangular matrix).
    If no basis is given, we assume L = I_n the identity matrix, which implies
    a diagonal covariance. If L is given as its inverse inv(L), then `s` is
    also assumed to be negated.

    Note, the log-eigenvalues `s` are related to the LDL decomposition and
    represent the triangular matrix eigenvalues, and *not* the eigenvalues of
    the matrix A = LDL, since L is not orthogonal.

    TODO: Training with Variational RNNs is still unstable with this param.

    See also: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    def __init__(
            self,
            loc,
            scale,
            basis: jax.Array | None = None,
            inverse: bool = False
    ):
        super().__init__(loc, scale, jnp.tril(basis))
        # Whether basis & scale represents the inverse covariance.
        self._inverse = inverse

        raise RuntimeError(f"Do not use {type(self)}. Refactor this class!")

    @property
    def inverted(self):
        return self._inverse

    def _inv_root(self) -> jax.Array:
        return jax.scipy.linalg.solve_triangular(
            self.basis.T, jnp.diag(jnp.exp(-self.scale * 0.5)),
            lower=False
        )  # -> Upper-Triangular

    def covariance(self) -> jax.Array:
        if not self.inverted:
            return self.basis * jnp.exp(self.scale) @ self.basis.T

        inv_root = self._inv_root()
        return inv_root * inv_root.T

    def precision(self) -> jax.Array:
        if self.inverted:
            return self.basis * jnp.exp(self.scale) @ self.basis.T

        inv_root = self._inv_root()
        return inv_root * inv_root.T

    def kl_divergence(
            self,
            other: MultivariateNormalExpTriangular,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Computes the Kullback-Leibler Divergence of the distribution.

        Computed according to, D_KL[self, other]:
            D_KL[N(m, S) || N(v, C)] = 1/2 (tr(inv(C)S) + (v-m)inv(C)(v-m)
                                        -k + log det(C) - log det(S))

        then, for exp(C) = exp(UDU^T) and exp(S) = exp(VFV^T) we get,
            tr(inv(C)S) = tr(inv(exp(UDU^T))exp(VFV^T))
                        = tr(inv(U exp(D) U^T)V exp(F) V^T)
                        = tr(U exp(-D) U^T V exp(F) V^T)
                        = square( ||exp(-0.5D) U^T V exp(0.5F) ||_F )

            log det(C)  = sum(D)
            log det(S)  = sum(F)

        This gives the divergence,
            D_KL[N(m, S) || N(v, C)] = 1/2 (sum(exp(F - D)) - k + sum(D - F) +
                                        ||(v - m) U^T exp(-D/2) ||^2)
        """
        k = np.prod(self.event_shape)

        if other.is_diag and self.is_diag:
            scale_other = jnp.diag(jnp.exp(-0.5 * other.scale))
            trace = jnp.exp(self.scale - other.scale).sum()

        else:
            scale_other = jnp.eye(k) if other.is_diag else other._inv_root()
            scale_self = jnp.eye(k) if self.is_diag else \
                self.basis * jnp.exp(0.5 * self.scale)

            trace_sqrt = jnp.linalg.norm(scale_other.T @ scale_self, ord='fro')
            trace = jnp.square(trace_sqrt)

        mahalanobis_factor = (other.mean() - self.mean()) @ scale_other
        squared_mahalanobis_norm = mahalanobis_factor @ mahalanobis_factor

        log_determinant_delta = (other.scale - self.scale).sum()
        print('Warning: un-tested code is probably wrong!')
        kl2 = trace + squared_mahalanobis_norm - k + log_determinant_delta
        return 0.5 * kl2, {
            'trace': trace,
            'mahalanobis': squared_mahalanobis_norm,
            'determinant': log_determinant_delta
        }

    def mahalanobis_distance(
            self,
            other: MultivariateNormalExpTriangular,
            **kwargs
    ):
        k = np.prod(self.event_shape)

        if other.is_diag and self.is_diag:
            scale = jnp.diag(jnp.exp(-0.5 * other.scale))

        else:
            scale = jnp.eye(k) if other.is_diag else other._inv_root()

        mahalanobis_factor = (other.mean() - self.mean()) @ scale
        return mahalanobis_factor @ mahalanobis_factor

    def entropy(self) -> jax.Array:
        """Computes the Differential entropy of the distribution.

        Since we parameterize the MVN through an exponentiated spectral
        decomposition, the entropy can be computed as a simple sum of the
        eigenvalues.

        Computed according to:
            H[N(m, C)] = k/2 log(2*pi*e) + 1/2 log det(C),

        then, for exp(C) = exp(UDU^T) we get,
            H[N(m, exp(C))] = k/2 log(2*pi*e) + 1/2 log det(exp(C))
                            = k/2 log(2*pi*e) + 1/2 log prod(exp(D))
                            = k/2 log(2*pi*e) + 1/2 sum(D)

        This also works for triangular matrices C = LL^T,
            log det(exp(C)) = log det(exp(LL^T)) = log prod_i exp(L_ii)
                            = sum(diag(L)) = sum(D)
        """
        k = np.prod(self.event_shape)
        return 0.5 * (k + k * jnp.log(2 * jnp.pi) + jnp.sum(self.scale))

    def _unstandardize(self, value: EventT) -> EventT:
        # Get the square-root of the exponentiated scale:
        #  sqrt(exp(C)) = sqrt(V exp(F) V^T) = V exp(0.5 F)
        if self.is_diag:
            flip = -1 if self.inverted else +1
            return self.mean() + jnp.exp(0.5 * self.scale * flip) * value

        if self.inverted:
            scale_factor = self._inv_root()
        else:
            scale_factor = self.basis * jnp.exp(0.5 * self.scale)

        return self.mean() + scale_factor @ value

    def _standardize(self, value: EventT) -> EventT:
        # Get the square-root of the inverse-exponentiated scale:
        #  sqrt(inv(exp(C))) = sqrt(V exp(-F) V^T) = V exp(-0.5 F)
        if self.is_diag:
            flip = -1 if self.inverted else +1
            return self.mean() + value * jnp.exp(-0.5 * self.scale * flip)

        if self.inverted:
            scale_factor = self.basis * jnp.exp(0.5 * self.scale)
        else:
            scale_factor = self._inv_root()

        return (value - self.mean()) @ scale_factor


class MultivariateNormalExpOrthogonal(MultivariateNormalExpEVs):
    """Parameterize a MVN according to N(m, exp(S)), where S = UDU^T

    The matrix S should be given by the log-eigenvalues D = diag(d_1:n) and
    an orthogonal basis U. If no basis is given, we assume the identity
    matrix, which implies a diagonal covariance.

    Note, the basis of a symmetric semi-PD matrix is always orthogonal.

    See also: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    def covariance(self) -> jax.Array:
        if self.is_diag:
            return jnp.exp(self.scale)
        return self.basis * jnp.exp(self.scale) @ self.basis.T

    def precision(self) -> jax.Array:
        if self.is_diag:
            return jnp.exp(-self.scale)
        return self.basis * jnp.exp(-self.scale) @ self.basis.T

    def kl_divergence(
            self,
            other: MultivariateNormalExpOrthogonal,
            **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Computes the Kullback-Leibler Divergence of the distribution.

        Computed according to, D_KL[self, other]:
            D_KL[N(m, S) || N(v, C)] = 1/2 (tr(inv(C)S) + (v-m)inv(C)(v-m)
                                        -k + log det(C) - log det(S))

        then, for exp(C) = exp(UDU^T) and exp(S) = exp(VFV^T) we get,
            tr(inv(C)S) = tr(inv(exp(UDU^T))exp(VFV^T))
                        = tr(inv(U exp(D) U^T)V exp(F) V^T)
                        = tr(U exp(-D) U^T V exp(F) V^T)
                        = square( ||exp(-0.5D) U^T V exp(0.5F) ||_F )

            log det(C)  = sum(D)
            log det(S)  = sum(F)

        This gives the divergence,
            D_KL[N(m, S) || N(v, C)] =
                1/2 * (
                    sum(D - F) - k + ||(v - m) U^T exp(-D/2) ||^2
                    + square( ||exp(-0.5D) U^T V exp(0.5F) ||_F )
                )
        """
        k = np.prod(self.event_shape)

        if other.is_diag and self.is_diag:
            trace = (jnp.exp(self.scale - other.scale)).sum()
        else:
            basis_other = jnp.eye(k) if other.is_diag else other.basis
            basis_self = jnp.eye(k) if self.is_diag else self.basis

            scale_factor = basis_other * jnp.exp(-0.5 * other.scale)
            trace = jnp.square(
                scale_factor.T @ (basis_self * jnp.exp(0.5 * self.scale)),
            ).sum()

        log_determinant_delta = (other.scale - self.scale).sum()
        mahalanobis_norm = self.mahalanobis_distance(other)

        kl2 = trace + mahalanobis_norm - k + log_determinant_delta

        return 0.5 * kl2, {
            'trace': 0.5 * trace,
            'mahalanobis': 0.5 * mahalanobis_norm,
            'determinant': 0.5 * log_determinant_delta
        }

    def mahalanobis_distance(
            self,
            other: MultivariateNormalExpOrthogonal,
            **kwargs
    ):
        k = np.prod(self.event_shape)

        if other.is_diag and self.is_diag:
            scale = jnp.diag(jnp.exp(-0.5 * other.scale))
        else:
            basis_other = jnp.eye(k) if other.is_diag else other.basis
            scale = basis_other * jnp.exp(-0.5 * other.scale)

        mahalanobis_factor = (other.mean() - self.mean()) @ scale
        return mahalanobis_factor @ mahalanobis_factor

    def entropy(self) -> jax.Array:
        """Computes the Differential entropy of the distribution.

        Since we parameterize the MVN through an exponentiated spectral
        decomposition, the entropy can be computed as a simple sum of the
        eigenvalues.

        Computed according to:
            H[N(m, C)] = k/2 log(2*pi*e) + 1/2 log det(C),

        then, for exp(C) = exp(UDU^T) we get,
            H[N(m, exp(C))] = k/2 log(2*pi*e) + 1/2 log det(exp(C))
                            = k/2 log(2*pi*e) + 1/2 log prod(exp(D))
                            = k/2 log(2*pi*e) + 1/2 sum(D)

        This also works for triangular matrices C = LL^T,
            log det(exp(C)) = log det(exp(LL^T)) = log prod_i exp(L_ii)
                            = sum(diag(L)) = sum(D)
        """
        k = np.prod(self.event_shape)
        return 0.5 * (k + k * jnp.log(2 * jnp.pi) + jnp.sum(self.scale))

    def _unstandardize(self, value: EventT) -> EventT:
        # Get the square-root of the exponentiated scale:
        #  sqrt(exp(C)) = sqrt(V exp(F) V^T) = V exp(0.5 F)
        if self.is_diag:
            return self.mean() + jnp.exp(0.5 * self.scale) * value

        return self.mean() + (self.basis * jnp.exp(0.5 * self.scale)) @ value

    def _standardize(self, value: EventT) -> EventT:
        # Get the square-root of the inverse-exponentiated scale:
        #  sqrt(inv(exp(C))) = sqrt(V exp(-F) V^T) = V exp(-0.5 F)
        if self.is_diag:
            return (value - self.mean()) * jnp.exp(-0.5 * self.scale)

        scale_factor = self.basis.T * jnp.exp(-0.5 * self.scale)
        return (value - self.mean()) @ scale_factor
