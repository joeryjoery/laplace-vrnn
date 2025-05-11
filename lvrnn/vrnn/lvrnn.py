from __future__ import annotations
from typing import Any, Literal, Sequence, TYPE_CHECKING
from functools import partial
from abc import ABC

from jaxtyping import PyTree, PRNGKeyArray

import flax.linen as nn

import jax
import jax.numpy as jnp

from axme.core import Scope

from lvrnn.distributions import SerializeTree, mvn
from .interface import RecurrentLatentVariableModel, RLVMState, StateAdapter

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class LVRNNState:
    precision: jax.Array
    step: int


@dataclass
class HistoryLVRNNState(LVRNNState):
    inputs: jax.Array


@dataclass
class ApproxLVRNNState(LVRNNState):
    jacobian_sum: jax.Array


class JointCore(nn.Module):
    core: nn.RNNCellBase
    var_mean: nn.Module
    adapter: StateAdapter

    @nn.compact
    def __call__(
            self,
            cell: PyTree[jax.Array],
            inputs: jax.Array
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        """Wraps an RNN implementation with a Linear output Projection. """
        new_cell, out = self.core(cell, inputs)
        mean = self.var_mean(out)
        return new_cell, mean

    def jacfun(
            self,
            cell: PyTree[jax.Array],
            inputs: jax.Array
    ) -> jax.Array:
        """Compute the Jacobian of an RNN-update step.

        Given a hidden state (cell) Z = (c, s) where c can be a hidden
        component, and s an output component. For example, the LSTM: c = cell
        and s = hidden state, for GRU: c = None, s = hidden state.

        Evaluate the Jacobian nabla_s f(Z, x) at input x.
        """

        _, jac_mean = jax.jacfwd(self, argnums=0)(cell, inputs)
        _, jac_hidden_mean = self.adapter.split_transformable(jac_mean)

        return jac_hidden_mean


class LaplaceVRNN(RecurrentLatentVariableModel, ABC):
    """Extend an RNN core to get its Laplace Approximated state Posterior."""

    # Stddev of the Gaussian prior at t=0
    prior_stddev: float = 1.0

    # Whether to model a diagonal precision
    diagonal: bool = False

    # Whether to scale the precision by `n` to model the mean distribution.
    normalize: bool = False

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState[mvn.MultivariateNormal, HistoryLVRNNState]:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, _ = self.adapter.split_transformable(cell)

        precision = jnp.full(self.features, 1.0 / self.prior_stddev)
        if not self.diagonal:
            precision = jnp.diag(precision)

        mvn = self._parameters_to_mvn(jnp.zeros(self.features), precision)
        init_state = (cell, mvn)

        return RLVMState(
            cell=init_state,
            state=LVRNNState(precision=precision, step=0)
        )

    def _compute_metrics(
            self,
            posterior: mvn.MultivariateNormal,
            prior: mvn.MultivariateNormal,
            jacobians: jax.Array,
            inner_state: Any
    ) -> dict[str, jax.Array]:
        placeholder = jnp.zeros()
        return {
            k: placeholder for k in (
                'average_jacobian_variance',
                'norm_average_jacobian',
                'norm_step_jacobian',
                'norm_mean_difference',
                'norm_step_difference',
                'pseudo_score',
                'marginal_entropy',
                'differential_entropy',
                'step'
            )
        }

    def _parameters_to_mvn(
            self,
            mean,
            precision,
            jitter: float = 1e-6
    ) -> SerializeTree[mvn.MultivariateNormal]:
        if self.diagonal:
            return SerializeTree(
                mvn.MultivariateNormalTriangular,
                mean, jax.lax.rsqrt(precision + jitter),
                static_kwargs=dict(diagonal=True, inverse=False)
            )

        chol = jnp.linalg.cholesky(
            precision + jitter * jnp.eye(len(precision))
        )
        return SerializeTree(
            mvn.MultivariateNormalTriangular,
            mean, chol,
            static_kwargs=dict(diagonal=False, inverse=True)
        )


class HistoryLaplaceVRNN(LaplaceVRNN):
    """Computes the Laplace Approximation with the historical trajectory.

    This model assumes a Gaussian predictive around the core outputs.

    Since this model is non-parametric, this can lead to memory-issues when
    taking gradients and requires tuning of the memory-buffer.
    """
    buffer_size: int = 100

    # damping: Diagonal constant added to all precisions if not accumulated.
    damping: float = 1e-3
    accumulate: Literal['precision', 'mean', 'both', 'None'] | None = None

    def _compute_metrics(
            self,
            posterior: mvn.MultivariateNormal,
            prior: mvn.MultivariateNormal,
            jacobians: jax.Array,
            step: int
    ) -> dict[str, dict[str, jax.Array] | jax.Array]:

        # Differential Entropy for a diagonal covariance
        marginal_entropy = 0.5 * (
                1 + jnp.log(2 * jnp.pi) + jnp.log(posterior.variance())
        )
        kl_div, kl_metrics = posterior.kl_divergence(prior)

        # Estimate a pseudo-score function and log related statistics.
        delta_nll = posterior.mean() - prior.mean()

        euclidean = (delta_nll ** 2).sum()
        pseudo_score = jax.vmap(jnp.multiply, in_axes=(0, None))(
            jacobians, delta_nll[..., None]
        ).sum()

        norm_average_jacobian = jnp.sum(((jacobians.sum(axis=0) / step) ** 2))
        average_norm_jacobian = jnp.sum(
            jnp.square(jacobians).sum(axis=0) / step
        )
        norm_step_jacobian = (jacobians.at[step - 1].get() ** 2).sum()

        return {
            'jacobian': {
                'norm_of_average': norm_average_jacobian,
                'norm_of_step': norm_step_jacobian,
                'average_norm': average_norm_jacobian
            },
            'pseudo_score': {
                'value': pseudo_score,
                'euclidean': euclidean
            },
            'posterior_entropy': {
                'marginal': marginal_entropy.mean(),
                'differential': posterior.entropy()
            },
            'kl_divergence': {'value': kl_div, **kl_metrics},
            'step': step
        }

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState[mvn.MultivariateNormal, HistoryLVRNNState]:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, _ = self.adapter.split_transformable(cell)

        precision = jnp.full(self.features, 1.0 / self.prior_stddev)
        if not self.diagonal:
            precision = jnp.diag(precision)

        mvn_dist = self._parameters_to_mvn(jnp.zeros(self.features), precision)
        init_state = (cell, mvn_dist)

        buffer = jnp.zeros((self.buffer_size, *input_shape))

        return RLVMState(
            cell=init_state,
            state=HistoryLVRNNState(precision=precision, step=0, inputs=buffer)
        )

    def _step(self, cell, inputs, history, get_jacobians: bool = True):
        """Helper method to compute outputs for the Laplace Approximation.

        We wrap the RNN-core and linear output layer in one module and lift
        the parameters to the current scope to compute the step and jacobian
        functions as pure functions. This implementation does a bit more work
        compared to flax.linen.vmap or flax.linen.jvp, but is much simpler
        to test for correctness.

        TODO: Current implementation is incompatible with apply RNGs.
        """
        joint_core = JointCore(
            self.core,
            nn.Dense(self.features),
            self.adapter
        )

        core_params = self.param(
            self.__class__.__name__,
            lambda *a: joint_core.init(*a)[Scope.Params],
            cell, inputs
        )

        new_cell, mean = joint_core.apply(
            {Scope.Params: core_params},
            cell, inputs
        )

        if not get_jacobians:
            return new_cell, mean

        # Note, the Jacobians are pseudo-statistics due to recurrent
        # non-stationarity. The RNN is also not consistent.
        jacobians = jax.vmap(
            partial(joint_core.apply, method=joint_core.jacfun),
            in_axes=(None, None, 0)
        )({Scope.Params: core_params}, new_cell, history)

        return new_cell, mean, jacobians

    @nn.compact
    def __call__(
            self,
            prev_state: RLVMState[mvn.MultivariateNormal, HistoryLVRNNState],
            inputs: jax.Array
    ) -> tuple[
        RLVMState[mvn.MultivariateNormal, HistoryLVRNNState],
        SerializeTree[mvn.MultivariateNormal]
    ]:
        # 1) State-management
        cell, prev_out = prev_state.cell
        lvrnn_state = prev_state.state

        # Add input to history-buffer (wraps circularly!).
        idx = lvrnn_state.step % self.buffer_size
        history = lvrnn_state.inputs.at[idx].set(inputs)
        mask = jnp.arange(self.buffer_size) <= lvrnn_state.step

        # 2) Computation of posterior parameters and core-transition.
        new_cell, mean, jacobians = self._step(cell, inputs, history)

        # Zero out buffer-elements that are still unobserved.
        jacobians = jax.vmap(jnp.multiply)(jacobians, mask)

        # Precision: sum_i jac_i @ jac_i.T, dim(jac) = (batch, out, in)
        if self.diagonal:
            # Sum: batch-wise + column-wise.
            precision = jnp.sum(jacobians * jacobians, axis=(0, 2))
        else:
            # Sum: batch-wise
            precision = jnp.einsum('abc,adc->bd', jacobians, jacobians)

        if self.normalize:
            n = jnp.minimum(lvrnn_state.step + 1, self.buffer_size)
            precision = precision / n

        # The mean should not optimize for the covariance/ precision.
        # Disabling this will also slow down training with orders of magnitude.
        precision = jax.lax.stop_gradient(precision)

        if (self.accumulate == 'precision') or (self.accumulate == 'both'):
            # Option A: Recurrent Convolution of Gaussian Distributions
            #   -> Accumulate Precision matrices.
            # Option B: Dampen the precision with an isotropic prior.
            #   -> Fit around a MAP solution
            prior = lvrnn_state.precision  # Previous-Precision
        else:
            prior = jnp.full(self.features, self.damping)
            if not self.diagonal:
                prior = jnp.diag(prior)

        if (self.accumulate == 'mean') or (self.accumulate == 'both'):
            # Recurrent Convolution of Gaussian Distributions
            #  -> Accumulate Means
            # Use a stop-gradient on the previous mean so that previous
            # steps are not optimized to improve future steps.
            mean = mean + jax.lax.stop_gradient(prev_out.get.mean())

        # 3) Parametrize Gaussian Posterior given Parameters
        posterior_precision = prior + precision
        dist = self._parameters_to_mvn(mean, posterior_precision)

        # 4) Optional Logging
        if self.is_mutable_collection(Scope.Intermediates):
            metrics = self._compute_metrics(
                dist.get,
                prev_out.get,
                jacobians,
                lvrnn_state.step + 1
            )
            self.sow(
                Scope.Intermediates,
                type(self).__name__ + '/metrics',
                metrics
            )

        new_lvrnn_state = HistoryLVRNNState(
            precision=posterior_precision, step=lvrnn_state.step + 1,
            inputs=history
        )

        return RLVMState(cell=(new_cell, dist), state=new_lvrnn_state), dist


class LinearizedLaplaceVRNN(LaplaceVRNN):
    """Uses a recursive first-order approximation to the full Hessian.

    Computes the cumulative sum of the Precision matrices with a
    linearized correction term.
    """
    # TODO: Always enable this through a custom JVP implementation.
    #  From testing, the autodiff compiles the jacobian computation within
    #  autodiff. This should not happen as it makes testing super slow.
    #  Disabling the jacobian accumulation makes the model an order of ~2
    #  faster.
    stop_jacobian_gradients: bool = True

    def _compute_metrics(
            self,
            posterior: mvn.MultivariateNormal,
            prior: mvn.MultivariateNormal,
            jacobian_state: tuple[jax.Array, jax.Array],
            core_state: Any,
            step: int
    ) -> dict[str, jax.Array]:
        # State unpacking
        jacobian, info_jacobian = jacobian_state

        marginal_entropy = 0.5 * (
                1 + jnp.log(2 * jnp.pi) + jnp.log(posterior.variance())
        )

        # Estimate sample-score related statistics
        delta_mu = posterior.mean() - prior.mean()
        norm_step_difference = jnp.square(delta_mu).sum()
        norm_step_jacobian = jnp.square(jacobian).sum()
        pseudo_score = jacobian @ delta_mu.sum()

        avg_jacobian_state_norm = jnp.square(info_jacobian).sum() / step

        return {
            'norm_step_jacobian': norm_step_jacobian,
            'norm_step_difference': norm_step_difference,
            'pseudo_score': pseudo_score,
            'marginal_entropy': marginal_entropy,
            'differential_entropy': posterior.entropy(),
            'kl_divergence': prior.kl_divergence(posterior),
            'jacobian_state_norm': avg_jacobian_state_norm,
            'step': step
        }

    def initialize_carry(
            self, rng: PRNGKeyArray, input_shape: Sequence[int]
    ) -> RLVMState[mvn.MultivariateNormal, ApproxLVRNNState]:
        cell = self.core.initialize_carry(rng, tuple(input_shape))
        cell, out = self.adapter.split_transformable(cell)

        precision = jnp.full(self.features, 1.0 / self.prior_stddev)
        if not self.diagonal:
            precision = jnp.diag(precision)

        mvn_dist = self._parameters_to_mvn(jnp.zeros(self.features), precision)
        init_state = (cell, mvn_dist)

        # Jacobian of precision w.r.t. hidden state dim: (n, n, out)
        jacobian_sum = jnp.zeros(
            (self.features,) * (2 - int(self.diagonal)) + out.shape
        )

        return RLVMState(
            cell=init_state, state=ApproxLVRNNState(
                precision=precision, step=0, jacobian_sum=jacobian_sum
            )
        )

    def _precision_and_jacfun(
            self,
            cell: PyTree[jax.Array],
            inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        jacobian = self.jacfun(cell, inputs)

        if self.diagonal:
            # Sum: column-wise.
            precision = jnp.sum(jacobian * jacobian, axis=1)
        else:
            precision = jnp.einsum('bc,dc->bd', jacobian, jacobian)

        return precision, jacobian

    @nn.compact
    def __call__(
            self,
            prev_state: RLVMState[mvn.MultivariateNormal, ApproxLVRNNState],
            inputs: jax.Array
    ) -> tuple[
        RLVMState[mvn.MultivariateNormal, ApproxLVRNNState],
        SerializeTree[mvn.MultivariateNormal]
    ]:
        # 1) State-management
        cell, prev_out = prev_state.cell
        lvrnn_state = prev_state.state

        # 2) Computation of posterior parameters and core-transition.
        new_cell, mean = self.joint_eval(cell, inputs)

        # Get the precision matrix at the current timestep.
        precision, jacobian = self._precision_and_jacfun(cell, inputs)
        if self.stop_precision_gradients:
            precision = jax.lax.stop_gradient(precision)

        # Extract linearization points from RNN states.
        _, old_phi = self.adapter.split_transformable(cell)
        _, phi = self.adapter.split_transformable(new_cell)

        # 1st Order Taylor-correction based on historical Jacobians.
        correction = lvrnn_state.jacobian_sum @ (phi - old_phi)
        # TODO: Symmetry of correction?
        posterior_precision = precision + correction

        if self.normalize:
            # TODO: rework this snippet to check how to accumulatively scale.
            n = jnp.minimum(lvrnn_state.step, self.buffer_size)
            posterior_precision = precision / n

        # Accumulate the tangents for the stepwise Precision matrices.
        tangent_cell, _ = jax.jacfwd(self._precision_and_jacfun, has_aux=True)(
            cell, inputs
        )
        _, tangent = self.adapter.split_transformable(tangent_cell)

        if self.stop_jacobian_gradients or self.stop_precision_gradients:
            tangent = jax.lax.stop_gradient(tangent)

        # TODO: Implement custom JVP for compute-efficiency.
        jacobian_sum = tangent + lvrnn_state.jacobian_sum
        # jacobian_sum = lvrnn_state.jacobian_sum

        # 3) Parametrize Gaussian Posterior given Parameters
        dist = self._parameters_to_mvn(mean, posterior_precision)

        # 4) Optional Logging
        if self.is_mutable_collection(Scope.Intermediates):
            metrics = self._compute_metrics(
                dist.get,
                prev_out.get,
                (jacobian, jacobian_sum),
                new_cell,
                lvrnn_state.step + 1
            )

            self.sow(Scope.Intermediates, type(self).__name__, metrics)

        new_lvrnn_state = ApproxLVRNNState(
                precision=posterior_precision,
                step=lvrnn_state.step + 1,
                jacobian_sum=jacobian_sum
            )

        return RLVMState(cell=(new_cell, dist), state=new_lvrnn_state), dist
