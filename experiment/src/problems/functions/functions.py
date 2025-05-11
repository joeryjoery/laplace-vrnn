import flax.linen as nn

import jax
import jax.numpy as jnp

from axme.core import Scope


class Slope(nn.Module):
    """Generate Linear Functions with random slopes."""
    bound: tuple[jax.Array, jax.Array] = (-5.0, 5.0)

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        shift = self.param(
            'shift',
            nn.initializers.normal(),
            (), x.dtype
        )
        slope = self.param(
            'slope',
            nn.initializers.uniform(scale=self.bound[1] - self.bound[0]),
            x.shape, x.dtype
        ) + self.bound[0]

        return (x * slope).sum() + shift


class Fourier(nn.Module):
    """Generate random functions as the sum of randomized sine waves

    i.e. (simplified),
        z = scale * x - shift,
        y = a_0 + sum_(i=1...n) a_i cos(2pi * i * z - phi_i),
    where a_i are amplitudes and phi_i are phase shifts.

    See the Amplitude-Phase form,
     - https://en.wikipedia.org/wiki/Fourier_series
    """
    n: int = 3
    amplitude: float = 1.0
    period: float = 1.0

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        a = self.param(
            'amplitude',
            nn.initializers.uniform(scale=self.amplitude * 2),
            (self.n, *x.shape), x.dtype
        ) - 1.0
        shift = self.param(
            'shift',
            nn.initializers.uniform(scale=jnp.pi),
            jnp.shape(x), x.dtype
        )
        phase = self.param(
            'phase',
            nn.initializers.uniform(scale=jnp.pi),
            (self.n - 1, *jnp.shape(x)), x.dtype
        )

        x = x - shift

        # (heuristic) Monotonically scale amplitudes.
        a = a * jnp.arange(1, len(a) + 1) / (self.n - 1)

        waves = jnp.cos(
            (2 * jnp.pi * jnp.arange(1, self.n) * x - phase) / self.period
        )

        return a[0] / 2.0 + jnp.sum(a[1:] * waves)


class DiscreteFourier(Fourier):
    """Generate discretized random functions with Fourier

    Rounds off the base class implementation to the nearest step-size.

    See Fourier.
    """
    step_size: float = 0.25

    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        out = super().__call__(x)
        step = jnp.clip(self.step_size, a_min=1e-4)
        return jnp.round(out / step) * step


class Norm(nn.Module):
    """Generate random functions as the l2 norm of an affine input transform

    i.e.,
        y = ||Sx + b||^2,
    where S is a diagonal matrix.
    """
    shift_bounds: tuple[jax.Array, jax.Array] = (-1.0, 1.0)
    scale_bounds: tuple[jax.Array, jax.Array] = (0.5, 2.0)

    flip_sign: bool = True

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:

        shift = self.param(
            'shift',
            nn.initializers.uniform(
                scale=self.shift_bounds[1] - self.shift_bounds[0]
            ),
            x.shape, x.dtype
        )
        shift = shift + self.shift_bounds[0]

        scale = self.param(
            'scale',
            nn.initializers.uniform(
                scale=self.scale_bounds[1] - self.scale_bounds[0]
            ),
            x.shape, x.dtype
        )
        scale = scale + self.scale_bounds[0]

        flip = 1.0
        if self.flip_sign:
            # 50/50 positive/ negative sign
            flip = self.param(
                'flip',
                nn.initializers.normal(),
                (), jnp.float32
            )

        return jnp.square(scale * x + shift).sum() * jnp.sign(flip)


class Rational(nn.Module):
    """Generate random functions as the fraction of two random polynomials

    i.e.,
        y = (a_n x^n + ... a_1 x + a_0) / (b_m x^m + ... + b_1 x + b_0),
    this is implemented in terms of a log series,
        a_n x^n \approx sign(x)^n * exp(log(a_n+1) + n * log(|x|))

    """
    numerator_order: int = 3
    denominator_order: int = 2
    scale_bounds: tuple[jax.Array, jax.Array] = (-5.0, 5.0)

    # Beware of asymptotes
    clip_bounds: tuple[float, float] = (-10.0, 10.0)

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:

        order = self.param(
            'order',
            nn.initializers.uniform(scale=1.0),
            (self.numerator_order + self.denominator_order + 1,),
            x.dtype
        )
        num_order, den_order = jnp.split(
            order, (self.numerator_order + 1,)
        )

        scale = self.param(
            'scale',
            nn.initializers.uniform(
                scale=self.scale_bounds[1] - self.scale_bounds[0]
            ),
            (self.numerator_order + self.denominator_order + 1,),
            x.dtype
        )
        scale = scale + self.scale_bounds[0]
        num_scale, den_scale = jnp.split(
            scale, (self.numerator_order + 1,)
        )
        num_scale = num_scale * (num_order > 0.5)
        den_scale = den_scale * (den_order > 0.5)

        # Get coefficient series
        num_series = jnp.arange(self.numerator_order + 1)
        den_series = jnp.arange(1, self.denominator_order + 1)

        # Get signs
        sign_num = jnp.power(jnp.sign(x), num_series)
        sign_den = jnp.power(jnp.sign(x), den_series)

        log_num = (num_order > 0.5) * num_series * jnp.log(1e-4 + jnp.abs(x))
        log_den = (den_order > 0.5) * den_series * jnp.log(1e-4 + jnp.abs(x))

        y = (jnp.sum(sign_num * num_scale * jnp.exp(log_num)) /
             jnp.sum(1 + sign_den * den_scale * jnp.exp(log_den)))

        return jnp.clip(y, *self.clip_bounds)


class Polynomial(nn.Module):
    order: int = 1

    scale_bounds: tuple[jax.Array, jax.Array] = (-5.0, 5.0)

    # Beware of asymptotes
    clip_bounds: tuple[float, float] = (-10.0, 10.0)

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:

        scale = self.param(
            'scale',
            nn.initializers.uniform(
                scale=self.scale_bounds[1] - self.scale_bounds[0]
            ),
            (self.order + 1,),
            x.dtype
        )
        scale = scale + self.scale_bounds[0]

        # Get polynomial expansion
        power_series = jnp.power(x, jnp.arange(self.order + 1))

        return jnp.clip((scale * power_series).sum(), *self.clip_bounds)


class Mixture(nn.Module):
    """Utility to allow Module branching based on a given index.

    See the flax doc on flax.linen.switch for more info.
    """
    branches: list[nn.Module]

    @nn.compact
    def __call__(self, *args, index: int | None):
        def head_fn(i):
            return lambda mdl, *ins: mdl.branches[i](*ins)
        branches = [head_fn(i) for i in range(len(self.branches))]

        # run all branches on init
        if self.is_mutable_collection(Scope.Params):
            for branch in branches:
                _ = branch(self, *args)

        if index is None:
            index = jax.random.randint(
                self.make_rng(type(self).__name__), (), 0, len(self.branches)
            )

        return nn.switch(index, branches, self, *args)
