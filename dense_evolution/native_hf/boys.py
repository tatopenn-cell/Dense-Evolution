"""The Boys function F_n(x), needed to evaluate any integral involving
1/r12 (nuclear attraction, electron repulsion) between Gaussians.

    F_n(x) = integral_0^1  t^(2n) exp(-x t^2) dt

which can be written in closed form via the regularized lower incomplete
gamma function P:

    F_n(x) = Gamma(n + 1/2) * P(n + 1/2, x) / (2 * x^(n + 1/2))

For x -> 0 this formula divides 0/0, so we fall back to the first-order
Taylor expansion F_n(x) ~= 1/(2n+1) - x/(2n+3), which is accurate to
better than machine epsilon once x is small enough that no other term
in the calculation could still be sensitive to it.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln

_TAYLOR_CUTOFF = 1e-12


def boys(n: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate F_n(x) elementwise. n and x broadcast against each other."""
    x = jnp.asarray(x)
    n = jnp.asarray(n)

    x_safe = jnp.where(x < _TAYLOR_CUTOFF, 1.0, x)
    log_gamma = gammaln(n + 0.5)
    closed_form = jnp.exp(log_gamma) * gammainc(n + 0.5, x_safe) / (2.0 * x_safe ** (n + 0.5))

    taylor = 1.0 / (2.0 * n + 1.0) - x / (2.0 * n + 3.0)

    return jnp.where(x < _TAYLOR_CUTOFF, taylor, closed_form)
