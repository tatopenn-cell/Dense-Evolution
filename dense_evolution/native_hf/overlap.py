"""Overlap integrals between Gaussian shells via the Obara-Saika recursion.

For two 1D Gaussians centered at A, B with exponents a, b, let
p = a+b and P = (aA+bB)/p be the center of their product (Gaussian
product theorem). The overlap S[i,j] = <(x-A)^i exp(-a(x-A)^2) |
(x-B)^j exp(-b(x-B)^2)> obeys two recursions:

  Vertical (build up the first index from the base case S[0,0]):
    S[i,0] = (P-A) S[i-1,0] + (i-1)/(2p) S[i-2,0]

  Horizontal (build up the second index by shifting angular momentum
  from center A to center B, exact for any two centers -- this is why
  it needs no exponent-dependent term):
    S[i,j] = (A-B) S[i,j-1] + S[i+1,j-1]

Both are linear recursions in one index with a 2-term memory, so each
maps directly onto jax.lax.scan: the whole angular-momentum ladder for
a shell pair compiles to one XLA loop instead of a Python for-loop.
"""

import functools

import jax
import jax.numpy as jnp

from dense_evolution.native_hf.gaussians import GaussianShell1D, GaussianShell3D


def _base_overlap_1d(g1: GaussianShell1D, g2: GaussianShell1D) -> jax.Array:
    p = g1.exponent + g2.exponent
    mu = (g1.exponent * g2.exponent) / p
    K = jnp.exp(-mu * jnp.square(g1.center - g2.center))
    return jnp.sqrt(jnp.pi / p) * K


def _vertical_step(p_minus_a, inv_two_p, carry, i):
    s_prev, s_prev2 = carry
    s_next = p_minus_a * s_prev + (i - 1) * inv_two_p * s_prev2
    return (s_next, s_prev), s_next


def _build_first_index(s00, g1: GaussianShell1D, g2: GaussianShell1D, n_extra: int) -> jax.Array:
    """Returns S[i, 0] for i = 0..n_extra."""
    if n_extra == 0:
        return s00[None]
    p = g1.exponent + g2.exponent
    P = (g1.exponent * g1.center + g2.exponent * g2.center) / p
    step = functools.partial(_vertical_step, P - g1.center, 1.0 / (2.0 * p))
    _, rest = jax.lax.scan(step, (s00, jnp.zeros_like(s00)), jnp.arange(1, n_extra + 1))
    return jnp.concatenate([s00[None], rest])


def _horizontal_step(a_minus_b, column, _):
    shifted = jnp.roll(column, shift=-1)
    new_column = a_minus_b * column + shifted
    return new_column, new_column


def _build_second_index(first_col: jax.Array, g1: GaussianShell1D, g2: GaussianShell1D) -> jax.Array:
    """first_col holds S[:,0]. Returns S[:, 0..g2.degree] (rows beyond
    validity are garbage and get sliced away by the caller)."""
    if g2.degree == 0:
        return first_col[:, None]
    step = functools.partial(_horizontal_step, g1.center - g2.center)
    _, rest = jax.lax.scan(step, first_col, jnp.arange(1, g2.degree + 1))
    return jnp.concatenate([first_col[:, None], jnp.moveaxis(rest, 0, -1)], axis=-1)


def overlap_1d(g1: GaussianShell1D, g2: GaussianShell1D) -> jax.Array:
    """S[i,j] for 0<=i<=g1.degree, 0<=j<=g2.degree. Shape (g1.degree+1, g2.degree+1)."""
    s00 = _base_overlap_1d(g1, g2)
    first_col = _build_first_index(s00, g1, g2, g1.degree + g2.degree)
    full = _build_second_index(first_col, g1, g2)
    return full[: g1.degree + 1, :]


@jax.jit
def overlap_3d(g1: GaussianShell3D, g2: GaussianShell3D) -> jax.Array:
    """S[ix,iy,iz,jx,jy,jz] for the full Cartesian shell pair.

    Shape: (L1+1,L1+1,L1+1, L2+1,L2+1,L2+1) where L1, L2 are the shell
    degrees (unphysical (i,j,k) combinations with i+j+k > degree are
    simply never read by the caller).
    """
    axes = [overlap_1d(g1.component(d), g2.component(d)) for d in range(3)]
    return jnp.einsum("ad,be,cf->abcdef", *axes)
