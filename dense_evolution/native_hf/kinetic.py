"""Kinetic energy integrals, obtained from overlap integrals for free.

The second derivative of a Gaussian (x-B)^j exp(-b(x-B)^2) with respect
to its own center reduces to a combination of Gaussians of degree j-2,
j and j+2, which gives kinetic integrals as a fixed linear combination
of overlap integrals evaluated at a boosted degree:

    T[i,j] = j(j-1) S[i,j-2] - 2b(2j+1) S[i,j] + 4b^2 S[i,j+2]

T here is <i| d^2/dx^2 |j>; the physical kinetic energy operator is
-1/2 (d^2/dx^2 + d^2/dy^2 + d^2/dz^2), so callers must negate and halve
the sum of the three Cartesian terms.
"""

import jax
import jax.numpy as jnp

from dense_evolution.native_hf.gaussians import GaussianShell3D
from dense_evolution.native_hf.overlap import overlap_1d


def _kinetic_1d_from_overlap(S: jax.Array, b: jax.Array) -> jax.Array:
    """S has shape (n_i, n_j) with n_j >= 3 (degree-boosted). Returns the
    kinetic matrix of shape (n_i, n_j - 2)."""
    n_j_out = S.shape[1] - 2
    j = jnp.arange(n_j_out)[None, :]

    term_down = jnp.pad(j[:, 2:] * (j[:, 2:] - 1) * S[:, :-4], ((0, 0), (2, 0))) if n_j_out > 2 else jnp.zeros((S.shape[0], n_j_out))
    term_same = -2.0 * b * (2.0 * j + 1.0) * S[:, :-2]
    term_up = 4.0 * b * b * S[:, 2:]

    return term_down + term_same + term_up


@jax.jit
def kinetic_3d(g1: GaussianShell3D, g2: GaussianShell3D) -> jax.Array:
    """<g1| d^2/dx^2 + d^2/dy^2 + d^2/dz^2 |g2>, shape matching overlap_3d.

    Multiply by -0.5 to get the physical kinetic-energy matrix elements.
    """
    b = jnp.asarray(g2.exponent)
    g2_boosted = GaussianShell3D(degree=g2.degree + 2, exponent=g2.exponent, center=g2.center)

    S = [overlap_1d(g1.component(d), g2_boosted.component(d)) for d in range(3)]
    T = [_kinetic_1d_from_overlap(S[d], b) for d in range(3)]
    S_trim = [s[:, :-2] for s in S]

    term_x = jnp.einsum("ad,be,cf->abcdef", T[0], S_trim[1], S_trim[2])
    term_y = jnp.einsum("ad,be,cf->abcdef", S_trim[0], T[1], S_trim[2])
    term_z = jnp.einsum("ad,be,cf->abcdef", S_trim[0], S_trim[1], T[2])

    return term_x + term_y + term_z
