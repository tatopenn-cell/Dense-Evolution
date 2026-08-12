"""Nuclear attraction and electron repulsion integrals.

Both integral types reduce to the same building block: the n-th order
Hermite Coulomb integral

    V_n(P, C) = K * F_n(p |P-C|^2)

where F_n is the Boys function, P the center of a Gaussian product and
C either a nuclear position (one-electron case) or the center of a
second electron pair's product Gaussian (two-electron case). Angular
momentum on each of up to four centers is then built up from V_n by
three kinds of linear recursion (Obara-Saika):

  * vertical transfer  -- raises angular momentum on the "bra" center
    while also shifting the Boys-function order n. This is the only
    recursion that touches the Boys function directly.
  * horizontal transfer -- shifts angular momentum from one center to
    its partner on the same electron (exact via Gaussian product
    translation, same recursion used for overlap integrals).
  * electron transfer -- shifts angular momentum from electron 1's
    pair to electron 2's pair (only needed for the two-electron
    repulsion integral).

Each recursion has a 2-term memory in its recursion index, so each maps
onto jax.lax.scan.
"""

import functools

import jax
import jax.numpy as jnp

from dense_evolution.native_hf.boys import boys
from dense_evolution.native_hf.gaussians import GaussianShell3D, product_center, product_prefactor


def _hermite_base(order: int, g1: GaussianShell3D, g2: GaussianShell3D, scale: jax.Array, C: jax.Array) -> jax.Array:
    """V[n,0,0,0] for n = 0..order-1, i.e. the (0,0,0) angular-momentum
    slice of the Hermite Coulomb integral at every Boys order we'll need."""
    K = product_prefactor(g1, g2)
    P = product_center(g1, g2)
    dist_sq = jnp.sum(jnp.square(P - C))
    orders = jnp.arange(order)
    return K * boys(orders, scale * dist_sq)


def _vertical_step(scale, p, PA, PC, carry, i):
    v_prev, v_prev2 = carry
    v_prev_shift = jnp.roll(v_prev, shift=-1, axis=0)
    v_prev2_shift = jnp.roll(v_prev2, shift=-1, axis=0)

    v_next = (
        PA * v_prev
        - (scale / p) * PC * v_prev_shift
        + ((i - 1) / (2.0 * p)) * v_prev2
        - (((i - 1) * scale) / (2.0 * p * p)) * v_prev2_shift
    )
    return (v_next, v_prev), v_next


def _raise_bra_degree(V0: jax.Array, degree: int, scale: jax.Array, p: jax.Array, A: jax.Array, C: jax.Array, P: jax.Array) -> jax.Array:
    """Appends a new trailing axis of size `degree` to V0 built via the
    vertical transfer recursion (one Cartesian direction at a time --
    call this three times, once per x/y/z, to build a full 3D shell)."""
    step = functools.partial(_vertical_step, scale, p, P - A, P - C)
    init = (V0, jnp.zeros_like(V0))
    _, rest = jax.lax.scan(step, init, jnp.arange(1, degree), unroll=True)
    return jnp.concatenate((V0[..., None], jnp.moveaxis(rest, 0, -1)), axis=-1)


def _hermite_coulomb_tensor(g1: GaussianShell3D, g2: GaussianShell3D, scale: jax.Array, C: jax.Array) -> jax.Array:
    """The (degree+1)^3 tensor V[ix,iy,iz] = V_0-order Hermite integral
    with all angular momentum on the bra (g1) side, at Boys order 0."""
    a, A = jnp.asarray(g1.exponent), jnp.asarray(g1.center)
    b, B = jnp.asarray(g2.exponent), jnp.asarray(g2.center)
    p = a + b
    P = (a * A + b * B) / p

    if g1.degree == 0:
        return _hermite_base(1, g1, g2, scale, C)[0, None, None, None]

    n = g1.degree + 1
    V = _hermite_base(3 * n, g1, g2, scale, C)
    for axis in range(3):
        V = _raise_bra_degree(V, n, scale, p, A[axis], C[axis], P[axis])
        if axis < 2:
            V = V[:-n, ...]
    return V[0, ...]


def _horizontal_step(diff, column, _):
    shifted = jnp.roll(column, shift=-1)
    new_column = diff * column + shifted
    return new_column, new_column


def _shift_degree(tensor: jax.Array, axis: int, new_size: int, center_from: jax.Array, center_to: jax.Array) -> jax.Array:
    """Adds a trailing axis of size new_size to `tensor`, transferring
    angular momentum from `axis` to it via the (exact) product-Gaussian
    translation recursion. Used both for same-electron transfers
    (nuclear attraction, and g1->g2 / g3->g4 in repulsion) and can be
    reused unmodified for the cross-electron transfer below because the
    recursion only differs in its coefficients, supplied by the caller."""
    if new_size <= 1:
        return tensor[..., None]
    moved = jnp.moveaxis(tensor, axis, -1)
    step = functools.partial(_horizontal_step, center_from - center_to)
    _, rest = jax.lax.scan(step, moved, jnp.arange(1, new_size), unroll=True)
    combined = jnp.concatenate((moved[..., None], jnp.moveaxis(rest, 0, -1)), axis=-1)
    return jnp.moveaxis(combined, -2, axis)


@jax.jit
def nuclear_attraction(g1: GaussianShell3D, g2: GaussianShell3D, nucleus: jax.Array) -> jax.Array:
    """<g1| 1/|r - nucleus| |g2>, shape (L1+1,)*3 + (L2+1,)*3."""
    a, A = jnp.asarray(g1.exponent), jnp.asarray(g1.center)
    b, B = jnp.asarray(g2.exponent), jnp.asarray(g2.center)
    p = a + b

    padded_g1 = GaussianShell3D(degree=g1.degree + g2.degree, exponent=a, center=A)
    I = (2.0 * jnp.pi / p) * _hermite_coulomb_tensor(padded_g1, g2, p, nucleus)

    for axis in range(3):
        I = _shift_degree(I, axis, g2.degree + 1, A[axis], B[axis])
        I = I[(slice(0, g1.degree + 1),) * (axis + 1) + (Ellipsis,)]
    return I


def _electron_transfer_step(p, q, alpha, carry, j):
    I_prev, I_prev2 = carry
    I_prev_up = jnp.roll(I_prev, shift=-1, axis=-1)
    I_prev_down = jnp.pad(I_prev[..., :-1], ((0, 0),) * (I_prev.ndim - 1) + ((1, 0),))
    idx = jnp.arange(I_prev.shape[-1]).reshape((1,) * (I_prev.ndim - 1) + (-1,))

    I_next = (
        alpha * I_prev
        + (idx / (2.0 * q)) * I_prev_down
        + ((j - 1) / (2.0 * q)) * I_prev2
        - (p / q) * I_prev_up
    )
    return (I_next, I_prev), I_next


def _transfer_to_second_electron(tensor: jax.Array, axis: int, new_size: int, exponents, centers) -> jax.Array:
    if new_size <= 1:
        return tensor[..., None]
    a, b, c, d = exponents
    A, B, C, D = centers
    p, q = a + b, c + d
    alpha = -(1.0 / q) * (b * (A - B) + d * (C - D))

    moved = jnp.moveaxis(tensor, axis, -1)
    step = functools.partial(_electron_transfer_step, p, q, alpha)
    init = (moved, jnp.zeros_like(moved))
    _, rest = jax.lax.scan(step, init, jnp.arange(1, new_size), unroll=True)
    combined = jnp.concatenate((moved[..., None], jnp.moveaxis(rest, 0, -1)), axis=-1)
    return jnp.moveaxis(combined, -2, axis)


@jax.jit
def electron_repulsion(g1: GaussianShell3D, g2: GaussianShell3D, g3: GaussianShell3D, g4: GaussianShell3D) -> jax.Array:
    """<g1 g2| 1/r12 |g3 g4>, shape (L1+1,)^3 + (L2+1,)^3 + (L3+1,)^3 + (L4+1,)^3."""
    a, A = jnp.asarray(g1.exponent), jnp.asarray(g1.center)
    b, B = jnp.asarray(g2.exponent), jnp.asarray(g2.center)
    c, C = jnp.asarray(g3.exponent), jnp.asarray(g3.center)
    d, D = jnp.asarray(g4.exponent), jnp.asarray(g4.center)

    padded_g3 = GaussianShell3D(degree=g3.degree + g4.degree, exponent=c, center=C)
    padded_g1 = GaussianShell3D(degree=g1.degree + g2.degree + padded_g3.degree, exponent=a, center=A)

    p, q = a + b, c + d
    scale = (p * q) / (p + q)
    Q = (c * C + d * D) / q
    K34 = jnp.exp(-((c * d) / q) * jnp.sum(jnp.square(C - D)))
    prefactor = 2.0 * jnp.pi ** 2.5 / (p * q * jnp.sqrt(p + q)) * K34

    I = prefactor * _hermite_coulomb_tensor(padded_g1, g2, scale, Q)

    exponents = jnp.array([a, b, c, d])
    for axis in range(3):
        I = _transfer_to_second_electron(
            I, axis, padded_g3.degree + 1, exponents,
            jnp.array([A[axis], B[axis], C[axis], D[axis]]),
        )
        I = I[(slice(0, g1.degree + g2.degree + 1),) * (axis + 1) + (Ellipsis,)]

    for axis in range(3):
        I = _shift_degree(I, axis, g2.degree + 1, A[axis], B[axis])
        I = I[(slice(0, g1.degree + 1),) * (axis + 1) + (Ellipsis,)]

    for axis in range(3):
        I = _shift_degree(I, axis + 3, g4.degree + 1, C[axis], D[axis])
        I = I[(slice(0, g1.degree + 1),) * 3 + (slice(0, g3.degree + 1),) * (axis + 1) + (Ellipsis,)]

    # axes are currently (g1, g3, g2, g4); reorder to (g1, g2, g3, g4)
    return jnp.moveaxis(I, [3, 4, 5], [6, 7, 8])
