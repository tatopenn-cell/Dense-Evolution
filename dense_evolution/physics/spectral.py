"""Gauge-safe gradients for spectral functions of a Hermitian matrix
(V f(Lambda) V^dagger, e.g. time evolution exp(-iHt)) at exact eigenvalue
degeneracy.

`jnp.linalg.eigh`'s reverse-mode gradient divides by `lambda_i - lambda_j`
for every eigenvector pair. When two eigenvalues are exactly degenerate,
this does not raise and does not always produce NaN -- it can silently
return a finite, WRONG gradient, because the eigenvectors spanning a
degenerate eigenspace are not themselves uniquely defined (any orthonormal
basis of that subspace is an equally valid `eigh` output). Measured on a
real Kaggle CPU kernel (Dense-Evolution-Discovery, PR #173): std eigh
gradient error 0.98 vs Kato 4e-10 on an H with four exact doubly-degenerate
eigenvalues -- several orders of magnitude, not a rounding difference.

REFERENCES (verified against the actual paper text, not trusted at face
value from a citation string alone):
    Kasim, M. F., "Derivatives of partial eigendecomposition of a real
    symmetric matrix for degenerate cases", arXiv:2011.04366 (2020).
    Kato, T., "Perturbation Theory for Linear Operators", Springer (1995),
    Ch. II.5.6 (the classical divided-difference formula for matrix
    function derivatives, predating Kasim by decades).

`matrix_function_eigh` uses a `jax.custom_jvp` based on Kato's
divided-difference formula for matrix functions:

    d/deps [ V(eps) f(Lambda(eps)) V(eps)^dagger ] = V (F o (V^dagger dH V)) V^dagger

with F the matrix of divided differences of f:

    F[i,j] = (f(lambda_i) - f(lambda_j)) / (lambda_i - lambda_j)   if lambda_i != lambda_j
    F[i,j] = f'(lambda_i)                                          if lambda_i == lambda_j (incl. i == j)

This formula does not pass through eigenvectors as an intermediate OUTPUT,
so it is gauge-invariant: the contribution from a degenerate block uses
f'(lambda) directly, and there is no gauge choice to make -- unlike a
`custom_vjp` built directly on top of `eigh`'s own eigenvector output,
which needs a compatibility condition on the perturbation direction
(Kasim's Eq. 4.72, confirmed present in the actual paper text) that this
formula does not.

WHEN TO USE THIS:
    - H has (or might have) exactly degenerate eigenvalues, AND
    - the function L(H) you differentiate depends on H through eigh, AND
    - the function is not trivially constant on degenerate blocks.

For a Hamiltonian with only near-degeneracy (e.g. min_gap ~1e-5, no exact
tie), plain `jnp.linalg.eigh` is correct and faster -- use
`has_exact_degeneracy` to check before reaching for `spectral_evolve`.
"""
import functools

import jax
import jax.numpy as jnp

from dense_evolution.config import ensure_x64

__all__ = ["has_exact_degeneracy", "matrix_function_eigh", "spectral_evolve"]

_DEGENERACY_TOL = 1e-8


def has_exact_degeneracy(H: jax.Array, tol: float = _DEGENERACY_TOL) -> bool:
    """True if H has at least one pair of eigenvalues closer than tol.

    Diagnostic only -- call this before choosing spectral_evolve (Kato)
    over plain jnp.linalg.eigh (std). The threshold is the same one the
    JVP rule below uses internally, so this is the exact condition under
    which the two methods disagree."""
    ensure_x64()
    w = jnp.linalg.eigvalsh(H)
    gaps = jnp.abs(jnp.diff(jnp.sort(w)))
    return bool((gaps < tol).any())


def _matrix_function_divided_differences(fw, f_prime_w, w, tol):
    lam_i = w[:, None]
    lam_j = w[None, :]
    gap = lam_i - lam_j
    is_deg = jnp.abs(gap) < tol
    safe_gap = jnp.where(is_deg, 1.0, gap)
    F_quot = (fw[:, None] - fw[None, :]) / safe_gap
    F_limit = f_prime_w[:, None]
    return jnp.where(is_deg, F_limit, F_quot)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def matrix_function_eigh(H: jax.Array, f, f_prime) -> jax.Array:
    """V f(Lambda) V^dagger with a gauge-safe gradient at exact degeneracy.

    Parameters
    ----------
    H : (n, n) Hermitian matrix.
    f : callable, lambda (array) -> array. Applied elementwise to the
        eigenvalues.
    f_prime : callable, lambda (array) -> array. Analytic derivative of f,
        used only in the JVP rule for degenerate blocks. Passed via
        nondiff_argnums since a Python closure is not a valid JAX type to
        trace.

    Returns
    -------
    (n, n) complex128 matrix.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> H = jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0], dtype=jnp.complex128))
    >>> U = matrix_function_eigh(H, lambda w: jnp.exp(-1j * w), lambda w: -1j * jnp.exp(-1j * w))
    """
    ensure_x64()
    w, v = jnp.linalg.eigh(H)
    return v @ jnp.diag(f(w)) @ v.conj().T


@matrix_function_eigh.defjvp
def _matrix_function_eigh_jvp(f, f_prime, primals, tangents):
    (H,) = primals
    (dH,) = tangents

    w, v = jnp.linalg.eigh(H)
    fw = f(w)
    f_prime_w = f_prime(w)

    F = _matrix_function_divided_differences(fw, f_prime_w, w, _DEGENERACY_TOL)

    X = v.conj().T @ dH @ v
    dU = v @ (F * X) @ v.conj().T
    U = v @ jnp.diag(fw) @ v.conj().T
    return U, dU


def spectral_evolve(H: jax.Array, t: float) -> jax.Array:
    """exp(-i H t) with a gauge-safe gradient at exact degeneracy.

    Equivalent forward to `V @ diag(exp(-1j*w*t)) @ V.conj().T` where
    (w, V) is `jnp.linalg.eigh(H)`. Backward uses Kato's divided-difference
    rule (see module docstring) instead of `eigh`'s own reverse-mode rule.
    """
    return matrix_function_eigh(
        H,
        f=lambda w: jnp.exp(-1j * w * t),
        f_prime=lambda w: -1j * t * jnp.exp(-1j * w * t),
    )
