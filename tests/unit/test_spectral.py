"""Regression tests for dense_evolution.physics.spectral.

The critical test is `test_std_eigh_fails_at_degeneracy`: it asserts that
plain jnp.linalg.eigh's gradient is WRONG at exact degeneracy. If it ever
starts passing correctly (e.g. JAX fixes this upstream), that test should
fail loudly -- the correct signal that spectral.py's workaround is no
longer needed, not a silent success.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from dense_evolution.physics.spectral import has_exact_degeneracy, matrix_function_eigh, spectral_evolve


def _make_degenerate_H(seed=42, n_levels=4, degeneracy=2):
    """H = U diag(d) U^dagger with d having n_levels distinct values, each
    repeated degeneracy times. U unitary via QR of a complex random
    matrix. Gap between levels is O(1); gap within a level is exactly 0."""
    n = n_levels * degeneracy
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(M)
    phases = np.diag(R) / np.abs(np.diag(R))
    U = jnp.asarray((Q * phases[None, :]).astype(jnp.complex128))
    levels = jnp.linspace(1.0, 4.0, n_levels, dtype=jnp.complex128)
    d = jnp.repeat(levels, degeneracy)
    H = U @ jnp.diag(d) @ U.conj().T
    return 0.5 * (H + H.conj().T)


def _finite_difference_grad(L_fn, H, P, h=1e-5):
    return float(L_fn(H + h * P) - L_fn(H - h * P)) / (2 * h)


def test_has_exact_degeneracy_true():
    H = _make_degenerate_H()
    assert has_exact_degeneracy(H)


def test_has_exact_degeneracy_false_on_generic():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    H = 0.5 * (M + M.conj().T)
    H = jnp.asarray(H.astype(jnp.complex128))
    assert not has_exact_degeneracy(H)


def test_forward_matches_std_eigh():
    """At degeneracy, forward pass must agree with std eigh bit-for-bit."""
    H = _make_degenerate_H()
    t = 0.7
    w, v = jnp.linalg.eigh(H)
    U_std = v @ jnp.diag(jnp.exp(-1j * w * t)) @ v.conj().T
    U_kato = spectral_evolve(H, t)
    assert jnp.allclose(U_std, U_kato, atol=1e-13)


def test_kato_gradient_matches_finite_differences():
    """Kato's JVP must match central finite differences at degeneracy."""
    H = _make_degenerate_H()
    rng = np.random.default_rng(7)
    P = rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)
    P = 0.5 * (P + P.conj().T)
    P = jnp.asarray(P.astype(jnp.complex128))
    t = 1.0

    A = rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)
    A = 0.5 * (A + A.conj().T)
    A = jnp.asarray(A.astype(jnp.complex128))

    def L(H_):
        return jnp.real(jnp.sum(A * spectral_evolve(H_, t)))

    fd = _finite_difference_grad(L, H, P)
    grad_H = np.asarray(jax.grad(L)(H))
    grad_dir = float(np.real(np.sum(grad_H * np.asarray(P))))
    assert abs(grad_dir - fd) < 1e-6, (grad_dir, fd)


def test_std_eigh_fails_at_degeneracy():
    """Guard test: std eigh's gradient is wrong at exact degeneracy.

    If this ever starts passing (i.e. std eigh becomes correct here), the
    spectral_evolve workaround can be retired."""
    H = _make_degenerate_H()
    rng = np.random.default_rng(11)
    P = rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)
    P = 0.5 * (P + P.conj().T)
    P = jnp.asarray(P.astype(jnp.complex128))
    t = 1.0

    A = rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)
    A = 0.5 * (A + A.conj().T)
    A = jnp.asarray(A.astype(jnp.complex128))

    def L_std(H_):
        w, v = jnp.linalg.eigh(H_)
        U = v @ jnp.diag(jnp.exp(-1j * w * t)) @ v.conj().T
        return jnp.real(jnp.sum(A * U))

    fd = _finite_difference_grad(L_std, H, P)
    grad_H = np.asarray(jax.grad(L_std)(H))
    grad_dir = float(np.real(np.sum(grad_H * np.asarray(P))))
    assert abs(grad_dir - fd) > 1e-3, (
        f"std eigh is now correct at degeneracy (diff={abs(grad_dir - fd):.2e}); "
        "the spectral_evolve workaround can be retired."
    )


def test_matrix_function_eigh_matches_std_on_non_degenerate():
    """On a non-degenerate H, Kato and std must agree to machine precision."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    H = jnp.asarray((0.5 * (M + M.conj().T)).astype(jnp.complex128))
    t = 1.3

    U_kato = spectral_evolve(H, t)
    w, v = jnp.linalg.eigh(H)
    U_std = v @ jnp.diag(jnp.exp(-1j * w * t)) @ v.conj().T
    assert jnp.allclose(U_kato, U_std, atol=1e-13)

    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    A = jnp.asarray((0.5 * (A + A.conj().T)).astype(jnp.complex128))

    def L(H_):
        return jnp.real(jnp.sum(A * spectral_evolve(H_, t)))

    grad = jax.grad(L)(H)
    assert jnp.isfinite(grad).all()


def test_spectral_evolve_is_unitary():
    """Sanity: exp(-iHt) must be unitary regardless of degeneracy."""
    H = _make_degenerate_H()
    U = spectral_evolve(H, 0.9)
    err = jnp.linalg.norm(U.conj().T @ U - jnp.eye(H.shape[0], dtype=jnp.complex128))
    assert err < 1e-12
