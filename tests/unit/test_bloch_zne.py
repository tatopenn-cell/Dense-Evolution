import numpy as np
import jax.numpy as jnp
import pytest

from dense_evolution.mitigation import (
    bloch_zne, bloch_zne_jit, project_to_physical, richardson_extrapolate,
)

FACTORS = [1.0, 2.0, 3.0]
PAULI = (np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex))


def rho_of(r):
    return 0.5 * (np.eye(2) + sum(c * P for c, P in zip(r, PAULI)))


def test_inside_ball_is_plain_richardson():
    v = np.array([[0.1, 0.2, 0.5], [0.08, 0.15, 0.45], [0.07, 0.12, 0.41]])
    np.testing.assert_allclose(bloch_zne(v, FACTORS), richardson_extrapolate(v, np.array(FACTORS)), atol=1e-14)


def test_outside_ball_is_rescaled():
    r = np.asarray(bloch_zne([[0.0, 0.0, 0.96], [0.0, 0.0, 0.9], [0.0, 0.0, 0.87]], FACTORS))
    np.testing.assert_allclose(r, [0.0, 0.0, 1.0], atol=1e-14)


def test_matches_project_to_physical_and_jit():
    rng = np.random.default_rng(243)
    for _ in range(30):
        v = rng.normal(size=(3, 3)) * 0.6
        raw = np.asarray(richardson_extrapolate(v, np.array(FACTORS)))
        r = np.asarray(bloch_zne(v, FACTORS))
        assert np.linalg.norm(r) <= 1 + 1e-12
        np.testing.assert_allclose(rho_of(r), project_to_physical(rho_of(raw)), atol=1e-12)
        np.testing.assert_allclose(bloch_zne_jit(jnp.asarray(v), jnp.asarray(FACTORS)), r, atol=1e-14)


def test_never_further_from_ideal():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        v = np.array([n * (1 - 0.05 * f) for f in FACTORS]) + rng.normal(scale=0.1, size=(3, 3))
        raw = np.asarray(richardson_extrapolate(v, np.array(FACTORS)))
        assert np.linalg.norm(np.asarray(bloch_zne(v, FACTORS)) - n) <= np.linalg.norm(raw - n) + 1e-12


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        bloch_zne([[0.0, 0.0, 1.0]], FACTORS)
