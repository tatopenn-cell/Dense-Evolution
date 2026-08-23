import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from dense_evolution import amplitude_damping_channel


def _rho1():
    return jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)   # |1><1|


def _rho0():
    return jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)   # |0><0|


def test_ground_state_is_untouched():
    # The real asymmetry the channel exists to model: |0><0| passes through
    # unchanged regardless of gamma -- no excess excitation.
    out = amplitude_damping_channel(_rho0(), 0.7)
    assert np.allclose(np.array(out), np.array(_rho0()))


def test_gamma_zero_is_identity_map():
    rho = jnp.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]], dtype=jnp.complex128)
    out = amplitude_damping_channel(rho, 0.0)
    assert np.allclose(np.array(out), np.array(rho))


def test_gamma_one_fully_decays_excited_state():
    out = amplitude_damping_channel(_rho1(), 1.0)
    assert np.allclose(np.array(out), np.array(_rho0()), atol=1e-9)


def test_preserves_trace():
    rho = jnp.array([[0.3, 0.1j], [-0.1j, 0.7]], dtype=jnp.complex128)
    for gamma in (0.0, 0.25, 0.5, 0.9, 1.0):
        out = amplitude_damping_channel(rho, gamma)
        assert abs(complex(jnp.trace(out)) - 1.0) < 1e-9


def test_population_only_moves_excited_to_ground():
    out = amplitude_damping_channel(_rho1(), 0.4)
    assert float(jnp.real(out[1, 1])) == pytest.approx(0.6, abs=1e-9)
    assert float(jnp.real(out[0, 0])) == pytest.approx(0.4, abs=1e-9)
