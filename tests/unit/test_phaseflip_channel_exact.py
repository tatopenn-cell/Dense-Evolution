import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dense_evolution as de
from dense_evolution import phaseflip_channel_exact


def test_p_zero_is_identity_map():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = jnp.asarray(a @ a.conj().T, dtype=jnp.complex128)
    rho = rho / jnp.trace(rho)
    out = phaseflip_channel_exact(rho, 0.0)
    np.testing.assert_allclose(np.array(out), np.array(rho), atol=1e-9)


def test_preserves_trace_and_hermiticity_single_qubit():
    rho = jnp.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]], dtype=jnp.complex128)
    for p in (0.0, 0.25, 0.5, 0.9, 1.0):
        out = phaseflip_channel_exact(rho, p)
        assert abs(complex(jnp.trace(out)) - 1.0) < 1e-9
        np.testing.assert_allclose(np.array(out), np.array(out).conj().T, atol=1e-9)


def test_single_qubit_matches_known_kraus_result():
    # K0=sqrt(1-p)*I, K1=sqrt(p)*Z -- diagonal entries untouched, off-diagonal
    # scaled by (1-p) - p = 1-2p.
    rho = jnp.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]], dtype=jnp.complex128)
    p = 0.3
    out = np.array(phaseflip_channel_exact(rho, p))
    assert out[0, 0] == pytest.approx(0.6, abs=1e-9)
    assert out[1, 1] == pytest.approx(0.4, abs=1e-9)
    assert out[0, 1] == pytest.approx((0.2 - 0.1j) * (1.0 - 2.0 * p), abs=1e-9)


def test_matches_monte_carlo_average_of_noise_model_within_statistical_tolerance():
    # Cross-check against the statevector Monte Carlo model this function
    # is the exact (infinite-trial) limit of -- see this function's own
    # docstring. A moderate trial count with a generous tolerance, not the
    # 200000-trial/2e-4 precision used to validate this during development.
    sim = de.DenseSVSimulator(3)
    sim.run_circuit([("h", 0), ("cx", 0, 1), ("cx", 1, 2)])
    sv = jnp.asarray(sim.get_statevector(), dtype=jnp.complex128)
    rho_ideal = jnp.outer(sv, jnp.conj(sv))

    p = 0.05
    rho_exact = phaseflip_channel_exact(rho_ideal, p)

    keys = jax.random.split(jax.random.PRNGKey(0), 20000)

    def one_trial(key):
        return de.NoiseModel.apply_to_sv(sv, 3, model="phaseflip", p=p, jax_key=key)

    sv_batch = jax.vmap(one_trial)(keys)
    rho_mc = jnp.einsum("ti,tj->ij", sv_batch, jnp.conj(sv_batch)) / 20000

    assert float(jnp.max(jnp.abs(rho_exact - rho_mc))) < 0.01
