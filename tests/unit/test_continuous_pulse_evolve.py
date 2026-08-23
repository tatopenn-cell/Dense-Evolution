import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from dense_evolution import continuous_pulse_evolve

X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)


def test_constant_coefficient_matches_single_expm_step():
    # A constant Hamiltonian sliced into many small steps must reproduce
    # exp(-i*H*t_total) applied once, up to Trotter/discretization error
    # from finite dt -- here H is already piecewise-constant per slice
    # (not approximated), so the match should be to numerical precision.
    n_slices = 200
    dt = 0.01
    coeffs = jnp.full((n_slices,), 0.7)
    psi0 = jnp.array([1.0, 0.0], dtype=jnp.complex128)

    final_psi, trajectory = continuous_pulse_evolve(psi0, lambda c: c * X, coeffs, dt)

    from jax.scipy.linalg import expm
    total_t = n_slices * dt
    expected = expm(-1j * 0.7 * X * total_t) @ psi0

    assert trajectory is None
    assert np.allclose(np.array(final_psi), np.array(expected), atol=1e-8)


def test_time_dependent_coefficient_matches_manual_python_loop():
    # Cross-check the scan against a plain, unrolled Python loop doing the
    # exact same per-slice exp(-i*H*dt) step -- the correctness baseline
    # independent of jax.lax.scan machinery.
    n_slices = 50
    dt = 0.02
    t = jnp.arange(n_slices) * dt
    coeffs = jnp.sin(t) ** 2   # an arbitrary smooth time-dependent profile
    psi0 = jnp.array([0.0, 1.0], dtype=jnp.complex128)

    def hamiltonian_fn(c):
        return 0.5 * c * X + 0.3 * (1 - c) * Y

    final_psi, _ = continuous_pulse_evolve(psi0, hamiltonian_fn, coeffs, dt)

    from jax.scipy.linalg import expm
    psi_manual = psi0
    for c in coeffs:
        H_t = 0.5 * c * X + 0.3 * (1 - c) * Y
        psi_manual = expm(-1j * H_t * dt) @ psi_manual

    assert np.allclose(np.array(final_psi), np.array(psi_manual), atol=1e-8)


def test_observable_fn_returns_trajectory_of_expected_length():
    n_slices = 30
    dt = 0.05
    coeffs = jnp.linspace(0.0, 1.0, n_slices)
    psi0 = jnp.array([1.0, 0.0], dtype=jnp.complex128)

    final_psi, trajectory = continuous_pulse_evolve(
        psi0, lambda c: c * X, coeffs, dt,
        observable_fn=lambda psi: jnp.abs(psi[1]) ** 2,
    )

    assert trajectory.shape == (n_slices,)
    # Final trajectory entry must equal the |1> occupation of final_psi.
    assert np.allclose(float(trajectory[-1]), float(jnp.abs(final_psi[1]) ** 2))


def test_preserves_norm():
    n_slices = 80
    dt = 0.03
    coeffs = jnp.cos(jnp.linspace(0.0, 3.0, n_slices))
    psi0 = jnp.array([0.6, 0.8], dtype=jnp.complex128)

    final_psi, _ = continuous_pulse_evolve(psi0, lambda c: c * X, coeffs, dt)

    assert abs(float(jnp.sum(jnp.abs(final_psi) ** 2)) - 1.0) < 1e-9
