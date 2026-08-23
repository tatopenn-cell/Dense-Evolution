import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from dense_evolution import continuous_dissipative_evolve, global_depolarizing_channel


def _pure_state_rho(dim, index=0):
    rho = jnp.zeros((dim, dim), dtype=jnp.complex128)
    return rho.at[index, index].set(1.0)


def test_zero_parameter_profile_leaves_rho_unchanged():
    rho0 = _pure_state_rho(4)
    params = jnp.zeros((20,))

    final_rho, trajectory = continuous_dissipative_evolve(rho0, global_depolarizing_channel, params)

    assert trajectory is None
    assert np.allclose(np.array(final_rho), np.array(rho0))


def test_matches_manual_sequential_composition():
    # Cross-check the scan against a plain Python loop applying the same
    # channel step by step -- the correctness baseline independent of
    # jax.lax.scan machinery.
    rho0 = _pure_state_rho(4)
    n_slices = 15
    params = jnp.linspace(0.01, 0.05, n_slices)

    final_rho, _ = continuous_dissipative_evolve(rho0, global_depolarizing_channel, params)

    rho_manual = rho0
    for p in params:
        rho_manual = global_depolarizing_channel(rho_manual, p)

    assert np.allclose(np.array(final_rho), np.array(rho_manual), atol=1e-10)


def test_rise_and_decay_profile_preserves_trace_and_pushes_toward_mixed_state():
    # A synthetic event profile shaped like the real cosmic-ray-induced
    # error burst (McEwen et al., arXiv:2104.05219): a fast rise to a peak
    # followed by an exponential decay back to baseline -- here in
    # dimensionless slice units, not literal microseconds/milliseconds.
    n_slices = 100
    slice_idx = jnp.arange(n_slices)
    peak_idx = 10
    rise = jnp.clip(slice_idx / peak_idx, 0.0, 1.0)
    decay = jnp.exp(-jnp.clip(slice_idx - peak_idx, 0, None) / 25.0)
    params = 0.3 * rise * decay   # peaks at p=0.3, decays back toward 0

    rho0 = _pure_state_rho(4)
    final_rho, trajectory = continuous_dissipative_evolve(
        rho0, global_depolarizing_channel, params,
        observable_fn=lambda rho: jnp.real(jnp.trace(rho)),
    )

    assert trajectory.shape == (n_slices,)
    assert np.allclose(np.array(trajectory), 1.0, atol=1e-9)   # trace preserved throughout
    assert abs(float(jnp.real(jnp.trace(final_rho))) - 1.0) < 1e-9
    # Some population must have been pushed off the diagonal peak toward
    # the maximally mixed state during the burst, then only partially
    # recover (channel is not perfectly invertible/undone by design).
    assert float(jnp.real(final_rho[0, 0])) < 1.0
