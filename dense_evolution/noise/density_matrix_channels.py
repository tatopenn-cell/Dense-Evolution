"""Noise applied directly to a density matrix, instead of a statevector --
what density-matrix ZNE (`dense_evolution.mitigation.zne_density_matrix`)
needs its noise ensemble built from."""
import jax.numpy as jnp

__all__ = ["global_depolarizing_channel", "amplitude_damping_channel"]


def global_depolarizing_channel(rho: jnp.ndarray, p: float) -> jnp.ndarray:
    """Global n-qubit depolarizing channel, D_p(rho) = (1-p)*rho + (p/dim)*I.

    Distinct from `NoiseModel`'s `'depolarizing'` model, which applies an
    independent PER-QUBIT local Kraus channel -- a different physical map
    from this GLOBAL channel, which mixes the whole `dim`-dimensional state
    toward the fully mixed state as one unit. Use this one when modeling
    e.g. state-prep/measurement (SPAM) error reported as a single joint
    depolarizing parameter over the whole register, not per-qubit gate
    noise (promoted from a real reproduction of arXiv:2608.16716's own
    SPAM model, Dense-Evolution-Discovery Experiment 33).
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    dim = rho.shape[0]
    identity = jnp.eye(dim, dtype=jnp.complex128)
    return (1.0 - p) * rho + (p / dim) * identity


def amplitude_damping_channel(rho: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Single-qubit amplitude-damping channel: E0 @ rho @ E0.conj().T +
    E1 @ rho @ E1.conj().T, with E0=diag(1, sqrt(1-gamma)) and
    E1=[[0,sqrt(gamma)],[0,0]] -- population only ever moves |1>->|0>,
    never the reverse.

    Distinct from `global_depolarizing_channel` (symmetric, mixes toward
    the fully-mixed state regardless of which state is |1> or |0>) -- this
    one is asymmetric by construction, the real signature of energy-relaxation
    (T1) processes and of quasiparticle poisoning (promoted from a real
    reproduction of arXiv:2104.05219's measured cosmic-ray-induced error
    bursts, Dense-Evolution-Discovery Experiment 34, where this asymmetry is
    exactly the mechanism's own reported signature: decay errors only, no
    excess excitation errors).

    Single-qubit only (rho must be 2x2) -- unlike `global_depolarizing_channel`,
    this is not dimension-generic, since amplitude damping is inherently a
    per-qubit process, not a joint-register one.
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    e0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]], dtype=jnp.complex128)
    e1 = jnp.array([[0.0, jnp.sqrt(gamma)], [0.0, 0.0]], dtype=jnp.complex128)
    return e0 @ rho @ e0.conj().T + e1 @ rho @ e1.conj().T
