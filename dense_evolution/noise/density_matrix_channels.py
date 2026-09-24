"""Noise applied directly to a density matrix, instead of a statevector --
what density-matrix ZNE (`dense_evolution.mitigation.zne_density_matrix`)
needs its noise ensemble built from."""
import jax.numpy as jnp

__all__ = ["global_depolarizing_channel", "amplitude_damping_channel", "phaseflip_channel_exact"]


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


def phaseflip_channel_exact(rho: jnp.ndarray, p: float) -> jnp.ndarray:
    """Exact (zero-variance) multi-qubit phaseflip channel: K0=sqrt(1-p)*I,
    K1=sqrt(p)*Z, applied independently per qubit across the whole
    register, matching `NoiseModel`'s `'phaseflip'` statevector model
    (`noise/kraus/phaseflip.py`) in the limit of infinitely many Monte
    Carlo trials -- this function computes that limit directly instead of
    averaging finite trials, for use as a zero-sampling-variance
    "classical node" in Classically Augmented ZNE (Scheiber et al., "CA-ZNE:
    high-noise Richardson nodes replaced by classically simulated
    estimates", arXiv:2607.25746).

    Since Z is diagonal in the computational basis, conjugating rho by Z_q
    (Pauli-Z on qubit q, identity elsewhere) only flips the sign of
    off-diagonal entries whose row and column disagree on bit q --
    (Z_q rho Z_q)_ij = rho_ij * s_i * s_j, s_k = -1 if bit q of k else +1
    -- so each qubit's channel (1-p)*rho + p*(Z_q rho Z_q) is one
    elementwise multiply, no explicit operator ever built.
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    dim = rho.shape[0]
    n_qubits = dim.bit_length() - 1
    idx = jnp.arange(dim)
    for q in range(n_qubits):
        sign = 1.0 - 2.0 * ((idx >> q) & 1)
        flip_sign = jnp.outer(sign, sign)
        rho = (1.0 - p) * rho + p * (flip_sign * rho)
    return rho
