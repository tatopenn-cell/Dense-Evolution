"""
Multi-qubit partial trace, von Neumann entropy, and mutual information.

Nothing like this existed anywhere in the package before: the only prior
partial trace (dashboard_core/state_visuals.py's private
`_reduced_density_matrix`) is single-qubit-only and uses the *opposite*,
little-endian convention (qubit 0 = least significant bit). Everything
here uses this package's own convention instead, matching
dense_evolution.observables/pauli_hamiltonian_to_matrix: qubit 0 is the
*most* significant bit of the basis-state index. Do not mix the two --
reusing dashboard_core's helper here would silently transpose which
qubits get traced out.

Originated in research/wormhole_syk.py (a traversable-wormhole-inspired
quantum teleportation reproduction) -- promoted here because these are
generic quantum-information utilities, not specific to that experiment.
Any state can have a subsystem's reduced density matrix, entropy, or the
mutual information between two subsystems computed with these three
functions; the wormhole work needed all three because the physically
meaningful readout there (a message injected into one system showing up
correlated with a reference qubit) is *not* visible in any single-qubit
expectation value -- see mutual_information's docstring.
"""

import numpy as np

__all__ = ['partial_trace', 'von_neumann_entropy', 'mutual_information']


def partial_trace(state, n_qubits, keep_qubits):
    """Reduced density matrix on `keep_qubits`, tracing out the rest.

    Parameters
    ----------
    state : np.ndarray
        A pure statevector of length 2**n_qubits.
    n_qubits : int
    keep_qubits : list[int]
        Qubit indices (this package's MSB-first convention) to keep.

    Returns
    -------
    np.ndarray
        Density matrix of shape (2**len(keep_qubits), 2**len(keep_qubits)).
    """
    keep_qubits = sorted(keep_qubits)
    trace_qubits = [q for q in range(n_qubits) if q not in keep_qubits]
    psi = np.transpose(np.asarray(state).reshape([2] * n_qubits), keep_qubits + trace_qubits)
    keep_dim, trace_dim = 2 ** len(keep_qubits), 2 ** len(trace_qubits)
    psi = psi.reshape(keep_dim, trace_dim)
    return psi @ psi.conj().T


def von_neumann_entropy(rho):
    """S(rho) = -Tr(rho log rho), computed from rho's eigenvalues. Nearly-
    zero eigenvalues (which a numerically pure/near-pure state produces,
    and which are mathematically forbidden from being exactly negative
    for a real density matrix but can land at a tiny negative float) are
    clipped before the log rather than raising or propagating a NaN."""
    eigs = np.clip(np.linalg.eigvalsh(rho).real, 1e-14, None)
    return float(-np.sum(eigs * np.log(eigs)))


def mutual_information(state, n_qubits, qubits_a, qubits_b):
    """I(A:B) = S(A) + S(B) - S(A union B), the standard quantum mutual
    information between two disjoint subsystems of a pure global state.

    Why this and not a single-qubit expectation value: a qubit entangled
    in a Bell pair (or more generally, maximally mixed on its own) has a
    marginal <Z> of exactly 0 regardless of what operation was applied to
    its partner -- this is the no-signaling theorem, not a measurement
    limitation, and no amount of clever circuit design around a
    single-qubit readout can get around it. Mutual information *can*
    reveal correlations a marginal expectation value structurally cannot,
    because it depends on the *joint* state of A and B, not either one
    alone. Verified in tests/test_entropy.py against the exact textbook
    value for a Bell pair (I = 2*ln(2), maximal) and a GHZ state.
    """
    s_a = von_neumann_entropy(partial_trace(state, n_qubits, qubits_a))
    s_b = von_neumann_entropy(partial_trace(state, n_qubits, qubits_b))
    s_ab = von_neumann_entropy(partial_trace(state, n_qubits, list(qubits_a) + list(qubits_b)))
    return s_a + s_b - s_ab
