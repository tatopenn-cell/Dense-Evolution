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

__all__ = ['partial_trace', 'von_neumann_entropy', 'mutual_information', 'central_charge']


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
    alone. Verified in tests/unit/test_entropy.py against the exact textbook
    value for a Bell pair (I = 2*ln(2), maximal) and a GHZ state.
    """
    s_a = von_neumann_entropy(partial_trace(state, n_qubits, qubits_a))
    s_b = von_neumann_entropy(partial_trace(state, n_qubits, qubits_b))
    s_ab = von_neumann_entropy(partial_trace(state, n_qubits, list(qubits_a) + list(qubits_b)))
    return s_a + s_b - s_ab


def central_charge(Ls, S, n_qubits):
    """Fit an open-chain entanglement entropy curve S(L) to the Calabrese-
    Cardy CFT prediction S(L) = (c/6)*ln[(2N/pi)*sin(pi*L/N)] + const
    (Calabrese & Cardy, J. Stat. Mech. 2004, P06002, eq. 4/19 combined via
    the standard open-chain doubling trick) and return (c, r_squared).

    Backend-agnostic: `S` can come from any source (exact diagonalization
    via `partial_trace`/`von_neumann_entropy` on this package's own
    `DenseSVSimulator`, `MPSSimulator`, `Chunk`, or elsewhere) -- this
    doesn't compute the entropy itself, only fits an already-measured
    curve. Meant as a benchmark diagnostic: does a given backend/
    truncation scheme preserve genuine critical CFT scaling, and with
    what effective central charge?

    A high r_squared alone does NOT mean the extracted c is trustworthy --
    Dense-Evolution-Discovery Experiment 36 found fitting at a finite-size
    pseudo-critical point (a susceptibility peak, not the true CFT point)
    gives a deceptively clean fit (r_squared=0.999997) to a wrong answer
    (c off by 2x). Only trust this near a genuine, independently-verified
    critical point.

    Parameters
    ----------
    Ls : array-like of int
        Subsystem sizes, each counted from one physical boundary of an
        open chain of `n_qubits` sites (not a bulk interval -- see
        Discovery Experiment 36 for the periodic/bulk c/3 case instead).
    S : array-like of float
        Entanglement entropy at each L in `Ls`, same length.
    n_qubits : int
        Total open-chain length N.

    Returns
    -------
    c : float
        Extracted central charge (theory: 0.5 for Ising, 1.0 for a free
        boson/XX chain, ...).
    r_squared : float
        Fit quality, in [0, 1] for a sane fit (can go negative for a
        pathological fit worse than the mean).
    """
    Ls = np.asarray(Ls, dtype=float)
    S = np.asarray(S, dtype=float)
    x = np.log((2.0 * n_qubits / np.pi) * np.sin(np.pi * Ls / n_qubits))
    slope, intercept = np.polyfit(x, S, 1)
    c = 6.0 * slope
    pred = slope * x + intercept
    ss_res = float(np.sum((S - pred) ** 2))
    ss_tot = float(np.sum((S - S.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(c), r_squared
