"""
Measurement-related conveniences: turning a statevector into finite-shot
counts the way real hardware and every other SDK report results (Qiskit's
`get_counts()`, PennyLane's `qml.counts()`), and comparing two pure states
directly without building a density matrix first.

Sampling a statevector into shot counts is another pattern found
hand-duplicated across this package's own experiment scripts -- always
the same `rng.choice(dim, p=probs)` + `np.bincount` (or `np.unique`)
couple of lines, rewritten fresh each time.
"""
import numpy as np

__all__ = ['sample_counts', 'statevector_fidelity']


def sample_counts(statevector, n_shots, rng=None):
    """
    Simulate n_shots projective measurements of every qubit in the
    computational basis, returning a Qiskit-style counts dict.

    Parameters
    ----------
    statevector : array-like, shape (2**n_qubits,)
    n_shots : int
        Number of measurement shots to sample, must be >= 1.
    rng : numpy.random.Generator, optional
        A fresh `numpy.random.default_rng()` is used if not given --
        pass one explicitly for reproducible sampling.

    Returns
    -------
    dict[str, int]
        Bitstring keys read left-to-right as qubit 0 upward (matching
        DenseSVSimulator's own qubit-0-is-MSB layout and
        pauli_expectation's string convention), each mapped to how many
        of the n_shots samples landed on it. Only bitstrings with at
        least one hit appear (unobserved outcomes are simply absent, as
        in Qiskit's get_counts()).

    Examples
    --------
    >>> import dense_evolution as de
    >>> sim = de.DenseSVSimulator(2)
    >>> sim.run_circuit([('h', 0), ('cx', 0, 1)])
    >>> de.sample_counts(sim.get_statevector(), 1000)
    {'00': 494, '11': 506}
    """
    statevector = np.asarray(statevector)
    dim = statevector.shape[0]
    n_qubits = dim.bit_length() - 1
    if 1 << n_qubits != dim:
        raise ValueError(f"statevector length {dim} is not a power of 2")
    if n_shots < 1:
        raise ValueError(f"n_shots must be >= 1, got {n_shots}")

    probs = np.abs(statevector) ** 2
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("statevector has zero or non-finite total probability")
    probs = probs / total

    if rng is None:
        rng = np.random.default_rng()
    outcomes = rng.choice(dim, size=n_shots, p=probs)
    values, freqs = np.unique(outcomes, return_counts=True)

    return {format(int(v), f'0{n_qubits}b'): int(f) for v, f in zip(values, freqs)}


def statevector_fidelity(statevector_a, statevector_b):
    """
    Fidelity |<a|b>|^2 between two pure statevectors -- the cheap, direct
    pure-state counterpart to `uhlmann_fidelity` (which compares two full
    density matrices, needed for the mixed/noisy states that
    `zne_density_matrix` and friends work with). Use this one whenever
    both states are pure; it never builds an O(dim^2) density matrix.

    Parameters
    ----------
    statevector_a, statevector_b : array-like, shape (2**n_qubits,)
        Must have the same shape.

    Returns
    -------
    float
        In [0, 1] up to floating-point error. 1.0 for identical states
        (up to global phase), 0.0 for orthogonal states.
    """
    a = np.asarray(statevector_a)
    b = np.asarray(statevector_b)
    if a.shape != b.shape:
        raise ValueError(f"statevector shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(np.vdot(a, b)) ** 2)
