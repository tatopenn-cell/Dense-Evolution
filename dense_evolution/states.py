"""
Common state-preparation circuits, returned as gate-tuple lists ready to
feed straight into `run_circuit` (or to concatenate with more gates first).
The GHZ-state snippet in particular -- `[('h', 0), ('cx', 0, 1), ('cx', 1,
2), ...]` -- shows up hand-written at the top of practically every
experiment and test script built on this package; `ghz_state` is that
snippet, written once.
"""
from .topology import entangling_layer

__all__ = ['ghz_state']


def ghz_state(n_qubits):
    """
    Build the GHZ-state preparation circuit:
    (|00...0> + |11...1>) / sqrt(2).

    Implementation: H on qubit 0, then a linear CX chain (qubit 0 -> 1,
    1 -> 2, ..., n-2 -> n-1) propagating the superposition outward --
    reuses `entangling_layer(n_qubits, pattern='linear')` for the chain.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, must be >= 2 (a single qubit has no partner to
        entangle with, so an n=1 "GHZ state" is undefined here).

    Returns
    -------
    list[tuple]
        e.g. ghz_state(3) == [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]

    Examples
    --------
    >>> import dense_evolution as de
    >>> sim = de.DenseSVSimulator(3)
    >>> sim.run_circuit(de.ghz_state(3))
    >>> sim.get_probabilities()[[0, 7]]  # |000> and |111>, each 0.5
    """
    if n_qubits < 2:
        raise ValueError(f"ghz_state needs at least 2 qubits, got {n_qubits}")
    return [('h', 0)] + entangling_layer(n_qubits, pattern='linear', gate='cx')
