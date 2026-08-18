"""
Quantum Fourier Transform circuit builder -- the textbook cascade of H and
controlled-phase gates, plus the trailing qubit-order swap, returned as a
gate-tuple list ready for run_circuit. A standard building block (phase
estimation, Shor's algorithm, arithmetic circuits) that every general-
purpose SDK ships (Qiskit's `QFT`, Cirq's `cirq.QuantumFourierTransformGate`).
"""
import numpy as np

__all__ = ['qft']


def _invert_op(op):
    """H and SWAP are self-inverse; CP(theta)^-1 = CP(-theta)."""
    if op[0] == 'cp':
        gate, ctrl, tgt, angle = op
        return (gate, ctrl, tgt, -angle)
    return op


def qft(n_qubits, inverse=False, do_swaps=True):
    """
    Build the Quantum Fourier Transform circuit on n_qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, must be >= 1.
    inverse : bool
        Build the inverse QFT instead (same gate set, reversed order,
        negated phase angles).
    do_swaps : bool
        Include the trailing qubit-reversal swaps that put the output in
        the same qubit order as the input. The H/CP cascade alone
        produces the Fourier-transformed amplitudes in bit-reversed
        qubit order; set this to False only if the caller will account
        for that reversal itself (e.g. chaining straight into another
        subroutine that expects it).

    Returns
    -------
    list[tuple]

    Examples
    --------
    >>> import dense_evolution as de
    >>> sim = de.DenseSVSimulator(3)
    >>> sim.run_circuit(de.qft(3))
    >>> sim.get_probabilities()  # uniform: QFT|000> = equal superposition
    """
    if n_qubits < 1:
        raise ValueError(f"qft needs at least 1 qubit, got {n_qubits}")

    ops = []
    for j in range(n_qubits):
        ops.append(('h', j))
        for k in range(j + 1, n_qubits):
            angle = np.pi / (2 ** (k - j))
            ops.append(('cp', k, j, angle))
    if do_swaps:
        for i in range(n_qubits // 2):
            ops.append(('swap', i, n_qubits - 1 - i))

    if inverse:
        ops = [_invert_op(op) for op in reversed(ops)]
    return ops
