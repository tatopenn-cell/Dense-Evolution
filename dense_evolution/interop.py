from typing import Optional, Tuple
import numpy as np

from .parser import QASMParser, QASMCircuit
from .simulator import DenseSVSimulator

try:
    import qiskit.qasm2 as _qasm2
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def _require_qiskit():
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit interop requires the 'qiskit' package. "
            "Install it with: pip install dense-evolution[qiskit]")


def _require_pennylane():
    if not HAS_PENNYLANE:
        raise ImportError(
            "PennyLane interop requires the 'pennylane' package. "
            "Install it with: pip install dense-evolution[pennylane]")


def _to_qiskit_bit_order(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Reindex a probability array from Dense-Evolution's native MSB-first
    convention (qubit 0 = most significant bit of the index, the same
    convention as apply_gate_1q/apply_gate_2q/measure/beast-mode) to
    Qiskit's little-endian convention (qubit 0 = least significant bit),
    so the result is directly comparable to Statevector(...).probabilities().

    A plain bit-reversal permutation of the index — verified against
    Statevector.from_instruction(...).probabilities() on an asymmetric
    circuit (exact match only after this reversal, not before).
    """
    perm = [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
    return probs[perm]


def from_qiskit(circuit) -> QASMCircuit:
    """Convert a Qiskit QuantumCircuit into a QASMCircuit via OpenQASM 2.0
    (qiskit.qasm2.dumps), reusing the existing QASMParser rather than a
    bespoke gate-by-gate translator."""
    _require_qiskit()
    qasm_str = _qasm2.dumps(circuit)
    return QASMParser().parse(qasm_str)


def from_pennylane(circuit, *args, **kwargs) -> QASMCircuit:
    """Convert a PennyLane QNode or QuantumTape/QuantumScript into a
    QASMCircuit via OpenQASM 2.0 (qml.to_openqasm), reusing the existing
    QASMParser.

    qml.to_openqasm returns the QASM string directly for an
    already-built tape/QuantumScript, but returns a wrapper function that
    must be called with the QNode's own arguments for a parametric QNode
    — both cases are handled here.
    """
    _require_pennylane()
    result = qml.to_openqasm(circuit, measure_all=False)
    qasm_str = result if isinstance(result, str) else result(*args, **kwargs)
    return QASMParser().parse(qasm_str)


def run_qiskit_circuit(
    circuit,
    use_float32: bool = True,
    sim: Optional[DenseSVSimulator] = None,
) -> Tuple[DenseSVSimulator, np.ndarray]:
    """Run a Qiskit QuantumCircuit on DenseSVSimulator. Returns
    (sim, probabilities) with probabilities reordered into Qiskit's own
    little-endian bit convention, so they compare directly against
    Statevector(circuit).probabilities() — see _to_qiskit_bit_order."""
    circ = from_qiskit(circuit)
    if sim is None:
        sim = DenseSVSimulator(n_qubits=circ.n_qubits, use_float32=use_float32)
    sim.run_circuit(circ.to_tuples())
    probs = np.asarray(sim.get_probabilities())
    return sim, _to_qiskit_bit_order(probs, circ.n_qubits)


def run_pennylane_circuit(
    circuit,
    *args,
    use_float32: bool = True,
    sim: Optional[DenseSVSimulator] = None,
    **kwargs,
) -> Tuple[DenseSVSimulator, np.ndarray]:
    """Run a PennyLane QNode/tape on DenseSVSimulator. Returns
    (sim, probabilities) in Dense-Evolution's native ordering, WITHOUT any
    bit-reversal — unlike run_qiskit_circuit, because PennyLane's own wire
    convention (wire 0 = most significant) already matches Dense-Evolution's
    MSB-first convention. Do not "symmetrize" this with the Qiskit version;
    that would silently misorder circuits that are asymmetric under qubit
    reversal (verified directly: no permutation needed here, one is
    required for Qiskit — the two frameworks are genuinely different)."""
    circ = from_pennylane(circuit, *args, **kwargs)
    if sim is None:
        sim = DenseSVSimulator(n_qubits=circ.n_qubits, use_float32=use_float32)
    sim.run_circuit(circ.to_tuples())
    probs = np.asarray(sim.get_probabilities())
    return sim, probs
