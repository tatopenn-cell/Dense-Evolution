"""
Real simulation engine for the dashboard.

OpenQASM text -> a Qiskit QuantumCircuit -> executed on dense_evolution's
actual DenseSVSimulator (not Qiskit's own simulator) -> statevector,
probabilities and shot counts, all reordered into Qiskit's little-endian
qubit convention so they line up with the Circuit tab's qubit labels and
with qiskit.visualization's functions (which assume that convention).

No synthetic/placeholder data anywhere here: every quantity returned is
computed from a real run of the real engine.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

import dense_evolution as de

__all__ = ['SimulationResult', 'run_circuit_from_qasm']


def _to_qiskit_bit_order(values: np.ndarray, n_qubits: int) -> np.ndarray:
    """Bit-reverse index order: Dense-Evolution is MSB-first (qubit 0 =
    most significant bit of the index), Qiskit is little-endian (qubit 0
    = least significant bit) -- the same remap dense_evolution.interop
    applies to probabilities internally, used here for the raw complex
    statevector too since it is the identical index permutation."""
    perm = [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
    return values[perm]


@dataclass
class SimulationResult:
    qiskit_circuit: QuantumCircuit
    n_qubits: int
    statevector: np.ndarray    # complex128, Qiskit bit order
    probabilities: np.ndarray  # Qiskit bit order
    counts: dict                # Qiskit-style bitstring -> shot count


def run_circuit_from_qasm(
    qasm_text: str,
    n_shots: int = 1000,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Parse `qasm_text` and run it on dense_evolution's real
    DenseSVSimulator, returning every quantity the dashboard's tabs need."""
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_text)
    n_qubits = qiskit_circuit.num_qubits
    if n_qubits < 1:
        raise ValueError("circuit must declare at least 1 qubit")

    sim, probabilities = de.run_qiskit_circuit(qiskit_circuit, use_float32=False)
    statevector = _to_qiskit_bit_order(np.asarray(sim.sv), n_qubits)

    rng = np.random.default_rng(seed)
    # sample_counts expects/returns dense_evolution's native MSB-first
    # bitstring convention -- reverse each key to relabel into Qiskit's
    # little-endian convention (same physical samples, just relabeled to
    # match the Circuit/Probabilities tabs' qubit numbering).
    counts_native = de.sample_counts(np.asarray(sim.sv), n_shots, rng=rng)
    counts = {key[::-1]: n for key, n in counts_native.items()}

    return SimulationResult(
        qiskit_circuit=qiskit_circuit,
        n_qubits=n_qubits,
        statevector=statevector,
        probabilities=probabilities,
        counts=counts,
    )
