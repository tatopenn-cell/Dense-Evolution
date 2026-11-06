"""OpenQASM of an entangling layer, for the Topology tab of the dashboard."""

import dense_evolution as de
from dense_evolution.circuits.topology import VALID_PATTERNS

from .qasm_library import gate_tuples_to_qasm

__all__ = ["topology_qasm"]

_GATES = ("cx", "cz", "cy")


def topology_qasm(
    n_qubits: int,
    pattern: str,
    gate: str = "cx",
    reverse: bool = False,
    hub: int = 0,
    hadamards: bool = True,
) -> str:
    """OpenQASM 2.0 of an entangling layer built by ``entangling_layer``.

    Args:
        n_qubits: number of qubits, at least 2.
        pattern: one of ``dense_evolution.circuits.topology.VALID_PATTERNS``.
        gate: two-qubit gate of every edge: ``"cx"``, ``"cz"`` or ``"cy"``.
        reverse: swap control and target of every edge.
        hub: hub qubit, used only by the ``"star"`` pattern.
        hadamards: start with one Hadamard on every qubit.

    Returns:
        The OpenQASM text of the circuit.

    Raises:
        ValueError: for fewer than 2 qubits, an unknown pattern, an unknown
            gate, or a ``hub`` outside ``0 <= hub < n_qubits`` with ``"star"``.

    Examples:
        >>> print(topology_qasm(3, "linear", hadamards=False), end="")
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        cx q[0],q[1];
        cx q[1],q[2];
        measure q -> c;
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"pattern must be one of {VALID_PATTERNS}, got {pattern!r}")
    if gate not in _GATES:
        raise ValueError(f"gate must be one of {_GATES}, got {gate!r}")
    if pattern == "star" and not 0 <= hub < n_qubits:
        raise ValueError(f"hub must satisfy 0 <= hub < {n_qubits}, got {hub}")
    ops = de.entangling_layer(
        n_qubits, pattern=pattern, gate=gate, reverse=reverse, hub=hub
    )
    if hadamards:
        ops = [("h", q) for q in range(n_qubits)] + ops
    return gate_tuples_to_qasm(ops, n_qubits)
