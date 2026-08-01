"""
Turns the ops list produced by the graphical (drag-and-drop) circuit
builder component into a real Qiskit QuantumCircuit -- the same object
type dashboard_core.engine already runs on the real dense_evolution
DenseSVSimulator. No separate execution path for graphically-built
circuits: they go through the exact same engine as typed OpenQASM.
"""

import numpy as np
from qiskit import QuantumCircuit

__all__ = ['GATE_PALETTE', 'ops_to_qiskit_circuit']

# Palette offered by the drag-and-drop grid (dashboard_core.circuit_builder_component).
# kind:
#   'single'  -- dropped directly on one qubit's cell
#   'control' -- the (.) end of a 2-qubit gate; pairs with a 'target' in the same column
#   'target'  -- the other end of a 2-qubit gate, carries which gate (x/y/z -> cx/cy/cz)
#   'swap'    -- two 'swap' markers in the same column form a SWAP
GATE_PALETTE = [
    {"id": "h", "label": "H", "kind": "single", "gate": "h"},
    {"id": "x", "label": "X", "kind": "single", "gate": "x"},
    {"id": "y", "label": "Y", "kind": "single", "gate": "y"},
    {"id": "z", "label": "Z", "kind": "single", "gate": "z"},
    {"id": "s", "label": "S", "kind": "single", "gate": "s"},
    {"id": "t", "label": "T", "kind": "single", "gate": "t"},
    {"id": "rx", "label": "Rx(π/2)", "kind": "single", "gate": "rx"},
    {"id": "ry", "label": "Ry(π/2)", "kind": "single", "gate": "ry"},
    {"id": "rz", "label": "Rz(π/2)", "kind": "single", "gate": "rz"},
    {"id": "ctrl", "label": "●", "kind": "control", "gate": None},
    {"id": "tgt_x", "label": "⊕", "kind": "target", "gate": "x"},
    {"id": "tgt_z", "label": "● Z", "kind": "target", "gate": "z"},
    {"id": "tgt_y", "label": "⊗ Y", "kind": "target", "gate": "y"},
    {"id": "swap", "label": "×", "kind": "swap", "gate": None},
]

_SINGLE_QUBIT_METHODS = {"h", "x", "y", "z", "s", "t"}
_ROTATION_METHODS = {"rx", "ry", "rz"}

# Fixed rotation angle for graphically-placed Rx/Ry/Rz -- the grid has no
# angle-entry UI yet (out of scope for this pass), so these place a real,
# working pi/2 rotation rather than a fake/placeholder gate.
_DEFAULT_ROTATION_ANGLE = np.pi / 2


def ops_to_qiskit_circuit(n_qubits: int, ops: list) -> QuantumCircuit:
    """Build a real QuantumCircuit from the builder's op list.

    Parameters
    ----------
    n_qubits : int
    ops : list[dict]
        Each dict has 'gate' (one of the GATE_PALETTE gate ids: h/x/y/z/s/t/
        rx/ry/rz/cx/cy/cz/swap) and 'qubits' (list[int]).

    Returns
    -------
    QuantumCircuit
        Ends with measure_all(), matching the QASM presets' convention.
    """
    if n_qubits < 1:
        raise ValueError("circuit must have at least 1 qubit")

    qc = QuantumCircuit(n_qubits)
    for op in ops:
        gate = op.get("gate")
        qubits = op.get("qubits", [])
        for q in qubits:
            if not (0 <= q < n_qubits):
                raise ValueError(f"qubit index {q} out of range for {n_qubits}-qubit circuit")

        if gate in _SINGLE_QUBIT_METHODS:
            (q,) = qubits
            getattr(qc, gate)(q)
        elif gate in _ROTATION_METHODS:
            (q,) = qubits
            getattr(qc, gate)(_DEFAULT_ROTATION_ANGLE, q)
        elif gate == "swap":
            a, b = qubits
            qc.swap(a, b)
        elif gate in ("cx", "cy", "cz"):
            control, target = qubits
            getattr(qc, gate)(control, target)
        else:
            raise ValueError(f"unknown gate from circuit builder: {gate!r}")

    qc.measure_all()
    return qc
