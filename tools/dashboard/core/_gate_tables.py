"""Gate-name classification, shared by qasm_library.py and
circuit_diagram.py (prog.txt, dashboard_core audit point 2). Was: two
independently hand-copied sets of the same 5 constants, each with a
comment claiming to mirror a different, non-existent "canonical" source
(dense_evolution/compiler.py's gate categories, dashboard_core/engine.py's
dispatch tables) -- neither actually defines these as a literal data
structure. The two copies had already drifted apart in practice:
circuit_diagram.py's _ONE_QUBIT_PARAM was missing the 'u1'/'phase'
aliases qasm_library.py's copy had, so a circuit using either of those
two (both real gate names dense_evolution.circuits.gates.GATE_IDS
recognizes, both mapped to the same gate as 'p') crashed
draw_native_circuit_diagram with "unsupported gate for native circuit
diagram" -- a real bug, not just a duplication risk, fixed by making
this the one place these sets are defined.

Matches dense_evolution.circuits.gates.GATE_IDS's own gate-name
vocabulary (the actual source of truth for which names QASMParser
accepts) -- verified directly, not assumed, before writing this down.
"""

__all__ = [
    '_ONE_QUBIT_STATIC', '_ONE_QUBIT_PARAM', '_TWO_QUBIT_STATIC',
    '_TWO_QUBIT_PARAM', '_THREE_QUBIT_STATIC',
]

_ONE_QUBIT_STATIC = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "id"}
_ONE_QUBIT_PARAM = {"rx", "ry", "rz", "p", "u1", "phase"}
_TWO_QUBIT_STATIC = {"cx", "cz", "cy", "swap"}
_TWO_QUBIT_PARAM = {"cp", "crz"}
_THREE_QUBIT_STATIC = {"ccx"}
