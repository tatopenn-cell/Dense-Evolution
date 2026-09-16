"""Small assertion helpers shared across a few test files (not fixtures --
plain functions, imported explicitly where needed)."""
import numpy as np


def norm(sim):
    return float(np.linalg.norm(sim.get_statevector()))


def probs(sim):
    return sim.get_probabilities()


def _bit_reversal_perm(n_qubits):
    """MSB-first <-> little-endian bit-reversal permutation (prog.txt
    test-suite audit, issue #269 point 10) -- the same one-line list
    comprehension used to live independently in both
    test_dashboard_engine.py::_reference_qiskit_bit_order (a deliberately
    naive reference, kept separate from dense_evolution.interop's own
    cached _to_qiskit_bit_order so the two can be checked against each
    other) and test_interop_stim_bridge.py::_to_le_order (STIM's
    convention). Both keep their own name/docstring/call signature --
    they serve different documented purposes -- only this shared
    permutation logic moved here."""
    return [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
