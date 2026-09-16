"""Regression test for _helpers.py::_bit_reversal_perm (prog.txt
test-suite audit, issue #269 point 10) -- consolidated from three
independent copies (test_dashboard_engine.py, test_interop.py's import
of the real dense_evolution.interop one, test_interop_stim_bridge.py)
into this one shared helper."""
import numpy as np

from _helpers import _bit_reversal_perm


def test_permutation_is_involution():
    for n_qubits in (1, 2, 3, 4, 5):
        perm = _bit_reversal_perm(n_qubits)
        values = np.arange(2 ** n_qubits)
        once = values[perm]
        twice = once[perm]
        np.testing.assert_array_equal(twice, values)


def test_single_qubit_is_identity():
    assert _bit_reversal_perm(1) == [0, 1]


def test_two_qubit_matches_known_permutation():
    # MSB-first index 1 (0b01, qubit0=0,qubit1=1) <-> LSB-first index 2 (0b10)
    assert _bit_reversal_perm(2) == [0, 2, 1, 3]
