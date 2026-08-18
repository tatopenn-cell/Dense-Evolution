"""
Tests for dense_evolution.interop.to_stim (STIM bridge).

STIM is an optional dependency — pytest.importorskip guards the
correctness tests so this file stays green without it installed, on top
of CI installing it explicitly.
"""
import numpy as np
import pytest

import dense_evolution as de
from dense_evolution.interop import qiskit_pennylane as interop
from dense_evolution.interop import to_stim
from dense_evolution.simulator import DenseSVSimulator

stim = pytest.importorskip("stim")


def _to_le_order(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reorder an MSB-first probability array (Dense-Evolution's own
    convention) into little-endian order (STIM's/Qiskit's convention) via
    a bit-reversal permutation. Takes probabilities directly -- callers
    must not pass raw amplitudes here (squaring already-squared
    probabilities silently halves the apparent magnitude and was a real
    bug caught in this exact test)."""
    perm = [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
    return probs[perm]


class TestToStimCliffordCircuits:

    def test_bell_state_matches_dense_sv_simulator(self):
        ops = [['h', 0], ['cx', 0, 1]]
        n_qubits = 2

        sim = DenseSVSimulator(n_qubits=n_qubits, use_float32=False)
        sim.run_circuit(ops)
        de_probs = np.asarray(sim.get_probabilities())

        circuit = to_stim(ops, n_qubits)
        tsim = stim.TableauSimulator()
        tsim.do(circuit)
        stim_probs = np.abs(np.asarray(tsim.state_vector())) ** 2
        # STIM's state_vector() is little-endian (qubit 0 = LSB), same as
        # Qiskit's convention -- reorder DE's MSB-first probs to compare.
        de_probs_le = _to_le_order(de_probs, n_qubits)
        assert np.allclose(de_probs_le, stim_probs, atol=1e-9)

    def test_larger_clifford_circuit_matches_dense_sv_simulator(self):
        ops = [['h', 0], ['s', 0], ['cx', 0, 1], ['cy', 1, 2],
               ['sx', 2], ['sdg', 1], ['x', 0], ['y', 1], ['z', 2],
               ['h', 2], ['cz', 0, 2]]
        n_qubits = 3

        sim = DenseSVSimulator(n_qubits=n_qubits, use_float32=False)
        sim.run_circuit(ops)
        de_probs = np.asarray(sim.get_probabilities())

        circuit = to_stim(ops, n_qubits)
        tsim = stim.TableauSimulator()
        tsim.do(circuit)
        stim_probs = np.abs(np.asarray(tsim.state_vector())) ** 2

        de_probs_le = _to_le_order(de_probs, n_qubits)
        assert np.allclose(de_probs_le, stim_probs, atol=1e-9)

    def test_untouched_high_qubit_still_sets_circuit_width(self):
        circuit = to_stim([['h', 0]], n_qubits=4)
        assert circuit.num_qubits == 4


class TestToStimNonCliffordRejection:

    def test_parametric_rotation_raises_value_error(self):
        with pytest.raises(ValueError, match='rx'):
            to_stim([['h', 0], ['rx', 0, 0.3]], n_qubits=1)

    def test_t_gate_raises_value_error(self):
        with pytest.raises(ValueError, match='t'):
            to_stim([['t', 0]], n_qubits=1)


class TestToStimImportGuard:

    def test_missing_stim_raises_clear_importerror(self, monkeypatch):
        monkeypatch.setattr(interop, 'HAS_STIM', False)
        with pytest.raises(ImportError, match='stim'):
            to_stim([['h', 0]], n_qubits=1)
