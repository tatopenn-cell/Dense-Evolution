"""
Unit tests for dense_evolution/qft.py -- cross-checked against the
brute-force analytic DFT matrix, not just against its own derivation.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.circuits.qft import qft


def _dft_matrix(dim):
    """M[j, k] = exp(2*pi*i*j*k/dim) / sqrt(dim), so QFT|j> = sum_k M[j,k]|k>."""
    j_idx, k_idx = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    return np.exp(2j * np.pi * j_idx * k_idx / dim) / np.sqrt(dim)


class TestQft:

    @pytest.mark.parametrize('n_qubits', [1, 2, 3, 4])
    def test_matches_brute_force_dft_matrix(self, n_qubits):
        dim = 2 ** n_qubits
        M = _dft_matrix(dim)
        for j in range(dim):
            sim = DenseSVSimulator(n_qubits, use_float32=False)
            init = np.zeros(dim, dtype=complex)
            init[j] = 1.0
            sim.set_initial_state(init)
            sim.run_circuit(qft(n_qubits))
            got = sim.get_statevector()
            assert np.max(np.abs(got - M[j])) < 1e-9

    def test_qft_of_zero_state_is_uniform_superposition(self):
        n_qubits = 3
        sim = DenseSVSimulator(n_qubits, use_float32=False)
        sim.run_circuit(qft(n_qubits))
        probs = sim.get_probabilities()
        assert np.allclose(probs, 1.0 / 2 ** n_qubits, atol=1e-9)

    def test_round_trip_qft_and_inverse_recovers_original_state(self):
        n_qubits = 3
        dim = 2 ** n_qubits
        rng = np.random.default_rng(0)
        psi0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi0 /= np.linalg.norm(psi0)

        sim = DenseSVSimulator(n_qubits, use_float32=False)
        sim.set_initial_state(psi0.copy())
        sim.run_circuit(qft(n_qubits) + qft(n_qubits, inverse=True))
        assert np.max(np.abs(sim.get_statevector() - psi0)) < 1e-9

    def test_without_swaps_differs_from_with_swaps(self):
        assert qft(3, do_swaps=False) != qft(3, do_swaps=True)
        assert qft(3, do_swaps=False) == qft(3, do_swaps=True)[:-1]

    def test_rejects_fewer_than_one_qubit(self):
        with pytest.raises(ValueError):
            qft(0)
