"""
Tests for dashboard_core.vqe.run_vqe -- real VQE (Adam + adjoint
differentiation on lightning.qubit), both the hardware-efficient and
UCCSD ansatz families, plus the n_layers=0/maxiter=0 fast path that
returns just the bare Hartree-Fock reference circuit.

H2 only (fastest real molecule this project has) to keep this file's
real optimization runs fast -- each still a genuine gradient-descent
run against H2's real Hamiltonian, not a stub. No Qiskit involved
(vqe.py never imports it), so no macOS skip needed here.
"""

import numpy as np
import pytest

from dashboard_core.vqe import run_vqe

H2_SYMBOLS = ['H', 'H']
H2_GEOMETRY = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]])
H2_EXACT = -1.1372701748786913
H2_HF = -1.1166843872194083


class TestFastPath:

    def test_maxiter_zero_returns_bare_hartree_fock_energy(self):
        result = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, n_layers=0, maxiter=0)
        assert result['ansatz_type'] == 'hartree_fock'
        assert result['n_params'] == 0
        assert result['n_qubits'] == 4
        assert result['hf_occupation'] == [1, 1, 0, 0]
        assert result['vqe_energy_hartree'] == pytest.approx(H2_HF, abs=1e-9)
        assert result['exact_energy_hartree'] == pytest.approx(H2_EXACT, abs=1e-6)

    def test_fast_path_qasm_is_just_x_gates_and_measure(self):
        result = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, n_layers=0, maxiter=0)
        lines = [l.strip() for l in result['qasm'].splitlines() if l.strip()]
        gate_lines = [l for l in lines if not l.startswith(('OPENQASM', 'include', 'qreg', 'creg'))]
        assert gate_lines == ['x q[0];', 'x q[1];', 'measure q -> c;']


class TestHardwareEfficientAnsatz:

    def test_converges_close_to_exact_ground_state(self):
        result = run_vqe(
            H2_SYMBOLS, H2_GEOMETRY, charge=0,
            ansatz_type='hardware_efficient', n_layers=4, maxiter=100, seed=0,
        )
        assert result['ansatz_type'] == 'hardware_efficient'
        assert result['n_layers'] == 4
        assert result['n_params'] == 4 * 4  # n_qubits * n_layers
        assert result['vqe_energy_hartree'] == pytest.approx(H2_EXACT, abs=1e-5)

    def test_energy_improves_over_the_optimization(self):
        result = run_vqe(
            H2_SYMBOLS, H2_GEOMETRY, charge=0,
            ansatz_type='hardware_efficient', n_layers=4, maxiter=100, seed=0,
        )
        history = result['energy_history']
        assert len(history) >= 100
        # Real gradient descent: final energy is (variationally) lower
        # than the first random-initialization energy, and no worse than
        # the bare HF starting point energy would be.
        assert history[-1] < history[0]
        assert history[-1] <= H2_HF + 1e-6

    def test_same_seed_is_deterministic(self):
        r1 = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='hardware_efficient',
                      n_layers=2, maxiter=20, seed=3)
        r2 = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='hardware_efficient',
                      n_layers=2, maxiter=20, seed=3)
        assert r1['vqe_energy_hartree'] == r2['vqe_energy_hartree']


class TestUccsdAnsatz:

    def test_converges_close_to_exact_ground_state(self):
        result = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='uccsd', maxiter=60, seed=0)
        assert result['ansatz_type'] == 'uccsd'
        assert result['n_layers'] is None
        # H2's real fermionic structure: 1 single + 1 double excitation
        # from the qml.qchem.excitations decomposition -- far fewer
        # parameters than the hardware-efficient template above.
        assert result['n_params'] == 3
        assert result['vqe_energy_hartree'] == pytest.approx(H2_EXACT, abs=1e-3)

    def test_uccsd_has_fewer_parameters_than_hardware_efficient(self):
        uccsd = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='uccsd', maxiter=1, seed=0)
        hw = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='hardware_efficient',
                      n_layers=8, maxiter=1, seed=0)
        assert uccsd['n_params'] < hw['n_params']


class TestQasmOutputIsReal:

    def test_generated_circuit_reproduces_the_reported_energy_on_dense_evolution(self):
        # Round-trip check: the returned OpenQASM, executed on
        # dense_evolution's own engine (not PennyLane), must give the
        # same energy PennyLane reported during optimization -- proof
        # the QASM is a faithful translation, not just plausible-looking
        # text.
        import dense_evolution as de

        result = run_vqe(H2_SYMBOLS, H2_GEOMETRY, charge=0, ansatz_type='uccsd', maxiter=40, seed=0)
        parsed = de.QASMParser().parse(result['qasm'])
        sim = de.DenseSVSimulator(parsed.n_qubits, use_float32=False)
        sim.run_circuit(parsed.to_tuples())
        sv = np.asarray(sim.sv)

        H, n_qubits = build_h2_pauli_terms()
        energy = float(np.real(sv.conj() @ H @ sv))
        assert energy == pytest.approx(result['vqe_energy_hartree'], abs=1e-6)


def build_h2_pauli_terms():
    from dashboard_core.hamiltonians import build_molecular_hamiltonian
    return build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
