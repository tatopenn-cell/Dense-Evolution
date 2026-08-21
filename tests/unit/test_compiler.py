"""
Unit tests for dense_evolution/compiler.py -- QuantumTranspiler's Toffoli/
SWAP/iSWAP/ECR/U2/U3 decomposition and pass-through transpilation.

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why.
"""
import numpy as np
import pytest

from dense_evolution import QuantumTranspiler, GATES, DenseSVSimulator
from dense_evolution.circuits.gates import PARAMETRIC_GATES

from _helpers import probs

# ─────────────────────────────────────────────────────────────
# TRANSPILER
# ─────────────────────────────────────────────────────────────

class TestTranspiler:

    def test_ccx_decomposition_length(self):
        result = QuantumTranspiler.decompose_toffoli(0, 1, 2)
        assert len(result) == 15

    def test_swap_decomposition_length(self):
        result = QuantumTranspiler.decompose_swap(0, 1)
        assert len(result) == 3

    def test_transpile_passes_through_basic_gates(self):
        circuit = [('h', 0), ('x', 1), ('cx', 0, 1)]
        result = QuantumTranspiler.transpile(circuit)
        assert result == circuit

    def test_transpile_expands_ccx(self):
        circuit = [('ccx', 0, 1, 2)]
        result = QuantumTranspiler.transpile(circuit)
        assert len(result) == 15
        assert all(op[0] in ('h', 'cx', 't', 'tdg') for op in result)

    def test_toffoli_correctness(self, sim3):
        """CCX|110⟩ = |111⟩"""
        sim3.apply_gate_1q(GATES['x'], 0)
        sim3.apply_gate_1q(GATES['x'], 1)
        sim3.run_circuit([('ccx', 0, 1, 2)])
        p = probs(sim3)
        assert p[7] > 0.99  # |111⟩

    def test_toffoli_no_flip_without_both_controls(self, sim3):
        """CCX|100⟩ = |100⟩ (only one control active)"""
        sim3.apply_gate_1q(GATES['x'], 0)
        sim3.run_circuit([('ccx', 0, 1, 2)])
        p = probs(sim3)
        assert p[4] > 0.99  # |100⟩


class TestU3Decomposition:
    """prog.txt: run_circuit_jit's GATE_IDS was missing u2/u3 (and ecr/
    iswap, not yet closed) -- U3(theta,phi,lam) = e^{i(phi+lam)/2} *
    Rz(phi)*Ry(theta)*Rz(lam) is an EXACT ZYZ decomposition (not just
    equivalent up to global phase), verified against the literal
    PARAMETRIC_GATES['u3'] matrix here."""

    def test_decompose_u3_matches_literal_matrix_exactly(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            theta, phi, lam = rng.uniform(-np.pi, np.pi, size=3)
            sim = DenseSVSimulator(1, use_float32=False)
            sim.run_circuit([('u3', 0, theta, phi, lam)])
            sv_decomposed = np.asarray(sim.get_statevector())

            sim_direct = DenseSVSimulator(1, use_float32=False)
            sim_direct.apply_gate_1q(np.asarray(PARAMETRIC_GATES['u3'](theta, phi, lam)), 0)
            sv_literal = np.asarray(sim_direct.get_statevector())

            np.testing.assert_allclose(sv_decomposed, sv_literal, atol=1e-10)

    def test_decompose_u2_matches_literal_matrix_exactly(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            phi, lam = rng.uniform(-np.pi, np.pi, size=2)
            sim = DenseSVSimulator(1, use_float32=False)
            sim.run_circuit([('u2', 0, phi, lam)])
            sv_decomposed = np.asarray(sim.get_statevector())

            sim_direct = DenseSVSimulator(1, use_float32=False)
            sim_direct.apply_gate_1q(np.asarray(PARAMETRIC_GATES['u2'](phi, lam)), 0)
            sv_literal = np.asarray(sim_direct.get_statevector())

            np.testing.assert_allclose(sv_decomposed, sv_literal, atol=1e-10)

    def test_run_circuit_and_run_circuit_jit_agree_on_u3(self):
        sim_eager = DenseSVSimulator(1, use_float32=False)
        sim_eager.run_circuit([('u3', 0, 0.9, -1.2, 2.1)])
        sim_jit = DenseSVSimulator(1, use_float32=False)
        sim_jit.run_circuit_jit([('u3', 0, 0.9, -1.2, 2.1)])
        np.testing.assert_allclose(
            np.asarray(sim_eager.get_statevector()), np.asarray(sim_jit.get_statevector()), atol=1e-10)

    def test_transpile_structural_only_u3_tuple_passes_through_unchanged(self):
        """dense_evolution.solvers.autodiff._build_template runs a
        structural-only pass through transpile -- (name, *qubits), no
        parameter values -- for gates whose real value is injected later
        via a traced theta. decompose_u3 needs real theta/phi/lam and
        can't run on a qubit-only tuple; transpile must leave it alone
        (not crash on missing arguments) so the caller's own downstream
        "u2/u3 unsupported here" check still fires."""
        circuit = [('u3', 0)]  # no params -- exactly _build_template's shape
        result = QuantumTranspiler.transpile(circuit)
        assert result == circuit

    def test_transpile_structural_only_u2_tuple_passes_through_unchanged(self):
        circuit = [('u2', 0)]
        result = QuantumTranspiler.transpile(circuit)
        assert result == circuit

    def test_gphase_scales_whole_statevector(self):
        """GPhase(a) = e^{ia}*I -- applying it to one qubit of an
        entangled register must multiply the ENTIRE statevector by
        e^{ia}, not just that qubit's local amplitudes."""
        alpha = 0.37
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        sv_before = np.asarray(sim.get_statevector()).copy()
        sim.run_circuit([('gphase', 0, alpha)])
        sv_after = np.asarray(sim.get_statevector())
        np.testing.assert_allclose(sv_after, sv_before * np.exp(1j * alpha), atol=1e-10)

    def test_gphase_jit_matches_eager(self):
        alpha = -1.1
        sim_eager = DenseSVSimulator(2, use_float32=False)
        sim_eager.run_circuit([('h', 0), ('cx', 0, 1), ('gphase', 0, alpha)])
        sim_jit = DenseSVSimulator(2, use_float32=False)
        sim_jit.run_circuit_jit([('h', 0), ('cx', 0, 1), ('gphase', 0, alpha)])
        np.testing.assert_allclose(
            np.asarray(sim_eager.get_statevector()), np.asarray(sim_jit.get_statevector()), atol=1e-10)


class TestIswapEcrDecomposition:
    """prog.txt: iSWAP and ECR were the last 2 gates missing from
    GATE_IDS/run_circuit_jit (after u2/u3/ccx/swap were closed above).
    Decompositions cross-checked against Qiskit's own iSwapGate/ECRGate
    transpiled to a {cx,h,s,sx,x} basis (qiskit==2.5.0), then verified
    numerically against this project's own literal GATES matrices on
    random 2-qubit states -- not derived from memory."""

    def _random_2q_state(self, seed):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        return v / np.linalg.norm(v)

    @pytest.mark.parametrize("gate_name", ["iswap", "ecr"])
    def test_decomposition_matches_literal_matrix_exactly(self, gate_name):
        for seed in range(10):
            psi0 = self._random_2q_state(seed)

            sim_direct = DenseSVSimulator(2, use_float32=False)
            sim_direct.sv = np.asarray(psi0, dtype=complex)
            sim_direct.apply_gate_2q(GATES[gate_name], 0, 1)
            sv_direct = np.asarray(sim_direct.get_statevector())

            sim_decomp = DenseSVSimulator(2, use_float32=False)
            sim_decomp.sv = np.asarray(psi0, dtype=complex)
            sim_decomp.run_circuit([(gate_name, 0, 1)], transpile=True)
            sv_decomp = np.asarray(sim_decomp.get_statevector())

            np.testing.assert_allclose(sv_direct, sv_decomp, atol=1e-9)

    @pytest.mark.parametrize("gate_name", ["iswap", "ecr"])
    def test_run_circuit_and_run_circuit_jit_agree(self, gate_name):
        ops = [('h', 0), ('rz', 1, 0.4), (gate_name, 0, 1)]
        sim_eager = DenseSVSimulator(2, use_float32=False)
        sim_eager.run_circuit(ops)
        sim_jit = DenseSVSimulator(2, use_float32=False)
        sim_jit.run_circuit_jit(ops)
        np.testing.assert_allclose(
            np.asarray(sim_eager.get_statevector()), np.asarray(sim_jit.get_statevector()), atol=1e-9)

    def test_all_six_formerly_unsupported_gates_now_run_on_jit(self):
        """Full gate parity check: ccx/swap/u2/u3/iswap/ecr -- every gate
        GATES/PARAMETRIC_GATES defines -- now runs on run_circuit_jit,
        not just the eager path. run_circuit's auto-dispatch (see
        TestRunCircuitAutoDispatch) has no remaining gate that forces the
        eager fallback."""
        cases = [
            ("ccx", [('h', 0), ('h', 1), ('ccx', 0, 1, 2)], 3),
            ("swap", [('h', 0), ('x', 1), ('swap', 0, 1)], 2),
            ("ecr", [('h', 0), ('ecr', 0, 1)], 2),
            ("iswap", [('h', 0), ('iswap', 0, 1)], 2),
            ("u2", [('u2', 0, 0.3, 0.5)], 1),
            ("u3", [('u3', 0, 0.3, 0.5, 0.7)], 1),
        ]
        for gate_name, ops, n in cases:
            sim_eager = DenseSVSimulator(n, use_float32=False)
            sim_eager.run_circuit(ops)
            sim_jit = DenseSVSimulator(n, use_float32=False)
            sim_jit.run_circuit_jit(ops)
            np.testing.assert_allclose(
                np.asarray(sim_eager.get_statevector()), np.asarray(sim_jit.get_statevector()), atol=1e-9,
                err_msg=f"{gate_name} disagreed between eager and jit",
            )


class TestRunCircuitAutoDispatch:
    """run_circuit now auto-delegates to the compiled path whenever every
    gate (post-transpile) is in GATE_IDS -- so a caller who has never
    heard of run_circuit_jit still gets it automatically."""

    def test_matches_run_circuit_jit_for_a_supported_circuit(self, sim3):
        ops = [('h', 0), ('cx', 0, 1), ('rz', 2, 0.5), ('cx', 1, 2)]
        sim3.run_circuit(ops)
        sim_jit = DenseSVSimulator(3, use_float32=False)
        sim_jit.run_circuit_jit(ops)
        np.testing.assert_allclose(
            np.asarray(sim3.get_statevector()), np.asarray(sim_jit.get_statevector()), atol=1e-10)

    def test_falls_back_to_eager_for_an_unsupported_gate(self, sim2):
        # ecr has no GATE_IDS entry yet (unlike u2/u3/ccx/swap, all of
        # which are now either natively supported or decomposed) -- must
        # still work via the eager per-gate loop, not raise.
        sim2.run_circuit([('h', 0), ('ecr', 0, 1)])
        assert abs(np.linalg.norm(np.asarray(sim2.get_statevector())) - 1.0) < 1e-10
