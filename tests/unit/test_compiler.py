"""
Unit tests for dense_evolution/compiler.py -- QuantumTranspiler's Toffoli/
SWAP decomposition and pass-through transpilation.

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why.
"""
from dense_evolution import QuantumTranspiler, GATES

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
