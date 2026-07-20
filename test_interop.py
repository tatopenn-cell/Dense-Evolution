"""
Tests for dense_evolution.interop (Qiskit / PennyLane bridge).

Both frameworks are optional dependencies — pytest.importorskip guards each
class so this file stays green in environments without qiskit/pennylane
installed, on top of CI installing both explicitly.
"""
import numpy as np
import pytest

import dense_evolution as de
from dense_evolution import interop
from dense_evolution.interop import (
    from_qiskit, from_pennylane, run_qiskit_circuit, run_pennylane_circuit,
    _to_qiskit_bit_order,
)


# ─────────────────────────────────────────────────────────────
# Import safety — must hold regardless of qiskit/pennylane presence
# ─────────────────────────────────────────────────────────────

class TestImportSafety:

    def test_root_import_never_fails(self):
        # this is the regression itself: interop.py's try/except pattern
        # must make `import dense_evolution` safe even if qiskit/pennylane
        # were both absent — can't literally uninstall them mid-suite, so
        # this asserts the exported symbols exist and mirrors registry.py's
        # HAS_JAX/HAS_CUPY pattern instead.
        assert hasattr(de, 'from_qiskit')
        assert hasattr(de, 'from_pennylane')
        assert hasattr(de, 'run_qiskit_circuit')
        assert hasattr(de, 'run_pennylane_circuit')

    def test_missing_qiskit_raises_clear_importerror(self, monkeypatch):
        monkeypatch.setattr(interop, 'HAS_QISKIT', False)
        with pytest.raises(ImportError, match='qiskit'):
            from_qiskit(None)

    def test_missing_pennylane_raises_clear_importerror(self, monkeypatch):
        monkeypatch.setattr(interop, 'HAS_PENNYLANE', False)
        with pytest.raises(ImportError, match='pennylane'):
            from_pennylane(None)


# ─────────────────────────────────────────────────────────────
# Qiskit
# ─────────────────────────────────────────────────────────────

class TestQiskitInterop:

    qiskit = pytest.importorskip('qiskit')

    @staticmethod
    def _asymmetric_circuit():
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 2)
        qc.crz(0.3, 1, 2)
        return qc

    def test_from_qiskit_structure(self):
        qc = self._asymmetric_circuit()
        circ = from_qiskit(qc)
        assert circ.n_qubits == 3
        names = [op['name'] for op in circ.ops]
        assert names == ['h', 'cx', 'rx', 'crz']

    def test_run_qiskit_circuit_matches_statevector_probabilities(self):
        from qiskit.quantum_info import Statevector
        qc = self._asymmetric_circuit()
        qk_probs = Statevector.from_instruction(qc).probabilities()
        _, de_probs = run_qiskit_circuit(qc, use_float32=False)
        np.testing.assert_allclose(de_probs, qk_probs, atol=1e-6)

    def test_bit_order_regression_asymmetric_circuit(self):
        # X only on qubit 0 of 3 -> must land on qiskit index 1 (q0 = LSB),
        # not index 4 (which would be the DE-native MSB-first index) —
        # pins the exact convention, not just "some permutation happened to
        # work" on a symmetric circuit.
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.x(0)
        _, probs = run_qiskit_circuit(qc, use_float32=False)
        nonzero = np.where(probs > 1e-9)[0]
        assert list(nonzero) == [1]

    def test_custom_gate_definition_does_not_corrupt_following_statement(self):
        # qiskit.qasm2.dumps emits composite gates (e.g. mcx) as a `gate
        # NAME params { ... }` block on a single line — same brace-block
        # corruption class as QASM3 for/if/while/def, fixed by widening
        # _RE_BLOCK_HEAD to also strip `gate` blocks. mcx itself has no
        # physical implementation in this simulator (unknown gate name,
        # silent no-op elsewhere in run_circuit too) — that part is a real,
        # separate, documented limitation, not something this test hides.
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.mcx([0, 1, 2], 3)
        circ = from_qiskit(qc)
        assert circ.n_qubits == 4
        assert [op['name'] for op in circ.ops] == ['h', 'mcx']

    def test_to_qiskit_bit_order_is_involution(self):
        # bit-reversal applied twice must return the original array
        rng = np.random.default_rng(0)
        probs = rng.random(2 ** 3)
        once = _to_qiskit_bit_order(probs, 3)
        twice = _to_qiskit_bit_order(once, 3)
        np.testing.assert_allclose(twice, probs)


# ─────────────────────────────────────────────────────────────
# PennyLane
# ─────────────────────────────────────────────────────────────

class TestPennyLaneInterop:

    pennylane = pytest.importorskip('pennylane')

    def test_from_pennylane_qnode_structure(self):
        import pennylane as qml
        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(0.5, wires=2)
            qml.CRZ(0.3, wires=[1, 2])
            return qml.probs(wires=[0, 1, 2])

        circ = from_pennylane(circuit)
        assert circ.n_qubits == 3
        assert [op['name'] for op in circ.ops] == ['h', 'cx', 'rx', 'crz']

    def test_from_pennylane_tape_input(self):
        import pennylane as qml
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])

        circ = from_pennylane(tape)
        assert circ.n_qubits == 2
        assert [op['name'] for op in circ.ops] == ['h', 'cx']

    def test_run_pennylane_circuit_matches_qml_probs_no_reordering(self):
        import pennylane as qml
        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(0.5, wires=2)
            qml.CRZ(0.3, wires=[1, 2])
            return qml.probs(wires=[0, 1, 2])

        pl_probs = np.asarray(circuit())
        _, de_probs = run_pennylane_circuit(circuit, use_float32=False)
        np.testing.assert_allclose(de_probs, pl_probs, atol=1e-6)

    def test_bit_order_regression_asymmetric_circuit(self):
        # Same asymmetric single-qubit-X probe as the Qiskit test, but here
        # NO reordering should be needed at all — PennyLane's own wire
        # convention already matches Dense-Evolution's MSB-first indexing.
        import pennylane as qml
        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=[0, 1, 2])

        _, probs = run_pennylane_circuit(circuit, use_float32=False)
        nonzero = np.where(probs > 1e-9)[0]
        assert list(nonzero) == [4]  # MSB-first: X on qubit 0 -> index 100b = 4
