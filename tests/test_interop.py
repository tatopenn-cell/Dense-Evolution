"""
Tests for dense_evolution.interop (Qiskit / PennyLane bridge).

Both frameworks are optional dependencies — pytest.importorskip guards each
class so this file stays green in environments without qiskit/pennylane
installed, on top of CI installing both explicitly.
"""
import sys

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

# qiskit itself (its compiled Rust extension) is what destabilizes the
# process on macOS CI runners, not any one specific call into it —
# reproduced first as a deterministic SIGSEGV inside
# qiskit.circuit.QuantumCircuit.__init__ on the simplest possible call,
# QuantumCircuit(3) (Python 3.10/3.11/3.12, macos-latest/arm64); after
# skipping that test's *execution* the crash didn't go away, it just
# moved to a non-deterministic segfault during interpreter shutdown --
# because `qiskit = pytest.importorskip('qiskit')` as a class-body
# statement still runs at collection time regardless of a skipif marker
# on the class, so qiskit was still being loaded into the process either
# way. The only fix that actually removes the trigger is to stop
# importing qiskit at all on macOS, by never defining the class there in
# the first place. Not a Dense-Evolution bug: every Dense-Evolution-only
# test (including the rest of this file, TestPennyLaneInterop) passes
# cleanly and deterministically on macOS. Re-enable once this is
# confirmed fixed upstream or traced to a specific dependency conflict.
if sys.platform == 'darwin':

    @pytest.mark.skip(reason="qiskit destabilizes the process on macOS CI runners -- see comment above")
    class TestQiskitInterop:
        pass

else:

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

    def test_non_monotonic_wire_order_does_not_get_renumbered(self):
        # Found via independent fuzz testing: qml.to_openqasm (and the old
        # tape.to_openqasm()) number exported QASM qubits by the order
        # wires are FIRST TOUCHED in the circuit, not by wire index —
        # PauliX(wires=2) then CNOT(wires=[2,1]) used to export as
        # `x q[0]; cx q[0],q[1];`, silently renumbering wire 2->q[0] and
        # wire 1->q[1]. Verified directly this produced a topologically
        # different circuit whenever wires weren't touched in ascending
        # order. Fixed by passing explicit wires= to force true wire order.
        import pennylane as qml
        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[2, 1])
            return qml.probs(wires=range(4))

        ref = np.asarray(circuit())
        _, ours = run_pennylane_circuit(circuit, use_float32=False)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_non_monotonic_wire_order_fuzz(self):
        # Same style fuzz test that originally caught the bug (9/20 passed
        # before the fix) — regression guard against it coming back.
        import pennylane as qml
        dev = qml.device('default.qubit', wires=4)
        rng = np.random.default_rng(1)
        for trial in range(20):
            n_ops = rng.integers(5, 12)
            piano = []
            for _ in range(n_ops):
                tipo = rng.integers(0, 4)
                if tipo < 3:
                    piano.append((int(tipo), int(rng.integers(4))))
                else:
                    a, b = rng.choice(4, 2, replace=False)
                    piano.append((3, int(a), int(b)))

            def circuit(piano=piano):
                for op in piano:
                    if op[0] == 0: qml.Hadamard(wires=op[1])
                    elif op[0] == 1: qml.PauliX(wires=op[1])
                    elif op[0] == 2: qml.RZ(0.7, wires=op[1])
                    else: qml.CNOT(wires=[op[1], op[2]])
                return qml.probs(wires=range(4))

            qnode = qml.QNode(circuit, dev)
            ref = np.asarray(qnode())
            _, ours = run_pennylane_circuit(qnode, use_float32=False)
            assert np.allclose(ref, ours, atol=1e-6), f"trial {trial} mismatch, piano={piano}"

    def test_elaborate_circuit_fuzz(self):
        # The fuzz test above (test_non_monotonic_wire_order_fuzz) is
        # deliberately small -- 4 qubits, 5-12 gates, a 4-gate alphabet
        # (H/X/RZ/CNOT) -- built to isolate one specific wire-ordering
        # bug, not to stand in for a general cross-validation check.
        # This is a separate, larger check: 8 qubits, 40-80 gates, and
        # a much richer gate alphabet spanning fixed single-qubit gates
        # (H, X, Y, Z, S, T), parametric single-qubit gates (RX, RY, RZ),
        # and both a 2-controlled and a 3-qubit entangling gate (CNOT,
        # CZ, SWAP, Toffoli) -- exercising far more of the gate library
        # (see README's "Gate Library" table) and a much larger Hilbert
        # space (2^8=256 amplitudes vs. 2^4=16) than any existing
        # PennyLane cross-check in this repo. 25 random trials, each an
        # independent circuit topology, not 25 repeats of one shape.
        import pennylane as qml
        n_wires = 8
        dev = qml.device('default.qubit', wires=n_wires)
        rng = np.random.default_rng(7)

        FIXED_1Q = ['h', 'x', 'y', 'z', 's', 't']
        PARAM_1Q = ['rx', 'ry', 'rz']
        GATE_2Q = ['cnot', 'cz', 'swap']

        for trial in range(25):
            n_ops = int(rng.integers(40, 81))
            piano = []
            for _ in range(n_ops):
                kind = rng.choice(['fixed1q', 'param1q', 'gate2q', 'toffoli'],
                                   p=[0.35, 0.35, 0.25, 0.05])
                if kind == 'fixed1q':
                    piano.append((rng.choice(FIXED_1Q), int(rng.integers(n_wires))))
                elif kind == 'param1q':
                    piano.append((rng.choice(PARAM_1Q), int(rng.integers(n_wires)),
                                   float(rng.uniform(0, 2 * np.pi))))
                elif kind == 'gate2q':
                    a, b = rng.choice(n_wires, 2, replace=False)
                    piano.append((rng.choice(GATE_2Q), int(a), int(b)))
                else:
                    a, b, c = rng.choice(n_wires, 3, replace=False)
                    piano.append(('toffoli', int(a), int(b), int(c)))

            def circuit(piano=piano):
                for op in piano:
                    name = op[0]
                    if name == 'h': qml.Hadamard(wires=op[1])
                    elif name == 'x': qml.PauliX(wires=op[1])
                    elif name == 'y': qml.PauliY(wires=op[1])
                    elif name == 'z': qml.PauliZ(wires=op[1])
                    elif name == 's': qml.S(wires=op[1])
                    elif name == 't': qml.T(wires=op[1])
                    elif name == 'rx': qml.RX(op[2], wires=op[1])
                    elif name == 'ry': qml.RY(op[2], wires=op[1])
                    elif name == 'rz': qml.RZ(op[2], wires=op[1])
                    elif name == 'cnot': qml.CNOT(wires=[op[1], op[2]])
                    elif name == 'cz': qml.CZ(wires=[op[1], op[2]])
                    elif name == 'swap': qml.SWAP(wires=[op[1], op[2]])
                    elif name == 'toffoli': qml.Toffoli(wires=[op[1], op[2], op[3]])
                return qml.probs(wires=range(n_wires))

            qnode = qml.QNode(circuit, dev)
            ref = np.asarray(qnode())
            _, ours = run_pennylane_circuit(qnode, use_float32=False)
            assert np.allclose(ref, ours, atol=1e-6), f"trial {trial} mismatch, piano={piano}"

    def test_non_monotonic_wire_order_bare_tape(self):
        # Same fix, tape (not QNode) input path.
        import pennylane as qml
        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires=2)
            qml.CNOT(wires=[2, 1])

        circ = from_pennylane(tape)
        # wire 2 touched first, wire 1 second, but sorted-order export
        # means q[0]=wire1, q[1]=wire2 -> cx control is q[1], target is q[0]
        assert circ.n_qubits == 2
        assert [op['name'] for op in circ.ops] == ['x', 'cx']
        assert circ.ops[0]['qubits'] == [1]        # x on wire 2 -> q[1]
        assert circ.ops[1]['qubits'] == [1, 0]      # cx(wire2, wire1) -> q[1], q[0]
