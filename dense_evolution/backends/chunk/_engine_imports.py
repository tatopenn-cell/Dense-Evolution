import numpy as np

# ── Flexible import with stub fallback ─────────────────────────────────────────
# Both except-branches below are for deployment layouts other than the
# installed package this test suite always runs against (a flat/Colab
# notebook cell with no `dense_evolution.` prefix, or an incomplete
# install missing simulator/compiler/gates entirely) -- genuinely
# unreachable here without artificially breaking sys.path, so excluded
# from coverage rather than chased with an artificial test.
try:
    from simulator import DenseSVSimulator
    from compiler import QuantumTranspiler  # pragma: no cover
    from circuits.compiler import _gate_1q_matrix  # pragma: no cover
    from gates import GATE_IDS  # pragma: no cover
except ModuleNotFoundError:
    try:
        from dense_evolution.simulator import DenseSVSimulator
        from dense_evolution.compiler import QuantumTranspiler
        from dense_evolution.circuits.compiler import _gate_1q_matrix
        from dense_evolution.gates import GATE_IDS
    except ModuleNotFoundError:  # pragma: no cover
        class DenseSVSimulator:  # type: ignore[no-redef]
            def __init__(self, n_qubits, **kwargs):
                self.n     = n_qubits
                self.dim   = 2 ** n_qubits
                self.dtype = np.complex128
                self.sv    = np.zeros(self.dim, dtype=self.dtype)
                self.sv[0] = 1.0
            def run_circuit_jit(self, circuit_slice): pass
            def memory_mb(self) -> float:
                return (self.dim * np.dtype(self.dtype).itemsize) / 1_000_000

            def _unavailable(self, name):
                raise NotImplementedError(
                    f"DenseSVSimulator stub: {name}() is not implemented here -- this "
                    "class only exists because the real dense_evolution/JAX modules "
                    "were not importable in this deployment (see this module's own "
                    "comment above); it is not a working simulator."
                )

            def get_probabilities(self): self._unavailable("get_probabilities")
            def get_statevector(self): self._unavailable("get_statevector")
            def apply_gate_1q(self, gate, q1): self._unavailable("apply_gate_1q")
            def apply_gate_2q(self, gate, q1, q2): self._unavailable("apply_gate_2q")

        class QuantumTranspiler:  # type: ignore[no-redef]
            @staticmethod
            def transpile(circuit): return circuit

        def _gate_1q_matrix(g_id, param, dtype):  # type: ignore[no-redef]
            raise NotImplementedError(
                "_gate_1q_matrix stub: real dense_evolution/JAX not importable in this deployment"
            )

        GATE_IDS: dict = {}

__all__ = ["DenseSVSimulator", "QuantumTranspiler", "GATE_IDS", "_gate_1q_matrix"]
