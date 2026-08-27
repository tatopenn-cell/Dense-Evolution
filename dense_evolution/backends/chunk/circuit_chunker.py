from typing import List, Optional

from ._engine_imports import DenseSVSimulator, QuantumTranspiler
from .guard import SafeMemoryGuard

__all__ = ["CircuitChunker"]


class CircuitChunker:
    """
    Transpile a circuit once, then execute it in fixed-size gate-slices so
    XLA sees the same trace shape on every compilation.

    A SafeMemoryGuard is checked **before every slice** — if RAM drops below
    15% the current slice is aborted with MemoryPressureError before JAX
    attempts the allocation.

    Parameters
    ----------
    simulator_instance : DenseSVSimulator
        Physical simulator (sized to safe_qubits, not logical n_qubits).
    memory_threshold   : float
        Passed to SafeMemoryGuard.  Default 0.15 (15%).
    """

    def __init__(
        self,
        simulator_instance: Optional[DenseSVSimulator] = None,
        memory_threshold: float = 0.15,
    ):
        self.sim   = simulator_instance
        self._guard = SafeMemoryGuard(threshold_pct=memory_threshold)

    def split_circuit(self, circuit: List, chunk_size: int = 500) -> None:
        """
        Execute *circuit* in slices of *chunk_size* gates.

        Raises
        ------
        RuntimeError        if no simulator instance is attached.
        MemoryPressureError if RAM drops below threshold before a slice.
        """
        if self.sim is None:
            raise RuntimeError(
                "CircuitChunker: no simulator instance attached. "
                "Pass simulator_instance= at construction or assign .sim."
            )

        target: List = QuantumTranspiler.transpile(circuit)
        n_slices     = (len(target) + chunk_size - 1) // chunk_size

        for idx, i in enumerate(range(0, len(target), chunk_size)):
            # ── Anti-OOM check before every slice ──────────────────────────────────────────────────────────────
            self._guard.check(f"slice {idx + 1}/{n_slices}")
            self.sim.run_circuit_jit(target[i : i + chunk_size])
