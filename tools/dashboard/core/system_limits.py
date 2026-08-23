"""
Real, per-machine qubit limits -- detects actual available RAM instead of
a number picked to fit whatever machine this was developed on. A modest
8 GB dev laptop and a 128 GB workstation get genuinely different limits.

Reuses dense_evolution.chunk.get_dynamic_chunk, the same real,
already-tested RAM-based sizing function the Chunk engine itself uses
(confirmed in feature/streamlit-dashboard's older simulation_runner.py:
"Decided by de.Chunk's own dynamic, available-RAM-based sizing... not a
hardcoded qubit-count constant, so it adapts to the machine it's running
on") -- rather than a second, slightly different hand-rolled formula.
The actual safety enforcement (rejecting a request that would genuinely
overflow RAM) is separate: dashboard_core.engine.run_circuit_from_qasm
calls dense_evolution.chunk.SafeMemoryGuard.check_allocation directly,
which has no floor -- this module is only the UI's suggested/default cap.
"""

import dense_evolution as de

__all__ = ['max_safe_dense_qubits']


def max_safe_dense_qubits() -> dict:
    """Suggested max qubit count for the Composer's Qubits field, from
    dense_evolution.chunk.get_dynamic_chunk(complex128) -- floor 16,
    ceiling 27 by that function's own design (a chunk always has *some*
    minimum useful size, and 27 qubits/2 GB is its own practical ceiling
    for a single dense block). MPS's contract_to_statevector has a
    separate, RAM-independent hard ceiling of 24 qubits; picking 25-27
    with the MPS backend surfaces that function's own real error rather
    than being silently blocked here."""
    guard = de.chunk.SafeMemoryGuard()
    status = guard.status()
    max_qubits = de.chunk.get_dynamic_chunk(de.chunk.jnp.complex128 if de.chunk.HAS_JAX else de.chunk.np.complex128)

    return {
        "total_mb": status["total_mb"],
        "available_mb": status["available_mb"],
        "threshold_pct": guard.threshold_pct,
        "max_qubits_dense": max_qubits,
    }
