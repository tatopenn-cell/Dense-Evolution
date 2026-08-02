"""
Tests for dashboard_core.system_limits.max_safe_dense_qubits -- the real,
per-machine RAM-aware qubit cap the Composer's UI uses instead of a fixed
number (dense_evolution.chunk.SafeMemoryGuard + get_dynamic_chunk under
the hood, both real psutil-backed).
"""

from dashboard_core.system_limits import max_safe_dense_qubits


def test_returns_real_structure_with_sane_values():
    limits = max_safe_dense_qubits()
    assert set(limits.keys()) == {"total_mb", "available_mb", "threshold_pct", "max_qubits_dense"}
    assert limits["total_mb"] > 0
    assert 0 < limits["available_mb"] <= limits["total_mb"]
    assert 0 < limits["threshold_pct"] < 1
    # get_dynamic_chunk's own documented clamp (dashboard_core/system_limits.py).
    assert 16 <= limits["max_qubits_dense"] <= 27


def test_max_qubits_is_an_int():
    limits = max_safe_dense_qubits()
    assert isinstance(limits["max_qubits_dense"], int)


def test_repeated_calls_are_consistent_within_a_tight_window():
    # Real system RAM can shift between calls, but not enough to move the
    # qubit cap on two back-to-back calls in a test process.
    a = max_safe_dense_qubits()
    b = max_safe_dense_qubits()
    assert a["max_qubits_dense"] == b["max_qubits_dense"]
