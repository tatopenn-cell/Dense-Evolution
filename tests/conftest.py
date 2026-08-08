"""
Shared fixtures for the whole test suite, auto-loaded by pytest for every
file in this directory -- extracted from the original monolithic
test_dense_evolution.py when it was split by module (test_simulator.py,
test_registry.py, test_compiler.py, test_parser.py, test_chunk.py,
test_healing.py, test_integration.py).
"""
import os
import sys

import pytest

from dense_evolution import DenseSVSimulator


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """
    On macOS, force an immediate process exit instead of letting normal
    Python interpreter shutdown run.

    Reproduced on CI (macos-latest, Python 3.10, arm64): every test
    passes, coverage.xml is written, pytest prints its own "N passed"
    summary -- then, a few seconds later, the process dies with SIGSEGV
    (exit 139) during interpreter finalization, after pytest's own work
    is already done. Bisected by first ruling out Qiskit (an earlier fix
    removed it from the process on macOS entirely; the crash persisted
    unchanged), which leaves native-extension teardown -- most likely
    JAX/XLA's runtime shutdown, a known category of issue on some
    platforms -- as the remaining explanation. `trylast=True` ensures
    this runs after pytest-cov's own pytest_sessionfinish (which writes
    coverage.xml), so nothing meaningful is skipped; os._exit() bypasses
    Python's normal atexit/finalization machinery entirely, which is
    exactly the phase that was crashing.
    """
    if sys.platform == 'darwin':
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exitstatus)

# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sim2():
    """Fresh 2-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)

@pytest.fixture
def sim3():
    """Fresh 3-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)

@pytest.fixture
def sim4():
    """Fresh 4-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=4, use_gpu=False, use_float32=False)
