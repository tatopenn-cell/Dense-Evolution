"""
Shared fixtures for the whole test suite, auto-loaded by pytest for every
file in this directory -- extracted from the original monolithic
test_dense_evolution.py when it was split by module (test_simulator.py,
test_registry.py, test_compiler.py, test_parser.py, test_chunk.py,
test_healing.py, test_integration.py).
"""
import os
import pathlib
import sys

import pytest

import dense_evolution
from dense_evolution import DenseSVSimulator

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
assert _REPO_ROOT in pathlib.Path(dense_evolution.__file__).resolve().parents, (
    f"dense_evolution imported from {dense_evolution.__file__}, not from this repo "
    f"checkout ({_REPO_ROOT}) -- a stale installed copy (e.g. a leftover 'pip install .' "
    f"or a twine build) is shadowing the local source; every test below would silently "
    f"run against the wrong code."
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_memory_probe: opt out of the autouse deterministic memory-budget "
        "fixture, for tests that exercise _device_memory_budget_bytes/"
        "_device_total_bytes themselves rather than code that merely calls them.",
    )


@pytest.fixture(autouse=True)
def _deterministic_memory_budget(request, monkeypatch):
    """SafeMemoryGuard reads the ACTIVE compute device's real memory (or
    host RAM via psutil on CPU) -- see guard.py's own module docstring.
    That makes any test that builds a Chunk/SafeMemoryGuard without
    explicitly faking the probe pass or fail depending on what else is
    using RAM on the machine at that moment (confirmed directly: a full
    `pytest tests/unit/test_mps.py` run raised a real, non-deterministic
    MemoryPressureError from cumulative RAM use partway through the
    file). Fixed at a comfortable 50% free of 8 GB by default here so the
    guard's pass/fail branch is a property of the code, not the host --
    tests that specifically exercise SafeMemoryGuard's own thresholds
    override this locally with their own monkeypatch of the same two
    functions. Tests marked real_memory_probe (those exercising the probe
    functions' own device-detection logic) opt out entirely."""
    if request.node.get_closest_marker("real_memory_probe") is not None:
        return

    from dense_evolution.backends.chunk import guard as guard_mod

    total_bytes = 8 * 1024 ** 3
    available_bytes = 0.5 * total_bytes
    monkeypatch.setattr(guard_mod, "_device_memory_budget_bytes",
                         lambda device=None: (available_bytes, "test-fixed"))
    monkeypatch.setattr(guard_mod, "_device_total_bytes",
                         lambda device=None: total_bytes)


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
    return DenseSVSimulator(n_qubits=2, use_float32=False)

@pytest.fixture
def sim3():
    """Fresh 3-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=3, use_float32=False)

@pytest.fixture
def sim4():
    """Fresh 4-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=4, use_float32=False)
