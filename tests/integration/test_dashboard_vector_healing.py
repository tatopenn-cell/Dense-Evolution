"""
Tests for dashboard_core.vector_healing -- the run_vector_healing
wrapper around ia_utils.vector_healing.enhanced_dense_healing_hybrid.
Covers the real happy path plus the deferred-ImportError guard added
around that import (previously unconditional, so a failure there would
have taken down every unrelated dashboard_core feature at import time).
"""
import importlib
import sys

import numpy as np
import pytest

import dashboard_core.vector_healing as vector_healing
from dashboard_core.vector_healing import run_vector_healing, VectorHealingResult


def test_run_vector_healing_returns_real_result_same_shape():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 4))
    result = run_vector_healing(vectors)
    assert isinstance(result, VectorHealingResult)
    assert len(result.healed_vectors) == 20
    assert len(result.healed_vectors[0]) == 4


def test_run_vector_healing_rejects_non_2d_input():
    with pytest.raises(ValueError):
        run_vector_healing(np.zeros(5))


def test_run_vector_healing_raises_clear_error_if_ia_utils_missing(monkeypatch):
    # BUG FIX: `from ia_utils.vector_healing import ...` used to be
    # unconditional -- if it ever failed (missing/stripped install, or
    # a transitive dependency gap), the whole dashboard_core.vector_healing
    # module failed to import, taking down unrelated features with it.
    # Now the import is deferred and only raises (with a clear message)
    # when run_vector_healing is actually called.
    monkeypatch.setattr(vector_healing, "enhanced_dense_healing_hybrid", None)
    monkeypatch.setattr(vector_healing, "_IMPORT_ERROR", ImportError("simulated missing ia_utils"))
    with pytest.raises(ImportError, match="run_vector_healing requires ia_utils.vector_healing"):
        run_vector_healing(np.zeros((5, 2)))


def test_module_import_guard_actually_triggers_on_a_real_import_failure(monkeypatch):
    # The test above only simulates the *consequence* of a failed import
    # (patching the resulting module attributes after a successful real
    # import) -- this one forces the actual `except ImportError` branch
    # at module-load time, via the standard sys.modules=None trick
    # (Python's import system treats that as "this import must fail").
    #
    # prog.txt test-suite audit (issue #269 point 6): the sys.modules
    # entry is now set via monkeypatch.setitem instead of manual
    # get/pop/set bookkeeping -- but the *timing* of the restore still
    # has to be manual, not left to monkeypatch's own fixture teardown:
    # reloading dashboard_core.vector_healing while the import is broken
    # is itself a lasting side effect (the ALREADY-IMPORTED module's own
    # enhanced_dense_healing_hybrid/_IMPORT_ERROR attributes get set to
    # the broken state), and the reload-back below needs sys.modules
    # already restored to the REAL module to succeed -- if it ran first
    # and monkeypatch's automatic teardown ran after, the reload-back
    # would still see the broken entry and fail. monkeypatch.undo() runs
    # the restore immediately, in the right order relative to the
    # reload-back, instead of waiting for fixture teardown.
    monkeypatch.setitem(sys.modules, "ia_utils.vector_healing", None)
    try:
        importlib.reload(vector_healing)
        assert vector_healing.enhanced_dense_healing_hybrid is None
        assert vector_healing._IMPORT_ERROR is not None
        with pytest.raises(ImportError, match="run_vector_healing requires ia_utils.vector_healing"):
            vector_healing.run_vector_healing(np.zeros((5, 2)))
    finally:
        monkeypatch.undo()
        importlib.reload(vector_healing)
        assert vector_healing.enhanced_dense_healing_hybrid is not None
