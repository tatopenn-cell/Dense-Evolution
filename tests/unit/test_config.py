"""
Regression tests for dense_evolution/config.py -- guards against the
exact critical bug flagged in an external code review (analisi e
consigli deepseek.txt): registry.py, compiler.py and statevector.py
used to call jax.config.update("jax_enable_x64", True) unconditionally
at MODULE IMPORT time, so merely `import dense_evolution` silently
overrode a precision a caller had already configured for unrelated JAX
code running earlier in the same process.

PR #130 introduced config.py's lazy ensure_x64() to fix this -- but a
second, more subtle instance of the same bug survived one level deeper:
dense_evolution/circuits/registry.py instantiated a module-level
`HARDWARE_REGISTRY = QuantumHardwareRegistry()` singleton (never
referenced anywhere else -- dead code), whose __init__ calls
ensure_x64() unconditionally. That eager construction ran at import
time regardless, so `import dense_evolution` still forced
jax_enable_x64=True even after the "lazy" refactor -- verified directly
here (this exact test, run against the code before the HARDWARE_REGISTRY
removal, failed: starting from jax_enable_x64=False, a bare import
turned it True). Fixed by deleting that dead singleton.

Subprocess isolation is required for these tests to mean anything: jax
and dense_evolution are both already imported by the time any test in
this suite runs (by pytest's own collection or by other test modules),
so re-importing dense_evolution in-process would hit sys.modules's
cache and never re-execute its module-level code at all. Each test here
spawns a fresh Python interpreter, sets a known precision, imports
dense_evolution, and checks nothing changed.
"""
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


class TestImportDoesNotTogglePrecision:

    def test_import_does_not_flip_true_to_false(self):
        result = _run(
            "import jax; jax.config.update('jax_enable_x64', True)\n"
            "import dense_evolution\n"
            "assert jax.config.jax_enable_x64 is True, "
            "f'import flipped jax_enable_x64 from True to {jax.config.jax_enable_x64}'\n"
        )
        assert result.returncode == 0, result.stderr

    def test_import_does_not_flip_false_to_true(self):
        """The regression this whole file exists for: with x64 already
        explicitly False, a bare `import dense_evolution` (no simulator,
        no registry, nothing constructed) must leave it False."""
        result = _run(
            "import jax; jax.config.update('jax_enable_x64', False)\n"
            "import dense_evolution\n"
            "assert jax.config.jax_enable_x64 is False, "
            "f'import flipped jax_enable_x64 from False to {jax.config.jax_enable_x64}'\n"
        )
        assert result.returncode == 0, result.stderr


class TestLazyEnsureX64:
    """Confirms the OTHER half of config.py's contract: precision isn't
    just left alone at import time -- it's still enabled lazily, on
    first real use, exactly as ensure_x64()'s docstring promises."""

    def test_constructing_a_simulator_enables_x64_by_default(self):
        result = _run(
            "import jax; jax.config.update('jax_enable_x64', False)\n"
            "import dense_evolution as de\n"
            "de.DenseSVSimulator(1)\n"
            "assert jax.config.jax_enable_x64 is True, "
            "'constructing DenseSVSimulator should lazily enable x64 by default'\n"
        )
        assert result.returncode == 0, result.stderr

    def test_explicit_set_precision_false_sticks_after_constructing_a_simulator(self):
        result = _run(
            "import jax\n"
            "import dense_evolution as de\n"
            "de.set_precision(False)\n"
            "de.DenseSVSimulator(1)\n"
            "assert jax.config.jax_enable_x64 is False, "
            "'set_precision(False) must not be overridden by a later ensure_x64() call'\n"
        )
        assert result.returncode == 0, result.stderr
