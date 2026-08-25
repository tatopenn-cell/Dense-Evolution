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

Removing that singleton exposed a THIRD, previously-masked bug:
dense_evolution/circuits/gates.py built its static GATES dict eagerly
at module import time via jax.numpy -- if jax_enable_x64 wasn't already
True at that exact moment, JAX silently truncated dtype=complex
(complex128) to complex64 right there, permanently (these are plain
constants, computed once, never rebuilt later even after ensure_x64()
turns x64 on for real simulation work). HARDWARE_REGISTRY's eager
ensure_x64() call happened to always win the race against gates.py's
own import, accidentally masking this the whole time. Verified
directly: removing HARDWARE_REGISTRY alone (without also fixing
gates.py) broke tests/unit/test_simulator.py's norm-preservation checks
(require <1e-12, failed at ~1.7e-8 -- exactly the scale of complex64
contamination in an otherwise-complex128 computation). Fixed by building
GATES with plain NumPy instead, which is never subject to JAX's
process-wide x64 flag.

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


class TestGatesSurviveImportTimePrecision:
    """The regression this class exists for: GATES is built once, at
    dense_evolution's own import time -- if it were built via jax.numpy
    (as it used to be), starting from jax_enable_x64=False would bake a
    permanent complex64 truncation into every gate matrix, silently,
    with no way to recover it later even after x64 gets enabled for real
    simulation work. Plain NumPy has no such process-wide flag, so GATES
    must stay genuinely complex128 regardless of the precision active at
    the moment dense_evolution is imported."""

    def test_gates_are_complex128_even_when_x64_starts_false(self):
        result = _run(
            "import jax; jax.config.update('jax_enable_x64', False)\n"
            "import numpy as np\n"
            "from dense_evolution.circuits.gates import GATES\n"
            "bad = {name: str(m.dtype) for name, m in GATES.items() if m.dtype != np.complex128}\n"
            "assert not bad, f'GATES entries not complex128 (x64 started False): {bad}'\n"
        )
        assert result.returncode == 0, result.stderr

    def test_gates_are_plain_numpy_not_jax_arrays(self):
        """GATES must be plain numpy arrays specifically -- a jax array
        would be re-subject to the process-wide x64 flag at creation
        time, reintroducing the exact bug this class guards against."""
        result = _run(
            "import numpy as np\n"
            "from dense_evolution.circuits.gates import GATES\n"
            "bad = {name: type(m).__name__ for name, m in GATES.items() if not isinstance(m, np.ndarray)}\n"
            "assert not bad, f'GATES entries not plain numpy arrays: {bad}'\n"
        )
        assert result.returncode == 0, result.stderr
