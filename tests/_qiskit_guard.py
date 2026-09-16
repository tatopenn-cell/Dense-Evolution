"""Shared macOS/qiskit instability guard (prog.txt test-suite audit, issue
#269 point 4) -- centralizes the condition and skip reason that
test_interop.py and test_interop_calibration_noise.py each used to
duplicate by hand.

QISKIT_UNSTABLE_ON_DARWIN gates an `if/else` at MODULE level in each
call site, not a `@pytest.mark.skipif` on the class: `qiskit =
pytest.importorskip('qiskit')` as a class-body statement still runs at
collection time regardless of a skipif marker on the class, so a plain
skipif would still trigger the crash this guard exists to avoid. Only
never defining the real class at all on Darwin (the if/else) stops
qiskit from being imported into the process in the first place. See the
call sites for the full explanation of the underlying macOS segfault.
"""
import sys

QISKIT_UNSTABLE_ON_DARWIN = sys.platform == 'darwin'
QISKIT_DARWIN_SKIP_REASON = "qiskit destabilizes the process on macOS CI runners -- see tests/_qiskit_guard.py"
