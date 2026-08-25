"""
Centralized JAX precision control for dense_evolution.

jax_enable_x64 is a process-wide JAX flag: once set, it affects every
JAX array in the process, not just dense_evolution's own. Before this
module existed, registry.py, compiler.py and statevector.py each called
jax.config.update("jax_enable_x64", True) unconditionally at MODULE
IMPORT time -- so merely `import dense_evolution`, even without ever
constructing a simulator, silently overrode a precision a caller had
already configured for unrelated JAX code running earlier in the same
process. mps.py and chunk.py never did this; their own docstrings
already documented the intended convention (see mps.py): dense_evolution
enables x64 lazily, only when something that actually needs complex128
precision is used, not as an import-time side effect.

This module is that single point. ensure_x64() is idempotent and safe
to call from every float64 entry point (DenseSVSimulator.__init__,
QuantumHardwareRegistry.__init__, circuit_to_energy_fn); set_precision()
is the public opt-out for a caller who wants to choose precision
explicitly, before constructing anything, and have that choice stick.
"""
import jax

_explicit = False


def set_precision(float64: bool = True) -> None:
    """
    Explicitly configure JAX's process-wide numeric precision.

    Call this yourself, before constructing anything, if you need
    control over exactly when/whether dense_evolution enables x64 --
    for example because another float32-only JAX library must be
    initialized first. Once called, dense_evolution's own lazy
    ensure_x64() (used internally by DenseSVSimulator, etc.) no longer
    forces float64 back on, so an explicit set_precision(False) sticks.
    """
    global _explicit
    jax.config.update("jax_enable_x64", float64)
    _explicit = True


def ensure_x64() -> None:
    """Enable jax_enable_x64 unless a caller has already explicitly
    configured precision via set_precision(). Called internally,
    lazily, by dense_evolution components that need complex128
    precision -- never at import time."""
    if not _explicit:
        jax.config.update("jax_enable_x64", True)
