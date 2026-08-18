"""Utility subpackage: circuit drawing and measurement/sampling helpers.

Part of the Phase 2 subpackage split (see prog.txt) -- `utils/` is the
first subpackage moved, chosen because it has no internal dense_evolution
dependents (leaf module), matching dynamiqs's own `core/` vs `apis/`
split philosophy applied here as "role" grouping rather than "physics
domain" grouping.
"""

from .drawing import draw_circuit
from .measurement import sample_counts, statevector_fidelity

__all__ = ["draw_circuit", "sample_counts", "statevector_fidelity"]
