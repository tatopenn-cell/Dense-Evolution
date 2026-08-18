"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.utils.measurement as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.measurement import ...`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.utils.measurement directly in new code.
"""

from dense_evolution.utils.measurement import sample_counts, statevector_fidelity

__all__ = ["sample_counts", "statevector_fidelity"]
