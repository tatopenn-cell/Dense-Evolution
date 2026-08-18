"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.utils.drawing as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.drawing import draw_circuit`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.utils.drawing directly in new code.
"""

from dense_evolution.utils.drawing import draw_circuit

__all__ = ["draw_circuit"]
