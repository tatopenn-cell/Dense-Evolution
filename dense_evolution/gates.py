"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.gates as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.gates import GATES`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.gates directly in new code.
"""
from dense_evolution.circuits.gates import GATES, PARAMETRIC_GATES, GATE_IDS

__all__ = ['GATES', 'PARAMETRIC_GATES', 'GATE_IDS']
