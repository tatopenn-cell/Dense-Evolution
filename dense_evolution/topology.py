"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.topology as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.topology import entangling_layer`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.topology directly in new code.
"""
from dense_evolution.circuits.topology import entangling_layer, VALID_PATTERNS

__all__ = ['entangling_layer', 'VALID_PATTERNS']
