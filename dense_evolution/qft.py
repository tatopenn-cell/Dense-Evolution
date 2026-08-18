"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.qft as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.qft import qft`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.qft directly in new code.
"""
from dense_evolution.circuits.qft import qft

__all__ = ['qft']
