"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.backends.statevector as part of the Phase 2 subpackage
split (see prog.txt). Kept so `from dense_evolution.simulator import
DenseSVSimulator` (used by external consumers, e.g. Dense-Evolution-
Discovery) keeps working unchanged. Import from
dense_evolution.backends.statevector directly in new code.
"""
from dense_evolution.backends.statevector import DenseSVSimulator

__all__ = ["DenseSVSimulator"]
