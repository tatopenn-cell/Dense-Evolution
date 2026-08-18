"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.backends.mps as part of the Phase 2 subpackage split (see
prog.txt). Kept so `from dense_evolution.mps import MPSSimulator` (used
by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.backends.mps directly in new code.
"""
from dense_evolution.backends.mps import MPSSimulator

__all__ = ["MPSSimulator"]
