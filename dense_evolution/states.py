"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.physics.states as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.states import ghz_state`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.physics.states directly in new code.
"""
from dense_evolution.physics.states import ghz_state

__all__ = ['ghz_state']
