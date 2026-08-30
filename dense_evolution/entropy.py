"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.physics.entropy as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.entropy import partial_trace, von_neumann_entropy, mutual_information`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.physics.entropy directly in new code.
"""
from dense_evolution.physics.entropy import partial_trace, von_neumann_entropy, mutual_information, central_charge

__all__ = ['partial_trace', 'von_neumann_entropy', 'mutual_information', 'central_charge']
