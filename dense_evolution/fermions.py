"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.physics.fermions as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.fermions import majorana_pauli_terms`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.physics.fermions directly in new code.
"""
from dense_evolution.physics.fermions import majorana_pauli_terms

__all__ = ['majorana_pauli_terms']
