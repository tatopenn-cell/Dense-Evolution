"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.physics.qec as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.qec import pauli_commutes, compute_syndrome, erasure_aware_decode`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.physics.qec directly in new code.
"""
from dense_evolution.physics.qec import pauli_commutes, compute_syndrome, erasure_aware_decode

__all__ = ['pauli_commutes', 'compute_syndrome', 'erasure_aware_decode']
