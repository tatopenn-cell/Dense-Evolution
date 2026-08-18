"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.trotter as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.trotter import pauli_rotation_ops`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.trotter directly in new code.
"""
from dense_evolution.circuits.trotter import pauli_rotation_ops, trotter_evolve_ops

__all__ = ['pauli_rotation_ops', 'trotter_evolve_ops']
