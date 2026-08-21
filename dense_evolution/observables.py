"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.physics.observables as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.observables import pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.physics.observables directly in new code.
"""
from dense_evolution.physics.observables import (
    pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix, pauli_sum_matvec,
)

__all__ = ['pauli_expectation', 'pauli_sum_expectation', 'pauli_hamiltonian_to_matrix', 'pauli_sum_matvec']
