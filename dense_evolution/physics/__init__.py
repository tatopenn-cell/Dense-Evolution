"""Physics subpackage: state preparation, observables, entropy, fermions, QEC."""
from .states import ghz_state
from .observables import pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix, pauli_sum_matvec
from .entropy import partial_trace, von_neumann_entropy, mutual_information
from .fermions import majorana_pauli_terms
from .qec import pauli_commutes, compute_syndrome, erasure_aware_decode

__all__ = [
    "ghz_state",
    "pauli_expectation", "pauli_sum_expectation", "pauli_hamiltonian_to_matrix", "pauli_sum_matvec",
    "partial_trace", "von_neumann_entropy", "mutual_information",
    "majorana_pauli_terms",
    "pauli_commutes", "compute_syndrome", "erasure_aware_decode",
]
