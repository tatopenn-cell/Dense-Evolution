"""Physics subpackage: state preparation, observables, entropy, fermions, QEC."""
from .states import ghz_state
from .observables import (pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix,
                           pauli_sum_matvec, multiply_pauli_terms,
                           pauli_sum_matvec_jax, pauli_sum_expectation_jax, PauliSumOperator)
from .entropy import partial_trace, von_neumann_entropy, mutual_information, central_charge
from .fermions import majorana_pauli_terms, total_parity_operator, hubbard_hamiltonian_pauli_terms
from .qec import (pauli_commutes, compute_syndrome, erasure_aware_decode, pymatching_decode,
                   blind_minimum_weight_decode, nearest_coset_decode)

__all__ = [
    "ghz_state",
    "pauli_expectation", "pauli_sum_expectation", "pauli_hamiltonian_to_matrix", "pauli_sum_matvec",
    "multiply_pauli_terms",
    "pauli_sum_matvec_jax", "pauli_sum_expectation_jax", "PauliSumOperator",
    "partial_trace", "von_neumann_entropy", "mutual_information", "central_charge",
    "majorana_pauli_terms", "total_parity_operator", "hubbard_hamiltonian_pauli_terms",
    "pauli_commutes", "compute_syndrome", "erasure_aware_decode", "pymatching_decode", "blind_minimum_weight_decode",
    "nearest_coset_decode",
]
