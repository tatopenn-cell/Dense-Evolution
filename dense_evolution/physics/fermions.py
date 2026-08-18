"""
Majorana-fermion -> qubit (Jordan-Wigner) mapping.

Standard convention, one qubit per two Majorana modes:
    chi_{2j-1} = (prod_{k<j} Z_k) X_j
    chi_{2j}   = (prod_{k<j} Z_k) Y_j
mode_index is 1-indexed (chi_1 .. chi_{2*n_qubits}). Each chi_i is
Hermitian and satisfies chi_i^2 = I by this normalization; the anticommutation
relation {chi_a, chi_b} = 2*delta_ab*I holds exactly (verified in
tests/test_fermions.py against the actual matrices, not assumed from the
textbook formula alone).

Originated in research/wormhole_syk.py (a traversable-wormhole-inspired
quantum teleportation reproduction, arXiv:2604.10090) -- promoted here
because Jordan-Wigner fermion mapping is a generic building block, not
specific to that one experiment, and nothing like it existed anywhere in
this package before (dashboard_core/hamiltonians.py only has PennyLane's
molecule-specific Hartree-Fock Jordan-Wigner, not a general Majorana map).

Combine the returned Pauli term with dense_evolution.pauli_hamiltonian_to_matrix
to build any Majorana-operator Hamiltonian as a dense matrix, e.g. a
sparse SYK model: H = sum_{ijkl} J_ijkl * chi_i*chi_j*chi_k*chi_l.
"""

__all__ = ['majorana_pauli_terms']


def majorana_pauli_terms(mode_index, n_qubits):
    """Jordan-Wigner term for one Majorana mode.

    Parameters
    ----------
    mode_index : int
        1-indexed Majorana mode, 1 <= mode_index <= 2*n_qubits.
    n_qubits : int
        Number of qubits the fermionic system is mapped onto
        (n_majorana_modes = 2*n_qubits).

    Returns
    -------
    (float, dict)
        A (coeff, pauli_dict) term -- coeff is always 1.0, pauli_dict is
        {qubit: 'X'|'Y'|'Z'} -- ready for
        dense_evolution.pauli_hamiltonian_to_matrix.
    """
    if not (1 <= mode_index <= 2 * n_qubits):
        raise ValueError(f"mode_index must be in [1, {2*n_qubits}], got {mode_index}")
    j = (mode_index - 1) // 2
    is_even = (mode_index % 2 == 0)
    pauli = {k: 'Z' for k in range(j)}
    pauli[j] = 'Y' if is_even else 'X'
    return (1.0, pauli)
