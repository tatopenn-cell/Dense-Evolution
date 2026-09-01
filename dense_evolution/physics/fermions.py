"""
Majorana-fermion -> qubit (Jordan-Wigner) mapping.

Standard convention, one qubit per two Majorana modes:
    chi_{2j-1} = (prod_{k<j} Z_k) X_j
    chi_{2j}   = (prod_{k<j} Z_k) Y_j
mode_index is 1-indexed (chi_1 .. chi_{2*n_qubits}). Each chi_i is
Hermitian and satisfies chi_i^2 = I by this normalization; the anticommutation
relation {chi_a, chi_b} = 2*delta_ab*I holds exactly (verified in
tests/unit/test_fermions.py against the actual matrices, not assumed from the
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

total_parity_operator (the "Klein factor" for a set of Majorana modes) was
promoted alongside majorana_pauli_terms from Dense-Evolution-Discovery's
wormhole_magic_entropy.py (2026-08-29): a second, independently-Jordan-
Wigner-mapped fermionic register (e.g. the "R" side of a two-copy/thermofield-
double construction) puts its Majoranas on disjoint qubits, so cross-register
Majorana products COMMUTE by construction instead of anticommuting -- the
well-known "Klein factor" problem from bosonization / fermionic-entanglement
literature (e.g. Fidkowski-Kitaev). Multiplying one register's operators by
its own total_parity_operator before combining them with the other
register's restores the correct anticommutation; see that function's
docstring for the algebraic proof.

hubbard_hamiltonian_pauli_terms uses the OTHER standard Jordan-Wigner
convention -- ordinary spin-orbital creation/annihilation operators
(c_q = sigma+_q * Z-string, not Majoranas) -- promoted from
Dense-Evolution-Discovery's hubbard_square_arovas.py (2026-09-01), which
reproduces Arovas, Bandyopadhyay & Zhu, "The Hubbard Model" (Annual Review
of Condensed Matter Physics 2022, arXiv:2103.12097). Nothing like it
existed anywhere in this package before: dashboard_core/hamiltonians.py
only has PennyLane's molecule-specific Hartree-Fock Jordan-Wigner (routed
through PennyLane's own internal mapping, not this module's Pauli-term
machinery), not a general lattice-fermion-model builder.
"""
from .observables import multiply_pauli_terms

__all__ = ['majorana_pauli_terms', 'total_parity_operator', 'hubbard_hamiltonian_pauli_terms']


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


def total_parity_operator(mode_indices, n_qubits):
    """Total fermion-parity ("Klein factor") operator for a set of Majorana
    modes: i^(N/2) times the ORDERED product of majorana_pauli_terms(m,
    n_qubits) for every m in `mode_indices` (N = len(mode_indices)),
    computed via multiply_pauli_terms (exact symbolic Pauli algebra, no
    numerical approximation).

    The i^(N/2) PHASE CORRECTION is not optional decoration -- found
    necessary by testing, not assumed from the general anticommutation
    argument alone: for N mutually anticommuting HERMITIAN operators, the
    raw ordered product Pi = chi_1*chi_2*...*chi_N satisfies
    Pi^dagger = chi_N*...*chi_1 = (-1)**(N*(N-1)/2) * Pi (reversing N
    anticommuting factors takes N*(N-1)/2 transpositions, each contributing
    -1) -- so Pi itself is Hermitian only when N*(N-1)/2 is even (N=4, 8,
    12, ... i.e. N%4==0), and ANTI-Hermitian when N*(N-1)/2 is odd (N=2, 6,
    10, ... i.e. N%4==2). This was caught by a test at N=6 (n_qubits=3)
    that the original N=8-only manual check never exercised. Multiplying
    by i^(N/2) fixes this for every even N: verified numerically for
    N=2,4,6,8,10 that i**(N/2) * Pi is exactly Hermitian AND squares to
    exactly the identity in every case (see tests/unit/test_fermions.py).

    Why this fixes cross-register anticommutation: an even number of
    mutually anticommuting, squares-to-identity operators, correctly
    phase-normalized to be Hermitian and square to I (as above), still
    anticommutes with each individual factor -- multiplying an overall
    scalar phase never changes an operator's (anti)commutation relations
    with OTHER operators. Concretely: if psi_L (from THIS register) and
    psi_R (from an independently Jordan-Wigner-mapped SECOND register,
    e.g. dense_evolution.physics.fermions.majorana_pauli_terms called
    again with its own qubit offset) act on disjoint qubits, they commute
    by construction -- but P_L = total_parity_operator(all_of_this_
    register's_modes, n_qubits) anticommutes with every psi_L, so
    replacing psi_R with multiply_pauli_terms([P_L, psi_R]) (P_L applied
    first) makes it anticommute with psi_L instead, while leaving
    {psi_R, psi_R'} (two operators from the SAME second register)
    unchanged, since P_L^2 = I factors out trivially:
    {P_L*psi_R, P_L*psi_R'} = P_L^2 * {psi_R, psi_R'}.

    len(mode_indices) must be even -- an odd-length parity operator would
    anticommute with an EVEN number of factors' worth of sign flips, i.e.
    NOT anticommute with its own members, defeating the purpose (raises
    ValueError rather than silently returning something that doesn't have
    the intended algebraic property).

    Parameters
    ----------
    mode_indices : iterable of int
        1-indexed Majorana modes (same indexing as majorana_pauli_terms),
        typically ALL of one register's modes (range(1, n_majorana+1)) to
        get that register's total parity, though any even-length subset
        is algebraically valid.
    n_qubits : int
        Same meaning as majorana_pauli_terms's n_qubits.

    Returns
    -------
    (complex, dict)
        A (coeff, pauli_dict) term, same shape as majorana_pauli_terms's
        return -- ready for pauli_hamiltonian_to_matrix / pauli_expectation
        / another multiply_pauli_terms call.

    Examples
    --------
    >>> total_parity_operator([1, 2], n_qubits=1)  # i^1 * chi_1*chi_2 = i*(i*Z) = -Z
    ((-1+0j), {0: 'Z'})
    """
    mode_indices = list(mode_indices)
    n = len(mode_indices)
    if n % 2 != 0:
        raise ValueError(
            f"total_parity_operator needs an EVEN number of modes to anticommute "
            f"with each of its own members, got {n}"
        )
    factors = [majorana_pauli_terms(m, n_qubits) for m in mode_indices]
    coeff, pauli_dict = multiply_pauli_terms(factors)
    return coeff * (1j ** (n / 2)), pauli_dict


def hubbard_hamiltonian_pauli_terms(n_sites, t, U, periodic=True):
    """Jordan-Wigner mapping of the 1D Hubbard-ring Hamiltonian
    H = -t * sum_<ij>,sigma (c^dagger_i,sigma c_j,sigma + h.c.)
        + U * sum_i n_i,up * n_i,down
    onto n_qubits=2*n_sites qubits: qubits [0, n_sites) are the spin-up
    orbitals, qubits [n_sites, 2*n_sites) spin-down, both site-ordered.
    <ij> runs over nearest-neighbor sites on a 1D ring (site i to site
    (i+1) % n_sites); pass periodic=False to drop the wraparound bond
    (site n_sites-1 to site 0) and get an open chain instead.

    The wraparound bond needed a self-test before being trusted: some
    Jordan-Wigner conventions need an extra fermion-parity correction for
    a periodic-boundary term written as a *short* Pauli string. This
    function instead always uses the full-length Jordan-Wigner string
    between the two mapped qubit indices (c_i^dagger c_j = sigma+_i *
    (Z-string between i and j) * sigma-_j, i<j), which is the exact
    fermionic identity for ANY pair of modes regardless of whether they
    are lattice-adjacent -- so no extra correction is needed here, and
    this was verified directly against an independent brute-force
    fermionic operator construction (not just argued from the formula):
    max diff 0.00e+00 (machine-exact) at n_sites=2,3,4, both periodic and
    open (Dense-Evolution-Discovery's hubbard_square_arovas.py).

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (n_qubits = 2*n_sites).
    t : float
        Hopping amplitude.
    U : float
        On-site interaction strength.
    periodic : bool, default True
        Include the wraparound bond (site n_sites-1 to site 0). With
        n_sites=4, this is the "Hubbard square" studied in Arovas,
        Bandyopadhyay & Zhu, "The Hubbard Model" (Annual Review of
        Condensed Matter Physics 2022, arXiv:2103.12097) -- Table 2 (p.6)
        gives a closed-form small-U/t perturbative ground-state energy
        for this exact model, verified directly against exact
        diagonalization in the Discovery experiment above, and identifies
        this ground state's orbital symmetry as x^2-y^2 (i.e. B1g/d-wave),
        checkable via the sign pattern of pairing correlations
        <Delta_0^dagger Delta_j> with Delta_i = c_i,up * c_i,down
        (positive for axis neighbors, negative for the diagonal).

    Returns
    -------
    list of (float, dict)
        Pauli terms in the same (coeff, {qubit: 'X'|'Y'|'Z'}) form
        majorana_pauli_terms returns -- ready for
        dense_evolution.pauli_hamiltonian_to_matrix or
        pauli_sum_expectation.
    """
    n_qubits = 2 * n_sites
    terms = []

    edges = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    if not periodic:
        edges = [(i, j) for i, j in edges if not (i == n_sites - 1 and j == 0)]

    for i, j in edges:
        for offset in (0, n_sites):
            qi, qj = offset + i, offset + j
            low, high = min(qi, qj), max(qi, qj)
            z_span = range(low + 1, high)

            pauli_xx = {k: 'Z' for k in z_span}
            pauli_xx[low] = 'X'
            pauli_xx[high] = 'X'
            terms.append((-0.5 * t, pauli_xx))

            pauli_yy = {k: 'Z' for k in z_span}
            pauli_yy[low] = 'Y'
            pauli_yy[high] = 'Y'
            terms.append((-0.5 * t, pauli_yy))

    for i in range(n_sites):
        idx_up, idx_dn = i, n_sites + i
        terms.append((0.25 * U, {}))
        terms.append((-0.25 * U, {idx_up: 'Z'}))
        terms.append((-0.25 * U, {idx_dn: 'Z'}))
        terms.append((0.25 * U, {idx_up: 'Z', idx_dn: 'Z'}))

    return terms
