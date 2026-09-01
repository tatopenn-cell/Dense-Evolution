"""
Unit tests for dense_evolution/fermions.py -- Majorana-fermion -> qubit
(Jordan-Wigner) mapping. Cross-checked against the actual dense Pauli
matrices via pauli_hamiltonian_to_matrix, not just the textbook formula.
"""
import numpy as np
import pytest

from dense_evolution import majorana_pauli_terms, hubbard_hamiltonian_pauli_terms
from dense_evolution.observables import pauli_hamiltonian_to_matrix
from dense_evolution.physics.fermions import total_parity_operator


def test_backward_compat_shim_fermions_reexports_majorana_pauli_terms():
    # dense_evolution.fermions is the Phase 2 backward-compat shim left at
    # the old top-level path -- nothing in this suite imports through it
    # directly (everything sources majorana_pauli_terms from the top-level
    # dense_evolution package instead, which now gets it from
    # dense_evolution.physics.fermions), so without this the shim's own
    # lines go uncovered and a broken shim would go undetected by CI.
    from dense_evolution.fermions import majorana_pauli_terms as shim_mpt
    assert shim_mpt is majorana_pauli_terms


def _chi_matrix(mode_index, n_qubits):
    coeff, pauli_dict = majorana_pauli_terms(mode_index, n_qubits)
    return coeff * pauli_hamiltonian_to_matrix([(1.0, pauli_dict)], n_qubits)


class TestMajoranaPauliTerms:

    def test_anticommutation_relation_exact(self):
        """{chi_a, chi_b} = 2*delta_ab*I for every pair, N=6 -> 3 qubits."""
        n_qubits = 3
        n_majorana = 2 * n_qubits
        chis = [_chi_matrix(m, n_qubits) for m in range(1, n_majorana + 1)]
        identity = np.eye(2 ** n_qubits, dtype=complex)
        max_error = 0.0
        for a in range(n_majorana):
            for b in range(n_majorana):
                anticomm = chis[a] @ chis[b] + chis[b] @ chis[a]
                expected = 2 * identity if a == b else np.zeros_like(identity)
                max_error = max(max_error, float(np.max(np.abs(anticomm - expected))))
        assert max_error == pytest.approx(0.0, abs=1e-10)

    def test_each_mode_is_hermitian(self):
        n_qubits = 3
        for m in range(1, 2 * n_qubits + 1):
            chi = _chi_matrix(m, n_qubits)
            assert np.max(np.abs(chi - chi.conj().T)) == pytest.approx(0.0, abs=1e-12)

    def test_each_mode_squares_to_identity(self):
        n_qubits = 3
        identity = np.eye(2 ** n_qubits, dtype=complex)
        for m in range(1, 2 * n_qubits + 1):
            chi = _chi_matrix(m, n_qubits)
            assert np.max(np.abs(chi @ chi - identity)) == pytest.approx(0.0, abs=1e-12)

    def test_odd_mode_is_x_on_its_qubit(self):
        # chi_1 = X_0 (j=0, is_even=False)
        coeff, pauli_dict = majorana_pauli_terms(1, n_qubits=3)
        assert coeff == 1.0
        assert pauli_dict == {0: 'X'}

    def test_even_mode_is_y_on_its_qubit_with_z_string(self):
        # chi_4: mode_index=4 -> j=(4-1)//2=1, is_even=True -> Z_0 Y_1
        coeff, pauli_dict = majorana_pauli_terms(4, n_qubits=3)
        assert coeff == 1.0
        assert pauli_dict == {0: 'Z', 1: 'Y'}

    def test_out_of_range_mode_index_raises(self):
        with pytest.raises(ValueError):
            majorana_pauli_terms(0, n_qubits=3)
        with pytest.raises(ValueError):
            majorana_pauli_terms(7, n_qubits=3)

    def test_boundary_mode_indices_are_valid(self):
        n_qubits = 3
        majorana_pauli_terms(1, n_qubits)
        majorana_pauli_terms(2 * n_qubits, n_qubits)


class TestTotalParityOperator:
    """The 'Klein factor' for a set of Majorana modes -- see the
    algebraic proof in the function's own docstring; these tests verify
    it against the actual dense matrices, not just re-derive the proof."""

    def _parity_matrix(self, mode_indices, n_qubits):
        coeff, pauli_dict = total_parity_operator(mode_indices, n_qubits)
        return coeff * pauli_hamiltonian_to_matrix([(1.0, pauli_dict)], n_qubits)

    def test_odd_length_raises(self):
        with pytest.raises(ValueError):
            total_parity_operator([1, 2, 3], n_qubits=2)

    def test_anticommutes_with_every_member_mode(self):
        """The core claim: P = total_parity_operator(all modes) anticommutes
        with each INDIVIDUAL chi_m for m in those modes -- this is what
        makes it usable as a Klein factor to fix cross-register commutation
        (see dense_evolution.physics.fermions module docstring)."""
        n_qubits = 3
        n_majorana = 2 * n_qubits
        P = self._parity_matrix(list(range(1, n_majorana + 1)), n_qubits)
        for m in range(1, n_majorana + 1):
            chi = _chi_matrix(m, n_qubits)
            anticomm = P @ chi + chi @ P
            assert np.max(np.abs(anticomm)) == pytest.approx(0.0, abs=1e-10), f"mode {m} did not anticommute"

    def test_matches_ordered_matrix_product_times_phase_correction(self):
        """total_parity_operator's symbolic algebra must agree with
        i**(N/2) times literally multiplying the individual chi matrices
        in order -- the i**(N/2) phase correction is what makes the
        result Hermitian for every even N, not just N%4==0 (see the
        function's own docstring for why the raw product alone isn't
        enough)."""
        n_qubits = 3
        modes = [1, 2, 3, 4]
        P = self._parity_matrix(modes, n_qubits)
        manual = np.eye(2 ** n_qubits, dtype=complex)
        for m in modes:
            manual = manual @ _chi_matrix(m, n_qubits)
        manual *= 1j ** (len(modes) / 2)
        assert np.max(np.abs(P - manual)) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
    def test_hermitian_and_squares_to_identity_for_every_even_n(self, n_qubits):
        """Regression test for the N%4==2 case (N=2, 6, 10, ...), where the
        UNcorrected raw product is anti-Hermitian, not Hermitian -- caught
        by this exact parametrization at n_qubits=3 (N=6) during
        development; n_qubits=1 (N=2) and n_qubits=5 (N=10) are also in
        the same N%4==2 family and are checked here too, alongside the
        N%4==0 cases (n_qubits=2,4) which happened to already work even
        before the phase correction was added."""
        n_majorana = 2 * n_qubits
        P = self._parity_matrix(list(range(1, n_majorana + 1)), n_qubits)
        identity = np.eye(2 ** n_qubits, dtype=complex)
        assert np.max(np.abs(P - P.conj().T)) == pytest.approx(0.0, abs=1e-10)
        assert np.max(np.abs(P @ P - identity)) == pytest.approx(0.0, abs=1e-10)

    def test_fixes_cross_register_commutation_end_to_end(self):
        """The actual motivating use case: two independently-Jordan-Wigner-
        mapped registers (L, R), tensored together (disjoint qubits) --
        bare chi_L and chi_R COMMUTE by construction; dressing chi_R with
        L's total parity operator makes it ANTIcommute with chi_L instead."""
        n_qubits_per_side = 2
        n_majorana_per_side = 2 * n_qubits_per_side
        n_full = 2 * n_qubits_per_side

        def embed(mode_index, offset):
            coeff, pauli_dict = majorana_pauli_terms(mode_index, n_qubits_per_side)
            shifted = {q + offset: p for q, p in pauli_dict.items()}
            return coeff * pauli_hamiltonian_to_matrix([(1.0, shifted)], n_full)

        chi_L1 = embed(1, offset=0)
        chi_R1 = embed(1, offset=n_qubits_per_side)

        # Bare tensor-product convention: commute, not anticommute.
        bare_commutator = chi_L1 @ chi_R1 - chi_R1 @ chi_L1
        assert np.max(np.abs(bare_commutator)) == pytest.approx(0.0, abs=1e-10)

        # Dress chi_R1 with P_L (L's total parity, embedded at offset 0).
        coeff_pl, pauli_pl = total_parity_operator(list(range(1, n_majorana_per_side + 1)), n_qubits_per_side)
        P_L = coeff_pl * pauli_hamiltonian_to_matrix([(1.0, pauli_pl)], n_full)
        chi_R1_dressed = P_L @ chi_R1

        anticomm = chi_L1 @ chi_R1_dressed + chi_R1_dressed @ chi_L1
        assert np.max(np.abs(anticomm)) == pytest.approx(0.0, abs=1e-10)


def _annihilation_matrix(n_qubits, q):
    """Standard full-length Jordan-Wigner c_q, built independently of
    hubbard_hamiltonian_pauli_terms's XX/YY-decomposed hopping terms --
    used only as an independent brute-force reference below."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_at(op, pos):
        m = np.array([[1.0]], dtype=complex)
        for i in range(n_qubits):
            m = np.kron(m, op if i == pos else I2)
        return m

    sigma_minus = 0.5 * (X + 1j * Y)
    result = np.eye(2 ** n_qubits, dtype=complex)
    for p in range(q):
        result = result @ kron_at(Z, p)
    return result @ kron_at(sigma_minus, q)


def _hubbard_matrix_bruteforce(n_sites, t, U, periodic=True):
    """Fermionic-operator construction, independent of the XX/YY Pauli
    decomposition hubbard_hamiltonian_pauli_terms uses."""
    n_qubits = 2 * n_sites
    c = [_annihilation_matrix(n_qubits, q) for q in range(n_qubits)]
    H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    edges = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    if not periodic:
        edges = [(i, j) for i, j in edges if not (i == n_sites - 1 and j == 0)]
    for i, j in edges:
        for off in (0, n_sites):
            qi, qj = off + i, off + j
            H += -t * (c[qi].conj().T @ c[qj] + c[qj].conj().T @ c[qi])
    for i in range(n_sites):
        n_up = c[i].conj().T @ c[i]
        n_dn = c[n_sites + i].conj().T @ c[n_sites + i]
        H += U * (n_up @ n_dn)
    return H


class TestHubbardHamiltonianPauliTerms:
    """Cross-checked against the real Arovas, Bandyopadhyay & Zhu, "The
    Hubbard Model" (Annual Review of Condensed Matter Physics 2022,
    arXiv:2103.12097) review, not just internal self-consistency -- see
    Dense-Evolution-Discovery's hubbard_square_arovas.py (Experiment 38)
    for the full derivation and additional physics checks (Mott
    localization, d-wave pairing sign pattern)."""

    def test_periodic_wraparound_bond_matches_bruteforce_fermionic_construction(self):
        """The one place a naive Jordan-Wigner implementation could
        plausibly need an extra parity correction -- verified exact, not
        assumed, at several system sizes and both periodic/open."""
        t, U = 1.0, 0.5
        for n_sites in (2, 3, 4):
            for periodic in (True, False):
                terms = hubbard_hamiltonian_pauli_terms(n_sites, t, U, periodic=periodic)
                H_pauli = np.asarray(pauli_hamiltonian_to_matrix(terms, 2 * n_sites))
                H_bruteforce = _hubbard_matrix_bruteforce(n_sites, t, U, periodic=periodic)
                max_diff = np.max(np.abs(H_pauli - H_bruteforce))
                assert max_diff < 1e-10, f"n_sites={n_sites} periodic={periodic}: diff={max_diff:.2e}"

    def test_ground_state_energy_matches_arovas_table2_small_u(self):
        """Table 2 (p.6)'s N=4 perturbative formula, E0 = -4t + (3/4)U -
        (13/128)*U^2/t, checked deep in its own regime of validity
        (U/t=0.05) against exact diagonalization through this function's
        own Pauli terms -- real numbers reproduced from
        Dense-Evolution-Discovery Experiment 38, not fabricated here."""
        n_sites, t, U = 4, 1.0, 0.05
        n_qubits = 2 * n_sites
        H_terms = hubbard_hamiltonian_pauli_terms(n_sites, t, U, periodic=True)
        H = np.asarray(pauli_hamiltonian_to_matrix(H_terms, n_qubits))

        N_terms = []
        for q in range(n_qubits):
            N_terms.append((0.5, {}))
            N_terms.append((-0.5, {q: 'Z'}))
        N = np.asarray(pauli_hamiltonian_to_matrix(N_terms, n_qubits))

        evals, evecs = np.linalg.eigh(H)
        populations = np.real(np.diag(evecs.conj().T @ N @ evecs))
        half_filled = np.abs(populations - n_sites) < 1e-6
        energy_exact = float(np.min(evals[half_filled]))

        energy_pert = -4.0 * t + 0.75 * U - (13.0 / 128.0) * (U ** 2 / t)
        rel_diff = abs(energy_exact - energy_pert) / abs(energy_exact)
        assert rel_diff < 1e-3, f"rel_diff={rel_diff:.2e} too large deep in the small-U regime"
        # Reproduces the exact value found in Experiment 38: -3.962753
        assert energy_exact == pytest.approx(-3.962753, abs=1e-5)
