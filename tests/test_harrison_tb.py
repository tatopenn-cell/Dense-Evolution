"""
Unit tests for dense_evolution/harrison_tb.py -- Harrison's universal
sp3 tight-binding parameters. Checks the scaling law, Slater-Koster
block construction, and the two Hamiltonian builders (dimer, periodic
zinc-blende), including the real Si-Si/Ga-As/GaAs numbers recorded in
Dense-Evolution-Discovery' validation writeup.
"""
import numpy as np
import pytest

from dense_evolution.harrison_tb import (
    ELEMENTS, ETA, HBAR2_OVER_M_EV_ANG2,
    hopping_integral, sp3_bond_block, sp3_dimer_hamiltonian, zincblende_hamiltonian,
)


class TestHoppingIntegral:

    def test_matches_scaling_law_formula(self):
        eta, d = 1.84, 2.35
        assert hopping_integral(eta, d) == pytest.approx(eta * HBAR2_OVER_M_EV_ANG2 / d ** 2)

    def test_rejects_nonpositive_bond_length(self):
        with pytest.raises(ValueError):
            hopping_integral(1.0, 0.0)
        with pytest.raises(ValueError):
            hopping_integral(1.0, -1.0)


class TestSp3BondBlock:

    def test_rejects_non_unit_direction_vector(self):
        with pytest.raises(ValueError):
            sp3_bond_block(1.0, 1.0, 1.0, 2.35)

    def test_ss_diagonal_element_is_pure_hopping_integral(self):
        block = sp3_bond_block(0.0, 0.0, 1.0, 2.35)
        assert block[0, 0] == pytest.approx(hopping_integral(ETA['ss_sigma'], 2.35))

    def test_sp_off_diagonal_antisymmetric_sign_convention(self):
        block = sp3_bond_block(0.0, 0.0, 1.0, 2.35)
        assert block[0, 3] == pytest.approx(-block[3, 0])


class TestSp3DimerHamiltonian:

    def test_rejects_unknown_element(self):
        with pytest.raises(ValueError):
            sp3_dimer_hamiltonian('Xx', 'Si', 2.35)

    def test_si_si_dimer_is_hermitian_and_matches_recorded_eigenvalues(self):
        H = sp3_dimer_hamiltonian('Si', 'Si', 2.35)
        assert H.shape == (8, 8)
        assert np.allclose(H, H.conj().T)
        eig = np.sort(np.linalg.eigvalsh(H).real)
        expected = [-16.626, -12.25, -9.847, -7.638, -7.638, -5.402, -5.402, -1.418]
        assert eig == pytest.approx(expected, abs=1e-2)

    def test_ga_as_dimer_is_hermitian_and_matches_recorded_eigenvalues(self):
        H = sp3_dimer_hamiltonian('Ga', 'As', 2.45)
        assert np.allclose(H, H.conj().T)
        eig = np.sort(np.linalg.eigvalsh(H).real)
        expected = [-18.386, -12.344, -9.239, -8.228, -8.228, -4.582, -4.582, -1.541]
        assert eig == pytest.approx(expected, abs=1e-2)

    def test_diagonal_blocks_use_each_atoms_own_term_values(self):
        H = sp3_dimer_hamiltonian('Si', 'Ge', 2.4)
        assert H[0, 0] == pytest.approx(ELEMENTS['Si']['eps_s'])
        assert H[4, 4] == pytest.approx(ELEMENTS['Ge']['eps_s'])


class TestZincblendeHamiltonian:

    def test_rejects_unknown_element(self):
        with pytest.raises(ValueError):
            zincblende_hamiltonian([0., 0., 0.], 'Ga', 'Xx', 5.65)

    def test_hermitian_at_gamma_and_generic_k(self):
        for k in ([0., 0., 0.], [0.3, -0.5, 1.1]):
            H = zincblende_hamiltonian(k, 'Ga', 'As', 5.6533)
            assert H.shape == (8, 8)
            assert np.allclose(H, H.conj().T)

    def test_gaas_direct_gap_at_gamma_matches_recorded_value(self):
        H = zincblende_hamiltonian([0., 0., 0.], 'Ga', 'As', 5.6533)
        eig = np.sort(np.linalg.eigvalsh(H).real)
        gap = eig[4] - eig[3]
        # Recorded 2026-08-05 in Dense-Evolution-Discovery' validation
        # writeup: 2.906 eV, ~105% off the 1.42 eV experimental gap --
        # this is Harrison's universal parameters' known behavior, not a
        # target to hit exactly (see vhd_tb for the material-specific fix).
        assert gap == pytest.approx(2.906, abs=1e-2)
