"""
Unit tests for dense_evolution/vhd_tb.py -- Vogl-Hjalmarson-Dow's
material-specific sp3s* tight-binding parameters. Checks Hermiticity
across the full material table, the real GaAs/Si/Ge gap numbers
recorded in Dense-Evolution-Ising-Tests' validation writeup, and that
Si/Ge's conduction-band minimum is correctly found off-Gamma.
"""
import numpy as np
import pytest

from dense_evolution.vhd_tb import (
    MATERIALS, sp3s_star_hamiltonian, direct_gap_at_gamma, band_extrema_along_path,
)


class TestSp3sStarHamiltonian:

    def test_rejects_unknown_material(self):
        with pytest.raises(ValueError):
            sp3s_star_hamiltonian((0., 0., 0.), 'Xx')

    def test_accepts_material_namedtuple_directly(self):
        H = sp3s_star_hamiltonian((0., 0., 0.), MATERIALS['GaAs'])
        assert H.shape == (10, 10)

    @pytest.mark.parametrize('name', sorted(MATERIALS))
    def test_hermitian_at_generic_k_for_every_material(self, name):
        H = sp3s_star_hamiltonian((0.1, 0.2, 0.3), name)
        assert np.allclose(H, H.conj().T), f'{name}: not Hermitian'


class TestDirectGapAtGamma:

    def test_gaas_matches_recorded_value(self):
        # Recorded 2026-08-05: 1.55 eV vs. 1.42 eV experimental (9.2% error),
        # vs. harrison_tb's universal-parameter 2.906 eV (104.7% error).
        assert direct_gap_at_gamma('GaAs') == pytest.approx(1.55, abs=1e-2)


class TestBandExtremaAlongPath:

    def test_si_indirect_gap_and_cbm_location_off_gamma(self):
        vbm, vbm_k, cbm, cbm_k, gap = band_extrema_along_path('Si', (0, 0, 0), (1, 0, 0))
        assert vbm == pytest.approx(0.0, abs=1e-6)
        assert vbm_k == pytest.approx([0., 0., 0.])
        assert cbm_k[0] > 0.0, 'Si conduction-band minimum must be off-Gamma along Gamma->X'
        # Recorded 2026-08-05: 1.171 eV vs. 1.12 eV experimental (4.6% error).
        assert gap == pytest.approx(1.171, abs=1e-2)

    def test_ge_conduction_minimum_is_at_l_not_x(self):
        _, _, cbm_x, _, gap_x = band_extrema_along_path('Ge', (0, 0, 0), (1, 0, 0))
        _, _, cbm_l, cbm_k_l, gap_l = band_extrema_along_path('Ge', (0, 0, 0), (0.5, 0.5, 0.5))
        assert cbm_l < cbm_x, 'Ge conduction-band minimum must be at L, lower than the X valley'
        assert cbm_k_l == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)
        # Recorded 2026-08-05: 0.765 eV vs. 0.66 eV experimental (15.9% error).
        assert gap_l == pytest.approx(0.765, abs=1e-2)

    def test_rejects_unknown_material(self):
        with pytest.raises(ValueError):
            band_extrema_along_path('Xx', (0, 0, 0), (1, 0, 0))
