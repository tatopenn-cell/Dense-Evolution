"""
Tests for dashboard_core.band_structure -- real sp3s* tight-binding
band scan (dense_evolution.sp3s_star_hamiltonian) plus the interactive
Plotly figure, checked against known physics (Si's real direct gap at
Gamma), not just "no exception".
"""

import numpy as np
import pytest

from plotly.graph_objs import Figure

from dashboard_core.band_structure import scan_bands_along_path, band_structure_figure


def test_scan_returns_expected_shapes():
    t, bands = scan_bands_along_path('Si', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), n_points=11)
    assert t.shape == (11,)
    assert bands.shape == (11, 10)


def test_bands_are_sorted_ascending_at_every_k_point():
    _, bands = scan_bands_along_path('Si', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), n_points=11)
    for row in bands:
        assert list(row) == sorted(row)


def test_gamma_point_vbm_cbm_gap_matches_direct_gap_at_gamma():
    import dense_evolution as de
    t, bands = scan_bands_along_path('Si', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), n_points=5)
    assert t[0] == pytest.approx(0.0)
    gamma_gap = bands[0, 4] - bands[0, 3]
    assert gamma_gap == pytest.approx(de.direct_gap_at_gamma('Si'), abs=1e-9)


def test_band_structure_figure_returns_a_plotly_figure():
    t, bands = scan_bands_along_path('Si', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), n_points=11)
    fig = band_structure_figure(t, bands, 'Si')
    assert isinstance(fig, Figure)
    assert len(fig.data) == 10
