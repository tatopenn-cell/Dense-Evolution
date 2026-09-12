"""
Real sp3s* tight-binding band structure along a k-path
(dense_evolution.sp3s_star_hamiltonian, the same real Hamiltonian
direct_gap_at_gamma/band_extrema_along_path already use -- all 10 bands
here, not just the VBM/CBM pair those two track), plus an interactive
Plotly figure with real hover tooltips (k-point and energy per band).

This is the one interactive (hover/zoom) visualization in the dashboard;
everywhere else in dashboard_core.visuals uses static matplotlib. The
choice here is unrelated to that module's own reason for staying
matplotlib-only (Qiskit's macOS instability, see visuals.py) -- this is
purely about wanting real per-point hover data that a static image
cannot provide, and needed adding plotly as a declared dependency
(pyproject.toml's `dashboard` extra) since it was only ever an
incidental transitive package before.
"""

import numpy as np
import plotly.graph_objects as go

import dense_evolution as de

__all__ = ['scan_bands_along_path', 'band_structure_figure']


def scan_bands_along_path(material, k_start, k_end, n_points=201):
    """All 10 sp3s* tight-binding bands (real eigenvalues of
    dense_evolution.sp3s_star_hamiltonian at each sampled k-point, not
    just the single VBM/CBM pair band_extrema_along_path tracks) along
    the straight line from k_start to k_end, both in the same 2*pi/a
    cubic-axis units as sp3s_star_hamiltonian itself.

    Returns (t, bands): t is the fractional position along the path
    (0..1, n_points), bands is a (n_points, 10) array with band j's
    energy at t[i] in bands[i, j] (sorted ascending at every k-point).
    """
    k_start = np.asarray(k_start, dtype=float)
    k_end = np.asarray(k_end, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    ks = k_start[None, :] + t[:, None] * (k_end - k_start)[None, :]

    bands = np.empty((n_points, 10))
    for i, k in enumerate(ks):
        bands[i] = np.sort(np.linalg.eigvalsh(de.sp3s_star_hamiltonian(k, material)).real)
    return t, bands


def band_structure_figure(t, bands, material_name, k_start_label='Γ', k_end_label='X'):
    """Interactive Plotly band-structure plot: one line per band, real
    hover tooltip per point showing the exact k-path position and
    energy (not read off a static image). Bands 3/4 (0-indexed, the
    valence-band maximum / conduction-band minimum pair
    band_extrema_along_path itself tracks) are highlighted."""
    fig = go.Figure()
    n_bands = bands.shape[1]
    for j in range(n_bands):
        is_frontier = j in (3, 4)
        fig.add_trace(go.Scatter(
            x=t, y=bands[:, j], mode='lines',
            name=f'banda {j}' + (' (VBM)' if j == 3 else ' (CBM)' if j == 4 else ''),
            line=dict(width=3 if is_frontier else 1, color='#d62728' if j == 3 else '#1f77b4' if j == 4 else '#999999'),
            hovertemplate=f'banda {j}<br>t=%{{x:.3f}}<br>E=%{{y:.4f}} eV<extra></extra>',
        ))
    fig.update_layout(
        title=f'Struttura a bande sp3s* -- {material_name} ({k_start_label}→{k_end_label})',
        xaxis_title=f'{k_start_label} → {k_end_label}',
        yaxis_title='Energia (eV)',
        hovermode='closest',
        template='plotly_white',
    )
    return fig
