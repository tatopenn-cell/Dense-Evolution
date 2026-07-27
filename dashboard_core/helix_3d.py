"""
Interactive Plotly 3D helix of the statevector's amplitude density.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import numpy as np
import plotly.graph_objects as go


def build_3d_helix_patch(n_qubits: int = 4, probabilities=None) -> go.Figure:
    """Interactive Plotly 3D helix of the statevector's amplitude density.
    Adapted from dash.py:2303 (canonical): drops the globals()-based
    n_qubits fallback and the self-monkey-patching behavior — call directly
    with explicit n_qubits/probabilities instead."""
    n_qubits = int(np.clip(n_qubits, 3, 12))
    hilbert_dim = 2 ** n_qubits

    t = np.linspace(0, 4 * np.pi, hilbert_dim)
    r = np.linspace(0.2, 1.0, hilbert_dim)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.linspace(-1, 1, hilbert_dim)

    if probabilities is not None and len(probabilities) == hilbert_dim:
        prob_weights = np.array(probabilities)
    else:
        prob_weights = np.ones(hilbert_dim) / hilbert_dim
        prob_weights[0] = 0.4
        prob_weights[-1] = 0.3

    sizes = 3 + 25 * (prob_weights / np.max(prob_weights))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color='#8A2BE2', width=2),
        name='Quantum Coherence Spine'
    ))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=sizes, color=prob_weights, colorscale='Viridis',
            opacity=0.8, line=dict(color='#FFA500', width=1)
        ),
        name='Amplitude State Density'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#0D0E15',
        plot_bgcolor='#0D0E15',
        showlegend=False
    )
    return fig
