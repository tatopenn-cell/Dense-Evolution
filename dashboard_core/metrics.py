"""
Scalar metric formatting/computation shared between build_panel_overview
and the native st.metric grid.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

from typing import Dict

import numpy as np


def _format_duration_ms(ms: float) -> str:
    if ms >= 1000:
        return f'{ms / 1000:.2f} s'
    return f'{ms:.1f} ms'


def compute_overview_metrics(res: Dict, noise_model: str = 'ideal', noise_p: float = 0.0) -> list:
    """The 13 scalar metrics shown in the Overview panel, as a list of
    {'label', 'value', 'help'} dicts — `value` is kept short so it never
    overflows a fixed-width st.metric tile (e.g. a 15-qubit dominant-state
    bitstring or "2^15 = 32768" both blow past that width); `help` carries
    the full-precision original as a hover tooltip. Single source of truth
    for both build_panel_overview (which no longer renders these as
    matplotlib text — illegible once scaled down in a browser) and the
    native st.metric grid the UI renders instead."""
    prob = res['prob']
    idx_max = res['idx_max']
    n_qubits = res['n_qubits']
    n_states = len(prob)
    prob_max = prob[idx_max]

    concurrence = 1.0 - prob_max
    spectral_std = float(np.std(prob))
    purity_approx = float(np.sum(prob ** 2))
    t_ms = res["tempo"] * 1e3

    return [
        {'label': 'Qubits', 'value': f'{n_qubits}', 'help': None},
        {'label': 'Hilbert Dim', 'value': f'{n_states:,}', 'help': f'2^{n_qubits} basis states'},
        {'label': 'Gates', 'value': f'{res["porte_count"]}', 'help': 'Gates processed'},
        {'label': 'Entropy', 'value': f'{res["entropy"]:.4f} bit', 'help': f'Shannon entropy: {res["entropy"]:.6f} bit'},
        {'label': 'Concurrence', 'value': f'{concurrence:.4f}', 'help': f'Concurrence index: {concurrence:.6f}'},
        {'label': 'Purity', 'value': f'{purity_approx:.4f}', 'help': f'Tr(ρ²) = {purity_approx:.6f}'},
        {'label': 'Spectral σ', 'value': f'{spectral_std:.4f}', 'help': f'Spectral std-dev: {spectral_std:.7f}'},
        {'label': 'Top State', 'value': f'#{idx_max}', 'help': f'|{res["stato_dominante"]}⟩'},
        {'label': 'P(top)', 'value': f'{prob_max:.4f}', 'help': f'Probability of dominant state: {prob_max:.6f}'},
        {'label': 'RAM', 'value': f'{res["ram"]:.2f} MB', 'help': f'Statevector memory: {res["ram"]:.6f} MB'},
        {'label': 'Time', 'value': _format_duration_ms(t_ms), 'help': f'Wall-clock: {t_ms:.3f} ms'},
        {'label': 'Noise', 'value': noise_model, 'help': 'Noise model applied to this run'},
        {'label': 'Noise p', 'value': f'{noise_p:.3f}', 'help': f'Noise probability: {noise_p:.4f}'},
    ]
