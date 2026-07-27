"""
Mitigation (ZNE) panel -- Phase 3 of the dashboard refactor.

Visualizes run_mitigation_sweep's result: how fidelity degrades across
noise scales, how the zero-noise-extrapolated estimate compares to the
raw noisy sample and the ideal circuit, and -- when predictive healing is
enabled -- makes the delta_preemp correction visible instead of leaving
it a silent internal number.
"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_theme import C, MONO, _ax_style, _badge


def build_panel_mitigation(mitigation_res: dict, base_res: dict) -> plt.Figure:
    """mitigation_res: run_mitigation_sweep's return dict.
    base_res: the plain run_simulation() result at the base noise scale
    (only used for n_qubits/dominant-state formatting fallback -- the
    sweep's own prob_per_scale[0] is the actual base-noise sample)."""
    fig, (ax_fid, ax_prob) = plt.subplots(2, 1, figsize=(16, 12), facecolor=C['bg'])

    noise_factors = mitigation_res['noise_factors']
    fidelity_per_scale = mitigation_res['fidelity_per_scale']
    fidelity_zne = mitigation_res['fidelity_zne']

    # ROW 1 — fidelity per noise scale, with ZNE-corrected / ideal reference lines
    _ax_style(ax_fid, 'Zero-Noise Extrapolation — Fidelity vs Noise Scale',
              'Noise scale factor (× base p)', 'Fidelity')
    x = np.arange(len(noise_factors))
    ax_fid.bar(x, fidelity_per_scale, color=C['noise'], alpha=0.75, width=0.5,
               label='raw noisy (measured)')
    ax_fid.set_xticks(x)
    ax_fid.set_xticklabels([f'{f:g}×' for f in noise_factors])
    ax_fid.axhline(1.0, color=C['accent'], lw=1.0, ls='--', alpha=0.7,
                   label='ideal (F=1)')
    ax_fid.axhline(fidelity_zne, color=C['title'], lw=1.4, ls='-', alpha=0.9,
                   label='ZNE-corrected')
    ax_fid.legend(loc='lower left', fontsize=8, framealpha=0.15,
                  labelcolor=C['label'])
    ax_fid.set_ylim(0.0, max(1.05, fidelity_zne * 1.1, max(fidelity_per_scale) * 1.1))

    healing_txt = (
        f"healing: ON (Δpre_emp={mitigation_res['delta_preemp']:.4f})"
        if mitigation_res['healing_enabled'] else "healing: off (plain ZNE)"
    )
    _badge(ax_fid, f"F_ZNE={fidelity_zne:.6f}  ·  {healing_txt}", C['title'])

    # ROW 2 — top-K basis-state probability overlay: ideal / raw noisy (base
    # scale) / ZNE-corrected (same dim_vis truncation idea as build_panel_mosaico)
    prob_ideal = np.asarray(mitigation_res['prob_ideal'])
    prob_noisy_base = np.asarray(mitigation_res['prob_per_scale'][0])
    prob_zne = np.asarray(mitigation_res['prob_zne'])
    dim_vis = min(1024, len(prob_ideal))

    _ax_style(ax_prob, 'Probability Distribution — Ideal vs Raw vs ZNE-corrected',
              '|n⟩ computational basis (truncated)', 'P(|n⟩)')
    xs = np.arange(dim_vis)
    ax_prob.plot(xs, prob_ideal[:dim_vis], color=C['accent'], lw=1.2, alpha=0.85,
                 label='ideal')
    ax_prob.plot(xs, prob_noisy_base[:dim_vis], color=C['noise'], lw=1.0, alpha=0.6,
                 label='raw noisy (1×)')
    ax_prob.plot(xs, prob_zne[:dim_vis], color=C['title'], lw=1.2, alpha=0.9,
                 ls='--', label='ZNE-corrected')
    ax_prob.legend(loc='upper right', fontsize=8, framealpha=0.15,
                   labelcolor=C['label'])

    fig.tight_layout()
    return fig
