"""
Matplotlib panel builders — adapted from dash.py's matplotlib figure code.
Converted to explicit-parameter pure functions: no globals() reads, no
ipywidgets, no plt.ioff()/plt.ion() (Streamlit doesn't need interactive
mode toggling), figures returned for st.pyplot().

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor). build_panel_fisica/vqe_results/md_results/performance
(Phase 2) now reuse plot_theme.C's semantic colors wherever a hardcoded
hex literal was an EXACT match (entropy/purity/grad/noise/theta/dom/accent)
-- zero visual change, verified. Near-miss shades (close but not identical
to a C[...] value, e.g. '#00e5ff' vs C['energy']='#00e0ff') were
deliberately left as local literals rather than forced onto the "closest"
C key, which would have been a real (if small) unauthorized color drift.
build_panel_mosaico already had its own centralized art palette
(BG_ART/PANEL_ART/CYAN_N/PINK_N/PURP_N/GOLD_N, moved in Phase 1) --
intentionally distinct from C, not touched here.
"""

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import seaborn as sns

from .plot_theme import (
    C, MONO, _cmap_prob, _ax_style, _series_plot,
    _noise_profile_plot, _badge, BG_ART, PANEL_ART, CYAN_N, PINK_N, PURP_N, GOLD_N,
    FIG_BG_DARK,
)


def build_panel_overview(res: Dict, df_vqe: pd.DataFrame, corr_matrix: pd.DataFrame,
                          noise_model: str = 'ideal', noise_p: float = 0.0) -> plt.Figure:
    """
    Layout  (8 rows × 2 cols)
    R0  header bar (full width)
    R1  probability distribution  |  top-N states ranked
    R2  wavefunction helix 3D     |  simulation metrics table
    R3  noise analysis            |  NISQ shot histogram
    R4  VQE energy [enhanced]     |  VQE entropy [enhanced]
    R5  purity [enhanced]         |  gradient [enhanced + plateau]
    R6  noise factor [enhanced]   |  theta correction [enhanced]
    R7  correlation heatmap (full width)

    Adapted from dash.py:1833 (canonical). df_vqe/corr_matrix/noise_model/noise_p
    are explicit params here instead of globals()/widget lookups.
    """
    prob          = res['prob']
    n_qubits      = res['n_qubits']
    idx_max       = res['idx_max']
    t_elapsed     = res['tempo']
    ram_mb        = res['ram']
    gates         = res['porte_count']
    shots_data    = res['shots_data']
    prob_max      = prob[idx_max]
    n_states      = len(prob)

    df_vqe   = df_vqe if df_vqe is not None else pd.DataFrame()
    mat_cor  = corr_matrix if corr_matrix is not None else pd.DataFrame()
    prob_noisy = prob if noise_model != 'ideal' else None
    prob_ref   = prob

    fig = plt.figure(figsize=(22, 34), facecolor=C['bg'])
    gs  = gridspec.GridSpec(
        8, 2,
        figure=fig,
        height_ratios=[0.10, 1.0, 1.0, 0.90, 0.85, 0.85, 0.85, 1.20],
        hspace=0.58, wspace=0.28,
        left=0.050, right=0.972, top=0.978, bottom=0.028,
    )

    # ROW 0 — header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(C['bg']); ax_h.axis('off')
    ax_h.text(0.0, 0.90,
              f'QUANTUM CIRCUIT OVERVIEW  ·  {res["nome"]}',
              transform=ax_h.transAxes, fontsize=14, fontweight='bold',
              color=C['title'], va='top', **MONO)
    stat_str = (f'{n_qubits} qb  ·  2^{n_qubits} = {n_states}  ·  '
                f'{gates} gates  ·  {t_elapsed*1e3:.2f} ms  ·  '
                f'{ram_mb:.3f} MB  ·  noise: {noise_model}  p={noise_p:.4f}')
    ax_h.text(0.0, 0.22, stat_str,
              transform=ax_h.transAxes, fontsize=8,
              color=C['label'], va='top', **MONO)
    fig.add_artist(plt.Line2D(
        [0.050, 0.972], [0.966, 0.966],
        transform=fig.transFigure, color=C['border'], lw=0.8))

    # ROW 1 — probability distribution | top-N states
    ax_pb = fig.add_subplot(gs[1, 0])
    _ax_style(ax_pb, 'Probability Distribution  P(|n⟩)',
              '|n⟩ computational basis', 'P(|n⟩)')
    norm_p   = Normalize(prob.min(), prob.max())
    bar_cols = _cmap_prob(norm_p(prob))
    ax_pb.bar(np.arange(n_states), prob,
              color=bar_cols, width=1.0, edgecolor='none', alpha=0.85)
    ax_pb.bar(idx_max, prob_max, color=C['dom'], width=1.0,
              edgecolor='none', alpha=0.95, zorder=3)
    ax_pb.axhline(1.0 / n_states, color=C['label'],
                  lw=0.7, ls=':', alpha=0.55)
    ax_pb.set_xlim(-0.5, n_states - 0.5)
    step = max(1, n_states // min(16, n_states))
    ax_pb.set_xticks(np.arange(0, n_states, step))
    ax_pb.tick_params(axis='x', rotation=45)

    ax_tp = fig.add_subplot(gs[1, 1])
    n_top = min(12, n_states)
    top_i = np.argsort(prob)[-n_top:][::-1]
    top_p = prob[top_i]
    top_lb = [f'|{bin(i)[2:].zfill(n_qubits)}⟩' for i in top_i]
    _ax_style(ax_tp, f'Top-{n_top} States by Probability', 'P(|n⟩)', '')
    norm_t  = Normalize(top_p.min(), top_p.max())
    hbar_c  = plt.cm.plasma(norm_t(top_p))
    yp      = np.arange(n_top)
    hbars   = ax_tp.barh(yp, top_p, color=hbar_c,
                         height=0.60, edgecolor='none', alpha=0.88)
    ax_tp.set_yticks(yp)
    ax_tp.set_yticklabels(top_lb, fontsize=8, color=C['tick'], **MONO)
    for idx_b, (bar, pv) in enumerate(zip(hbars, top_p)):
        ax_tp.text(pv + max(top_p) * 0.012, idx_b,
                   f'{pv:.5f}', va='center',
                   fontsize=7, color=C['label'], **MONO)
    ax_tp.set_xlim(0, max(top_p) * 1.22)
    ax_tp.invert_yaxis()

    # ROW 2 — wavefunction helix 3D (full width; the old "Simulation Metrics"
    # text panel that used to share this row moved to native st.metric tiles
    # in the UI layer — too small to read once matplotlib scales down to fit
    # a browser column, see compute_overview_metrics())
    ax_3d = fig.add_subplot(gs[2, :], projection='3d')
    ax_3d.set_facecolor(C['panel'])
    dim_v = min(512, n_states)
    amps  = np.sqrt(prob[:dim_v])
    phi_v = np.linspace(0, 6 * np.pi, dim_v)
    x3    = amps * np.cos(phi_v)
    y3    = amps * np.sin(phi_v)
    z3    = np.linspace(0, 1, dim_v)
    ax_3d.scatter(x3, y3, z3, c=amps, cmap='cool',
                  s=18, alpha=0.92, linewidths=0, depthshade=False)
    ax_3d.plot(x3, y3, z3, color=C['energy'], alpha=0.35, lw=0.8)
    ax_3d.set_title('Wavefunction Helix  ψ(|n⟩)',
                    color=C['title'], fontsize=9.5,
                    fontweight='bold', pad=4, **MONO)
    for attr, lbl in [('xlabel', 'Re(ψ)'),
                       ('ylabel', 'Im(ψ)'), ('zlabel', '|n⟩')]:
        getattr(ax_3d, f'set_{attr}')(lbl, color=C['label'],
                                      fontsize=7.5, labelpad=1)
    ax_3d.tick_params(colors=C['tick'], labelsize=6.5)
    for pane in [ax_3d.xaxis.pane,
                 ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(C['border'])
    ax_3d.view_init(elev=24, azim=50)

    # ROW 3 — noise analysis | NISQ shot histogram
    ax_ns = fig.add_subplot(gs[3, 0])
    _noise_profile_plot(ax_ns, noise_model, noise_p,
                        n_qubits, prob_ref, prob_noisy)

    ax_sh = fig.add_subplot(gs[3, 1])
    _ax_style(ax_sh,
              f'NISQ Shot Histogram  ({len(shots_data):,} samples)',
              '|n⟩ basis state', 'Counts')
    counts   = np.bincount(shots_data, minlength=n_states).astype(float)
    expected = prob * len(shots_data)
    norm_sh  = Normalize(counts.min(), counts.max())
    sh_cols  = plt.cm.viridis(norm_sh(counts))
    ax_sh.bar(np.arange(n_states), counts,
              color=sh_cols, width=1.0, edgecolor='none', alpha=0.80)
    ax_sh.plot(np.arange(n_states), expected,
               color=C['warn'], lw=1.2, alpha=0.75,
               ls='--', label='expected')
    ax_sh.set_xlim(-0.5, n_states - 0.5)
    ax_sh.legend(loc='upper right', fontsize=7,
                 framealpha=0.15, labelcolor=C['label'])
    sigma = np.sqrt(len(shots_data) * prob_max * (1 - prob_max))
    ax_sh.annotate(
        f'σ(|dom⟩) ≈ {sigma:.1f}',
        xy=(idx_max, counts[idx_max]),
        xytext=(10, 10), textcoords='offset points',
        color=C['warn'], fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color=C['warn'], lw=0.8),
        **MONO,
    )

    # ROW 4 — VQE energy | entropy
    _series_plot(fig.add_subplot(gs[4, 0]), df_vqe,
                 'VQE_Energy', C['energy'], 'VQE Energy', 'Epoch', 'E  (Ha)',
                 badge=True, badge_fmt='Eₙ={:.4g} Ha', mark_convergence=True)
    _series_plot(fig.add_subplot(gs[4, 1]), df_vqe,
                 'Entropy', C['entropy'],
                 'Von Neumann Entropy', 'Epoch', 'S  (bit)',
                 badge=True)

    # ROW 5 — purity | gradient
    _series_plot(fig.add_subplot(gs[5, 0]), df_vqe,
                 'Purity', C['purity'],
                 'State Purity  Tr(ρ²)', 'Epoch', 'Tr(ρ²)',
                 badge=True, ref_line=1.0)
    _series_plot(fig.add_subplot(gs[5, 1]), df_vqe,
                 'Gradient', C['grad'],
                 '‖∇L‖  Gradient Norm', 'Epoch', '‖∇L‖',
                 badge=True, detect_plateau=True)

    # ROW 6 — noise factor | theta correction
    _series_plot(fig.add_subplot(gs[6, 0]), df_vqe,
                 'Noise_Factor', C['noise'],
                 'Noise Factor', 'Epoch', 'Factor',
                 badge=True, ref_line=1.0)
    _series_plot(fig.add_subplot(gs[6, 1]), df_vqe,
                 'Theta_Correction', C['theta'],
                 'θ  Correction', 'Epoch', 'Δθ  (rad)',
                 badge=True, ref_line=0.0)

    # ROW 7 — correlation heatmap (full width)
    ax_c = fig.add_subplot(gs[7, :])
    ax_c.set_facecolor(C['panel'])
    for sp in ax_c.spines.values():
        sp.set_edgecolor(C['border']); sp.set_linewidth(0.7)

    if not mat_cor.empty:
        _n    = len(mat_cor)
        _afs  = max(6.0, min(9.0, 72.0 / _n))
        _mask = np.triu(np.ones_like(mat_cor, dtype=bool), k=1)
        _labs = [c.replace('_', '\n') for c in mat_cor.columns]
        sns.heatmap(
            mat_cor,
            mask=_mask,
            annot=True, fmt='.2f',
            cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, center=0.0,
            ax=ax_c,
            square=True,
            linewidths=0.22, linecolor=C['bg'],
            annot_kws={'size': _afs, 'fontfamily': 'monospace'},
            xticklabels=_labs, yticklabels=_labs,
            cbar_kws={'label': 'Pearson r',
                      'shrink': 0.65, 'pad': 0.01,
                      'format': '%.1f'},
        )
        ax_c.set_title('Pearson Correlation  ·  MD Telemetry',
                       color=C['title'], fontsize=9.5,
                       fontweight='bold', pad=6,
                       loc='left', **MONO)
        ax_c.tick_params(axis='x', colors=C['tick'],
                         rotation=30, labelsize=7.5)
        ax_c.tick_params(axis='y', colors=C['tick'],
                         rotation=0,  labelsize=7.5)
        cbar = ax_c.collections[0].colorbar
        cbar.ax.yaxis.label.set_color(C['label'])
        cbar.ax.tick_params(colors=C['tick'], labelsize=6.5)
        cbar.outline.set_edgecolor(C['border'])
    else:
        ax_c.axis('off')
        ax_c.text(0.5, 0.5,
                  'Correlation matrix — enable MD simulation to populate',
                  ha='center', va='center', color=C['label'],
                  fontsize=9.5, transform=ax_c.transAxes, **MONO)

    return fig


def build_panel_fisica(res, seed: int = 42) -> plt.Figure:
    """Adapted from dash.py:2121 (canonical). `seed` is an explicit param
    instead of the original's `w_seed.value if 'w_seed' in globals() else 42`."""
    probabilita = res['prob']
    idx_max = res['idx_max']
    shannon_entropy = res['entropy']

    prob_max = probabilita[idx_max]
    concurrence_val = 1.0 - prob_max
    deviazione_spettrale = np.std(probabilita)

    dim_vis = min(1024, len(probabilita))
    sv_vis = np.sqrt(probabilita[:dim_vis])

    fig = plt.figure(figsize=(22, 12), facecolor=FIG_BG_DARK)

    metrics = [
        (0.12, "S H A N N O N - E N T R O P Y", f"{shannon_entropy:.4f} b", "#b400ff"),
        (0.38, "C O N C U R R E N C E - I N D E X", f"{concurrence_val:.4f}", "#ff007f"),
        (0.62, "P E A K - P R O B A B I L I T Y", f"{prob_max*100:.2f}%", "#00c8ff"),
        (0.88, "S P E C T R A L - D E V I A T I O N", f"{deviazione_spettrale:.5f}", C['accent'])
    ]
    for x, label, val, col in metrics:
        fig.text(x, 0.960, label, color='#7d8590', fontsize=10, ha='center', fontfamily='monospace')
        fig.text(x, 0.910, val, color=col, fontsize=30, fontweight='bold', ha='center', fontfamily='monospace')

    ax2 = fig.add_axes([0.52, 0.10, 0.45, 0.70], projection='3d')
    ax2.set_facecolor('#0d1117')
    # `seed` (the parameter above) was only ever used to build an `rng` that
    # this deterministic scatter (linspace/cos/sin below) never actually
    # consumed -- dead code removed rather than wired up, since nothing here
    # needs randomization. `seed` itself is left in the signature: callers
    # already pass it, and dropping the parameter is a separate API change.
    angoli = np.linspace(0, 2 * np.pi, dim_vis)
    raggio = np.sqrt(range(dim_vis))
    x_c = raggio * np.cos(angoli)
    y_c = raggio * np.sin(angoli)
    z_c = sv_vis
    ax2.scatter(x_c, y_c, z_c, c=z_c, cmap='plasma', s=80, alpha=0.7, edgecolors='#f0f6fc', lw=0.1)
    ax2.set_title("Topografia Tridimensionale delle Ampiezza d'Onda ($2^n$)", color=C['accent'], fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')

    ax3 = fig.add_axes([0.05, 0.10, 0.45, 0.38], projection='3d')
    ax3.set_facecolor('#0d1117')
    X, Y = np.meshgrid(np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
    R = np.sqrt(X**2 + Y**2)
    frequenza_onda = max(0.5, shannon_entropy / 2.0)
    ampiezza_onda = max(0.1, deviazione_spettrale * 5.0)
    Z = np.sin(R * frequenza_onda) * np.exp(-R * 0.3) * ampiezza_onda
    ax3.plot_surface(X, Y, Z, cmap='magma', alpha=0.85, antialiased=True, lw=0)
    ax3.set_title("Onda di Risonanza e Spettro Coerenza Spaziale", color=C['accent'], fontsize=12, fontweight='bold', pad=10)
    ax3.axis('off')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_facecolor('#0d1117')
    num_barre = min(32, len(probabilita))
    indici_barre = np.arange(num_barre)
    zero_base = np.zeros(num_barre)
    dx = dy = 0.6
    dz = probabilita[:num_barre]
    ax1.bar3d(indici_barre, zero_base, zero_base, dx, dy, dz, color='#00c8ff', alpha=0.7, shade=True)
    ax1.set_title("Distribuzione Vettoriale Primitivi Quantistici", color=C['accent'], fontsize=12, fontweight='bold')
    ax1.axis('off')

    return fig


def build_panel_mosaico(res) -> plt.Figure:
    """MOSAICO: fractal/artistic transform of the statevector. Adapted from
    dash.py:1059 (fully self-contained in the original, straight port)."""
    probabilita = res['prob']
    shannon_entropy = res['entropy']
    idx_max = res['idx_max']

    dim_vis = min(1024, len(probabilita))
    prob_vis = probabilita[:dim_vis]
    ampiezze = np.sqrt(prob_vis)

    fig = plt.figure(figsize=(22, 12), facecolor=BG_ART)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22)

    fig.text(0.08, 0.94, f"Ψ_SHANNON: {shannon_entropy:.4f} b", color=CYAN_N, fontsize=14, fontweight='bold', fontfamily='monospace')
    fig.text(0.38, 0.94, f"Ξ_CONCURRENCE: {1.0 - probabilita[idx_max]:.5f}", color=PINK_N, fontsize=14, fontweight='bold', fontfamily='monospace')
    fig.text(0.68, 0.94, f"Ω_PEAK_STATE: |{res['stato_dominante']}>", color=GOLD_N, fontsize=14, fontweight='bold', fontfamily='monospace')

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL_ART)
    side = int(np.ceil(np.sqrt(dim_vis)))
    pad_size = (side * side) - dim_vis
    prob_padded = np.pad(prob_vis, (0, pad_size), mode='constant') if pad_size > 0 else prob_vis
    ax0.imshow(prob_padded.reshape(side, side), cmap='twilight_shifted', aspect='equal', origin='lower', interpolation='bicubic')
    ax0.set_title("Mosaico Olografico Spettrale", color=CYAN_N, fontsize=11, fontweight='bold', fontfamily='monospace', pad=10)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(PANEL_ART)
    x_wave = np.linspace(0, 4 * np.pi, dim_vis)
    y_wave = np.sin(x_wave * (shannon_entropy / 2.0)) * ampiezze
    points = np.array([x_wave, y_wave]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=plt.Normalize(prob_vis.min(), prob_vis.max()))
    lc.set_array(prob_vis)
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)
    ax1.set_xlim(x_wave.min(), x_wave.max())
    ax1.set_ylim(y_wave.min() - 0.05, y_wave.max() + 0.05)
    ax1.set_title("Flusso di Coerenza Variazionale", color=PINK_N, fontsize=11, fontweight='bold', fontfamily='monospace', pad=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1:3], projection='3d')
    ax2.set_facecolor(PANEL_ART)
    phi = np.linspace(0, 8 * np.pi, dim_vis)
    raggio_frattale = np.exp(0.04 * phi)
    x_3d = raggio_frattale * np.sin(phi) * ampiezze
    y_3d = raggio_frattale * np.cos(phi) * ampiezze
    z_3d = phi * prob_vis
    ax2.scatter(x_3d, y_3d, z_3d, c=prob_vis, cmap='inferno', s=ampiezze*600, alpha=0.8, edgecolors=CYAN_N, lw=0.2)
    ax2.plot(x_3d, y_3d, z_3d, color=PURP_N, alpha=0.18, linewidth=0.6)
    ax2.set_title("Topografia Frattale dello Spazio di Hilbert ($2^n$)", color=GOLD_N, fontsize=12, fontweight='bold', fontfamily='monospace')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 1:3], projection='3d')
    ax3.set_facecolor(PANEL_ART)
    X, Y = np.meshgrid(np.linspace(-4, 4, 110), np.linspace(-4, 4, 110))
    R = np.sqrt(X**2 + Y**2)
    modulazione_fase = np.cos(X * (shannon_entropy / 4.0)) * np.sin(Y * 1.9106)
    Z = np.sin(R * (shannon_entropy / 2.0)) * np.exp(-R * 0.25) * (modulazione_fase * 0.4)
    ax3.plot_surface(X, Y, Z, cmap='magma', alpha=0.4, antialiased=True, lw=0)
    ax3.plot_wireframe(X, Y, Z, color=CYAN_N, alpha=0.10, rstride=4, cstride=4)
    ax3.set_title("Onda di Risonanza Quantistica Asimmetrica", color=CYAN_N, fontsize=12, fontweight='bold', fontfamily='monospace')
    ax3.axis('off')

    return fig


def build_panel_vqe_results(df_vqe: pd.DataFrame) -> plt.Figure:
    """Adapted from dash.py:1272 (canonical). Takes `df_vqe` directly instead
    of the original's globals()['df_vqe_telemetry'] lookup (the original's own
    `res` argument was unused)."""
    if df_vqe is None or df_vqe.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nessun dato VQE disponibile. Eseguire una simulazione VQE.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(3, 2, figsize=(20, 20), facecolor=FIG_BG_DARK)
    fig.suptitle('VQE Optimization Results', color=C['accent'], fontsize=24, fontweight='bold', fontfamily='monospace')

    plots = [
        (axes[0, 0], 'VQE_Energy', '#00e5ff', 'VQE Energy per Epoch', 'Energy (Ha)'),
        (axes[0, 1], 'Entropy', C['entropy'], 'Entropy per Epoch', 'Entropy (bit)'),
        (axes[1, 0], 'Purity', C['purity'], 'Purity per Epoch', 'Purity'),
        (axes[1, 1], 'Gradient', C['grad'], 'Gradient per Epoch', 'Gradient'),
        (axes[2, 0], 'Noise_Factor', C['noise'], 'Noise Factor per Epoch', 'Noise Factor'),
        (axes[2, 1], 'Theta_Correction', C['theta'], 'Theta Correction per Epoch', 'Theta (rad)'),
    ]
    for ax, col, color, title, ylabel in plots:
        ax.plot(df_vqe.index, df_vqe[col], color=color, linewidth=2)
        ax.set_title(title, color='#dce3f5', fontsize=14)
        ax.set_xlabel('Epoch', color='#4e6490', fontsize=12)
        ax.set_ylabel(ylabel, color='#4e6490', fontsize=12)
        ax.tick_params(axis='x', colors='#354f7a')
        ax.tick_params(axis='y', colors='#354f7a')
        ax.set_facecolor('#080c1a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#111829')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def build_panel_md_results(df_md: pd.DataFrame, corr_matrix: pd.DataFrame) -> plt.Figure:
    """Adapted from dash.py:1371 (canonical) — already had an explicit-param
    signature in the original, straight port."""
    if df_md is None or df_md.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nessun dato MD disponibile. Eseguire una simulazione MD.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    fig = plt.figure(figsize=(22, 20), facecolor=FIG_BG_DARK)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Molecular Dynamics Simulation Results', color=C['accent'], fontsize=24, fontweight='bold', fontfamily='monospace')

    plots = [
        (gs[0, 0], 'Energia_VQE_Ha', '#00e5ff', 'VQE Energy (Ha) during MD', 'Energy (Ha)'),
        (gs[0, 1], 'Entropia_von_Neumann_Bit', C['entropy'], 'Von Neumann Entropy (Bit) during MD', 'Entropy (Bit)'),
        (gs[1, 0], 'Purita_Stato', C['purity'], 'State Purity during MD', 'Purity'),
        (gs[1, 1], 'Gradiente_Operatore', C['grad'], 'Operator Gradient during MD', 'Gradient'),
    ]
    for gs_cell, col, color, title, ylabel in plots:
        ax = fig.add_subplot(gs_cell)
        ax.plot(df_md.index, df_md[col], color=color, linewidth=2)
        ax.set_title(title, color='#dce3f5', fontsize=14)
        ax.set_xlabel('MD Step', color='#4e6490', fontsize=12)
        ax.set_ylabel(ylabel, color='#4e6490', fontsize=12)
        ax.set_facecolor('#080c1a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#111829')

    ax4 = fig.add_subplot(gs[2, :])
    sns.heatmap(
        corr_matrix,
        annot=True, fmt='.2f', cmap='RdYlBu_r',
        ax=ax4,
        linewidths=0.25, linecolor=FIG_BG_DARK,
        annot_kws={'size': 8.5, 'color': '#dde4f5', 'fontfamily': 'monospace'},
        cbar_kws={'label': 'Pearson r', 'shrink': 0.72, 'pad': 0.01},
    )
    ax4.set_title('Correlation Matrix of MD Telemetry', color='#dce3f5', fontsize=14)
    ax4.tick_params(axis='x', colors='#354f7a', rotation=28, labelsize=8)
    ax4.tick_params(axis='y', colors='#354f7a', rotation=0, labelsize=8)
    ax4.set_facecolor('#080c1a')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════
# Hamiltonian panel — genuinely new, not a port. dash.py's "Custom
# Hamiltonian" panel option (dash.py:3025-3029) just printed the raw H-matrix
# text to console; built here as an actual energy-spectrum chart instead.
# ══════════════════════════════════════════════════════════════════════

def build_panel_hamiltonian(hamiltonian_values, name: str = 'Custom') -> plt.Figure:
    """Bar chart of a Hamiltonian's diagonal energy spectrum (the `H_matrix`
    diagonal built in run_vqe_telemetry when hamiltonian_values is passed)."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C['bg'])

    if not hamiltonian_values:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Nessuna Hamiltoniana selezionata.',
                ha='center', va='center', color=C['label'], fontsize=11, transform=ax.transAxes, **MONO)
        return fig

    values = np.asarray(hamiltonian_values, dtype=float)
    _ax_style(ax, f'Spettro Energetico — {name}', 'Autostato |n⟩', 'Energia (Ha)')
    norm = Normalize(values.min(), values.max())
    colors = plt.cm.viridis(norm(values))
    ax.bar(np.arange(len(values)), values, color=colors, width=0.9, edgecolor='none', alpha=0.9)
    ax.axhline(0.0, color=C['label'], lw=0.6, ls=':', alpha=0.6)
    _badge(ax, f'dim={len(values)}  ·  E_min={values.min():.3f}  ·  E_max={values.max():.3f}', C['accent'])

    return fig


# ══════════════════════════════════════════════════════════════════════
# Performance panel — genuinely new, not a port. build_panel_performance
# in dash.py (lines 893 and 2904, both identical) was never implemented:
# it just prints a placeholder and returns None. Built here from data that
# already exists (res['tempo']/['ram']/..., run history) plus a real,
# explicitly-triggered benchmark scan (adapted from
# DiagnosticTools.core_trigger_benchmark, dash.py:2460, canonical version).
# ══════════════════════════════════════════════════════════════════════

def build_panel_performance(res: Dict, run_history: list) -> plt.Figure:
    """New panel (see module docstring above). Time/RAM per run from
    `run_history` (a list of dicts with at least 'nome', 'n_qubits',
    'tempo', 'ram' — the same fields already returned by run_simulation)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=FIG_BG_DARK)

    if not run_history:
        for ax in axes:
            ax.axis('off')
        fig.text(0.5, 0.5, 'Nessuno storico run disponibile. Esegui una simulazione.',
                  ha='center', va='center', color=C['label'], fontsize=12, **MONO)
        return fig

    names   = [r['nome'] for r in run_history]
    tempi   = [r['tempo'] * 1e3 for r in run_history]
    rams    = [r['ram'] for r in run_history]
    qubits  = [r['n_qubits'] for r in run_history]
    x       = np.arange(len(run_history))

    ax0 = axes[0]
    _ax_style(ax0, 'Wall-clock Time per Run', 'Run #', 'ms')
    bars0 = ax0.bar(x, tempi, color=C['energy'], alpha=0.85)
    for xi, q in zip(x, qubits):
        ax0.text(xi, 0, f'{q}q', ha='center', va='bottom', fontsize=7, color=C['label'], **MONO)

    ax1 = axes[1]
    _ax_style(ax1, 'Statevector RAM per Run', 'Run #', 'MB')
    ax1.bar(x, rams, color=C['purity'], alpha=0.85)

    fig.suptitle(f'Performance — {len(run_history)} run(s) in this session',
                 color=C['title'], fontsize=14, fontweight='bold', **MONO)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    return fig


_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def compute_bloch_vector(sv, n_qubits: int, qubit_index: int):
    """The real Bloch vector (Tr(ρX), Tr(ρY), Tr(ρZ)) for one qubit's
    reduced density matrix ρ, traced out of the full `sv` statevector --
    standard single-qubit visualization in Qiskit (plot_bloch_vector) and
    PennyLane/qsim demos, previously absent from this dashboard.

    Qubit-index convention verified empirically against this simulator's
    own measurement labeling (format(idx, '0{n}b'), qubit 0 = leftmost/
    most-significant bit): reshaping the flat statevector into `n_qubits`
    axes of size 2 puts axis `i` in exactly that same qubit-0-first order,
    so no bit-reversal is needed here.

    Returns (bx, by, bz), each a float in [-1, 1] (exactly on the unit
    sphere surface for a pure, unentangled qubit; strictly inside it —
    a real physical effect, not a bug — when qubit_index is entangled
    with the rest of the register, e.g. (0,0,0) for either half of a
    Bell pair)."""
    sv = np.asarray(sv).reshape([2] * n_qubits)
    sv = np.moveaxis(sv, qubit_index, 0).reshape(2, -1)
    rho = sv @ sv.conj().T
    return (
        float(np.real(np.trace(rho @ _PAULI_X))),
        float(np.real(np.trace(rho @ _PAULI_Y))),
        float(np.real(np.trace(rho @ _PAULI_Z))),
    )


def build_panel_bloch_sphere(sv, n_qubits: int, qubit_index: int = 0) -> plt.Figure:
    """Plots compute_bloch_vector's result on a 3D Bloch sphere -- a
    wireframe sphere, the three real (X/Y/Z) axes, and an arrow from the
    origin to the qubit's actual state. A short arrow (inside the sphere,
    not reaching the surface) is the correct, honest rendering of a
    genuinely mixed/entangled reduced state, not a rendering error."""
    bx, by, bz = compute_bloch_vector(sv, n_qubits, qubit_index)

    fig = plt.figure(figsize=(6, 6), facecolor=FIG_BG_DARK)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='3d')
    ax.set_facecolor(C['panel'])

    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color=C['border'], linewidth=0.4, alpha=0.5)

    axis_len = 1.3
    for vec, label in (([axis_len, 0, 0], 'X'), ([0, axis_len, 0], 'Y'), ([0, 0, axis_len], 'Z')):
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=C['label'], lw=0.8)
        ax.text(*vec, label, color=C['label'], fontsize=10, **MONO)

    ax.quiver(0, 0, 0, bx, by, bz, color=C['accent'], linewidth=2.5, arrow_length_ratio=0.12)
    ax.scatter([bx], [by], [bz], color=C['accent'], s=40)

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.suptitle(
        f'Bloch Sphere — qubit {qubit_index}  '
        f'(x={bx:.3f}, y={by:.3f}, z={bz:.3f}, |v|={(bx**2+by**2+bz**2)**0.5:.3f})',
        color=C['title'], fontsize=12, fontweight='bold', **MONO,
    )
    return fig
