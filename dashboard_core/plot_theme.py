"""
Shared matplotlib styling primitives: palette, axis styling, and the
time-series plot helpers used across the panel builders.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor). _series_plot/_series_enhanced/_energy_enhanced's
former redundancy was unified into a single _series_plot (Phase 2) --
same visual output, one parameterized function instead of three.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

C = dict(
    bg      = '#03050e',
    panel   = '#070b18',
    grid    = '#0f1628',
    border  = '#18243c',
    title   = '#e2eaf8',
    label   = '#4a6080',
    tick    = '#2e4a6a',
    prob    = '#38bdf8',
    energy  = '#00e0ff',
    entropy = '#f43f7a',
    purity  = '#4ade80',
    grad    = '#fbbf24',
    noise   = '#fb923c',
    theta   = '#a78bfa',
    dom     = '#ffffff',
    accent  = '#00ff9d',
    warn    = '#ff6b35',
)

MONO   = {'fontfamily': 'monospace'}
FILL_A = 0.08

_cmap_prob = LinearSegmentedColormap.from_list(
    'tureq_prob', ['#0a1f3d', '#38bdf8', '#f43f7a'], N=256
)

BG_ART    = '#030305'
PANEL_ART = '#07070a'
CYAN_N    = '#00f3ff'
PINK_N    = '#ff0055'
PURP_N    = '#7a00ff'
GOLD_N    = '#ffaa00'

# Repeated across build_panel_fisica/vqe_results/md_results/performance's
# plt.figure(facecolor=...) -- distinct from C['bg'] (a different literal
# in the original), centralized here (Phase 2) as one named constant
# instead of 5 duplicated string literals in panels.py.
FIG_BG_DARK = '#010409'


def _ax_style(ax, title='', xlabel='', ylabel='', spine_alpha=0.6):
    ax.set_facecolor(C['panel'])
    for sp in ax.spines.values():
        sp.set_edgecolor(C['border'])
        sp.set_linewidth(0.7)
        sp.set_alpha(spine_alpha)
    ax.tick_params(colors=C['tick'], which='both', length=3, width=0.5, labelsize=7.5)
    ax.xaxis.label.set_color(C['label']); ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_color(C['label']); ax.yaxis.label.set_fontsize(8)
    ax.grid(True, ls='--', lw=0.30, alpha=0.25, color=C['grid'])
    if title:
        ax.set_title(title, color=C['title'], fontsize=9.5, fontweight='bold',
                     pad=6, loc='left', **MONO)
    if xlabel: ax.set_xlabel(xlabel, **MONO)
    if ylabel: ax.set_ylabel(ylabel, **MONO)


def _fill(ax, x, y, col):
    ax.fill_between(x, y, alpha=FILL_A, color=col)


def _rolling(y, win):
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values


def _badge(ax, text, color):
    """Top-right value badge."""
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=color, **MONO,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=C['bg'], edgecolor=C['border'], alpha=0.72))


def _series_plot(ax, df, col, color, title, xlabel, ylabel, *,
                  show_raw=True, badge=False, badge_fmt='{:.4g}',
                  ref_line=None, detect_plateau=False,
                  mark_convergence=False):
    """Unified time-series: raw line + fill + rolling mean + last-value
    annotation, with optional badge / reference-line / barren-plateau-span /
    convergence-marker overlays.

    Unifies what used to be three near-duplicate functions (_series_plot,
    which this absorbs, plus _series_enhanced and _energy_enhanced, Phase 2
    of the dashboard refactor) into one parameterized function -- same
    visual output as each of the three original call shapes:
      - old _series_plot(...)                    == _series_plot(...) (defaults)
      - old _series_enhanced(..., ref_line=X)     == _series_plot(..., badge=True, ref_line=X)
      - old _energy_enhanced(ax, df)              == _series_plot(ax, df, 'VQE_Energy',
                                                       C['energy'], 'VQE Energy', 'Epoch',
                                                       'E  (Ha)', badge=True,
                                                       badge_fmt='Eₙ={:.4g} Ha',
                                                       mark_convergence=True)
    """
    _ax_style(ax, title, xlabel, ylabel)
    if df.empty or col not in df.columns:
        ax.axis('off')
        ax.text(0.5, 0.5, f'[ {col} — no data ]',
                ha='center', va='center', color=C['label'],
                fontsize=8.5, transform=ax.transAxes, **MONO)
        return
    x = df.index.values
    y = df[col].values
    if show_raw:
        ax.plot(x, y, color=color, lw=1.4, alpha=0.7)
    _fill(ax, x, y, color)
    win = max(3, len(y) // 15)
    rm  = _rolling(y, win)
    ax.plot(x, rm, color=C['title'], lw=1.1, alpha=0.6, ls='--',
            label='rolling mean')
    ax.annotate(f'{y[-1]:.4g}',
                xy=(x[-1], y[-1]), xytext=(-4, 6),
                textcoords='offset points',
                color=color, fontsize=7.5, fontweight='bold',
                ha='right', **MONO)

    x_f = df.index.to_numpy(dtype=float)

    if badge:
        _badge(ax, badge_fmt.format(y[-1]), color)

    if ref_line is not None:
        ax.axhline(ref_line, color=color, lw=0.6, ls='--', alpha=0.28)

    if detect_plateau:
        _thr   = 0.01 * np.abs(y).max() if np.abs(y).max() > 0 else 1e-9
        _bp    = np.abs(y) < _thr
        if _bp.sum() > 3:
            _edges = np.where(np.diff(_bp.astype(int)))[0]
            if len(_edges) >= 2:
                ax.axvspan(x_f[_edges[0]], x_f[_edges[1]],
                           alpha=0.11, color=C['warn'])
                ax.text(
                    (x_f[_edges[0]] + x_f[_edges[1]]) / 2,
                    y.max() * 0.88,
                    'plateau', ha='center', fontsize=6.5,
                    color=C['warn'], alpha=0.80, **MONO,
                )

    if mark_convergence and len(y) > 2:
        _conv = int(np.argmin(np.gradient(y)))
        ax.axvline(x_f[_conv], color=C['warn'], lw=0.8, ls=':', alpha=0.55)
        ax.annotate(
            f'∇min@{_conv}',
            xy=(x_f[_conv], y[_conv]),
            xytext=(6, -14), textcoords='offset points',
            color=C['warn'], fontsize=7.0,
            arrowprops=dict(arrowstyle='-', color=C['warn'], lw=0.5),
            **MONO,
        )


def _interp_colour(c1_hex, c2_hex, t):
    t = float(np.clip(t, 0, 1))
    def h(s): return [int(s.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4)]
    r1, g1, b1 = h(c1_hex)
    r2, g2, b2 = h(c2_hex)
    to_hex = lambda v: f'{int(v*255):02x}'
    return f'#{to_hex(r1+(r2-r1)*t)}{to_hex(g1+(g2-g1)*t)}{to_hex(b1+(b2-b1)*t)}'


def _noise_profile_plot(ax, noise_model, noise_p, n_qubits,
                        prob_ideal, prob_noisy=None):
    _ax_style(ax, 'Noise Analysis', '', '')
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1.08)
    ax.axis('off')

    model_col = C['warn'] if noise_model != 'ideal' else C['accent']
    ax.text(0.5, 0.97, noise_model.upper().replace('_', ' '),
            ha='center', va='top', color=model_col,
            fontsize=16, fontweight='bold', transform=ax.transAxes, **MONO)

    bar_y  = 0.80
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, bar_y - 0.025), 0.90, 0.05,
        boxstyle='round,pad=0.005',
        facecolor=C['grid'], edgecolor=C['border'], lw=0.8,
        transform=ax.transAxes, zorder=2))
    fill_w = 0.90 * min(noise_p / 0.10, 1.0)
    if fill_w > 0:
        fill_col = _interp_colour('#00ff9d', '#ff6b35', noise_p / 0.10)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, bar_y - 0.025), fill_w, 0.05,
            boxstyle='round,pad=0.005',
            facecolor=fill_col, edgecolor='none',
            transform=ax.transAxes, zorder=3))
    ax.text(0.5, bar_y + 0.06, f'p = {noise_p:.4f}',
            ha='center', va='bottom', color=C['title'],
            fontsize=11, fontweight='bold', transform=ax.transAxes, **MONO)
    ax.text(0.05, bar_y - 0.07, '0', ha='left', va='top',
            color=C['label'], fontsize=7.5, transform=ax.transAxes, **MONO)
    ax.text(0.95, bar_y - 0.07, '0.10', ha='right', va='top',
            color=C['label'], fontsize=7.5, transform=ax.transAxes, **MONO)

    if prob_noisy is not None and noise_model != 'ideal':
        fid_bc = float(np.sum(np.sqrt(np.maximum(prob_ideal, 0) *
                                      np.maximum(prob_noisy,  0))))
        tvd    = float(0.5 * np.sum(np.abs(prob_ideal - prob_noisy)))
        rows = [
            ('Bhattacharyya Fidelity',  f'{fid_bc:.6f}', C['purity']),
            ('Total Variation Distance', f'{tvd:.6f}',   C['entropy']),
            ('Noise Channel', noise_model.replace('_', ' '), C['noise']),
        ]
    else:
        rows = [
            ('Channel', 'ideal — no noise applied', C['accent']),
            ('Fidelity', '1.000000', C['purity']),
            ('TVD', '0.000000', C['label']),
        ]
    for j, (k, v, col) in enumerate(rows):
        yy = 0.58 - j * 0.14
        ax.text(0.03, yy, k, ha='left',  va='center', color=C['label'],
                fontsize=8, transform=ax.transAxes, **MONO)
        ax.text(0.97, yy, v, ha='right', va='center', color=col,
                fontsize=8.5, fontweight='bold',
                transform=ax.transAxes, **MONO)
        ax.axhline(yy - 0.05, xmin=0.01, xmax=0.99, color=C['grid'], lw=0.35)

    if noise_model in ('ideal', 'depolarizing'):
        ins = ax.inset_axes([0.04, 0.04, 0.92, 0.26])
        ins.set_facecolor(C['panel'])
        for sp in ins.spines.values():
            sp.set_edgecolor(C['border']); sp.set_linewidth(0.5)
        ps = np.linspace(0, 0.1, 200)
        d  = 2 ** n_qubits
        fid_curve = ((1.0 - ps * (d - 1) / d) ** n_qubits).clip(0, 1)
        ins.plot(ps, fid_curve, color=C['purity'], lw=1.2)
        ins.axvline(noise_p, color=C['warn'], lw=0.9, ls='--', alpha=0.8)
        ins.fill_between(ps, fid_curve, alpha=0.07, color=C['purity'])
        ins.set_xlim(0, 0.10); ins.set_ylim(0, 1.05)
        ins.tick_params(colors=C['tick'], labelsize=6.5, length=2)
        ins.set_xlabel('p',    color=C['label'], fontsize=7, **MONO)
        ins.set_ylabel('F(p)', color=C['label'], fontsize=7, **MONO)
        ins.set_title('Theoretical Fidelity Curve',
                      color=C['label'], fontsize=7, pad=2, **MONO)
        ins.grid(True, ls=':', lw=0.3, alpha=0.25, color=C['grid'])
