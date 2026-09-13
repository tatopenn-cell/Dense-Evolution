"""
Every visualization here is native (plain matplotlib/numpy) -- no Qiskit
anywhere in this module. The circuit diagram was already native
(dashboard_core.circuit_diagram); histogram/Q-sphere/Bloch spheres are
now native too (dashboard_core.state_visuals), replacing
qiskit.visualization.{plot_histogram, plot_state_qsphere,
plot_bloch_multivector} -- the last Qiskit call sites this dashboard had.
Qiskit's macOS instability was never proven scoped to just
QuantumCircuit (see circuit_diagram.py's own docstring), so making these
three merely "optional" instead of actually removing Qiskit would have
left the same open question for a different set of functions; there's no
reason left to keep it as a dependency here at all.
"""

import matplotlib.pyplot as plt

from .circuit_diagram import draw_native_circuit_diagram
from .state_visuals import native_histogram_figure, native_qsphere_figure, native_bloch_multivector_figure

__all__ = [
    'draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure',
    'energy_landscape_figure', 'zne_bar_figure', 'qec_syndrome_map_figure',
]

# dense_evolution.registry's dark_background styling used to be a bare
# import-time side effect -- fixed (prog.txt, core audit point 2, PR
# #256) into an explicit, opt-in apply_dark_theme(), so it no longer
# fires just from importing dense_evolution. Nothing in this dashboard
# calls it today, but plt.style.use() is a genuinely global, process-wide
# mutation, so the wrapper below isn't guarding a stale concern (prog.txt,
# dashboard_core audit point 5a) -- it's still real protection: rendering
# inside style.context('default') pins every figure here to matplotlib's
# light default regardless of whatever ELSE in the same process (a future
# caller of apply_dark_theme(), another library, a user's own
# matplotlibrc) may have changed globally.
_LIGHT_STYLE = 'default'


def draw_circuit_figure(ops, n_qubits: int):
    """Native matplotlib circuit diagram (dashboard_core.circuit_diagram).

    Examples
    --------
    >>> from dashboard_core.visuals import draw_circuit_figure
    >>> fig = draw_circuit_figure([('h', 0), ('cx', 0, 1)], n_qubits=2)
    >>> type(fig).__name__
    'Figure'
    """
    with plt.style.context(_LIGHT_STYLE):
        return draw_native_circuit_diagram(ops, n_qubits)


def histogram_figure(counts: dict, statevector=None):
    """Native shot-count histogram (dashboard_core.state_visuals).
    statevector (optional): color bars by amplitude phase instead of a
    flat color -- see native_histogram_figure's own docstring."""
    with plt.style.context(_LIGHT_STYLE):
        return native_histogram_figure(counts, statevector=statevector)


def qsphere_figure(statevector):
    """Native Q-sphere (dashboard_core.state_visuals) -- the multi-qubit
    generalization of the Bloch sphere used by IBM Quantum Composer."""
    with plt.style.context(_LIGHT_STYLE):
        return native_qsphere_figure(statevector)


def bloch_multivector_figure(statevector):
    """Native per-qubit Bloch spheres (dashboard_core.state_visuals)."""
    with plt.style.context(_LIGHT_STYLE):
        return native_bloch_multivector_figure(statevector)


def energy_landscape_figure(
    values_i, values_j, energies, param_i, param_j,
    converged_i, converged_j, converged_energy,
):
    """Contour plot of a real VQE energy landscape scan (dashboard_core.
    vqe.scan_hardware_efficient_energy_landscape) over two parameters,
    with the converged optimum marked at its actual (converged_i,
    converged_j) coordinates -- not assumed to sit at the grid's center,
    since the caller decides the scan range independently."""
    with plt.style.context(_LIGHT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        contour = ax.contourf(values_i, values_j, energies.T, levels=30, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='Energia (Hartree)')
        ax.scatter(
            [converged_i], [converged_j],
            color='red', marker='o', s=60, label=f'Minimo VQE ({converged_energy:.6f} Hartree)',
        )
        ax.set_xlabel(f'θ[{param_i}]')
        ax.set_ylabel(f'θ[{param_j}]')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        return fig


def zne_bar_figure(noise_factors, noisy_expectations, noisy_sems, zne_extrapolated, ideal_expectation):
    """Real ZNE measurement chart with a +-1 SEM (standard error of the
    mean, not the raw per-trial spread) band around each measured point
    -- the Monte Carlo uncertainty on run_zne_mitigation's own
    n_trials-averaged expectation values, not a decorative estimate."""
    with plt.style.context(_LIGHT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(
            noise_factors, noisy_expectations, yerr=noisy_sems,
            fmt='o-', color='#1f77b4', capsize=4, label='Misurato (media Monte Carlo ±1 SEM)',
        )
        ax.axhline(ideal_expectation, color='green', linestyle='--', label=f'Ideale ({ideal_expectation:.4f})')
        ax.scatter([0], [zne_extrapolated], color='red', marker='*', s=150, zorder=5,
                   label=f'Estrapolato a 0 ({zne_extrapolated:.4f})')
        ax.set_xlabel('Fattore di scala del rumore')
        ax.set_ylabel('<P> misurato')
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        return fig


def qec_syndrome_map_figure(stabilizers, syndrome, pauli_error=None):
    """Real syndrome-measurement map: physical qubits laid out in a line,
    each stabilizer drawn as a bracket connecting the qubits it acts on
    (its own non-identity Pauli positions), colored red if its syndrome
    bit is 1 (anticommutes / detected, from dense_evolution's own
    compute_syndrome output) or gray if 0 (commutes / undetected) --
    not a schematic, the real per-stabilizer outcome. Qubits touched by
    pauli_error (if given) are marked filled and labeled with their
    Pauli letter."""
    with plt.style.context(_LIGHT_STYLE):
        n_qubits = len(stabilizers[0]) if stabilizers else (len(pauli_error) if pauli_error else 1)
        error_positions = {i for i, p in enumerate(pauli_error) if p != 'I'} if pauli_error else set()

        fig_height = 2.2 + 0.4 * max(1, len(stabilizers))
        fig, ax = plt.subplots(figsize=(max(4, n_qubits * 1.2), fig_height))

        for i in range(n_qubits):
            if i in error_positions:
                ax.scatter([i], [0], s=350, color='#d62728', edgecolors='black', zorder=3)
                ax.annotate(pauli_error[i], (i, 0), ha='center', va='center', color='white',
                            fontsize=9, fontweight='bold', zorder=4)
            else:
                ax.scatter([i], [0], s=350, color='white', edgecolors='black', zorder=3)
            ax.annotate(f'q{i}', (i, -0.35), ha='center', fontsize=8)

        for s_idx, (stab, bit) in enumerate(zip(stabilizers, syndrome)):
            positions = [i for i, p in enumerate(stab) if p != 'I']
            if not positions:
                continue
            height = 0.5 + 0.4 * s_idx
            color = '#d62728' if bit else '#999999'
            lo, hi = min(positions), max(positions)
            ax.plot([lo, lo, hi, hi], [0.15, height, height, 0.15], color=color, linewidth=2)
            ax.annotate(stab, ((lo + hi) / 2, height + 0.05), ha='center', fontsize=8, color=color)

        ax.set_xlim(-0.7, n_qubits - 0.3)
        ax.set_ylim(-0.6, fig_height - 1.9)
        ax.axis('off')
        fig.tight_layout()
        return fig
