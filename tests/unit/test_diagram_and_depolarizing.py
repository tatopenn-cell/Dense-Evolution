import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from dense_evolution import plot_circuit, global_depolarizing_channel


def test_plot_circuit_returns_figure_with_single_and_multi_qubit_gates():
    fig = plot_circuit([('x', 1), ('rz', 0, 0.5), ('iswap', 0, 1)], n_qubits=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_global_depolarizing_channel_preserves_trace():
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    out = global_depolarizing_channel(rho, 0.3)
    assert abs(complex(np.trace(out)) - 1.0) < 1e-9


def test_global_depolarizing_channel_at_p_zero_is_identity_map():
    rho = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
    out = global_depolarizing_channel(rho, 0.0)
    assert np.allclose(out, rho)


def test_global_depolarizing_channel_at_p_one_is_maximally_mixed():
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    out = global_depolarizing_channel(rho, 1.0)
    assert np.allclose(out, np.eye(4) / 4)
