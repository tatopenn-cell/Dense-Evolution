"""
Unit tests for dashboard_core.scars_engine -- the public-API duplication of
ui_pages/quantum_scars.py's private PXP helpers, added so
launch_interactive_panel's ipywidgets Quantum Scars panel doesn't depend on
the repo-only ui_pages package (see the module's own docstring). Mirrors
the properties already checked against the private originals in
test_quantum_scars.py, against this module's public functions instead.
"""

import numpy as np
import pytest

from dashboard_core import scars_engine as se

N_QUBITS = 6  # dim=64, diagonalizes instantly


@pytest.fixture(scope="module")
def pxp():
    return se.build_pxp(N_QUBITS)


def test_build_pxp_hilbert_dimension(pxp):
    assert pxp["dim"] == 2 ** N_QUBITS
    assert pxp["eigenvalues"].shape == (pxp["dim"],)
    assert pxp["eigenvectors"].shape == (pxp["dim"], pxp["dim"])


def test_build_pxp_hamiltonian_is_hermitian(pxp):
    h = pxp["h_pxp"]
    np.testing.assert_allclose(h, h.conj().T, atol=1e-10)


def test_build_pxp_valid_dim_bounded(pxp):
    assert 0 < pxp["valid_dim"] <= pxp["dim"]


def test_build_pxp_tower_ceiling_bounded(pxp):
    assert 0 < pxp["tower_ceiling"] <= 1 + 1e-9


def test_build_pxp_is_cached(pxp):
    # Same n_qubits -> the exact same dict object, not recomputed.
    assert se.build_pxp(N_QUBITS) is pxp


def test_build_pxp_different_n_qubits_not_cached_together():
    other = se.build_pxp(N_QUBITS + 1)
    assert other["n_qubits"] == N_QUBITS + 1
    assert other["dim"] == 2 ** (N_QUBITS + 1)


# ── run_experiment ───────────────────────────────────────────────────────

def test_run_experiment_no_noise_no_protection_shape_and_bounds(pxp):
    fidelity = se.run_experiment(
        pxp, n_trajectories=1, noise_p=0.0, protection="Nessuna",
        weight_threshold=0.02, base_seed=0,
    )
    assert fidelity.shape == (se.N_CHUNK,)
    assert np.all(fidelity >= -1e-9)
    assert np.all(fidelity <= 1 + 1e-9)


def test_run_experiment_is_deterministic_given_seed(pxp):
    a = se.run_experiment(pxp, n_trajectories=2, noise_p=0.02, protection="Nessuna",
                           weight_threshold=0.02, base_seed=7)
    b = se.run_experiment(pxp, n_trajectories=2, noise_p=0.02, protection="Nessuna",
                           weight_threshold=0.02, base_seed=7)
    np.testing.assert_array_equal(a, b)


def test_run_experiment_protection_modes_do_not_raise(pxp):
    for protection in ("Nessuna", "Proiezione vincolo (economica)", "Proiezione torre (ideale)"):
        fidelity = se.run_experiment(
            pxp, n_trajectories=1, noise_p=0.02, protection=protection,
            weight_threshold=0.02, base_seed=1,
        )
        assert fidelity.shape == (se.N_CHUNK,)
