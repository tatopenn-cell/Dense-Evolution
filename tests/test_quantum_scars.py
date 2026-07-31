"""
Unit tests for ui_pages/quantum_scars.py's numerical core (Hamiltonian
construction, exact-diagonalization propagation, and the two projection
strategies). Plain pytest against the module's private helpers directly --
no separate core module like dashboard_core.py exists for this page.
AppTest-level UI smoke testing was already done manually and is not
duplicated here.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from ui_pages import quantum_scars as qs

N_QUBITS = 6  # dim=64, diagonalizes instantly


@pytest.fixture(scope="module")
def pxp():
    return qs._build_pxp(N_QUBITS)


def _random_state(dim, seed):
    rng = np.random.default_rng(seed)
    sv = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return sv / np.linalg.norm(sv)


# ── _is_valid_state ─────────────────────────────────────────────────────

def test_is_valid_state_rejects_adjacent_ones():
    # 0b110000 has two adjacent 1-bits -> invalid
    assert qs._is_valid_state(0b110000, N_QUBITS) is False


def test_is_valid_state_accepts_neel_pattern():
    # 0b010101 alternates, never two adjacent 1-bits -> valid
    assert qs._is_valid_state(0b010101, N_QUBITS) is True


def test_is_valid_state_all_zero_is_valid():
    assert qs._is_valid_state(0, N_QUBITS) is True


def test_is_valid_state_matches_independent_bruteforce():
    # Cross-check every state against an independent string-based
    # implementation, unrelated to _build_pxp's own usage of the function.
    for i in range(2 ** N_QUBITS):
        expected = "11" not in format(i, f"0{N_QUBITS}b")
        assert qs._is_valid_state(i, N_QUBITS) == expected


# ── _build_pxp ───────────────────────────────────────────────────────────

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


# ── _propagate ───────────────────────────────────────────────────────────

def test_propagate_fidelity_one_at_t_zero(pxp):
    neel = np.zeros(pxp["dim"], dtype=np.complex128)
    neel[pxp["neel_idx"]] = 1.0
    propagated = qs._propagate(pxp, neel, dt=0.0)
    fidelity = np.abs(np.vdot(neel, propagated)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_propagate_preserves_norm(pxp):
    neel = np.zeros(pxp["dim"], dtype=np.complex128)
    neel[pxp["neel_idx"]] = 1.0
    sv = neel.copy()
    for _ in range(20):
        sv = qs._propagate(pxp, sv, dt=0.2)
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-9)


# ── _project_tower ───────────────────────────────────────────────────────

def test_project_tower_stays_normalized(pxp):
    sv = _random_state(pxp["dim"], seed=1)
    projected = qs._project_tower(pxp, sv)
    assert np.linalg.norm(projected) == pytest.approx(1.0, abs=1e-9)


def test_project_tower_zero_weight_outside_subspace(pxp):
    sv = _random_state(pxp["dim"], seed=2)
    projected = qs._project_tower(pxp, sv)
    weight_in_tower = np.sum(np.abs(pxp["tower_vectors"].conj().T @ projected) ** 2)
    assert weight_in_tower == pytest.approx(1.0, abs=1e-9)


def test_project_tower_orthogonal_state_returns_near_zero(pxp):
    # Build a state supported only on eigenvectors NOT in the tower ->
    # exercises the norm < 1e-12 safe-return branch without dividing by zero.
    tower_set = set(np.argsort(
        np.abs(pxp["eigenvectors"][pxp["neel_idx"], :]) ** 2
    )[::-1][: N_QUBITS + 1].tolist())
    non_tower_idx = next(i for i in range(pxp["dim"]) if i not in tower_set)
    sv = pxp["eigenvectors"][:, non_tower_idx]
    projected = qs._project_tower(pxp, sv)
    assert np.linalg.norm(projected) < 1e-9


# ── _project_constraint ──────────────────────────────────────────────────

def test_project_constraint_stays_normalized(pxp):
    sv = _random_state(pxp["dim"], seed=3)
    projected = qs._project_constraint(pxp, sv)
    assert np.linalg.norm(projected) == pytest.approx(1.0, abs=1e-9)


def test_project_constraint_zero_weight_on_invalid_states(pxp):
    sv = _random_state(pxp["dim"], seed=4)
    projected = qs._project_constraint(pxp, sv)
    assert np.allclose(projected[~pxp["valid_mask"]], 0.0)


# ── _invalid_weight ───────────────────────────────────────────────────────

def test_invalid_weight_neel_state_is_zero(pxp):
    neel = np.zeros(pxp["dim"], dtype=np.complex128)
    neel[pxp["neel_idx"]] = 1.0
    assert qs._invalid_weight(pxp, neel) == pytest.approx(0.0, abs=1e-12)


def test_invalid_weight_uniform_state_matches_complement_of_valid_dim(pxp):
    uniform = np.ones(pxp["dim"], dtype=np.complex128) / np.sqrt(pxp["dim"])
    expected = 1.0 - pxp["valid_dim"] / pxp["dim"]
    assert qs._invalid_weight(pxp, uniform) == pytest.approx(expected, abs=1e-9)
