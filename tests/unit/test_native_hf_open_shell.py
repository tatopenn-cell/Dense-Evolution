"""Open-shell SCF: run_uhf and run_cuhf (constrained UHF = ROHF,
Tsuchimochi & Scuseria, arXiv:1008.1607).

Anchors: closed-shell UHF and CUHF must reproduce RHF; CUHF must give the
exact <S^2> = S_z(S_z+1) of a restricted open-shell determinant; the OH
doublet and O2 triplet CUHF energies were checked against an independent
implementation of the paper's equations (agreement 2e-12 Ha) before being
frozen here.
"""
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from dense_evolution.native_hf.assembly import (
    build_core_hamiltonian,
    build_overlap_matrix,
    build_repulsion_tensor,
)
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.scf import run_cuhf, run_scf, run_uhf


def _molecule(Z, geometry):
    g = np.array(geometry, dtype=float)
    q = [float(z) for z in Z]
    shells = build_molecule_shells(Z, g, "sto-3g")
    return (build_overlap_matrix(shells), build_core_hamiltonian(shells, q, g),
            build_repulsion_tensor(shells), q, g)


@pytest.fixture(scope="module")
def h2o():
    return _molecule([8, 1, 1], [[0, 0, 0.2217], [0, 1.4309, -0.8867], [0, -1.4309, -0.8867]])


@pytest.fixture(scope="module")
def oh():
    return _molecule([8, 1], [[0, 0, 0], [0, 0, 1.83]])


def test_closed_shell_uhf_and_cuhf_equal_rhf(h2o):
    S, H, V, q, g = h2o
    e_rhf = run_scf(S, H, V, 10, q, g).total_energy
    uhf = run_uhf(S, H, V, 10, q, g, n_unpaired=0)
    cuhf = run_cuhf(S, H, V, 10, q, g, n_unpaired=0)
    assert uhf.total_energy == pytest.approx(e_rhf, abs=1e-9)
    assert cuhf.total_energy == pytest.approx(e_rhf, abs=1e-9)
    assert uhf.spin_squared == pytest.approx(0.0, abs=1e-9)
    assert cuhf.spin_squared == pytest.approx(0.0, abs=1e-9)


def test_oh_doublet_uhf_is_contaminated_cuhf_is_not(oh):
    S, H, V, q, g = oh
    uhf = run_uhf(S, H, V, 9, q, g)
    cuhf = run_cuhf(S, H, V, 9, q, g)
    assert uhf.converged and cuhf.converged
    assert (uhf.n_alpha, uhf.n_beta) == (5, 4)
    assert uhf.spin_squared > 0.75 + 1e-4
    assert cuhf.spin_squared == pytest.approx(0.75, abs=1e-9)
    assert cuhf.total_energy == pytest.approx(-74.3613919722, abs=1e-8)
    assert uhf.total_energy < cuhf.total_energy


def test_o2_triplet_cuhf_energy_and_spin():
    S, H, V, q, g = _molecule([8, 8], [[0, 0, 0], [0, 0, 2.28]])
    cuhf = run_cuhf(S, H, V, 16, q, g, n_unpaired=2)
    assert cuhf.converged
    assert cuhf.spin_squared == pytest.approx(2.0, abs=1e-9)
    assert cuhf.total_energy == pytest.approx(-147.3711494748, abs=1e-8)


def test_warm_start_reproduces_the_converged_state(oh):
    S, H, V, q, g = oh
    for run in (run_uhf, run_cuhf):
        first = run(S, H, V, 9, q, g)
        again = run(S, H, V, 9, q, g, C_alpha_init=first.orbital_coefficients_alpha,
                    C_beta_init=first.orbital_coefficients_beta)
        assert again.total_energy == pytest.approx(first.total_energy, abs=1e-9)
        assert again.n_iterations <= first.n_iterations


def test_energy_history_is_nan_padded(oh):
    S, H, V, q, g = oh
    res = run_cuhf(S, H, V, 9, q, g, max_iterations=80)
    assert res.energy_history.shape == (80,)
    assert np.all(np.isnan(np.asarray(res.energy_history)[res.n_iterations:]))


@pytest.mark.parametrize("run", [run_uhf, run_cuhf])
@pytest.mark.parametrize("n_electrons, n_unpaired", [(9, 0), (2, 3)])
def test_invalid_electron_counts_raise(oh, run, n_electrons, n_unpaired):
    S, H, V, q, g = oh
    with pytest.raises(ValueError):
        run(S, H, V, n_electrons, q, g, n_unpaired=n_unpaired)
