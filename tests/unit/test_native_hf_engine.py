"""Direct numerical tests for the native_hf integral engine and SCF loop
(prog.txt P1) -- the only prior coverage was test_native_hf_bridge.py's
96 lines on the AO->MO transform; the integral engine (boys, overlap,
kinetic, coulomb, assembly) and the SCF loop itself had no direct
numerical test at all. Values below were verified correct on the current
code before being frozen here (see prog.txt), not invented.
"""
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from scipy.special import hyp1f1

from dense_evolution.native_hf.boys import boys
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import (
    build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor,
)
from dense_evolution.native_hf.scf import run_scf

_BOHR_PER_ANGSTROM = 1.8897259886


def _linear_two_atom_geometry_bohr(bond_length_angstrom):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length_angstrom]]) * _BOHR_PER_ANGSTROM


def _water_geometry_bohr(bond_length_angstrom, angle_degrees):
    half_angle = np.radians(angle_degrees) / 2.0
    r = bond_length_angstrom
    geometry_angstrom = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(half_angle), 0.0, r * np.cos(half_angle)],
        [-r * np.sin(half_angle), 0.0, r * np.cos(half_angle)],
    ])
    return geometry_angstrom * _BOHR_PER_ANGSTROM


class TestBoys:

    @pytest.mark.parametrize("n", [0, 2, 4, 6])
    @pytest.mark.parametrize("x", [0.0, 1e-13, 1e-10, 1e-6, 1e-2, 1.0, 10.0, 50.0, 200.0])
    def test_boys_matches_reference(self, n, x):
        reference = hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)
        value = float(boys(n, x))
        assert value == pytest.approx(reference, rel=1e-12, abs=1e-300)


class TestOverlapMatrix:

    def test_h2_sto3g_overlap_matrix(self):
        geometry_bohr = _linear_two_atom_geometry_bohr(0.735)
        shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")
        S = build_overlap_matrix(shells)

        assert S.shape == (2, 2)
        assert np.allclose(np.diag(S), 1.0, atol=1e-10)
        assert S == pytest.approx(S.T, abs=1e-12)
        assert S[0, 1] == pytest.approx(0.6631457753793818, abs=1e-8)


class TestScfEnergies:

    def test_scf_h2_sto3g(self):
        geometry_bohr = _linear_two_atom_geometry_bohr(0.735)
        shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, [1.0, 1.0], geometry_bohr)
        repulsion = build_repulsion_tensor(shells)
        result = run_scf(S, H_core, repulsion, 2, [1.0, 1.0], geometry_bohr)

        assert result.converged is True
        assert result.total_energy == pytest.approx(-1.116999, abs=1e-6)

    def test_scf_h2o_sto3g(self):
        geometry_bohr = _water_geometry_bohr(0.9584, 104.5)
        shells = build_molecule_shells([8, 1, 1], geometry_bohr, "sto-3g")
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, [8.0, 1.0, 1.0], geometry_bohr)
        repulsion = build_repulsion_tensor(shells)
        result = run_scf(S, H_core, repulsion, 10, [8.0, 1.0, 1.0], geometry_bohr)

        assert result.converged is True
        assert result.total_energy == pytest.approx(-74.96310432637894, abs=1e-6)

    def test_scf_converges_on_near_degenerate(self):
        # The real Si2/STO-3G near-degenerate-orbital case that motivated
        # DIIS (dense_evolution/native_hf/scf.py's own module docstring):
        # plain undamped density substitution never converged here (100/100
        # iterations, still oscillating between two numerically-degenerate
        # orbital pairs); DIIS converges in 11 iterations.
        geometry_bohr = _linear_two_atom_geometry_bohr(2.184)
        shells = build_molecule_shells([14, 14], geometry_bohr, "sto-3g")
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, [14.0, 14.0], geometry_bohr)
        repulsion = build_repulsion_tensor(shells)
        result = run_scf(S, H_core, repulsion, 28, [14.0, 14.0], geometry_bohr)

        assert result.converged is True
        assert result.n_iterations < 20


class TestRepulsionTensorSymmetry:

    def test_h2_sto3g_repulsion_tensor_eight_fold_symmetry(self):
        geometry_bohr = _linear_two_atom_geometry_bohr(0.735)
        shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")
        V = build_repulsion_tensor(shells)

        assert V == pytest.approx(V.transpose(1, 0, 2, 3), abs=1e-12)
        assert V == pytest.approx(V.transpose(0, 1, 3, 2), abs=1e-12)
        assert V == pytest.approx(V.transpose(2, 3, 0, 1), abs=1e-12)
        assert V == pytest.approx(V.transpose(3, 2, 1, 0), abs=1e-12)
