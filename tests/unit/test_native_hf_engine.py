"""Direct numerical tests for the native_hf integral engine and SCF loop
(prog.txt P1) -- the only prior coverage was test_native_hf_bridge.py's
96 lines on the AO->MO transform; the integral engine (boys, overlap,
kinetic, coulomb, assembly) and the SCF loop itself had no direct
numerical test at all. Values below were verified correct on the current
code before being frozen here (see prog.txt), not invented.

Two different kinds of assertion live in this file, worth telling apart:
several are anchored against something external to this codebase --
boys(n,x) vs scipy's own hyp1f1, the H2/STO-3G and H2O/STO-3G SCF
energies against real literature reference values (see prog.txt's own
evidence table), the ERI tensor's 8-fold permutational symmetry
(a physical invariant true of any correct implementation, not a
property of this one), and -- added for d-shell (angular momentum L=2)
support -- the d-orbital overlap integral against independent scipy
quadrature, the d-orbital kinetic integral against an exact sympy
symbolic Laplacian, and the dxy/dxx normalization ratio against the
closed-form sqrt(3) result every Cartesian-Gaussian implementation has
to get right. These verify correctness: a wrong answer here means a
wrong answer, full stop.

The other 37 assertions (the frozen overlap-matrix entry, the frozen
Si2 iteration-count bound, etc.) are auto-generated -- frozen from
whatever this code currently outputs, without independent verification
against a known-correct external reference. These block regressions
(a future change that shifts the number gets caught), but a systematic
error already present when a value was frozen would be locked in, not
revealed. If a future audit finds one of the 37 wrong, that is real
information about a real bug -- it does not mean this test suite lied,
it means this suite was never able to catch that particular error in
the first place.
"""
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from scipy.special import hyp1f1

from dense_evolution.native_hf.boys import boys
from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.assembly import (
    build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor,
)
from dense_evolution.native_hf.scf import run_scf
from dense_evolution.native_hf.gaussians import GaussianShell3D
from dense_evolution.native_hf.overlap import overlap_3d
from dense_evolution.native_hf.kinetic import kinetic_3d
from dense_evolution.native_hf.cartesian import cartesian_powers, cartesian_normalization_ratios

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


# ── d-shell (angular momentum L=2) support (prog.txt) ────────────────────
# Cartesian d shells have 6 components (dxx,dyy,dzz,dxy,dxz,dyz); unlike
# s/p, they do NOT all share the same normalization constant (dxx and dxy
# have different self-overlap). These tests check the new per-component
# normalization ratio, and the underlying overlap/kinetic recursions
# themselves, against sources external to native_hf entirely (scipy
# quadrature, sympy exact symbolic integration) -- not just self-
# consistency within the library. A full molecular SCF run on a real
# d-containing basis (RHF/6-31G* on neon) is too slow to run on every
# push (~10 minutes, see prog.txt) and lives instead as an experiment in
# Dense-Evolution-Discovery.

class TestDShellNormalization:

    def test_cartesian_powers_count_matches_degree_formula(self):
        for degree in range(4):
            assert len(cartesian_powers(degree)) == (degree + 1) * (degree + 2) // 2

    @pytest.mark.parametrize("exponent", [0.3, 0.9, 5.7])
    def test_d_shell_normalization_ratio_matches_overlap_self_consistency(self, exponent):
        g = GaussianShell3D(degree=2, exponent=jnp.asarray(exponent), center=jnp.zeros(3))
        S = overlap_3d(g, g)
        powers = cartesian_powers(2)
        ref_self_overlap = float(S[tuple(powers[0]) + tuple(powers[0])])
        measured = np.array([
            np.sqrt(ref_self_overlap / float(S[lx, ly, lz, lx, ly, lz]))
            for lx, ly, lz in powers
        ])
        assert measured == pytest.approx(cartesian_normalization_ratios(2), abs=1e-12)

    def test_dxy_over_dxx_normalization_ratio_is_sqrt3(self):
        # The one number every Cartesian-Gaussian implementation gets
        # asked about: dxy needs a sqrt(3) larger normalization constant
        # than dxx, because <dxx|dxx> = 3*<dxy|dxy> for the same exponent.
        ratios = cartesian_normalization_ratios(2)
        powers = cartesian_powers(2).tolist()
        dxx_ratio = ratios[powers.index([2, 0, 0])]
        dxy_ratio = ratios[powers.index([1, 1, 0])]
        assert dxy_ratio / dxx_ratio == pytest.approx(np.sqrt(3.0), abs=1e-12)


class TestDShellIntegralsAgainstExternalReferences:

    def test_d_overlap_matches_independent_scipy_quadrature(self):
        from scipy import integrate

        a = 0.9
        g = GaussianShell3D(degree=2, exponent=jnp.asarray(a), center=jnp.zeros(3))
        S = overlap_3d(g, g)

        Ix4, _ = integrate.quad(lambda x: x**4 * np.exp(-2 * a * x * x), -np.inf, np.inf)
        Ix2, _ = integrate.quad(lambda x: x**2 * np.exp(-2 * a * x * x), -np.inf, np.inf)
        I0, _ = integrate.quad(lambda x: np.exp(-2 * a * x * x), -np.inf, np.inf)

        dxx_quad = Ix4 * I0 * I0
        dxy_quad = Ix2 * Ix2 * I0

        assert float(S[2, 0, 0, 2, 0, 0]) == pytest.approx(dxx_quad, rel=1e-10)
        assert float(S[1, 1, 0, 1, 1, 0]) == pytest.approx(dxy_quad, rel=1e-10)

    def test_d_kinetic_matches_independent_sympy_exact_integration(self):
        sp = pytest.importorskip("sympy")

        x, y, z, a_sym = sp.symbols("x y z a", positive=True, real=True)
        gaussian = sp.exp(-a_sym * (x**2 + y**2 + z**2))
        dxx_expr = x**2 * gaussian
        dxy_expr = x * y * gaussian

        def laplacian_expectation(f_expr):
            lap = sp.diff(f_expr, x, 2) + sp.diff(f_expr, y, 2) + sp.diff(f_expr, z, 2)
            val = sp.integrate(f_expr * lap, (x, -sp.oo, sp.oo))
            val = sp.integrate(val, (y, -sp.oo, sp.oo))
            return sp.integrate(val, (z, -sp.oo, sp.oo))

        a_val = 0.9
        dxx_exact = float(laplacian_expectation(dxx_expr).subs(a_sym, a_val))
        dxy_exact = float(laplacian_expectation(dxy_expr).subs(a_sym, a_val))

        g = GaussianShell3D(degree=2, exponent=jnp.asarray(a_val), center=jnp.zeros(3))
        T = kinetic_3d(g, g)

        assert float(T[2, 0, 0, 2, 0, 0]) == pytest.approx(dxx_exact, rel=1e-10)
        assert float(T[1, 1, 0, 1, 1, 0]) == pytest.approx(dxy_exact, rel=1e-10)


class TestDShellBasisLoading:

    def test_631gstar_oxygen_yields_a_real_d_shell(self):
        geometry_bohr = np.array([[0.0, 0.0, 0.0]])
        shells = build_molecule_shells([8], geometry_bohr, "6-31g*")
        degrees = sorted({s.degree for s in shells})
        assert 2 in degrees
        assert n_cartesian_functions(shells) == sum(len(cartesian_powers(s.degree)) for s in shells)

    def test_core_hamiltonian_symmetric_with_a_real_d_shell(self):
        # Fast (no ERI): overlap + core Hamiltonian only, on the smallest
        # real d-containing basis available -- the full electron-repulsion
        # tensor for a mixed s/p/d basis is a separate, slow check (see
        # the Dense-Evolution-Discovery experiment referenced above).
        geometry_bohr = np.array([[0.0, 0.0, 0.0]])
        shells = build_molecule_shells([10], geometry_bohr, "6-31g*")
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, [10.0], geometry_bohr)

        assert S == pytest.approx(S.T, abs=1e-9)
        assert np.allclose(np.diag(S), 1.0, atol=1e-8)
        assert H_core == pytest.approx(H_core.T, abs=1e-9)
