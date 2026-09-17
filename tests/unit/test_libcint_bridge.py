"""
Correctness tests for native_hf.libcint_bridge, the optional PySCF/libcint
ERI tensor bridge (see its own module docstring, and prog.txt, for why it
exists: ~0.04s vs 158-532s for native_hf's own JAX path on Ne/6-31G*).

The end-to-end energy tests were first verified on Google Colab (this
sandbox can't install PySCF locally), then reproduced here so a real
pyscf install in CI keeps covering the actual bridge logic, not just the
ImportError path -- same reasoning as this repo's stim/pymatching tests.
`pytest.importorskip` is per-test, not module-level, so the pure-Python
shell-matching/permutation tests below still run without pyscf installed.
"""
import jax
import numpy as np
import pytest

from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.libcint_bridge import (
    _match_pyscf_shell_ao_starts,
    _permutation_libcint_to_native_hf,
)

_BOHR_PER_ANGSTROM = 1.8897259886
NE_631GSTAR_PYSCF_RHF_ENERGY = -128.47440651990485


class _FakeShell:
    def __init__(self, atom_index, degree, exponents):
        self.atom_index = atom_index
        self.degree = degree
        self.exponents = np.asarray(exponents)


class _FakeMol:
    """Duck-typed stand-in for a PySCF Mole -- only the four accessors
    _match_pyscf_shell_ao_starts actually calls, so these error-path tests
    don't need PySCF installed at all."""

    def __init__(self, bas):
        self._bas = bas  # list of (atom, degree, exponents)
        sizes = [len(cartesian_powers(d)) for _, d, _ in bas]
        self._ao_loc = np.concatenate([[0], np.cumsum(sizes)])

    @property
    def nbas(self):
        return len(self._bas)

    def bas_atom(self, ib):
        return self._bas[ib][0]

    def bas_angular(self, ib):
        return self._bas[ib][1]

    def bas_exp(self, ib):
        return self._bas[ib][2]

    def ao_loc_nr(self):
        return self._ao_loc


def test_permutation_unimplemented_degree_raises():
    with pytest.raises(NotImplementedError):
        _permutation_libcint_to_native_hf(3)


def test_shell_matching_raises_on_real_disagreement():
    mol = _FakeMol([(0, 0, np.array([1.0, 0.5]))])
    shells = [_FakeShell(atom_index=0, degree=1, exponents=[1.0, 0.5])]
    with pytest.raises(ValueError):
        _match_pyscf_shell_ao_starts(mol, shells)


def test_shell_matching_finds_reordered_shells():
    # Same pattern actually seen on Ne/6-31G*: native_hf keeps s,p,s
    # order, the fake "libcint" mol groups them s,s,p.
    mol = _FakeMol([(0, 0, np.array([6.0])), (0, 0, np.array([1.0])), (0, 1, np.array([1.0]))])
    shells = [
        _FakeShell(atom_index=0, degree=0, exponents=[6.0]),
        _FakeShell(atom_index=0, degree=1, exponents=[1.0]),
        _FakeShell(atom_index=0, degree=0, exponents=[1.0]),
    ]
    starts = _match_pyscf_shell_ao_starts(mol, shells)
    assert starts == [0, 2, 1]


@pytest.fixture
def _x64():
    # Function-scoped, not module-level: restores the previous value after
    # this one test so it doesn't leak into other test files that run in
    # the same pytest process.
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_h2_sto3g_bridge_matches_native_hf_eri(_x64):
    pytest.importorskip("pyscf")
    from dense_evolution.native_hf.basis import build_molecule_shells
    from dense_evolution.native_hf.assembly import build_repulsion_tensor
    from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint

    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]) * _BOHR_PER_ANGSTROM
    shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")

    V_native = build_repulsion_tensor(shells)
    V_bridge = build_repulsion_tensor_libcint([1, 1], geometry_bohr, "sto-3g")

    assert V_bridge == pytest.approx(V_native, abs=1e-8)


def test_h2_sto3g_overlap_core_libcint_matches_native_hf(_x64):
    pytest.importorskip("pyscf")
    from dense_evolution.native_hf.basis import build_molecule_shells
    from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian
    from dense_evolution.native_hf.libcint_bridge import build_overlap_and_core_hamiltonian_libcint

    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]) * _BOHR_PER_ANGSTROM
    shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")

    S_native = build_overlap_matrix(shells)
    H_core_native = build_core_hamiltonian(shells, [1.0, 1.0], geometry_bohr)
    S_bridge, H_core_bridge = build_overlap_and_core_hamiltonian_libcint([1, 1], geometry_bohr, "sto-3g")

    assert S_bridge == pytest.approx(S_native, abs=1e-8)
    assert H_core_bridge == pytest.approx(H_core_native, abs=1e-8)


def test_ne_631gstar_fully_libcint_pipeline_matches_anchor(_x64):
    # Same mixed s/p/d anchor as test_ne_631gstar_bridge_matches_pyscf_anchor,
    # but S/H_core ALSO come from the libcint bridge now (not just the ERI
    # tensor) -- the real point found and fixed via a live Kaggle CPU kernel:
    # native_hf's own one-electron assembly ran out of memory during XLA JIT
    # compilation on a larger real molecule at this same basis, even with the
    # ERI tensor already libcint-backed. This proves the one-electron bridge
    # reproduces the identical converged energy, not just plausible-looking
    # matrices.
    pytest.importorskip("pyscf")
    from dense_evolution.native_hf.libcint_bridge import (
        build_overlap_and_core_hamiltonian_libcint, build_repulsion_tensor_libcint,
    )
    from dense_evolution.native_hf.scf import run_scf

    geometry_bohr = np.array([[0.0, 0.0, 0.0]])
    S, H_core = build_overlap_and_core_hamiltonian_libcint([10], geometry_bohr, "6-31g*")
    V = build_repulsion_tensor_libcint([10], geometry_bohr, "6-31g*")

    result = run_scf(S, H_core, V, 10, [10.0], geometry_bohr)
    assert result.converged
    assert result.total_energy == pytest.approx(NE_631GSTAR_PYSCF_RHF_ENERGY, abs=1e-6)


def test_run_scf_converges_without_caller_enabling_x64_first():
    # The real bug this guards: run_scf's own default tolerances (1e-10)
    # are unreachable in JAX's default float32 mode (machine epsilon
    # ~1.19e-7), so without run_scf enabling x64 itself, `converged` comes
    # back False even for a trivially-converged H2/STO-3G case whose
    # total_energy is already numerically correct -- found via a real
    # Kaggle run whose own convergence gate then nulled out a valid energy.
    # Deliberately does NOT use the `_x64` fixture: starts from whatever
    # jax_enable_x64 already is (commonly False), to prove run_scf fixes
    # this on its own rather than relying on the caller.
    pytest.importorskip("pyscf")
    from dense_evolution.native_hf.libcint_bridge import (
        build_overlap_and_core_hamiltonian_libcint, build_repulsion_tensor_libcint,
    )
    from dense_evolution.native_hf.scf import run_scf

    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]) * _BOHR_PER_ANGSTROM
    S, H_core = build_overlap_and_core_hamiltonian_libcint([1, 1], geometry_bohr, "sto-3g")
    V = build_repulsion_tensor_libcint([1, 1], geometry_bohr, "sto-3g")

    result = run_scf(S, H_core, V, 2, [1.0, 1.0], geometry_bohr)
    assert result.converged
    assert result.n_iterations < 200


def test_ne_631gstar_bridge_matches_pyscf_anchor(_x64):
    # The real point: a mixed s/p/d basis, where native_hf and libcint
    # disagree on both shell order (native_hf keeps basis-file order,
    # libcint groups by ascending degree) and per-component d normalization
    # -- exercises the shell-identity matching and rescale in
    # libcint_bridge.py, not just a trivial pass-through.
    pytest.importorskip("pyscf")
    from dense_evolution.native_hf.basis import build_molecule_shells
    from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian
    from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint
    from dense_evolution.native_hf.scf import run_scf

    geometry_bohr = np.array([[0.0, 0.0, 0.0]])
    shells = build_molecule_shells([10], geometry_bohr, "6-31g*")

    S = build_overlap_matrix(shells)
    H_core = build_core_hamiltonian(shells, [10.0], geometry_bohr)
    V_bridge = build_repulsion_tensor_libcint([10], geometry_bohr, "6-31g*")

    assert V_bridge == pytest.approx(V_bridge.transpose(1, 0, 2, 3), abs=1e-9)
    assert V_bridge == pytest.approx(V_bridge.transpose(0, 1, 3, 2), abs=1e-9)
    assert V_bridge == pytest.approx(V_bridge.transpose(2, 3, 0, 1), abs=1e-9)

    result = run_scf(S, H_core, V_bridge, 10, [10.0], geometry_bohr)
    assert result.converged
    assert result.total_energy == pytest.approx(NE_631GSTAR_PYSCF_RHF_ENERGY, abs=1e-6)
