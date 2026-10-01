"""
Correctness tests for native_hf.libcint_bridge, which calls the libcint
shared library bundled in the dense-evolution wheels directly through
ctypes (~0.04s vs 158-532s for native_hf's own JAX path on Ne/6-31G*).
The integral tests skip when no libcint library is present (source
install); .github/workflows/libcint.yml builds the library for Windows,
macOS and Linux and runs them against the installed wheel.
"""
import jax
import numpy as np
import pytest

from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.libcint_bridge import (
    _permutation_libcint_to_native_hf,
    load_libcint,
)

_BOHR_PER_ANGSTROM = 1.8897259886
NE_631GSTAR_PYSCF_RHF_ENERGY = -128.47440651990485


def _require_libcint():
    try:
        load_libcint()
    except ImportError as exc:
        pytest.skip(str(exc))


def test_permutation_unimplemented_degree_raises():
    with pytest.raises(NotImplementedError):
        _permutation_libcint_to_native_hf(3)


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
    _require_libcint()
    from dense_evolution.native_hf.basis import build_molecule_shells
    from dense_evolution.native_hf.assembly import build_repulsion_tensor
    from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint

    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]) * _BOHR_PER_ANGSTROM
    shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")

    V_native = build_repulsion_tensor(shells)
    V_bridge = build_repulsion_tensor_libcint([1, 1], geometry_bohr, "sto-3g")

    assert V_bridge == pytest.approx(V_native, abs=1e-8)


def test_h2_sto3g_overlap_core_libcint_matches_native_hf(_x64):
    _require_libcint()
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
    _require_libcint()
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
    _require_libcint()
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
    # A mixed s/p/d basis, where native_hf and libcint disagree on
    # Cartesian component order and per-component d normalization --
    # exercises the permutation and rescale in libcint_bridge.py.
    _require_libcint()
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
