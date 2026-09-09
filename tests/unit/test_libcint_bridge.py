"""
Correctness tests for native_hf.libcint_bridge, the optional PySCF/libcint
ERI tensor bridge (see its own module docstring, and prog.txt, for why it
exists: ~0.04s vs 158-532s for native_hf's own JAX path on Ne/6-31G*).

Both cases here were first verified on Google Colab (this sandbox can't
install PySCF locally), then reproduced here so a real pyscf install in
CI keeps covering the actual bridge logic, not just the ImportError path
-- same reasoning as this repo's stim/pymatching tests.
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

import jax
jax.config.update("jax_enable_x64", True)

from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor
from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint
from dense_evolution.native_hf.scf import run_scf

_BOHR_PER_ANGSTROM = 1.8897259886
NE_631GSTAR_PYSCF_RHF_ENERGY = -128.47440651990485


def test_h2_sto3g_bridge_matches_native_hf_eri():
    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]) * _BOHR_PER_ANGSTROM
    shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")

    V_native = build_repulsion_tensor(shells)
    V_bridge = build_repulsion_tensor_libcint([1, 1], geometry_bohr, "sto-3g")

    assert V_bridge == pytest.approx(V_native, abs=1e-8)


def test_ne_631gstar_bridge_matches_pyscf_anchor():
    # The real point: a mixed s/p/d basis, where native_hf and libcint
    # disagree on both shell order (native_hf keeps basis-file order,
    # libcint groups by ascending degree) and per-component d normalization
    # -- exercises the shell-identity matching and rescale in
    # libcint_bridge.py, not just a trivial pass-through.
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
