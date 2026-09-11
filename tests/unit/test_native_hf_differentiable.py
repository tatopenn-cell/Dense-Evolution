import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from dense_evolution.native_hf.differentiable import build_energy_fn
from dense_evolution.native_hf.scf import nuclear_repulsion_energy

_BOHR_PER_ANGSTROM = 1.8897259886


@pytest.fixture(autouse=True, scope="module")
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _h2_geometry(bond_length_bohr):
    return jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length_bohr]])


def test_h2_sto3g_gradient_matches_central_finite_difference():
    d0 = 0.735 * _BOHR_PER_ANGSTROM
    energy_fn = build_energy_fn(
        atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0], n_electrons=2,
        basis_name="sto-3g", reference_geometry_bohr=_h2_geometry(d0),
    )

    def energy_of_bond_length(d):
        return energy_fn(_h2_geometry(d))

    grad_analytic = float(jax.grad(energy_of_bond_length)(d0))

    h = 1e-5
    grad_fd = float((energy_of_bond_length(d0 + h) - energy_of_bond_length(d0 - h)) / (2 * h))

    assert grad_analytic == pytest.approx(grad_fd, abs=1e-8)


def test_nuclear_repulsion_gradient_matches_central_finite_difference():
    d0 = 0.735 * _BOHR_PER_ANGSTROM

    def e_nuc(d):
        return nuclear_repulsion_energy([1.0, 1.0], _h2_geometry(d))

    grad_analytic = float(jax.grad(e_nuc)(d0))

    h = 1e-5
    grad_fd = float((e_nuc(d0 + h) - e_nuc(d0 - h)) / (2 * h))

    assert grad_analytic == pytest.approx(grad_fd, abs=1e-9)


def test_energy_fn_matches_run_scf_total_energy():
    from dense_evolution.native_hf.basis import build_molecule_shells
    from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor
    from dense_evolution.native_hf.scf import run_scf

    d0 = 0.735 * _BOHR_PER_ANGSTROM
    geometry = _h2_geometry(d0)

    energy_fn = build_energy_fn(
        atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0], n_electrons=2,
        basis_name="sto-3g", reference_geometry_bohr=geometry,
    )
    energy = float(energy_fn(geometry))

    shells = build_molecule_shells([1, 1], np.asarray(geometry), "sto-3g")
    S = build_overlap_matrix(shells)
    H_core = build_core_hamiltonian(shells, [1.0, 1.0], geometry)
    repulsion = build_repulsion_tensor(shells)
    result = run_scf(S, H_core, repulsion, 2, [1.0, 1.0], geometry)

    assert energy == pytest.approx(result.total_energy, abs=1e-9)
