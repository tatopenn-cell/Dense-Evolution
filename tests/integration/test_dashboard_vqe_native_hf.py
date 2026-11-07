"""run_vqe on an element outside PennyLane's STO-3G table: native_hf builds the Hamiltonian."""
import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("basis_set_exchange")

from dashboard_core import hamiltonians as ham  # noqa: E402
from dashboard_core.vqe import run_vqe  # noqa: E402

SI2 = [name for name in ham.get_all_molecules() if name.startswith("Si2")][0]


def test_the_catalog_keeps_the_active_space_of_each_molecule():
    molecules = ham.get_all_molecules()
    assert molecules[SI2]["active_electrons"] == 4
    assert molecules[SI2]["active_orbitals"] == 4
    assert molecules[SI2]["n_qubits"] == 8
    h2 = [m for name, m in molecules.items() if name.startswith("H2 ")][0]
    assert h2["active_electrons"] is None


def test_the_electron_count_of_an_element_that_pennylane_does_not_bundle():
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.184]])
    assert ham._electron_count(["Si", "Si"], geometry, 0) == 28
    assert ham._electron_count(["Si", "Si"], geometry, 1) == 27
    assert ham._electron_count(["H", "H"], np.array([[0.0, 0, 0], [0, 0, 0.74]]), 0) == 2


@pytest.mark.slow
def test_vqe_on_the_silicon_dimer_runs_on_native_hf_with_the_active_space():
    molecule = ham.get_all_molecules()[SI2]
    result = run_vqe(
        molecule["symbols"],
        molecule["geometry"],
        molecule["charge"],
        n_layers=1,
        maxiter=3,
        active_electrons=molecule["active_electrons"],
        active_orbitals=molecule["active_orbitals"],
    )
    assert np.isfinite(result["vqe_energy_hartree"])
    assert int(np.sum(result["hf_occupation"])) == 4
    assert len(result["hf_occupation"]) == 8
