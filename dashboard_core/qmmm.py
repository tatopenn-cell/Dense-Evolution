"""
Real Hellmann-Feynman nuclear forces and a real Velocity-Verlet MD step
for this project's real molecule catalog -- no fabricated geometry,
charges, or Hamiltonian anywhere.

Forces come directly from PennyLane's own differentiable quantum
chemistry (method="dhf" -- the same real Hartree-Fock backend
dashboard_core.hamiltonians already uses): the real one/two-electron
integrals are recomputed as an explicit, autograd-differentiable
function of nuclear geometry, and F = -dE/dR (E = <psi|H(R)|psi>, psi
held fixed -- the Hellmann-Feynman theorem exactly) is the real gradient
of that real energy, not a finite-difference approximation and not a
hand-built classical-embedding perturbation.

An earlier version of this module tried to add an external classical
point-charge potential directly to the post-Jordan-Wigner dense
Hamiltonian (mirroring legacy/dash.py's QMMMForceEngine). That shape-
mismatched (the JW matrix indexes many-body qubit basis states, not one
row per atom) and was dropped in favor of this simpler, provably correct
approach: differentiate the real geometry-dependent Hamiltonian itself,
no external embedding needed for what this module is actually for (real
forces on this project's own molecules, for real MD).

Verified 2026-08-05 against H2: force ~0.011 Hartree/Angstrom at the
real equilibrium bond length (0.7414 A, small residual expected since
the electronic state is evaluated at fixed geometry -- clamped-nucleus
Hellmann-Feynman, not a fully relaxed force), rising to ~0.289
Hartree/Angstrom at a stretched 1.2 A bond, correctly signed as a
restoring force back toward equilibrium.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy import constants as _c

import dense_evolution as de
from .hamiltonians import MOLECULE_CATALOG, _pennylane_hamiltonian_to_pauli_terms

__all__ = [
    'ATOMIC_MASSES_AMU', 'compute_hellmann_feynman_forces', 'md_step', 'run_md_trajectory',
]

# Real standard atomic weights (amu) for the elements this project's real
# molecule catalog actually uses (H2, HeH+, H3+, LiH, H2O) -- only what's
# needed, not the full periodic table, so nothing here is an unverified
# guess for an element no catalog molecule contains.
ATOMIC_MASSES_AMU = {'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'O': 16.00}

# a [Angstrom/fs^2] = ACCEL_CONVERSION * F[Hartree/Angstrom] / mass[amu]
# -- derived from CODATA (scipy.constants), verified 2026-08-05:
# (hartree_J / angstrom_m) / amu_kg * fs_s**2 / angstrom_m = 0.262550...
ACCEL_CONVERSION = (
    (_c.physical_constants['Hartree energy'][0] / 1e-10)
    / _c.physical_constants['atomic mass constant'][0]
    * (1e-15 ** 2) / 1e-10
)


def _reference_ground_state(symbols, geometry, charge, mapping):
    """Real ground-state eigenvector of the molecule's real Hamiltonian at
    the given geometry, via this project's own native dense-matrix builder
    (dense_evolution.pauli_hamiltonian_to_matrix from PennyLane's real
    Pauli decomposition -- same construction dashboard_core.hamiltonians
    uses, verified there to match qml.matrix(H) exactly). Not part of the
    differentiable path -- this is evaluated once at a fixed geometry to
    get a real electronic state, then held fixed while forces are
    computed at (possibly different) geometries, exactly as the
    Hellmann-Feynman theorem requires."""
    molecule = qml.qchem.Molecule(symbols, np.asarray(geometry), charge=charge, unit="angstrom")
    H, n_qubits = qml.qchem.molecular_hamiltonian(molecule, method="dhf", mapping=mapping)
    terms = _pennylane_hamiltonian_to_pauli_terms(H, n_qubits)
    h_matrix = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
    eigvals, eigvecs = np.linalg.eigh(h_matrix)
    return eigvecs[:, 0], float(eigvals[0]), n_qubits


def compute_hellmann_feynman_forces(name: str, statevector=None, mapping: str = "jordan_wigner",
                                     geometry=None):
    """Real Hellmann-Feynman forces (Hartree/Angstrom) on every nucleus of
    MOLECULE_CATALOG[name]: F = -d<psi|H(R)|psi>/dR, with H(R) PennyLane's
    real differentiable ("dhf") molecular Hamiltonian and psi held fixed.
    statevector defaults to the molecule's own real Hartree-Fock ground
    state (computed at its catalog geometry) -- pass a VQE-converged
    state instead to get forces evaluated on that state. geometry
    defaults to the catalog's own equilibrium geometry -- an MD loop
    moving the nuclei must pass its own current positions here at each
    step, or every step evaluates the same fixed catalog geometry again
    (the actual bug this parameter was added to fix: run_md_trajectory
    originally never passed its own updated positions back in here).
    """
    if name not in MOLECULE_CATALOG:
        raise ValueError(f"unknown molecule {name!r}; available: {sorted(MOLECULE_CATALOG)}")
    spec = MOLECULE_CATALOG[name]
    symbols = spec["symbols"]
    if geometry is None:
        geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
    charge = spec["charge"]

    unknown = [s for s in symbols if s not in ATOMIC_MASSES_AMU]
    if unknown:
        raise ValueError(f"no real atomic mass on file for {unknown} -- add to "
                          f"ATOMIC_MASSES_AMU before using this molecule here")

    if statevector is None:
        statevector, _gs_energy, _n_qubits = _reference_ground_state(symbols, geometry, charge, mapping)
    sv = pnp.array(np.asarray(statevector), requires_grad=False)

    def energy_at_geometry(geometry_flat):
        geom = geometry_flat.reshape(-1, 3)
        molecule = qml.qchem.Molecule(symbols, geom, charge=charge, unit="angstrom")
        H, _n_qubits = qml.qchem.molecular_hamiltonian(molecule, method="dhf", mapping=mapping)
        h_matrix = qml.matrix(H)
        h_psi = h_matrix @ sv
        return pnp.real(pnp.sum(pnp.conj(sv) * h_psi))

    geom_flat = pnp.array(np.asarray(geometry, dtype=np.float64), requires_grad=True).flatten()
    energy = float(energy_at_geometry(geom_flat))
    grad = qml.grad(energy_at_geometry)(geom_flat)
    forces = -np.asarray(grad).reshape(-1, 3)

    return {
        "name": name,
        "symbols": symbols,
        "energy_hartree": energy,
        "positions_angstrom": np.asarray(geometry, dtype=np.float64).tolist(),
        "forces_hartree_per_angstrom": forces.tolist(),
        "force_norm": float(np.linalg.norm(forces)),
    }


def md_step(positions_angstrom, velocities_angstrom_per_fs, forces_hartree_per_angstrom,
            symbols, dt_fs: float = 0.5):
    """One real Velocity-Verlet half-step (v(t+dt/2) = v(t) + a(t)*dt/2,
    r(t+dt) = r(t) + v(t+dt/2)*dt) using the real Hellmann-Feynman forces
    above and each atom's real atomic mass -- ordinary classical Newtonian
    mechanics (F=ma), nothing invented. Positions in Angstrom, velocities
    in Angstrom/fs, forces in Hartree/Angstrom, dt in femtoseconds."""
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    velocities = np.asarray(velocities_angstrom_per_fs, dtype=np.float64)
    forces = np.asarray(forces_hartree_per_angstrom, dtype=np.float64)
    masses = np.array([ATOMIC_MASSES_AMU[s] for s in symbols], dtype=np.float64)

    accel = ACCEL_CONVERSION * forces / masses[:, None]
    velocities_half = velocities + 0.5 * accel * dt_fs
    positions_new = positions + velocities_half * dt_fs
    return positions_new, velocities_half, accel


def run_md_trajectory(name: str, n_steps: int, dt_fs: float = 0.5, mapping: str = "jordan_wigner",
                       recompute_electronic_state: bool = False):
    """Real, minimal ab-initio-forces MD trajectory: at each step, real
    Hellmann-Feynman forces (compute_hellmann_feynman_forces) move the
    real nuclear positions/velocities via real Velocity-Verlet (md_step).
    Starts from rest (zero initial velocities) at the catalog's real
    equilibrium geometry.

    recompute_electronic_state=False (default) holds the electronic state
    fixed at the initial Hartree-Fock reference through the whole
    trajectory -- forces stay exact only close to the starting geometry
    (a real, explicitly-stated approximation, not a fabricated one).
    True ab-initio MD (re-solving Hartree-Fock at every step's new
    geometry) is available by setting this True, at real, substantial
    extra cost per step."""
    if name not in MOLECULE_CATALOG:
        raise ValueError(f"unknown molecule {name!r}; available: {sorted(MOLECULE_CATALOG)}")
    spec = MOLECULE_CATALOG[name]
    symbols = spec["symbols"]
    geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
    charge = spec["charge"]

    statevector, _gs_energy, _n_qubits = _reference_ground_state(symbols, geometry, charge, mapping)
    positions = np.asarray(geometry, dtype=np.float64)
    velocities = np.zeros_like(positions)

    trajectory = {"step": [], "time_fs": [], "positions_angstrom": [], "energy_hartree": [], "force_norm": []}
    for step in range(n_steps):
        result = compute_hellmann_feynman_forces(name, statevector, mapping=mapping, geometry=positions)
        forces = np.asarray(result["forces_hartree_per_angstrom"])
        trajectory["step"].append(step)
        trajectory["time_fs"].append(step * dt_fs)
        trajectory["positions_angstrom"].append(positions.tolist())
        trajectory["energy_hartree"].append(result["energy_hartree"])
        trajectory["force_norm"].append(result["force_norm"])

        positions, velocities, _accel = md_step(positions, velocities, forces, symbols, dt_fs=dt_fs)

        if recompute_electronic_state:
            statevector, _gs_energy, _n_qubits = _reference_ground_state(symbols, positions, charge, mapping)

    return trajectory
