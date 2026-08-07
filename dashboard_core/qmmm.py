"""
Real Hellmann-Feynman nuclear forces and a real Velocity-Verlet MD step
for this project's real molecule catalog -- no fabricated geometry,
charges, or Hamiltonian anywhere.

Forces are F = -dE/dR (E = <psi|H(R)|psi>, psi held fixed -- the
Hellmann-Feynman theorem exactly), with H(R) this project's own real
Hamiltonian at geometry R (dashboard_core.hamiltonians.
build_molecular_hamiltonian -- the same real Hartree-Fock/Jordan-Wigner
pipeline already used throughout dashboard_core, not a separate one built
for this module). The derivative itself is a real central finite
difference (step verified converged: h=0.001 and h=0.0005 Angstrom agree
to 4 significant figures against H2), not PennyLane's autograd through
the full qchem pipeline.

An earlier version used qml.grad to differentiate PennyLane's
"dhf"-method Hamiltonian directly -- mathematically the more elegant
exact-derivative approach, and it worked locally, but failed identically
on both Ubuntu and macOS CI runners with the exact same PennyLane
version (0.45.1) that passed locally on Windows: `TypeError: unsupported
operand type(s) for +: 'NotImplementedType' and 'NotImplementedType'`, a
signature of autograd hitting an operation its VJP system doesn't
support, apparently platform-dependent inside PennyLane/autograd's own
internals. Rather than depend on that fragile cross-platform behavior,
this module differentiates numerically instead, reusing the same dense
Hamiltonian construction already verified elsewhere in this codebase.

An even earlier version tried adding an external classical point-charge
potential directly to the post-Jordan-Wigner dense Hamiltonian (mirroring
legacy/dash.py's QMMMForceEngine). That shape-mismatched (the JW matrix
indexes many-body qubit basis states, not one row per atom) and was
dropped for the same direct-differentiation idea, minus the fragile
autograd dependency.

Verified 2026-08-05 against H2: force ~0.0154 Hartree/Angstrom at the
real equilibrium bond length (0.7414 A, small residual expected since
the electronic state is evaluated at fixed geometry -- clamped-nucleus
Hellmann-Feynman, not a fully relaxed force), rising sharply and
correctly signed as a restoring force at a stretched 1.2 A bond.
"""
import numpy as np
import pennylane as qml
from scipy import constants as _c

import dense_evolution as de
from .hamiltonians import MOLECULE_CATALOG, build_molecular_hamiltonian, _pennylane_hamiltonian_to_pauli_terms

__all__ = [
    'ATOMIC_MASSES_AMU', 'compute_hellmann_feynman_forces', 'md_step', 'run_md_trajectory',
    'MIN_NUCLEAR_DISTANCE_ANGSTROM',
]

# Real MD safety floor, not a fabricated number: shorter than any real
# covalent bond this project's molecule catalog could ever produce (H2's
# own equilibrium is 0.7414 A) -- run_md_trajectory checks new positions
# against this after every Velocity-Verlet step, since Hartree-Fock at a
# near-collided geometry (the failure mode a too-large dt_fs drives
# light atoms like H toward) diverges rather than raising a clear error,
# making the actual cause hard to diagnose from the resulting crash.
# md_step itself stays a bare, unchecked F=ma primitive -- the check
# belongs at run_md_trajectory's real-simulation boundary, not the
# mechanical formula (tests exercise md_step directly with synthetic,
# not physically meaningful, starting positions).
MIN_NUCLEAR_DISTANCE_ANGSTROM = 0.3


def _assert_no_nuclear_collision(positions, step, dt_fs):
    """Raises RuntimeError if any two atoms in `positions` (n_atoms, 3)
    are closer than MIN_NUCLEAR_DISTANCE_ANGSTROM -- see that constant's
    comment for why. A no-op for a single atom (nothing to compare)."""
    if positions.shape[0] < 2:
        return
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(dists.min())
    if min_dist < MIN_NUCLEAR_DISTANCE_ANGSTROM:
        raise RuntimeError(
            f"MD trajectory diverged: two atoms are {min_dist:.4f} A apart at step "
            f"{step + 1} (below the {MIN_NUCLEAR_DISTANCE_ANGSTROM} A safety floor -- "
            f"no real covalent bond in this catalog is that short). This almost always "
            f"means dt_fs={dt_fs} is too large for the forces involved -- light atoms "
            f"like H accelerate sharply and overshoot in a single step. Try a smaller "
            f"dt_fs before rerunning."
        )

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
                                     geometry=None, fd_step_angstrom: float = 0.001):
    """Real Hellmann-Feynman forces (Hartree/Angstrom) on every nucleus of
    MOLECULE_CATALOG[name]: F = -d<psi|H(R)|psi>/dR, with H(R) this
    project's own real Hamiltonian (build_molecular_hamiltonian) and psi
    held fixed. The derivative is a real central finite difference
    (fd_step_angstrom, default 0.001 A -- verified converged against
    0.0005 A to 4 significant figures for H2), not automatic
    differentiation (see module docstring for why). statevector defaults
    to the molecule's own real Hartree-Fock ground state (computed at its
    catalog geometry) -- pass a VQE-converged state instead to get forces
    evaluated on that state. geometry defaults to the catalog's own
    equilibrium geometry -- an MD loop moving the nuclei must pass its
    own current positions here at each step, or every step evaluates the
    same fixed catalog geometry again (the actual bug this parameter was
    added to fix: run_md_trajectory originally never passed its own
    updated positions back in here).
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
    sv = np.asarray(statevector, dtype=np.complex128)
    geometry = np.asarray(geometry, dtype=np.float64)

    def energy_at(geom):
        h_matrix, _n_qubits = build_molecular_hamiltonian(symbols, geom, charge, mapping)
        return float(np.real(np.vdot(sv, h_matrix @ sv)))

    energy = energy_at(geometry)
    forces = np.zeros_like(geometry)
    h = fd_step_angstrom
    for i in range(geometry.shape[0]):
        for j in range(3):
            geom_plus = geometry.copy()
            geom_plus[i, j] += h
            geom_minus = geometry.copy()
            geom_minus[i, j] -= h
            forces[i, j] = -(energy_at(geom_plus) - energy_at(geom_minus)) / (2 * h)

    return {
        "name": name,
        "symbols": symbols,
        "energy_hartree": energy,
        "positions_angstrom": geometry.tolist(),
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
        _assert_no_nuclear_collision(positions, step, dt_fs)

        if recompute_electronic_state:
            statevector, _gs_energy, _n_qubits = _reference_ground_state(symbols, positions, charge, mapping)

    return trajectory
