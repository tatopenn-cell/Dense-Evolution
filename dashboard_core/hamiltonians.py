"""
Real molecular Hamiltonians, built on demand from actual atomic geometry
via PennyLane's qchem module (Hartree-Fock + Jordan-Wigner fermion-to-
qubit mapping, method='dhf' -- native to PennyLane, no PySCF/OpenFermion
dependency needed).

Ported from feature/streamlit-dashboard (git branch), where this was
built and verified against known values before the dashboard rebuild.
Only the real molecular catalog comes along here -- the old diagonal
"toy model" library and the VQE optimization loop stay out for now
(kept minimal on purpose, brought back separately if/when needed).
"""

import numpy as np

__all__ = [
    'MOLECULE_CATALOG', 'build_molecular_hamiltonian',
    'get_compatible_molecules', 'get_all_molecules', 'get_molecule_n_qubits',
    'get_molecular_hamiltonian_matrix', 'ground_state_energy',
    'linear_chain_geometry', 'ring_geometry',
]


def linear_chain_geometry(n_atoms: int, bond_length_angstrom: float):
    """N atoms on a line, each bond_length_angstrom apart -- the real,
    general shape behind every diatomic entry in the catalog (H2, HeH+,
    LiH), extended to any atom count."""
    if n_atoms < 1:
        raise ValueError("linear_chain_geometry needs at least 1 atom")
    return np.array([[0.0, 0.0, i * bond_length_angstrom] for i in range(n_atoms)])


def ring_geometry(n_atoms: int, bond_length_angstrom: float):
    """N atoms on a regular polygon (equal bond_length_angstrom between
    neighbors), circumradius R = bond_length / (2*sin(pi/n)) -- standard
    regular-polygon geometry. At n_atoms=3 this is exactly an equilateral
    triangle -- the same real D3h geometry H3+'s catalog entry uses, just
    generalized to any ring size (still only meaningful up to whatever
    qubit count this simulator's exact diagonalization / VQE range can
    handle -- this function itself has no such limit, the caller does)."""
    if n_atoms < 3:
        raise ValueError("ring_geometry needs at least 3 atoms")
    R = bond_length_angstrom / (2 * np.sin(np.pi / n_atoms))
    angles = 2 * np.pi * np.arange(n_atoms) / n_atoms
    return np.array([[R * np.cos(a), R * np.sin(a), 0.0] for a in angles])


def _triangular_h3_geometry(bond_length_angstrom: float):
    """H3+'s real, published equilateral-triangle (D3h) ground-state
    geometry -- the n_atoms=3 case of ring_geometry."""
    return ring_geometry(3, bond_length_angstrom)


def _linear_two_atom_geometry(bond_length_angstrom: float):
    return linear_chain_geometry(2, bond_length_angstrom)


# Bond lengths are real, published equilibrium geometries. LiH needs no
# active-space reduction (its full STO-3G Hamiltonian is already exactly
# 12 qubits). H2O's full STO-3G Hamiltonian is 14 qubits -- too large for
# dense diagonalization (2**14 x 2**14 complex128 is ~34 GB) and beyond
# the ~12-qubit range where this VQE ansatz still optimizes comfortably,
# so it's given a real frozen-core active space (freezing the O 1s core
# orbital: 10 electrons -> 8 active electrons, 7 orbitals -> 6 active
# orbitals = 12 qubits), a standard, physically honest quantum-chemistry
# approximation -- not a fabricated circuit.
MOLECULE_CATALOG = {
    "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]": {
        "symbols": ["H", "H"],
        "geometry": lambda: _linear_two_atom_geometry(0.7414),
        "charge": 0,
        "active_electrons": None,
        "active_orbitals": None,
    },
    "HeH+ (Idruro di Elio, catione) - R = 0.7743 A [equilibrio reale]": {
        "symbols": ["He", "H"],
        "geometry": lambda: _linear_two_atom_geometry(0.7743),
        "charge": 1,
        "active_electrons": None,
        "active_orbitals": None,
    },
    "H3+ (Ione Triidrogeno) - triangolo equilatero D3h, R = 0.8738 A [equilibrio reale]": {
        "symbols": ["H", "H", "H"],
        "geometry": lambda: _triangular_h3_geometry(0.8738),
        "charge": 1,
        "active_electrons": None,
        "active_orbitals": None,
    },
    "LiH (Idruro di Litio) - R = 1.5949 A [equilibrio reale]": {
        "symbols": ["Li", "H"],
        "geometry": lambda: _linear_two_atom_geometry(1.5949),
        "charge": 0,
        "active_electrons": None,
        "active_orbitals": None,
    },
    "H2O (Acqua) - angolo 104.5 deg, R(O-H) = 0.9584 A [equilibrio reale, frozen-core O(1s)]": {
        "symbols": ["O", "H", "H"],
        "geometry": lambda: _water_geometry(0.9584, 104.5),
        "charge": 0,
        "active_electrons": 8,
        "active_orbitals": 6,
    },
}

_pennylane_hamiltonian_cache = {}
_dense_hamiltonian_cache = {}


def _get_pennylane_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals):
    """Real Hartree-Fock + fermion-to-qubit mapping (PennyLane qchem),
    returning the PennyLane operator (not yet densified) and n_qubits.
    Split out from build_molecular_hamiltonian so callers that only need
    n_qubits (e.g. listing the catalog) don't pay for qml.matrix(H) --
    building a dense matrix is the expensive/memory-heavy step, not the
    HF+mapping step itself. Cached on its own so the catalog listing and
    a later "compute ground state" / VQE call on the same molecule share
    the one real HF computation instead of repeating it."""
    import pennylane as qml

    key = (tuple(symbols), tuple(map(tuple, np.asarray(geometry).round(10))), charge, mapping,
           active_electrons, active_orbitals)
    if key in _pennylane_hamiltonian_cache:
        return _pennylane_hamiltonian_cache[key]

    molecule = qml.qchem.Molecule(symbols, np.asarray(geometry), charge=charge, unit="angstrom")
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule, method="dhf", mapping=mapping,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
    )
    _pennylane_hamiltonian_cache[key] = (H, n_qubits)
    return H, n_qubits


def _water_geometry(bond_length_angstrom: float, angle_degrees: float):
    """Real equilibrium water geometry: O at the origin, both O-H bonds
    at the given real bond length and real H-O-H bond angle (104.5 deg),
    placed symmetrically about the z-axis in the xz-plane."""
    half_angle = np.radians(angle_degrees) / 2.0
    r = bond_length_angstrom
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(half_angle), 0.0, r * np.cos(half_angle)],
        [-r * np.sin(half_angle), 0.0, r * np.cos(half_angle)],
    ])


def build_molecular_hamiltonian(symbols, geometry, charge: int = 0, mapping: str = "jordan_wigner",
                                 active_electrons=None, active_orbitals=None):
    """Runs real Hartree-Fock + fermion-to-qubit mapping (PennyLane
    qchem) on the given geometry and returns (H_dense, n_qubits). The
    eigenvalue spectrum (and therefore the ground-state energy) is
    mapping-invariant -- Jordan-Wigner and Bravyi-Kitaev represent the
    identical physical Hamiltonian in a different qubit basis -- so this
    only changes which qubit operators appear, never the energies this
    function's callers report. Cached: Hartree-Fock isn't free, and the
    UI can re-request the same molecule repeatedly."""
    import pennylane as qml

    H, n_qubits = _get_pennylane_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)

    dense_key = (tuple(symbols), tuple(map(tuple, np.asarray(geometry).round(10))), charge, mapping,
                 active_electrons, active_orbitals)
    if dense_key in _dense_hamiltonian_cache:
        return _dense_hamiltonian_cache[dense_key]

    H_dense = np.asarray(qml.matrix(H), dtype=np.complex128)
    result = (H_dense, n_qubits)
    _dense_hamiltonian_cache[dense_key] = result
    return result


def get_molecule_n_qubits(symbols, geometry, charge=0, mapping="jordan_wigner",
                           active_electrons=None, active_orbitals=None):
    """The qubit count a molecule's real Hamiltonian needs, without
    paying for a dense matrix build -- cheap enough to call for every
    catalog entry when just listing what's available."""
    _, n_qubits = _get_pennylane_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)
    return n_qubits


def get_all_molecules(catalog=None, mapping="jordan_wigner"):
    """Every catalog molecule, each annotated with its real qubit count
    under the given mapping -- unfiltered, so the UI can always show the
    whole catalog and let the molecule choice drive the circuit's qubit
    count (not the other way around)."""
    catalog = catalog if catalog is not None else MOLECULE_CATALOG
    out = {}
    for name, spec in catalog.items():
        geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
        n_qubits = get_molecule_n_qubits(
            spec["symbols"], geometry, spec["charge"], mapping=mapping,
            active_electrons=spec.get("active_electrons"), active_orbitals=spec.get("active_orbitals"),
        )
        out[name] = {
            "symbols": spec["symbols"],
            "geometry": np.asarray(geometry).tolist(),
            "charge": spec["charge"],
            "n_qubits": n_qubits,
        }
    return out


def get_compatible_molecules(n_qubits, catalog=None, mapping="jordan_wigner"):
    """Filters MOLECULE_CATALOG down to molecules whose real Hamiltonian
    needs exactly n_qubits. Kept for callers that want a qubit-filtered
    view; the main catalog UI uses get_all_molecules instead so every
    molecule is always visible."""
    catalog = catalog if catalog is not None else MOLECULE_CATALOG
    if n_qubits is None or n_qubits <= 0:
        return {}
    all_molecules = get_all_molecules(catalog, mapping=mapping)
    return {name: catalog[name] for name, info in all_molecules.items() if info["n_qubits"] == n_qubits}


def get_molecular_hamiltonian_matrix(name, catalog=None, mapping="jordan_wigner"):
    """Resolves a MOLECULE_CATALOG entry by name to its (cached) dense
    Hermitian Hamiltonian matrix, under the given fermion-to-qubit
    mapping (spectrum is identical either way, see build_molecular_hamiltonian)."""
    catalog = catalog if catalog is not None else MOLECULE_CATALOG
    spec = catalog[name]
    geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
    H_dense, _ = build_molecular_hamiltonian(
        spec["symbols"], geometry, spec["charge"], mapping=mapping,
        active_electrons=spec.get("active_electrons"), active_orbitals=spec.get("active_orbitals"),
    )
    return H_dense


def ground_state_energy(H_dense) -> float:
    """Exact ground-state energy via dense diagonalization -- a real,
    checkable number (Hartree) for a Hamiltonian this small (H2/HeH+/H3+
    all fit well within exact diagonalization), not an estimate."""
    eigvals = np.linalg.eigvalsh(H_dense)
    return float(eigvals.min())


def mix_hamiltonians(H_a, H_b, weight_a: float = 0.5, weight_b: float = 0.5):
    """Real weighted combination H_mix = weight_a*H_a + weight_b*H_b of
    two molecular Hamiltonians acting on the same qubit space (same
    electron/qubit count -- the only condition that makes the sum mean
    anything, mirroring the old dashboard's own "mix molecules that
    share an electron space" behavior). A real-weighted sum of two
    Hermitian matrices is itself Hermitian, so H_mix is a real, valid
    Hamiltonian with a real spectrum -- not a fabricated hybrid, just
    linear algebra applied to two already-real operators."""
    if H_a.shape != H_b.shape:
        dim_a, dim_b = H_a.shape[0], H_b.shape[0]
        raise ValueError(
            f"cannot mix: different qubit spaces ({int(np.log2(dim_a))} vs {int(np.log2(dim_b))} qubits)"
        )
    H_mix = weight_a * H_a + weight_b * H_b
    if not np.allclose(H_mix, H_mix.conj().T, atol=1e-9):
        raise ValueError("mixed Hamiltonian is not Hermitian -- this should be unreachable")
    return H_mix
