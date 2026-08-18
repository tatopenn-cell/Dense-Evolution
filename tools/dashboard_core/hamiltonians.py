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

For elements PennyLane's own bundled STO-3G table doesn't cover (row 3+,
e.g. Silicon), this falls back to dense_evolution.native_hf -- a
from-scratch, jax-vmap-vectorized Hartree-Fock engine (Obara-Saika
integrals, Roothaan-Hall SCF) that sources basis-set data from
basis_set_exchange instead, so any element it has STO-3G parameters for
works. Only the Hartree-Fock/integral stage is native; the resulting
converged result is still handed to PennyLane's own fermionic_observable
+ jordan_wigner for the qubit mapping (see native_hf/bridge.py), since
that stage is already fast and well-tested.
"""

import numpy as np

import dense_evolution as de

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
    # Real published equilibrium bond length (Balamurugan & Prasad,
    # "Effect of hydrogen on ground state structures of small silicon
    # clusters", arXiv:cond-mat/0108426). Si isn't in PennyLane's own
    # bundled STO-3G table, so this routes through native_hf (see module
    # docstring) -- verified against an independent reference
    # (lowdanie/hartree-fock-solver) to 10 significant figures. Given only
    # 4 active electrons/orbitals (freezing all 20 core electrons: Si's
    # 1s,2s,2p x2 atoms), this active space is too small to reproduce
    # 2.184 A as its own energy minimum (checked directly: a 10-point scan
    # from 1.9-4.0 A found its minimum at the 1.9 A edge of the range, not
    # an interior point) -- included anyway, with this caveat stated
    # plainly, rather than silently picking a geometry that flatters the
    # active-space choice.
    "Si2 (Disilicio) - R = 2.184 A [equilibrio reale, active space minimo]": {
        "symbols": ["Si", "Si"],
        "geometry": lambda: _linear_two_atom_geometry(2.184),
        "charge": 0,
        "active_electrons": 4,
        "active_orbitals": 4,
    },
}

_pennylane_hamiltonian_cache = {}
_dense_hamiltonian_cache = {}

# Same physical floor as dashboard_core.qmmm.MIN_NUCLEAR_DISTANCE_ANGSTROM
# (kept as a separate constant here, not imported, since qmmm.py already
# imports from this module -- importing back would be circular). Any real
# atomic radius is well above this; two nuclei closer than this in an
# *input* geometry (as opposed to qmmm's own post-MD-step divergence
# check) means malformed input, not physics.
MIN_NUCLEAR_DISTANCE_ANGSTROM = 0.3


def _validate_geometry(symbols, geometry):
    """BUG FIX: build_molecular_hamiltonian had no input validation at
    all -- a symbols/geometry length mismatch surfaced as a raw
    IndexError deep inside PennyLane's own internals (verified directly:
    2 symbols + 1-row geometry -> 'IndexError: index 1 is out of bounds
    for axis 0 with size 1', no indication the actual problem is the
    caller's mismatched input), and non-finite coordinates (e.g. a NaN
    from an upstream bug) were silently accepted and produced a NaN
    Hamiltonian with no error at all (verified directly: NaN in a
    geometry row -> np.any(np.isnan(H_dense)) is True, no exception).
    Called once, at the real entry point every other function here
    funnels through (_get_pennylane_hamiltonian), not duplicated at each
    of its public callers.
    """
    geometry = np.asarray(geometry, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError(
            f"geometry must have shape (n_atoms, 3), got {geometry.shape}")
    if len(symbols) != geometry.shape[0]:
        raise ValueError(
            f"{len(symbols)} symbols but {geometry.shape[0]} geometry rows -- "
            f"these must match one-to-one")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry contains non-finite values (NaN/Inf)")
    n_atoms = geometry.shape[0]
    if n_atoms > 1:
        diffs = geometry[:, None, :] - geometry[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if dists[i, j] < MIN_NUCLEAR_DISTANCE_ANGSTROM:
            raise ValueError(
                f"atoms {i} and {j} are {dists[i, j]:.4f} A apart, below the "
                f"{MIN_NUCLEAR_DISTANCE_ANGSTROM} A physically-realistic floor "
                f"-- check the geometry for a units mistake (e.g. Bohr instead "
                f"of Angstrom) or a duplicated atom")
    return geometry


_native_hamiltonian_cache = {}


def _get_native_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals):
    """Same contract as _get_pennylane_hamiltonian (returns (H, n_qubits),
    cached), but for elements outside PennyLane's bundled STO-3G table --
    routes through dense_evolution.native_hf instead (see module
    docstring). Jordan-Wigner only for now: native_hf/bridge.py calls
    qml.jordan_wigner directly rather than taking a mapping parameter,
    since every current caller (MOLECULE_CATALOG) already defaults to
    jordan_wigner -- raising here instead of silently ignoring a
    different requested mapping."""
    if mapping != "jordan_wigner":
        raise NotImplementedError(
            f"native_hf fallback only supports mapping='jordan_wigner' (got {mapping!r}) "
            f"-- native_hf/bridge.py calls qml.jordan_wigner directly, not a general mapper."
        )

    from basis_set_exchange.lut import element_Z_from_sym
    from dense_evolution.native_hf.bridge import build_qubit_hamiltonian

    geometry = _validate_geometry(symbols, geometry)

    key = (tuple(symbols), tuple(map(tuple, geometry.round(10))), charge,
           active_electrons, active_orbitals)
    if key in _native_hamiltonian_cache:
        return _native_hamiltonian_cache[key]

    atomic_numbers = [element_Z_from_sym(s) for s in symbols]
    n_electrons = sum(atomic_numbers) - charge

    H, n_qubits, _hf_result = build_qubit_hamiltonian(
        atomic_numbers, geometry, n_electrons,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
    )
    _native_hamiltonian_cache[key] = (H, n_qubits)
    return H, n_qubits


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
    from pennylane.qchem.basis_data import STO3G

    geometry = _validate_geometry(symbols, geometry)

    key = (tuple(symbols), tuple(map(tuple, geometry.round(10))), charge, mapping,
           active_electrons, active_orbitals)
    if key in _pennylane_hamiltonian_cache:
        return _pennylane_hamiltonian_cache[key]

    # PennyLane's own error for an unsupported element (raised deep inside
    # molecular_hamiltonian, e.g. requesting Fe/Mo/Xe) is real and honest
    # but written for someone already inside PennyLane's own codebase --
    # "consider using load_data=True ... basis-set-exchange". Checked
    # against STO3G's real key set (not a hardcoded/guessed list, so this
    # never goes stale against a future PennyLane version) and re-raised
    # with the concrete supported set and which of the caller's own
    # symbols are the problem, before paying for anything else.
    unsupported = sorted({s for s in symbols if s not in STO3G})
    if unsupported:
        raise ValueError(
            f"No built-in STO-3G basis data for: {', '.join(unsupported)}. "
            f"This pipeline (PennyLane qchem, method='dhf') only ships minimal-basis "
            f"parameters for {', '.join(sorted(STO3G))} -- heavier elements (transition "
            f"metals, anything past Ne) need an external basis-set source PennyLane "
            f"doesn't bundle, not a dense_evolution limitation."
        )

    molecule = qml.qchem.Molecule(symbols, np.asarray(geometry), charge=charge, unit="angstrom")
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule, method="dhf", mapping=mapping,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
    )
    _pennylane_hamiltonian_cache[key] = (H, n_qubits)
    return H, n_qubits


def _get_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals):
    """Dispatches to PennyLane's own qchem pipeline when every symbol is
    in its bundled STO-3G table, or to native_hf otherwise -- the single
    real entry point every public function below funnels through, so
    "does this molecule need the native fallback" is decided in exactly
    one place."""
    from pennylane.qchem.basis_data import STO3G

    if all(s in STO3G for s in symbols):
        return _get_pennylane_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)
    return _get_native_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)


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


def _pennylane_hamiltonian_to_pauli_terms(H, n_qubits):
    """Extracts (coeff, {qubit: 'X'|'Y'|'Z'}) terms from a PennyLane
    Hamiltonian/Sum operator -- the same real Pauli decomposition
    PennyLane itself uses internally, just handed over in the plain form
    dense_evolution.pauli_hamiltonian_to_matrix accepts, so this
    project's own engine builds the dense matrix instead of going through
    qml.matrix(). Verified (see dense_evolution/tests/test_observables.py
    and this session's own cross-checks) to reproduce qml.matrix(H)
    exactly for H2/HeH+/H3+, not an approximation."""
    coeffs, ops = H.terms()
    terms = []
    for coeff, op in zip(coeffs, ops):
        pauli = {}
        factors = op.operands if hasattr(op, 'operands') else [op]
        for factor in factors:
            wires = factor.wires
            if not len(wires) or factor.name == 'Identity':
                continue
            pauli[int(wires[0])] = factor.name[-1]  # 'PauliZ' -> 'Z', etc.
        real_coeff = float(np.real(complex(coeff)))
        terms.append((real_coeff, pauli))
    return terms


def build_molecular_hamiltonian(symbols, geometry, charge: int = 0, mapping: str = "jordan_wigner",
                                 active_electrons=None, active_orbitals=None):
    """Runs real Hartree-Fock + fermion-to-qubit mapping (PennyLane
    qchem) on the given geometry and returns (H_dense, n_qubits). The
    eigenvalue spectrum (and therefore the ground-state energy) is
    mapping-invariant -- Jordan-Wigner and Bravyi-Kitaev represent the
    identical physical Hamiltonian in a different qubit basis -- so this
    only changes which qubit operators appear, never the energies this
    function's callers report. Cached: Hartree-Fock isn't free, and the
    UI can re-request the same molecule repeatedly.

    The dense matrix itself is built by this project's own
    dense_evolution.pauli_hamiltonian_to_matrix from PennyLane's real
    Pauli decomposition, not qml.matrix() -- verified to match qml.matrix
    exactly (same ground-state energy, same matrix, atol=1e-8) for every
    catalog molecule."""
    H, n_qubits = _get_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)

    dense_key = (tuple(symbols), tuple(map(tuple, np.asarray(geometry).round(10))), charge, mapping,
                 active_electrons, active_orbitals)
    if dense_key in _dense_hamiltonian_cache:
        return _dense_hamiltonian_cache[dense_key]

    # H_dense is dim x dim (dim = 2**n_qubits), not just dim, and its only
    # consumer (ground_state_energy) runs a full dense np.linalg.eigvalsh
    # on it -- LAPACK's own eigh workspace needs comparable scratch memory
    # on top of the matrix itself, and the geometry generators in the
    # Composer's UI (linear_chain_geometry/ring_geometry) let a visitor
    # build an arbitrarily long chain, with no smaller natural ceiling than
    # whatever PennyLane's own Hartree-Fock step tolerates. Same real
    # anti-OOM guard as dashboard_core.engine.run_circuit_from_qasm and
    # mitigation.py's ZNE panels, sized for what this actually allocates
    # (the x3 covers the matrix + its eigh scratch space + the cached copy
    # this function stores in _dense_hamiltonian_cache).
    dim = 2 ** n_qubits
    required_mb = dim * dim * 16 / 1e6 * 3
    de.chunk.SafeMemoryGuard().check_allocation(required_mb, context=f"{n_qubits}-qubit molecular Hamiltonian")

    terms = _pennylane_hamiltonian_to_pauli_terms(H, n_qubits)
    H_dense = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
    result = (H_dense, n_qubits)
    _dense_hamiltonian_cache[dense_key] = result
    return result


def get_molecule_n_qubits(symbols, geometry, charge=0, mapping="jordan_wigner",
                           active_electrons=None, active_orbitals=None):
    """The qubit count a molecule's real Hamiltonian needs, without
    paying for a dense matrix build -- cheap enough to call for every
    catalog entry when just listing what's available."""
    _, n_qubits = _get_hamiltonian(symbols, geometry, charge, mapping, active_electrons, active_orbitals)
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
