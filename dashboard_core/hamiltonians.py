"""
Hamiltonian sources for the VQE panel: a small library of diagonal toy
models (Ising, Heisenberg, Hubbard, ...) plus a catalog of REAL molecular
Hamiltonians built on demand from actual atomic geometry via PennyLane's
qchem module (Hartree-Fock + Jordan-Wigner, method='dhf' -- native to
PennyLane, no PySCF/OpenFermion dependency needed).

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- the toy-model dict below is that original move,
unchanged. The molecular catalog is new: the file used to also carry
~10 "real molecule" entries (H2, HeH+, LiH, N2, H2O, NH3, CH4, BeH2, HF)
labeled with specific literature bond lengths (e.g. "H2 - R = 0.74 Å
[Equilibrio Reale]") whose actual coefficient arrays were hand-typed
numbers, not computed from that geometry or from any real electronic
structure calculation -- convincing-looking, but fabricated. Real
molecules now go through build_molecular_hamiltonian() instead, which
computes them for real; nothing here claims to be real chemistry without
actually being it.

Represented Hamiltonians come in two shapes, both understood by the VQE
panel (dashboard_core.vqe_engine._run_vqe_telemetry_body, which branches
on numpy .ndim): a flat list/array of length 2**n_qubits (interpreted as
a diagonal matrix -- the toy models below, and whatever a user pastes
into the Custom JSON textarea) or a dense (2**n_qubits, 2**n_qubits)
Hermitian matrix (the real molecular Hamiltonians -- Jordan-Wigner maps
electronic structure onto X/Y/Z terms, not just Z, so these are never
diagonal in the computational basis).
"""

LIBRERIA_HAMILTONIANE = {
    # ---------------------------------------------------------------------
    # --- MODELLI GIOCATTOLO / DI FISICA (spettro diagonale semplificato,
    # --- non chimica reale -- vedi MOLECULE_CATALOG per quella)
    # ---------------------------------------------------------------------
    "H3+ (Ione Triidrogeno) - modello giocattolo lineare 3q":
        [-1.28, -0.94, -0.51, -0.08, 0.33, 0.76, 1.18, 1.55],
    "Modello di Lipkin (Fisica Nucleare 3q Baseline)":
        [-2.00, -1.41, -0.82, -0.22, 0.35, 0.91, 1.54, 2.11],
    "Catena di Heisenberg XXX (Antiferromagnetica 3q)":
        [-1.82, -1.22, -0.61, -0.11, 0.42, 0.93, 1.34, 1.85],
    "Modello Ising Lineare (4 Qubit Baseline)":
        [-1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5],
    "Modello Hubbard (Sito 2x2, Half-Filling 4q)":
        [-3.21, -2.75, -2.24, -1.72, -1.18, -0.64, -0.11, 0.42, 0.95, 1.48, 2.01, 2.54, 3.06, 3.58, 4.11, 4.64],
    "Interazione Cooper (Superconduttività BCS, modello giocattolo 4q)":
        [-1.95, -1.68, -1.41, -1.12, -0.84, -0.55, -0.24, 0.05, 0.36, 0.68, 0.98, 1.29, 1.58, 1.88, 2.19, 2.48],
}


# ═══════════════════════════════════════════════════════════════════════
# Molecole reali -- geometria vera (Å), calcolate al volo con PennyLane
# ═══════════════════════════════════════════════════════════════════════

def _triangular_h3_geometry(bond_length_angstrom: float):
    """H3+ has a real, well-documented equilateral-triangle (D3h) ground-
    state geometry -- not the linear-chain approximation the old static
    library used under the same name. This is also the one genuinely
    "ring/cyclic" molecule small enough for this simulator's exact dense
    diagonalization (6 qubits, dim=64) -- larger rings (e.g. benzene)
    would need far more qubits than exact diagonalization can handle
    here, so this catalog deliberately stays small rather than promising
    scale it doesn't have."""
    import numpy as np
    r = bond_length_angstrom
    h = r * (3 ** 0.5) / 2
    return np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [r / 2, h, 0.0]])


# name -> (symbols, geometry_builder, charge). geometry_builder is either
# a fixed np.ndarray or a zero-arg callable building one (kept lazy so
# numpy only imports when a molecule's Hamiltonian is actually needed).
def _linear_two_atom_geometry(bond_length_angstrom: float):
    import numpy as np
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length_angstrom]])


MOLECULE_CATALOG = {
    # Bond lengths are real, published equilibrium geometries, not
    # invented to look plausible:
    "H2 (Idrogeno) - R = 0.7414 Å [equilibrio reale]": {
        "symbols": ["H", "H"],
        "geometry": lambda: _linear_two_atom_geometry(0.7414),
        "charge": 0,
    },
    "HeH+ (Idruro di Elio, catione) - R = 0.7743 Å [equilibrio reale]": {
        "symbols": ["He", "H"],
        "geometry": lambda: _linear_two_atom_geometry(0.7743),
        "charge": 1,
    },
    "H3+ (Ione Triidrogeno) - triangolo equilatero D3h, R = 0.8738 Å [equilibrio reale]": {
        "symbols": ["H", "H", "H"],
        "geometry": lambda: _triangular_h3_geometry(0.8738),
        "charge": 1,
    },
}

_molecular_hamiltonian_cache = {}


def build_molecular_hamiltonian(symbols, geometry, charge: int = 0):
    """Computes a REAL molecular electronic-structure Hamiltonian from
    real atomic symbols + 3D geometry (Å), via PennyLane's native
    Hartree-Fock solver (method='dhf' -- no PySCF/OpenFermion needed,
    pennylane is already a project dependency). Returns
    (H_dense: np.ndarray complex128, n_qubits: int) -- Jordan-Wigner maps
    the fermionic Hamiltonian onto real X/Y/Z Pauli terms, so H_dense is
    generally NOT diagonal even though its eigenvalues (and the physics)
    are real.

    Cached in-memory by (symbols, geometry, charge) -- Hartree-Fock is
    not free, and the same catalog molecule gets selected repeatedly
    across VQE runs."""
    import numpy as np
    import pennylane as qml

    key = (tuple(symbols), tuple(map(tuple, np.asarray(geometry).round(10))), charge)
    if key in _molecular_hamiltonian_cache:
        return _molecular_hamiltonian_cache[key]

    molecule = qml.qchem.Molecule(symbols, np.asarray(geometry), charge=charge, unit="angstrom")
    H, n_qubits = qml.qchem.molecular_hamiltonian(molecule, method="dhf")
    H_dense = np.asarray(qml.matrix(H), dtype=np.complex128)

    result = (H_dense, n_qubits)
    _molecular_hamiltonian_cache[key] = result
    return result


def get_compatible_molecules(n_qubits, catalog=None):
    """Like get_compatible_hamiltonians, but for MOLECULE_CATALOG: builds
    (and caches) each molecule's real Hamiltonian to learn its actual
    qubit count, then filters to the ones matching n_qubits. Building is
    cheap for this catalog's small molecules (4-6 qubits) and cached
    after the first call, so this is safe to call on every UI refresh."""
    catalog = catalog if catalog is not None else MOLECULE_CATALOG
    if n_qubits is None or n_qubits <= 0:
        return {}
    compatible = {}
    for name, spec in catalog.items():
        geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
        _, mol_n_qubits = build_molecular_hamiltonian(spec["symbols"], geometry, spec["charge"])
        if mol_n_qubits == n_qubits:
            compatible[name] = spec
    return compatible


def get_molecular_hamiltonian_matrix(name, catalog=None):
    """Resolves a MOLECULE_CATALOG entry by name to its (cached) dense
    Hamiltonian matrix -- what the VQE panel actually needs once the user
    has picked a molecule from the dropdown."""
    catalog = catalog if catalog is not None else MOLECULE_CATALOG
    spec = catalog[name]
    geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
    H_dense, _ = build_molecular_hamiltonian(spec["symbols"], geometry, spec["charge"])
    return H_dense


# ═══════════════════════════════════════════════════════════════════════

def get_compatible_hamiltonians(n_qubits, library=None):
    """Filters a Hamiltonian library down to entries whose diagonal length
    matches 2**n_qubits. Adapted from update_hamiltonian_options_and_state's
    filter (dash.py:2825) — `values is not None and len(values) == expected_dim`."""
    library = library if library is not None else LIBRERIA_HAMILTONIANE
    if n_qubits is None or n_qubits <= 0:
        return {}
    expected_dim = 2 ** n_qubits
    return {name: values for name, values in library.items()
            if values is not None and len(values) == expected_dim}


def save_custom_hamiltonian(library, name, values_json_str):
    """Validates and inserts a custom Hamiltonian into `library` (mutated in
    place). Adapted from core_trigger_save_hamiltonian (dash.py:2920), minus
    the blocking input() call for the name (replaced by an explicit `name`
    param — the UI layer collects it via st.text_input instead).
    Returns (success: bool, message: str)."""
    import json as _json

    if not name:
        return False, "Nome dell'Hamiltoniana non valido."
    if name in library:
        return False, f"Un'Hamiltoniana con il nome '{name}' esiste già. Scegli un nome diverso."
    try:
        values = _json.loads(values_json_str)
    except _json.JSONDecodeError:
        return False, "Errore di parsing JSON: assicurati che l'input sia un array JSON valido."
    if not isinstance(values, list) or not all(isinstance(x, (int, float)) for x in values):
        return False, "L'input deve essere un array JSON di numeri (es. [1.0, 2.0, 3.0])."

    library[name] = values
    return True, f"Hamiltoniana '{name}' salvata con successo!"
