"""
Preset Hamiltonian library and lookup/save helpers.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

LIBRERIA_HAMILTONIANE = {
    # ---------------------------------------------------------------------
    # --- GRUPPO 1: BENCHMARK CHIMICI REALI (2 QUBIT / DIM=4 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H2 (Idrogeno) - R = 0.50 Å [Compressione]": [-0.51, -0.12, 0.35, 0.85],
    "H2 (Idrogeno) - R = 0.74 Å [Equilibrio Reale]": [-1.13, -0.45, 0.12, 0.64],
    "H2 (Idrogeno) - R = 1.20 Å [Dissociazione]": [-0.92, -0.68, -0.15, 0.22],
    "HeH+ (Idruro di Elio) - R = 0.93 Å [Equilibrio]": [-1.41, -0.82, -0.22, 0.45],
    "H2 (Idrogeno) - R = 1.50 Å [Asintoto Dissoc.]": [-0.78, -0.65, -0.31, 0.11],
    "HeH+ (Idruro di Elio) - R = 0.50 Å [Stallo Interno]": [-0.22, 0.14, 0.76, 1.48],
    "HeH+ (Idruro di Elio) - R = 1.60 Å [Limite Ionico]": [-1.05, -0.91, -0.44, 0.02],
    "LiH (Idruro di Litio) - STO-3G (Sotto-spazio 2q)": [-1.62, -0.98, -0.11, 0.54],

    # ---------------------------------------------------------------------
    # --- GRUPPO 2: CHIMICA ED ENTANGLEMENT AVANZATO (3 QUBIT / DIM=8 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H3+ (Ione Triidrogeno Linear) - R = 0.85 Å": [-1.28, -0.94, -0.51, -0.08, 0.33, 0.76, 1.18, 1.55],
    "H3+ (Ione Triidrogeno Triang) - R = 0.90 Å": [-1.34, -1.01, -0.62, -0.14, 0.28, 0.69, 1.09, 1.42],
    "Modello di Lipkin (Fisica Nucleare 3q Baseline)": [-2.00, -1.41, -0.82, -0.22, 0.35, 0.91, 1.54, 2.11],
    "Catena di Heisenberg XXX (Antiferromagnetica 3q)": [-1.82, -1.22, -0.61, -0.11, 0.42, 0.93, 1.34, 1.85],

    # ---------------------------------------------------------------------
    # --- GRUPPO 3: MASSA MOLECOLARE INTERMEDIA (4 QUBIT / DIM=16 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "Modello Ising Lineare (4 Qubit Baseline)": [-1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5],
    "LiH (Idruro di Litio) - R = 1.40 Å [Minimo]": [-2.31, -2.01, -1.65, -1.22, -0.85, -0.41, 0.02, 0.44, 0.88, 1.25, 1.61, 1.98, 2.34, 2.71, 3.05, 3.42],
    "LiH (Idruro di Litio) - R = 2.20 Å [Torsione]": [-1.89, -1.62, -1.31, -0.98, -0.62, -0.22, 0.15, 0.52, 0.91, 1.28, 1.64, 1.99, 2.33, 2.68, 3.01, 3.35],
    "BH3 (Borano Parziale) - R = 1.15 Å": [-2.85, -2.42, -2.01, -1.55, -1.11, -0.65, -0.21, 0.22, 0.64, 1.05, 1.47, 1.88, 2.29, 2.69, 3.08, 3.49],
    "H4 (Catena di Idrogeno Quadrata) - R = 1.00 Å": [-2.14, -1.82, -1.44, -1.02, -0.61, -0.18, 0.24, 0.65, 1.08, 1.49, 1.91, 2.32, 2.72, 3.11, 3.51, 3.92],
    "H4 (Catena di Idrogeno Lineare) - R = 1.25 Å": [-2.45, -2.11, -1.72, -1.34, -0.92, -0.51, -0.08, 0.34, 0.76, 1.17, 1.58, 1.99, 2.38, 2.78, 3.16, 3.55],
    "Modello Hubbard (Sito 2x2, Half-Filling 4q)": [-3.21, -2.75, -2.24, -1.72, -1.18, -0.64, -0.11, 0.42, 0.95, 1.48, 2.01, 2.54, 3.06, 3.58, 4.11, 4.64],
    "Interazione Cooper (Superconduttività BC 4q)": [-1.95, -1.68, -1.41, -1.12, -0.84, -0.55, -0.24, 0.05, 0.36, 0.68, 0.98, 1.29, 1.58, 1.88, 2.19, 2.48],

    # ---------------------------------------------------------------------
    # --- GRUPPO 4: MACROMOLECOLE PRE-CALCOLATE (5-6 QUBIT / DIM=32-64 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H2O (Acqua Embedding Core) - R = 0.96 Å [32 Val]": [
        -4.12, -3.79, -3.47, -3.15, -2.83, -2.51, -2.19, -1.86, -1.54, -1.22, -0.90, -0.58, -0.26, 0.06, 0.38, 0.70,
         1.02,  1.34,  1.67,  1.99,  2.31,  2.63,  2.95,  3.27,  3.59,  3.91,  4.24,  4.56,  4.88, 5.20, 5.52, 5.85
    ],
    "NH3 (Ammoniaca Sotto-guscio) - R = 1.01 Å [32 Val]": [
        -4.85, -4.49, -4.13, -3.78, -3.42, -3.06, -2.71, -2.35, -2.00, -1.64, -1.28, -0.93, -0.57, -0.21, 0.14, 0.50,
         0.85,  1.21,  1.56,  1.92,  2.28,  2.63,  2.99,  3.35,  3.70,  4.06,  4.41,  4.77,  5.13, 5.48, 5.84, 6.22
    ],
    "CH4 (Metano Orbitale Ibrido) - R = 1.09 Å [32 Val]": [
        -5.12, -4.71, -4.31, -3.91, -3.51, -3.11, -2.71, -2.31, -1.90, -1.50, -1.10, -0.70, -0.30, 0.10, 0.50, 0.90,
         1.31,  1.71,  2.11,  2.51,  2.91,  3.31,  3.71,  4.11,  4.52,  4.92,  5.32,  5.72,  6.12, 6.52, 6.92, 7.34
    ],
    "BeH2 (Idruro di Berillio Active Space) [64 Val]": [
        -6.42, -6.19, -5.96, -5.73, -5.50, -5.27, -5.04, -4.81, -4.58, -4.35, -4.12, -3.89, -3.66, -3.43, -3.20, -2.97,
        -2.74, -2.51, -2.28, -2.05, -1.82, -1.59, -1.36, -1.13, -0.90, -0.67, -0.44, -0.21,  0.02,  0.25,  0.48,  0.71,
         0.94,  1.17,  1.40,  1.63,  1.86,  2.09,  2.32,  2.55,  2.78,  3.01,  3.24,  3.47,  3.70,  3.93,  4.16,  4.39,
         4.62,  4.85,  5.08,  5.31,  5.54,  5.77,  6.00,  6.23,  6.46,  6.69,  6.92,  7.15,  7.38,  7.61,  7.84,  8.11
    ],
    "N2 (Azoto Molecolare Singlet-State) [64 Val]": [
        -8.95, -8.63, -8.31, -7.99, -7.67, -7.35, -7.03, -6.71, -6.39, -6.07, -5.75, -5.43, -5.11, -4.79, -4.47, -4.15,
        -3.83, -3.51, -3.19, -2.87, -2.55, -2.23, -1.91, -1.59, -1.27, -0.95, -0.63, -0.31,  0.01,  0.33,  0.65,  0.97,
         1.29,  1.61,  1.93,  2.25,  2.57,  2.89,  3.21,  3.53,  3.85,  4.17,  4.49,  4.81,  5.13,  5.45,  5.77,  6.09,
         6.41,  6.73,  7.05,  7.37,  7.69,  8.01,  8.33,  8.65,  8.97,  9.29,  9.61,  9.93, 10.25, 10.57, 10.89, 11.24
    ],
    "HF (Acido Fluoridrico Valence Space) [64 Val]": [
        -7.14, -6.87, -6.61, -6.34, -6.08, -5.81, -5.54, -5.28, -5.01, -4.75, -4.48, -4.21, -3.95, -3.68, -3.42, -3.15,
        -2.88, -2.62, -2.35, -2.09, -1.82, -1.55, -1.29, -1.02, -0.76, -0.49, -0.22,  0.04,  0.31,  0.57,  0.84,  1.10,
         1.37,  1.64,  1.90,  2.17,  2.43,  2.70,  2.96,  3.23,  3.50,  3.76,  4.03,  4.29,  4.56,  4.82,  5.09,  5.35,
         5.62,  5.89,  6.15,  6.42,  6.68,  6.95,  7.21,  7.48,  7.75,  8.01,  8.28,  8.54,  8.81,  9.07,  9.34,  9.65
    ],

    "Spettro Uniforme Classico (Baseline linspace)": None,
}


def get_compatible_hamiltonians(n_qubits, library=None):
    """Filters a Hamiltonian library down to entries whose diagonal length
    matches 2**n_qubits. Adapted from update_hamiltonian_options_and_state's
    filter (dash.py:2825) — `values is not None and len(values) == expected_dim`,
    which is also why "Spettro Uniforme Classico" (value None) never actually
    appears as selectable in the original either; ported faithfully."""
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
