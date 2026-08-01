"""
PXP model (Rydberg blockade) exact-diagonalization engine backing the
Quantum Scars live demo (ui_pages/quantum_scars.py's Streamlit page, and
launch_interactive_panel's ipywidgets equivalent). Deliberately duplicated
from that page's private helpers rather than imported -- same reasoning as
`_heal_telemetry` in interactive_panel.py: ui_pages is repo-only, not part
of the installable `dashboard_core` package.

Pure numpy/scipy, no Streamlit import. `build_pxp` results are cached in a
plain module-level dict keyed by n_qubits (exact diagonalization of a dense
2**n_qubits matrix is genuinely expensive and depends only on n_qubits) --
the ipywidgets equivalent of the Streamlit page's `st.cache_resource`.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from dense_evolution.registry import NoiseModel

__all__ = ['build_pxp', 'run_experiment', 'DT_CHUNK', 'N_CHUNK']

DT_CHUNK = 0.2
N_CHUNK = 100

_pxp_cache = {}


def _is_valid_state(state_int, n):
    prev_bit = 0
    for _ in range(n):
        bit = state_int & 1
        if bit == 1 and prev_bit == 1:
            return False
        prev_bit = bit
        state_int >>= 1
    return True


def build_pxp(n_qubits: int) -> dict:
    """Exact diagonalization of the PXP Hamiltonian (Rydberg blockade) for
    a chain of `n_qubits`, plus the Néel-state overlap tower and the
    constrained (Rydberg-blockade-valid) subspace mask. Cached per
    n_qubits -- call again with the same n_qubits and the cached dict is
    returned instantly instead of re-diagonalized."""
    if n_qubits in _pxp_cache:
        return _pxp_cache[n_qubits]

    dim = 2 ** n_qubits
    identity_1q = sp.identity(2, format="csr", dtype=np.complex128)
    x_1q = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    z_1q = sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)

    def op(o, q):
        mats = [identity_1q] * n_qubits
        mats[q] = o
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format="csr")
        return out

    z_cache = {q: op(z_1q, q) for q in range(n_qubits)}
    x_cache = {q: op(x_1q, q) for q in range(n_qubits)}
    identity = sp.identity(dim, format="csr", dtype=np.complex128)
    p_cache = {q: 0.5 * (identity + z_cache[q]) for q in range(n_qubits)}

    h_pxp = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(n_qubits):
        term = x_cache[i]
        if i > 0:
            term = p_cache[i - 1] @ term
        if i < n_qubits - 1:
            term = term @ p_cache[i + 1]
        h_pxp = h_pxp + term
    h_pxp = h_pxp.toarray()

    eigenvalues, eigenvectors = eigh(h_pxp)

    neel_bits = [0 if i % 2 == 0 else 1 for i in range(n_qubits)]
    neel_idx = int("".join(map(str, neel_bits)), 2)

    neel_overlap = np.abs(eigenvectors[neel_idx, :]) ** 2
    k = n_qubits + 1
    tower_indices = np.argsort(neel_overlap)[::-1][:k]
    tower_vectors = eigenvectors[:, tower_indices]
    tower_ceiling = float(neel_overlap[tower_indices].sum())

    valid_mask = np.array([_is_valid_state(i, n_qubits) for i in range(dim)], dtype=bool)

    result = {
        "n_qubits": n_qubits,
        "dim": dim,
        "h_pxp": h_pxp,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "eigenvectors_h": eigenvectors.conj().T,
        "neel_idx": neel_idx,
        "tower_vectors": tower_vectors,
        "tower_ceiling": tower_ceiling,
        "valid_mask": valid_mask,
        "valid_dim": int(valid_mask.sum()),
    }
    _pxp_cache[n_qubits] = result
    return result


def _propagate(pxp, sv, dt):
    c = pxp["eigenvectors_h"] @ sv
    c = c * np.exp(-1j * pxp["eigenvalues"] * dt)
    return pxp["eigenvectors"] @ c


def _project_tower(pxp, sv):
    v = pxp["tower_vectors"]
    sv_proj = v @ (v.conj().T @ sv)
    norm = np.linalg.norm(sv_proj)
    return sv_proj if norm < 1e-12 else sv_proj / norm


def _project_constraint(pxp, sv):
    mask = pxp["valid_mask"]
    sv_proj = np.where(mask, sv, 0.0)
    norm = np.linalg.norm(sv_proj)
    return sv_proj if norm < 1e-12 else sv_proj / norm


def _invalid_weight(pxp, sv):
    return float(np.sum(np.abs(sv[~pxp["valid_mask"]]) ** 2))


def run_experiment(pxp: dict, n_trajectories: int, noise_p: float, protection: str,
                    weight_threshold: float, base_seed: int) -> np.ndarray:
    """Averages `n_trajectories` noisy quantum trajectories starting from
    the Néel state, returning the fidelity revival |<Néel|psi(t)>|^2 at
    each of N_CHUNK timesteps (dt=DT_CHUNK). `protection` is one of
    'Nessuna', 'Proiezione vincolo (economica)', 'Proiezione torre (ideale)'."""
    dim = pxp["dim"]
    n_qubits = pxp["n_qubits"]
    neel = np.zeros(dim, dtype=np.complex128)
    neel[pxp["neel_idx"]] = 1.0

    fidelity_acc = np.zeros(N_CHUNK)
    for trajectory in range(n_trajectories):
        rng = np.random.default_rng(base_seed + trajectory)
        sv = neel.copy()
        for step in range(N_CHUNK):
            sv = _propagate(pxp, sv, DT_CHUNK)
            if noise_p > 0:
                sv = NoiseModel.apply_to_sv(sv, n=n_qubits, model="depolarizing", p=noise_p, rng=rng)
            if protection == "Proiezione vincolo (economica)" and _invalid_weight(pxp, sv) > weight_threshold:
                sv = _project_constraint(pxp, sv)
            elif protection == "Proiezione torre (ideale)":
                sv = _project_tower(pxp, sv)
            fidelity_acc[step] += np.abs(np.vdot(neel, sv)) ** 2

    return fidelity_acc / n_trajectories
