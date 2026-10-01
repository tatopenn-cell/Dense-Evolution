"""
Quantum Scars page — interactive live demo of the PXP model (Rydberg
blockade), the system studied in the "quantum_scar_investigation" report:
github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/tree/main/scripts/quantum_scar_investigation

That investigation found no genuine many-body scar in Dense Evolution's
frustrated Ising grids (wrong observable + gauge equivalence), then
validated the same verification pipeline against the PXP model, where
scars are real and well documented. This page lets you reproduce the core
PXP result interactively: start from the Néel state, inject real
depolarizing noise via NoiseModel.apply_to_sv, and compare the fidelity
revival with/without protection.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import streamlit as st

from dense_evolution.registry import NoiseModel
from ui_pages.components import render_metric_grid, render_page_banner, render_run_guard, render_matplotlib_figure

DT_CHUNK = 0.2
N_CHUNK = 100


def _is_valid_state(state_int, n):
    prev_bit = 0
    for _ in range(n):
        bit = state_int & 1
        if bit == 1 and prev_bit == 1:
            return False
        prev_bit = bit
        state_int >>= 1
    return True


# st.cache_resource (first use in this codebase): exact diagonalization here
# is scipy.linalg.eigh on a dense 2**n_qubits matrix -- genuinely expensive
# and depends only on n_qubits, so it's cached in memory across reruns and
# widget interactions instead of recomputed on every slider tweak.
@st.cache_resource(show_spinner="Costruzione H_PXP e diagonalizzazione esatta...")
def _build_pxp(n_qubits: int):
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

    return {
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


def _run_experiment(pxp, n_trajectories, noise_p, protection, weight_threshold, base_seed):
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


def render():
    render_page_banner(
        "Quantum Many-Body Scars: PXP Live Demo",
        """Dinamica esatta del modello PXP (blocco di Rydberg), lo stesso sistema usato per
        validare la pipeline di verifica nell'indagine
        <a href="https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/tree/main/scripts/quantum_scar_investigation"
           style="color:#00e5ff;" target="_blank">quantum_scar_investigation</a>.
        Parte dallo stato di Néel, inietta rumore reale
        (<code>dense_evolution.registry.NoiseModel.apply_to_sv</code>, media su più
        traiettorie quantistiche) e confronta i revival di fedeltà con/senza protezione.""",
        accent="#00e5ff", bg_from="#001014", bg_to="#012026",
    )

    with st.sidebar:
        st.header("⚙️ Configurazione")
        n_qubits = st.slider(
            "Numero di qubit (catena PXP)", min_value=6, max_value=12, value=10,
            help="Oltre 10 la diagonalizzazione esatta (e la simulazione) rallenta sensibilmente.",
        )
        noise_p = st.slider("Rumore depolarizzante per sito (p)", 0.0, 0.05, 0.01, step=0.005)
        n_trajectories = st.slider("Traiettorie quantistiche (media)", 5, 30, 10)
        protection = st.selectbox(
            "Protezione",
            ["Nessuna", "Proiezione vincolo (economica)", "Proiezione torre (ideale)"],
            help=(
                "Vincolo: nessuna diagonalizzazione aggiuntiva, economica ma parziale. "
                "Torre: limite teorico, richiede lo spettro esatto — non realizzabile su "
                "hardware reale così com'è."
            ),
        )
        run_clicked = st.button("🚀 Esegui esperimento", type="primary")

    if run_clicked:
        pxp = _build_pxp(n_qubits)
        with st.spinner("Simulazione delle traiettorie quantistiche in corso..."):
            fidelity_protected = _run_experiment(
                pxp, n_trajectories=n_trajectories, noise_p=noise_p,
                protection=protection, weight_threshold=0.02, base_seed=1000,
            )
            fidelity_clean = _run_experiment(
                pxp, n_trajectories=1, noise_p=0.0,
                protection="Nessuna", weight_threshold=0.02, base_seed=0,
            )
        st.session_state["scar_result"] = {
            "n_qubits": n_qubits, "noise_p": noise_p, "protection": protection,
            "fidelity_protected": fidelity_protected, "fidelity_clean": fidelity_clean,
            "valid_dim": pxp["valid_dim"], "dim": pxp["dim"], "tower_ceiling": pxp["tower_ceiling"],
        }

    result = render_run_guard(
        "scar_result",
        message="Configura i parametri nella sidebar e premi **Esegui esperimento** per iniziare.",
    )
    if result is None:
        return

    render_metric_grid([
        {"label": "Qubit", "value": result["n_qubits"]},
        {"label": "Dimensione Hilbert", "value": result["dim"]},
        {"label": "Sottospazio valido (vincolo)", "value": result["valid_dim"]},
        {"label": "Soffitto torre (peso Néel)", "value": f"{result['tower_ceiling']*100:.1f}%",
         "help": "Peso massimo che lo stato di Néel puro ha sulla torre di N+1 autostati — "
                 "il limite teorico che nessuna correzione può superare."},
    ])

    with st.container(border=True):
        st.subheader("📈 Fedeltà di revival")
        times = np.arange(1, N_CHUNK + 1) * DT_CHUNK
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(times, result["fidelity_clean"], label="Nessun rumore (riferimento)",
                color="gold", linewidth=1.5, linestyle="--")
        ax.plot(times, result["fidelity_protected"],
                label=f"p={result['noise_p']} — {result['protection']}",
                color="#00e5ff", linewidth=2.2)
        ax.set_xlabel("Tempo t")
        ax.set_ylabel("Fedeltà |⟨Néel|ψ(t)⟩|²")
        ax.set_title("Revival della scar PXP")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        render_matplotlib_figure(fig)

    st.caption(
        "Report completo dell'indagine (inclusa la scar smentita nell'Ising frustrato, "
        "lo scan sistematico e i tentativi di protezione economica): "
        "[quantum_scar_investigation su GitHub]"
        "(https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/tree/main/scripts/quantum_scar_investigation)."
    )
