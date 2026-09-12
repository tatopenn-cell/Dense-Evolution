"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Sidebar organized into grouped sections (Costruisci / Risultati / Chimica /
Rumore / Sistema) instead of a single flat row of tabs -- the flat-tab
layout (still visible in this file's git history) stopped scaling once
Chimica/VQE and Rumore/ZNE were wired in: dashboard_core already exposes
these (see dashboard_core/__init__.py), they just were not connected to
any control here yet.

Every number shown here comes from a real run of dense_evolution's own
DenseSVSimulator/native VQE/ZNE machinery (dashboard_core), rendered with
dashboard_core.visuals -- no placeholder data anywhere.

Run with:
    pip install streamlit
    streamlit run app.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import dense_evolution
import streamlit as st

import dashboard_core as dc


@st.cache_data(show_spinner="Calcolo il catalogo molecole (Hartree-Fock reale, solo la prima volta)...")
def _cached_all_molecules():
    return dc.get_all_molecules()


st.set_page_config(
    page_title=f"Dense Evolution v{dense_evolution.__version__} - Dashboard",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Dense Evolution — Dashboard")

# ── Barra di salute, sempre visibile in cima alla sidebar ───────────────
with st.sidebar:
    limits = dc.max_safe_dense_qubits()
    st.caption(
        f"🖥️ {limits['available_mb']:,.0f} / {limits['total_mb']:,.0f} MB liberi · "
        f"fino a **{limits['max_qubits_dense']} qubit** in denso su questa macchina"
    )
    st.divider()
    section = st.radio(
        "Sezione", ["Costruisci", "Risultati", "Chimica", "Rumore", "Sistema"],
        key="nav_section",
    )
    st.divider()

    if section == "Costruisci":
        st.header("Circuito")
        preset_name = st.selectbox(
            "Preset", ["Custom"] + list(dc.QASM_LIBRARY.keys()), key="preset_select",
        )
        default_qasm = dc.QASM_LIBRARY.get(preset_name, dc.QASM_LIBRARY["Bell state (2 qubit)"])
        qasm_text = st.text_area(
            "OpenQASM 2.0", value=default_qasm, height=220, key=f"qasm_text__{preset_name}",
        )
        n_shots = st.number_input("Shots", min_value=1, max_value=100_000, value=1000, step=100)
        seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1)
        run_clicked = st.button("▶ Esegui", type="primary", width="stretch")
    else:
        run_clicked = False

if run_clicked:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            qasm_text, n_shots=int(n_shots), seed=int(seed),
        )
        st.session_state["error"] = None
    except Exception as exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(exc)

result = st.session_state.get("result")
error = st.session_state.get("error")


# ── COSTRUISCI ───────────────────────────────────────────────────────────
if section == "Costruisci":
    st.caption(
        "Stato attuale: " + (
            f"circuito a {result.n_qubits} qubit già eseguito — vai su Risultati, "
            "oppure aggiungi rumore in Rumore." if result is not None and not error
            else "premi ▶ Esegui nella sidebar per lanciare il circuito Bell precaricato."
        )
    )
    tab_builder, tab_circuit = st.tabs(["Editor Grafico", "Circuito"])

    with tab_builder:
        st.caption(
            "Trascina le porte sulla griglia per costruire un circuito a mano -- "
            "un controllo ● e un target nella stessa colonna formano un gate a 2 qubit, "
            "due × nella stessa colonna formano uno SWAP."
        )
        n_qubits_builder = st.number_input(
            "Qubit", min_value=1, max_value=8, value=3, step=1, key="n_qubits_builder",
        )
        builder_ops = dc.mount_circuit_builder(
            int(n_qubits_builder), n_columns=12, key=f"circuit_builder_{int(n_qubits_builder)}",
        )
        if st.button("→ Carica nel Circuit Editor"):
            if not builder_ops:
                st.warning("Nessuna porta piazzata sulla griglia.")
            else:
                native_ops = dc.ops_to_native_tuples(int(n_qubits_builder), builder_ops)
                st.session_state["preset_select"] = "Custom"
                st.session_state["qasm_text__Custom"] = dc.gate_tuples_to_qasm(native_ops, int(n_qubits_builder))
                st.rerun()

    with tab_circuit:
        if result is None:
            st.info("Premi ▶ Esegui nella sidebar per lanciare il circuito sul motore reale.")
        elif error:
            st.error(f"Errore nell'esecuzione del circuito: {error}")
        else:
            st.pyplot(dc.draw_circuit_figure(result.ops, result.n_qubits))

    if error:
        st.error(f"Errore nell'esecuzione del circuito: {error}")


# ── RISULTATI ────────────────────────────────────────────────────────────
elif section == "Risultati":
    if result is None:
        st.info("Nessun circuito eseguito ancora — vai su Costruisci e premi ▶ Esegui.")
    elif error:
        st.error(f"Errore nell'esecuzione del circuito: {error}")
    else:
        st.caption(
            "Stato attuale: circuito a "
            f"{result.n_qubits} qubit — puoi aggiungere rumore (Rumore) o costruirne "
            "uno nuovo (Costruisci)."
        )
        tab_sv, tab_prob, tab_qsphere, tab_bloch = st.tabs(
            ["Statevector", "Probabilità", "Q-sphere", "Bloch per qubit"]
        )
        with tab_sv:
            st.caption(f"{result.n_qubits} qubit — {len(result.statevector)} ampiezze (convenzione Qiskit)")
            rows = [
                {
                    "state": format(i, f"0{result.n_qubits}b"),
                    "amplitude (re)": float(amp.real),
                    "amplitude (im)": float(amp.imag),
                    "|amplitude|": abs(amp),
                    "phase (rad)": float(np.angle(amp)),
                }
                for i, amp in enumerate(result.statevector)
                if abs(amp) > 1e-10
            ]
            st.dataframe(rows, width="stretch")
        with tab_prob:
            st.caption("1000 shot reali campionati dallo statevector calcolato")
            st.pyplot(dc.histogram_figure(result.counts))
        with tab_qsphere:
            st.pyplot(dc.qsphere_figure(result.statevector))
        with tab_bloch:
            st.caption("Una sfera di Bloch per ogni qubit -- dalla sua matrice densità ridotta.")
            st.pyplot(dc.bloch_multivector_figure(result.statevector))


# ── CHIMICA ──────────────────────────────────────────────────────────────
elif section == "Chimica":
    st.header("Molecole & Hamiltoniane")
    molecules = dc.get_all_molecules()
    mol_name = st.selectbox("Molecola", list(molecules.keys()), key="mol_select")
    mol = molecules[mol_name]
    st.caption(f"Questa molecola userà **{mol['n_qubits']} qubit**.")

    if st.button("Costruisci Hamiltoniana"):
        try:
            H_dense, n_qubits = dc.build_molecular_hamiltonian(
                mol["symbols"], mol["geometry"], mol["charge"],
            )
            st.session_state["ham_result"] = (H_dense, n_qubits)
            st.session_state["ham_error"] = None
        except Exception as exc:
            st.session_state["ham_result"] = None
            st.session_state["ham_error"] = str(exc)

    ham_error = st.session_state.get("ham_error")
    ham_result = st.session_state.get("ham_result")
    if ham_error:
        st.error(f"Errore: {ham_error}")
    elif ham_result is not None:
        H_dense, n_qubits = ham_result
        exact_e = dc.ground_state_energy(H_dense)
        st.metric("Energia esatta di riferimento (Hartree)", f"{exact_e:.6f}")

    st.divider()
    st.header("VQE")
    ansatz_type = st.selectbox("Ansatz", ["hardware_efficient", "uccsd"], key="vqe_ansatz")
    n_layers = st.slider("n_layers (solo hardware_efficient)", 1, 12, 4, key="vqe_n_layers")
    maxiter = st.slider("Iterazioni (Adam)", 0, 500, 150, key="vqe_maxiter")
    with st.expander("Avanzate"):
        step_size = st.number_input("step_size", value=0.1, format="%.4f", key="vqe_step_size")
        seed_vqe = st.number_input("seed", value=0, step=1, key="vqe_seed")

    if st.button("Esegui VQE", type="primary"):
        try:
            vqe_result = dc.run_vqe(
                mol["symbols"], mol["geometry"], mol["charge"],
                ansatz_type=ansatz_type, n_layers=int(n_layers), maxiter=int(maxiter),
                step_size=float(step_size), seed=int(seed_vqe),
            )
            st.session_state["vqe_result"] = vqe_result
            st.session_state["vqe_error"] = None
        except Exception as exc:
            st.session_state["vqe_result"] = None
            st.session_state["vqe_error"] = str(exc)

    vqe_error = st.session_state.get("vqe_error")
    vqe_result = st.session_state.get("vqe_result")
    if vqe_error:
        st.error(f"Errore VQE: {vqe_error}")
    elif vqe_result is not None:
        col1, col2 = st.columns(2)
        col1.metric("Energia VQE (Hartree)", f"{vqe_result['vqe_energy_hartree']:.6f}")
        if vqe_result["exact_energy_hartree"] is not None:
            col2.metric("Energia esatta (Hartree)", f"{vqe_result['exact_energy_hartree']:.6f}")
        st.line_chart(vqe_result["energy_history"])
        st.caption(
            "L'energia converge verso lo stato fondamentale reale della molecola — "
            "più vicina a zero la differenza rispetto all'energia esatta, meglio ha "
            "approssimato il circuito."
        )
        if st.button("→ Carica il circuito VQE nell'Editor QASM"):
            st.session_state["preset_select"] = "Custom"
            st.session_state["qasm_text__Custom"] = vqe_result["qasm"]
            st.session_state["nav_section"] = "Costruisci"
            st.rerun()


# ── RUMORE ───────────────────────────────────────────────────────────────
elif section == "Rumore":
    st.header("Mitigazione ZNE")
    if result is None:
        st.info(
            "Serve un circuito già eseguito -- vai su Costruisci, premi ▶ Esegui, "
            "poi torna qui."
        )
    else:
        pauli_string = st.text_input(
            "Osservabile Pauli (lunga quanto i qubit, es. 'Z' o 'ZZ')",
            value="Z" * result.n_qubits, key="zne_pauli",
        )
        noise_model = st.selectbox(
            "Modello di rumore", ["depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"],
            key="zne_noise_model",
        )
        noise_p = st.slider("Intensità rumore (p)", 0.0, 0.5, 0.05, key="zne_noise_p")
        extrapolation_method = st.selectbox(
            "Metodo di estrapolazione", ["richardson", "polynomial"], key="zne_method",
        )
        n_trials = st.slider("n_trials (media Monte Carlo per punto)", 20, 500, 200, key="zne_n_trials")

        if st.button("Applica mitigazione ZNE", type="primary"):
            qasm_for_zne = dc.gate_tuples_to_qasm(result.ops, result.n_qubits)
            try:
                zne_result = dc.run_zne_mitigation(
                    qasm_for_zne, pauli_string=pauli_string, noise_model=noise_model,
                    noise_p=float(noise_p), extrapolation_method=extrapolation_method,
                    n_trials=int(n_trials),
                )
                st.session_state["zne_result"] = zne_result
                st.session_state["zne_error"] = None
            except Exception as exc:
                st.session_state["zne_result"] = None
                st.session_state["zne_error"] = str(exc)

        zne_error = st.session_state.get("zne_error")
        zne_result = st.session_state.get("zne_result")
        if zne_error:
            st.error(f"Errore: {zne_error}")
        elif zne_result is not None:
            col1, col2 = st.columns(2)
            col1.metric("Valore ideale", f"{zne_result.ideal_expectation:.4f}")
            col2.metric("Estrapolato a rumore zero", f"{zne_result.zne_extrapolated:.4f}")
            chart_data = {
                str(f): v for f, v in zip(zne_result.noise_factors, zne_result.noisy_expectations)
            }
            st.bar_chart(chart_data)
            st.caption(
                "Ogni barra è il valore misurato a quella scala di rumore (1x, 2x, 3x...) -- "
                "l'estrapolazione stima cosa accadrebbe a rumore zero."
            )


# ── SISTEMA ──────────────────────────────────────────────────────────────
elif section == "Sistema":
    st.header("Limiti di questa macchina")
    limits = dc.max_safe_dense_qubits()
    col1, col2, col3 = st.columns(3)
    col1.metric("RAM disponibile", f"{limits['available_mb']:,.0f} MB")
    col2.metric("RAM totale", f"{limits['total_mb']:,.0f} MB")
    col3.metric("Qubit massimi in denso", limits["max_qubits_dense"])
    st.caption(
        f"Soglia di sicurezza: {limits['threshold_pct'] * 100:.0f}% di RAM libera "
        "richiesta dopo ogni allocazione -- calcolata dal vero stato di questa macchina, "
        "non un numero fisso."
    )
