"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Sidebar organized into grouped sections instead of a single flat row of
tabs. Base sections (Costruisci / Risultati / Chimica / Rumore / Sistema)
are always visible; a "Modalità avanzata" toggle in the sidebar reveals
the rest (Dinamica / Wormhole / QEC / Magia & Divergenze / Materia
Condensata / Importa Circuito) -- per the project's own "principio di
semplicità": a first-time visitor sees 5 sections, not 11.

Every number shown here comes from a real run of dense_evolution's own
engine (dashboard_core + dense_evolution's public API) -- no placeholder
data anywhere. Where a real feature could not be exposed safely (a
free-text "paste Qiskit/PennyLane code" importer would need to exec()
arbitrary Python -- a genuine code-injection surface), it is scoped down
instead of faked; see the Importa Circuito section's own comment.

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

from dense_evolution.mitigation.magic_entropy import magic_entropy
from dense_evolution.mitigation.stabilizer_renyi_entropy import stabilizer_renyi_entropy
from dense_evolution.native_hf.differentiable import build_energy_fn

ANGSTROM_TO_BOHR = 1.8897259886


@st.cache_data(show_spinner="Calcolo il catalogo molecole (Hartree-Fock reale, solo la prima volta)...")
def _cached_all_molecules():
    return dc.get_all_molecules()


st.set_page_config(
    page_title=f"Dense Evolution v{dense_evolution.__version__} - Dashboard",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Dense Evolution — Dashboard")

# ── Stato iniziale non vuoto: il Bell state gira già al primo avvio ─────
if "result" not in st.session_state:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            dc.QASM_LIBRARY["Bell state (2 qubit)"], n_shots=1000, seed=42,
        )
        st.session_state["error"] = None
    except Exception as _exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(_exc)

_BASE_SECTIONS = ["Costruisci", "Risultati", "Chimica", "Rumore", "Sistema"]
_ADVANCED_SECTIONS = [
    "Dinamica", "Wormhole", "QEC", "Magia & Divergenze", "Materia Condensata",
    "Grandi Sistemi", "Importa Circuito",
]

# ── Barra di salute, sempre visibile in cima alla sidebar ───────────────
with st.sidebar:
    limits = dc.max_safe_dense_qubits()
    st.caption(
        f"🖥️ {limits['available_mb']:,.0f} / {limits['total_mb']:,.0f} MB liberi · "
        f"fino a **{limits['max_qubits_dense']} qubit** in denso su questa macchina"
    )
    st.divider()
    advanced_mode = st.toggle("Modalità avanzata", key="advanced_mode")
    all_sections = _BASE_SECTIONS + (_ADVANCED_SECTIONS if advanced_mode else [])
    if st.session_state.get("nav_section") not in all_sections:
        st.session_state["nav_section"] = "Costruisci"
    section = st.radio("Sezione", all_sections, key="nav_section")
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
        with st.expander("Rumore & backend"):
            run_noise_model = st.selectbox(
                "Modello di rumore", ["ideal", "depolarizing", "bitflip", "phaseflip",
                                      "amplitude_damping", "combined"],
                key="run_noise_model",
            )
            run_noise_p = st.slider(
                "Intensità rumore (p)", 0.0, 1.0, 0.0, key="run_noise_p",
                disabled=(run_noise_model == "ideal"),
            )
            run_backend = st.selectbox(
                "Backend", ["dense", "mps"], key="run_backend",
                help="mps (Matrix Product State) usa meno memoria su circuiti poco "
                     "entangled, fino a 24 qubit -- oltre, serve la modalità MPS di "
                     "Grandi Sistemi (non ancora collegata qui).",
            )
        run_clicked = st.button("▶ Esegui", type="primary", width="stretch")
    else:
        run_clicked = False

if run_clicked:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            qasm_text, n_shots=int(n_shots), seed=int(seed),
            noise_model=run_noise_model, noise_p=float(run_noise_p), backend=run_backend,
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
            else "premi ▶ Esegui nella sidebar per lanciare un nuovo circuito."
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
        if result.fidelity_vs_ideal is not None or result.backend == "mps":
            cols = st.columns(4)
            i = 0
            if result.fidelity_vs_ideal is not None:
                cols[i].metric("Fedeltà vs. ideale", f"{result.fidelity_vs_ideal:.4f}")
                i += 1
            if result.backend == "mps":
                cols[i].metric("Backend", "MPS")
                cols[i + 1].metric("Bond massimo usato", result.mps_max_bond_used)
                cols[i + 2].metric("Memoria MPS (MB)", f"{result.mps_memory_mb:.2f}")
            if result.fidelity_vs_ideal is not None:
                st.caption("Fedeltà = quanto il run rumoroso si discosta dal circuito ideale (1.0 = identico).")
        tab_sv, tab_prob, tab_qsphere, tab_bloch, tab_entropy = st.tabs(
            ["Statevector", "Probabilità", "Q-sphere", "Bloch per qubit", "Entropia & informazione mutua"]
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
        with tab_entropy:
            st.caption(
                "Un qubit entangled ha <Z>=0 anche se il suo partner ha subito un'operazione -- "
                "l'informazione mutua vede correlazioni che un valore di aspettazione singolo non "
                "può vedere per costruzione (teorema no-signaling). Numerazione qubit: come in "
                "Risultati (convenzione Qiskit)."
            )
            all_qubits = list(range(result.n_qubits))
            qa_text = st.text_input("Sottosistema A (indici separati da virgola)", value="0", key="entropy_a")
            qb_text = st.text_input("Sottosistema B (indici separati da virgola)", value="1" if result.n_qubits > 1 else "", key="entropy_b")
            if st.button("Calcola entropia e informazione mutua"):
                try:
                    qa = [int(x) for x in qa_text.split(",") if x.strip() != ""]
                    qb = [int(x) for x in qb_text.split(",") if x.strip() != ""]
                    if set(qa) & set(qb):
                        raise ValueError("i due sottosistemi devono essere disgiunti")
                    # Same Qiskit-little-endian -> native-MSB-first conversion
                    # as Magia & Divergenze: flip each index before calling
                    # partial_trace/mutual_information, which use dense_evolution's
                    # own convention, not Qiskit's.
                    n = result.n_qubits
                    native_a = [n - 1 - q for q in qa]
                    native_b = [n - 1 - q for q in qb]
                    rho_a = dense_evolution.partial_trace(result.statevector, n, native_a)
                    rho_b = dense_evolution.partial_trace(result.statevector, n, native_b)
                    s_a = dense_evolution.von_neumann_entropy(rho_a)
                    s_b = dense_evolution.von_neumann_entropy(rho_b)
                    i_ab = dense_evolution.mutual_information(result.statevector, n, native_a, native_b)
                    st.session_state["entropy_result"] = (s_a, s_b, i_ab)
                    st.session_state["entropy_error"] = None
                except Exception as exc:
                    st.session_state["entropy_result"] = None
                    st.session_state["entropy_error"] = str(exc)

            entropy_error = st.session_state.get("entropy_error")
            entropy_result = st.session_state.get("entropy_result")
            if entropy_error:
                st.error(f"Errore: {entropy_error}")
            elif entropy_result is not None:
                s_a, s_b, i_ab = entropy_result
                col1, col2, col3 = st.columns(3)
                col1.metric("S(A) (nat)", f"{s_a:.4f}")
                col2.metric("S(B) (nat)", f"{s_b:.4f}")
                col3.metric("I(A:B) (nat)", f"{i_ab:.4f}")


# ── CHIMICA ──────────────────────────────────────────────────────────────
elif section == "Chimica":
    st.header("Molecole & Hamiltoniane")
    molecules = _cached_all_molecules()
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
        # Heuristic flattening check, not a formal Barren Plateau detector:
        # BP is a training-time property (gradient variance vanishing with
        # system size); this only looks at whether energy_history's own
        # tail stopped moving, using data run_vqe already returns -- an
        # honest proxy for "the optimizer may be stuck", not a claim to
        # have measured the real phenomenon.
        history = np.asarray(vqe_result["energy_history"], dtype=np.float64)
        if len(history) >= 10:
            tail = history[-max(5, len(history) // 5):]
            spread = float(np.ptp(history)) or 1.0
            tail_std = float(np.std(np.diff(tail)))
            if tail_std < 1e-4 * spread:
                st.warning(
                    "La curva si è appiattita nelle ultime iterazioni (variazione energia "
                    "quasi nulla) — possibile plateau o minimo locale, non necessariamente "
                    "il minimo globale. Prova a ridurre n_layers o usa l'ansatz UCCSD."
                )
        if st.button("→ Carica il circuito VQE nell'Editor QASM"):
            st.session_state["preset_select"] = "Custom"
            st.session_state["qasm_text__Custom"] = vqe_result["qasm"]
            st.session_state["nav_section"] = "Costruisci"
            st.rerun()

    if advanced_mode:
        st.divider()
        with st.expander("Ottimizza geometria (dE/dR analitico, native_hf)"):
            st.caption(
                "Rilassa le posizioni nucleari verso il minimo di energia usando il "
                "gradiente analitico dE/dR (native_hf, PR #238) -- diverso da "
                "run_md_trajectory (Dinamica), che muove i nuclei con dinamica "
                "Newtoniana a stato elettronico fisso, non verso un minimo."
            )
            n_geom_steps = st.slider("Passi di discesa", 1, 50, 15, key="geom_opt_steps")
            geom_lr = st.number_input("Passo (learning rate, Bohr)", value=0.05, format="%.3f", key="geom_opt_lr")
            if st.button("Ottimizza geometria"):
                try:
                    import jax
                    from basis_set_exchange.lut import element_Z_from_sym

                    atomic_numbers = [element_Z_from_sym(s) for s in mol["symbols"]]
                    n_electrons = sum(atomic_numbers) - mol["charge"]
                    geometry_bohr = np.asarray(mol["geometry"], dtype=np.float64) * ANGSTROM_TO_BOHR
                    energy_fn = build_energy_fn(
                        atomic_numbers, [float(z) for z in atomic_numbers], n_electrons,
                        "sto-3g", geometry_bohr,
                    )
                    grad_fn = jax.grad(energy_fn)
                    geom = geometry_bohr.copy()
                    history = [float(energy_fn(geom))]
                    for _ in range(int(n_geom_steps)):
                        g = np.asarray(grad_fn(geom))
                        geom = geom - float(geom_lr) * g
                        history.append(float(energy_fn(geom)))
                    st.session_state["geom_opt_result"] = {
                        "history": history,
                        "geometry_angstrom": (geom / ANGSTROM_TO_BOHR).tolist(),
                    }
                    st.session_state["geom_opt_error"] = None
                except Exception as exc:
                    st.session_state["geom_opt_result"] = None
                    st.session_state["geom_opt_error"] = str(exc)

            geom_error = st.session_state.get("geom_opt_error")
            geom_res = st.session_state.get("geom_opt_result")
            if geom_error:
                st.error(f"Errore: {geom_error}")
            elif geom_res is not None:
                st.line_chart(geom_res["history"])
                st.caption(
                    f"Energia: {geom_res['history'][0]:.6f} -> {geom_res['history'][-1]:.6f} Hartree "
                    "(prima compilazione JAX può richiedere fino a qualche minuto)."
                )


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

    st.divider()
    st.header("Canali & rumore fisico")
    tab_cosmic, tab_osc, tab_dm, tab_heal = st.tabs(
        ["Raffica raggi cosmici", "Rumore oscillante", "Canale su matrice densità", "Ripara vettore"]
    )
    with tab_cosmic:
        st.caption(
            "Riproduce un evento reale misurato su un chip a 26 qubit (arXiv:2104.05219), "
            "non uno scenario inventato."
        )
        baseline_gamma = st.number_input("baseline_gamma", value=1e-4, format="%.6f", key="cosmic_gamma")
        if st.button("Simula raffica"):
            cosmic_result = dc.run_cosmic_ray_burst(float(baseline_gamma))
            st.bar_chart({str(t): p for t, p in zip(cosmic_result.times_us, cosmic_result.decay_probabilities)})
            st.caption(f"Rapporto picco/base: {cosmic_result.peak_ratio:.2f}x")
    with tab_osc:
        base_p = st.slider("base_p", 0.0, 0.5, 0.05, key="osc_base_p")
        freq = st.slider("freq", 0.1, 5.0, 1.0, key="osc_freq")
        amp = st.slider("amp", 0.0, 0.5, 0.1, key="osc_amp")
        if st.button("Simula rumore oscillante"):
            osc_result = dc.run_oscillating_noise(float(base_p), float(freq), float(amp))
            st.line_chart({str(f): p for f, p in zip(osc_result.factors, osc_result.p_eff)})
            st.caption("Se questa curva non è liscia, l'ipotesi di rumore liscio che ZNE assume si rompe.")
    with tab_dm:
        if result is None:
            st.info("Serve un circuito già eseguito.")
        else:
            channel = st.selectbox("Canale", ["global_depolarizing", "amplitude_damping"], key="dm_channel")
            param = st.slider("Parametro del canale", 0.0, 1.0, 0.1, key="dm_param")
            if st.button("Applica canale"):
                qasm_for_dm = dc.gate_tuples_to_qasm(result.ops, result.n_qubits)
                try:
                    dm_result = dc.run_density_matrix_channel(qasm_for_dm, channel, float(param))
                    st.bar_chart({
                        "ideale": dm_result.ideal_diagonal, "rumoroso": dm_result.noisy_diagonal,
                    })
                except Exception as exc:
                    st.error(f"Errore: {exc}")
    with tab_heal:
        if vqe_result := st.session_state.get("vqe_result"):
            st.caption("Ripara la traiettoria di energia dell'ultimo VQE eseguito in Chimica.")
            if st.button("Ripara vettore"):
                try:
                    vectors = np.asarray(vqe_result["energy_history"], dtype=np.float64).reshape(-1, 1)
                    heal_result = dc.run_vector_healing(vectors)
                    st.line_chart({
                        "originale": [v[0] for v in vectors.tolist()],
                        "riparato": [v[0] for v in heal_result.healed_vectors],
                    })
                    st.caption(
                        f"Correzione attivata: {heal_result.fallback_triggered} · "
                        f"errore di ricostruzione: {heal_result.reconstruction_error:.4g}"
                    )
                except ImportError as exc:
                    st.error(f"Richiede il pacchetto ia_utils: {exc}")
        else:
            st.info("Esegui prima un VQE in Chimica per avere una traiettoria da riparare.")


# ── DINAMICA ─────────────────────────────────────────────────────────────
elif section == "Dinamica":
    st.header("QM/MM & Traiettoria MD")
    molecules = _cached_all_molecules()
    mol_name = st.selectbox("Molecola", list(molecules.keys()), key="dyn_mol_select")

    if st.button("Calcola forze"):
        try:
            forces_result = dc.compute_hellmann_feynman_forces(mol_name)
            st.session_state["forces_result"] = forces_result
            st.session_state["forces_error"] = None
        except Exception as exc:
            st.session_state["forces_result"] = None
            st.session_state["forces_error"] = str(exc)

    forces_error = st.session_state.get("forces_error")
    forces_result = st.session_state.get("forces_result")
    if forces_error:
        st.error(f"Errore: {forces_error}")
    elif forces_result is not None:
        col1, col2 = st.columns(2)
        col1.metric("Energia (Hartree)", f"{forces_result['energy_hartree']:.6f}")
        col2.metric("Norma forza (Hartree/Å)", f"{forces_result['force_norm']:.6f}")

    st.divider()
    n_steps = st.slider("n_steps", 1, 50, 10, key="md_n_steps")
    dt_fs = st.number_input("dt_fs", value=0.5, format="%.3f", key="md_dt_fs")
    recompute = st.checkbox(
        "Ricalcola stato elettronico a ogni step (molto più lento, forze esatte anche lontano "
        "dalla geometria di partenza)", value=False, key="md_recompute",
    )
    if st.button("Avvia traiettoria MD", type="primary"):
        try:
            traj = dc.run_md_trajectory(
                mol_name, int(n_steps), dt_fs=float(dt_fs), recompute_electronic_state=recompute,
            )
            st.session_state["md_traj"] = traj
            st.session_state["md_error"] = None
        except Exception as exc:
            st.session_state["md_traj"] = None
            st.session_state["md_error"] = str(exc)

    md_error = st.session_state.get("md_error")
    md_traj = st.session_state.get("md_traj")
    if md_error:
        st.error(f"Errore: {md_error}")
    elif md_traj is not None:
        st.line_chart({"time_fs": md_traj["time_fs"], "energy_hartree": md_traj["energy_hartree"]})
        st.line_chart({"time_fs": md_traj["time_fs"], "force_norm": md_traj["force_norm"]})
        st.caption(
            "Energia e norma della forza lungo la traiettoria -- se la simulazione diverge "
            "(nuclei troppo vicini), la funzione stessa lo segnala con un errore chiaro invece "
            "di continuare silenziosamente."
        )


# ── WORMHOLE ─────────────────────────────────────────────────────────────
elif section == "Wormhole":
    st.header("Protocollo SYK / Teletrasporto")
    st.caption(
        "Traversable-wormhole-inspired quantum teleportation (arXiv:2604.10090) -- il segnale "
        "interessante è la DIFFERENZA tra mu positivo e negativo, non il valore singolo."
    )
    n_majorana = st.select_slider("n_majorana", options=[8, 12, 16, 20], value=8, key="wh_n_majorana")
    k_terms = st.slider("k_terms", 4, 20, 10, key="wh_k_terms")
    J = st.number_input("J", value=1.4142, format="%.4f", key="wh_J")
    mu = st.slider("mu", 1.0, 20.0, 12.0, key="wh_mu")
    t0 = st.number_input("t0", value=0.3, format="%.3f", key="wh_t0")
    t1 = st.number_input("t1", value=0.60, format="%.3f", key="wh_t1")
    with_message = st.checkbox("Inietta messaggio (with_message)", value=True, key="wh_with_message")
    method = st.selectbox("Metodo", ["esatto", "Trotter"], key="wh_method")

    if st.button("Trova una buona istanza"):
        seed = dc.select_good_instance(int(n_majorana), int(k_terms), float(J))
        st.session_state["wh_seed"] = seed
    seed = st.session_state.get("wh_seed", 61)
    st.caption(f"Seed in uso: **{seed}** (Trova una buona istanza cerca quello che dà il segnale più pulito)")

    if st.button("Esegui protocollo", type="primary"):
        fn = dc.run_wormhole_protocol_trotter if method == "Trotter" else dc.run_wormhole_protocol
        try:
            i_plus = fn(int(n_majorana), int(k_terms), float(J), float(mu), float(t0), float(t1), int(seed), with_message)
            i_minus = fn(int(n_majorana), int(k_terms), float(J), -float(mu), float(t0), float(t1), int(seed), with_message)
            st.session_state["wh_result"] = (i_plus, i_minus)
            st.session_state["wh_error"] = None
        except Exception as exc:
            st.session_state["wh_result"] = None
            st.session_state["wh_error"] = str(exc)

    wh_error = st.session_state.get("wh_error")
    wh_result = st.session_state.get("wh_result")
    if wh_error:
        st.error(f"Errore: {wh_error}")
    elif wh_result is not None:
        i_plus, i_minus = wh_result
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(4, 3))
        ax.barh(["mu > 0", "mu < 0"], [i_plus, -i_minus], color=["#648fff", "#dc267f"])
        ax.set_xlabel("informazione mutua (con segno)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"I(mu=+{mu:g}) = {i_plus:.5f} · I(mu=-{mu:g}) = {i_minus:.5f}")


# ── QEC ──────────────────────────────────────────────────────────────────
elif section == "QEC":
    st.header("Correzione di errori quantistici")
    stabilizers_text = st.text_area(
        "Generatori stabilizzatori (uno per riga, es. 'ZZI')", value="ZZI\nIZZ", key="qec_stabilizers",
    )
    stabilizers = [s.strip() for s in stabilizers_text.splitlines() if s.strip()]
    n_qubits_qec = len(stabilizers[0]) if stabilizers else 0
    st.caption(f"Codice a {n_qubits_qec} qubit fisici, {len(stabilizers)} stabilizzatori.")

    st.subheader("Calcolatrice sindrome")
    pauli_error = st.text_input(f"Errore Pauli ({n_qubits_qec} caratteri, es. 'IXI')", key="qec_error")
    if st.button("Calcola sindrome"):
        try:
            syndrome = dense_evolution.compute_syndrome(pauli_error, stabilizers)
            st.session_state["qec_syndrome"] = syndrome
        except Exception as exc:
            st.error(f"Errore: {exc}")
    if "qec_syndrome" in st.session_state:
        st.write(f"Sindrome: `{st.session_state['qec_syndrome']}`")

    st.subheader("Decodifica")
    observed_text = st.text_input("Sindrome osservata (es. '1,0')", value="1,0", key="qec_observed")
    heralded_text = st.text_input("Qubit erasi noti, opzionale (es. '2')", value="", key="qec_heralded")
    if st.button("Decodifica"):
        try:
            observed = tuple(int(x) for x in observed_text.split(",") if x.strip() != "")
            heralded = [int(x) for x in heralded_text.split(",") if x.strip() != ""]
            decoded = dense_evolution.decode_with_erasure_fallback(
                observed, heralded, n_qubits_qec, stabilizers,
            )
            st.session_state["qec_decoded"] = decoded
        except Exception as exc:
            st.error(f"Errore: {exc}")
    if "qec_decoded" in st.session_state:
        decoded = st.session_state["qec_decoded"]
        if decoded is None:
            st.warning("Sindrome ambigua o non risolvibile -- nessuna correzione univoca trovata.")
        else:
            st.success(f"Correzione trovata: `{decoded}`")


# ── MAGIA & DIVERGENZE ───────────────────────────────────────────────────
elif section == "Magia & Divergenze":
    st.header("Diagnostica magica")
    if result is None:
        st.info("Serve un circuito già eseguito -- vai su Costruisci.")
    else:
        st.metric("Stabilizer Renyi Entropy (tutto lo stato)", f"{stabilizer_renyi_entropy(result.statevector):.6f} bit")
        st.caption("Zero per ogni stato stabilizzatore, positiva per stati 'magici' (non-Clifford).")

        qubit_idx = st.number_input(
            "Qubit per magic_entropy (singolo qubit, numerazione Qiskit -- come in Risultati)",
            min_value=0, max_value=result.n_qubits - 1, value=0, key="magic_qubit",
        )
        # result.statevector is in Qiskit's little-endian order; partial_trace
        # uses dense_evolution's own MSB-first convention -- the two disagree
        # on which physical qubit index N means, so the Qiskit-facing index
        # has to be flipped before calling it (BUG FIX: this used to pass
        # qubit_idx straight through, silently tracing out a different qubit
        # than the one displayed everywhere else in the app).
        native_qubit_idx = result.n_qubits - 1 - int(qubit_idx)
        rho = dense_evolution.partial_trace(result.statevector, result.n_qubits, [native_qubit_idx])
        st.metric(f"magic_entropy (qubit {qubit_idx})", f"{magic_entropy(rho):.6f} bit")
        st.caption("Costruzione Key-Unitary a 3 copie -- 0 per |0>,|1>,|+>,|->,|+i>,|-i>, 0.811 per T e H.")


# ── MATERIA CONDENSATA ───────────────────────────────────────────────────
elif section == "Materia Condensata":
    st.header("Stato solido (tight-binding)")
    material = st.selectbox("Materiale", sorted(dense_evolution.VHD_MATERIALS.keys()), key="solid_material")
    if st.button("Calcola gap a Gamma"):
        gap = dense_evolution.direct_gap_at_gamma(material)
        st.metric(f"Gap diretto a Γ -- {material}", f"{gap:.4f} eV")
    if st.button("Scansiona banda Γ→X"):
        vbm, vbm_k, cbm, cbm_k, gap = dense_evolution.band_extrema_along_path(
            material, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("VBM (eV)", f"{vbm:.4f}")
        col2.metric("CBM (eV)", f"{cbm:.4f}")
        col3.metric("Gap reale (eV)", f"{gap:.4f}")
        st.caption(
            "Per materiali a gap indiretto (es. Si) questo è il numero corretto -- "
            "\"Calcola gap a Gamma\" da solo darebbe il valore sbagliato per quelli."
        )

    st.divider()
    st.header("Modello di Hubbard")
    n_sites = st.slider("n_sites", 2, 6, 4, key="hub_n_sites")
    t_hop = st.number_input("t (hopping)", value=1.0, format="%.3f", key="hub_t")
    U_int = st.number_input("U (repulsione on-site)", value=2.0, format="%.3f", key="hub_U")
    if st.button("Costruisci Hamiltoniana di Hubbard"):
        try:
            terms = dense_evolution.hubbard_hamiltonian_pauli_terms(int(n_sites), float(t_hop), float(U_int))
            n_qubits_hub = 2 * int(n_sites)
            H_hub = dense_evolution.pauli_hamiltonian_to_matrix(terms, n_qubits_hub)
            e0 = dc.ground_state_energy(H_hub)
            st.metric(f"Energia di stato fondamentale ({n_qubits_hub} qubit)", f"{e0:.6f}")
        except Exception as exc:
            st.error(f"Errore: {exc}")


# ── GRANDI SISTEMI ───────────────────────────────────────────────────────
elif section == "Grandi Sistemi":
    st.header("Grandi Sistemi (MPS oltre il limite denso)")
    st.caption(
        f"Sopra {dc.MPS_DENSE_CONTRACTION_LIMIT} qubit non esiste più un array "
        "statevector denso da costruire -- run_large_circuit_mps trova i "
        "k stati più probabili con probabilità ESATTE (non campionate), "
        "senza mai materializzare lo stato completo. Backend/rumore di "
        "Costruisci non si applicano qui: è un percorso di esecuzione separato."
    )
    default_large_qasm = dc.gate_tuples_to_qasm(dense_evolution.ghz_state(30), 30)
    large_qasm_text = st.text_area(
        "OpenQASM 2.0 (circuito grande)", value=default_large_qasm, height=200, key="large_qasm_text",
    )
    k_states = st.slider("k (quanti stati più probabili mostrare)", 1, 100, 32, key="large_k")
    large_seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1, key="large_seed")

    if st.button("Esegui in modalità MPS", type="primary"):
        try:
            large_result = dc.run_large_circuit_mps(large_qasm_text, k=int(k_states), seed=int(large_seed))
            st.session_state["large_mps_result"] = large_result
            st.session_state["large_mps_error"] = None
        except Exception as exc:
            st.session_state["large_mps_result"] = None
            st.session_state["large_mps_error"] = str(exc)

    large_error = st.session_state.get("large_mps_error")
    large_result = st.session_state.get("large_mps_result")
    if large_error:
        st.error(f"Errore: {large_error}")
    elif large_result is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Qubit", large_result.n_qubits)
        col2.metric("Bond massimo usato", large_result.mps_max_bond_used)
        col3.metric("Memoria MPS (MB)", f"{large_result.mps_memory_mb:.2f}")
        st.caption(f"Errore medio di troncamento (JSD): {large_result.mps_avg_jsd:.2e}")
        top_states_sorted = sorted(large_result.top_k_states, key=lambda t: -t[1])
        st.dataframe(
            [{"stato": bits, "probabilità": p} for bits, p in top_states_sorted],
            width="stretch",
        )

    with st.expander("Convergenza rispetto al bond dimension"):
        st.caption(
            "mps_avg_jsd sopra è un errore di troncamento medio su tutto lo stato -- non dice "
            "se l'osservabile che ti interessa si è davvero assestato al crescere del bond. "
            "Qui il circuito viene rieseguito a bond crescenti e si guarda se il valore atteso "
            "smette di cambiare. Se 'chi_used' al bond più alto è già uguale al suo tetto, il "
            "verdetto è 'undecidable': il troncamento non ha mai avuto margine per dimostrare "
            "la convergenza, non un converged/not_converged nascosto."
        )
        bc_qasm_text = st.text_area(
            "OpenQASM 2.0 (circuito per il check)", value=default_large_qasm, height=150, key="bc_qasm_text",
        )
        bc_observable = st.text_input(
            "Osservabile (stringa Pauli, es. ZIII)", value="Z" + "I" * 29, key="bc_observable",
        )
        bc_bonds_text = st.text_input("Bond da testare (crescenti, separati da virgola)", value="4,8,16,32", key="bc_bonds")
        bc_tol = st.number_input("Tolleranza", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f", key="bc_tol")
        if st.button("Esegui check di convergenza"):
            try:
                bonds = [int(b) for b in bc_bonds_text.split(",") if b.strip() != ""]
                bc_result = dc.run_bond_convergence_check(bc_qasm_text, [bc_observable], bonds, tol=float(bc_tol))
                st.session_state["bond_convergence_result"] = bc_result
                st.session_state["bond_convergence_error"] = None
            except Exception as exc:
                st.session_state["bond_convergence_result"] = None
                st.session_state["bond_convergence_error"] = str(exc)

        bc_error = st.session_state.get("bond_convergence_error")
        bc_result = st.session_state.get("bond_convergence_result")
        if bc_error:
            st.error(f"Errore: {bc_error}")
        elif bc_result is not None:
            verdict = bc_result.verdicts[0]
            verdict_label = {
                "converged": "✅ converged",
                "not_converged": "⚠️ not_converged",
                "undecidable": "❔ undecidable",
            }[verdict]
            st.metric("Verdetto", verdict_label)
            st.dataframe(
                [
                    {
                        "bond": b, "chi_used": c, "avg_jsd": j, "budget_violations": v,
                        "|<P>|": abs(val),
                    }
                    for b, c, j, v, val in zip(
                        bc_result.bonds, bc_result.chi_used, bc_result.avg_jsd,
                        bc_result.budget_violations, bc_result.values[0],
                    )
                ],
                width="stretch",
            )
            st.caption(
                "diffs (variazione tra bond consecutivi): "
                + ", ".join(f"{d:.2e}" for d in bc_result.diffs[0])
            )


# ── IMPORTA CIRCUITO ─────────────────────────────────────────────────────
elif section == "Importa Circuito":
    st.header("Rumore da hardware reale")
    st.caption(
        "Un importatore \"incolla codice Qiskit/PennyLane\" richiederebbe eseguire quel codice "
        "Python lato server (exec() su input arbitrario) -- una vera falla di code injection, "
        "non un feature. Questa sezione resta limitata a un backend Qiskit FISSO e noto "
        "(nessun codice arbitrario eseguito): estrae i tassi di errore reali calibrati."
    )
    if st.button("Carica calibrazione FakeSherbrooke"):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
            backend = FakeSherbrooke()
            specs = dense_evolution.noise_model_from_qiskit_backend(backend)
            st.dataframe(specs[:50], width="stretch")
            st.caption(f"{len(specs)} coppie (gate, qubit) con tasso di errore calibrato reale.")
        except ImportError as exc:
            st.error(f"Richiede qiskit-ibm-runtime: {exc}")
        except Exception as exc:
            st.error(f"Errore: {exc}")


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
