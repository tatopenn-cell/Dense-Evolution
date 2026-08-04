"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Rebuilt from scratch on the structure of IBM Quantum Composer -- the
standard layout every real quantum-computing dashboard (Qiskit/PennyLane/
qsim included) converges on: a graphical (drag-and-drop) circuit editor
synced with a code (OpenQASM) editor, plus Statevector / Probabilities /
Q-sphere views, all reflecting one real execution.

Every number shown here comes from a real run of dense_evolution's own
DenseSVSimulator (dashboard_core.engine), rendered with Qiskit's own real
visualization functions (dashboard_core.visuals) -- no placeholder data,
no hand-rolled plotting standing in for the real thing.

Advanced features built previously (VQE, real molecular Hamiltonians,
mitigation, AI vector-healing) are intentionally not here yet -- they
live on the feature/streamlit-dashboard branch and get reintegrated once
this base is solid.

Run with:
    pip install streamlit
    streamlit run app_dashboard.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import dense_evolution
import streamlit as st

import dashboard_core as dc

st.set_page_config(
    page_title=f"Dense Evolution v{dense_evolution.__version__} - Quantum Composer",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Dense Evolution — Quantum Composer")

with st.sidebar:
    st.header("Circuit")
    preset_name = st.selectbox(
        "Preset", ["Custom"] + list(dc.QASM_LIBRARY.keys()), key="preset_select",
    )
    # Each preset (and Custom) gets its own widget key, so switching the
    # dropdown actually swaps the visible text instead of Streamlit
    # keeping whatever was last typed under a single shared key.
    default_qasm = dc.QASM_LIBRARY.get(preset_name, dc.QASM_LIBRARY["Bell state (2 qubit)"])
    qasm_text = st.text_area(
        "OpenQASM 2.0", value=default_qasm, height=220, key=f"qasm_text__{preset_name}",
    )

    n_shots = st.number_input("Shots", min_value=1, max_value=100_000, value=1000, step=100)
    seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1)

    run_clicked = st.button("▶ Esegui", type="primary", width="stretch")

if run_clicked:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            qasm_text, n_shots=int(n_shots), seed=int(seed),
        )
        st.session_state["error"] = None
    except Exception as exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(exc)

error = st.session_state.get("error")
result = st.session_state.get("result")

tab_builder, tab_circuit, tab_statevector, tab_probabilities, tab_qsphere = st.tabs(
    ["Graphical Builder", "Circuit", "Statevector", "Probabilities", "Q-sphere"]
)

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

with tab_statevector:
    if result is None:
        st.info("Premi ▶ Esegui nella sidebar per lanciare il circuito sul motore reale.")
    elif not error:
        st.caption(f"{result.n_qubits} qubit — {len(result.statevector)} ampiezze (convenzione Qiskit)")
        # pyarrow (which st.dataframe serializes through) has no complex128
        # support -- split into real/imaginary float columns instead of
        # relying on Streamlit's internal best-effort fallback for it.
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

with tab_probabilities:
    if result is None:
        st.info("Premi ▶ Esegui nella sidebar per lanciare il circuito sul motore reale.")
    elif not error:
        st.caption(f"{int(n_shots)} shot reali campionati dallo statevector calcolato")
        st.pyplot(dc.histogram_figure(result.counts))

with tab_qsphere:
    if result is None:
        st.info("Premi ▶ Esegui nella sidebar per lanciare il circuito sul motore reale.")
    elif not error:
        st.pyplot(dc.qsphere_figure(result.statevector))

if error:
    st.sidebar.error(f"Errore nell'esecuzione del circuito: {error}")
