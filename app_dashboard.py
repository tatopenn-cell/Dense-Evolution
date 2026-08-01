"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Rebuilt from scratch on the structure of IBM Quantum Composer -- the
standard layout every real quantum-computing dashboard (Qiskit/PennyLane/
qsim included) converges on: a circuit editor plus Statevector /
Probabilities / Q-sphere views, kept in sync with one real execution.

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
    preset_name = st.selectbox("Preset", ["Custom"] + list(dc.QASM_LIBRARY.keys()))
    # Each preset (and Custom) gets its own widget key, so switching the
    # dropdown actually swaps the visible text instead of Streamlit
    # keeping whatever was last typed under a single shared key.
    default_qasm = dc.QASM_LIBRARY.get(preset_name, dc.QASM_LIBRARY["Bell state (2 qubit)"])
    qasm_text = st.text_area(
        "OpenQASM 2.0", value=default_qasm, height=220, key=f"qasm_text__{preset_name}",
    )

    n_shots = st.number_input("Shots", min_value=1, max_value=100_000, value=1000, step=100)
    seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1)

    run_clicked = st.button("▶ Esegui", type="primary", use_container_width=True)

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

if error:
    st.error(f"Errore nell'esecuzione del circuito: {error}")
elif result is None:
    st.info("Premi ▶ Esegui nella sidebar per lanciare il circuito sul motore reale.")
else:
    tab_circuit, tab_statevector, tab_probabilities, tab_qsphere = st.tabs(
        ["Circuit", "Statevector", "Probabilities", "Q-sphere"]
    )

    with tab_circuit:
        st.pyplot(dc.draw_circuit_figure(result.qiskit_circuit))

    with tab_statevector:
        st.caption(f"{result.n_qubits} qubit — {len(result.statevector)} ampiezze (convenzione Qiskit)")
        rows = [
            {
                "state": format(i, f"0{result.n_qubits}b"),
                "amplitude": complex(amp),
                "|amplitude|": abs(amp),
                "phase (rad)": float(np.angle(amp)),
            }
            for i, amp in enumerate(result.statevector)
            if abs(amp) > 1e-10
        ]
        st.dataframe(rows, use_container_width=True)

    with tab_probabilities:
        st.caption(f"{int(n_shots)} shot reali campionati dallo statevector calcolato")
        st.pyplot(dc.histogram_figure(result.counts))

    with tab_qsphere:
        st.pyplot(dc.qsphere_figure(result.statevector))
