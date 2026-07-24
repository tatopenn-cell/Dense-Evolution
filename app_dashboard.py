"""
Dense Evolution v8.1.7 - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Two pages sharing one AI vector-healing engine:
  - Quantum Simulator (ui_pages/quantum_simulator.py): circuit execution,
    VQE/MD telemetry — both routed through the AI shield before rendering.
  - Vector Healing (ui_pages/vector_healing.py): interactive sandbox for
    the same shield (ia_utils.vector_healing.enhanced_dense_healing_hybrid).

Run with:
    pip install streamlit
    streamlit run app_dashboard.py
"""

import streamlit as st

from ui_pages import quantum_simulator, vector_healing, quantum_scars

st.set_page_config(
    page_title="Dense Evolution v8.1.7 - Dashboard",
    page_icon="🧬",
    layout="wide",
)

pg = st.navigation([
    st.Page(quantum_simulator.render, title="Quantum Simulator", icon="⚛️", url_path="quantum-simulator", default=True),
    st.Page(vector_healing.render, title="Vector Healing", icon="🧬", url_path="vector-healing"),
    st.Page(quantum_scars.render, title="Quantum Scars", icon="🌀", url_path="quantum-scars"),
])
pg.run()
