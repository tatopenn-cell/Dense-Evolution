"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Pages sharing one AI vector-healing engine:
  - Quantum Simulator (ui_pages/quantum_simulator.py): circuit execution,
    VQE/MD telemetry — both routed through the AI shield before rendering.
  - Vector Healing (ui_pages/vector_healing.py): interactive sandbox for
    the same shield (ia_utils.vector_healing.enhanced_dense_healing_hybrid).
  - Research Bridge (ui_pages/research_bridge.py): package a hypothesis +
    real Dense-Evolution numbers into a paste-ready block, cross-check it
    against Google / a personal Colab / any API-keyed model, log the
    results. First (small, one-layer) piece of a planned multi-layer
    bridge -- more tool layers (e.g. PyMOL) added incrementally later.

Run with:
    pip install streamlit
    streamlit run app_dashboard.py
"""

import dense_evolution
import streamlit as st

from ui_pages import quantum_simulator, vector_healing, quantum_scars, research_bridge

st.set_page_config(
    page_title=f"Dense Evolution v{dense_evolution.__version__} - Dashboard",
    page_icon="🧬",
    layout="wide",
)

pg = st.navigation([
    st.Page(quantum_simulator.render, title="Quantum Simulator", icon="⚛️", url_path="quantum-simulator", default=True),
    st.Page(vector_healing.render, title="Vector Healing", icon="🧬", url_path="vector-healing"),
    st.Page(quantum_scars.render, title="Quantum Scars", icon="🌀", url_path="quantum-scars"),
    st.Page(research_bridge.render, title="Research Bridge", icon="🌉", url_path="research-bridge"),
])
pg.run()
