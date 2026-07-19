"""
Shared, reusable rendering pieces used by more than one page — keeps the
same visual language (bordered "card" grids) consistent across Vector
Healing and Quantum Simulator instead of each page hand-rolling its own.
"""

import streamlit as st


def render_metric_grid(metrics: list, columns: int = 4):
    """Renders a list of {'label', 'value', 'help'} dicts as a grid of
    st.metric tiles. `value` should already be pre-shortened by the caller
    (st.metric tiles don't wrap — long values overflow instead of fitting);
    the full-precision original belongs in `help` as a hover tooltip."""
    cols = st.columns(columns)
    for i, m in enumerate(metrics):
        cols[i % columns].metric(m["label"], m["value"], help=m.get("help"))


def render_ai_shield_card(title: str, metadata: dict):
    """A bordered card showing the 3 AI vector-healing telemetry values
    (fallback triggered / adaptive radius / reconstruction error) for a
    telemetry DataFrame that was just passed through
    ui_pages.ai_middleware.heal_telemetry()."""
    with st.container(border=True):
        st.markdown(f"**🛡️ {title}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fallback scattato", "Sì" if metadata["fallback_triggered"] else "No")
        c2.metric("Raggio adattivo", metadata["adaptive_radius_used"])
        c3.metric("Errore di ricostruzione", f"{metadata['reconstruction_error']:.4f}")
