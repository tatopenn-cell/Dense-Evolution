"""
Shared, reusable rendering pieces used by more than one page — keeps the
same visual language (bordered "card" grids, gradient banners, the
run-then-render guard) consistent across all dashboard pages instead of
each one hand-rolling its own copy.
"""

import matplotlib.pyplot as plt
import streamlit as st

import dense_evolution


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


# Neutral AI-shield metadata for the "no healing ran" case (empty/None
# input to ui_pages.ai_middleware.heal_telemetry, or a page that hasn't
# run yet) -- was hand-duplicated as an identical dict literal in
# ai_middleware.py and twice in quantum_simulator.py.
AI_SHIELD_NEUTRAL_META = {
    'fallback_triggered': False,
    'adaptive_radius_used': 0,
    'reconstruction_error': 0.0,
}


def render_page_banner(page_title: str, subtitle_html: str, accent: str = "#a78bfa",
                        bg_from: str = "#0a0014", bg_to: str = "#1a0226") -> None:
    """Gradient header banner at the top of every dashboard page.

    `page_title` is appended after "Dense Evolution v{dense_evolution.__version__} — "
    so the version shown always reflects what's actually installed instead
    of a hand-typed number that silently drifts out of sync per page
    (found: v8.1.9 / v8.1.7 / v8.1.22, three different stale values
    hardcoded across three pages before this).

    `subtitle_html` is rendered as-is inside a <p> — callers already build
    their own <code>/<a> markup for the subtitle text, this doesn't
    re-escape it."""
    st.markdown(
        f"""
        <div style="padding: 1.25rem 1.5rem; border-radius: 0.75rem;
                    background: linear-gradient(90deg, {bg_from}, {bg_to});
                    border: 1px solid {accent}44; margin-bottom: 1rem;">
            <h1 style="margin: 0; color: {accent};">Dense Evolution v{dense_evolution.__version__} — {page_title}</h1>
            <p style="margin: 0.5rem 0 0 0; color: #cccccc;">
                {subtitle_html}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_guard(*keys: str, message: str):
    """Retrieves each of `keys` from st.session_state. If any is missing
    (None), shows `message` via st.info and returns None — the caller
    should `return` immediately in that case, nothing left to render yet.
    Otherwise returns the retrieved values: a single value when one key
    was given, a tuple when several were.

    Encapsulates the "run → session_state → guard → render" flow repeated
    across quantum_simulator.py (2 keys: result + panels),
    vector_healing.py and quantum_scars.py (1 key each)."""
    values = [st.session_state.get(k) for k in keys]
    if any(v is None for v in values):
        st.info(message)
        return None
    return values[0] if len(values) == 1 else tuple(values)


def render_matplotlib_figure(fig, *, close: bool = True) -> None:
    """st.pyplot(fig), then plt.close(fig) unless close=False.

    For the "build a figure, show it immediately" call sites
    (vector_healing.py, quantum_scars.py) — NOT quantum_simulator.py,
    which builds every panel once in _execute_run and renders
    already-closed figures later in _render_tabs, a different and
    intentional pattern left as-is."""
    st.pyplot(fig)
    if close:
        plt.close(fig)
