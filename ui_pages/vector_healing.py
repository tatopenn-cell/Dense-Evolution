"""
Vector Healing page — interactive demo for
ia_utils.vector_healing.enhanced_dense_healing_hybrid. The same engine also
runs, for real, inside the Quantum Simulator page's VQE/MD telemetry
pipeline (see ui_pages/ai_middleware.py) — this page is the sandbox where
you can dial the corruption knobs yourself and watch it work.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from ia_utils.vector_healing import enhanced_dense_healing_hybrid
from ui_pages.components import render_ai_shield_card, render_page_banner, render_run_guard, render_matplotlib_figure

HIDDEN_DIM = 6


def _generate_corrupted_sequence(n_steps, hidden_dim, anomaly_pct, rng):
    t = np.linspace(0, 4 * np.pi, n_steps)
    freqs = rng.uniform(0.8, 1.2, size=hidden_dim)
    phases = rng.uniform(0, 2 * np.pi, size=hidden_dim)
    ideal = np.stack([np.sin(freqs[d] * t + phases[d]) for d in range(hidden_dim)], axis=1)
    corrupted = ideal + rng.normal(0, 0.05, size=ideal.shape)

    n_anomalies = max(1, int(round(n_steps * anomaly_pct / 100.0)))
    anomaly_idx = rng.choice(n_steps, size=min(n_anomalies, n_steps), replace=False)
    kinds = rng.choice(["nan", "inf", "spike"], size=len(anomaly_idx))

    for idx, kind in zip(anomaly_idx, kinds):
        dim = int(rng.integers(0, hidden_dim))
        if kind == "nan":
            corrupted[idx, dim] = np.nan
        elif kind == "inf":
            corrupted[idx, dim] = np.inf if rng.random() < 0.5 else -np.inf
        else:
            corrupted[idx, dim] = ideal[idx, dim] + rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)

    return ideal, corrupted


def render():
    render_page_banner(
        "AI Vector Healing Dashboard",
        """<code>ia_utils.vector_healing.enhanced_dense_healing_hybrid</code> è uno scudo anti-crash
        per sequenze di vettori (es. hidden states): ripulisce <code>NaN</code>/<code>Inf</code>,
        poi decide passo per passo — tramite la logica Φ-trigger di
        <code>dense_evolution.healing</code> — se il valore fa parte di un cambio di tendenza
        genuino (lasciato passare) o di uno spike isolato/rumore (sostituito con la mediana
        locale). Lo stesso scudo protegge in produzione anche la telemetria VQE/MD del Quantum
        Simulator.""",
        accent="#00e5ff", bg_from="#001014", bg_to="#012026",
    )

    with st.sidebar:
        st.header("⚙️ Configurazione")
        n_steps = st.slider("Numero di step / token", min_value=10, max_value=150, value=80)
        anomaly_pct = st.slider(
            "Percentuale anomalie (NaN / Inf / Spike)", min_value=5, max_value=30, value=10
        )
        run_clicked = st.button("🚀 Genera ed Esegui Healing", type="primary")

    if run_clicked:
        rng = np.random.default_rng()
        ideal, corrupted = _generate_corrupted_sequence(n_steps, HIDDEN_DIM, anomaly_pct, rng)
        healed, metadata = enhanced_dense_healing_hybrid(corrupted)
        st.session_state["healing_result"] = {
            "ideal": ideal,
            "corrupted": corrupted,
            "healed": healed,
            "metadata": metadata,
            "n_steps": n_steps,
        }

    result = render_run_guard(
        "healing_result",
        message="Configura i parametri nella sidebar e premi **Genera ed Esegui Healing** per iniziare.",
    )
    if result is None:
        return

    ideal, corrupted, healed, metadata = (
        result["ideal"], result["corrupted"], result["healed"], result["metadata"]
    )

    render_ai_shield_card("AI Vector-Healing Shield", metadata)

    with st.container(border=True):
        st.subheader("📈 Confronto vettoriale")
        channel = st.selectbox(
            "Dimensione da visualizzare",
            options=list(range(HIDDEN_DIM)),
            format_func=lambda d: f"Canale {d}",
        )

        x = np.arange(result["n_steps"])
        ideal_channel = ideal[:, channel]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, ideal_channel, label="Ideale", color="#888888", linewidth=1.5, linestyle="--")
        ax.plot(x, corrupted[:, channel], label="Corrotto", color="#ff4b4b", linewidth=1.2, alpha=0.75)
        ax.plot(x, healed[:, channel], label="Curato", color="#00c853", linewidth=2.2)

        margin = 1.5
        ax.set_ylim(ideal_channel.min() - margin, ideal_channel.max() + margin)
        ax.set_xlabel("Step / Token")
        ax.set_ylabel(f"Valore (canale {channel})")
        ax.set_title("Vector Healing — Ideale vs Corrotto vs Curato")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

        render_matplotlib_figure(fig)
