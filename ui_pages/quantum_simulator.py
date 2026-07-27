"""
Quantum Simulator page — circuit execution, VQE (real JAX-autodiff or
hash-seeded mock), MD telemetry, Hamiltonian library, and the
Overview/Fisica Stato/Mosaico/VQE Results/MD Results/Performance/3D
Helix/Hamiltonian panels (dashboard_core.py, ported from dash.py).

VQE and MD telemetry are routed through the same AI vector-healing shield
used standalone on the Vector Healing page (ui_pages/ai_middleware.py)
before any panel is built from them — this is real infrastructure, not a
cosmetic touch: if a run produces NaN/Inf/spike artifacts, the shield
cleans them up here exactly like it does on the Vector Healing page, and
its telemetry (fallback/radius/reconstruction-error) is shown alongside
the physics metrics it protects.
"""

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import dashboard_core as dc
from ui_pages.ai_middleware import heal_telemetry
from ui_pages.components import render_ai_shield_card, render_metric_grid


def _render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Circuito")
        source_mode = st.radio("Sorgente", ["Libreria Built-in", "Custom QASM Textarea"], key="qs_source")
        if source_mode == "Libreria Built-in":
            circuit_name = st.selectbox("Circuito", options=list(dc.QASM_LIBRARY.keys()), key="qs_circuit")
            qasm_text = ""
        else:
            circuit_name = "Custom Workspace"
            qasm_text = st.text_area(
                "OpenQASM 2.0", height=140,
                value='OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;',
                key="qs_qasm_text",
            )

        st.header("🧮 Motore")
        engine = st.selectbox(
            "Motore di simulazione", options=["dense", "mps"],
            format_func=lambda e: "Denso (DenseSVSimulator/Chunk)" if e == "dense" else "MPS (bond-troncato, ≤24 qubit)",
            key="qs_engine",
            help=(
                "Denso: sempre esatto, adatta memoria automaticamente (Chunk) fino al "
                "limite RAM della macchina. MPS: esatto per circuiti a bassa entanglement "
                "(vedi dense_evolution/mps.py), limitato a 24 qubit in questa dashboard "
                "-- oltre serve il campionamento sequenziale, non ancora collegato ai pannelli."
            ),
        )

        st.header("🌪️ Rumore")
        noise_model = st.selectbox("Modello", options=['ideal', 'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping', 'combined'], key="qs_noise")
        noise_p = st.slider("Probabilità p", 0.0, 0.5, 0.0, 0.01, key="qs_noise_p")
        shots = st.slider("Shots", 50, 5000, 512, 50, key="qs_shots")
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, key="qs_seed")
        double_precision = st.checkbox("Doppia precisione (float64)", value=False, key="qs_precision")

        st.header("🩹 Error Mitigation (ZNE)")
        zne_enabled = st.checkbox(
            "Abilita Zero-Noise Extrapolation", value=False, key="qs_zne_enabled",
            disabled=(noise_model == 'ideal'),
            help="Richiede un modello di rumore attivo -- non applicabile a 'ideal'.",
        )
        if noise_model == 'ideal' and zne_enabled:
            st.caption("⚠️ ZNE disabilitato: nessun rumore da estrapolare con 'ideal'.")
            zne_enabled = False
        if zne_enabled:
            zne_healing = st.checkbox(
                "Healing predittivo (Δpre_emp-adapted)", value=False, key="qs_zne_healing",
                help=(
                    "Corregge i 3 coefficienti di Richardson in base alla deviazione di "
                    "coerenza osservata (σ da shot-noise binomiale alla scala base), invece "
                    "del ZNE standard a pesi fissi."
                ),
            )
            zne_target_sigma = (
                st.number_input("Target σ ideale", value=10.0, key="qs_zne_target_sigma")
                if zne_healing else 10.0
            )
        else:
            zne_healing, zne_target_sigma = False, 10.0

        st.header("🧪 VQE")
        vqe_enabled = st.checkbox("Abilita telemetria VQE", value=True, key="qs_vqe_enabled")
        if vqe_enabled:
            vqe_epochs = st.slider("Epochs", 5, 100, 20, key="qs_vqe_epochs")
            vqe_lr = st.slider("Learning rate", 0.001, 0.5, 0.05, key="qs_vqe_lr")
            vqe_beta1 = st.slider("Adam β1", 0.5, 0.999, 0.9, key="qs_vqe_beta1")
            vqe_beta2 = st.slider("Adam β2", 0.9, 0.9999, 0.999, key="qs_vqe_beta2")
            confirm_heavy_vqe = st.checkbox(
                f"Confermo VQE reale anche su circuiti pesanti (>{dc.QM_MM_HEAVY_QUBIT_THRESHOLD} qubit — costo O(4ⁿ), può essere lento)",
                value=False, key="qs_confirm_heavy",
            )
        else:
            vqe_epochs, vqe_lr, vqe_beta1, vqe_beta2, confirm_heavy_vqe = 20, 0.05, 0.9, 0.999, False

        hamiltonian_values, hamiltonian_name = None, None
        if vqe_enabled:
            st.header("🧬 Hamiltoniana personalizzata")
            hamiltonian_enabled = st.checkbox("Abilita Hamiltoniana personalizzata", value=False, key="qs_ham_enabled")
            if hamiltonian_enabled:
                ham_library = st.session_state.setdefault("hamiltonian_library", dict(dc.LIBRERIA_HAMILTONIANE))
                current_qasm = dc.QASM_LIBRARY[circuit_name] if source_mode == "Libreria Built-in" else qasm_text
                inferred_n_qubits = dc.infer_qubit_count_from_qasm(current_qasm)
                compatible = dc.get_compatible_hamiltonians(inferred_n_qubits, ham_library)

                ham_mode = st.radio("Modalità", ["Libreria Built-in", "Custom JSON Textarea"], key="qs_ham_mode")
                if ham_mode == "Libreria Built-in":
                    if compatible:
                        hamiltonian_name = st.selectbox(
                            f"Hamiltoniana ({inferred_n_qubits} qubit, dim={2**inferred_n_qubits if inferred_n_qubits else '?'})",
                            options=list(compatible.keys()), key="qs_ham_sel",
                        )
                        hamiltonian_values = compatible[hamiltonian_name]
                    else:
                        st.caption(
                            f"Nessuna Hamiltoniana in libreria è compatibile con {inferred_n_qubits or '?'} qubit "
                            f"(serve un array di lunghezza 2^n = {2**inferred_n_qubits if inferred_n_qubits else '?'}). "
                            "Usa la modalità Custom JSON qui sotto."
                        )
                else:
                    ham_json_text = st.text_area(
                        "Array JSON (lunghezza = 2^n_qubits)", value="[-1.13, -0.45, 0.12, 0.64]",
                        height=80, key="qs_ham_json",
                    )
                    ham_save_name = st.text_input("Nome per salvare in libreria (opzionale)", key="qs_ham_save_name")
                    if st.button("💾 Salva in libreria", key="qs_ham_save_btn"):
                        ok, msg = dc.save_custom_hamiltonian(ham_library, ham_save_name, ham_json_text)
                        (st.success if ok else st.error)(msg)
                    try:
                        hamiltonian_values = json.loads(ham_json_text)
                        hamiltonian_name = ham_save_name or "Custom JSON"
                    except json.JSONDecodeError:
                        st.error("JSON non valido — verrà usata un'Hamiltoniana casuale per questa run.")
                        hamiltonian_values = None

        st.header("🌡️ Molecular Dynamics")
        md_enabled = st.checkbox("Abilita telemetria MD", value=True, key="qs_md_enabled")
        if md_enabled:
            md_steps = st.slider("MD steps", 10, 500, 80, key="qs_md_steps")
            md_temp = st.slider("Temperatura (K)", 10, 800, 300, key="qs_md_temp")
        else:
            md_steps, md_temp = 80, 300

        run_clicked = st.button("▶️ Esegui Simulazione", type="primary")

    return dict(
        source_mode=source_mode, circuit_name=circuit_name, qasm_text=qasm_text,
        engine=engine,
        noise_model=noise_model, noise_p=noise_p, shots=shots, seed=seed,
        double_precision=double_precision,
        zne_enabled=zne_enabled, zne_healing=zne_healing, zne_target_sigma=zne_target_sigma,
        vqe_enabled=vqe_enabled, vqe_epochs=vqe_epochs, vqe_lr=vqe_lr,
        vqe_beta1=vqe_beta1, vqe_beta2=vqe_beta2, confirm_heavy_vqe=confirm_heavy_vqe,
        hamiltonian_values=hamiltonian_values, hamiltonian_name=hamiltonian_name,
        md_enabled=md_enabled, md_steps=md_steps, md_temp=md_temp,
        run_clicked=run_clicked,
    )


def _execute_run(p: dict) -> None:
    with st.spinner("Esecuzione circuito..."):
        try:
            res = dc.run_simulation(
                p["source_mode"], p["circuit_name"], p["qasm_text"],
                p["noise_model"], p["noise_p"], p["shots"], int(p["seed"]),
                use_float32=not p["double_precision"], engine=p["engine"],
            )
        except Exception as e:
            st.error(f"Errore durante l'esecuzione del circuito: {e}")
            return

    mitigation_res = None
    if p["zne_enabled"]:
        with st.spinner("Mitigazione ZNE (3 scale di rumore)..."):
            try:
                mitigation_res = dc.run_mitigation_sweep(
                    p["source_mode"], p["circuit_name"], p["qasm_text"], p["noise_model"],
                    p["noise_p"], p["shots"], int(p["seed"]),
                    use_float32=not p["double_precision"], engine=p["engine"],
                    healing_enabled=p["zne_healing"], target_sigma_ideal=p["zne_target_sigma"],
                )
            except Exception as e:
                st.error(f"Errore durante la mitigazione ZNE: {e}")

    df_vqe = pd.DataFrame()
    vqe_ai_meta = {'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0}
    if p["vqe_enabled"]:
        if res['n_qubits'] > dc.QM_MM_HEAVY_QUBIT_THRESHOLD and not p["confirm_heavy_vqe"]:
            st.warning(
                f"Circuito a {res['n_qubits']} qubit: il VQE reale (se il circuito ha gate parametrici) "
                f"userebbe QMMMForceEngine con costo O(4ⁿ) ≈ dim²={2**res['n_qubits']:,}² — telemetria VQE "
                "saltata per questa run. Spunta la conferma nella sidebar per eseguirla comunque."
            )
        else:
            progress_bar = st.progress(0.0, text="VQE: avvio ottimizzazione...")
            log_lines = []
            log_container = st.empty()

            def _on_epoch(epoch, total, row):
                progress_bar.progress((epoch + 1) / total, text=f"VQE epoch {epoch + 1}/{total}")
                log_lines.append(
                    f"[epoch {epoch + 1:03d}/{total}] "
                    f"E={row['VQE_Energy']:.5f}  S={row['Entropy']:.4f}  "
                    f"purity={row['Purity']:.4f}  grad={row['Gradient']:.5f}"
                )
                log_container.code("\n".join(log_lines[-12:]), language="text")

            try:
                df_vqe = dc.run_vqe_telemetry(
                    res['sim'], res['parser'],
                    dc.QASM_LIBRARY[p["circuit_name"]] if p["source_mode"] == "Libreria Built-in" else p["qasm_text"],
                    p["circuit_name"], res['n_qubits'], not p["double_precision"],
                    p["vqe_epochs"], p["vqe_lr"], p["vqe_beta1"], p["vqe_beta2"], int(p["seed"]),
                    hamiltonian_values=p["hamiltonian_values"], on_epoch=_on_epoch,
                )
            except Exception as e:
                st.error(f"Errore durante la telemetria VQE: {e}")
            finally:
                progress_bar.empty()

            if not df_vqe.empty:
                df_vqe, vqe_ai_meta = heal_telemetry(df_vqe)
                if vqe_ai_meta["fallback_triggered"]:
                    log_lines.append(
                        f"⚠️ AI Shield: fallback mediano attivato (raggio adattivo="
                        f"{vqe_ai_meta['adaptive_radius_used']}, errore ricostruzione="
                        f"{vqe_ai_meta['reconstruction_error']:.4f})"
                    )
                log_container.code("\n".join(log_lines[-12:]), language="text")

    df_md, corr_matrix = pd.DataFrame(), pd.DataFrame()
    md_ai_meta = {'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0}
    if p["md_enabled"]:
        with st.spinner("Telemetria MD..."):
            df_md, _raw_corr = dc.run_md_telemetry(p["md_steps"], p["md_temp"])
            df_md, md_ai_meta = heal_telemetry(df_md)
            corr_matrix = df_md.corr(method="pearson")  # recomputed from the healed data, not the raw one

    history = st.session_state.setdefault("qs_run_history", [])
    history.append({
        'nome': res['nome'],
        'n_qubits': res['n_qubits'],
        'tempo': res['tempo'],
        'ram': res['ram'],
        'porte_count': res['porte_count'],
        'entropy': float(res['entropy']),
        'fidelity': res['fidelity'],
        'stato_dominante': res['stato_dominante'],
        'noise_model': p["noise_model"],
        'noise_p': p["noise_p"],
    })

    # Build every panel once, here, right after the data that feeds it
    # changes — not on every sidebar rerun. Figures are closed immediately
    # (plt.close) and the objects kept in session_state; st.pyplot() can
    # render an already-closed Figure object just fine, this only stops
    # them accumulating in pyplot's registry.
    with st.spinner("Rendering pannelli..."):
        metrics = dc.compute_overview_metrics(res, p["noise_model"], p["noise_p"])
        overview_fig = dc.build_panel_overview(res, df_vqe, corr_matrix, p["noise_model"], p["noise_p"])
        fisica_fig = dc.build_panel_fisica(res, seed=int(p["seed"]))
        mosaico_fig = dc.build_panel_mosaico(res)
        vqe_fig = dc.build_panel_vqe_results(df_vqe)
        md_fig = dc.build_panel_md_results(df_md, corr_matrix)
        perf_fig = dc.build_panel_performance(res, history)
        helix_fig = dc.build_3d_helix_patch(res['n_qubits'], res['prob'])
        ham_fig = dc.build_panel_hamiltonian(p["hamiltonian_values"], p["hamiltonian_name"] or 'nessuna')
        mitigation_fig = dc.build_panel_mitigation(mitigation_res, res) if mitigation_res else None

        overview_png = io.BytesIO()
        overview_fig.savefig(overview_png, format="png", dpi=300, facecolor='#010409', bbox_inches='tight')

        figs_to_close = [overview_fig, fisica_fig, mosaico_fig, vqe_fig, md_fig, perf_fig, ham_fig]
        if mitigation_fig is not None:
            figs_to_close.append(mitigation_fig)
        for fig in figs_to_close:
            plt.close(fig)

    st.session_state["qs_res"] = res
    st.session_state["qs_panels"] = {
        "metrics": metrics,
        "vqe_ai_meta": vqe_ai_meta,
        "md_ai_meta": md_ai_meta,
        "overview": overview_fig,
        "overview_png": overview_png.getvalue(),
        "fisica": fisica_fig,
        "mosaico": mosaico_fig,
        "vqe": vqe_fig,
        "md": md_fig,
        "hamiltonian": ham_fig,
        "performance": perf_fig,
        "helix": helix_fig,
        "mitigation": mitigation_fig,
    }


def _render_tabs() -> None:
    res = st.session_state.get("qs_res")
    panels = st.session_state.get("qs_panels")

    if res is None or panels is None:
        st.info("Configura il circuito nella sidebar e premi **Esegui Simulazione** per iniziare.")
        return

    run_history = st.session_state.get("qs_run_history", [])

    tabs = st.tabs(["Overview", "Fisica Stato", "Mosaico", "VQE Results", "MD Results", "Performance", "3D Helix", "Hamiltonian", "Mitigation (ZNE)"])

    with tabs[0]:
        render_ai_shield_card("AI Vector-Healing Shield — Telemetria VQE", panels["vqe_ai_meta"])
        with st.container(border=True):
            st.subheader("📊 Simulation Metrics")
            render_metric_grid(panels["metrics"])
        st.pyplot(panels["overview"])
        st.download_button(
            "🖼️ Esporta Overview come PNG (300 DPI)",
            data=panels["overview_png"],
            file_name="quantum_dashboard_overview.png",
            mime="image/png",
        )

    with tabs[1]:
        st.pyplot(panels["fisica"])

    with tabs[2]:
        st.pyplot(panels["mosaico"])

    with tabs[3]:
        st.pyplot(panels["vqe"])

    with tabs[4]:
        render_ai_shield_card("AI Vector-Healing Shield — Telemetria MD", panels["md_ai_meta"])
        st.pyplot(panels["md"])

    with tabs[5]:
        st.pyplot(panels["performance"])

        if run_history:
            st.dataframe(pd.DataFrame(run_history), use_container_width=True)
            st.download_button(
                "💾 Esporta provenance JSON",
                data=dc.build_provenance_json(run_history),
                file_name="quantum_provenance_archive.json",
                mime="application/json",
            )

        st.divider()
        st.caption(
            "Benchmark scan: alloca simulatori da 2 a 14 qubit e misura tempo/RAM. "
            "Genuinamente lento (allocazione densa fino a 2¹⁴) — esegui solo se necessario."
        )
        if st.button("🧪 Esegui Benchmark Scan (2→14 qubit)"):
            with st.spinner("Benchmark in corso..."):
                bench_df = dc.run_benchmark_scan()
            st.dataframe(bench_df, use_container_width=True)

    with tabs[6]:
        st.plotly_chart(panels["helix"], use_container_width=True)

    with tabs[7]:
        st.pyplot(panels["hamiltonian"])
        st.caption(
            "Se nessuna Hamiltoniana personalizzata è abilitata nella sidebar, il VQE usa uno spettro "
            "casuale ordinato (comportamento di default, invariato)."
        )

    with tabs[8]:
        if panels.get("mitigation") is None:
            st.info("Abilita ZNE nella sidebar e riesegui per vedere questo pannello.")
        else:
            st.pyplot(panels["mitigation"])


def render():
    st.markdown(
        """
        <div style="padding: 1.25rem 1.5rem; border-radius: 0.75rem;
                    background: linear-gradient(90deg, #0a0014, #1a0226);
                    border: 1px solid #a78bfa44; margin-bottom: 1rem;">
            <h1 style="margin: 0; color: #a78bfa;">Dense Evolution v8.1.7 — Quantum Simulator Dashboard</h1>
            <p style="margin: 0.5rem 0 0 0; color: #cccccc;">
                Esegui un circuito OpenQASM su <code>DenseSVSimulator</code>, opzionalmente con rumore,
                telemetria VQE (JAX autodiff + forze QM/MM Hellmann-Feynman) e dinamica molecolare —
                entrambe protette dallo stesso scudo AI Vector-Healing della pagina Vector Healing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    params = _render_sidebar()
    if params["run_clicked"]:
        _execute_run(params)
    _render_tabs()
