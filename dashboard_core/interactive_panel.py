"""
Interactive ipywidgets panel — Colab/Jupyter-native, no Streamlit, no
external tunnel/link. Lives entirely in the calling cell's output.

Full feature parity with app_dashboard.py's Streamlit dashboard -- all four
pages, switchable via a mode selector at the top instead of Streamlit's
st.navigation sidebar:
  - Quantum Simulator (ui_pages/quantum_simulator.py): same 9 output
    panels, same sidebar controls (circuit source, engine, noise,
    ZNE/predictive healing, VQE, custom Hamiltonian, MD) -- built on the
    same dashboard_core functions that page calls.
  - Vector Healing (ui_pages/vector_healing.py): the same
    enhanced_dense_healing_hybrid sandbox, driven by
    dashboard_core.vector_healing_demo instead of that page's private
    corrupted-sequence generator (ui_pages is repo-only, not importable
    from the installable package -- see that module's own docstring).
  - Quantum Scars (ui_pages/quantum_scars.py): the same PXP live demo,
    driven by dashboard_core.scars_engine, same reasoning.
  - Research Bridge (ui_pages/research_bridge.py): the same four bridges
    (Google chain, Colab, custom API, local CLI), driven directly by
    dashboard_core.research_bridge, which was already Streamlit-free.

All four are driven by ipywidgets instead of Streamlit's rerun-the-whole-
script model -- no st.session_state, no st.rerun(), plain closures and
direct widget-callback wiring instead.

Requires the `ipywidgets` extra (`pip install dense-evolution[dashboard]`).

Not a port of the old legacy/dash.py ipywidgets panel, which predates
this package's refactor, the ZNE feature, and several bug fixes found
while building it (see README changelog v8.1.27-8.1.32).
"""

import pandas as pd

from .qasm_library import QASM_LIBRARY, infer_qubit_count_from_qasm
from .hamiltonians import LIBRERIA_HAMILTONIANE, get_compatible_hamiltonians, save_custom_hamiltonian
from .simulation_runner import run_simulation
from .vqe_engine import QM_MM_HEAVY_QUBIT_THRESHOLD, run_vqe_telemetry
from .md_telemetry import run_md_telemetry
from .metrics import compute_overview_metrics
from .panels import (
    build_panel_overview, build_panel_fisica, build_panel_mosaico,
    build_panel_vqe_results, build_panel_md_results, build_panel_performance,
    build_panel_hamiltonian,
)
from .helix_3d import build_3d_helix_patch
from .mitigation_runner import run_mitigation_sweep
from .mitigation_panel import build_panel_mitigation

_NEUTRAL_AI_META = {'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0}


def _heal_telemetry(df):
    """Same logic as ui_pages/ai_middleware.py::heal_telemetry, duplicated
    (not imported) deliberately: ui_pages is repo-only, not part of the
    installable package `dashboard_core` ships in, so this module can't
    depend on it without breaking a bare `pip install dense-evolution`."""
    from ia_utils.vector_healing import enhanced_dense_healing_hybrid
    if df is None or df.empty:
        return (df if df is not None else pd.DataFrame()), dict(_NEUTRAL_AI_META)
    healed_values, metadata = enhanced_dense_healing_hybrid(df.to_numpy(dtype=float))
    healed_df = pd.DataFrame(healed_values, columns=df.columns, index=df.index)
    return healed_df, metadata


def launch_interactive_panel():
    """Builds and displays the full interactive panel (sidebar-equivalent
    controls + a 9-tab output area, same panels as the Streamlit
    dashboard) in the current Jupyter/Colab cell output. Call directly
    after import:

        import dashboard_core as dc
        dc.launch_interactive_panel()

    Returns the outer ipywidgets.VBox (already displayed)."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError as e:
        raise ImportError(
            "launch_interactive_panel() requires ipywidgets and IPython "
            "(Jupyter/Colab environment). Install with: "
            "pip install dense-evolution[dashboard]"
        ) from e
    import matplotlib.pyplot as plt
    import numpy as np

    run_history = []
    ham_library = dict(LIBRERIA_HAMILTONIANE)

    # ── Circuit source ───────────────────────────────────────────────
    w_source_mode = widgets.RadioButtons(
        options=['Libreria Built-in', 'Custom QASM Textarea'], value='Libreria Built-in',
        description='Sorgente:', style={'description_width': 'initial'},
    )
    w_circuit = widgets.Dropdown(
        options=list(QASM_LIBRARY.keys()), value='Bell |Φ+⟩', description='Circuito:',
        style={'description_width': 'initial'}, layout=widgets.Layout(width='420px'),
    )
    w_qasm_text = widgets.Textarea(
        value='OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;',
        description='OpenQASM 2.0:', style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px', height='90px'),
    )
    w_qasm_text.layout.display = 'none'

    def _on_source_mode_change(change):
        libreria = change['new'] == 'Libreria Built-in'
        w_circuit.layout.display = None if libreria else 'none'
        w_qasm_text.layout.display = 'none' if libreria else None

    w_source_mode.observe(_on_source_mode_change, names='value')

    # ── Engine / noise ───────────────────────────────────────────────
    w_engine = widgets.Dropdown(options=['dense', 'mps'], value='dense', description='Motore:')
    w_noise_model = widgets.Dropdown(
        options=['ideal', 'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping', 'combined'],
        value='ideal', description='Rumore:',
    )
    w_noise_p = widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description='p:')
    w_shots = widgets.IntSlider(value=512, min=50, max=5000, step=50, description='Shots:')
    w_seed = widgets.IntText(value=42, description='Seed:')
    w_double_precision = widgets.Checkbox(value=False, description='Doppia precisione (float64)')

    # ── ZNE ──────────────────────────────────────────────────────────
    w_zne_enabled = widgets.Checkbox(value=False, description='Abilita Zero-Noise Extrapolation')
    w_zne_healing = widgets.Checkbox(value=False, description='Healing predittivo (Δpre_emp-adapted)')
    w_zne_target_sigma = widgets.FloatText(value=10.0, description='Target σ ideale:',
                                            style={'description_width': 'initial'})

    # ── VQE ──────────────────────────────────────────────────────────
    w_vqe_enabled = widgets.Checkbox(value=True, description='Abilita telemetria VQE')
    w_vqe_epochs = widgets.IntSlider(value=20, min=5, max=100, description='Epochs:')
    w_vqe_lr = widgets.FloatLogSlider(value=0.05, min=-3, max=-0.3, description='Learning rate:',
                                       style={'description_width': 'initial'})
    w_vqe_beta1 = widgets.FloatSlider(value=0.9, min=0.5, max=0.999, step=0.001, description='Adam β1:')
    w_vqe_beta2 = widgets.FloatSlider(value=0.999, min=0.9, max=0.9999, step=0.0001, description='Adam β2:')
    w_confirm_heavy_vqe = widgets.Checkbox(
        value=False,
        description=f'Confermo VQE reale anche su circuiti pesanti (>{QM_MM_HEAVY_QUBIT_THRESHOLD} qubit)',
        style={'description_width': 'initial'},
    )

    # ── Custom Hamiltonian ───────────────────────────────────────────
    w_ham_enabled = widgets.Checkbox(value=False, description='Abilita Hamiltoniana personalizzata')
    w_ham_mode = widgets.RadioButtons(options=['Libreria Built-in', 'Custom JSON Textarea'],
                                       value='Libreria Built-in', description='Modalità:')
    w_ham_select = widgets.Dropdown(options=[], description='Hamiltoniana:',
                                     style={'description_width': 'initial'}, layout=widgets.Layout(width='500px'))
    w_ham_json = widgets.Textarea(value='[-1.13, -0.45, 0.12, 0.64]', description='Array JSON:',
                                   style={'description_width': 'initial'})
    w_ham_save_name = widgets.Text(value='', description='Nome (salva):',
                                    style={'description_width': 'initial'})
    w_ham_save_btn = widgets.Button(description='💾 Salva in libreria')
    w_ham_status = widgets.HTML(value='')
    w_ham_box = widgets.VBox([])  # populated/hidden based on w_ham_enabled

    def _current_qasm():
        return QASM_LIBRARY[w_circuit.value] if w_source_mode.value == 'Libreria Built-in' else w_qasm_text.value

    def _refresh_ham_options(*_):
        n_qubits = infer_qubit_count_from_qasm(_current_qasm())
        compatible = get_compatible_hamiltonians(n_qubits, ham_library)
        w_ham_select.options = list(compatible.keys())
        w_ham_status.value = (
            '' if compatible else
            f'<span style="color:#ff6b35">Nessuna Hamiltoniana compatibile con {n_qubits or "?"} qubit — usa Custom JSON.</span>'
        )

    def _on_ham_save_clicked(_btn):
        ok, msg = save_custom_hamiltonian(ham_library, w_ham_save_name.value, w_ham_json.value)
        w_ham_status.value = f'<span style="color:{"#00ff9d" if ok else "#ff6b35"}">{msg}</span>'
        if ok:
            _refresh_ham_options()

    w_ham_save_btn.on_click(_on_ham_save_clicked)
    w_circuit.observe(_refresh_ham_options, names='value')
    w_qasm_text.observe(_refresh_ham_options, names='value')
    w_source_mode.observe(_refresh_ham_options, names='value')

    def _on_ham_enabled_change(change):
        if not change['new']:
            w_ham_box.children = []
            return
        _refresh_ham_options()
        w_ham_box.children = [w_ham_mode, w_ham_select, w_ham_json, w_ham_save_name, w_ham_save_btn, w_ham_status]

    def _on_ham_mode_change(change):
        libreria = change['new'] == 'Libreria Built-in'
        w_ham_select.layout.display = None if libreria else 'none'
        for w in (w_ham_json, w_ham_save_name, w_ham_save_btn):
            w.layout.display = 'none' if libreria else None

    w_ham_enabled.observe(_on_ham_enabled_change, names='value')
    w_ham_mode.observe(_on_ham_mode_change, names='value')

    # ── MD ───────────────────────────────────────────────────────────
    w_md_enabled = widgets.Checkbox(value=True, description='Abilita telemetria MD')
    w_md_steps = widgets.IntSlider(value=80, min=10, max=500, description='MD steps:')
    w_md_temp = widgets.IntSlider(value=300, min=10, max=800, description='Temperatura (K):')

    # ── Run control ──────────────────────────────────────────────────
    w_run = widgets.Button(description='▶ Esegui Simulazione', button_style='primary')
    w_status = widgets.HTML(value='')

    # ── Output: 9 tabs, matching the Streamlit page exactly ──────────
    tab_names = ['Overview', 'Fisica Stato', 'Mosaico', 'VQE Results', 'MD Results',
                 'Performance', '3D Helix', 'Hamiltonian', 'Mitigation (ZNE)']
    tab_outputs = [widgets.Output() for _ in tab_names]
    tabs = widgets.Tab(children=tab_outputs)
    for i, name in enumerate(tab_names):
        tabs.set_title(i, name)

    def _show(idx, *figs):
        with tab_outputs[idx]:
            clear_output(wait=True)
            for fig in figs:
                if fig is None:
                    continue
                display(fig)
                if hasattr(fig, 'savefig'):  # matplotlib Figure, not plotly
                    plt.close(fig)

    def _on_run_clicked(_btn):
        w_run.disabled = True
        w_status.value = '<span style="color:#00c8ff">⏳ Esecuzione circuito...</span>'
        try:
            res = run_simulation(
                w_source_mode.value, w_circuit.value, w_qasm_text.value,
                w_noise_model.value, w_noise_p.value, w_shots.value, int(w_seed.value),
                use_float32=not w_double_precision.value, engine=w_engine.value,
            )
        except Exception as e:
            w_status.value = f'<span style="color:#ff6b35">Errore durante l\'esecuzione del circuito: {e}</span>'
            w_run.disabled = False
            return

        mitigation_res = None
        if w_zne_enabled.value:
            w_status.value = '<span style="color:#00c8ff">⏳ Mitigazione ZNE...</span>'
            if w_noise_model.value == 'ideal':
                w_status.value = '<span style="color:#ff6b35">ZNE richiede un modello di rumore attivo, non \'ideal\' — saltato.</span>'
            else:
                try:
                    mitigation_res = run_mitigation_sweep(
                        w_source_mode.value, w_circuit.value, w_qasm_text.value, w_noise_model.value,
                        w_noise_p.value, w_shots.value, int(w_seed.value),
                        use_float32=not w_double_precision.value, engine=w_engine.value,
                        healing_enabled=w_zne_healing.value, target_sigma_ideal=w_zne_target_sigma.value,
                    )
                except Exception as e:
                    w_status.value = f'<span style="color:#ff6b35">Errore durante la mitigazione ZNE: {e}</span>'

        # Computed once, reused by VQE, MD and the Hamiltonian panel below --
        # was three separate copies of the same extraction before.
        hamiltonian_values = None
        if w_ham_enabled.value:
            if w_ham_mode.value == 'Libreria Built-in' and w_ham_select.value:
                hamiltonian_values = ham_library.get(w_ham_select.value)
            elif w_ham_mode.value == 'Custom JSON Textarea':
                import json
                try:
                    hamiltonian_values = json.loads(w_ham_json.value)
                except json.JSONDecodeError:
                    hamiltonian_values = None

        df_vqe = pd.DataFrame()
        vqe_ai_meta = dict(_NEUTRAL_AI_META)
        if w_vqe_enabled.value:
            if res['n_qubits'] > QM_MM_HEAVY_QUBIT_THRESHOLD and not w_confirm_heavy_vqe.value:
                w_status.value = (
                    f'<span style="color:#ff6b35">Circuito a {res["n_qubits"]} qubit: telemetria VQE saltata '
                    f'(spunta "Confermo VQE reale..." per eseguirla comunque).</span>'
                )
            else:
                w_status.value = '<span style="color:#00c8ff">⏳ VQE...</span>'
                try:
                    df_vqe = run_vqe_telemetry(
                        res['sim'], res['parser'], _current_qasm(), w_circuit.value, res['n_qubits'],
                        not w_double_precision.value, w_vqe_epochs.value, w_vqe_lr.value,
                        w_vqe_beta1.value, w_vqe_beta2.value, int(w_seed.value),
                        hamiltonian_values=hamiltonian_values,
                    )
                except Exception as e:
                    w_status.value = f'<span style="color:#ff6b35">Errore durante la telemetria VQE: {e}</span>'
                if not df_vqe.empty:
                    df_vqe, vqe_ai_meta = _heal_telemetry(df_vqe)

        df_md, corr_matrix = pd.DataFrame(), pd.DataFrame()
        md_ai_meta = dict(_NEUTRAL_AI_META)
        md_is_real, md_note = False, ''
        if w_md_enabled.value:
            w_status.value = '<span style="color:#00c8ff">⏳ Telemetria MD...</span>'
            # Real dynamics under hamiltonian_values when it's compatible with
            # this circuit's statevector; otherwise run_md_telemetry falls
            # back to the labeled mock itself (see md_telemetry.py) -- no
            # branching needed here.
            df_md, _raw_corr = run_md_telemetry(
                w_md_steps.value, w_md_temp.value,
                hamiltonian_values=hamiltonian_values, sv=res['sim'].sv,
                n_qubits=res['n_qubits'], seed=int(w_seed.value),
            )
            md_is_real = df_md.attrs.get('is_real', False)
            md_note = df_md.attrs.get('note', '')
            df_md, md_ai_meta = _heal_telemetry(df_md)
            corr_matrix = df_md.corr(method='pearson')

        run_history.append({
            'nome': res['nome'], 'n_qubits': res['n_qubits'], 'tempo': res['tempo'], 'ram': res['ram'],
            'porte_count': res['porte_count'], 'entropy': float(res['entropy']), 'fidelity': res['fidelity'],
            'stato_dominante': res['stato_dominante'], 'noise_model': w_noise_model.value, 'noise_p': w_noise_p.value,
        })

        w_status.value = '<span style="color:#00c8ff">⏳ Rendering pannelli...</span>'
        metrics = compute_overview_metrics(res, w_noise_model.value, w_noise_p.value)
        overview_fig = build_panel_overview(res, df_vqe, corr_matrix, w_noise_model.value, w_noise_p.value)
        fisica_fig = build_panel_fisica(res, seed=int(w_seed.value))
        mosaico_fig = build_panel_mosaico(res)
        vqe_fig = build_panel_vqe_results(df_vqe)
        md_fig = build_panel_md_results(df_md, corr_matrix)
        perf_fig = build_panel_performance(res, run_history)
        helix_fig = build_3d_helix_patch(res['n_qubits'], res['prob'])
        ham_name = (w_ham_select.value if w_ham_mode.value == 'Libreria Built-in' else 'Custom JSON') if w_ham_enabled.value else None
        ham_fig = build_panel_hamiltonian(hamiltonian_values, ham_name or 'nessuna')
        mitigation_fig = build_panel_mitigation(mitigation_res, res) if mitigation_res else None

        _metrics_html = '<div style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(
            f'<div style="border:1px solid #333;border-radius:6px;padding:6px 10px">'
            f'<div style="font-size:11px;color:#888">{m["label"]}</div>'
            f'<div style="font-size:15px;font-weight:bold">{m["value"]}</div></div>'
            for m in metrics
        ) + '</div>'
        with tab_outputs[0]:
            clear_output(wait=True)
            display(widgets.HTML(
                f'<b>🛡️ AI Vector-Healing Shield — Telemetria VQE</b> — '
                f'Fallback: {"Sì" if vqe_ai_meta["fallback_triggered"] else "No"}, '
                f'Raggio: {vqe_ai_meta["adaptive_radius_used"]}, '
                f'Errore ricostruzione: {vqe_ai_meta["reconstruction_error"]:.4f}'
            ))
            display(widgets.HTML(_metrics_html))
            display(overview_fig)
            plt.close(overview_fig)
        _show(1, fisica_fig)
        _show(2, mosaico_fig)
        _show(3, vqe_fig)
        with tab_outputs[4]:
            clear_output(wait=True)
            md_badge_color = '#00ff9d' if md_is_real else '#ff9d00'
            md_badge_text = 'DATI REALI' if md_is_real else 'MOCK'
            display(widgets.HTML(
                f'<b style="color:{md_badge_color}">● {md_badge_text}</b> — {md_note}'
            ))
            display(widgets.HTML(
                f'<b>🛡️ AI Vector-Healing Shield — Telemetria MD</b> — '
                f'Fallback: {"Sì" if md_ai_meta["fallback_triggered"] else "No"}, '
                f'Raggio: {md_ai_meta["adaptive_radius_used"]}, '
                f'Errore ricostruzione: {md_ai_meta["reconstruction_error"]:.4f}'
            ))
            display(md_fig)
            plt.close(md_fig)
        _show(5, perf_fig)
        _show(6, helix_fig)  # plotly Figure -- not closed via plt.close (see _show)
        _show(7, ham_fig)
        _show(8, mitigation_fig)

        w_status.value = '<span style="color:#00ff9d">● Fatto</span>'
        w_run.disabled = False

    w_run.on_click(_on_run_clicked)

    sidebar = widgets.VBox([
        widgets.HTML('<b>⚙️ Circuito</b>'), w_source_mode, w_circuit, w_qasm_text,
        widgets.HTML('<b>🧮 Motore</b>'), w_engine,
        widgets.HTML('<b>🌪️ Rumore</b>'), w_noise_model, w_noise_p, w_shots, w_seed, w_double_precision,
        widgets.HTML('<b>🩹 Error Mitigation (ZNE)</b>'), w_zne_enabled, w_zne_healing, w_zne_target_sigma,
        widgets.HTML('<b>🧪 VQE</b>'), w_vqe_enabled, w_vqe_epochs, w_vqe_lr, w_vqe_beta1, w_vqe_beta2, w_confirm_heavy_vqe,
        widgets.HTML('<b>🧬 Hamiltoniana personalizzata</b>'), w_ham_enabled, w_ham_box,
        widgets.HTML('<b>🌡️ Molecular Dynamics</b>'), w_md_enabled, w_md_steps, w_md_temp,
        widgets.HBox([w_run, w_status]),
    ])

    panel_quantum_sim = widgets.VBox([sidebar, tabs])

    # ════════════════════════════════════════════════════════════════
    # Vector Healing — sandbox for ia_utils.vector_healing.enhanced_dense_healing_hybrid
    # (same panel as ui_pages/vector_healing.py's Streamlit page)
    # ════════════════════════════════════════════════════════════════
    from .vector_healing_demo import generate_corrupted_sequence
    from ia_utils.vector_healing import enhanced_dense_healing_hybrid

    VH_HIDDEN_DIM = 6
    vh_state = {}

    w_vh_n_steps = widgets.IntSlider(value=80, min=10, max=150, description='Step / token:',
                                      style={'description_width': 'initial'})
    w_vh_anomaly_pct = widgets.IntSlider(value=10, min=5, max=30, description='% Anomalie:',
                                          style={'description_width': 'initial'})
    w_vh_run = widgets.Button(description='▶ Genera ed Esegui Healing', button_style='primary')
    w_vh_status = widgets.HTML(value='')
    w_vh_channel = widgets.Dropdown(options=list(range(VH_HIDDEN_DIM)), value=0, description='Canale:')
    w_vh_shield_out = widgets.Output()
    w_vh_plot_out = widgets.Output()

    def _vh_render_channel(channel):
        if not vh_state:
            return
        ideal, corrupted, healed = vh_state['ideal'], vh_state['corrupted'], vh_state['healed']
        x = np.arange(vh_state['n_steps'])
        ideal_channel = ideal[:, channel]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, ideal_channel, label='Ideale', color='#888888', linewidth=1.5, linestyle='--')
        ax.plot(x, corrupted[:, channel], label='Corrotto', color='#ff4b4b', linewidth=1.2, alpha=0.75)
        ax.plot(x, healed[:, channel], label='Curato', color='#00c853', linewidth=2.2)
        margin = 1.5
        ax.set_ylim(ideal_channel.min() - margin, ideal_channel.max() + margin)
        ax.set_xlabel('Step / Token')
        ax.set_ylabel(f'Valore (canale {channel})')
        ax.set_title('Vector Healing — Ideale vs Corrotto vs Curato')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        with w_vh_plot_out:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)

    def _on_vh_run_clicked(_btn):
        w_vh_run.disabled = True
        w_vh_status.value = '<span style="color:#00c8ff">⏳ Generazione e healing...</span>'
        rng = np.random.default_rng()
        ideal, corrupted = generate_corrupted_sequence(
            w_vh_n_steps.value, VH_HIDDEN_DIM, w_vh_anomaly_pct.value, rng)
        healed, metadata = enhanced_dense_healing_hybrid(corrupted)
        vh_state.clear()
        vh_state.update(ideal=ideal, corrupted=corrupted, healed=healed, n_steps=w_vh_n_steps.value)
        with w_vh_shield_out:
            clear_output(wait=True)
            display(widgets.HTML(
                f'<b>🛡️ AI Vector-Healing Shield</b> — '
                f'Fallback: {"Sì" if metadata["fallback_triggered"] else "No"}, '
                f'Raggio: {metadata["adaptive_radius_used"]}, '
                f'Errore ricostruzione: {metadata["reconstruction_error"]:.4f}'
            ))
        _vh_render_channel(w_vh_channel.value)
        w_vh_status.value = '<span style="color:#00ff9d">● Fatto</span>'
        w_vh_run.disabled = False

    def _on_vh_channel_change(change):
        _vh_render_channel(change['new'])

    w_vh_run.on_click(_on_vh_run_clicked)
    w_vh_channel.observe(_on_vh_channel_change, names='value')

    panel_vector_healing = widgets.VBox([
        widgets.HTML('<b>🧬 AI Vector Healing Dashboard</b> — sandbox interattiva per lo scudo '
                      'anti-crash che protegge in produzione la telemetria VQE/MD del Quantum Simulator.'),
        widgets.VBox([
            widgets.HTML('<b>⚙️ Configurazione</b>'), w_vh_n_steps, w_vh_anomaly_pct,
            widgets.HBox([w_vh_run, w_vh_status]),
        ]),
        w_vh_shield_out,
        w_vh_channel,
        w_vh_plot_out,
    ])

    # ════════════════════════════════════════════════════════════════
    # Quantum Scars — live PXP (Rydberg blockade) demo
    # (same panel as ui_pages/quantum_scars.py's Streamlit page)
    # ════════════════════════════════════════════════════════════════
    from .scars_engine import build_pxp, run_experiment, DT_CHUNK, N_CHUNK

    w_qs_n_qubits = widgets.IntSlider(value=10, min=6, max=12, description='Qubit (catena PXP):',
                                       style={'description_width': 'initial'})
    w_qs_noise_p = widgets.FloatSlider(value=0.01, min=0.0, max=0.05, step=0.005,
                                        description='Rumore depol. (p):', style={'description_width': 'initial'})
    w_qs_n_traj = widgets.IntSlider(value=10, min=5, max=30, description='Traiettorie:')
    w_qs_protection = widgets.Dropdown(
        options=['Nessuna', 'Proiezione vincolo (economica)', 'Proiezione torre (ideale)'],
        value='Nessuna', description='Protezione:', style={'description_width': 'initial'},
    )
    w_qs_run = widgets.Button(description='▶ Esegui esperimento', button_style='primary')
    w_qs_status = widgets.HTML(value='')
    w_qs_metrics_out = widgets.Output()
    w_qs_plot_out = widgets.Output()

    def _on_qs_run_clicked(_btn):
        w_qs_run.disabled = True
        w_qs_status.value = '<span style="color:#00c8ff">⏳ Diagonalizzazione H_PXP...</span>'
        pxp = build_pxp(w_qs_n_qubits.value)
        w_qs_status.value = '<span style="color:#00c8ff">⏳ Simulazione delle traiettorie...</span>'
        fidelity_protected = run_experiment(
            pxp, n_trajectories=w_qs_n_traj.value, noise_p=w_qs_noise_p.value,
            protection=w_qs_protection.value, weight_threshold=0.02, base_seed=1000,
        )
        fidelity_clean = run_experiment(
            pxp, n_trajectories=1, noise_p=0.0, protection='Nessuna',
            weight_threshold=0.02, base_seed=0,
        )
        with w_qs_metrics_out:
            clear_output(wait=True)
            display(widgets.HTML(
                '<div style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(
                    f'<div style="border:1px solid #333;border-radius:6px;padding:6px 10px">'
                    f'<div style="font-size:11px;color:#888">{label}</div>'
                    f'<div style="font-size:15px;font-weight:bold">{value}</div></div>'
                    for label, value in [
                        ('Qubit', pxp['n_qubits']),
                        ('Dimensione Hilbert', pxp['dim']),
                        ('Sottospazio valido (vincolo)', pxp['valid_dim']),
                        ('Soffitto torre (peso Néel)', f"{pxp['tower_ceiling']*100:.1f}%"),
                    ]
                ) + '</div>'
            ))
        times = np.arange(1, N_CHUNK + 1) * DT_CHUNK
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(times, fidelity_clean, label='Nessun rumore (riferimento)',
                color='gold', linewidth=1.5, linestyle='--')
        ax.plot(times, fidelity_protected, label=f'p={w_qs_noise_p.value} — {w_qs_protection.value}',
                color='#00e5ff', linewidth=2.2)
        ax.set_xlabel('Tempo t')
        ax.set_ylabel('Fedeltà |⟨Néel|ψ(t)⟩|²')
        ax.set_title('Revival della scar PXP')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        with w_qs_plot_out:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)
        w_qs_status.value = '<span style="color:#00ff9d">● Fatto</span>'
        w_qs_run.disabled = False

    w_qs_run.on_click(_on_qs_run_clicked)

    panel_quantum_scars = widgets.VBox([
        widgets.HTML('<b>🌀 Quantum Many-Body Scars: PXP Live Demo</b> — parte dallo stato di Néel, '
                      'inietta rumore reale (NoiseModel.apply_to_sv) e confronta i revival di fedeltà '
                      'con/senza protezione. Oltre 10 qubit la diagonalizzazione esatta rallenta.'),
        widgets.VBox([
            widgets.HTML('<b>⚙️ Configurazione</b>'), w_qs_n_qubits, w_qs_noise_p, w_qs_n_traj, w_qs_protection,
            widgets.HBox([w_qs_run, w_qs_status]),
        ]),
        w_qs_metrics_out,
        w_qs_plot_out,
    ])

    # ════════════════════════════════════════════════════════════════
    # Research Bridge — hypothesis + real numbers vs. an external AI
    # (same panel as ui_pages/research_bridge.py's Streamlit page)
    # ════════════════════════════════════════════════════════════════
    from .research_bridge import (
        build_context_block, build_search_query, call_custom_api, new_log_entry,
        build_next_search_query, call_local_cli,
    )
    import urllib.parse
    import json as _json

    rb_log = []
    rb_chain = []

    w_rb_hypothesis = widgets.Textarea(
        description='Ipotesi:', placeholder='Cosa stai verificando?',
        layout=widgets.Layout(width='650px', height='80px'), style={'description_width': 'initial'},
    )
    w_rb_real_data = widgets.Textarea(
        description='Dati reali (opz.):', placeholder='Incolla output/CSV di un tuo esperimento...',
        layout=widgets.Layout(width='650px', height='90px'), style={'description_width': 'initial'},
    )
    w_rb_notes = widgets.Text(description='Note (opz.):', layout=widgets.Layout(width='650px'),
                               style={'description_width': 'initial'})
    w_rb_context_out = widgets.Output()

    def _rb_context_block():
        return build_context_block(w_rb_hypothesis.value, w_rb_real_data.value, w_rb_notes.value)

    def _rb_refresh_context(*_):
        with w_rb_context_out:
            clear_output(wait=True)
            if w_rb_hypothesis.value.strip():
                print(_rb_context_block())
            else:
                display(widgets.HTML('<i>Scrivi un\'ipotesi sopra per generare il blocco da incollare.</i>'))

    for _w in (w_rb_hypothesis, w_rb_real_data, w_rb_notes):
        _w.observe(_rb_refresh_context, names='value')

    # ── Google (chained ReAct search) ──────────────────────────────
    w_rb_react_url = widgets.Text(description='Endpoint API (opz.):', layout=widgets.Layout(width='500px'),
                                   style={'description_width': 'initial'})
    w_rb_react_key = widgets.Password(description='Chiave API (opz.):', style={'description_width': 'initial'})
    w_rb_react_model = widgets.Text(description='Modello (opz.):', style={'description_width': 'initial'})
    w_rb_google_history_out = widgets.Output()
    w_rb_google_query_out = widgets.Output()
    w_rb_google_response = widgets.Textarea(description='Risposta Google:',
                                             layout=widgets.Layout(width='650px', height='100px'),
                                             style={'description_width': 'initial'})
    w_rb_google_save = widgets.Button(description='✅ Salva questo passo')
    w_rb_google_stop = widgets.Button(description='🏁 Ferma e salva la catena')
    w_rb_google_status = widgets.HTML(value='')

    def _rb_current_query():
        if not rb_chain:
            return build_search_query(w_rb_hypothesis.value)
        return rb_chain[-1].get('_next_query', build_search_query(w_rb_hypothesis.value))

    def _rb_render_google():
        with w_rb_google_history_out:
            clear_output(wait=True)
            for i, step in enumerate(rb_chain):
                preview = step['result'][:300] + ('...' if len(step['result']) > 300 else '')
                display(widgets.HTML(f'<b>Passo {i + 1}:</b> <code>{step["query"]}</code><br>'
                                      f'<span style="color:#888">{preview}</span>'))
        with w_rb_google_query_out:
            clear_output(wait=True)
            query = _rb_current_query()
            url = 'https://www.google.com/search?q=' + urllib.parse.quote(query) + '&udm=50'
            label = 'Prima ricerca' if not rb_chain else f'Follow-up (passo {len(rb_chain) + 1})'
            display(widgets.HTML(
                f'<b>{label}:</b> <code>{query}</code><br>'
                f'<a href="{url}" target="_blank">Apri Google ({"nuova conversazione" if not rb_chain else "nella conversazione GIÀ aperta"})</a>'
            ))

    def _on_rb_google_save(_btn):
        if not w_rb_google_response.value.strip():
            return
        query = _rb_current_query()
        rb_chain.append({'query': query, 'result': w_rb_google_response.value.strip()})
        w_rb_google_response.value = ''
        if w_rb_react_url.value and w_rb_react_key.value:
            w_rb_google_status.value = '<span style="color:#00c8ff">⏳ Ragiono sulla prossima ricerca (ReAct)...</span>'
            try:
                next_step = build_next_search_query(
                    w_rb_hypothesis.value, rb_chain, w_rb_react_url.value,
                    w_rb_react_key.value, w_rb_react_model.value)
            except Exception as e:
                next_step = None
                w_rb_google_status.value = f'<span style="color:#ff6b35">Ragionamento fallito: {e}</span>'
            if next_step is not None:
                if next_step['done']:
                    summary = '\n\n'.join(f"Passo {i+1}: {s['query']} -> {s['result']}"
                                           for i, s in enumerate(rb_chain))
                    rb_log.append(new_log_entry(
                        f'Google (catena ReAct, {len(rb_chain)} passi)', w_rb_hypothesis.value, summary,
                        f"SINTESI FINALE: {next_step['reasoning']}", w_rb_react_model.value))
                    rb_chain.clear()
                    w_rb_google_status.value = f'<span style="color:#00ff9d">Catena completa: {next_step["reasoning"]}</span>'
                    _rb_render_log()
                else:
                    rb_chain[-1]['_next_query'] = next_step['query'] or query
                    w_rb_google_status.value = ''
        _rb_render_google()

    def _on_rb_google_stop(_btn):
        if not rb_chain:
            return
        summary = '\n\n'.join(f"Passo {i+1}: {s['query']} -> {s['result']}" for i, s in enumerate(rb_chain))
        rb_log.append(new_log_entry(f'Google (catena, {len(rb_chain)} passi)', w_rb_hypothesis.value,
                                     summary, rb_chain[-1]['result']))
        rb_chain.clear()
        w_rb_google_status.value = '<span style="color:#00ff9d">Catena salvata nel log.</span>'
        _rb_render_google()
        _rb_render_log()

    w_rb_google_save.on_click(_on_rb_google_save)
    w_rb_google_stop.on_click(_on_rb_google_stop)
    w_rb_hypothesis.observe(lambda _c: _rb_render_google(), names='value')

    tab_google = widgets.VBox([
        widgets.HTML('<i>Nessuna API per l\'AI Mode di google.com — apri la ricerca UNA volta, poi '
                      'continua nella STESSA conversazione per i follow-up.</i>'),
        w_rb_google_query_out,
        w_rb_google_response,
        widgets.HBox([w_rb_google_save, w_rb_google_stop]),
        w_rb_google_status,
        widgets.HTML('<b>Cronologia:</b>'), w_rb_google_history_out,
        widgets.Accordion(children=[widgets.VBox([w_rb_react_url, w_rb_react_key, w_rb_react_model])],
                           titles=('🧠 Chiave per il ragionamento tra un passo e l\'altro (opzionale)',)),
    ])

    # ── Colab (copy/paste) ──────────────────────────────────────────
    w_rb_colab_block_out = widgets.Output()
    w_rb_colab_response = widgets.Textarea(description='Risposta:', layout=widgets.Layout(width='650px', height='100px'),
                                            style={'description_width': 'initial'})
    w_rb_colab_save = widgets.Button(description='Salva nel log')
    w_rb_colab_status = widgets.HTML(value='')

    def _on_rb_colab_save(_btn):
        if not w_rb_colab_response.value.strip():
            return
        rb_log.append(new_log_entry('Colab personale', w_rb_hypothesis.value, _rb_context_block(),
                                     w_rb_colab_response.value.strip()))
        w_rb_colab_status.value = '<span style="color:#00ff9d">Salvato.</span>'
        _rb_render_log()

    w_rb_colab_save.on_click(_on_rb_colab_save)

    def _rb_refresh_colab_block(*_):
        with w_rb_colab_block_out:
            clear_output(wait=True)
            print(_rb_context_block())
            display(widgets.HTML('<a href="https://colab.research.google.com/" target="_blank">Apri Colab</a>'))

    for _w in (w_rb_hypothesis, w_rb_real_data, w_rb_notes):
        _w.observe(_rb_refresh_colab_block, names='value')

    tab_colab = widgets.VBox([
        widgets.HTML('<i>Copia il blocco intero, incollalo in una cella/chat del tuo notebook, incolla la risposta.</i>'),
        w_rb_colab_block_out, w_rb_colab_response,
        widgets.HBox([w_rb_colab_save, w_rb_colab_status]),
    ])

    # ── Custom API (automated) ───────────────────────────────────────
    w_rb_api_url = widgets.Text(description='Endpoint API:', layout=widgets.Layout(width='500px'),
                                 style={'description_width': 'initial'})
    w_rb_api_key = widgets.Password(description='Chiave API:', style={'description_width': 'initial'})
    w_rb_api_model = widgets.Text(description='Modello (opz.):', style={'description_width': 'initial'})
    w_rb_api_send = widgets.Button(description='🚀 Invia direttamente', button_style='primary')
    w_rb_api_out = widgets.Output()

    def _on_rb_api_send(_btn):
        with w_rb_api_out:
            clear_output(wait=True)
            if not (w_rb_api_url.value and w_rb_api_key.value):
                display(widgets.HTML('<span style="color:#ff6b35">Endpoint e chiave richiesti.</span>'))
                return
            print('⏳ Chiamata in corso...')
        try:
            reply = call_custom_api(_rb_context_block(), w_rb_api_url.value, w_rb_api_key.value, w_rb_api_model.value)
            rb_log.append(new_log_entry('Chiave personale', w_rb_hypothesis.value, _rb_context_block(),
                                         reply, w_rb_api_model.value))
            with w_rb_api_out:
                clear_output(wait=True)
                display(widgets.HTML('<span style="color:#00ff9d">Risposta ricevuta e salvata nel log.</span>'))
                print(reply)
            _rb_render_log()
        except Exception as e:
            with w_rb_api_out:
                clear_output(wait=True)
                display(widgets.HTML(f'<span style="color:#ff6b35">Chiamata fallita: {e}</span>'))

    w_rb_api_send.on_click(_on_rb_api_send)

    tab_custom = widgets.VBox([
        widgets.HTML('<i>Riconosce da solo l\'API Anthropic da un URL con "anthropic.com"; qualsiasi altro '
                      'endpoint è trattato come compatibile OpenAI chat-completions.</i>'),
        w_rb_api_url, w_rb_api_key, w_rb_api_model, w_rb_api_send, w_rb_api_out,
    ])

    # ── Local CLI (automated, no separate API cost) ──────────────────
    w_rb_cli_command = widgets.Text(description='Comando:', placeholder="es: claude -p",
                                     layout=widgets.Layout(width='500px'), style={'description_width': 'initial'})
    w_rb_cli_run = widgets.Button(description='💻 Esegui in locale', button_style='primary')
    w_rb_cli_out = widgets.Output()

    def _on_rb_cli_run(_btn):
        with w_rb_cli_out:
            clear_output(wait=True)
            if not w_rb_cli_command.value.strip():
                display(widgets.HTML('<span style="color:#ff6b35">Comando richiesto.</span>'))
                return
            print('⏳ Esecuzione in corso...')
        try:
            reply = call_local_cli(_rb_context_block(), w_rb_cli_command.value)
            rb_log.append(new_log_entry('IA locale (CLI)', w_rb_hypothesis.value, _rb_context_block(),
                                         reply, w_rb_cli_command.value))
            with w_rb_cli_out:
                clear_output(wait=True)
                display(widgets.HTML('<span style="color:#00ff9d">Risposta ricevuta e salvata nel log.</span>'))
                print(reply)
            _rb_render_log()
        except Exception as e:
            with w_rb_cli_out:
                clear_output(wait=True)
                display(widgets.HTML(f'<span style="color:#ff6b35">Esecuzione fallita: {e}</span>'))

    w_rb_cli_run.on_click(_on_rb_cli_run)

    tab_local = widgets.VBox([
        widgets.HTML('<i>Richiama un\'IA a riga di comando già autenticata sul PC (Claude Code o altro), '
                      'tramite un comando locale invece di una chiamata a pagamento.</i>'),
        w_rb_cli_command, w_rb_cli_run, w_rb_cli_out,
    ])

    # ── Log + export ──────────────────────────────────────────────────
    w_rb_log_out = widgets.Output()
    w_rb_log_export = widgets.Button(description='⬇️ Esporta log (JSON)')
    w_rb_log_export_status = widgets.HTML(value='')

    def _rb_render_log():
        with w_rb_log_out:
            clear_output(wait=True)
            if not rb_log:
                display(widgets.HTML('<i>Nessuna voce nel log ancora.</i>'))
                return
            display(widgets.HTML(f'<b>Log ({len(rb_log)} voci)</b>'))
            for entry in reversed(rb_log):
                display(widgets.HTML(
                    f'<div style="border:1px solid #333;border-radius:6px;padding:8px;margin:4px 0">'
                    f'<b>{entry["timestamp"]} — {entry["source"]}</b><br>'
                    f'<i>Ipotesi:</i> {entry["hypothesis"]}<br>'
                    + (f'<span style="color:#888">Modello: {entry["model"]}</span><br>' if entry.get("model") else '')
                    + f'<div style="color:#ccc;white-space:pre-wrap">{entry["response"][:500]}</div></div>'
                ))

    def _on_rb_log_export(_btn):
        path = 'research_bridge_log.json'
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(rb_log, f, indent=2, ensure_ascii=False)
        try:
            from google.colab import files as _colab_files
            _colab_files.download(path)
            w_rb_log_export_status.value = f'<span style="color:#00ff9d">Download avviato: {path}</span>'
        except ImportError:
            from IPython.display import FileLink
            with w_rb_log_out:
                display(FileLink(path))
            w_rb_log_export_status.value = f'<span style="color:#00ff9d">Salvato come {path} (link sopra).</span>'

    w_rb_log_export.on_click(_on_rb_log_export)

    rb_tabs = widgets.Tab(children=[tab_google, tab_colab, tab_custom, tab_local])
    for i, name in enumerate(['🔍 Google', '📓 Colab personale', '🔑 La tua IA (API)', '💻 IA locale (CLI)']):
        rb_tabs.set_title(i, name)

    panel_research_bridge = widgets.VBox([
        widgets.HTML('<b>🌉 Research Bridge</b> — impacchetta un\'ipotesi (più i numeri reali di un tuo '
                      'esperimento) in un blocco pronto da incollare, e confrontala con un\'IA esterna a scelta.'),
        w_rb_hypothesis, w_rb_real_data, w_rb_notes,
        widgets.HTML('<b>Blocco di contesto:</b>'), w_rb_context_out,
        rb_tabs,
        widgets.HTML('<b>Log</b>'), w_rb_log_out,
        widgets.HBox([w_rb_log_export, w_rb_log_export_status]),
    ])
    _rb_refresh_context()
    _rb_refresh_colab_block()
    _rb_render_google()
    _rb_render_log()

    # ════════════════════════════════════════════════════════════════
    # Mode switch — same show/hide pattern already used for w_qasm_text
    # above; the Quantum Simulator panel itself (sidebar + tabs) is
    # untouched, just wrapped.
    # ════════════════════════════════════════════════════════════════
    mode_panels = {
        '⚛️ Quantum Simulator': panel_quantum_sim,
        '🧬 Vector Healing': panel_vector_healing,
        '🌀 Quantum Scars': panel_quantum_scars,
        '🌉 Research Bridge': panel_research_bridge,
    }
    w_mode = widgets.ToggleButtons(options=list(mode_panels.keys()), value='⚛️ Quantum Simulator')
    for name, p in mode_panels.items():
        p.layout.display = None if name == w_mode.value else 'none'

    def _on_mode_change(change):
        for name, p in mode_panels.items():
            p.layout.display = None if name == change['new'] else 'none'

    w_mode.observe(_on_mode_change, names='value')

    panel = widgets.VBox([w_mode] + list(mode_panels.values()))
    display(panel)
    return panel
