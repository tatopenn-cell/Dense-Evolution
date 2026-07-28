"""
Interactive ipywidgets panel — Colab/Jupyter-native, no Streamlit, no
external tunnel/link. Lives entirely in the calling cell's output.

Full feature parity with ui_pages/quantum_simulator.py's Streamlit page
(same 9 output panels, same sidebar controls: circuit source, engine,
noise, ZNE/predictive healing, VQE, custom Hamiltonian, MD) -- built on
the same dashboard_core functions that page calls, just driven by
ipywidgets instead of Streamlit's rerun-the-whole-script model.

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

    panel = widgets.VBox([sidebar, tabs])
    display(panel)
    return panel
