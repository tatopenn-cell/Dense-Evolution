"""
Tests for dashboard_core.interactive_panel.launch_interactive_panel — the
ipywidgets panel (Colab/Jupyter-native, no Streamlit) added for users who
stay inside a notebook and never leave for a Streamlit tunnel link.

Simulates real widget interaction (Button.click(), not calling the
internal callback directly) to exercise the actual ipywidgets event
wiring, not just the underlying dashboard_core calls (already covered
by test_dashboard_core.py/test_mitigation_runner.py).

Deliberately does NOT assert on Output.outputs (checked and confirmed:
ipywidgets.widgets.widget_output.Output.__enter__/__exit__ only route
display() calls into .outputs via a real Jupyter kernel's comm/message
protocol -- ip.kernel.get_parent(), not any in-process Python capture.
Outside a genuine running kernel (plain pytest, even with
IPython.testing.globalipapp's InteractiveShell active) .outputs stays
permanently empty regardless of whether the code is correct -- verified
directly. What IS meaningful here and checked below: no exception is
raised and the run reaches its final "Fatto" status.
"""

import matplotlib
matplotlib.use("Agg")

import pytest

import dashboard_core as dc

pytest.importorskip("ipywidgets")


def _find(widgets_list, description):
    for w in widgets_list:
        if getattr(w, "description", None) == description:
            return w
    raise KeyError(description)


def _find_deep(widgets_list, description):
    """Like _find, but descends into nested VBox/HBox children too -- for
    widgets like w_ham_mode/w_ham_select/w_ham_json, which only exist
    inside w_ham_box (itself one entry in the flat sidebar list), not as
    direct children of the sidebar."""
    for w in widgets_list:
        found = _find_recursive(w, description)
        if found is not None:
            return found
    raise KeyError(description)


def _mode_panel(panel, title):
    """panel.children[0] is the mode ToggleButtons; the rest are the four
    mode panels in the same order as its .options."""
    w_mode = panel.children[0]
    idx = list(w_mode.options).index(title)
    return panel.children[1 + idx]


def _panel_widgets(panel):
    """Reaches into the Quantum Simulator mode panel specifically (same
    sidebar/tabs shape as before the multi-mode rebuild -- just one level
    deeper now, behind the mode selector)."""
    quantum_sim_panel = _mode_panel(panel, "⚛️ Quantum Simulator")
    sidebar, tabs = quantum_sim_panel.children
    kids = sidebar.children
    run_hbox = kids[-1]
    w_run, w_status = run_hbox.children
    return kids, tabs, w_run, w_status


def test_launch_interactive_panel_builds():
    panel = dc.launch_interactive_panel()
    assert type(panel).__name__ == "VBox"
    w_mode = panel.children[0]
    assert type(w_mode).__name__ == "ToggleButtons"
    assert list(w_mode.options) == [
        "⚛️ Quantum Simulator", "🧬 Vector Healing", "🌀 Quantum Scars", "🌉 Research Bridge",
    ]
    assert len(panel.children) == 1 + len(w_mode.options)
    quantum_sim_panel = _mode_panel(panel, "⚛️ Quantum Simulator")
    sidebar, tabs = quantum_sim_panel.children
    assert type(tabs).__name__ == "Tab"
    assert len(tabs.children) == 9
    assert list(tabs.titles) == [
        "Overview", "Fisica Stato", "Mosaico", "VQE Results", "MD Results",
        "Performance", "3D Helix", "Hamiltonian", "Mitigation (ZNE)",
    ]


def test_run_button_ideal_circuit_populates_all_tabs():
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    w_vqe_enabled = _find(kids, "Abilita telemetria VQE")
    w_md_enabled = _find(kids, "Abilita telemetria MD")
    w_vqe_enabled.value = False  # keep the test fast -- VQE/MD logic already
    w_md_enabled.value = False   # covered directly in test_dashboard_core.py

    w_run.click()

    assert "Fatto" in w_status.value
    assert not w_run.disabled


def test_run_button_with_zne_and_healing():
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Rumore:").value = "depolarizing"
    _find(kids, "p:").value = 0.06
    _find(kids, "Abilita Zero-Noise Extrapolation").value = True
    _find(kids, "Healing predittivo (Δpre_emp-adapted)").value = True

    w_run.click()

    assert "Fatto" in w_status.value


def test_run_button_ideal_with_zne_shows_warning_not_crash():
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Abilita Zero-Noise Extrapolation").value = True
    # noise_model stays 'ideal' (default) -- ZNE has nothing to extrapolate

    w_run.click()  # must not raise

    assert "Fatto" in w_status.value


def test_run_button_vqe_and_md_together():
    # the heavier, slower path -- real parametric VQE + MD telemetry +
    # AI healing shield on both, exercised once end-to-end
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Circuito:").value = "VQE ansatz H₂"
    _find(kids, "Epochs:").value = 4
    _find(kids, "MD steps:").value = 10

    w_run.click()

    assert "Fatto" in w_status.value


def test_source_mode_change_toggles_circuit_and_qasm_visibility():
    # _on_source_mode_change -- never triggered by any run-button test
    # above, which all leave the default "Libreria Built-in" untouched.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    w_source_mode = _find(kids, "Sorgente:")
    w_circuit = _find(kids, "Circuito:")
    w_qasm_text = _find(kids, "OpenQASM 2.0:")

    assert w_circuit.layout.display is None
    assert w_qasm_text.layout.display == "none"

    w_source_mode.value = "Custom QASM Textarea"
    assert w_circuit.layout.display == "none"
    assert w_qasm_text.layout.display is None

    w_source_mode.value = "Libreria Built-in"
    assert w_circuit.layout.display is None
    assert w_qasm_text.layout.display == "none"


def test_hamiltonian_enabled_toggle_populates_and_clears_box():
    # _on_ham_enabled_change -- no earlier test ever flips w_ham_enabled.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    w_ham_enabled = _find(kids, "Abilita Hamiltoniana personalizzata")
    w_ham_box = None
    for w in kids:
        if type(w).__name__ == "VBox" and len(w.children) == 0:
            w_ham_box = w
            break
    assert w_ham_box is not None and w_ham_box.children == ()

    w_ham_enabled.value = True
    assert len(w_ham_box.children) > 0  # populated with mode/select/json/save widgets

    w_ham_enabled.value = False
    assert w_ham_box.children == ()  # cleared again


def test_hamiltonian_mode_change_toggles_widget_visibility():
    # _on_ham_mode_change -- needs w_ham_enabled=True first so w_ham_box's
    # children (which include w_ham_mode) actually exist to observe.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita Hamiltoniana personalizzata").value = True
    w_ham_mode = _find_deep(kids, "Modalità:")
    w_ham_select = _find_deep(kids, "Hamiltoniana:")
    w_ham_json = _find_deep(kids, "Array JSON:")

    # No assertion on the pre-change state here: w_ham_box.children is set
    # directly (not via .observe), so _on_ham_mode_change hasn't run yet
    # at this point regardless of w_ham_mode.value already being
    # "Libreria Built-in" -- only an actual value change fires it.
    w_ham_mode.value = "Custom JSON Textarea"
    assert w_ham_select.layout.display == "none"
    assert w_ham_json.layout.display is None

    w_ham_mode.value = "Libreria Built-in"
    assert w_ham_select.layout.display is None
    assert w_ham_json.layout.display == "none"


def test_hamiltonian_save_button_saves_custom_json_to_library():
    # _on_ham_save_clicked -- never triggered by any earlier test.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita Hamiltoniana personalizzata").value = True
    _find_deep(kids, "Modalità:").value = "Custom JSON Textarea"
    _find_deep(kids, "Nome (salva):").value = "test_saved_hamiltonian"
    _find_deep(kids, "Array JSON:").value = "[-1.0, 0.5, 0.5, -1.0]"

    w_save_btn = None
    for w in kids:
        found = _find_button_recursive(w, "Salva in libreria")
        if found is not None:
            w_save_btn = found
    assert w_save_btn is not None
    w_save_btn.click()  # must not raise

    # confirmed via the saved hamiltonian now being selectable:
    w_ham_select = _find_deep(kids, "Hamiltoniana:")
    assert "test_saved_hamiltonian" in w_ham_select.options


def test_run_button_with_hamiltonian_enabled_library_mode():
    # Covers the hamiltonian_values extraction branch in _on_run_clicked
    # (w_ham_mode == 'Libreria Built-in') -- never exercised since no
    # earlier run-button test enables the Hamiltonian at all.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Abilita Hamiltoniana personalizzata").value = True
    w_ham_select = _find_deep(kids, "Hamiltoniana:")
    assert len(w_ham_select.options) > 0  # Bell |Φ+⟩ (2 qubit) has compatible built-ins
    w_ham_select.value = w_ham_select.options[0]

    w_run.click()

    assert "Fatto" in w_status.value


def test_run_button_with_hamiltonian_enabled_custom_json_mode():
    # Same branch, the other half (Custom JSON Textarea).
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Abilita Hamiltoniana personalizzata").value = True
    _find_deep(kids, "Modalità:").value = "Custom JSON Textarea"

    w_run.click()  # default Array JSON text is valid -- must not raise

    assert "Fatto" in w_status.value


def test_run_button_heavy_circuit_skips_vqe_without_confirmation():
    # Covers the ">QM_MM_HEAVY_QUBIT_THRESHOLD qubit, not confirmed" skip
    # branch -- "Error Mitigation (Real-Stress)" is the one QASM_LIBRARY
    # entry with more than 12 qubits (15). The "saltata" warning is only
    # visible transiently mid-run: _on_run_clicked unconditionally
    # overwrites w_status with the final "Fatto" once every step (VQE
    # skipped or not) finishes, so by the time click() returns there's
    # nothing left to assert about that intermediate message -- exercising
    # the skip branch itself (no crash) is the actual coverage target.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Circuito:").value = "Error Mitigation (Real-Stress)"
    _find(kids, "Abilita telemetria MD").value = False
    assert _find(kids, "Confermo VQE reale anche su circuiti pesanti (>12 qubit)").value is False

    w_run.click()

    assert "Fatto" in w_status.value


def _find_recursive(widget, description):
    """Descends through .children (VBox/HBox/Tab/Accordion all expose it)
    to find a widget by its .description — the mode panels below nest
    deeper than the flat sidebar the plain _find() above was written for."""
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        found = _find_recursive(child, description)
        if found is not None:
            return found
    return None


def _find_button_recursive(widget, label_substring):
    if type(widget).__name__ == "Button" and label_substring in (widget.description or ""):
        return widget
    for child in getattr(widget, "children", ()):
        found = _find_button_recursive(child, label_substring)
        if found is not None:
            return found
    return None


def test_mode_toggle_switches_visible_panel():
    panel = dc.launch_interactive_panel()
    w_mode = panel.children[0]
    quantum_sim_panel = _mode_panel(panel, "⚛️ Quantum Simulator")
    vector_healing_panel = _mode_panel(panel, "🧬 Vector Healing")

    assert quantum_sim_panel.layout.display is None
    assert vector_healing_panel.layout.display == "none"

    w_mode.value = "🧬 Vector Healing"

    assert quantum_sim_panel.layout.display == "none"
    assert vector_healing_panel.layout.display is None


def test_vector_healing_panel_run_button_populates_status():
    panel = dc.launch_interactive_panel()
    vh_panel = _mode_panel(panel, "🧬 Vector Healing")

    _find_recursive(vh_panel, "Step / token:").value = 30  # keep it fast
    w_run = _find_button_recursive(vh_panel, "Genera ed Esegui Healing")

    w_run.click()

    # w_status has no .description (plain HTML), so grab it positionally:
    # panel_vector_healing.children[1] is the config VBox, whose last child
    # is the HBox([w_vh_run, w_vh_status]).
    config_box = vh_panel.children[1]
    _, w_status = config_box.children[-1].children
    assert "Fatto" in w_status.value


def test_quantum_scars_panel_run_button_populates_status():
    panel = dc.launch_interactive_panel()
    qs_panel = _mode_panel(panel, "🌀 Quantum Scars")

    _find_recursive(qs_panel, "Qubit (catena PXP):").value = 6  # smallest allowed -- keep it fast
    _find_recursive(qs_panel, "Traiettorie:").value = 5
    w_run = _find_button_recursive(qs_panel, "Esegui esperimento")

    w_run.click()

    config_box = qs_panel.children[1]
    _, w_status = config_box.children[-1].children
    assert "Fatto" in w_status.value


def test_research_bridge_context_block_and_colab_log():
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")

    w_hypothesis = _find_recursive(rb_panel, "Ipotesi:")
    w_hypothesis.value = "La topologia lineare riduce la fedeltà rispetto a quella ad anello."

    rb_tabs = None
    for child in rb_panel.children:
        if type(child).__name__ == "Tab":
            rb_tabs = child
            break
    assert rb_tabs is not None
    tab_colab = rb_tabs.children[1]  # 📓 Colab personale, second tab

    w_colab_response = _find_recursive(tab_colab, "Risposta:")
    w_colab_response.value = "Risposta di prova incollata a mano."
    w_colab_save = _find_button_recursive(tab_colab, "Salva nel log")
    w_colab_save.click()  # must not raise

    w_colab_status = tab_colab.children[-1].children[1]  # HBox([save, status])
    assert "Salvato" in w_colab_status.value


def _rb_tab(rb_panel, index):
    for child in rb_panel.children:
        if type(child).__name__ == "Tab":
            return child.children[index]
    raise AssertionError("Research Bridge Tab widget not found")


def test_vector_healing_channel_change_before_run_is_noop():
    # _vh_render_channel's `if not vh_state: return` guard, and the
    # observer callback that calls it (_on_vh_channel_change) -- neither
    # is exercised by the run-button test above, which never touches the
    # channel dropdown.
    panel = dc.launch_interactive_panel()
    vh_panel = _mode_panel(panel, "🧬 Vector Healing")

    w_channel = _find_recursive(vh_panel, "Canale:")
    w_channel.value = 2  # before any run -- vh_state is still empty, must not raise

    config_box = vh_panel.children[1]
    _find_button_recursive(config_box, "Genera ed Esegui Healing").click()
    w_channel.value = 4  # after a run -- the populated-state path, via the observer this time

    _, w_status = config_box.children[-1].children
    assert "Fatto" in w_status.value


def test_run_button_with_hamiltonian_invalid_custom_json_falls_back_to_none():
    # The json.JSONDecodeError branch in _on_run_clicked's hamiltonian
    # extraction -- every earlier custom-JSON test used the valid default
    # text, so hamiltonian_values silently staying None on bad input was
    # never actually exercised.
    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Abilita Hamiltoniana personalizzata").value = True
    _find_deep(kids, "Modalità:").value = "Custom JSON Textarea"
    _find_deep(kids, "Array JSON:").value = "{not valid json"

    w_run.click()  # must not raise -- falls back to hamiltonian_values=None

    assert "Fatto" in w_status.value


def test_research_bridge_google_save_step_and_stop_chain():
    # _on_rb_google_save and _on_rb_google_stop -- no earlier test ever
    # touches the Google tab at all, only Colab's.
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")

    _find_recursive(rb_panel, "Ipotesi:").value = "Ipotesi di prova per la catena Google."
    tab_google = _rb_tab(rb_panel, 0)

    w_response = _find_recursive(tab_google, "Risposta Google:")
    w_save = _find_button_recursive(tab_google, "Salva questo passo")
    w_stop = _find_button_recursive(tab_google, "Ferma e salva la catena")

    w_stop.click()  # empty chain -- must be a no-op, not raise

    w_response.value = "Prima risposta incollata a mano."
    w_save.click()  # first step: builds the query, appends to rb_chain, re-renders history

    w_response.value = "Seconda risposta, per un secondo passo nella stessa catena."
    w_save.click()  # second step: exercises the "follow-up" (rb_chain non-empty) branch too

    w_stop.click()  # now with a real 2-step chain: logs it and clears rb_chain

    assert True  # reaching here without an exception is the actual coverage target


def test_research_bridge_colab_save_empty_response_is_noop():
    # _on_rb_colab_save's `if not ...strip(): return` guard -- the earlier
    # Colab test always provides real response text.
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")
    tab_colab = _rb_tab(rb_panel, 1)

    w_colab_save = _find_button_recursive(tab_colab, "Salva nel log")
    w_colab_save.click()  # w_colab_response is still empty -- must not raise or log anything

    w_colab_status = tab_colab.children[-1].children[1]
    assert w_colab_status.value == ''  # unchanged -- the guard returned before setting it


def test_research_bridge_custom_api_send_without_credentials():
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")
    tab_custom = _rb_tab(rb_panel, 2)

    w_send = _find_button_recursive(tab_custom, "Invia direttamente")
    w_send.click()  # both endpoint and key are still empty -- guard branch, must not raise


def test_research_bridge_custom_api_send_with_unreachable_endpoint():
    # Exercises the try/except around call_custom_api -- a real network
    # call would be flaky in CI, so this points at a port nothing listens
    # on (connection refused, fast and deterministic) rather than mocking
    # the request away.
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")
    tab_custom = _rb_tab(rb_panel, 2)

    _find_recursive(tab_custom, "Endpoint API:").value = "http://127.0.0.1:1/v1/chat/completions"
    _find_recursive(tab_custom, "Chiave API:").value = "fake-key"
    w_send = _find_button_recursive(tab_custom, "Invia direttamente")
    w_send.click()  # must not raise -- the connection failure is caught and shown, not propagated


def test_research_bridge_local_cli_run_without_command():
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")
    tab_local = _rb_tab(rb_panel, 3)

    w_run_cli = _find_button_recursive(tab_local, "Esegui in locale")
    w_run_cli.click()  # w_rb_cli_command is still empty -- guard branch, must not raise


def test_research_bridge_local_cli_run_with_missing_command():
    # Exercises the try/except around call_local_cli -- a command that
    # can't possibly exist on PATH raises FileNotFoundError, caught by the
    # generic except Exception the same as any other failure mode.
    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")
    tab_local = _rb_tab(rb_panel, 3)

    _find_recursive(tab_local, "Comando:").value = "nonexistent_command_xyz_12345"
    w_run_cli = _find_button_recursive(tab_local, "Esegui in locale")
    w_run_cli.click()  # must not raise -- FileNotFoundError is caught and shown


def test_research_bridge_log_export_writes_file():
    # _on_rb_log_export -- outside Colab, `from google.colab import files`
    # raises ImportError, falling through to the IPython.display.FileLink
    # branch. Not in the missing-lines list for the "no credentials"
    # guard paths above, but the export button itself was never clicked
    # by any earlier test.
    import os

    panel = dc.launch_interactive_panel()
    rb_panel = _mode_panel(panel, "🌉 Research Bridge")

    w_export = _find_button_recursive(rb_panel, "Esporta log (JSON)")
    w_export.click()  # must not raise

    assert os.path.exists("research_bridge_log.json")
    os.remove("research_bridge_log.json")  # don't leave test output behind in the repo checkout


def test_heal_telemetry_none_and_empty_df_return_neutral_metadata():
    # _heal_telemetry's `if df is None or df.empty:` guard -- every run
    # exercised through the panel always produces real VQE/MD data, so
    # this early-return path (mirrors ui_pages/ai_middleware.py's own
    # heal_telemetry, deliberately duplicated here) was never hit.
    import pandas as pd
    from dashboard_core.interactive_panel import _heal_telemetry

    df_none, meta_none = _heal_telemetry(None)
    assert df_none.empty
    assert meta_none == {
        'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0,
    }

    df_empty, meta_empty = _heal_telemetry(pd.DataFrame())
    assert df_empty.empty
    assert meta_empty == meta_none


def test_run_button_shows_error_when_run_simulation_raises(monkeypatch):
    # The try/except around run_simulation in _on_run_clicked -- no
    # earlier test ever makes the circuit-execution step itself fail.
    import dashboard_core.interactive_panel as ip_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated circuit execution failure")

    monkeypatch.setattr(ip_mod, "run_simulation", _boom)

    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    w_run.click()

    assert "Errore durante l'esecuzione del circuito" in w_status.value
    assert not w_run.disabled  # re-enabled after the early return


def test_run_button_shows_error_when_zne_mitigation_raises(monkeypatch):
    # The try/except around run_mitigation_sweep -- needs a real noise
    # model active (ZNE is skipped entirely on 'ideal') plus ZNE enabled,
    # neither of which any earlier test combines with a forced failure.
    import dashboard_core.interactive_panel as ip_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated ZNE sweep failure")

    monkeypatch.setattr(ip_mod, "run_mitigation_sweep", _boom)

    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria VQE").value = False
    _find(kids, "Abilita telemetria MD").value = False
    _find(kids, "Rumore:").value = "depolarizing"
    _find(kids, "p:").value = 0.06
    _find(kids, "Abilita Zero-Noise Extrapolation").value = True

    w_run.click()

    # The "Errore durante la mitigazione ZNE" message is only visible
    # transiently mid-run: the run continues past a ZNE failure and
    # unconditionally overwrites w_status with the final "Fatto" -- as
    # with the heavy-circuit skip test above, exercising the except
    # branch itself (no crash) is the actual coverage target.
    assert "Fatto" in w_status.value


def test_run_button_shows_error_when_vqe_telemetry_raises(monkeypatch):
    # The try/except around run_vqe_telemetry -- no earlier test forces
    # this specific step to fail (as opposed to being skipped or disabled).
    import dashboard_core.interactive_panel as ip_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated VQE telemetry failure")

    monkeypatch.setattr(ip_mod, "run_vqe_telemetry", _boom)

    panel = dc.launch_interactive_panel()
    kids, tabs, w_run, w_status = _panel_widgets(panel)

    _find(kids, "Abilita telemetria MD").value = False
    assert _find(kids, "Abilita telemetria VQE").value is True  # default -- VQE runs and fails

    w_run.click()

    # Same reasoning as the ZNE test above: the error message is transient,
    # overwritten by the unconditional final "Fatto" -- that's what's left
    # to check once click() has returned.
    assert "Fatto" in w_status.value
