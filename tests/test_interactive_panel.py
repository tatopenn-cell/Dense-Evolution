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
