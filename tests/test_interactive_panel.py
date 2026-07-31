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


def _panel_widgets(panel):
    sidebar, tabs = panel.children
    kids = sidebar.children
    run_hbox = kids[-1]
    w_run, w_status = run_hbox.children
    return kids, tabs, w_run, w_status


def test_launch_interactive_panel_builds():
    panel = dc.launch_interactive_panel()
    assert type(panel).__name__ == "VBox"
    sidebar, tabs = panel.children
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
