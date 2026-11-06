"""Analysis windows: left-panel settings with On/Off, one Run, one Results tab each."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
RUN = "▶ Run"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()


def _run(app):
    [button for button in app.sidebar.button if button.label == RUN][0].click().run()


def _results_tabs(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    return [tab.label for tab in app.main.tabs]


def test_windows_are_in_the_left_panel_and_off_by_default(app):
    labels = [e.label for e in app.sidebar.expander]
    for name in ("Noise", "State analysis"):
        assert name in labels
    for key in ("zne_on", "dm_on", "magic_on", "bc_on"):
        assert app.sidebar.toggle(key=key).value is False
    assert "ZNE mitigation" not in _results_tabs(app)


def test_old_sections_are_gone_from_the_top_menu(app):
    options = list(app.main.radio(key="nav_section").options)
    for gone in (
        "Magic & Divergences",
        "System",
        "All functions",
        "Noise",
        "Import Circuit",
        "Large Systems",
        "Chemistry",
        "Dynamics",
        "Wormhole",
        "QEC",
    ):
        assert gone not in options
    assert options == ["Build", "Results"]


def test_magic_window_adds_a_results_tab_computed_by_run(app):
    app.sidebar.toggle(key="magic_on").set_value(True).run()
    _run(app)
    assert not app.exception
    assert "Magic" in _results_tabs(app)
    values = [m.value for m in app.main.metric]
    assert any(v.startswith("1.") or v.startswith("0.") for v in values)


def test_density_channel_runs_with_the_main_circuit(app):
    app.sidebar.toggle(key="dm_on").set_value(True).run()
    _run(app)
    assert app.session_state["dm_error"] is None
    assert app.session_state["dm_result"] is not None
    assert "Density-matrix channel" in _results_tabs(app)


def test_zne_runs_with_the_main_circuit(app):
    app.sidebar.toggle(key="zne_on").set_value(True).run()
    app.sidebar.slider(key="zne_n_trials").set_value(20).run()
    _run(app)
    assert app.session_state["zne_error"] is None
    assert app.session_state["zne_result"] is not None


def test_bond_convergence_runs_with_the_main_circuit(app):
    app.sidebar.toggle(key="bc_on").set_value(True).run()
    app.sidebar.text_input(key="bc_bonds").set_value("2,4,8").run()
    _run(app)
    assert app.session_state["bond_convergence_error"] is None, app.session_state[
        "bond_convergence_error"
    ]
    assert app.session_state["bond_convergence_result"] is not None


def test_mps_beyond_the_dense_limit_lists_the_top_states(app):
    ghz = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[30];\nh q[0];\n' + "".join(
        f"cx q[{i}],q[{i + 1}];\n" for i in range(29)
    )
    app.sidebar.selectbox(key="preset_select").set_value("Custom").run()
    app.sidebar.text_area(key="qasm_text__Custom").set_value(ghz).run()
    app.sidebar.selectbox(key="run_backend").set_value("mps").run()
    _run(app)
    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["large_mps_result"].n_qubits == 30


def test_cosmic_ray_burst_runs_the_main_circuit_under_the_burst_profile(app):
    app.sidebar.toggle(key="burst_on").set_value(True).run()
    _run(app)
    assert app.session_state["burst_error"] is None, app.session_state["burst_error"]
    rows = app.session_state["burst_result"]["rows"]
    assert len(rows) == 4
    assert all(0.0 <= row["fidelity"] <= 1.0 for row in rows)
    assert rows[-1]["fidelity"] >= rows[2]["fidelity"]
    assert "Cosmic-ray burst" in _results_tabs(app)


def test_oscillating_noise_uses_the_zne_strength(app):
    app.sidebar.toggle(key="osc_on").set_value(True).run()
    _run(app)
    out = app.session_state["osc_result"]
    assert out.base_p == app.sidebar.slider(key="zne_noise_p").value
    assert out.p_eff[0] == pytest.approx(out.base_p)
    assert "Oscillating noise" in _results_tabs(app)


def test_hardware_calibration_lists_the_error_rates(app):
    pytest.importorskip("qiskit_ibm_runtime")
    app.sidebar.toggle(key="cal_on").set_value(True).run()
    _run(app)
    assert app.session_state["cal_error"] is None, app.session_state["cal_error"]
    assert len(app.session_state["cal_result"]) > 0


def test_function_explorer_runs_the_chosen_function_on_the_main_circuit(app):
    assert "Function explorer" in [e.label for e in app.sidebar.expander]
    assert "System" in [e.label for e in app.sidebar.expander]
    app.sidebar.toggle(key="fn_on").set_value(True).run()
    _run(app)
    assert not app.exception
    assert app.session_state["fn_error"] is None, app.session_state["fn_error"]
    assert app.session_state["fn_result"]["name"] == app.sidebar.selectbox(key="ex_fn").value
    assert "Function" in _results_tabs(app)


def test_step_by_step_is_a_window_and_the_gate_is_chosen_on_the_left(app):
    assert "Step by step" not in _results_tabs(app)
    app.sidebar.toggle(key="steps_on").set_value(True).run()
    app.sidebar.number_input(key="sota_step").set_value(2).run()
    assert "Step by step" in _results_tabs(app)
    assert any("Gate 2 of" in c.value for c in app.main.caption)


def test_exact_probabilities_replace_the_sampled_histogram(app):
    app.sidebar.toggle(key="prob_exact").set_value(True).run()
    app.main.radio(key="nav_section").set_value("Results").run()
    assert any("Exact probabilities" in c.value for c in app.main.caption)


def test_the_results_circuit_tab_lists_the_shots_one_by_one(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    assert "Measurements" in [tab.label for tab in app.main.tabs]
    frames = [e.value for e in app.main.dataframe]
    shots = [f for f in frames if list(f.columns) == ["measurement", "input", "output"]]
    assert len(shots) == 1 and len(shots[0]) == 100
    assert set(shots[0]["output"]) <= {"00", "11"}


def test_export_window_is_in_the_left_panel(app):
    assert "Export" in [e.label for e in app.sidebar.expander]
