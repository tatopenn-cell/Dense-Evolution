"""Chemistry, Dynamics, Wormhole, Error correction and Condensed matter windows."""
import math
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
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=300).run()


def _run(app):
    app.sidebar.button(key="run_button").click().run()


def _results_tabs(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    return [tab.label for tab in app.main.tabs]


def test_windows_exist_and_are_off_by_default(app):
    labels = [e.label for e in app.sidebar.expander]
    for name in ("Chemistry", "Protocols", "Condensed matter"):
        assert name in labels
    for key in ("chem_on", "dyn_on", "wh_on", "qec_on", "cm_on", "hub_on"):
        assert app.sidebar.toggle(key=key).value is False


def test_chemistry_on_loads_the_ansatz_and_off_restores_the_circuit(app):
    before = app.sidebar.text_area(key="qasm_text__Custom").value
    app.sidebar.slider(key="vqe_maxiter").set_value(3).run()
    app.sidebar.toggle(key="chem_on").set_value(True).run()
    _run(app)
    assert app.session_state["vqe_error"] is None, app.session_state["vqe_error"]
    assert app.session_state["vqe_result"] is not None
    ansatz = app.sidebar.text_area(key="qasm_text__Custom").value
    assert ansatz == app.session_state["vqe_result"]["qasm"] != before
    assert app.session_state["result"] is not None
    assert "Chemistry" in _results_tabs(app)
    app.sidebar.toggle(key="chem_on").set_value(False).run()
    assert app.sidebar.text_area(key="qasm_text__Custom").value == before


def test_wormhole_runs_with_the_run_button(app):
    app.sidebar.toggle(key="wh_on").set_value(True).run()
    _run(app)
    assert app.session_state["wh_error"] is None, app.session_state["wh_error"]
    i_plus, i_minus, mu = app.session_state["wh_result"]
    assert mu == 12.0
    assert "Wormhole" in _results_tabs(app)


def test_error_correction_finds_the_single_qubit_correction(app):
    app.sidebar.toggle(key="qec_on").set_value(True).run()
    _run(app)
    out = app.session_state["qec_result"]
    assert app.session_state["qec_error"] is None, app.session_state["qec_error"]
    assert len(out["stabilizers"]) == 4
    assert out["decoded"] is not None
    assert "Error correction" in _results_tabs(app)


def test_error_correction_reports_a_wrong_pauli_error(app):
    app.sidebar.toggle(key="qec_on").set_value(True).run()
    app.sidebar.text_input(key="qec_pauli").set_value("XX").run()
    _run(app)
    assert "needs exactly 5 characters" in app.session_state["qec_error"]


def test_condensed_matter_gap_and_hubbard(app):
    app.sidebar.toggle(key="cm_on").set_value(True).run()
    app.sidebar.toggle(key="hub_on").set_value(True).run()
    app.sidebar.slider(key="hub_n_sites").set_value(2).run()
    _run(app)
    assert app.session_state["cm_error"] is None, app.session_state["cm_error"]
    assert app.session_state["cm_result"]["gap_gamma"] > 0
    n_qubits, energy = app.session_state["hub_result"]
    assert n_qubits == 4
    assert math.isfinite(energy)
    assert "Condensed matter" in _results_tabs(app)
