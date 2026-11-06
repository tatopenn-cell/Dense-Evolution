"""The left panel of the dashboard is the same in every section."""
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
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()


def _run_button(app):
    return [button for button in app.sidebar.button if button.label == RUN][0]


def test_every_section_has_the_left_panel(app):
    app.main.toggle(key="advanced_mode").set_value(True).run()
    sections = list(app.main.radio(key="nav_section").options)
    assert len(sections) == 11
    for name in sections:
        app.main.radio(key="nav_section").set_value(name).run()
        assert not app.exception, name
        assert RUN in [button.label for button in app.sidebar.button], name
        assert app.sidebar.text_area(key="qasm_text__Custom").value, name


def test_running_from_results_stays_in_results(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    _run_button(app).click().run()
    assert not app.exception
    assert app.main.radio(key="nav_section").value == "Results"
    assert app.session_state["result"] is not None


def test_entropy_of_a_bell_pair_is_computed_from_the_left_panel(app):
    assert "Entropy & magic" in [e.label for e in app.sidebar.expander]
    app.main.radio(key="nav_section").set_value("Results").run()
    app.sidebar.button(key="entropy_run").click().run()
    s_a, s_b, i_ab = app.session_state["entropy_result"]
    expected = (math.log(2), math.log(2), 2 * math.log(2))
    assert (s_a, s_b, i_ab) == pytest.approx(expected, abs=1e-6)


def test_entropy_without_a_circuit_reports_the_reason(app):
    app.session_state["result"] = None
    app.sidebar.button(key="entropy_run").click().run()
    assert app.session_state["entropy_error"] == "run a circuit first"
