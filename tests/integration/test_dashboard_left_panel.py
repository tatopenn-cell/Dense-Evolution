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
    sections = list(app.main.radio(key="nav_section").options)
    assert len(sections) == 2
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


def test_entropy_of_a_bell_pair_is_shown_when_its_toggle_is_on(app):
    assert "State analysis" in [e.label for e in app.sidebar.expander]
    app.sidebar.toggle(key="entropy_on").set_value(True).run()
    _run_button(app).click().run()
    app.main.radio(key="nav_section").set_value("Results").run()
    assert "Entropy & mutual information" in [tab.label for tab in app.main.tabs]
    values = {m.label: float(m.value) for m in app.main.metric if m.label.endswith("(nat)")}
    expected = {
        "S(A) (nat)": math.log(2),
        "S(B) (nat)": math.log(2),
        "I(A:B) (nat)": 2 * math.log(2),
    }
    assert values == pytest.approx(expected, abs=1e-3)


def test_entropy_tab_needs_a_circuit(app):
    app.session_state["result"] = None
    app.sidebar.toggle(key="entropy_on").set_value(True).run()
    app.main.radio(key="nav_section").set_value("Results").run()
    assert "Entropy & mutual information" not in [tab.label for tab in app.main.tabs]
