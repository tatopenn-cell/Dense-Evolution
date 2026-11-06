"""The page menu of the dashboard sits in the main area; the windows are in the left panel."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
BASE = ["Build", "Results"]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()


def test_radio_is_in_the_main_area_and_not_in_the_sidebar(app):
    assert app.main.radio(key="nav_section").options == BASE
    with pytest.raises(KeyError):
        app.sidebar.radio(key="nav_section")


def test_there_is_no_advanced_switch(app):
    with pytest.raises(KeyError):
        app.main.toggle(key="advanced_mode")
    assert "Optimize geometry (analytic dE/dR)" in [t.label for t in app.sidebar.toggle]


def test_choosing_a_page_shows_it(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    assert "Probabilities & Q-sphere" in [tab.label for tab in app.main.tabs]
    app.main.radio(key="nav_section").set_value("Build").run()
    assert "Graphical Editor" in [tab.label for tab in app.main.tabs]


def test_results_tabs_are_fixed_and_wrap_onto_the_next_row(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    styles = [m.value for m in app.main.markdown if "results_tabs" in m.value]
    assert len(styles) == 1
    assert "flex-wrap:wrap" in styles[0]


def test_results_fixed_tabs_group_the_related_views(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    assert [tab.label for tab in app.main.tabs] == [
        "Statevector",
        "Probabilities & Q-sphere",
        "Qubits",
    ]
