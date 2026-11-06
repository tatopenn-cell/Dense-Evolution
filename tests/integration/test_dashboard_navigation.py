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
    assert [tab.label for tab in app.main.tabs] == []
    assert app.main.number_input(key="n_steps_builder").value == 12


def test_results_tabs_are_fixed_and_wrap_onto_the_next_row(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    styles = [m.value for m in app.main.markdown if "results_tabs" in m.value]
    assert len(styles) == 1
    assert "flex-wrap:wrap" in styles[0]


def test_results_fixed_tabs_group_the_related_views(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    labels = [tab.label for tab in app.main.tabs]
    top = [label for label in labels if label not in ("Probabilities", "Per qubit", "Outcomes")]
    assert top == ["Circuit", "Statevector", "Probabilities & Q-sphere", "Qubits"]


def test_theme_selector_is_at_the_top_right_with_four_themes(app):
    box = app.main.selectbox(key="theme")
    assert list(box.options) == ["Light", "Dark", "Pink fantasy", "Dark fantasy"]
    assert box.value == "Light"
    with pytest.raises(KeyError):
        app.sidebar.selectbox(key="theme")


def test_each_theme_changes_the_page_colors_without_breaking_it(app):
    colors = {}
    for name in ("Dark", "Pink fantasy", "Dark fantasy"):
        app.main.selectbox(key="theme").set_value(name).run()
        assert not app.exception, name
        styles = [m.value for m in app.main.markdown if "--primary-color" in m.value]
        assert len(styles) == 1, name
        assert "@" not in styles[0], name
        colors[name] = styles[0]
    assert len(set(colors.values())) == 3
    assert "invert" in colors["Dark"] and "invert" in colors["Dark fantasy"]
    assert "invert" not in colors["Pink fantasy"]
