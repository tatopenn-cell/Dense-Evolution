"""The section menu and the Advanced switch of the dashboard sit in the main area."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
BASE = ["Build", "Results", "Chemistry", "Noise", "System"]


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


def test_toggle_is_in_the_main_area_and_not_in_the_sidebar(app):
    assert app.main.toggle(key="advanced_mode").value is False
    with pytest.raises(KeyError):
        app.sidebar.toggle(key="advanced_mode")


def test_advanced_mode_adds_the_eight_advanced_sections(app):
    assert len(app.main.radio(key="nav_section").options) == 5
    app.main.toggle(key="advanced_mode").set_value(True).run()
    assert len(app.main.radio(key="nav_section").options) == 13


def test_choosing_a_section_shows_its_page(app):
    app.main.radio(key="nav_section").set_value("Results").run()
    assert "Q-sphere" in [tab.label for tab in app.main.tabs]
    app.main.toggle(key="advanced_mode").set_value(True).run()
    app.main.radio(key="nav_section").set_value("All functions").run()
    assert "All engine functions" in [h.value for h in app.main.header]


def test_switching_advanced_mode_off_returns_to_build(app):
    app.main.toggle(key="advanced_mode").set_value(True).run()
    app.main.radio(key="nav_section").set_value("QEC").run()
    assert app.main.radio(key="nav_section").value == "QEC"
    app.main.toggle(key="advanced_mode").set_value(False).run()
    assert app.main.radio(key="nav_section").value == "Build"
