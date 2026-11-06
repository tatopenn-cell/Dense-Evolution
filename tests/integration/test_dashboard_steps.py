"""The number of steps of the graphical editor is a control of the Build page."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()


def _columns(app):
    grid = app.session_state["ce_hist"][app.session_state["ce_idx"]]
    return {len(row) for row in grid}


def test_default_grid_has_twelve_steps(app):
    assert app.main.number_input(key="n_steps_builder").value == 12
    assert _columns(app) == {12}


def test_chosen_steps_resize_the_grid_of_the_editor(app):
    app.main.number_input(key="n_steps_builder").set_value(8).run()
    assert _columns(app) == {8}
    app.main.number_input(key="n_steps_builder").set_value(16).run()
    assert _columns(app) == {16}
