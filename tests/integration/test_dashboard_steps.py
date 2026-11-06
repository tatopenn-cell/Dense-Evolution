"""The number of steps of the graphical editor is a control of the Build page.

``circuit_editor.circuit_editor`` is a browser component that ``AppTest``
cannot read, so it is replaced by a function that records its arguments.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    import circuit_editor

    recorded = []

    def fake(n_qubits, n_columns=12):
        recorded.append(n_columns)
        return []

    monkeypatch.setattr(circuit_editor, "circuit_editor", fake)
    return recorded


def test_default_is_twelve_steps(calls):
    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()
    assert app.main.number_input(key="n_steps_builder").value == 12
    assert calls[-1] == 12


def test_chosen_steps_reach_the_editor(calls):
    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()
    app.main.number_input(key="n_steps_builder").set_value(8).run()
    assert calls[-1] == 8
