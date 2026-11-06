"""The qubits chosen in the Build page start in |1> when the circuit runs."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
NO_GATE = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nmeasure q -> c;\n'


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()
    at.sidebar.text_area(key="qasm_text__Custom").set_value(NO_GATE).run()
    return at


def _run(at):
    run = [b for b in at.sidebar.button if b.label == "▶ Run"][0]
    run.click().run()
    return at.session_state["result"].counts


def test_qubit_one_chosen_gives_the_outcome_10(app):
    app.main.multiselect(key="input_ones").set_value([1]).run()
    assert set(_run(app)) == {"10"}


def test_no_qubit_chosen_gives_the_outcome_00(app):
    assert set(_run(app)) == {"00"}
