"""The per-qubit output and the outcomes of a run are next to the circuit."""
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


def test_the_circuit_tab_has_the_per_qubit_and_outcomes_tabs(app):
    labels = [tab.label for tab in app.main.tabs]
    assert "Per qubit" in labels
    assert "Outcomes" in labels


def test_bell_pair_has_probability_one_half_on_each_qubit(app):
    frames = [element.value for element in app.main.dataframe]
    per_qubit = [f for f in frames if list(f.columns) == ["qubit", "P(1)"]]
    assert len(per_qubit) == 1
    assert list(per_qubit[0]["P(1)"]) == pytest.approx([0.5, 0.5], abs=1e-9)
