"""Build holds only the editor; the circuit and its outcomes are in Results."""
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


def _results(app):
    app.main.radio(key="nav_section").set_value("Results").run()


def test_build_has_only_the_graphical_editor(app):
    assert [tab.label for tab in app.main.tabs] == []
    assert app.main.number_input(key="n_qubits_builder").value == 3


def test_the_circuit_tab_of_results_shows_the_outcomes(app):
    _results(app)
    assert "Circuit" in [tab.label for tab in app.main.tabs]
    frames = [element.value for element in app.main.dataframe]
    assert any(list(f.columns)[:1] == ["outcome"] or "outcome" in f.columns for f in frames)


def test_bell_pair_has_probability_one_half_on_each_qubit(app):
    _results(app)
    frames = [element.value for element in app.main.dataframe]
    summary = [f for f in frames if {"qubit", "P(1)"} <= set(f.columns)]
    assert len(summary) == 1
    assert list(summary[0]["P(1)"]) == pytest.approx([0.5, 0.5], abs=1e-9)
