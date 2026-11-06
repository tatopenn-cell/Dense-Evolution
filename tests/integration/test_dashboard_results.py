"""Build holds only the editor; the circuit with its three views is in Results."""
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


def test_the_circuit_tab_of_results_keeps_its_three_views(app):
    _results(app)
    labels = [tab.label for tab in app.main.tabs]
    assert labels[0] == "Circuit"
    for inner in ("Probabilities", "Per qubit", "Outcomes"):
        assert inner in labels


def test_bell_pair_has_probability_one_half_on_each_qubit(app):
    _results(app)
    frames = [element.value for element in app.main.dataframe]
    summaries = [f for f in frames if {"qubit", "P(1)"} <= set(f.columns)]
    assert len(summaries) == 2
    for frame in summaries:
        assert list(frame["P(1)"]) == pytest.approx([0.5, 0.5], abs=1e-9)
