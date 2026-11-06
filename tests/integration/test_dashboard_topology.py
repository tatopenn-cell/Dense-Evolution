"""The Topology tab of the Build page loads an entangling layer into the editor."""
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


def test_circular_on_five_qubits_loads_the_wrap_around_edge(app):
    assert "Topology" in [tab.label for tab in app.main.tabs]
    app.main.selectbox(key="topo_pattern").set_value("circular").run()
    app.main.number_input(key="topo_qubits").set_value(5).run()
    app.main.button(key="topo_load").click().run()
    assert "cx q[4],q[0];" in app.sidebar.text_area(key="qasm_text__Custom").value


def test_hub_input_is_only_for_the_star_pattern(app):
    with pytest.raises(KeyError):
        app.main.number_input(key="topo_hub")
    app.main.selectbox(key="topo_pattern").set_value("star").run()
    assert app.main.number_input(key="topo_hub").value == 0
