"""The fixed Qubits window at the top of the left panel."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
RUN = "▶ Run"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=300).run()


def _run(app):
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()


def _preset(app, name):
    app.sidebar.selectbox(key="preset_select").set_value(name).run()


def _steps(app):
    return [int(o.split()[0]) for o in app.sidebar.select_slider(key="qubits_select").options]


def test_the_window_is_at_the_top_and_starts_from_the_circuit_size(app):
    slider = app.sidebar.select_slider(key="qubits_select")
    assert slider.value == 2
    assert slider.disabled is False
    assert _steps(app)[0] == 2
    assert _steps(app)[-1] == 1500
    assert list(slider.options)[-1] == "1500 · MPS"


def test_a_custom_circuit_grows_by_its_register_and_runs(app):
    app.sidebar.select_slider(key="qubits_select").set_value(4).run()
    assert "qreg q[4]" in app.sidebar.text_area(key="qasm_text__Custom").value
    _run(app)
    assert app.session_state["result"].n_qubits == 4


def test_the_qubits_the_gates_use_cannot_be_removed(app):
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[5];\nh q[0];\ncx q[0],q[3];\n'
    app.sidebar.text_area(key="qasm_text__Custom").set_value(qasm).run()
    _run(app)
    assert _steps(app)[0] == 4


def test_a_ghz_preset_scales_and_stays_a_ghz_state(app):
    _preset(app, "GHZ state (3 qubit)")
    assert app.sidebar.select_slider(key="qubits_select").value == 3
    app.sidebar.select_slider(key="qubits_select").set_value(6).run()
    _run(app)
    result = app.session_state["result"]
    assert result.n_qubits == 6
    assert set(result.counts) <= {"000000", "111111"}


def test_bell_goes_up_and_runs_on_mps_beyond_the_dense_limit(app):
    _preset(app, "Bell state (2 qubit)")
    app.sidebar.select_slider(key="qubits_select").set_value(40).run()
    _run(app)
    assert not app.exception
    large = app.session_state["large_mps_result"]
    assert large.n_qubits == 40
    assert app.session_state["result"] is None


def test_a_preset_with_a_fixed_size_is_locked(app):
    _preset(app, "Grover search (3 qubit, target |111>)")
    slider = app.sidebar.select_slider(key="qubits_select")
    assert slider.disabled is True
    assert _steps(app) == [3]


def test_the_dense_sizes_are_those_the_memory_guard_accepts(app):
    steps = _steps(app)
    dense = [n for n in steps if "MPS" not in str(n)]
    labels = list(app.sidebar.select_slider(key="qubits_select").options)
    cap = max(int(o.split()[0]) for o in labels if "MPS" not in o)
    assert cap >= 16
    assert all("MPS" in o for o in labels if int(o.split()[0]) > cap)
    assert dense
