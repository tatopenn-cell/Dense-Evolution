"""The circuit that ran is drawn on the graphical editor of Build."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"
RUN = "▶ Run"
HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()


def _grid(app):
    return app.session_state["ce_hist"][app.session_state["ce_idx"]]


def _run_qasm(app, body, n):
    qasm = f"{HEADER}qreg q[{n}];\ncreg c[{n}];\n{body}"
    app.sidebar.text_area(key="qasm_text__Custom").set_value(qasm).run()
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()


def test_the_bell_circuit_is_drawn_when_the_page_opens(app):
    grid = _grid(app)
    assert grid[0][0] == "h"
    assert grid[0][1] == "ctrl"
    assert grid[1][1] == "tgt_x"
    assert app.main.number_input(key="n_qubits_builder").value == 2
    assert app.main.number_input(key="n_steps_builder").value == 12


def test_running_a_loaded_circuit_draws_it_on_the_editor(app):
    _run_qasm(app, "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n", 3)
    assert not app.exception
    grid = _grid(app)
    assert len(grid) == 3
    assert grid[0][0] == "h"
    assert (grid[1][1], grid[2][2]) == ("tgt_x", "tgt_x")
    assert app.main.number_input(key="n_qubits_builder").value == 3


def test_gates_the_grid_cannot_draw_are_reported(app):
    _run_qasm(app, "h q[0];\nrx(0.3) q[0];\ncx q[0],q[1];\n", 2)
    assert "1 gate(s) cannot be drawn" in app.session_state["editor_note"]
    assert any("cannot be drawn" in c.value for c in app.main.caption)


def test_gates_placed_by_hand_are_not_overwritten_by_a_run(app):
    grid = [list(row) for row in _grid(app)]
    grid[0][5] = "x"
    app.session_state["ce_hist"] = [grid]
    app.session_state["ce_idx"] = 0
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()
    assert _grid(app)[0][5] == "x"


def test_grid_from_ops_packs_single_gates_and_gives_pairs_their_own_column(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    sys.modules.pop("circuit_editor", None)
    import circuit_editor as ce

    grid, skipped = ce.grid_from_ops(
        [("h", 0), ("h", 1), ("cx", 0, 1), ("swap", 0, 1), ("ccx", 0, 1, 2)], 3
    )
    assert skipped == 1
    assert [row[0] for row in grid] == ["h", "h", None]
    assert (grid[0][1], grid[1][1]) == ("ctrl", "tgt_x")
    assert (grid[0][2], grid[1][2]) == ("swap", "swap")
    assert len(grid[0]) == 12
