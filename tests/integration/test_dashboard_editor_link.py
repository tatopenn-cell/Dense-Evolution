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
    app.sidebar.text_area(key="qasm_text__Custom").set_value(qasm)
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()


def test_the_bell_circuit_is_drawn_when_the_page_opens(app):
    grid = _grid(app)
    assert grid[0][0] == "h"
    assert grid[0][1] == "ctrl"
    assert grid[1][1] == "x"
    assert app.main.number_input(key="n_qubits_builder").value == 2
    assert app.main.number_input(key="n_steps_builder").value == 12


def test_running_a_loaded_circuit_draws_it_on_the_editor(app):
    _run_qasm(app, "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n", 3)
    assert not app.exception
    grid = _grid(app)
    assert len(grid) == 3
    assert grid[0][0] == "h"
    assert (grid[1][1], grid[2][2]) == ("x", "x")
    assert app.main.number_input(key="n_qubits_builder").value == 3


def test_rotations_and_controlled_phases_keep_their_angle(app):
    _run_qasm(app, "rx(0.3) q[0];\ncp(0.5) q[0],q[1];\nccx q[0],q[1],q[2];\n", 3)
    grid = _grid(app)
    assert grid[0][0] == "rx:0.3"
    assert (grid[0][1], grid[1][1]) == ("ctrl", "p:0.5")
    assert (grid[0][2], grid[1][2], grid[2][2]) == ("ctrl", "ctrl", "x")
    assert app.session_state["editor_note"] is None


def test_the_qft_preset_is_drawn_completely(app):
    name = [o for o in app.sidebar.selectbox(key="preset_select").options if "Fourier" in o][0]
    app.sidebar.selectbox(key="preset_select").set_value(name).run()
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()
    assert not app.exception
    assert app.session_state["editor_note"] is None
    cells = [cell for row in _grid(app) for cell in row if cell]
    assert any(cell.startswith("p:") for cell in cells)


def test_a_circuit_beyond_the_editor_size_is_reported(app):
    _run_qasm(app, "h q[0];\n", 9)
    assert "up to 8 qubits" in app.session_state["editor_note"]
    assert any("up to 8 qubits" in c.value for c in app.main.caption)


def test_gates_placed_by_hand_are_not_overwritten_by_a_run(app):
    grid = [list(row) for row in _grid(app)]
    grid[0][5] = "x"
    app.session_state["ce_hist"] = [grid]
    app.session_state["ce_idx"] = 0
    app.session_state["ce_version"] += 1
    app.session_state["ce_start_version"] = None
    [b for b in app.sidebar.button if b.label == RUN][0].click().run()
    assert _grid(app)[0][5] == "x"


def test_grid_from_ops_packs_single_gates_and_gives_pairs_their_own_column(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    sys.modules.pop("circuit_editor", None)
    import circuit_editor as ce

    grid, skipped = ce.grid_from_ops(
        [("h", 0), ("h", 1), ("cx", 0, 1), ("swap", 0, 1), ("id", 0)], 3
    )
    assert skipped == 1
    assert [row[0] for row in grid] == ["h", "h", None]
    assert (grid[0][1], grid[1][1]) == ("ctrl", "x")
    assert (grid[0][2], grid[1][2]) == ("swap", "swap")
    assert len(grid[0]) == 12


def test_the_editor_ops_round_trip_through_the_grid(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    sys.modules.pop("circuit_editor", None)
    import circuit_editor as ce

    ops = [("h", 0), ("rx", 1, 0.3), ("cp", 0, 1, 0.5), ("ccx", 0, 1, 2), ("crz", 2, 0, 1.0)]
    grid, _ = ce.grid_from_ops(ops, 3)
    assert ce.ops_to_tuples(ce.grid_to_ops(grid)) == ops
    assert ce.decode_grid(ce.encode_grid(grid), 3, 12) == grid
    assert ce.grid_to_ops([["rx"]]) == [{"gate": "rx", "qubits": [0], "angle": 1.5707963267948966}]


def test_a_control_dot_on_any_gate_and_links_saved_with_separate_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    sys.modules.pop("circuit_editor", None)
    import circuit_editor as ce

    assert ce.grid_to_ops([["ctrl"], ["x"]]) == [{"gate": "cx", "qubits": [0, 1]}]
    assert ce.grid_to_ops([["ctrl"], ["tgt_x"]]) == [{"gate": "cx", "qubits": [0, 1]}]
    assert ce.grid_to_ops([["z"], ["ctrl"]]) == [{"gate": "cz", "qubits": [1, 0]}]
    assert ce.grid_to_ops([["ctrl"], ["h"]]) == []
    ids = [chip["id"] for chip in ce._palette()]
    assert not any(i.startswith("tgt_") for i in ids)


def test_an_anti_control_acts_on_zero_with_an_x_before_and_after(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    sys.modules.pop("circuit_editor", None)
    import circuit_editor as ce

    assert ce.grid_to_ops([["actrl"], ["x"]]) == [
        {"gate": "x", "qubits": [0]},
        {"gate": "cx", "qubits": [0, 1]},
        {"gate": "x", "qubits": [0]},
    ]
    assert ce.grid_to_ops([["actrl"], ["ctrl"], ["x"]])[1] == {"gate": "ccx", "qubits": [0, 1, 2]}
    assert "actrl" in [chip["id"] for chip in ce._palette()]
