"""docs/assets/composer/circuit_tools.js, run with Node, against the functions of the dashboard.

The page has the topologies, the input values of the qubits and the tables of a run; each one is the
translation of a function of tools/dashboard/circuit_tools.py or of dense_evolution/circuits/topology.py.
The references are those functions, which the page file does not compute.
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from dashboard import circuit_tools as ct

from dense_evolution.circuits.topology import VALID_PATTERNS, entangling_layer

TOOLS = Path(__file__).resolve().parents[2] / "docs" / "assets" / "composer" / "circuit_tools.js"
NL = chr(10)

_SCRIPT = (
    "global.window = {};"
    "eval(require('fs').readFileSync(process.argv[1], 'utf8'));"
    "const a = JSON.parse(process.argv[2]);"
    "const T = window.DECircuitTools;"
    "let r;"
    "try { r = {ok: T[a.fn](...a.args)}; } catch (e) { r = {error: String(e.message)}; }"
    "process.stdout.write(JSON.stringify(r));"
)


def _node():
    exe = shutil.which("node")
    if exe:
        return exe
    try:
        from playwright._impl._driver import compute_driver_executable

        return str(compute_driver_executable()[0])
    except (ImportError, AttributeError):
        return None


pytestmark = pytest.mark.skipif(_node() is None, reason="Node is not available")


def _js(fn: str, *args):
    done = subprocess.run(
        [_node(), "-e", _SCRIPT, str(TOOLS), json.dumps({"fn": fn, "args": list(args)})],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize("pattern", VALID_PATTERNS)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_the_edges_are_the_ones_of_entangling_layer(pattern, n, reverse):
    hubs = range(n) if pattern == "star" else [0]
    for hub in hubs:
        want = [[a, b] for _, a, b in entangling_layer(n, pattern, "cx", reverse, hub)]
        assert _js("topologyEdges", n, pattern, reverse, hub)["ok"] == want


@pytest.mark.parametrize("pattern", VALID_PATTERNS)
@pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
def test_the_qasm_of_a_topology_is_the_one_of_topology_qasm(pattern, gate):
    for n, reverse, hadamards in itertools.product([2, 5], [False, True], [False, True]):
        want = ct.topology_qasm(n, pattern, gate, reverse, 1 if pattern == "star" else 0, hadamards)
        got = _js("topologyQasm", n, pattern, gate, reverse, 1 if pattern == "star" else 0, hadamards)
        assert got["ok"] == want


@pytest.mark.parametrize(
    "args",
    [(1, "linear", "cx", False, 0, True), (4, "wheel", "cx", False, 0, True), (4, "star", "cx", False, 4, True), (4, "linear", "swap", False, 0, True)],
)
def test_the_page_refuses_what_topology_qasm_refuses(args):
    with pytest.raises(ValueError):
        ct.topology_qasm(*args)
    assert "error" in _js("topologyQasm", *args)


BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'


@pytest.mark.parametrize("ones", [[], [1], [1, 0], [5], [0, 0, 1]])
def test_the_input_values_are_the_ones_of_with_input_values(ones):
    assert _js("withInputValues", BELL, ones)["ok"] == ct.with_input_values(BELL, ones)


def test_a_negative_index_and_a_text_with_no_qreg_are_refused():
    for text, ones in ((BELL, [-1]), ("OPENQASM 2.0;" + NL + "h q[0];" + NL, [0])):
        with pytest.raises(ValueError):
            ct.with_input_values(text, ones)
        assert "error" in _js("withInputValues", text, ones)


@pytest.mark.parametrize("text", [BELL, "OPENQASM 2.0;" + NL, "qreg a[7];" + NL + "qreg b[2];" + NL])
def test_the_size_of_the_register_is_the_one_of_qubits_in_qasm(text):
    assert _js("qubitsInQasm", text)["ok"] == ct.qubits_in_qasm(text)


@pytest.mark.parametrize("counts", [{}, {"00": 503, "11": 497}, {"01": 5, "10": 5, "00": 90}, {"1": 0}])
def test_the_outcome_rows_are_the_ones_of_outcome_rows(counts):
    want = ct.outcome_rows(counts)
    got = _js("outcomeRows", counts)["ok"]
    assert [r["outcome"] for r in got] == [r["outcome"] for r in want]
    assert [r["shots"] for r in got] == [r["shots"] for r in want]
    assert [r["probability"] for r in got] == pytest.approx([r["probability"] for r in want], abs=1e-12)


def test_p1_of_a_bell_pair_is_one_half_for_both_qubits():
    rows = [{"state": "00", "abs": 2**-0.5}, {"state": "11", "abs": 2**-0.5}]
    assert _js("qubitP1", rows, 2)["ok"] == pytest.approx([0.5, 0.5], abs=1e-12)
    ghost = [{"state": "01", "abs": 1.0}]
    assert _js("qubitP1", ghost, 2)["ok"] == [1.0, 0.0]
