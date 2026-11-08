"""The translation of a Quirk circuit to OpenQASM keeps its meaning.

``docs/assets/composer/quirk_bridge.js`` is run with Node. The OpenQASM it writes is run on
the NumPy MPS simulator and compared with the state computed here from the definition of the
gate: a power of a Pauli gate is its principal power, found from the eigenvectors of the Pauli
matrix, so a phase error between the branches of a control shows as a fidelity below 1.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dense_evolution.backends import mps_numpy as mn
from dense_evolution.backends.qasm_subset import parse_qasm_subset

BRIDGE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "composer" / "quirk_bridge.js"

_SCRIPT = (
    "global.window = {};"
    "eval(require('fs').readFileSync(process.argv[1], 'utf8'));"
    "process.stdout.write(JSON.stringify("
    "window.DEQuirkBridge.quirkToQasm(JSON.parse(process.argv[2]))));"
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


pytestmark = pytest.mark.skipif(_node() is None, reason="Node is needed to run quirk_bridge.js")

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / math.sqrt(2)
T = np.diag([1, np.exp(1j * math.pi / 4)])
PAULI = {"X": X, "Y": Y, "Z": Z}
EXPONENTS = {"": 1, "^½": 0.5, "^-½": -0.5, "^¼": 0.25, "^-¼": -0.25, "^⅓": 1 / 3, "^-⅓": -1 / 3}


def _translate(circuit: dict) -> dict:
    done = subprocess.run(
        [_node(), "-e", _SCRIPT, str(BRIDGE), json.dumps(circuit)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def _run(qasm: str) -> np.ndarray:
    n, ops = parse_qasm_subset(qasm)
    sim = mn.MPSSimulatorNumpy(n, dtype=np.complex128)
    sim.run_circuit(ops)
    return sim.contract_to_statevector()


def _principal_power(pauli: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(pauli)
    return (v * np.exp(1j * math.pi * t * (1 - w) / 2)) @ v.conj().T


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)))


def _single(gate: np.ndarray, wire: int, n: int) -> np.ndarray:
    mats = [np.eye(2, dtype=complex)] * n
    mats[wire] = gate
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _controlled(gate: np.ndarray, control: int, target: int, n: int, on: int = 1) -> np.ndarray:
    keep = np.diag([1 - on, on]).astype(complex)
    other = np.diag([on, 1 - on]).astype(complex)
    return _single(keep, control, n) @ _single(gate, target, n) + _single(other, control, n)


def _plus_state(n: int, wires) -> np.ndarray:
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    for w in wires:
        state = _single(H, w, n) @ state
    return state


PREP = [["H"], ["Z^¼"]]


@pytest.mark.parametrize("suffix", list(EXPONENTS))
@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_a_power_of_a_pauli_gate_alone_and_under_one_control(axis, suffix):
    t = EXPONENTS[suffix]
    gate = _principal_power(PAULI[axis], t)
    ident = axis + suffix

    alone = _translate({"cols": [["H"], ["Z^¼"], [ident]]})
    expected = _single(gate, 0, 1) @ _single(T, 0, 1) @ _single(H, 0, 1)
    assert not alone["skipped"]
    assert _fidelity(_run(alone["qasm"]), expected[:, 0]) > 1 - 1e-9

    for mark, on in (("•", 1), ("◦", 0)):
        cols = [[1, "H"], [1, "Z^¼"], [mark, ident]]
        out = _translate({"cols": cols, "init": ["+"]})
        prep = _single(T, 1, 2) @ _single(H, 1, 2)
        start = prep @ _plus_state(2, [0])
        want = _controlled(gate, 0, 1, 2, on) @ start
        assert not out["skipped"], out
        assert _fidelity(_run(out["qasm"]), want) > 1 - 1e-9


def test_a_toffoli_gate_from_two_controls():
    out = _translate({"cols": [["•", "•", "X"]], "init": ["+", "+"]})
    start = _plus_state(3, [0, 1])
    ccx = np.eye(8, dtype=complex)
    ccx[[6, 7]] = ccx[[7, 6]]
    assert _fidelity(_run(out["qasm"]), ccx @ start) > 1 - 1e-9


def test_a_rotation_by_a_constant_angle_and_a_swap():
    out = _translate({"cols": [["H"], ["Z^¼"], [{"id": "Ryft", "arg": "pi/3"}]]})
    theta = math.pi / 3
    ry = np.array(
        [[math.cos(theta / 2), -math.sin(theta / 2)], [math.sin(theta / 2), math.cos(theta / 2)]],
        dtype=complex,
    )
    want = (ry @ T @ H)[:, 0]
    assert _fidelity(_run(out["qasm"]), want) > 1 - 1e-9

    swap = _translate({"cols": [["H"], [1, "H"], ["Z^¼"], ["Swap", "Swap"]]})
    n = 2
    start = _single(T, 0, n) @ _single(H, 1, n) @ _single(H, 0, n)
    start = start @ np.eye(4)[:, 0]
    sw = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
    assert _fidelity(_run(swap["qasm"]), sw @ start) > 1 - 1e-9


@pytest.mark.parametrize(
    "circuit, what",
    [
        ({"cols": [["H", "•"]]}, "H with controls"),
        ({"cols": [[{"id": "Rxft", "arg": "pi t"}]]}, "formula of time"),
        ({"cols": [["Measure"]]}, "gate Measure"),
        ({"cols": [["•", "•", "•", "X"]]}, "X with 3 controls"),
    ],
)
def test_what_has_no_translation_is_named_and_not_run(circuit, what):
    out = _translate(circuit)
    assert any(what in line for line in out["skipped"])
    assert len(out["qasm"].splitlines()) == 5


def test_displays_change_nothing_and_initial_states_are_prepared():
    out = _translate({"cols": [["Bloch"], ["Amps"], ["Chance"]], "init": ["1", "+"]})
    assert out["skipped"] == []
    assert _fidelity(_run(out["qasm"]), np.kron([0, 1], [1, 1]) / math.sqrt(2)) > 1 - 1e-9


_SCRIPT_BACK = (
    "global.window = {};"
    "eval(require('fs').readFileSync(process.argv[1], 'utf8'));"
    "process.stdout.write(JSON.stringify(window.DEQuirkBridge.qasmToQuirk(process.argv[2])));"
)
NL = chr(10)


def _to_quirk(qasm: str) -> dict:
    done = subprocess.run(
        [_node(), "-e", _SCRIPT_BACK, str(BRIDGE), qasm],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def _qasm(*lines: str, n: int = 3) -> str:
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    return NL.join(head + list(lines) + ["measure q -> c;"]) + NL


def test_a_circuit_written_in_openqasm_goes_to_quirk_and_back_with_the_same_state():
    text = _qasm(
        "h q[0]; x q[1]; y q[2]; z q[0]; s q[1]; sdg q[2]; t q[0]; tdg q[1]; sx q[2];",
        "rx(0.3) q[0]; ry(pi/3) q[1]; rz(pi/2*(1+1)) q[2];",
        "cx q[0],q[1]; cy q[1],q[2]; cz q[2],q[0]; swap q[0],q[2];",
        "cp(pi/2) q[0],q[1]; ccx q[0],q[1],q[2];",
    )
    drawn = _to_quirk(text)
    assert drawn["skipped"] == []
    assert drawn["n"] == 3
    assert all(len(col) == 3 for col in drawn["circuit"]["cols"])
    back = _translate(drawn["circuit"])
    assert back["skipped"] == []
    assert _fidelity(_run(back["qasm"]), _run(text)) > 1 - 1e-9


def test_a_preset_of_the_page_goes_to_quirk_and_back_with_the_same_state():
    qasm = _qasm("h q[0];", "cx q[0],q[1];", "cx q[1],q[2];")
    back = _translate(_to_quirk(qasm)["circuit"])
    assert _fidelity(_run(back["qasm"]), _run(qasm)) > 1 - 1e-9


def test_what_quirk_has_no_gate_for_is_named_and_a_big_circuit_says_its_size():
    out = _to_quirk(_qasm("crz(0.5) q[0],q[1];", "h q[2];"))
    assert any("crz" in line for line in out["skipped"])
    assert len(out["circuit"]["cols"]) == 1
    assert _to_quirk(_qasm("h q[0];", n=20))["n"] == 20
