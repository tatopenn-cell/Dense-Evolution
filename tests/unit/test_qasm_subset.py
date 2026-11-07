"""Tests of parse_qasm_subset, the OpenQASM reader of the browser simulator."""

import math

import pytest
from dashboard_core.qasm_library import gate_tuples_to_qasm

import dense_evolution as de
from dense_evolution.backends.qasm_subset import parse_qasm_subset

HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
PATTERNS = ("linear", "circular", "full", "star", "brick")


def _circuits():
    cases = [
        ("ghz8", de.ghz_state(8), 8),
        ("qft6", de.qft(6), 6),
        ("random5", de.random_circuit(5, 12, seed=13), 5),
    ]
    for pattern in PATTERNS:
        ops = [("h", q) for q in range(5)] + de.entangling_layer(5, pattern=pattern)
        cases.append((f"entangling-{pattern}", ops, 5))
    return cases


@pytest.mark.parametrize("label, ops, n", _circuits(), ids=[c[0] for c in _circuits()])
def test_the_text_of_gate_tuples_parses_back_to_the_same_tuples(label, ops, n):
    n_back, back = parse_qasm_subset(gate_tuples_to_qasm(ops, n))
    assert n_back == n
    assert len(back) == len(ops)
    for original, parsed in zip(ops, back):
        assert len(parsed) == len(original)
        assert parsed[0] == original[0]
        for want, got in zip(original[1:], parsed[1:]):
            if isinstance(want, float):
                assert abs(got - want) < 1e-12
            else:
                assert got == want


@pytest.mark.parametrize(
    "angle, expected",
    [
        ("pi/2", math.pi / 2),
        ("-pi/4", -math.pi / 4),
        ("3*pi/8", 3 * math.pi / 8),
        ("((pi)/2)", math.pi / 2),
        ("pi/2*(1+1)", math.pi),
        ("2*(pi/4)+0.1", 2 * (math.pi / 4) + 0.1),
        ("1e-3", 1e-3),
        (".5", 0.5),
        ("-.5", -0.5),
        ("0.3", 0.3),
    ],
)
def test_an_angle_is_an_arithmetic_expression_with_nested_parentheses(angle, expected):
    _, ops = parse_qasm_subset(HEAD + f"rz({angle}) q[0];\n")
    assert ops == [("rz", 0, pytest.approx(expected, abs=1e-15))]


def test_every_gate_of_the_subset_has_its_tuple():
    body = (
        "h q[0];\nrx(0.1) q[1];\nu1(0.2) q[2];\nphase(0.3) q[0];\n"
        "cx q[0],q[1];\nswap q[1],q[2];\ncp(0.4) q[0],q[2];\ncrz(0.5) q[2],q[0];\n"
        "ccx q[0],q[1],q[2];\nSDG q[1];\n"
    )
    _, ops = parse_qasm_subset(HEAD + body)
    assert [op[0] for op in ops] == [
        "h",
        "rx",
        "u1",
        "phase",
        "cx",
        "swap",
        "cp",
        "crz",
        "ccx",
        "sdg",
    ]
    assert ops[-2] == ("ccx", 0, 1, 2)
    assert ops[6] == ("cp", 0, 2, 0.4)


def test_several_statements_can_share_a_line_and_comments_are_dropped():
    _, ops = parse_qasm_subset(HEAD + "h q[0]; cx q[0],q[1]; // bell\n// only a comment\n")
    assert ops == [("h", 0), ("cx", 0, 1)]


def test_measure_barrier_and_further_cregs_are_skipped():
    text = HEAD + "creg d[1];\nbarrier q;\nh q[0];\nmeasure q -> c;\n"
    assert parse_qasm_subset(text) == (3, [("h", 0)])


@pytest.mark.parametrize(
    "body, line, words",
    [
        ("foo q[0];\n", 5, "unknown gate"),
        ("rx(foo) q[0];\n", 5, "angle"),
        ("rx(1/0) q[0];\n", 5, "division by zero"),
        ("rx((1+2) q[0];\n", 5, "unbalanced"),
        ("rx(1+) q[0];\n", 5, "angle"),
        ("rx q[0];\n", 5, "needs an angle"),
        ("h(0.1) q[0];\n", 5, "no angle"),
        ("h q[3];\n", 5, "outside the register"),
        ("h q[0]\n", 5, "missing ';'"),
        ("cx q[0];\n", 5, "2 qubits"),
        ("qreg r[2];\n", 5, "second qreg"),
        ("h q[0];\nh r[0];\n", 6, "bad qubit argument"),
    ],
)
def test_every_error_names_its_line(body, line, words):
    with pytest.raises(ValueError, match=rf"^line {line}: .*{words}"):
        parse_qasm_subset(HEAD + body)


def test_header_register_and_order_errors():
    with pytest.raises(ValueError, match=r"^line 1: only OPENQASM 2.0"):
        parse_qasm_subset("OPENQASM 3.0;\nqreg q[1];\n")
    with pytest.raises(ValueError, match=r"^line 2: gate before the qreg"):
        parse_qasm_subset("OPENQASM 2.0;\nh q[0];\n")
    with pytest.raises(ValueError, match="no qreg"):
        parse_qasm_subset("OPENQASM 2.0;\n")
    with pytest.raises(ValueError, match=r"^line 2: qreg size"):
        parse_qasm_subset("OPENQASM 2.0;\nqreg q[0];\n")
