"""Tests of the helpers of the Build page, on values written by hand."""
import re

import pytest
from dashboard import circuit_tools as ct

BELL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
)
HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
TAIL = "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"


def test_x_line_goes_after_the_declarations_and_before_the_gates():
    assert ct.with_input_values(BELL, [1]) == HEAD + "x q[1];\n" + TAIL


def test_no_qubit_returns_the_text_unchanged():
    assert ct.with_input_values(BELL, []) == BELL


def test_lines_are_in_increasing_order_of_the_index():
    assert ct.with_input_values(BELL, [1, 0]) == HEAD + "x q[0];\nx q[1];\n" + TAIL


def test_an_index_out_of_range_is_ignored():
    assert ct.with_input_values(BELL, [5]) == BELL


def test_a_negative_index_and_a_text_with_no_qreg_are_refused():
    with pytest.raises(ValueError, match="-1"):
        ct.with_input_values(BELL, [-1])
    with pytest.raises(ValueError, match="qreg"):
        ct.with_input_values("OPENQASM 2.0;\n", [0])


def test_qubits_in_qasm():
    assert ct.qubits_in_qasm("OPENQASM 2.0;\nqreg q[2];\n") == 2
    assert ct.qubits_in_qasm("OPENQASM 2.0;\n") == 0


def test_rows_are_most_frequent_first():
    assert ct.outcome_rows({"11": 497, "00": 503}) == [
        {"outcome": "00", "shots": 503, "probability": 0.503},
        {"outcome": "11", "shots": 497, "probability": 0.497},
    ]


def test_equal_shots_are_sorted_by_outcome():
    rows = ct.outcome_rows({"10": 5, "01": 5, "00": 5})
    assert [row["outcome"] for row in rows] == ["00", "01", "10"]


def test_probabilities_sum_to_one():
    rows = ct.outcome_rows({"000": 3, "001": 7, "111": 90})
    assert sum(row["probability"] for row in rows) == pytest.approx(1.0, abs=1e-12)


def test_empty_counts_give_no_row():
    assert ct.outcome_rows({}) == []


def _edges(text, gate="cx"):
    pattern = rf"^{gate} q\[(\d+)\],q\[(\d+)\];$"
    return [(int(a), int(b)) for a, b in re.findall(pattern, text, re.M)]


def _layer(n, pattern, **kwargs):
    return ct.topology_qasm(n, pattern, hadamards=False, **kwargs)


def test_linear_on_four_qubits():
    assert _edges(_layer(4, "linear")) == [(0, 1), (1, 2), (2, 3)]


def test_circular_on_four_qubits_adds_the_wrap_around_edge():
    assert _edges(_layer(4, "circular")) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_circular_on_two_qubits_has_one_edge():
    assert _edges(_layer(2, "circular")) == [(0, 1)]


def test_full_on_four_qubits_has_six_edges():
    edges = _edges(_layer(4, "full"))
    assert len(edges) == 6
    assert set(edges) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_star_on_four_qubits_for_two_hubs():
    assert _edges(_layer(4, "star", hub=0)) == [(0, 1), (0, 2), (0, 3)]
    assert _edges(_layer(4, "star", hub=2)) == [(2, 0), (2, 1), (2, 3)]


def test_brick_on_five_qubits_is_even_pairs_then_odd_pairs():
    assert _edges(_layer(5, "brick")) == [(0, 1), (2, 3), (1, 2), (3, 4)]


def test_reverse_swaps_control_and_target_of_every_edge():
    assert _edges(_layer(3, "linear", reverse=True)) == [(1, 0), (2, 1)]


def test_hadamards_start_the_circuit_only_when_asked():
    with_h = ct.topology_qasm(3, "linear", hadamards=True)
    without_h = ct.topology_qasm(3, "linear", hadamards=False)
    assert re.findall(r"^h q\[(\d+)\];$", with_h, re.M) == ["0", "1", "2"]
    assert "h q[" not in without_h


def test_the_gate_is_written_on_every_edge():
    text = _layer(4, "linear", gate="cz")
    assert _edges(text, "cz") == [(0, 1), (1, 2), (2, 3)]
    assert "cx " not in text


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_qubits": 1, "pattern": "linear"}, "n_qubits"),
        ({"n_qubits": 4, "pattern": "ring"}, "pattern"),
        ({"n_qubits": 4, "pattern": "star", "hub": 4}, "hub"),
        ({"n_qubits": 4, "pattern": "linear", "gate": "swap"}, "gate"),
    ],
)
def test_wrong_arguments_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ct.topology_qasm(**kwargs)


def test_measurements_keep_every_count_and_the_input_string():
    rows = ct.measurement_rows({"11": 3, "00": 2}, [1], 2, seed=5)
    assert len(rows) == 5
    assert sorted(r["output"] for r in rows) == ["00", "00", "11", "11", "11"]
    assert {r["input"] for r in rows} == {"10"}
    assert [r["measurement"] for r in rows] == [1, 2, 3, 4, 5]


def test_measurements_have_a_fixed_order_for_a_seed_and_a_limit():
    counts = {"01": 50, "10": 50}
    assert ct.measurement_rows(counts, [], 2, seed=3) == ct.measurement_rows(counts, [], 2, seed=3)
    assert len(ct.measurement_rows(counts, [], 2, limit=7)) == 7


def test_qiskit_code_of_a_circuit_with_angles_and_three_qubit_gate():
    code = ct.qiskit_code([("h", 0), ("rx", 1, 0.3), ("cp", 0, 1, 0.5), ("ccx", 0, 1, 2)], 3)
    assert code.splitlines() == [
        "from qiskit import QuantumCircuit",
        "",
        "qc = QuantumCircuit(3, 3)",
        "qc.h(0)",
        "qc.rx(0.3, 1)",
        "qc.cp(0.5, 0, 1)",
        "qc.ccx(0, 1, 2)",
        "qc.measure(range(3), range(3))",
    ]


def test_qiskit_code_rejects_a_gate_without_a_qiskit_method():
    with pytest.raises(ValueError):
        ct.qiskit_code([("zz", 0, 1)], 2)


def test_state_csv_of_a_bell_pair():
    lines = ct.state_csv([2**-0.5, 0, 0, 2**-0.5], 2).splitlines()
    assert lines[0] == "state,re,im,probability"
    assert [line.split(",")[0] for line in lines[1:]] == ["00", "01", "10", "11"]
    assert float(lines[1].split(",")[3]) == pytest.approx(0.5)


def test_the_smallest_register_is_the_highest_qubit_used_plus_one():
    assert ct.min_qubits("qreg q[5];\ncreg c[5];\nh q[0];\ncx q[0],q[2];\n") == 3
    assert ct.min_qubits("qreg q[4];\n") == 1


def test_the_register_size_changes_in_both_declarations_and_nothing_else():
    text = "qreg q[2];\ncreg c[2];\nh q[0];\n"
    assert ct.with_qubit_count(text, 4) == "qreg q[4];\ncreg c[4];\nh q[0];\n"


def test_dense_cap_keeps_fifteen_percent_of_the_ram_free():
    assert ct.dense_qubit_cap(8000, 16000) == 28
    assert ct.dense_qubit_cap(300, 300) == 23
    assert ct.dense_qubit_cap(16000, 16000, ceiling=20) == 20
