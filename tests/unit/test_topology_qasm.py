"""Tests of the OpenQASM of an entangling layer, on edges written by hand."""
import re

import pytest
from dashboard_core.topology_qasm import topology_qasm


def _edges(text, gate="cx"):
    pattern = rf"^{gate} q\[(\d+)\],q\[(\d+)\];$"
    return [(int(a), int(b)) for a, b in re.findall(pattern, text, re.M)]


def _layer(n, pattern, **kwargs):
    return topology_qasm(n, pattern, hadamards=False, **kwargs)


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
    with_h = topology_qasm(3, "linear", hadamards=True)
    without_h = topology_qasm(3, "linear", hadamards=False)
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
        topology_qasm(**kwargs)
