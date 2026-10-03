import numpy as np
import pytest

import dense_evolution as de
from dense_evolution.circuits import arithmetic as ar
from dense_evolution.circuits.arithmetic_qasm import (
    cuccaro_adder_qasm, draper_adder_qasm,
    constant_adder_qasm, modular_constant_adder_qasm,
)

rng = np.random.default_rng(352)


def run(qasm, sv):
    circ = de.QASMParser().parse(qasm)
    sim = de.DenseSVSimulator(circ.n_qubits)
    sim.set_initial_state(sv)
    sim.run_circuit_jit(circ.to_tuples())
    return np.asarray(sim.get_statevector())


def random_state(nq, mask=None):
    v = rng.normal(size=2 ** nq) + 1j * rng.normal(size=2 ** nq)
    if mask is not None:
        v = v * mask
    return v / np.linalg.norm(v)


def bit(nq, q):
    return (np.arange(2 ** nq) >> (nq - 1 - q)) & 1


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cuccaro_matches_add_registers(n):
    nq = 2 * n + 2
    sv = random_state(nq, bit(nq, nq - 1) == 0)
    ref = ar.add_registers(sv, nq, list(range(n)), list(range(n, 2 * n + 1)))
    np.testing.assert_allclose(run(cuccaro_adder_qasm(n), sv), ref, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_draper_matches_add_registers(n):
    nq = 2 * n + 1
    sv = random_state(nq)
    ref = ar.add_registers(sv, nq, list(range(n)), list(range(n, nq)))
    np.testing.assert_allclose(run(draper_adder_qasm(n), sv), ref, atol=1e-12)


@pytest.mark.parametrize("c", [0, 1, 5, 7, -3])
def test_constant_adder_matches_add_constant(c):
    sv = random_state(3)
    np.testing.assert_allclose(run(constant_adder_qasm(c, 3), sv), ar.add_constant(sv, 3, [0, 1, 2], c), atol=1e-12)


@pytest.mark.parametrize("n,N", [(2, 3), (3, 5), (4, 15)])
def test_modular_adder_matches_add_constant_mod(n, N):
    nq = n + 2
    value = sum(bit(nq, q) << q for q in range(n))
    mask = (bit(nq, n) == 0) & (bit(nq, n + 1) == 0) & (value < N)
    for c in range(N):
        sv = random_state(nq, mask)
        ref = ar.add_constant_mod(sv, nq, list(range(n)), c, N)
        np.testing.assert_allclose(run(modular_constant_adder_qasm(c, N, n), sv), ref, atol=1e-12)


def test_cuccaro_gate_counts():
    ops = [op[0] for op in de.QASMParser().parse(cuccaro_adder_qasm(4)).to_tuples()]
    assert ops.count("ccx") == 8 and ops.count("cx") == 17


@pytest.mark.parametrize("call", [
    lambda: cuccaro_adder_qasm(0),
    lambda: modular_constant_adder_qasm(1, 16, 4),
    lambda: modular_constant_adder_qasm(5, 5, 3),
])
def test_invalid_arguments_raise(call):
    with pytest.raises(ValueError):
        call()
