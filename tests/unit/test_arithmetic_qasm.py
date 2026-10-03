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


@pytest.mark.parametrize("n,N,a", [(2, 3, 2), (3, 5, 3), (3, 7, 4)])
def test_cmult_and_controlled_ua_match_permutations(n, N, a):
    from dense_evolution.circuits.arithmetic_qasm import cmult_mod_qasm, controlled_ua_qasm
    nq = 2 * n + 3
    x, b = list(range(1, n + 1)), list(range(n + 1, 2 * n + 2))
    xv = sum(bit(nq, q) << i for i, q in enumerate(x))
    bv = sum(bit(nq, q) << i for i, q in enumerate(b[:n]))
    clean = (bit(nq, 2 * n + 1) == 0) & (bit(nq, 2 * n + 2) == 0)
    sv = random_state(nq, clean & (bv < N))
    ref = ar.multiply_add_mod(sv, nq, x, b[:n], a, N, controls=(0,))
    np.testing.assert_allclose(run(cmult_mod_qasm(a, N, n), sv), ref, atol=1e-12)
    sv = random_state(nq, clean & (bv == 0) & (xv < N))
    ref = ar.multiply_mod(sv, nq, x, a, N, controls=(0,))
    np.testing.assert_allclose(run(controlled_ua_qasm(a, N, n), sv), ref, atol=1e-12)


def test_controlled_ua_rejects_non_invertible_a():
    from dense_evolution.circuits.arithmetic_qasm import controlled_ua_qasm
    with pytest.raises(ValueError):
        controlled_ua_qasm(3, 15, 4)


@pytest.mark.parametrize("a,allowed", [(7, {0, 64, 128, 192}), (11, {0, 128})])
def test_shor_order_finding_n15_lands_on_multiples_of_2m_over_r(a, allowed):
    from dense_evolution.circuits.shor import shor_order_finding
    for seed in range(4):
        y, m = shor_order_finding(a, 15, rng=seed)
        assert m == 8 and y in allowed


def test_shor_order_finding_noise_moves_outcomes_off_the_peaks():
    from dense_evolution.circuits.shor import shor_order_finding
    ys = [shor_order_finding(7, 15, rng=s, noise_model="depolarizing", p=0.2)[0] for s in range(8)]
    assert any(y not in {0, 64, 128, 192} for y in ys)
