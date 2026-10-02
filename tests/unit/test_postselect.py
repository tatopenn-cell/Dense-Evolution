import numpy as np
import pytest

import dense_evolution as de


def _run(n, ops, sv=None):
    sim = de.DenseSVSimulator(n)
    if sv is not None:
        sim.set_initial_state(np.asarray(sv))
    sim.run_circuit_jit(ops)
    return np.asarray(sim.get_statevector())


PREP = [('h', 0), ('ry', 1, 0.7), ('cx', 0, 1), ('rx', 2, 1.3), ('cx', 1, 2), ('rz', 0, 0.4)]


@pytest.mark.parametrize("q, outcome", [(0, '0'), (1, '1'), (2, '1')])
def test_computational_postselection_keeps_matching_amplitudes(q, outcome):
    sv = _run(3, PREP)
    keep = np.array([((i >> (2 - q)) & 1) == int(outcome) for i in range(8)])
    want = np.where(keep, sv, 0)
    p = np.sum(np.abs(want) ** 2)
    out, prob = de.postselect(sv, 3, q, outcome)
    np.testing.assert_allclose(prob, p, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out), want / np.sqrt(p), atol=1e-12)


@pytest.mark.parametrize("outcome, to_z, back", [
    ('+', [('h', 1)], [('h', 1)]),
    ('-', [('h', 1)], [('h', 1)]),
    ('+i', [('sdg', 1), ('h', 1)], [('h', 1), ('s', 1)]),
    ('-i', [('sdg', 1), ('h', 1)], [('h', 1), ('s', 1)]),
])
def test_other_bases_match_basis_change_circuit(outcome, to_z, back):
    sv = _run(3, PREP)
    z = '0' if outcome in ('+', '+i') else '1'
    mid, p_ref = de.postselect(_run(3, to_z, sv), 3, 1, z)
    ref = _run(3, back, mid)
    out, p = de.postselect(sv, 3, 1, outcome)
    np.testing.assert_allclose(p, p_ref, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-12)


def test_intermediate_postselection_equals_ancilla_postselected_at_the_end():
    after = [('h', 0), ('cx', 0, 2), ('ry', 1, 0.9)]
    sv = _run(3, PREP)
    mid, p_mid = de.postselect(sv, 3, 0, '1')
    direct = _run(3, after, mid)
    late = _run(4, PREP + [('cx', 0, 3)] + after)
    end, p_end = de.postselect(late, 4, 3, '1')
    np.testing.assert_allclose(p_end, p_mid, atol=1e-12)
    np.testing.assert_allclose(np.asarray(end).reshape(8, 2)[:, 1], direct, atol=1e-12)


def test_result_is_normalised():
    out, _ = de.postselect(_run(3, PREP), 3, 2, '-i')
    np.testing.assert_allclose(np.linalg.norm(np.asarray(out)), 1.0, atol=1e-12)


def test_zero_probability_outcome_raises():
    with pytest.raises(ValueError, match="zero probability"):
        de.postselect(_run(2, []), 2, 0, '1')


def test_unknown_state_raises():
    with pytest.raises(ValueError, match="state must be"):
        de.postselect(_run(2, []), 2, 0, 'x')
