"""
Tests for the erasure decoders in dense_evolution.physics.qec: maximum-likelihood (GF(2)), peeling,
Union-Find and erasure-weighted matching decoders, and the Spitz et al.
detection-event estimator, each checked against the maximum-likelihood one.
"""
import itertools
import numpy as np
import pytest

from dense_evolution.physics import qec as E
from dense_evolution.physics import qec as Q

STEANE = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']


def surface(d):
    n = d * d
    xs, zs = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qs = [r * d + c for r, c in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)) if 0 <= r < d and 0 <= c < d]
            typ = 'X' if (i + j) % 2 == 0 else 'Z'
            if len(qs) == 4 or (len(qs) == 2 and ((typ == 'X') == (i in (0, d)))):
                s = ['I'] * n
                for q in qs:
                    s[q] = typ
                (xs if typ == 'X' else zs).append(''.join(s))
    return xs + zs


def equiv(err, corr, stabs):
    if corr is None:
        return False
    sym = np.array([Q._pauli_to_symplectic(s) for s in stabs])
    r, p = Q._gf2_rref(sym)
    return Q._in_gf2_span(Q._pauli_to_symplectic(err) ^ Q._pauli_to_symplectic(corr), r, p)


def sample(n, p_erase, p_pauli, rng):
    er = [q for q in range(n) if rng.random() < p_erase]
    e = ''.join('IXZY'[rng.integers(0, 4)] if q in er else ('XZY'[rng.integers(0, 3)] if rng.random() < p_pauli else 'I')
                for q in range(n))
    return er, e


def test_ml_steane_three_erasures_fail_only_on_the_seven_logical_supports():
    bad = 0
    for S in itertools.combinations(range(7), 3):
        for b in itertools.product(range(4), repeat=3):
            e = ''.join('IXZY'[b[S.index(i)]] if i in S else 'I' for i in range(7))
            if not equiv(e, E.erasure_ml_decode(Q.compute_syndrome(e, STEANE), S, 7, STEANE), STEANE):
                bad += 1
                break
    assert bad == 7


def test_ml_failure_falls_with_distance_below_one_half_and_rises_above():
    rng = np.random.default_rng(0)

    def rate(d, p):
        st, n = surface(d), d * d
        f = 0
        for _ in range(200):
            er, e = sample(n, p, 0.0, rng)
            f += not equiv(e, E.erasure_ml_decode(Q.compute_syndrome(e, st), er, n, st), st)
        return f / 200

    assert rate(3, 0.3) > rate(7, 0.3)
    assert rate(3, 0.6) < rate(7, 0.6)


@pytest.mark.parametrize("d", [3, 5])
def test_peeling_and_union_find_are_right_wherever_ml_is_right(d):
    rng = np.random.default_rng(d)
    st, n = surface(d), d * d
    for _ in range(150):
        er, e = sample(n, 0.3, 0.0, rng)
        sy = Q.compute_syndrome(e, st)
        if equiv(e, E.erasure_ml_decode(sy, er, n, st), st):
            assert equiv(e, E.peeling_decode(st, sy, er, n), st)
            assert equiv(e, E.union_find_decode(st, sy, er, n), st)


def test_matching_with_zero_weight_erasures_is_right_wherever_ml_is_right():
    pytest.importorskip("pymatching")
    rng = np.random.default_rng(9)
    st, n = surface(5), 25
    for _ in range(150):
        er, e = sample(n, 0.3, 0.0, rng)
        sy = Q.compute_syndrome(e, st)
        if equiv(e, E.erasure_ml_decode(sy, er, n, st), st):
            assert equiv(e, E.matching_erasure_decode(st, sy, er, n), st)


def test_union_find_corrects_every_single_pauli_error_and_every_two_erasures_d3():
    st = surface(3)
    for q in range(9):
        for P in 'XZY':
            e = ''.join(P if i == q else 'I' for i in range(9))
            assert equiv(e, E.union_find_decode(st, Q.compute_syndrome(e, st), [], 9), st)
    for S in itertools.combinations(range(9), 2):
        for b in itertools.product(range(4), repeat=2):
            e = ''.join('IXZY'[b[S.index(i)]] if i in S else 'I' for i in range(9))
            assert equiv(e, E.union_find_decode(st, Q.compute_syndrome(e, st), S, 9), st)


def test_union_find_failure_falls_with_distance_for_mixed_noise():
    rng = np.random.default_rng(4)

    def rate(d):
        st, n = surface(d), d * d
        f = 0
        for _ in range(300):
            er, e = sample(n, 0.1, 0.02, rng)
            f += not equiv(e, E.union_find_decode(st, Q.compute_syndrome(e, st), er, n), st)
        return f / 300

    assert rate(3) > rate(7)
