"""QAOA and VQE of variational_numpy, against references that do not use the engine.

* The QAOA state is built here from Eqs. 2 to 7 of Farhi, Goldstone and Gutmann
  (arXiv:1411.4028) with dense matrices and compared with ``qaoa_expectation``.
* The ring of disagrees at p = 1 has the maximum 3n/4 that the paper gives.
* The gradient of the parameter-shift rule is compared with central finite differences.
* The ground energy of the transverse-field Ising chain comes from the eigenvalues of its
  dense matrix; a variational energy can never go below it.
* Single precision is compared with double precision on the same seeds.
"""

import math
import time

import numpy as np
import pytest

from dense_evolution.backends import variational_numpy as vn

RING4 = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]


def _ring(n):
    return [(i, (i + 1) % n, 1.0) for i in range(n)]


def _kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def _paper_expectation(n, edges, gammas, betas):
    """F_p from Eqs. 1 to 7, with dense matrices, qubit 0 the most significant bit."""
    dim = 2**n
    bits = (np.arange(dim)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    cost = np.zeros(dim)
    for j, k, w in edges:
        cost += w * 0.5 * (1 - (1 - 2 * bits[:, j]) * (1 - 2 * bits[:, k]))
    state = np.full(dim, 1 / math.sqrt(dim), dtype=complex)
    for g, b in zip(gammas, betas):
        state = np.exp(-1j * g * cost) * state
        mixer = np.array([[math.cos(b), -1j * math.sin(b)], [-1j * math.sin(b), math.cos(b)]])
        for q in range(n):
            state = _kron_all([mixer if i == q else np.eye(2) for i in range(n)]) @ state
    return float(np.real(np.vdot(state, cost * state)))


@pytest.mark.parametrize("p", [1, 2])
def test_qaoa_expectation_equals_the_paper_formulas(p):
    rng = np.random.default_rng(3)
    gammas, betas = rng.uniform(0, 2 * math.pi, p), rng.uniform(0, math.pi, p)
    want = _paper_expectation(4, RING4, gammas, betas)
    got64 = vn.qaoa_expectation(4, RING4, gammas, betas, np.complex128)
    got32 = vn.qaoa_expectation(4, RING4, gammas, betas, np.complex64)
    print(f"QAOA p={p}: paper {want:.9f}  complex128 {got64:.9f}  complex64 {got32:.9f}")
    assert got64 == pytest.approx(want, abs=1e-8)
    assert got32 == pytest.approx(want, abs=1e-4)


def test_a_weighted_graph_follows_the_paper_too():
    edges = [(0, 1, 2.0), (1, 2, 0.5), (0, 2, 1.5)]
    want = _paper_expectation(3, edges, [0.7], [0.3])
    assert vn.qaoa_expectation(3, edges, [0.7], [0.3], np.complex128) == pytest.approx(
        want, abs=1e-8
    )


def test_the_ring_of_disagrees_at_p1_reaches_three_quarters():
    n = 8
    best = max(
        vn.qaoa_expectation(n, _ring(n), [g], [b], np.complex128)
        for g in np.linspace(0, 2 * math.pi, 36, endpoint=False)
        for b in np.linspace(0, math.pi, 18, endpoint=False)
    )
    print(f"ring n={n}, p=1: best F_1 on the grid {best:.6f}, paper 3n/4 = {3 * n / 4}")
    assert best <= 3 * n / 4 + 1e-6
    assert best >= 0.95 * 3 * n / 4


def test_parameter_shift_equals_finite_differences():
    rng = np.random.default_rng(5)
    gammas, betas = rng.uniform(0, 2 * math.pi, 2), rng.uniform(0, math.pi, 2)
    d_gamma, d_beta = vn.qaoa_gradient(4, RING4, gammas, betas, np.complex128)
    eps = 1e-5
    for layer in range(2):
        up, down = gammas.copy(), gammas.copy()
        up[layer] += eps
        down[layer] -= eps
        fd = (
            vn.qaoa_expectation(4, RING4, up, betas, np.complex128)
            - vn.qaoa_expectation(4, RING4, down, betas, np.complex128)
        ) / (2 * eps)
        assert d_gamma[layer] == pytest.approx(fd, abs=1e-6)
        up, down = betas.copy(), betas.copy()
        up[layer] += eps
        down[layer] -= eps
        fd = (
            vn.qaoa_expectation(4, RING4, gammas, up, np.complex128)
            - vn.qaoa_expectation(4, RING4, gammas, down, np.complex128)
        ) / (2 * eps)
        assert d_beta[layer] == pytest.approx(fd, abs=1e-6)


def test_qaoa_in_single_precision_matches_double_precision_and_reaches_the_p1_maximum():
    n, edges = 6, _ring(6)
    out = {}
    for dtype in (np.complex64, np.complex128):
        t0 = time.perf_counter()
        out[dtype] = vn.qaoa(n, edges, p=2, maxiter=30, seed=1, dtype=dtype)
        out[dtype]["seconds"] = time.perf_counter() - t0
    best = exact = vn.exact_maxcut(n, edges)
    e32, e64 = out[np.complex64]["expectation"], out[np.complex128]["expectation"]
    print(
        f"QAOA ring n=6 p=2: float32 {e32:.6f} ({out[np.complex64]['seconds']:.1f} s)  "
        f"float64 {e64:.6f} ({out[np.complex128]['seconds']:.1f} s)  best cut {best}"
    )
    assert e32 <= exact + 1e-3 and e64 <= exact + 1e-9
    assert abs(e32 - e64) < 1e-2
    assert e64 >= 0.75 * n


def _tfim(n, g):
    terms = [(-1.0, vn.pauli_string(n, {i: "Z", i + 1: "Z"})) for i in range(n - 1)]
    terms += [(-g, vn.pauli_string(n, {i: "X"})) for i in range(n)]
    dense = np.zeros((2**n, 2**n), dtype=complex)
    for c, s in terms:
        dense += c * _kron_all([{"I": np.eye(2), "X": X, "Z": Z}[ch] for ch in s])
    return terms, float(np.linalg.eigvalsh(dense)[0])


def test_vqe_in_single_precision_matches_double_precision():
    n = 4
    terms, exact = _tfim(n, 1.0)
    res = {}
    for dtype in (np.complex64, np.complex128):
        t0 = time.perf_counter()
        res[dtype] = vn.vqe(terms, n, layers=3, maxiter=60, seed=0, dtype=dtype)
        res[dtype]["seconds"] = time.perf_counter() - t0
    e32, e64 = res[np.complex64]["energy"], res[np.complex128]["energy"]
    print(
        f"VQE TFIM n={n}: exact {exact:.6f}  float32 {e32:.6f} ({res[np.complex64]['seconds']:.1f} s)  "
        f"float64 {e64:.6f} ({res[np.complex128]['seconds']:.1f} s)"
    )
    assert e64 >= exact - 1e-9
    assert e32 >= exact - 1e-4
    assert abs(e32 - e64) < 5e-3
    for r in res.values():
        assert r["history"][-1] < r["history"][0]


def test_the_ansatz_has_n_times_layers_parameters_and_the_dtype_is_checked():
    assert len(vn.hardware_efficient_ops(np.zeros(6), 3, 2, (1, 0, 0))) == 1 + 6 + 4
    with pytest.raises(ValueError, match="expected 6 parameters"):
        vn.hardware_efficient_ops(np.zeros(5), 3, 2)
    with pytest.raises(ValueError, match="complex64"):
        vn.energy([], 1, [(1.0, "Z")], dtype=np.float32)
