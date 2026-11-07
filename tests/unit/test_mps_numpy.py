"""Cross-checks of MPSSimulatorNumpy against DenseSVSimulator and MPSSimulator.

The eight tests of item C of Task MPS-01: statevector against the dense
simulator up to global phase, bond dimension and top probabilities against
the JAX MPSSimulator, chi_search against its JAX form, the three non-local
gates, the Pauli expectation value, the standalone (no JAX) import, the
Python-int indices at 150 qubits, and the error paths.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import dense_evolution as de
from dense_evolution.backends.mps import MPSSimulator, _vectorized_chi_search
from dense_evolution.backends.mps_numpy import (MPSSimulatorNumpy, chi_search,
                                                pauli_expectation)


def _phase_invariant_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.vdot(b, a)
    if abs(overlap) < 1e-15:
        return float(np.max(np.abs(a - b)))
    phase = overlap / abs(overlap)
    return float(np.max(np.abs(a - phase * b)))


def _run_dense(ops, n: int) -> np.ndarray:
    sim = de.DenseSVSimulator(n_qubits=n, use_float32=False)
    sim.run_circuit_jit(ops)
    return np.asarray(sim.get_statevector())


def _run_mps_numpy(ops, n: int, max_bond: int = 64) -> "MPSSimulatorNumpy":
    mps = MPSSimulatorNumpy(n, max_bond=max_bond)
    mps.run_circuit(ops)
    return mps


def test_a_statevector_matches_dense_up_to_phase() -> None:
    n = 10
    cases = [
        ("ghz(10)", de.ghz_state(n)),
        ("qft(10)", de.qft(n)),
        ("random(10,20,s=1)", de.random_circuit(n, 20, seed=1)),
        ("random(10,20,s=2)", de.random_circuit(n, 20, seed=2)),
        ("random(10,20,s=3)", de.random_circuit(n, 20, seed=3)),
    ]
    for label, ops in cases:
        psi_dense = _run_dense(ops, n)
        mps = _run_mps_numpy(ops, n, max_bond=1024)
        psi_mps = mps.contract_to_statevector()
        err = _phase_invariant_max_abs(psi_dense, psi_mps)
        assert err < 1e-8, f"{label}: |psi_dense - phase*psi_mps|_inf = {err}"


def _drive_jax_mps_eager(mps, ops) -> None:
    """Drive MPSSimulator's eager path with the same gate matrices.

    ``MPSSimulator.run_circuit_jit`` rejects ``swap`` and every gate that
    is not in ``GATE_IDS``; the eager ``apply_gate_1q``/``apply_gate_2q``/
    ``apply_ccx`` methods are the same algorithm. The gate matrices come
    from ``gate_1q``/``gate_2q`` of this test's NumPy side, which test (a)
    independently verifies against ``DenseSVSimulator``.
    """
    import jax.numpy as jnp

    from dense_evolution.backends.mps_numpy import gate_1q, gate_2q

    static_1q = ("id", "i", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx")
    param_1q = ("rx", "ry", "rz", "p", "u1", "phase")
    static_2q = ("cx", "cy", "cz", "swap")
    param_2q = ("cp", "crz")
    for op in ops:
        name = op[0].lower()
        if name in static_1q:
            mps.apply_gate_1q(jnp.asarray(gate_1q(name)), int(op[1]))
        elif name in param_1q:
            mps.apply_gate_1q(jnp.asarray(gate_1q(name, float(op[2]))), int(op[1]))
        elif name in static_2q:
            mat = gate_2q(name).reshape(2, 2, 2, 2)
            mps.apply_gate_2q(jnp.asarray(mat), int(op[1]), int(op[2]))
        elif name in param_2q:
            mat = gate_2q(name, float(op[3])).reshape(2, 2, 2, 2)
            mps.apply_gate_2q(jnp.asarray(mat), int(op[1]), int(op[2]))
        elif name == "ccx":
            mps.apply_ccx(int(op[1]), int(op[2]), int(op[3]))
        else:
            raise ValueError(f"_drive_jax_mps_eager: unknown gate {name!r}")


def test_b_bond_dimension_and_top_probs_match_jax_mps() -> None:
    n = 12
    ops = de.random_circuit(n, 24, seed=5)
    mps_jax = MPSSimulator(n_qubits=n, max_bond=8, use_float32=False)
    mps_np = MPSSimulatorNumpy(n, max_bond=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _drive_jax_mps_eager(mps_jax, ops)
        mps_np.run_circuit(ops)

    chi_jax = [int(c) for c in np.asarray(mps_jax._real_chi)]
    chi_np = [int(c) for c in mps_np.bond_dimensions()]
    assert len(chi_jax) == len(chi_np) == n + 1
    assert chi_jax == chi_np, f"bond dims differ: jax={chi_jax} np={chi_np}"

    idx_j, p_j = mps_jax.get_top_k_probable_states(16)
    idx_n, p_n = mps_np.get_top_k_probable_states(16)
    d_j = {int(i): float(p) for i, p in zip(idx_j, p_j)}
    d_n = {int(i): float(p) for i, p in zip(idx_n, p_n)}
    shared = set(d_j) & set(d_n)
    assert len(shared) >= 12, (
        f"only {len(shared)} indices shared between top-16 sets; "
        f"jax={sorted(d_j)} np={sorted(d_n)}"
    )
    for idx in shared:
        assert abs(d_j[idx] - d_n[idx]) < 1e-6, (idx, d_j[idx], d_n[idx])
    assert abs(float(np.sum(p_j)) - float(np.sum(p_n))) < 1e-6


def test_c_chi_search_matches_vectorized_jax() -> None:
    import jax.numpy as jnp

    rng = np.random.default_rng(20260419)
    for trial in range(200):
        dim = int(rng.integers(2, 41))
        raw = rng.random(dim)
        s = np.sort(raw)[::-1]
        s = s / float(np.linalg.norm(s))

        chi_np, jsd_np = chi_search(s, 1e-12, 1e-5, 16)
        chi_jx, jsd_jx = _vectorized_chi_search(jnp.asarray(s), 1e-12, 1e-5, 16)
        assert int(chi_np) == int(chi_jx), (trial, dim, chi_np, chi_jx)
        assert abs(float(jsd_np) - float(jsd_jx)) < 1e-9, (trial, dim, jsd_np, jsd_jx)


def test_d_nonlocal_cx_cp_ccx_match_dense() -> None:
    n = 8
    base = [("h", q) for q in range(n)]

    cases = [
        ("cx 0->5", base + [("cx", 0, 5)]),
        ("cp 5->1 angle=0.7", base + [("cp", 5, 1, 0.7)]),
        ("ccx 4,0,2", base + [("ccx", 4, 0, 2)]),
    ]
    for label, ops in cases:
        psi_dense = _run_dense(ops, n)
        mps = _run_mps_numpy(ops, n, max_bond=1024)
        psi_mps = mps.contract_to_statevector()
        err = _phase_invariant_max_abs(psi_dense, psi_mps)
        assert err < 1e-9, f"{label}: err = {err}"


def test_e_pauli_expectation_matches_dense() -> None:
    n = 5
    ops = de.random_circuit(n, 20, seed=17)
    psi = _run_dense(ops, n)
    mps = _run_mps_numpy(ops, n)
    for s in ("ZZIIX", "XIIIX", "IIYII"):
        expected = complex(de.pauli_expectation(psi, s, n_qubits=n))
        got = complex(pauli_expectation(mps, s))
        err = abs(expected - got)
        assert err < 1e-9, f"{s}: expected={expected} got={got} err={err}"


def test_f_standalone_import_without_jax() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
import importlib.util
import sys
sys.modules["jax"] = None

for name, path in (
    ("mps_numpy", "dense_evolution/backends/mps_numpy.py"),
    ("qasm_subset", "dense_evolution/backends/qasm_subset.py"),
):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

import mps_numpy

n = 100
ops = [("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)]
sim = mps_numpy.MPSSimulatorNumpy(n)
sim.run_circuit(ops)
idx, p = sim.get_top_k_probable_states(2)
assert sorted(idx) == [0, 2 ** n - 1], sorted(idx)
assert abs(float(p[0]) - 0.5) < 1e-9
assert abs(float(p[1]) - 0.5) < 1e-9
print("STANDALONE_OK")
"""
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert r.returncode == 0, r.stderr
    assert "STANDALONE_OK" in r.stdout


def test_g_top_k_indices_are_python_ints_on_150_qubits() -> None:
    n = 150
    mps = MPSSimulatorNumpy(n)
    mps.run_circuit([("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)])
    idx, p = mps.get_top_k_probable_states(2)
    for i in idx:
        assert isinstance(i, int), f"index {i!r} has type {type(i).__name__}"
    assert sorted(idx) == [0, 2**n - 1]
    assert abs(float(p[0]) - 0.5) < 1e-9
    assert abs(float(p[1]) - 0.5) < 1e-9


def test_h_unknown_gate_and_statevector_limit() -> None:
    mps = MPSSimulatorNumpy(3)
    with pytest.raises(ValueError):
        mps.run_circuit([("not_a_gate", 0)])

    mps_25 = MPSSimulatorNumpy(25)
    with pytest.raises(MemoryError):
        mps_25.contract_to_statevector()
