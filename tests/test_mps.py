"""
Unit tests for dense_evolution/mps.py -- the Matrix Product State
simulator, ported from a private prototype after independently verifying
that its plain SVD truncation (no Lloyd-Max quantization, which measured
a real ~0.5% error against DenseSVSimulator for no bond-dimension benefit
and was dropped in this port) reproduces DenseSVSimulator exactly.

Cross-checks against the real DenseSVSimulator on entangling circuits are
the primary correctness signal here, not just internal self-consistency.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import dense_evolution as de
from dense_evolution.mps import (
    MPSSimulator, _jsd_vectors, _vectorized_chi_search, _expand_nonlocal_2q_positions,
)

INV2 = 1.0 / np.sqrt(2.0)
H_GATE = INV2 * np.array([[1, 1], [1, -1]], dtype=complex)


def _entangling_circuit_probs_dense(n, seed=7, layers=4):
    sim = de.DenseSVSimulator(n_qubits=n, use_gpu=False, use_float32=False)
    ops = [["h", q, -1] for q in range(n)]
    rng = np.random.default_rng(seed)
    for _ in range(layers):
        for q in range(0, n - 1, 2):
            ops.append(["cx", q + 1, q])
        for q in range(n):
            ops.append(["rz", q, float(rng.uniform(0.1, 1.5))])
        for q in range(1, n - 1, 2):
            ops.append(["cx", q + 1, q])
    sim.run_circuit_jit_beast_mode(ops)
    return np.array(sim.get_probabilities())


def _entangling_circuit_probs_mps(n, seed=7, layers=4, **mps_kwargs):
    mps = MPSSimulator(n_qubits=n, **mps_kwargs)
    for q in range(n):
        mps.apply_gate_1q(H_GATE, q)
    rng = np.random.default_rng(seed)
    for _ in range(layers):
        for q in range(0, n - 1, 2):
            mps.apply_cx(q + 1, q)
        for q in range(n):
            theta = float(rng.uniform(0.1, 1.5))
            rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
            mps.apply_gate_1q(rz, q)
        for q in range(1, n - 1, 2):
            mps.apply_cx(q + 1, q)
    sv = mps.contract_to_statevector()
    return np.abs(sv) ** 2, mps


# ── _jsd_vectors ─────────────────────────────────────────────────────────

def test_jsd_identical_distributions_is_zero():
    p = np.array([0.5, 0.3, 0.2])
    assert _jsd_vectors(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jsd_handles_zero_entries_without_warning():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.5, 0.5, 0.0])
    with np.errstate(all="raise"):
        val = _jsd_vectors(p, q)
    assert np.isfinite(val)


# ── basic gate mechanics ─────────────────────────────────────────────────

def test_bell_state():
    mps = MPSSimulator(n_qubits=2, max_bond=8)
    mps.apply_gate_1q(H_GATE, 0)
    mps.apply_cx(0, 1)
    sv = mps.contract_to_statevector()
    prob = np.abs(sv) ** 2
    assert prob[0] == pytest.approx(0.5, abs=1e-9)
    assert prob[3] == pytest.approx(0.5, abs=1e-9)
    assert prob[1] == pytest.approx(0.0, abs=1e-9)
    assert prob[2] == pytest.approx(0.0, abs=1e-9)


def test_ghz_chain_n_qubits():
    n = 6
    mps = MPSSimulator(n_qubits=n, max_bond=8)
    mps.apply_gate_1q(H_GATE, 0)
    for q in range(n - 1):
        mps.apply_cx(q, q + 1)
    sv = mps.contract_to_statevector()
    prob = np.abs(sv) ** 2
    assert prob[0] == pytest.approx(0.5, abs=1e-9)
    assert prob[-1] == pytest.approx(0.5, abs=1e-9)
    assert prob.sum() == pytest.approx(1.0, abs=1e-9)


def test_statevector_stays_normalized_after_many_gates():
    n = 5
    mps = MPSSimulator(n_qubits=n, max_bond=16)
    rng = np.random.default_rng(3)
    for q in range(n):
        mps.apply_gate_1q(H_GATE, q)
    for _ in range(3):
        for q in range(n - 1):
            mps.apply_cx(q, q + 1)
    sv = mps.contract_to_statevector()
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-9)


def test_nonlocal_2q_gate_via_swap_chain():
    n = 4
    mps = MPSSimulator(n_qubits=n, max_bond=16)
    mps.apply_gate_1q(H_GATE, 0)
    mps.apply_cx(0, 3)  # non-adjacent
    sv = mps.contract_to_statevector()
    prob = np.abs(sv) ** 2
    # H on q0 then CX(0,3): entangles q0/q3, q1/q2 stay |0>
    assert prob[0b0000] == pytest.approx(0.5, abs=1e-9)
    assert prob[0b1001] == pytest.approx(0.5, abs=1e-9)


def test_toffoli_matches_classical_truth_table():
    # |110> -controlled by q0,q1-> flips q2: expect |111>
    mps = MPSSimulator(n_qubits=3, max_bond=8)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    mps.apply_gate_1q(x, 0)
    mps.apply_gate_1q(x, 1)
    mps.apply_ccx(0, 1, 2)
    sv = mps.contract_to_statevector()
    prob = np.abs(sv) ** 2
    assert prob[0b111] == pytest.approx(1.0, abs=1e-6)


# ── regression: _apply_nonlocal_2q used to invert ctrl/tgt when q1 > q2 ──
# (found via an external code review, independently reproduced and fixed;
# apply_gate_2q's adjacent-qubit branch already normalized q1>q2 -- the
# non-adjacent SWAP-chain branch didn't, silently swapping which qubit was
# control vs target for any non-adjacent gate called with q1 > q2)

def test_nonlocal_2q_gate_ctrl_greater_than_tgt():
    n = 4
    mps = MPSSimulator(n_qubits=n, max_bond=16)
    mps.apply_gate_1q(H_GATE, 3)
    mps.apply_cx(3, 0)  # non-adjacent, ctrl(3) > tgt(0)
    sv = mps.contract_to_statevector()
    prob = np.abs(sv) ** 2
    # H on q3 then CX(ctrl=3,tgt=0): entangles q3/q0, q1/q2 stay |0>
    assert prob[0b0000] == pytest.approx(0.5, abs=1e-9)
    assert prob[0b1001] == pytest.approx(0.5, abs=1e-9)


def test_nonlocal_2q_gate_ctrl_greater_than_tgt_matches_dense_simulator():
    n = 5
    ops = [["h", 0, -1], ["h", 2, -1],
           ["cx", 4, 0],   # non-adjacent, ctrl > tgt -- the buggy case
           ["cx", 3, 1],   # non-adjacent, ctrl > tgt -- again
           ["cx", 0, 4]]   # non-adjacent, ctrl < tgt, for contrast
    sim = de.DenseSVSimulator(n_qubits=n, use_gpu=False, use_float32=False)
    sim.run_circuit_jit_beast_mode(ops)
    prob_dense = np.array(sim.get_probabilities())

    mps = MPSSimulator(n_qubits=n, max_bond=16)
    mps.apply_gate_1q(H_GATE, 0)
    mps.apply_gate_1q(H_GATE, 2)
    mps.apply_cx(4, 0)
    mps.apply_cx(3, 1)
    mps.apply_cx(0, 4)
    prob_mps = np.abs(mps.contract_to_statevector()) ** 2

    tvd = 0.5 * np.sum(np.abs(prob_dense - prob_mps))
    assert tvd < 1e-6, f"TVD={tvd} -- MPS diverged from DenseSVSimulator on ctrl>tgt non-adjacent gates"


def test_ghz_via_out_of_order_nonlocal_cnots():
    # Exact repro of the originally-reported bug: H(0), then three
    # non-adjacent CNOTs where the middle one has ctrl > tgt.
    n = 4
    mps = MPSSimulator(n_qubits=n, jsd_budget=1e-6)
    mps.apply_gate_1q(H_GATE, 0)
    mps.apply_cx(0, 3)
    mps.apply_cx(3, 1)  # non-adjacent, ctrl > tgt -- the reported bug
    mps.apply_cx(1, 2)
    prob = np.abs(mps.contract_to_statevector()) ** 2
    assert prob[0b0000] == pytest.approx(0.5, abs=1e-6)
    assert prob[0b1111] == pytest.approx(0.5, abs=1e-6)
    assert np.sum(prob) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("use_float64", [False, True])
def test_nonlocal_2q_gate_ctrl_greater_than_tgt_both_dtypes(use_float64):
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", use_float64)
    try:
        n = 4
        mps = MPSSimulator(n_qubits=n, max_bond=16)
        mps.apply_gate_1q(H_GATE, 3)
        mps.apply_cx(3, 0)
        prob = np.abs(mps.contract_to_statevector()) ** 2
        assert prob[0b0000] == pytest.approx(0.5, abs=1e-6)
        assert prob[0b1001] == pytest.approx(0.5, abs=1e-6)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_toffoli_non_adjacent_unordered_controls():
    # apply_ccx composes from apply_cx calls that aren't guaranteed
    # adjacent-and-increasing -- exercises _apply_nonlocal_2q for a real
    # 3-qubit gate, not just a raw CNOT.
    n = 4
    mps = MPSSimulator(n_qubits=n, max_bond=16)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    mps.apply_gate_1q(x, 3)
    mps.apply_gate_1q(x, 0)
    mps.apply_ccx(3, 0, 1)  # controls at 3 and 0 (non-adjacent, unordered), target 1
    prob = np.abs(mps.contract_to_statevector()) ** 2
    # controls q3=1,q0=1 -> target q1 flips 0->1; q2 stays 0 -> |1101>
    assert prob[0b1101] == pytest.approx(1.0, abs=1e-6)


# ── the actual regression check: cross-validation vs DenseSVSimulator ────

def test_matches_dense_simulator_on_entangling_circuit():
    n = 8
    prob_dense = _entangling_circuit_probs_dense(n)
    prob_mps, mps = _entangling_circuit_probs_mps(n, max_bond=64, jsd_budget=1e-5)
    tvd = 0.5 * np.sum(np.abs(prob_dense - prob_mps))
    assert tvd < 1e-9, f"TVD={tvd} -- MPS diverged from DenseSVSimulator"
    assert mps.max_bond_used() > 1, "test circuit should produce real entanglement"


def test_matches_dense_simulator_smaller_bond_still_exact_here():
    # Same circuit, tighter max_bond: still fits (bond used well under 64
    # in the unconstrained run), confirms the JSD-budget growth loop
    # converges to the same exact answer, not just when given headroom.
    n = 8
    prob_dense = _entangling_circuit_probs_dense(n)
    prob_mps, mps = _entangling_circuit_probs_mps(n, max_bond=32, jsd_budget=1e-5)
    tvd = 0.5 * np.sum(np.abs(prob_dense - prob_mps))
    assert tvd < 1e-9


# ── budget_violations / jsd_budget-not-honored signal ────────────────────

def test_budget_violations_flagged_when_max_bond_too_small():
    # max_bond=2 is deliberately far too small for this circuit's real
    # entanglement -- verified directly: TVD ~0.97 against DenseSVSimulator
    # (a badly wrong result) while avg_JSD alone read a deceptively low
    # 0.0534. budget_violations/the UserWarning are the real signal that
    # the result can't be trusted at this max_bond.
    n = 8
    prob_dense = _entangling_circuit_probs_dense(n, layers=15)
    with pytest.warns(UserWarning, match="jsd_budget.*not honored"):
        prob_mps, mps = _entangling_circuit_probs_mps(n, layers=15, max_bond=2, jsd_budget=1e-5)
    tvd = 0.5 * np.sum(np.abs(prob_dense - prob_mps))
    assert tvd > 0.5, "sanity: this max_bond should produce a badly wrong result"
    assert mps.budget_violations > 0
    assert "budget_violations=" in mps.summary()


def test_budget_violations_zero_when_max_bond_is_adequate():
    # Same circuit shape as the exact-match tests above -- max_bond=64 is
    # generous enough that jsd_budget is always satisfiable, so this must
    # NOT warn and budget_violations must stay 0 (no false positives).
    n = 8
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, mps = _entangling_circuit_probs_mps(n, max_bond=64, jsd_budget=1e-5)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []
    assert mps.budget_violations == 0


# ── large-n sampling path (no full statevector ever materialized) ────────

def test_sampled_probabilities_low_entanglement_large_n():
    # GHZ-chain at 40 qubits: bond dimension stays 2 throughout regardless
    # of n, so this must run instantly and never touch a 2**40 array.
    n = 40
    mps = MPSSimulator(n_qubits=n, max_bond=8)
    mps.apply_gate_1q(H_GATE, 0)
    for q in range(n - 1):
        mps.apply_cx(q, q + 1)
    assert mps.max_bond_used() <= 2

    dist = mps.get_probabilities_sampled(n_samples=500, seed=1)
    zeros = "0" * n
    ones = "1" * n
    total = sum(dist.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    # only these two bitstrings should ever appear for a GHZ chain
    assert set(dist.keys()) <= {zeros, ones}
    assert zeros in dist and ones in dist


def test_contract_to_statevector_raises_above_24_qubits():
    mps = MPSSimulator(n_qubits=25, max_bond=4)
    with pytest.raises(MemoryError):
        mps.contract_to_statevector()

# dashboard_core.run_simulation(..., engine='mps') integration tests moved
# out with dashboard_core/ itself -- dashboard_core lives on WIP branches
# only (feature/composer, feature/streamlit-dashboard), not main, so a
# main-only test importing it is a guaranteed ModuleNotFoundError here.

# ── get_top_k_probable_states (corrected greedy beam search) ─────────────

def test_top_k_probability_values_are_exact():
    # Whichever states the beam search finds, their reported probability
    # must match the exact contraction exactly (not approximately).
    n = 8
    prob_dense, _ = _entangling_circuit_probs_mps(n, max_bond=64, jsd_budget=1e-5)
    mps = MPSSimulator(n_qubits=n, max_bond=64, jsd_budget=1e-5)
    for q in range(n):
        mps.apply_gate_1q(H_GATE, q)
    rng = np.random.default_rng(7)
    for _ in range(4):
        for q in range(0, n - 1, 2):
            mps.apply_cx(q + 1, q)
        for q in range(n):
            theta = float(rng.uniform(0.1, 1.5))
            rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
            mps.apply_gate_1q(rz, q)
        for q in range(1, n - 1, 2):
            mps.apply_cx(q + 1, q)

    idx_k, prob_k = mps.get_top_k_probable_states(k=32)
    for i, p in zip(idx_k, prob_k):
        assert p == pytest.approx(prob_dense[i], abs=1e-9)


def test_top_k_recall_improves_with_beam_width():
    n = 10
    mps_ref = MPSSimulator(n_qubits=n, max_bond=64, jsd_budget=1e-5)
    rng = np.random.default_rng(3)
    for q in range(n):
        mps_ref.apply_gate_1q(H_GATE, q)
    for _ in range(3):
        for q in range(0, n - 1, 2):
            mps_ref.apply_cx(q + 1, q)
        for q in range(n):
            theta = float(rng.uniform(0.1, 1.5))
            rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
            mps_ref.apply_gate_1q(rz, q)
        for q in range(1, n - 1, 2):
            mps_ref.apply_cx(q + 1, q)
    sv_exact = mps_ref.contract_to_statevector()
    prob_exact = np.abs(sv_exact) ** 2
    true_top = set(np.argsort(-prob_exact)[:8].tolist())

    def build():
        m = MPSSimulator(n_qubits=n, max_bond=64, jsd_budget=1e-5)
        rng2 = np.random.default_rng(3)
        for q in range(n):
            m.apply_gate_1q(H_GATE, q)
        for _ in range(3):
            for q in range(0, n - 1, 2):
                m.apply_cx(q + 1, q)
            for q in range(n):
                theta = float(rng2.uniform(0.1, 1.5))
                rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
                m.apply_gate_1q(rz, q)
            for q in range(1, n - 1, 2):
                m.apply_cx(q + 1, q)
        return m

    idx_small, _ = build().get_top_k_probable_states(k=8)
    idx_large, _ = build().get_top_k_probable_states(k=64)
    hits_small = len(set(idx_small.tolist()[:8]) & true_top)
    hits_large = len(set(idx_large.tolist()[:8]) & true_top)
    assert hits_large >= hits_small


def test_top_k_never_touches_full_statevector_at_large_n():
    # 30 qubits: contract_to_statevector would refuse (>24). The beam
    # search must still work -- it never builds a (2**30,) array.
    n = 30
    mps = MPSSimulator(n_qubits=n, max_bond=8)
    mps.apply_gate_1q(H_GATE, 0)
    for q in range(n - 1):
        mps.apply_cx(q, q + 1)
    idx_k, prob_k = mps.get_top_k_probable_states(k=8)
    assert prob_k.sum() <= 1.0 + 1e-9
    assert len(idx_k) > 0


# ── _vectorized_chi_search: JIT-fusion groundwork (issue: MPS is ~140x ──
# slower than Qiskit Aer's MPS backend on a 60-qubit stress circuit,
# measured 88.9s vs 0.64s -- root cause: zero @jax.jit anywhere in this
# file, including a host-syncing Python `while` loop in _svd_truncate to
# search for the truncated bond dimension. _vectorized_chi_search replaces
# that loop with one vectorized pass; this test is the correctness gate
# before it's wired into _svd_truncate itself (a later step) -- it must
# reproduce the while loop's (chi_new, jsd_val) exactly, not approximately,
# across real _svd_truncate calls captured from real entangled circuits.

def _capture_real_svd_truncate_calls(n_qubits, max_bond, jsd_budget, seed, layers):
    """Runs a real entangling circuit through MPSSimulator, capturing the
    TRUE full (pre-truncation) singular-value spectrum _svd_truncate saw
    internally for every 2-qubit gate, plus what the real while loop
    decided (chi_new, jsd_val) -- so _vectorized_chi_search can be checked
    against real data, not synthetic arrays."""
    captured = []
    orig_svd_truncate = MPSSimulator._svd_truncate

    def spy(self, theta_mat):
        full_S = np.array(jnp.linalg.svd(theta_mat, full_matrices=False)[1])
        U, S, Vh, trunc_err, jsd_val = orig_svd_truncate(self, theta_mat)
        captured.append((full_S, self.eps, self.jsd_budget, self.chi, U.shape[1], jsd_val))
        return U, S, Vh, trunc_err, jsd_val

    MPSSimulator._svd_truncate = spy
    try:
        rng = np.random.default_rng(seed)
        mps = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
        cx = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=complex).reshape(2, 2, 2, 2)
        mps.apply_gate_1q(H_GATE, 0)
        for _ in range(layers):
            for i in range(n_qubits):
                theta = float(rng.uniform(0.1, 1.5))
                rx = jnp.array([[jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)],
                                 [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)]], dtype=complex)
                mps.apply_gate_1q(rx, i)
            for i in range(0, n_qubits - 1, 2):
                mps.apply_gate_2q(cx, i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                mps.apply_gate_2q(cx, i, i + 1)
    finally:
        MPSSimulator._svd_truncate = orig_svd_truncate
    return captured


@pytest.mark.parametrize("n_qubits,max_bond,jsd_budget,seed,layers", [
    (14, 32, 1e-5, 0, 6),   # the config that first validated this during design
    (10, 8,  1e-3, 1, 4),   # small max_bond -- more likely to hit budget violations
    (20, 64, 1e-6, 2, 5),   # larger, tighter budget
    (6,  4,  1e-2, 3, 8),   # tiny bond, many layers -- stresses the violation fallback
])
def test_vectorized_chi_search_matches_real_while_loop(n_qubits, max_bond, jsd_budget, seed, layers):
    captured = _capture_real_svd_truncate_calls(n_qubits, max_bond, jsd_budget, seed, layers)
    assert len(captured) > 0, "expected at least one real _svd_truncate call"
    for full_S, eps, budget, chi, real_chi_new, real_jsd_val in captured:
        vec_chi_new, vec_jsd_val = _vectorized_chi_search(jnp.array(full_S), eps, budget, chi)
        assert vec_chi_new == real_chi_new
        assert vec_jsd_val == pytest.approx(real_jsd_val, abs=1e-9)


# ── _expand_nonlocal_2q_positions: JIT-fusion groundwork, step 2 ────────
# Moves the SWAP-chain expansion _apply_nonlocal_2q does at runtime to a
# pre-compile-time Python function, so a future jax.lax.scan-based kernel
# can consume a flat, pre-expanded op list instead of dispatching SWAPs
# inside the traced region.

@pytest.mark.parametrize("n_qubits", [6, 10, 15])
def test_expand_nonlocal_2q_positions_matches_real_dispatch_sequence(n_qubits):
    """Exhaustive: every (q1, q2) pair for this n_qubits must produce the
    exact same sequence of adjacent-pair apply_gate_2q calls the real
    runtime SWAP-chain dispatch produces."""
    CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                      dtype=complex).reshape(2, 2, 2, 2)

    def real_sequence(q1, q2):
        calls = []
        orig_apply_gate_2q = MPSSimulator.apply_gate_2q

        def spy(self, gate_2q, a, b):
            calls.append((a, b))
            return orig_apply_gate_2q(self, gate_2q, a, b)

        MPSSimulator.apply_gate_2q = spy
        try:
            mps = MPSSimulator(n_qubits=n_qubits, max_bond=8)
            mps._apply_nonlocal_2q(CNOT, q1, q2)
        finally:
            MPSSimulator.apply_gate_2q = orig_apply_gate_2q
        return calls

    for q1 in range(n_qubits):
        for q2 in range(n_qubits):
            if q1 == q2:
                continue
            real_seq = real_sequence(q1, q2)
            pred_seq, gate_index, needs_transpose = _expand_nonlocal_2q_positions(q1, q2)
            assert pred_seq == real_seq, f"(q1={q1}, q2={q2}): predicted {pred_seq} != real {real_seq}"
            assert needs_transpose == (q1 > q2)


def test_expand_nonlocal_2q_positions_replay_matches_final_state():
    """End-to-end, not just index bookkeeping: replaying the predicted
    sequence by hand (SWAP at every position except gate_index, the real
    gate -- transposed if needs_transpose -- at gate_index) through the
    existing eager apply_gate_2q must produce the identical final quantum
    state as calling apply_gate_2q(gate, q1, q2) directly (which routes
    through the real runtime _apply_nonlocal_2q dispatch internally)."""
    n_qubits = 8
    swap = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                      dtype=complex).reshape(2, 2, 2, 2)
    CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                      dtype=complex).reshape(2, 2, 2, 2)

    for q1, q2 in [(0, 3), (3, 0), (2, 7), (7, 2), (1, 6)]:
        # Reference: real dispatch, via the public API exactly as used elsewhere.
        mps_real = MPSSimulator(n_qubits=n_qubits, max_bond=16)
        mps_real.apply_gate_1q(H_GATE, 0)
        mps_real.apply_gate_1q(H_GATE, 4)
        mps_real.apply_gate_2q(CNOT, q1, q2)
        sv_real = np.asarray(mps_real.contract_to_statevector())

        # Replay: predicted sequence, executed by hand through the same
        # eager apply_gate_2q primitive, no call into _apply_nonlocal_2q.
        mps_replay = MPSSimulator(n_qubits=n_qubits, max_bond=16)
        mps_replay.apply_gate_1q(H_GATE, 0)
        mps_replay.apply_gate_1q(H_GATE, 4)
        seq, gate_index, needs_transpose = _expand_nonlocal_2q_positions(q1, q2)
        gate_to_apply = jnp.transpose(CNOT, (1, 0, 3, 2)) if needs_transpose else CNOT
        for i, (a, b) in enumerate(seq):
            mps_replay.apply_gate_2q(gate_to_apply if i == gate_index else swap, a, b)
        sv_replay = np.asarray(mps_replay.contract_to_statevector())

        fidelity = np.abs(np.vdot(sv_real, sv_replay)) ** 2
        assert fidelity == pytest.approx(1.0, abs=1e-9), f"(q1={q1}, q2={q2}): fidelity={fidelity}"


# ── run_circuit_jit: JIT-fusion groundwork, step 3 ───────────────────────
# The fused whole-circuit jax.lax.scan path. Correctness bar is the same
# one this file already established for the eager path: cross-validation
# against DenseSVSimulator on real entangling circuits, not just internal
# self-consistency. The eager apply_gate_1q/apply_gate_2q/_apply_nonlocal_2q
# methods are untouched by this -- run_circuit_jit is a new, additional
# entry point.

def test_run_circuit_jit_bell_state():
    mps = MPSSimulator(n_qubits=2, max_bond=8)
    mps.run_circuit_jit([["h", 0], ["cx", 0, 1]])
    prob = np.abs(np.asarray(mps.contract_to_statevector())) ** 2
    assert prob[0b00] == pytest.approx(0.5, abs=1e-9)
    assert prob[0b11] == pytest.approx(0.5, abs=1e-9)


def test_run_circuit_jit_ghz_chain():
    n = 6
    ops = [["h", 0]] + [["cx", q, q + 1] for q in range(n - 1)]
    mps = MPSSimulator(n_qubits=n, max_bond=8)
    mps.run_circuit_jit(ops)
    prob = np.abs(np.asarray(mps.contract_to_statevector())) ** 2
    assert prob[0] == pytest.approx(0.5, abs=1e-9)
    assert prob[2 ** n - 1] == pytest.approx(0.5, abs=1e-9)


def test_run_circuit_jit_nonlocal_ctrl_greater_than_tgt_ghz():
    # The exact scenario the MPS ctrl/tgt inversion bug was found on
    # earlier this session (H, CX(0,3), CX(3,1) non-adjacent+ctrl>tgt,
    # CX(1,2)) -- must stay correct through the fused path too.
    n = 4
    mps = MPSSimulator(n_qubits=n, max_bond=16)
    mps.run_circuit_jit([["h", 0], ["cx", 0, 3], ["cx", 3, 1], ["cx", 1, 2]])
    sv = np.asarray(mps.contract_to_statevector())
    exact = np.zeros(2 ** n, dtype=complex)
    exact[0] = exact[-1] = 1 / np.sqrt(2)
    fidelity = np.abs(np.vdot(sv, exact)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_run_circuit_jit_toffoli_matches_classical_truth_table():
    mps = MPSSimulator(n_qubits=3, max_bond=8)
    mps.run_circuit_jit([["x", 0], ["x", 1], ["ccx", 0, 1, 2]])
    # MPSSimulator has no run_circuit_jit-native ccx decomposition -- use
    # the transpiler's own, same source of truth compiler.py/chunk.py use.
    prob = np.abs(np.asarray(mps.contract_to_statevector())) ** 2
    assert prob[0b111] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n_qubits,max_bond,jsd_budget,seed,layers", [
    (12, 64, 1e-6, 5, 4),   # generous max_bond -- exact match expected
    (16, 32, 1e-5, 7, 3),
])
def test_run_circuit_jit_matches_dense_simulator(n_qubits, max_bond, jsd_budget, seed, layers):
    rng = np.random.default_rng(seed)
    ops = [["h", q] for q in range(n_qubits)]
    for _ in range(layers):
        for q in range(0, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])
        for q in range(n_qubits):
            ops.append(["rz", q, float(rng.uniform(0.1, 1.5))])
        for q in range(1, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])

    sim_dense = de.DenseSVSimulator(n_qubits=n_qubits, use_gpu=False, use_float32=False)
    sim_dense.run_circuit_jit_beast_mode(ops)
    prob_dense = np.array(sim_dense.get_probabilities())

    mps = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    mps.run_circuit_jit(ops)
    prob_mps = np.abs(np.asarray(mps.contract_to_statevector())) ** 2

    tvd = 0.5 * np.sum(np.abs(prob_dense - prob_mps))
    assert tvd < 1e-6, f"TVD={tvd} -- run_circuit_jit diverged from DenseSVSimulator"


def test_run_circuit_jit_matches_eager_even_under_budget_violation():
    # max_bond=8 is deliberately too small for this circuit's real
    # entanglement (triggers real budget_violations, same as
    # test_budget_violations_flagged_when_max_bond_too_small above) --
    # DenseSVSimulator is NOT the right correctness bar here (neither
    # MPS path can match it at this max_bond, by construction, that's
    # what "budget violation" means). What must hold: the fused path's
    # lossy truncation must match the eager path's own lossy truncation
    # almost exactly -- fusion must not introduce error *beyond* what the
    # eager implementation itself already accepts at this max_bond.
    n_qubits, max_bond, jsd_budget, seed, layers = 10, 8, 1e-3, 6, 5
    rng = np.random.default_rng(seed)
    ops = [["h", q] for q in range(n_qubits)]
    for _ in range(layers):
        for q in range(0, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])
        for q in range(n_qubits):
            ops.append(["rz", q, float(rng.uniform(0.1, 1.5))])
        for q in range(1, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])

    mps_eager = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    rz_cache = {}
    for op in ops:
        if op[0] == "h":
            mps_eager.apply_gate_1q(H_GATE, op[1])
        elif op[0] == "cx":
            mps_eager.apply_cx(op[1], op[2])
        elif op[0] == "rz":
            theta = op[2]
            if theta not in rz_cache:
                rz_cache[theta] = jnp.array(
                    [[jnp.exp(-1j * theta / 2), 0], [0, jnp.exp(1j * theta / 2)]], dtype=complex)
            mps_eager.apply_gate_1q(rz_cache[theta], op[1])
    assert mps_eager.budget_violations > 0, "test setup should genuinely stress max_bond"
    sv_eager = np.asarray(mps_eager.contract_to_statevector())

    mps_fused = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    mps_fused.run_circuit_jit(ops)
    sv_fused = np.asarray(mps_fused.contract_to_statevector())

    fidelity = np.abs(np.vdot(sv_eager, sv_fused)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


# ── bookkeeping parity: JIT-fusion groundwork, step 4 ────────────────────
# entanglement_entropy/_bond_history/jsd_per_bond/truncation_errors/
# budget_violations must end up populated the same way after
# run_circuit_jit as after the equivalent eager apply_gate_1q/apply_cx
# calls -- same information, populated from jax.lax.scan's stacked
# per-step diagnostics instead of Python list.append()s inside a loop.

def test_run_circuit_jit_bookkeeping_matches_eager():
    n_qubits, max_bond, jsd_budget, seed, layers = 10, 8, 1e-3, 6, 5
    rng = np.random.default_rng(seed)
    ops = [["h", q] for q in range(n_qubits)]
    for _ in range(layers):
        for q in range(0, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])
        for q in range(n_qubits):
            ops.append(["rz", q, float(rng.uniform(0.1, 1.5))])
        for q in range(1, n_qubits - 1, 2):
            ops.append(["cx", q + 1, q])

    mps_eager = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    rz_cache = {}
    for op in ops:
        if op[0] == "h":
            mps_eager.apply_gate_1q(H_GATE, op[1])
        elif op[0] == "cx":
            mps_eager.apply_cx(op[1], op[2])
        elif op[0] == "rz":
            theta = op[2]
            if theta not in rz_cache:
                rz_cache[theta] = jnp.array(
                    [[jnp.exp(-1j * theta / 2), 0], [0, jnp.exp(1j * theta / 2)]], dtype=complex)
            mps_eager.apply_gate_1q(rz_cache[theta], op[1])

    mps_fused = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    mps_fused.run_circuit_jit(ops)

    assert mps_fused._bond_history == mps_eager._bond_history
    assert mps_fused.budget_violations == mps_eager.budget_violations
    assert len(mps_fused.jsd_per_bond) == len(mps_eager.jsd_per_bond)
    for fused_v, eager_v in zip(mps_fused.jsd_per_bond, mps_eager.jsd_per_bond):
        assert fused_v == pytest.approx(eager_v, abs=1e-9)
    for fused_v, eager_v in zip(mps_fused.truncation_errors, mps_eager.truncation_errors):
        assert fused_v == pytest.approx(eager_v, abs=1e-6)
    assert np.max(np.abs(mps_fused.entanglement_entropy - mps_eager.entanglement_entropy)) < 1e-9
    assert mps_eager.budget_violations > 0, "test setup should genuinely stress max_bond"


@pytest.mark.parametrize("use_float64", [False, True])
def test_run_circuit_jit_both_dtypes(use_float64):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", use_float64)
    try:
        mps = MPSSimulator(n_qubits=4, max_bond=8)
        mps.run_circuit_jit([["h", 0], ["cx", 0, 3], ["cx", 3, 1], ["cx", 1, 2]])
        sv = np.asarray(mps.contract_to_statevector())
        exact = np.zeros(16, dtype=complex)
        exact[0] = exact[-1] = 1 / np.sqrt(2)
        fidelity = np.abs(np.vdot(sv, exact)) ** 2
        assert fidelity == pytest.approx(1.0, abs=1e-6)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_run_circuit_jit_rejects_unknown_gate():
    mps = MPSSimulator(n_qubits=2, max_bond=4)
    with pytest.raises(ValueError):
        mps.run_circuit_jit([["not_a_real_gate", 0]])

