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

import dense_evolution as de
from dense_evolution.mps import MPSSimulator, _jsd_vectors

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

# ── dashboard integration: dc.run_simulation(..., engine='mps') ──────────

def test_dashboard_run_simulation_mps_matches_dense_bell():
    import dashboard_core as dc

    qasm_bell = ('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
                 'h q[0]; cx q[0],q[1]; measure q -> c;')

    res_dense = dc.run_simulation(
        'Custom Workspace', 'Custom Workspace', qasm_bell,
        'ideal', 0.0, 100, 42, use_float32=True, engine='dense',
    )
    res_mps = dc.run_simulation(
        'Custom Workspace', 'Custom Workspace', qasm_bell,
        'ideal', 0.0, 100, 42, use_float32=True, engine='mps',
    )
    assert np.allclose(res_dense['prob'], res_mps['prob'], atol=1e-9)
    # res['sim'] must stay a DenseSVSimulator regardless of engine, so
    # everything downstream (VQE's res['sim'] usage, memory_mb, etc.)
    # keeps working unchanged.
    assert type(res_mps['sim']).__name__ == 'DenseSVSimulator'


def test_dashboard_run_simulation_mps_matches_dense_entangling():
    import dashboard_core as dc

    n = 8
    gates = ["h q[{}];".format(i) for i in range(n)]
    gates += ["cx q[{}],q[{}];".format(i, i + 1) for i in range(0, n - 1, 2)]
    gates += ["rz(0.7) q[{}];".format(i) for i in range(n)]
    gates += ["cx q[{}],q[{}];".format(i, i + 1) for i in range(1, n - 1, 2)]
    qasm = (f'OPENQASM 2.0; include "qelib1.inc"; qreg q[{n}]; creg c[{n}]; '
            + " ".join(gates) + " measure q -> c;")

    res_dense = dc.run_simulation(
        'Custom Workspace', 'Custom Workspace', qasm,
        'ideal', 0.0, 100, 42, use_float32=False, engine='dense',
    )
    res_mps = dc.run_simulation(
        'Custom Workspace', 'Custom Workspace', qasm,
        'ideal', 0.0, 100, 42, use_float32=False, engine='mps',
    )
    tvd = 0.5 * np.sum(np.abs(res_dense['prob'] - res_mps['prob']))
    assert tvd < 1e-9


def test_dashboard_run_simulation_mps_rejects_over_24_qubits():
    import dashboard_core as dc

    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[30]; creg c[30]; h q[0]; measure q -> c;'
    with pytest.raises(ValueError, match="24 qubit"):
        dc.run_simulation(
            'Custom Workspace', 'Custom Workspace', qasm,
            'ideal', 0.0, 100, 42, use_float32=True, engine='mps',
        )

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

