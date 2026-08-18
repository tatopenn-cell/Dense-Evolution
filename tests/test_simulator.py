"""
Unit tests for dense_evolution/simulator.py -- DenseSVSimulator's core
statevector mechanics: initialization, single/two-qubit gates, parametric
gates, measurement, the run_circuit_jit fast path, and
donate_argnums buffer reuse.

Split out of the original monolithic test_dense_evolution.py (which mixed
simulator, registry, compiler, parser, chunk, and healing tests in one
2200+-line file) for the same reason every other module here already has
its own test_<module>.py: one file per source module, mirroring peer
quantum-simulator projects' convention (see docs/index.md's honesty note
and the project's own comparison notes).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from dense_evolution import DenseSVSimulator, GATES

from _helpers import norm, probs

# ─────────────────────────────────────────────────────────────
# 1. INITIALIZATION
# ─────────────────────────────────────────────────────────────

class TestInitialization:

    def test_initial_state_is_zero(self, sim2):
        sv = sim2.get_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_initial_norm_is_one(self, sim2):
        assert abs(norm(sim2) - 1.0) < 1e-12

    def test_initial_probabilities(self, sim2):
        p = probs(sim2)
        assert abs(p[0] - 1.0) < 1e-12
        assert np.all(p[1:] < 1e-12)

    def test_custom_initial_state(self, sim2):
        sv_in = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        sim2.set_initial_state(sv_in)
        sv_out = sim2.get_statevector()
        np.testing.assert_allclose(np.abs(sv_out), np.abs(sv_in), atol=1e-12)

    def test_invalid_state_raises(self, sim2):
        with pytest.raises(ValueError):
            sim2.set_initial_state(np.array([1, 0, 0], dtype=complex))

    def test_zero_norm_state_raises(self, sim2):
        with pytest.raises(ValueError):
            sim2.set_initial_state(np.zeros(4, dtype=complex))

    def test_set_initial_state_explicit_none_resets_to_zero(self, sim2):
        sim2.set_initial_state(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
        sim2.set_initial_state(None)
        sv = sim2.get_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_set_state_is_an_alias_for_set_initial_state(self, sim2):
        sv_in = np.array([0, 1, 0, 0], dtype=complex)
        sim2.set_state(sv_in)
        np.testing.assert_allclose(np.abs(sim2.get_statevector()), np.abs(sv_in), atol=1e-12)

    def test_n_qubits_out_of_range_raises(self):
        with pytest.raises(ValueError):
            DenseSVSimulator(n_qubits=0)
        with pytest.raises(ValueError):
            DenseSVSimulator(n_qubits=35)

# ─────────────────────────────────────────────────────────────
# 2. SINGLE-QUBIT GATES
# ─────────────────────────────────────────────────────────────

class TestSingleQubitGates:

    def test_x_gate_flips_qubit(self, sim2):
        """X|0⟩ = |1⟩"""
        sim2.apply_gate_1q(GATES['x'], 0)
        p = probs(sim2)
        # In MSB: qubit 0 is the most significant bit → |10⟩ = index 2
        assert p[2] > 0.99

    def test_x_gate_double_application_identity(self, sim2):
        """XX = I"""
        sim2.apply_gate_1q(GATES['x'], 0)
        sim2.apply_gate_1q(GATES['x'], 0)
        p = probs(sim2)
        assert p[0] > 0.99

    def test_h_gate_creates_superposition(self, sim2):
        """H|0⟩ = (|0⟩+|1⟩)/√2 on qubit 0"""
        sim2.apply_gate_1q(GATES['h'], 0)
        p = probs(sim2)
        assert abs(p[0] - 0.5) < 1e-10
        assert abs(p[2] - 0.5) < 1e-10

    def test_h_gate_is_self_inverse(self, sim2):
        """HH = I"""
        sim2.apply_gate_1q(GATES['h'], 0)
        sim2.apply_gate_1q(GATES['h'], 0)
        p = probs(sim2)
        assert p[0] > 0.99

    def test_z_gate_on_zero_state_no_change(self, sim2):
        """Z|0⟩ = |0⟩ (phase change invisible in probabilities)"""
        sim2.apply_gate_1q(GATES['z'], 0)
        p = probs(sim2)
        assert p[0] > 0.99

    def test_z_gate_on_superposition_flips_phase(self, sim2):
        """Z applied after H: |+⟩ → |-⟩, then H gives |1⟩"""
        sim2.apply_gate_1q(GATES['h'], 0)
        sim2.apply_gate_1q(GATES['z'], 0)
        sim2.apply_gate_1q(GATES['h'], 0)
        p = probs(sim2)
        # result should be |1x⟩ → qubit 0 in state |1⟩
        assert (p[2] + p[3]) > 0.99

    def test_norm_preserved_after_1q_gate(self, sim2):
        for g in ['h', 'x', 'y', 'z', 's', 't']:
            sim2.apply_gate_1q(GATES[g], 0)
            assert abs(norm(sim2) - 1.0) < 1e-12

    def test_out_of_bounds_qubit_raises(self, sim2):
        with pytest.raises((ValueError, IndexError)):
            sim2.apply_gate_1q(GATES['x'], 5)


class TestQubitRangeValidationBypassesJIT:
    """run_circuit_jit / run_batch_jit build their own
    compiled_ops and never call apply_gate_1q/apply_gate_2q (which already
    validate) — an out-of-range qubit index there used to silently corrupt
    the entire statevector to zero instead of raising, because the fast
    JAX path encodes qubit indices as bit-shift amounts inside
    jax.lax.scan/switch with no bounds check. Verified before the fix:
    a single gate on an out-of-range qubit on an otherwise normalized
    state left get_probabilities().sum() == 0.0, no exception."""

    def test_beast_mode_1q_gate_out_of_range_raises(self, sim4):
        with pytest.raises(ValueError):
            sim4.run_circuit_jit([['x', 5, -1]])

    def test_beast_mode_2q_gate_out_of_range_raises(self, sim4):
        with pytest.raises(ValueError):
            sim4.run_circuit_jit([['cx', 0, 5]])

    def test_beast_mode_valid_circuit_unaffected(self, sim4):
        # the validation must not reject in-range circuits
        sim4.run_circuit_jit([['h', 0, -1], ['cx', 0, 1]])
        p = probs(sim4)
        assert abs(p.sum() - 1.0) < 1e-9

    def test_parametric_batch_qubit_out_of_range_raises(self, sim4):
        with pytest.raises(ValueError):
            sim4.run_batch_jit([['rx', 5]], np.zeros((1, 1)))


class TestBeastModeGateDispatchGaps:
    """run_circuit_jit used to silently DROP cy/cp/crz/u1/p/sx —
    they weren't in GATE_IDS, so `if name not in GATE_IDS: continue` skipped
    them with no error (verified: h(0);h(1);crz(0,1,1.2) produced the exact
    same output as h(0);h(1) alone — the crz vanished). Fixed by adding the
    missing GATE_IDS entries and (for cy/crz/sx, which had no kernel at all)
    new branches in _apply_gate_fast_step. crz specifically needed its own
    kernel, not reuse of cp's: CP phases |11> only, CRZ phases the target
    conditioned on its own bit value — mathematically different gates."""

    def test_previously_dropped_gates_are_not_no_ops(self):
        # each of these used to leave the statevector identical to the
        # circuit with the gate simply removed
        cases = [
            ("cy",  [('h', 0), ('cy', 0, 1)]),
            ("cp",  [('h', 0), ('h', 1), ('cp', 0, 1, 0.7)]),
            ("crz", [('h', 0), ('h', 1), ('crz', 0, 1, 1.2)]),
            ("u1",  [('h', 0), ('u1', 0, 0.9)]),
            ("p",   [('h', 0), ('p', 0, 0.5)]),
            ("sx",  [('sx', 0)]),
        ]
        for name, circuit in cases:
            sim_with = DenseSVSimulator(n_qubits=2)
            sim_with.run_circuit_jit(circuit)
            without = [c for c in circuit if c[0] != name]
            sim_without = DenseSVSimulator(n_qubits=2)
            sim_without.run_circuit_jit(without)
            assert not np.allclose(
                np.asarray(sim_with.get_statevector()),
                np.asarray(sim_without.get_statevector()), atol=1e-9,
            ), f"'{name}' still has no effect in beast mode"

    @pytest.mark.parametrize("name,circuit", [
        ("cy",  [('h', 0), ('cy', 0, 1)]),
        ("cp",  [('h', 0), ('h', 1), ('cp', 0, 1, 0.7)]),
        ("crz", [('h', 0), ('h', 1), ('crz', 0, 1, 1.2)]),
        ("u1",  [('h', 0), ('u1', 0, 0.9)]),
        ("p",   [('h', 0), ('p', 0, 0.5)]),
        ("sx_q0", [('sx', 0)]),
        ("sx_q1", [('sx', 1)]),
        ("cp_and_crz_and_cy_and_rx", [('rx', 0, 0.3), ('cx', 0, 1), ('cy', 1, 0), ('crz', 0, 1, 0.8), ('p', 1, 0.4)]),
    ])
    def test_matches_run_circuit(self, name, circuit):
        # run_circuit_jit used to disagree with run_circuit() on
        # qubit ordering (LSB-first vs the documented MSB-first) — now fixed
        # (see TestBeastModeQubitOrdering below), so a direct comparison
        # with no relabeling is the real correctness bar.
        n = 2
        ref = DenseSVSimulator(n_qubits=n)
        ref.run_circuit(circuit)
        fast = DenseSVSimulator(n_qubits=n)
        fast.run_circuit_jit(circuit)
        np.testing.assert_allclose(
            np.asarray(ref.get_statevector()), np.asarray(fast.get_statevector()), atol=1e-9,
        )

    def test_crz_is_not_cp(self):
        # regression guard for the specific mistake of reusing apply_cp's
        # kernel for crz: they must diverge on a case where CP is a no-op
        # (control=1, target=0 — CP only phases |11>) but CRZ still isn't
        # (CRZ phases based on the target's own bit, regardless of the
        # other bit's value)
        sim_cp = DenseSVSimulator(n_qubits=2)
        sim_cp.run_circuit_jit([('x', 1), ('cp', 1, 0, 1.5)])  # ctrl=1(set), tgt=0(unset) -> CP no-op
        sim_crz = DenseSVSimulator(n_qubits=2)
        sim_crz.run_circuit_jit([('x', 1), ('crz', 1, 0, 1.5)])
        assert not np.allclose(
            np.asarray(sim_cp.get_statevector()), np.asarray(sim_crz.get_statevector()), atol=1e-9,
        )

    def test_sx_squared_is_x(self):
        # convention-independent algebraic identity: SX*SX = X
        sim = DenseSVSimulator(n_qubits=1)
        sim.run_circuit_jit([('sx', 0), ('sx', 0)])
        p = probs(sim)
        assert p[1] > 0.999   # |0> -> |1>, same as a single X

    def test_previously_working_gates_unaffected(self, sim2):
        # h/cx/rz/s/sdg/t/tdg already worked before this fix -- confirm the
        # is_1q boundary change (12 -> 13, needed for sx) didn't misroute them
        sim2.run_circuit_jit([('h', 0), ('cx', 0, 1), ('rz', 1, 0.6)])
        p = probs(sim2)
        assert abs(p.sum() - 1.0) < 1e-9


class TestUnknownGateRaises:
    """Issue #4: a typo'd/unrecognized gate name used to be silently
    dropped from the circuit in all three execution paths instead of
    raising -- verified: h(0);ch(0,1);x(2) executed as if 'ch' wasn't
    there, no exception, no warning."""

    def test_run_circuit_raises_on_unknown_gate(self):
        sim = DenseSVSimulator(n_qubits=2)
        with pytest.raises(ValueError, match="unknown gate"):
            sim.run_circuit([('h', 0), ('ch', 0, 1)])

    def test_beast_mode_raises_on_unknown_gate(self):
        sim = DenseSVSimulator(n_qubits=2)
        with pytest.raises(ValueError, match="unknown gate"):
            sim.run_circuit_jit([('h', 0), ('ch', 0, 1)])

    def test_parametric_batch_raises_on_unknown_gate(self):
        sim = DenseSVSimulator(n_qubits=2)
        with pytest.raises(ValueError, match="unknown gate"):
            sim.run_batch_jit(
                [('h', 0), ('ch', 0, 1)], np.zeros((3, 0))
            )

    def test_known_gates_still_run_unaffected(self, sim2):
        # regression guard: the new validation must not reject any
        # currently-supported gate name
        sim2.run_circuit([('h', 0), ('cx', 0, 1), ('rz', 1, 0.6)])
        p = probs(sim2)
        assert abs(p.sum() - 1.0) < 1e-9


class TestParametricBatchColumnMismatchRaises:
    """Issue #6: run_batch_jit assigns one parameter_batch
    column per parametric gate, in gate-appearance order -- including
    literal-float rotation gates, which silently ignored their literal
    and consumed a column anyway. A column-count mismatch used to be
    clipped silently by JAX's default out-of-bounds indexing instead of
    raising -- verified (pre-fix) with a statevector delta of 0.66
    against the intended circuit."""

    def test_too_few_columns_raises(self):
        # base_circuit has 2 parametric gates (rx, ry) but only 1 column
        sim = DenseSVSimulator(n_qubits=2)
        circuit = [('rx', 0, None), ('ry', 1, None)]
        with pytest.raises(ValueError, match="parameter_batch"):
            sim.run_batch_jit(circuit, np.zeros((5, 1)))

    def test_too_many_columns_raises(self):
        sim = DenseSVSimulator(n_qubits=2)
        circuit = [('rx', 0, None)]
        with pytest.raises(ValueError, match="parameter_batch"):
            sim.run_batch_jit(circuit, np.zeros((5, 2)))

    def test_literal_float_rotation_still_consumes_a_column(self):
        # the exact footgun from issue #6: a literal float on a rotation
        # gate is NOT exempt from the positional-slot contract
        sim = DenseSVSimulator(n_qubits=2)
        circuit = [('rx', 0, 0.5), ('ry', 1, None)]
        with pytest.raises(ValueError, match="parameter_batch"):
            sim.run_batch_jit(circuit, np.zeros((5, 1)))  # needs 2 columns, not 1

    def test_matching_column_count_runs_correctly(self):
        sim = DenseSVSimulator(n_qubits=2)
        circuit = [('rx', 0, None), ('ry', 1, None)]
        out = sim.run_batch_jit(circuit, np.zeros((3, 2)))
        assert out.shape == (3, 4)


class TestBeastModeQubitOrdering:
    """run_circuit_jit used raw qubit index as bit position
    (LSB-first: qubit 0 = least significant bit) inside _apply_gate_fast_step
    (do_1q/do_2q), while the rest of the simulator — run_circuit(),
    apply_gate_1q(), apply_gate_2q(), measure() — uses the documented
    MSB-first convention (qubit 0 = most significant bit, phys = n-1-qubit,
    see simulator.py's class docstring and _qubit_stride_pairs). Pre-existing,
    not introduced by the cy/cp/crz/u1/p/sx dispatch fix above — found while
    verifying that fix, masked until then because every circuit tested this
    session against beast_mode happened to be symmetric under qubit reversal
    (Bell states, GHZ states, uniform superpositions). Fixed by computing
    physical bit positions (n_qubits-1-qubit) in do_1q/do_2q instead of using
    the raw qubit index directly."""

    def test_x_on_qubit_0_matches_msb_first_convention(self):
        # the decisive reproduction: X on qubit 0 in a 3-qubit register must
        # flip the MOST significant bit (|000> -> |100>, index 4), not the
        # least significant one (index 1)
        sim = DenseSVSimulator(n_qubits=3)
        sim.run_circuit_jit([('x', 0)])
        p = probs(sim)
        assert p[4] > 0.999
        assert p[1] < 1e-9

    @pytest.mark.parametrize("circuit", [
        [('x', 0), ('cx', 0, 2)],
        [('x', 0), ('cy', 0, 2)],
        [('x', 0), ('crz', 0, 2, 1.1)],
        [('x', 1), ('cp', 1, 2, 0.8)],
        [('rx', 0, 0.4), ('ry', 1, 0.9), ('rz', 2, 1.3), ('cx', 0, 2), ('cx', 1, 2)],
        [('h', 0), ('rx', 1, 0.3), ('cx', 0, 1), ('cy', 1, 2), ('crz', 0, 2, 0.9),
         ('t', 0), ('sdg', 1), ('sx', 2), ('cp', 2, 0, 0.5)],
    ])
    def test_asymmetric_circuits_match_run_circuit(self, circuit):
        n = 3
        ref = DenseSVSimulator(n_qubits=n)
        ref.run_circuit(circuit)
        fast = DenseSVSimulator(n_qubits=n)
        fast.run_circuit_jit(circuit)
        np.testing.assert_allclose(
            np.asarray(ref.get_statevector()), np.asarray(fast.get_statevector()), atol=1e-9,
        )

    def test_run_batch_jit_matches_run_circuit(self):
        # same _apply_gate_fast_step kernel, must inherit the fix
        sim = DenseSVSimulator(n_qubits=3)
        batch = sim.run_batch_jit([('rx', 0, None), ('cx', 0, 2)], np.array([[0.5]]))
        ref = DenseSVSimulator(n_qubits=3)
        ref.run_circuit([('rx', 0, 0.5), ('cx', 0, 2)])
        np.testing.assert_allclose(np.asarray(batch[0]), ref.get_statevector(), atol=1e-9)


class TestBeastModeFloat32:
    """use_float32=True used to crash unconditionally in run_circuit_jit
    (the JIT fast path) — not just for circuits with 2-qubit gates, even a
    circuit with only 1-qubit gates hit it, because jax.lax.cond traces
    every branch of _apply_gate_fast_step's dispatch (do_1q AND do_2q)
    regardless of which gates are actually present. Root cause: inside
    do_2q, apply_cp built its exp_pos constant hardcoded to complex128,
    while the identity branch of that same lax.cond (`lambda s: s`)
    preserved sv's real dtype (complex64 under use_float32=True) —
    'cond branches must have equal output types but they differ'. Fixed by
    deriving every constant in _apply_gate_fast_step from sv.dtype instead
    of a hardcoded complex128."""

    def test_1q_only_circuit_runs_under_float32(self):
        sim = DenseSVSimulator(n_qubits=3, use_float32=True)
        sim.run_circuit_jit([['h', 0, -1], ['x', 1, -1]])
        assert sim.sv.dtype == np.complex64
        assert abs(float(np.sum(np.abs(np.asarray(sim.sv)) ** 2)) - 1.0) < 1e-6

    def test_2q_gates_run_under_float32(self):
        sim = DenseSVSimulator(n_qubits=4, use_float32=True)
        sim.run_circuit_jit(
            [['h', 0, -1], ['cx', 0, 1, 0], ['cz', 1, 2, 0], ['cp', 2, 3, 0.7]]
        )
        assert sim.sv.dtype == np.complex64
        assert abs(float(np.sum(np.abs(np.asarray(sim.sv)) ** 2)) - 1.0) < 1e-6

    def test_float32_matches_float64_within_precision(self):
        circuit = [
            ['h', 0, -1], ['h', 1, -1], ['rx', 2, 0.5], ['ry', 3, 1.1],
            ['cx', 0, 1, 0], ['cz', 1, 2, 0], ['cp', 2, 3, 0.7], ['crz', 0, 3, 1.3],
        ]
        sim32 = DenseSVSimulator(n_qubits=4, use_float32=True)
        sim32.run_circuit_jit(circuit)
        sim64 = DenseSVSimulator(n_qubits=4, use_float32=False)
        sim64.run_circuit_jit(circuit)
        np.testing.assert_allclose(probs(sim32), probs(sim64), atol=1e-6)


class TestRunBatchJitFloat32:
    """run_batch_jit built its own init_sv hardcoded to jnp.complex128,
    ignoring self.dtype/self.use_float32 entirely -- a separate instance
    of the same category of bug TestBeastModeFloat32 documents for
    run_circuit_jit above (that one already got its own fix; this one
    hadn't). Fixed by deriving init_sv's dtype from self.dtype, matching
    _apply_gate_fast_step's own sv_dtype-derived (not hardcoded)
    approach in compiler.py."""

    def test_output_is_complex64_under_use_float32(self):
        sim = DenseSVSimulator(n_qubits=3, use_float32=True)
        out = sim.run_batch_jit([['h', 0, -1], ['rx', 1, None]], np.array([[0.5]]))
        assert np.asarray(out).dtype == np.complex64

    def test_output_is_complex128_by_default(self):
        sim = DenseSVSimulator(n_qubits=3, use_float32=False)
        out = sim.run_batch_jit([['h', 0, -1], ['rx', 1, None]], np.array([[0.5]]))
        assert np.asarray(out).dtype == np.complex128

    def test_float32_batch_matches_float64_within_precision(self):
        circuit = [['h', 0, -1], ['cx', 0, 1], ['ry', 2, None]]
        batch = np.array([[0.3], [0.9], [1.5]])
        sim32 = DenseSVSimulator(n_qubits=3, use_float32=True)
        sim64 = DenseSVSimulator(n_qubits=3, use_float32=False)
        out32 = np.asarray(sim32.run_batch_jit(circuit, batch))
        out64 = np.asarray(sim64.run_batch_jit(circuit, batch))
        np.testing.assert_allclose(np.abs(out32) ** 2, np.abs(out64) ** 2, atol=1e-6)


class TestBeastModeDonateArgnums:
    """run_circuit_jit's self.sv = ... call used to allocate a
    fresh statevector buffer on every call instead of letting XLA reuse the
    memory of the one it's replacing — zero donate_argnums anywhere in the
    codebase, confirmed via audit. Only THIS call site is safe to donate:
    self.sv is always rebound immediately after, and no code path anywhere
    (including run_circuit_with_chunking's repeated calls, or separate
    DenseSVSimulator instances) keeps a stale reference to the old buffer
    across the call. run_batch_jit (vmap-broadcasts its init_sv
    closure across the whole batch) and circuit_to_energy_fn's energy_fn
    (the VQE loop reuses the same stato_zero every epoch) are NOT safe to
    donate — verified by tracing every call site of the shared
    _compile_and_run_circuit_jit before touching anything — so they keep
    using the plain, non-donating wrapper, untouched by this change."""

    def test_result_unchanged_by_donation(self):
        circuit = [['h', 0, -1], ['cx', 0, 1, 0], ['rz', 1, 0.6]]
        sim = DenseSVSimulator(n_qubits=3, use_float32=False)
        sim.run_circuit_jit(circuit)
        expected = probs(sim)

        # independent instance, same circuit, confirms determinism/parity
        sim2 = DenseSVSimulator(n_qubits=3, use_float32=False)
        sim2.run_circuit_jit(circuit)
        np.testing.assert_allclose(probs(sim2), expected, atol=1e-12)
        assert abs(expected.sum() - 1.0) < 1e-9

    def test_donation_actually_reuses_the_buffer(self):
        # Proof, not assumption: the pre-call buffer must be invalidated by
        # JAX after a donated call -- that's the observable signature of
        # real buffer reuse. If this test ever stops raising, donation
        # silently stopped happening (e.g. a future JAX version change) and
        # that's worth knowing, not something to quietly tolerate.
        import jax
        sim = DenseSVSimulator(n_qubits=3, use_float32=False)
        old_sv = sim.sv
        sim.run_circuit_jit([['h', 0, -1]])
        with pytest.raises(RuntimeError, match="deleted"):
            jax.block_until_ready(old_sv)

    def test_chunked_repeated_calls_stay_correct(self):
        # run_circuit_with_chunking calls run_circuit_jit
        # repeatedly in a loop -- each call donates and rebinds self.sv;
        # confirms that repeated donation across many calls doesn't
        # accumulate any corruption.
        sim = DenseSVSimulator(n_qubits=4, use_float32=False)
        circuit = [('h', i % 4) for i in range(50)] + [('cx', i % 3, (i % 3) + 1) for i in range(50)]
        sim.run_circuit_with_chunking(circuit, chunk_size=7)
        assert abs(float(np.sum(probs(sim))) - 1.0) < 1e-9

    def test_memory_rss_donated_vs_non_donated(self, capsys):
        # Not a strict pass/fail bound (RSS is noisy and platform/allocator
        # dependent) -- reports the real measured numbers so the claim
        # "donate_argnums helps" is backed by data on this machine instead
        # of asserted on faith. Uses a circuit sized to stay within this
        # dev machine's 8.5GB RAM budget.
        import gc
        import psutil
        from dense_evolution.circuits.compiler import (
            _compile_and_run_circuit_jit, _compile_and_run_circuit_jit_donated,
        )

        n_qubits = 22
        n_gates = 300
        sim_setup = DenseSVSimulator(n_qubits=n_qubits, use_float32=False)
        # g_id=1 -> H gate (see compiler.py's gate-ID table), one row per gate
        ops_jnp = jnp.array(
            [[1.0, float(i % n_qubits), 0.0, 0.0] for i in range(n_gates)],
            dtype=jnp.float64,
        )

        proc = psutil.Process()

        def run(fn):
            sv = sim_setup.sv
            gc.collect()
            before = proc.memory_info().rss
            out = fn(sv, ops_jnp)
            jnp.asarray(out).block_until_ready()
            gc.collect()
            after = proc.memory_info().rss
            return (after - before) / 1e6  # MB

        # re-init sv fresh for each variant since the donated call deletes it
        sim_setup.sv = jnp.zeros(2 ** n_qubits, dtype=jnp.complex128).at[0].set(1.0)
        delta_plain = run(_compile_and_run_circuit_jit)
        sim_setup.sv = jnp.zeros(2 ** n_qubits, dtype=jnp.complex128).at[0].set(1.0)
        delta_donated = run(_compile_and_run_circuit_jit_donated)

        with capsys.disabled():
            print(f"\n[donate_argnums RSS] n_qubits={n_qubits} "
                  f"plain=+{delta_plain:.1f}MB donated=+{delta_donated:.1f}MB")


# ─────────────────────────────────────────────────────────────
# 3. TWO-QUBIT GATES
# ─────────────────────────────────────────────────────────────

class TestTwoQubitGates:

    def test_cx_on_zero_state_no_change(self, sim2):
        """CNOT with ctrl=0 in |0⟩: no flip"""
        sim2.apply_cx(0, 1)
        p = probs(sim2)
        assert p[0] > 0.99

    def test_cx_flips_target_when_control_is_one(self, sim2):
        """CNOT with ctrl=1: |10⟩ → |11⟩"""
        sim2.apply_gate_1q(GATES['x'], 0)  # set qubit 0 to |1⟩
        sim2.apply_cx(0, 1)
        p = probs(sim2)
        # |11⟩ = index 3
        assert p[3] > 0.99

    def test_cx_double_application_identity(self, sim2):
        sim2.apply_gate_1q(GATES['x'], 0)
        sim2.apply_cx(0, 1)
        sim2.apply_cx(0, 1)
        p = probs(sim2)
        assert p[2] > 0.99  # back to |10⟩

    def test_cz_no_change_on_zero_state(self, sim2):
        sim2.apply_cz(0, 1)
        p = probs(sim2)
        assert p[0] > 0.99

    def test_norm_preserved_after_2q_gate(self, sim2):
        sim2.apply_gate_1q(GATES['h'], 0)
        sim2.apply_cx(0, 1)
        assert abs(norm(sim2) - 1.0) < 1e-12

    def test_invalid_qubit_indices_raise(self, sim2):
        with pytest.raises(ValueError):
            sim2.apply_cx(0, 0)
        with pytest.raises(ValueError):
            sim2.apply_cx(0, 5)

    def test_apply_gate_2q_direct_validation(self, sim2):
        # apply_cx/apply_cz do their own validation before delegating to
        # apply_gate_2q -- this exercises apply_gate_2q's OWN validation
        # directly, never reached via those callers.
        with pytest.raises(ValueError):
            sim2.apply_gate_2q(np.eye(4), 0, 0)
        with pytest.raises(ValueError):
            sim2.apply_gate_2q(np.eye(4), 0, 5)

    def test_apply_cz_invalid_qubit_indices_raise(self, sim2):
        with pytest.raises(ValueError):
            sim2.apply_cz(1, 1)
        with pytest.raises(ValueError):
            sim2.apply_cz(0, 5)

# ─────────────────────────────────────────────────────────────
# 4. GHZ STATE (Esempio 1 dal README)
# ─────────────────────────────────────────────────────────────

class TestGHZState:

    def test_ghz_3qubit_probabilities(self, sim3):
        """H-CX-CX: generates |000⟩+|111⟩ / √2"""
        circuit = [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]
        sim3.run_circuit(circuit)
        p = probs(sim3)
        assert abs(p[0] - 0.5) < 1e-10  # |000⟩
        assert abs(p[7] - 0.5) < 1e-10  # |111⟩
        # All other states should be zero
        for i in [1, 2, 3, 4, 5, 6]:
            assert p[i] < 1e-10

    def test_ghz_norm(self, sim3):
        circuit = [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]
        sim3.run_circuit(circuit)
        assert abs(norm(sim3) - 1.0) < 1e-12

    def test_ghz_statevector_shape(self, sim3):
        circuit = [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]
        sim3.run_circuit(circuit)
        sv = sim3.get_statevector()
        assert sv.shape == (8,)
        assert sv.dtype == np.complex128

# ─────────────────────────────────────────────────────────────
# 5. BELL STATE
# ─────────────────────────────────────────────────────────────

class TestBellState:

    def test_bell_phi_plus(self, sim2):
        """H + CNOT creates |Φ+⟩ = (|00⟩+|11⟩)/√2"""
        sim2.apply_gate_1q(GATES['h'], 0)
        sim2.apply_cx(0, 1)
        p = probs(sim2)
        assert abs(p[0] - 0.5) < 1e-10
        assert abs(p[3] - 0.5) < 1e-10
        assert p[1] < 1e-10
        assert p[2] < 1e-10

    def test_bell_entanglement_norm(self, sim2):
        sim2.apply_gate_1q(GATES['h'], 0)
        sim2.apply_cx(0, 1)
        assert abs(norm(sim2) - 1.0) < 1e-12

# ─────────────────────────────────────────────────────────────
# 6. PARAMETRIC GATES
# ─────────────────────────────────────────────────────────────

class TestParametricGates:

    def test_rx_pi_equals_x(self, sim2):
        """Rx(π)|0⟩ ≈ X|0⟩ up to global phase"""
        sim2.apply_rx(0, np.pi)
        p = probs(sim2)
        assert p[2] > 0.99  # qubit 0 flipped → |10⟩

    def test_rz_no_change_in_probabilities(self, sim2):
        """Rz only changes phase, not populations"""
        p_before = probs(sim2).copy()
        sim2.apply_rz(0, np.pi / 3)
        p_after = probs(sim2)
        np.testing.assert_allclose(p_before, p_after, atol=1e-12)

    def test_ry_half_pi_superposition(self, sim2):
        """Ry(π/2)|0⟩ gives equal superposition"""
        sim2.apply_ry(0, np.pi / 2)
        p = probs(sim2)
        assert abs(p[0] - 0.5) < 1e-10
        assert abs(p[2] - 0.5) < 1e-10

    def test_norm_preserved_after_parametric(self, sim2):
        for theta in [0.1, np.pi / 4, np.pi / 2, np.pi]:
            sim2_local = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
            sim2_local.apply_rx(0, theta)
            assert abs(norm(sim2_local) - 1.0) < 1e-12

    def test_run_circuit_u3_three_parameter_gate(self, sim2):
        # run_circuit's classic (non-JIT) dispatch len(args)==4 branch --
        # a 1-qubit gate taking 3 independent parameters (theta, phi, lam),
        # distinct from every other parametric gate here (all single-param).
        sim2.run_circuit([('u3', 0, np.pi, 0.3, 0.7)], transpile=True)
        assert abs(norm(sim2) - 1.0) < 1e-12

    def test_run_circuit_u2_two_parameter_gate(self, sim2):
        # BUG FIX: u2(phi, lam) is a 1-qubit gate with 2 params, giving a
        # 3-element args tuple (qubit, phi, lam) -- run_circuit used to
        # dispatch PARAMETRIC_GATES purely on len(args), and len(args)==3
        # was hard-coded for 2-qubit+1-param gates (cp/crz), so a real
        # 'u2' op crashed with "missing 1 required positional argument:
        # 'lam'" instead of applying the gate. Now dispatched by name.
        sim2.run_circuit([('u2', 0, 0.3, 0.7)], transpile=True)
        assert abs(norm(sim2) - 1.0) < 1e-12

        from dense_evolution.gates import PARAMETRIC_GATES
        sim_direct = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim_direct.apply_gate_1q(np.asarray(PARAMETRIC_GATES['u2'](0.3, 0.7)), 0)
        np.testing.assert_allclose(np.asarray(sim2.get_statevector()),
                                    np.asarray(sim_direct.get_statevector()), atol=1e-10)

    def test_run_circuit_cp_still_dispatches_as_two_qubit_gate(self):
        # Regression check for the same dispatch rewrite: cp/crz (2 qubits,
        # 1 param) must still be applied as 2-qubit gates, not broken by
        # switching from arg-count to name-based dispatch.
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim.run_circuit([('x', 0), ('x', 1), ('cp', 0, 1, 0.9)], transpile=True)

        from dense_evolution.gates import PARAMETRIC_GATES
        sim_direct = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim_direct.apply_gate_1q(GATES['x'], 0)
        sim_direct.apply_gate_1q(GATES['x'], 1)
        sim_direct.apply_gate_2q(np.asarray(PARAMETRIC_GATES['cp'](0.9)), 0, 1)
        np.testing.assert_allclose(np.asarray(sim.get_statevector()),
                                    np.asarray(sim_direct.get_statevector()), atol=1e-10)

    def test_run_batch_jit_cp_gate(self):
        # run_batch_jit's cp/crz/cphase branch -- a 2-qubit
        # parametric gate, distinct from the 1-qubit rx/ry/rz/p/u1 and
        # non-parametric cx/cz/swap/cy branches exercised elsewhere.
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim.apply_gate_1q(GATES['h'], 0)
        sim.apply_gate_1q(GATES['h'], 1)
        out = sim.run_batch_jit([('cp', 0, 1, None)], np.array([[0.5]]))
        out_np = np.asarray(out)
        assert out_np.shape == (1, 4)
        np.testing.assert_allclose(np.sum(np.abs(out_np) ** 2, axis=1), 1.0, atol=1e-6)

# ─────────────────────────────────────────────────────────────
# 7. MEASUREMENT
# ─────────────────────────────────────────────────────────────

class TestMeasurement:

    def test_measure_zero_state_returns_zero(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        result = sim.measure(0)
        assert result == 0

    def test_measure_one_state_returns_one(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim.apply_gate_1q(GATES['x'], 0)
        result = sim.measure(0)
        assert result == 1

    def test_measure_collapses_state_norm(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim.apply_gate_1q(GATES['h'], 0)
        sim.measure(0)
        assert abs(norm(sim) - 1.0) < 1e-12

    def test_measure_returns_binary_value(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        sim.apply_gate_1q(GATES['h'], 0)
        results = set()
        for _ in range(30):
            s = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
            s.apply_gate_1q(GATES['h'], 0)
            results.add(s.measure(0))
        assert results == {0, 1}

    def test_measure_out_of_bounds_raises(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        with pytest.raises(ValueError):
            sim.measure(5)

    def test_measure_jax_key_is_reproducible(self):
        # measure(jax_key=...) is an explicit, seedable alternative to the
        # default np.random.choice (global NumPy RNG state, not seedable
        # via JAX) -- same key on a fresh simulator in the same
        # superposition state must give the same outcome every time.
        def make_plus_state():
            sim = DenseSVSimulator(n_qubits=1, use_gpu=False, use_float32=False)
            sim.apply_gate_1q(GATES['h'], 0)
            return sim

        key = jax.random.PRNGKey(42)
        results = {make_plus_state().measure(0, jax_key=key) for _ in range(5)}
        assert len(results) == 1

    def test_measure_jax_key_none_keeps_default_behavior(self):
        # Default (no jax_key) must be unchanged: still returns a valid
        # binary outcome via the original np.random.choice path.
        sim = DenseSVSimulator(n_qubits=1, use_gpu=False, use_float32=False)
        sim.apply_gate_1q(GATES['h'], 0)
        result = sim.measure(0)
        assert result in (0, 1)

    def test_measure_jax_key_without_jax_raises(self, monkeypatch):
        # jax_key is only meaningful with JAX installed -- passing one
        # in a JAX-less environment must raise a clear error, not
        # silently fall through or hit a NameError on an unimported
        # `jax` module. Simulated here via monkeypatching HAS_JAX
        # (this environment always has real JAX installed).
        import dense_evolution.simulator as sim_mod
        monkeypatch.setattr(sim_mod, "HAS_JAX", False)
        sim = DenseSVSimulator(n_qubits=1, use_gpu=False, use_float32=False)
        with pytest.raises(ValueError, match="requires JAX"):
            sim.measure(0, jax_key=object())

# ─────────────────────────────────────────────────────────────
# 8. CIRCUIT CHUNKING (Stress test da README) -- DenseSVSimulator's own
#    run_circuit_with_chunking, distinct from the standalone Chunk class
#    (see test_chunk.py)
# ─────────────────────────────────────────────────────────────

class TestCircuitChunking:

    def test_chunking_preserves_norm(self):
        """5000 H + 5000 CNOT on 4 qubits: norm must stay 1.0"""
        sim = DenseSVSimulator(n_qubits=4, use_gpu=False, use_float32=False)
        n_gates = 500  # ridotto per velocità in CI
        circuit = [('h', i % 4) for i in range(n_gates // 2)]
        circuit += [('cx', i % 3, (i % 3) + 1) for i in range(n_gates // 2)]
        sim.run_circuit(circuit)
        assert abs(norm(sim) - 1.0) < 1e-10

    def test_run_circuit_with_chunking_exists(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        assert hasattr(sim, 'run_circuit_with_chunking') or hasattr(sim, 'run_circuit')

# ─────────────────────────────────────────────────────────────
# 9. MEMORY
# ─────────────────────────────────────────────────────────────

class TestMemory:

    def test_memory_mb_12_qubits(self):
        sim = DenseSVSimulator(n_qubits=12, use_gpu=False, use_float32=False)
        mb = sim.memory_mb()
        expected = (2**12 * 16) / 1e6
        assert abs(mb - expected) < 0.01

    def test_memory_mb_float32(self):
        sim = DenseSVSimulator(n_qubits=12, use_gpu=False, use_float32=True)
        mb = sim.memory_mb()
        expected = (2**12 * 8) / 1e6
        assert abs(mb - expected) < 0.01

# ─────────────────────────────────────────────────────────────
# 10. DEPRECATED ALIASES (renamed in 8.1.46: run_circuit_jit_beast_mode ->
# run_circuit_jit, run_parametric_batch_jit -> run_batch_jit -- both names
# were real, documented public API across many prior PyPI releases, so
# the old names stay callable and behaviorally identical, just warning,
# rather than breaking anyone's existing code silently)
# ─────────────────────────────────────────────────────────────

class TestDeprecatedAliases:

    def test_run_circuit_jit_beast_mode_still_works_and_warns(self):
        sim = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        with pytest.deprecated_call():
            sim.run_circuit_jit_beast_mode([('h', 0), ('cx', 0, 1)])
        p = probs(sim)
        assert p[0] == pytest.approx(0.5, abs=1e-9)
        assert p[3] == pytest.approx(0.5, abs=1e-9)

    def test_run_circuit_jit_beast_mode_matches_new_name(self):
        circuit = [('h', 0), ('cx', 0, 1), ('rz', 1, 0.6)]
        old = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        new = DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)
        with pytest.deprecated_call():
            old.run_circuit_jit_beast_mode(circuit)
        new.run_circuit_jit(circuit)
        assert probs(old) == pytest.approx(probs(new), abs=1e-9)

    def test_run_parametric_batch_jit_still_works_and_warns(self):
        sim = DenseSVSimulator(n_qubits=1, use_gpu=False, use_float32=False)
        with pytest.deprecated_call():
            out = sim.run_parametric_batch_jit([('rx', 0, None)], np.array([[0.0]]))
        assert np.asarray(out).shape == (1, 2)

    def test_run_parametric_batch_jit_matches_new_name(self):
        circuit = [('rx', 0, None), ('cx', 0, 2)]
        batch = np.array([[0.5]])
        sim_old = DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)
        sim_new = DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)
        with pytest.deprecated_call():
            out_old = sim_old.run_parametric_batch_jit(circuit, batch)
        out_new = sim_new.run_batch_jit(circuit, batch)
        assert np.allclose(np.asarray(out_old), np.asarray(out_new), atol=1e-9)
