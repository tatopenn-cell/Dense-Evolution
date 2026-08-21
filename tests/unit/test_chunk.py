"""
Unit tests for dense_evolution/chunk.py -- the Chunk anti-OOM engine
(single-process multi-chunk dispatch, the jax.lax.scan-fused multi-chunk
kernel, utility surfaces, and real multi-device distributed dispatch).

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why.
"""
import numpy as np
import pytest
import jax

from dense_evolution import Chunk, DenseSVSimulator

# ─────────────────────────────────────────────────────────────
# CHUNK PUBLIC API
# ─────────────────────────────────────────────────────────────

class TestChunkPublicAPI:
    """Regression guard for audit finding #3: the README's own Anti-OOM
    quick-start (`from dense_evolution import Chunk`) raised ImportError —
    Chunk was never re-exported from the package root, only reachable via
    `dense_evolution.chunk`. Chunk also lacked get_probabilities()/
    get_statevector(), unlike DenseSVSimulator, so even the right import
    path needed an undocumented `np.abs(sim.sv)**2` workaround."""

    def test_chunk_importable_from_package_root(self):
        # this is the regression itself: it would have raised ImportError
        from dense_evolution import Chunk as ChunkFromRoot
        assert ChunkFromRoot is Chunk

    def test_chunk_get_probabilities_matches_manual_computation(self):
        sim = Chunk(6)
        sim.run_chunk([['h', i] for i in range(6)], 500)
        probs = np.asarray(sim.get_probabilities())
        manual = np.abs(np.asarray(sim.sv)) ** 2
        np.testing.assert_allclose(probs, manual, atol=1e-12)
        assert abs(probs.sum() - 1.0) < 1e-9
        assert np.allclose(probs, 1.0 / 64, atol=1e-9)  # uniform after H on all 6 qubits

    def test_chunk_sv_setter_num_chunks_1_round_trips_through_inner_sim(self):
        # sv's setter, num_chunks==1 branch (self._chunk_sims is None):
        # writes straight through to the physical inner simulator instead of
        # splitting across chunks -- never exercised by any existing test,
        # which only ever reads sim.sv, never assigns to it.
        sim = Chunk(6)
        assert sim._chunk_sims is None  # confirms this test is on the branch it targets

        new_sv = np.zeros(2 ** 6, dtype=sim.sv.dtype)
        new_sv[5] = 1.0  # an arbitrary basis state, distinct from the |000000> default
        sim.sv = new_sv

        np.testing.assert_allclose(np.asarray(sim.sv), new_sv, atol=1e-12)
        np.testing.assert_allclose(np.asarray(sim._inner_sim.sv), new_sv, atol=1e-12)


class TestChunkMultiPiece:
    """Chunk(n_qubits) beyond the RAM-safe budget used to silently simulate
    FEWER qubits than requested (inner simulator sized to
    min(n_qubits, chunk_size_bits)) instead of real multi-chunk splitting —
    num_chunks/chunk_dim were computed but never acted on. Found testing
    Chunk(n_qubits=28) directly: get_probabilities() returned 2**27 elements,
    not 2**28.

    These tests force a small chunk_size_bits via monkeypatching
    get_dynamic_chunk (so num_chunks>1 is cheap to test) and cross-check the
    multi-chunk dispatch against a plain DenseSVSimulator(n_qubits) running
    the identical circuit — the only real correctness bar here, since this
    is bit-manipulation-heavy code where a plausible-looking-but-wrong
    formula is easy to miss by inspection alone."""

    @pytest.fixture
    def force_chunk_bits(self, monkeypatch):
        """Returns a function to force MemoryChunker's safe budget to a
        fixed small value, so num_chunks>1 can be tested without needing
        real multi-GB allocations."""
        import dense_evolution.chunk as chunk_mod

        def _force(bits):
            monkeypatch.setattr(chunk_mod, "get_dynamic_chunk", lambda dtype_target: bits)

        return _force

    def _compare_to_reference(self, n_qubits, circuit):
        c = Chunk(n_qubits)
        c.run_chunk(circuit)
        sv_chunk = np.asarray(c.get_statevector())

        ref = DenseSVSimulator(n_qubits)
        ref.run_circuit(circuit, transpile=True)
        sv_ref = np.asarray(ref.get_statevector())

        return sv_chunk, sv_ref

    def test_empty_circuit_canary(self, force_chunk_bits):
        # cheapest way to catch "every chunk seeded its own |0...0>" in
        # isolation, before it's buried in a larger circuit's diff
        force_chunk_bits(4)
        c = Chunk(6)  # chunk_size_bits=4 -> num_chunks=4
        assert c.num_chunks == 4
        probs = np.asarray(c.get_probabilities())
        assert probs.shape == (64,)
        assert abs(probs.sum() - 1.0) < 1e-9
        assert abs(probs[0] - 1.0) < 1e-9

        ref = DenseSVSimulator(6)
        np.testing.assert_allclose(np.asarray(c.get_statevector()), ref.get_statevector(), atol=1e-12)

    @pytest.mark.parametrize("qubit", [3])  # local qubit (m=2 for n=6,bits=4)
    def test_1q_local(self, force_chunk_bits, qubit):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', qubit)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("qubit", [0, 1])  # chunk-select: MSB (0) and non-MSB (1)
    def test_1q_chunk_select(self, force_chunk_bits, qubit):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', qubit)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    def test_2q_local_local(self, force_chunk_bits):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', 3), ('cx', 3, 4)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_local(self, force_chunk_bits, gate):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', 0), (gate, 0, 3)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_local_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', 3), (gate, 3, 0)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(4)
        sv_chunk, sv_ref = self._compare_to_reference(6, [('h', 0), ('h', 1), (gate, 0, 1)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    def test_parametric_2q_gates_all_four_locations(self, force_chunk_bits):
        # cp/crz aren't in GATE_IDS (run_circuit_jit's table) —
        # canary for silently dropping them via the wrong dispatch table
        force_chunk_bits(4)
        cases = [
            [('h', 0), ('cp', 0, 3, 0.7)],                  # ctrl chunk-select, tgt local
            [('h', 3), ('crz', 3, 0, 1.1)],                 # ctrl local, tgt chunk-select
            [('h', 0), ('h', 1), ('cp', 0, 1, 0.9)],         # both chunk-select
            [('h', 3), ('h', 4), ('crz', 3, 4, 0.4)],        # both local
        ]
        for circuit in cases:
            sv_chunk, sv_ref = self._compare_to_reference(6, circuit)
            np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    def test_random_mixed_circuits_num_chunks_8(self, force_chunk_bits):
        # num_chunks=8 (m=3) specifically exercises the middle chunk-select
        # bit (index 1 of 3), not just the most-significant selector bit —
        # catches formulas that hardcode the full-register bit position
        # instead of the chunk-index-local one.
        force_chunk_bits(4)
        n = 7
        rng = np.random.default_rng(1234)
        gates_1q = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']
        gates_1q_param = ['rx', 'ry', 'rz', 'p']
        gates_2q = ['cx', 'cz', 'cy']
        gates_2q_param = ['cp', 'crz']

        for _trial in range(8):
            circuit = []
            for _ in range(20):
                kind = rng.integers(0, 4)
                if kind == 0:
                    circuit.append((rng.choice(gates_1q), int(rng.integers(0, n))))
                elif kind == 1:
                    circuit.append((rng.choice(gates_1q_param), int(rng.integers(0, n)),
                                     float(rng.uniform(-3.14, 3.14))))
                elif kind == 2:
                    q1, q2 = rng.choice(n, size=2, replace=False)
                    circuit.append((rng.choice(gates_2q), int(q1), int(q2)))
                else:
                    q1, q2 = rng.choice(n, size=2, replace=False)
                    circuit.append((rng.choice(gates_2q_param), int(q1), int(q2),
                                     float(rng.uniform(-3.14, 3.14))))
            sv_chunk, sv_ref = self._compare_to_reference(n, circuit)
            np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-8,
                                        err_msg=f"circuit={circuit}")

    def test_num_chunks_1_path_untouched(self, force_chunk_bits):
        # sanity: with a budget that covers n_qubits, behaviour must be
        # identical to before this feature existed (single inner sim)
        force_chunk_bits(16)
        c = Chunk(6)
        assert c.num_chunks == 1
        c.run_chunk([('h', i) for i in range(6)])
        probs = np.asarray(c.get_probabilities())
        assert np.allclose(probs, 1.0 / 64, atol=1e-9)

    def test_chunk_get_statevector_matches_sv(self):
        sim = Chunk(4)
        sim.run_chunk([['h', 0]], 500)
        np.testing.assert_array_equal(np.asarray(sim.get_statevector()), np.asarray(sim.sv))

    def test_2q_gate_spanning_first_to_last_qubit(self, force_chunk_bits):
        # Explicit long-range case: control on qubit 0 (chunk-select, most
        # significant) and target on the LAST local qubit (opposite end of
        # the register), not just an adjacent chunk-select/local pair --
        # the bit-shift math for chunk index vs local index is most likely
        # to have an off-by-one/wrong-stride bug at the extremes.
        force_chunk_bits(4)  # n=6, chunk_size_bits=4 -> m=2, local qubits [2,3,4,5]
        sv_chunk, sv_ref = self._compare_to_reference(
            6, [('h', 0), ('h', 5), ('cx', 0, 5)])
        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)

    # ── MemoryPressureError actually firing, not just "doesn't crash" ────

    def test_memory_pressure_error_fires_on_insufficient_ram_num_chunks_1(self, monkeypatch):
        import dense_evolution.chunk as chunk_mod

        class _FakeVM:
            total = 8 * 1024 ** 3       # 8 GB
            available = 0.05 * 8 * 1024 ** 3  # 5% free -- below any sane threshold
            percent = 95.0

        monkeypatch.setattr(chunk_mod.psutil, "virtual_memory", lambda: _FakeVM())
        with pytest.raises(chunk_mod.MemoryPressureError, match="MEMORIA CRITICA"):
            Chunk(10, memory_threshold=0.15)

    def test_memory_pressure_error_fires_on_insufficient_ram_num_chunks_gt_1(self, monkeypatch, force_chunk_bits):
        import dense_evolution.chunk as chunk_mod

        force_chunk_bits(4)  # forces num_chunks>1 at small n_qubits

        class _FakeVM:
            total = 8 * 1024 ** 3
            available = 0.05 * 8 * 1024 ** 3
            percent = 95.0

        monkeypatch.setattr(chunk_mod.psutil, "virtual_memory", lambda: _FakeVM())
        with pytest.raises(chunk_mod.MemoryPressureError, match="MEMORIA INSUFFICIENTE"):
            Chunk(6, memory_threshold=0.15)

    def test_memory_pressure_error_not_raised_with_ample_ram(self, monkeypatch, force_chunk_bits):
        # Negative control: the same fake-psutil machinery, but with
        # generous available memory, must NOT raise -- confirms the two
        # tests above are catching a real threshold check, not e.g. an
        # unconditional raise or a monkeypatch that broke construction
        # entirely.
        import dense_evolution.chunk as chunk_mod

        force_chunk_bits(4)

        class _FakeVM:
            total = 8 * 1024 ** 3
            available = 0.90 * 8 * 1024 ** 3
            percent = 10.0

        monkeypatch.setattr(chunk_mod.psutil, "virtual_memory", lambda: _FakeVM())
        c = Chunk(6, memory_threshold=0.15)
        assert c.num_chunks == 4


class TestChunkMultiPieceJIT:
    """dense_evolution.chunk's multi-chunk dispatch (num_chunks>1) used to
    apply gates one at a time via a Python loop calling
    sim.apply_gate_1q/apply_gate_2q (neither @jax.jit'd, no jax.lax.scan) —
    6x slower than run_circuit_jit on an identical workload
    (10 qubits/200 gates/4 forced chunks: 2.2s vs 0.37s). Replaced with a
    single jax.lax.scan over the whole circuit, operating on the stacked
    (num_chunks, chunk_dim) representation directly (never materializing a
    (2**n_qubits,) array — the anti-OOM property Chunk exists for). The 6
    gate/qubit-location cases were ported formula-for-formula from the old
    Python-loop implementation (dense_evolution/chunk.py git history) and
    verified against it case-by-case before it was removed — TestChunkMultiPiece
    above (unchanged, all 18 tests still pass against the new kernel) is
    that cross-check. This class covers what's new: dtype and the actual
    measured speedup."""

    @pytest.fixture
    def force_chunk_bits(self, monkeypatch):
        import dense_evolution.chunk as chunk_mod

        def _force(bits):
            monkeypatch.setattr(chunk_mod, "get_dynamic_chunk", lambda dtype_target: bits)

        return _force

    @pytest.mark.parametrize("use_float32", [False, True])
    def test_dtype_consistency_across_all_cases(self, force_chunk_bits, use_float32):
        # Every constant in the new kernel must derive from the stacked
        # array's own dtype, never a hardcoded complex128 — the exact
        # mistake that once broke beast-mode's use_float32 path (a
        # lax.cond/where branch mismatch that surfaces at trace time, not
        # silently), so this exercises all 6 cases under complex64 too.
        force_chunk_bits(4)
        n = 6
        circuit = [
            ('h', 3), ('rx', 4, 0.6),              # 1q local
            ('h', 0), ('ry', 1, 0.4),               # 1q chunk-select
            ('cx', 3, 4),                            # 2q local-local
            ('cp', 0, 4, 0.7),                        # ctrl chunk, tgt local
            ('crz', 4, 1, 1.1),                       # ctrl local, tgt chunk
            ('cy', 0, 1),                              # both chunk-select
        ]
        chunk_sim = Chunk(n, use_float32=use_float32)
        chunk_sim.run_chunk(circuit)
        ref = DenseSVSimulator(n, use_float32=use_float32)
        ref.run_circuit(circuit, transpile=True)
        atol = 1e-5 if use_float32 else 1e-9
        np.testing.assert_allclose(
            np.asarray(chunk_sim.get_statevector()), np.asarray(ref.get_statevector()), atol=atol)

    def test_measured_speedup_over_python_loop_dispatch(self, force_chunk_bits, capsys):
        # Not a strict pass/fail bound (timing is noisy) — reports the real
        # measured number, same discipline as the donate_argnums RSS
        # measurement: honest data, not a claim asserted on faith. The
        # comparison point (2.2s on this exact benchmark, pre-fix) is
        # recorded in the changelog, not re-measured here (the old
        # implementation no longer exists to compare against directly).
        import time
        force_chunk_bits(8)
        n_qubits = 10
        circuit = ([('h', i % n_qubits) for i in range(100)]
                   + [('cx', i % (n_qubits - 1), (i % (n_qubits - 1)) + 1) for i in range(100)])

        chunk_sim = Chunk(n_qubits=n_qubits, memory_threshold=0.01)
        assert chunk_sim.num_chunks == 4
        t0 = time.perf_counter()
        chunk_sim.run_chunk(circuit)
        elapsed = time.perf_counter() - t0

        with capsys.disabled():
            print(f"\n[multi-chunk JIT speed] n_qubits={n_qubits} num_chunks=4 "
                  f"200 gates: {elapsed:.4f}s (pre-fix Python-loop dispatch: ~2.2s)")

    @pytest.mark.parametrize("gate_op", [
        ("u3", ('u3', 2, 0.3, 0.5, 0.7)),
        ("u2", ('u2', 2, 0.4, 0.9)),
        ("ecr", ('ecr', 2, 3)),
        ("iswap", ('iswap', 2, 3)),
    ], ids=lambda p: p[0])
    def test_gphase_derived_gates_match_reference(self, force_chunk_bits, gate_op):
        # Regression test: QuantumTranspiler.transpile decomposes u2/u3/ecr/
        # iswap into sequences that include a 'gphase' op (GATE_IDS['gphase']
        # = 14). This kernel's own 1-qubit jax.lax.switch table used to be
        # clipped to jnp.clip(g_id, 0, 13) -- one branch short of the 15
        # _apply_gate_fast_step (compiler.py) has -- so gate_id 14 silently
        # aliased to branch 13 (SX) instead of applying e^{i*alpha}*I. Caught
        # by forcing num_chunks>1 (the bug only lives in this multi-chunk
        # kernel, not the single-chunk path, which just forwards to
        # DenseSVSimulator) and comparing against it as ground truth.
        _, op = gate_op
        force_chunk_bits(4)
        n = 6
        circuit = [('h', 0), ('cx', 0, 1), op, ('h', 4), ('cx', 4, 5)]

        chunk_sim = Chunk(n, use_float32=False)
        assert chunk_sim.num_chunks > 1
        chunk_sim.run_chunk(circuit)
        sv_chunk = np.asarray(chunk_sim.get_statevector())

        ref = DenseSVSimulator(n, use_float32=False)
        ref.run_circuit(circuit, transpile=True)
        sv_ref = np.asarray(ref.get_statevector())

        np.testing.assert_allclose(sv_chunk, sv_ref, atol=1e-9)


class TestChunkUtilities:
    """Coverage-driven tests for chunk.py's smaller utility surfaces --
    get_dynamic_chunk's non-default dtype branches, SafeMemoryGuard/
    MemoryChunker's __repr__/geometry/validation, _compile_multi_chunk_ops's
    unknown-gate-skip and empty-circuit paths, CircuitChunker's no-simulator
    guard, and Chunk's property forwarders in num_chunks>1 mode (never
    exercised by TestChunkMultiPiece's own tests, which only ever compare
    statevectors/probabilities against DenseSVSimulator, not these
    introspection properties directly)."""

    def test_get_dynamic_chunk_numpy_complex128(self):
        from dense_evolution.chunk import get_dynamic_chunk
        bits = get_dynamic_chunk(np.complex128)
        assert 16 <= bits <= 27

    def test_get_dynamic_chunk_other_dtype(self):
        from dense_evolution.chunk import get_dynamic_chunk
        bits = get_dynamic_chunk(np.float32)
        assert 16 <= bits <= 27

    def test_safe_memory_guard_rejects_invalid_threshold(self):
        from dense_evolution.chunk import SafeMemoryGuard
        with pytest.raises(ValueError):
            SafeMemoryGuard(threshold_pct=0.0)
        with pytest.raises(ValueError):
            SafeMemoryGuard(threshold_pct=1.0)
        with pytest.raises(ValueError):
            SafeMemoryGuard(threshold_pct=-0.1)

    def test_safe_memory_guard_repr(self):
        from dense_evolution.chunk import SafeMemoryGuard
        guard = SafeMemoryGuard(threshold_pct=0.15)
        r = repr(guard)
        assert "SafeMemoryGuard" in r and "threshold=15%" in r

    def test_memory_chunker_geometry_and_repr(self):
        from dense_evolution.chunk import MemoryChunker
        mc = MemoryChunker(n_qubits=4)
        num_chunks, chunk_dim, chunk_size_bits = mc.geometry()
        assert num_chunks == mc.num_chunks
        assert chunk_dim == mc.chunk_dim
        assert chunk_size_bits == mc.chunk_size_bits
        r = repr(mc)
        assert "MemoryChunker" in r and "num_chunks=" in r

    def test_compile_multi_chunk_ops_skips_unknown_gate(self):
        from dense_evolution.chunk import _compile_multi_chunk_ops
        # 'not_a_real_gate' isn't in GATE_IDS -- must be silently skipped
        # (same documented behavior as beast-mode's own GATE_IDS lookup),
        # 'h' on qubit 0 must still be compiled.
        rows = _compile_multi_chunk_ops([('not_a_real_gate', 0), ('h', 0)])
        rows_np = np.asarray(rows)
        assert rows_np.shape[0] == 1

    def test_compile_multi_chunk_ops_empty_circuit(self):
        from dense_evolution.chunk import _compile_multi_chunk_ops
        rows = _compile_multi_chunk_ops([])
        rows_np = np.asarray(rows)
        assert rows_np.shape == (0, 4)

    def test_circuit_chunker_requires_simulator_instance(self):
        from dense_evolution.chunk import CircuitChunker
        chunker = CircuitChunker()  # no simulator_instance
        with pytest.raises(RuntimeError, match="no simulator instance"):
            chunker.split_circuit([('h', 0)])

    def test_chunk_repr(self):
        c = Chunk(n_qubits=3)
        r = repr(c)
        assert "Chunk(" in r and "num_chunks=" in r

    def test_chunk_multi_piece_property_forwarding(self, monkeypatch):
        # Same force_chunk_bits pattern as TestChunkMultiPiece, but this
        # time actually touching the properties themselves (chunk_size_bits,
        # chunk_dim, dtype, memory_geometry, the sv setter, memory_mb) in
        # num_chunks>1 mode, which no existing test does directly.
        import dense_evolution.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "get_dynamic_chunk", lambda dtype_target: 4)

        c = Chunk(n_qubits=6)  # chunk_size_bits=4 -> num_chunks=4
        assert c.num_chunks == 4
        assert c.chunk_size_bits == 4
        assert c.chunk_dim == 2 ** 4
        assert c.dtype is not None
        assert c.memory_geometry.num_chunks == 4
        assert c.memory_mb() > 0

        # sv getter/setter round-trip through the multi-chunk split path
        original_sv = np.asarray(c.sv).copy()
        c.sv = original_sv  # exercises the setter's num_chunks>1 branch
        np.testing.assert_allclose(np.asarray(c.sv), original_sv, atol=1e-12)


class TestChunkDistributed:
    """Chunk.run_chunk_distributed (issue #1): dispatches the multi-chunk
    kernel across a real JAX device mesh (jax.shard_map + jax.lax.ppermute
    for the cross-chunk edge exchange) instead of one process's RAM, one
    physical chunk per device (v1 scope). JAX's device count is fixed at
    process start, so these tests need >= 8 devices to exercise the
    interesting (num_chunks>1) cases and are skipped otherwise -- run
    with XLA_FLAGS=--xla_force_host_platform_device_count=8 (or more) to
    actually execute them; see CI, which sets this for a dedicated step.

    Correctness bar is the same one TestChunkMultiPiece already
    established: cross-check against a plain DenseSVSimulator running the
    identical circuit -- not against the single-process multi-chunk path,
    to avoid two implementations of the same bug looking like agreement."""

    MIN_DEVICES = 8

    @pytest.fixture(autouse=True)
    def _require_devices(self):
        if jax.device_count() < self.MIN_DEVICES:
            pytest.skip(
                f"needs >= {self.MIN_DEVICES} JAX devices, only "
                f"{jax.device_count()} available -- run with XLA_FLAGS="
                f"--xla_force_host_platform_device_count={self.MIN_DEVICES}"
            )

    @pytest.fixture
    def force_chunk_bits(self, monkeypatch):
        import dense_evolution.chunk as chunk_mod

        def _force(bits):
            monkeypatch.setattr(chunk_mod, "get_dynamic_chunk", lambda dtype_target: bits)

        return _force

    def _compare_to_reference(self, n_qubits, circuit):
        c = Chunk(n_qubits)
        c.run_chunk_distributed(circuit)
        sv_dist = np.asarray(c.get_statevector())

        ref = DenseSVSimulator(n_qubits)
        ref.run_circuit(circuit, transpile=True)
        sv_ref = np.asarray(ref.get_statevector())
        return sv_dist, sv_ref

    def test_1q_local(self, force_chunk_bits):
        force_chunk_bits(3)  # n_qubits=6 -> num_chunks=8, m=3
        sv_dist, sv_ref = self._compare_to_reference(6, [('h', 3), ('rx', 4, 0.4), ('rz', 5, 0.9)])
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    def test_1q_chunk_select(self, force_chunk_bits):
        force_chunk_bits(3)
        sv_dist, sv_ref = self._compare_to_reference(6, [('h', 0), ('h', 1), ('h', 2)])
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    def test_2q_local_local(self, force_chunk_bits):
        force_chunk_bits(3)
        sv_dist, sv_ref = self._compare_to_reference(6, [('h', 3), ('cx', 3, 4), ('cx', 4, 5)])
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_local(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 0), ('h', 1), ('h', 2), ('h', 5), (gate, 0, 5)]
        sv_dist, sv_ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_local_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 5), ('h', 0), (gate, 5, 0)]
        sv_dist, sv_ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 0), ('h', 1), (gate, 0, 1)]
        sv_dist, sv_ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    def test_parametric_2q_gates(self, force_chunk_bits):
        force_chunk_bits(3)
        circuit = [('h', q) for q in range(6)] + [('crz', 0, 1, 0.5), ('cp', 3, 4, 0.7)]
        sv_dist, sv_ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    def test_random_mixed_circuit(self, force_chunk_bits):
        force_chunk_bits(3)
        rng = np.random.default_rng(11)
        pool_1q, pool_2q = ['h', 'x', 'y', 'z', 's'], ['cx', 'cz', 'cy']
        circuit = []
        for _ in range(30):
            if rng.random() < 0.5:
                circuit.append((rng.choice(pool_1q), int(rng.integers(0, 6))))
            else:
                q1, q2 = rng.choice(6, size=2, replace=False)
                circuit.append((rng.choice(pool_2q), int(q1), int(q2)))
        sv_dist, sv_ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)

    def test_dtype_float32(self, force_chunk_bits):
        force_chunk_bits(2)  # n_qubits=4 -> num_chunks=4, m=2
        circuit = [('h', 0), ('h', 1), ('cx', 0, 2), ('rz', 3, 0.6)]
        c = Chunk(4, use_float32=True)
        c.run_chunk_distributed(circuit)
        sv_dist = np.asarray(c.get_statevector())
        ref = DenseSVSimulator(4, use_float32=True)
        ref.run_circuit(circuit, transpile=True)
        sv_ref = np.asarray(ref.get_statevector())
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-4)
        assert sv_dist.dtype == np.complex64

    def test_num_chunks_1_raises_clear_error(self):
        # dispatch_distributed only makes sense for num_chunks>1; a Chunk
        # that fits in a single chunk must raise, not silently no-op or
        # fall back to the single-process path.
        c = Chunk(2)  # tiny -- always fits in one chunk
        assert c.num_chunks == 1
        with pytest.raises(RuntimeError, match="num_chunks"):
            c.run_chunk_distributed([('h', 0)])

    def test_insufficient_devices_raises_clear_error(self, force_chunk_bits, monkeypatch):
        # force num_chunks to exceed the actual device count, even though
        # we have >= MIN_DEVICES for other tests in this class
        force_chunk_bits(3)
        c = Chunk(6)  # num_chunks=8
        monkeypatch.setattr(jax, "device_count", lambda: 4)
        with pytest.raises(RuntimeError, match="devices"):
            c.run_chunk_distributed([('h', 0)])

    @pytest.mark.parametrize("gate_op", [
        ("u3", ('u3', 4, 0.3, 0.5, 0.7)),
        ("ecr", ('ecr', 4, 5)),
    ], ids=lambda p: p[0])
    def test_gphase_derived_gates_match_reference(self, force_chunk_bits, gate_op):
        # Same GATE_IDS['gphase']=14 staleness bug as
        # TestChunkMultiPieceJIT.test_gphase_derived_gates_match_reference,
        # but in _build_distributed_chunk_step's own separately-duplicated
        # 1-qubit switch table (a third copy of the same table, independent
        # of the non-distributed kernel's).
        _, op = gate_op
        force_chunk_bits(3)  # n_qubits=6 -> num_chunks=8, m=3
        sv_dist, sv_ref = self._compare_to_reference(
            6, [('h', 0), ('h', 1), ('h', 2), op])
        np.testing.assert_allclose(sv_dist, sv_ref, atol=1e-9)


class TestChunkStreaming:
    """Chunk(..., streaming=True) / run_chunk_streaming(): a third
    execution mode alongside run_chunk() (every chunk allocated eagerly in
    __init__, device memory scales with num_chunks) and
    run_chunk_distributed() (needs jax.device_count() >= num_chunks). This
    one keeps num_chunks chunks in HOST RAM between gates and moves at
    most 2 onto the compute device at a time -- device memory is bounded
    by a small, dynamically-computed budget regardless of num_chunks, at
    the cost of one host<->device transfer per chunk-select-qubit gate.
    Same "amplitudes sent pairwise" structure as the distributed-memory
    literature (LaRose, arXiv:1801.01037) and the same load/compute/
    write-back cycle IBM's secondary-storage Sycamore simulation uses
    (Pednault et al., arXiv:1910.09534) -- see run_chunk_streaming's own
    docstring for the full citation and the one optimization (batching
    consecutive gates per chunk pair) it doesn't yet implement.

    Correctness bar: cross-check against a plain DenseSVSimulator running
    the identical (transpiled) circuit -- same standard the other two
    execution modes' test classes use."""

    @pytest.fixture
    def force_chunk_bits(self, monkeypatch):
        import dense_evolution.chunk as chunk_mod

        def _force(bits):
            monkeypatch.setattr(chunk_mod, "get_dynamic_chunk", lambda dtype_target: bits)

        return _force

    def _compare_to_reference(self, n_qubits, circuit, use_float32=False):
        c = Chunk(n_qubits, use_float32=use_float32, streaming=True)
        c.run_chunk_streaming(circuit)
        sv_stream = np.asarray(c.get_statevector())

        ref = DenseSVSimulator(n_qubits, use_float32=use_float32)
        ref.run_circuit(circuit, transpile=True)
        sv_ref = np.asarray(ref.get_statevector())
        return sv_stream, sv_ref

    def test_1q_local(self, force_chunk_bits):
        force_chunk_bits(3)  # n_qubits=6 -> num_chunks=8, m=3
        sv, ref = self._compare_to_reference(6, [('h', 3), ('rx', 4, 0.4), ('rz', 5, 0.9)])
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_1q_chunk_select(self, force_chunk_bits):
        force_chunk_bits(3)
        sv, ref = self._compare_to_reference(6, [('h', 0), ('h', 1), ('h', 2)])
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_2q_local_local(self, force_chunk_bits):
        force_chunk_bits(3)
        sv, ref = self._compare_to_reference(6, [('h', 3), ('cx', 3, 4), ('cx', 4, 5)])
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_local(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 0), ('h', 1), ('h', 2), ('h', 5), (gate, 0, 5)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_local_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 5), ('h', 0), (gate, 5, 0)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    @pytest.mark.parametrize("gate", ["cx", "cz", "cy"])
    def test_2q_control_chunk_select_target_chunk_select(self, force_chunk_bits, gate):
        force_chunk_bits(3)
        circuit = [('h', 0), ('h', 1), (gate, 0, 1)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_parametric_2q_gates(self, force_chunk_bits):
        force_chunk_bits(3)
        circuit = [('h', q) for q in range(6)] + [('crz', 0, 1, 0.5), ('cp', 3, 4, 0.7)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    @pytest.mark.parametrize("gate_op", [
        ("u3", ('u3', 3, 0.3, 0.5, 0.7)),
        ("u2", ('u2', 2, 0.4, 0.9)),
        ("ecr", ('ecr', 2, 3)),
        ("iswap", ('iswap', 2, 3)),
    ], ids=lambda p: p[0])
    def test_gphase_derived_gates_match_reference(self, force_chunk_bits, gate_op):
        _, op = gate_op
        force_chunk_bits(3)
        circuit = [('h', q) for q in range(6)] + [op, ('cx', 3, 4)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_random_mixed_circuit_large_num_chunks(self, force_chunk_bits):
        # num_chunks=128 (n_qubits=9, chunk_size_bits=2) -- stresses the
        # "skip untouched chunks entirely" optimization in
        # _apply_2q_streaming's ctrl-chunk-select cases at real scale, not
        # just num_chunks=8 like the other tests in this class.
        force_chunk_bits(2)
        rng = np.random.default_rng(42)
        pool_1q, pool_2q = ['h', 'x', 'y', 'z', 's', 'rx', 'ry', 'rz'], ['cx', 'cz', 'cy', 'crz', 'cp']
        n = 9
        circuit = []
        for _ in range(80):
            if rng.random() < 0.35:
                q1, q2 = rng.choice(n, size=2, replace=False)
                name = rng.choice(pool_2q)
                if name in ('crz', 'cp'):
                    circuit.append((name, int(q1), int(q2), float(rng.uniform(0, 6.28))))
                else:
                    circuit.append((name, int(q1), int(q2)))
            else:
                name = rng.choice(pool_1q)
                q = int(rng.integers(n))
                if name in ('rx', 'ry', 'rz'):
                    circuit.append((name, q, float(rng.uniform(0, 6.28))))
                else:
                    circuit.append((name, q))
        sv, ref = self._compare_to_reference(n, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_dtype_float32(self, force_chunk_bits):
        force_chunk_bits(2)  # n_qubits=4 -> num_chunks=4, m=2
        circuit = [('h', 0), ('h', 1), ('cx', 0, 2), ('rz', 3, 0.6)]
        sv, ref = self._compare_to_reference(4, circuit, use_float32=True)
        np.testing.assert_allclose(sv, ref, atol=1e-4)
        assert sv.dtype == np.complex64

    def test_num_chunks_1_forwards_correctly(self, force_chunk_bits):
        # streaming=True is a no-op when num_chunks==1 -- both constructor
        # branches produce the same single inner simulator.
        force_chunk_bits(10)  # n_qubits=4 fits in one chunk
        c = Chunk(4, streaming=True)
        assert c.num_chunks == 1
        sv, ref = self._compare_to_reference(4, [('h', 0), ('cx', 0, 1), ('rz', 2, 0.5)])
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_run_chunk_raises_on_streaming_instance(self, force_chunk_bits):
        force_chunk_bits(3)
        c = Chunk(6, streaming=True)
        with pytest.raises(RuntimeError, match="streaming"):
            c.run_chunk([('h', 0)])

    def test_dispatch_distributed_raises_on_streaming_instance(self, force_chunk_bits):
        force_chunk_bits(3)
        c = Chunk(6, streaming=True)
        with pytest.raises(RuntimeError, match="streaming"):
            c.dispatch_distributed([('h', 0)])

    def test_run_chunk_streaming_raises_without_streaming_true(self, force_chunk_bits):
        force_chunk_bits(3)
        c = Chunk(6)  # streaming=False (default), num_chunks=8
        with pytest.raises(RuntimeError, match="streaming=True"):
            c.run_chunk_streaming([('h', 0)])

    def test_device_budget_too_small_raises_memory_pressure_error(self, force_chunk_bits):
        import dense_evolution.chunk as chunk_mod

        force_chunk_bits(3)
        c = Chunk(6, streaming=True)
        with pytest.raises(chunk_mod.MemoryPressureError, match="run_chunk_streaming needs room"):
            c.run_chunk_streaming([('h', 0)], device_budget_mb=0.0001)

    def test_dynamic_default_budget_runs_without_explicit_arg(self, force_chunk_bits):
        # No device_budget_mb passed -- must fall back to a real, computed
        # budget (dense_evolution.chunk._device_memory_budget_bytes), not
        # crash for lack of an explicit number.
        force_chunk_bits(3)
        circuit = [('h', 0), ('h', 3), ('cx', 3, 4), ('crz', 0, 4, 0.3)]
        sv, ref = self._compare_to_reference(6, circuit)
        np.testing.assert_allclose(sv, ref, atol=1e-9)

    def test_device_transfer_count_matches_hand_derived_formula(self, force_chunk_bits, monkeypatch):
        # Not a peak-concurrency check (self._host_chunks entries are
        # always plain numpy between iterations -- the on-device arrays
        # only ever exist as loop-local Python variables, invisible to any
        # inspection of self._host_chunks; the "at most 2 chunks on-device
        # at once" guarantee is a structural property of the code in
        # _apply_1q_streaming/_apply_2q_streaming, verified by reading
        # them, not by runtime instrumentation here). What IS directly
        # measurable and worth pinning down: the TOTAL number of
        # host<->device transfers for a known circuit, hand-derived case
        # by case (num_chunks=8, m=3 throughout):
        #   6x 'h' on q in {0,1,2} (chunk-select) -> 4 pairs x 2 = 8 each  = 24
        #   6x 'h' on q in {3,4,5} (local)        -> 8 chunks x 1 each    = 24
        #   cy(0,1)      both chunk-select, only ctrl-set pairs touched  =  4
        #   crz(3,1,.4)  ctrl local/tgt chunk, all 4 pairs, masked        =  8
        #   cx(3,4)      both local, every chunk touched                 =  8
        #                                                          total = 68
        import dense_evolution.chunk as chunk_mod

        force_chunk_bits(3)
        c = Chunk(6, streaming=True)
        counts = []
        original_device_put = chunk_mod.jax.device_put

        def counting_device_put(x, *a, **kw):
            counts.append(1)
            return original_device_put(x, *a, **kw)

        monkeypatch.setattr(chunk_mod.jax, "device_put", counting_device_put)
        circuit = [('h', q) for q in range(6)] + [('cy', 0, 1), ('crz', 3, 1, 0.4), ('cx', 3, 4)]
        c.run_chunk_streaming(circuit)
        assert len(counts) == 68, (
            f"{len(counts)} device_put calls, expected exactly 68 (see the "
            f"hand-derived breakdown above) -- the per-gate transfer pattern "
            f"changed, re-derive and update this count if that was intentional"
        )
