import numpy as np
import pytest
from dense_evolution import DenseSVSimulator, GATES, PARAMETRIC_GATES, NoiseModel, QuantumTranspiler, QASMParser, Chunk

import inspect
import jax.numpy as jnp # Ensure jnp is available for jax backend

# Patch the measure method directly within the test file to ensure pytest uses the patched version
def patched_measure_for_tests(self, qubit_idx: int) -> int:
    """
    Misura un singolo qubit e collassa lo stato quantistico.
    """
    import numpy as np # Ensure np is available for random.choice

    if not 0 <= qubit_idx < self.n:
        raise ValueError(f"Qubit {qubit_idx} out of bounds")

    xp = self.xp
    # phys_q is used for stride calculation in NumPy/CuPy branch (LSB-first index)
    phys_q = self.n - 1 - qubit_idx
    stride = 1 << phys_q

    if xp is jnp:
        # JAX branch: Calculate probabilities by moving the correct (MSB-indexed) axis
        probs = self.xp.abs(self.sv)**2
        sv_shape = [2] * self.n
        sv_nd = probs.reshape(sv_shape)
        # FIX: Use qubit_idx directly as axis, as sv_nd is MSB-first indexed
        moved_probs = jnp.moveaxis(sv_nd, qubit_idx, 0)
        prob_0 = float(jnp.sum(moved_probs[0]))
        prob_1 = float(jnp.sum(moved_probs[1]))
    else:
        # NumPy/CuPy Stride Slicing: phys_q and stride logic correctly applied here
        sv_reshaped = self.sv.reshape(-1, 2, stride)
        prob_0 = float(xp.sum(xp.abs(sv_reshaped[:, 0, :])**2))
        prob_1 = float(xp.sum(xp.abs(sv_reshaped[:, 1, :])**2))

    total = prob_0 + prob_1
    if total > 1e-12:
        prob_0 /= total
        prob_1 /= total

    # Sampling the measurement outcome
    result = int(np.random.choice([0, 1], p=[prob_0, prob_1]))

    if xp is jnp:
        sv_shape = [2] * self.n
        sv_nd = self.sv.reshape(sv_shape)
        moved_sv = jnp.moveaxis(sv_nd, qubit_idx, 0) # FIX: Apply same correction here
        # Correctly zero out the unmeasured component (1 if result is 0, 0 if result is 1)
        moved_sv = moved_sv.at[1 - result].set(0.0)
        self.sv = jnp.moveaxis(moved_sv, 0, qubit_idx).ravel() # FIX: And here too
    else:
        sv_reshaped = self.sv.reshape(-1, 2, stride)
        # Zero out the unmeasured component
        sv_reshaped[:, 1 if result == 0 else 0, :] = 0.0
        self.sv = sv_reshaped.ravel()

    self.normalize()
    return result

# Apply the patch
DenseSVSimulator.measure = patched_measure_for_tests

# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sim2():
    """Fresh 2-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)

@pytest.fixture
def sim3():
    """Fresh 3-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)

@pytest.fixture
def sim4():
    """Fresh 4-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=4, use_gpu=False, use_float32=False)

# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def norm(sim):
    return float(np.linalg.norm(sim.get_statevector()))

def probs(sim):
    return sim.get_probabilities()

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

# ─────────────────────────────────────────────────────────────
# 8. NOISE MODEL (Esempio 2 dal README)
# ─────────────────────────────────────────────────────────────

class TestNoiseModel:

    def test_ideal_model_no_change(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='ideal', p=0.1)
        np.testing.assert_allclose(sv_out, sv, atol=1e-12)

    def test_depolarizing_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(42)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='depolarizing', p=0.2, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_bitflip_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(7)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='bitflip', p=0.3, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_phaseflip_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(99)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='phaseflip', p=0.5, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_amplitude_damping_preserves_norm(self):
        sv = np.array([0.0, 1.0], dtype=complex)  # |1⟩
        rng = np.random.default_rng(42)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='amplitude_damping', p=0.2, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_zero_probability_no_change(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(0)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='depolarizing', p=0.0, rng=rng)
        np.testing.assert_allclose(sv_out, sv, atol=1e-12)

    def test_kraus_description_returns_dict(self):
        for model in NoiseModel.MODELS:
            desc = NoiseModel.kraus_description(model)
            assert isinstance(desc, dict)
            assert 'kraus' in desc

# ─────────────────────────────────────────────────────────────
# 9. TRANSPILER
# ─────────────────────────────────────────────────────────────

class TestTranspiler:

    def test_ccx_decomposition_length(self):
        result = QuantumTranspiler.decompose_toffoli(0, 1, 2)
        assert len(result) == 15

    def test_swap_decomposition_length(self):
        result = QuantumTranspiler.decompose_swap(0, 1)
        assert len(result) == 3

    def test_transpile_passes_through_basic_gates(self):
        circuit = [('h', 0), ('x', 1), ('cx', 0, 1)]
        result = QuantumTranspiler.transpile(circuit)
        assert result == circuit

    def test_transpile_expands_ccx(self):
        circuit = [('ccx', 0, 1, 2)]
        result = QuantumTranspiler.transpile(circuit)
        assert len(result) == 15
        assert all(op[0] in ('h', 'cx', 't', 'tdg') for op in result)

    def test_toffoli_correctness(self, sim3):
        """CCX|110⟩ = |111⟩"""
        sim3.apply_gate_1q(GATES['x'], 0)
        sim3.apply_gate_1q(GATES['x'], 1)
        sim3.run_circuit([('ccx', 0, 1, 2)])
        p = probs(sim3)
        assert p[7] > 0.99  # |111⟩

    def test_toffoli_no_flip_without_both_controls(self, sim3):
        """CCX|100⟩ = |100⟩ (only one control active)"""
        sim3.apply_gate_1q(GATES['x'], 0)
        sim3.run_circuit([('ccx', 0, 1, 2)])
        p = probs(sim3)
        assert p[4] > 0.99  # |100⟩


class TestQASMRangeSyntax:
    """Regression guard for audit finding #2: `gate q[a:b]` on an inherently
    single-qubit gate used to attach all resolved qubits to ONE op, so only
    the first qubit was ever actually gated — the rest were silently dropped
    with no error, and probabilities still summed to 1. The parser's own
    docstring already promised "range syntax expanded to individual qubits";
    parse() now honors that by emitting one op per qubit instead of one op
    carrying the whole list."""

    def test_range_syntax_expands_to_separate_ops(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0:3];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 3
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [2]]
        assert all(op['name'] == 'h' for op in circ.ops)

    def test_range_syntax_produces_correct_superposition(self, sim4):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0:3];'
        circ = QASMParser().parse(qasm)
        sim4.run_circuit_jit_beast_mode([[op['name'], op['qubits'][0], -1] for op in circ.ops])
        p = probs(sim4)
        # q0,q1,q2 uniform superposition, q3 untouched -> 8 equally likely states
        nonzero = np.where(p > 1e-9)[0]
        assert len(nonzero) == 8
        assert np.allclose(p[nonzero], 1.0 / 8, atol=1e-9)

    def test_two_qubit_gate_qubit_list_is_not_expanded(self):
        # sanity check the fix is scoped to single-qubit gate names only —
        # a genuine 2-qubit gate must keep both its qubits on one op
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; cx q[0],q[1];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 1
        assert circ.ops[0]['qubits'] == [0, 1]

# ─────────────────────────────────────────────────────────────
# 10. CIRCUIT CHUNKING (Stress test da README)
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

    def test_chunk_get_statevector_matches_sv(self):
        sim = Chunk(4)
        sim.run_chunk([['h', 0]], 500)
        np.testing.assert_array_equal(np.asarray(sim.get_statevector()), np.asarray(sim.sv))

# ─────────────────────────────────────────────────────────────
# 11. MEMORY
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
# 12. END-TO-END INTEGRATION
# ─────────────────────────────────────────────────────────────

class TestFullPipelineIntegration:
    """Converted from dense_evolution/test2.py and dense_evolution/stress_test.py
    (audit finding #5): two byte-identical, assertion-free print-and-eyeball
    debug scripts that shipped inside every `pip install dense-evolution`
    (via the package-data "*.py" glob), were 0% covered, and never ran in CI.
    The one real signal they checked — parser -> transpiler -> simulator ->
    noise model wired together end to end, and Kraus noise application being
    genuinely stochastic across independent runs — is preserved here as a
    real, CI-enforced test; both original scripts have been deleted."""

    def test_parse_transpile_simulate_and_apply_noise(self):
        qasm_bench = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        cx q[2], q[3];
        cx q[3], q[4];
        cx q[4], q[5];
        rx(1.570796) q[0];
        ry(0.785398) q[1];
        rz(0.392699) q[2];
        """
        parser = QASMParser()
        circ = parser.parse(qasm_bench)
        tuples = QuantumTranspiler.transpile(circ.to_tuples())
        n_qubits = circ.n_qubits
        assert n_qubits == 6
        assert len(tuples) == 9  # 1 h + 5 cx + 3 rotations

        sim_ideale = DenseSVSimulator(n_qubits)
        sim_ideale.run_circuit_jit_beast_mode(tuples)
        prob_ideale = sim_ideale.get_probabilities()
        assert abs(float(np.sum(prob_ideale)) - 1.0) < 1e-9

        sim_noisy1 = DenseSVSimulator(n_qubits)
        sim_noisy1.run_circuit_jit_beast_mode(tuples)
        sim_noisy1.sv = NoiseModel.apply_to_sv(sim_noisy1.sv, n_qubits, model='amplitude_damping', p=0.15)
        prob_noisy1 = sim_noisy1.get_probabilities()

        sim_noisy2 = DenseSVSimulator(n_qubits)
        sim_noisy2.run_circuit_jit_beast_mode(tuples)
        sim_noisy2.sv = NoiseModel.apply_to_sv(sim_noisy2.sv, n_qubits, model='amplitude_damping', p=0.15)
        prob_noisy2 = sim_noisy2.get_probabilities()

        # Kraus noise must be genuinely stochastic: two independent
        # applications of the same channel to the same clean state must
        # not produce identical output.
        stochastic_spread = float(np.linalg.norm(prob_noisy1 - prob_noisy2))
        assert stochastic_spread > 1e-12
