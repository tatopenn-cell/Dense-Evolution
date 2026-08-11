"""
Tests for dense_evolution.autodiff.circuit_to_energy_fn — the differentiable
VQE engine extracted from dashboard_core.py (_build_vqe_template /
_vqe_energy_fn) so it's reachable as public API, independent of Streamlit/
pandas/dashboard, and composable with the Qiskit/PennyLane interop bridge.
"""
import numpy as np
import pytest

import dense_evolution as de
from dense_evolution import autodiff

jax = pytest.importorskip("jax")
import jax.numpy as jnp


VQE_QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; cx q[0],q[1]; rz(0.2) q[1]; cx q[0],q[1]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; measure q -> c;'
)


def _random_hamiltonian(n_qubits, seed=7):
    rng = np.random.default_rng(seed)
    values = np.sort(rng.uniform(-2.5, 2.5, 2 ** n_qubits))
    return jnp.diag(jnp.array(values, dtype=jnp.float64))


class TestCircuitToEnergyFn:

    def test_n_params_matches_parametric_gate_count(self):
        circ = de.QASMParser().parse(VQE_QASM)
        _, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        # ry, rx, rz, ry, rx = 5 parametric gates in VQE_QASM
        assert n_params == 5

    def test_gradient_matches_finite_difference(self):
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)

        rng = np.random.default_rng(3)
        theta0 = rng.uniform(-np.pi, np.pi, n_params)

        energy_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))
        (_, _), grad = energy_and_grad(jnp.asarray(theta0), h_matrix)

        eps = 1e-6
        fd_grad = np.zeros(n_params)
        for i in range(n_params):
            tp, tm = theta0.copy(), theta0.copy()
            tp[i] += eps
            tm[i] -= eps
            (ep, _), _ = energy_and_grad(jnp.asarray(tp), h_matrix)
            (em, _), _ = energy_and_grad(jnp.asarray(tm), h_matrix)
            fd_grad[i] = float((ep - em) / (2 * eps))

        np.testing.assert_allclose(np.asarray(grad), fd_grad, atol=1e-6)

    def test_default_stato_zero_is_ground_state(self):
        circ = de.QASMParser().parse('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; measure q -> c;')
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        assert n_params == 0
        h_matrix = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        energy, sv = energy_fn(jnp.array([]), h_matrix)
        # empty circuit -> stays |00>, energy = h_matrix[0,0]
        assert abs(float(energy) - 1.0) < 1e-9
        assert abs(abs(complex(sv[0])) - 1.0) < 1e-9

    def test_optimization_loop_standalone_energy_descends(self):
        # Same correctness bar as test_dashboard_core.py's
        # test_vqe_energy_trends_downward_over_epochs, but built directly on
        # circuit_to_energy_fn with a hand-rolled Adam loop — no
        # run_vqe_telemetry/dashboard_core involved at all. Demonstrates the
        # public API is self-sufficient for real VQE outside the dashboard.
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits, seed=11)

        rng = np.random.default_rng(11)
        theta = rng.uniform(-np.pi, np.pi, n_params)
        m, v = np.zeros(n_params), np.zeros(n_params)
        lr, beta1, beta2 = 0.1, 0.9, 0.999

        energy_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))
        energies = []
        for epoch in range(1, 41):
            (energy, _), grad = energy_and_grad(jnp.asarray(theta), h_matrix)
            energies.append(float(energy))
            grad = np.asarray(grad)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            theta -= (lr / (np.sqrt(v_hat) + 1e-8)) * m_hat

        assert np.mean(energies[-5:]) < np.mean(energies[:5])

    def test_from_pennylane_circuit_is_now_differentiable(self):
        # This is the gap found while auditing the interop bridge last
        # turn: jax.grad through run_pennylane_circuit silently returns 0.0
        # because from_pennylane bakes theta into a plain float. Composed
        # through circuit_to_energy_fn instead (which works on the
        # QASMCircuit from_pennylane returns, before that float baking
        # matters), a real circuit must now show a non-zero gradient.
        pennylane = pytest.importorskip("pennylane")
        import pennylane as qml

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta[0], wires=0)
            qml.RX(theta[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        theta0 = np.array([0.5, 0.5])
        circ = de.from_pennylane(circuit, theta0)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        assert n_params == 2

        h_matrix = _random_hamiltonian(circ.n_qubits, seed=5)

        def energy(theta):
            e, _ = energy_fn(theta, h_matrix)
            return e

        grad = jax.grad(energy)(jnp.asarray(theta0))
        assert float(jnp.linalg.norm(grad)) > 1e-6


class TestUnrecognizedGateRaises:
    """BUG FIX: _build_template used to silently `continue` past any op
    whose gate name has no GATE_IDS entry -- e.g. 'u2'/'u3' (multi-
    parameter gates this template's single-param-per-row shape can't
    represent, unlike GATE_IDS's other entries) -- dropping the gate
    from the traced circuit entirely with no error, giving a silently
    WRONG energy/gradient. Now raises instead of dropping."""

    def test_u2_gate_raises_instead_of_silently_dropping(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; u2(0.3,0.7) q[0];'
        circ = de.QASMParser().parse(qasm)
        with pytest.raises(ValueError, match="u2"):
            de.circuit_to_energy_fn(circ, circ.n_qubits)

    def test_u3_gate_raises_instead_of_silently_dropping(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; u3(0.3,0.7,1.1) q[0];'
        circ = de.QASMParser().parse(qasm)
        with pytest.raises(ValueError, match="u3"):
            de.circuit_to_energy_fn(circ, circ.n_qubits)

    def test_known_gates_still_work_unaffected(self):
        # Regression check: the error path must not affect circuits that
        # only use gates with a real GATE_IDS entry.
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        assert n_params == 5


class TestEnergyFnNoiseSpec:
    """circuit_to_energy_fn's `noise=` argument (registry.NoiseSpec, a
    JAX PyTree): applies NoiseModel.apply_to_sv natively inside the same
    traced computation as theta, instead of as an external Python-side
    step the caller has to splice in around energy_fn. jax_key is a
    pytree leaf (not an aux_data/static field), so it flows through
    jit/grad/vmap the way any other JAX array does -- no external
    key-management workaround, no OS-entropy fallback."""

    def test_default_none_matches_pre_noise_behavior(self):
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(1).uniform(-np.pi, np.pi, n_params))
        e_no_arg, _ = energy_fn(theta, h_matrix)
        e_explicit_none, _ = energy_fn(theta, h_matrix, None, None)
        assert float(e_no_arg) == pytest.approx(float(e_explicit_none))

    def test_same_key_is_reproducible(self):
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(2).uniform(-np.pi, np.pi, n_params))
        noise = de.NoiseSpec(model='depolarizing', p=0.15, jax_key=jax.random.PRNGKey(42))
        e1, _ = energy_fn(theta, h_matrix, noise=noise)
        e2, _ = energy_fn(theta, h_matrix, noise=noise)
        assert float(e1) == pytest.approx(float(e2))

    def test_noisy_energy_differs_from_ideal(self):
        # p=0.9, jax_key=PRNGKey(0) verified to fire real noise for this
        # circuit's 2 qubits. A single trajectory can legitimately draw
        # zero errors on all qubits (probability (1-p)^n_qubits, ~1% even
        # at p=0.9 here) -- registry.py's apply_to_sv used to draw
        # 2**(n_qubits-1) independent per-branch decisions per qubit,
        # which made SOME branch differing from ideal almost certain
        # regardless of p/seed; now it's one real Bernoulli(p) draw per
        # qubit per shot, so an arbitrary (p, seed) pair is no longer
        # guaranteed to show a deviation -- this test needs one verified
        # to actually fire, not just any nonzero p.
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(3).uniform(-np.pi, np.pi, n_params))
        e_ideal, _ = energy_fn(theta, h_matrix)
        noise = de.NoiseSpec(model='depolarizing', p=0.9, jax_key=jax.random.PRNGKey(0))
        e_noisy, _ = energy_fn(theta, h_matrix, noise=noise)
        assert float(e_ideal) != pytest.approx(float(e_noisy))

    def test_zero_probability_reduces_to_ideal_even_when_traced(self):
        # p=0.0 passed as a *traced* pytree leaf (under jax.jit) used to
        # crash with TracerBoolConversionError -- apply_to_sv's early
        # `if p <= 0.0: return sv` shortcut couldn't take a Python bool
        # of a tracer. Fixed to try/except around that optimization only;
        # every channel's math already reduces to a no-op at p=0
        # (`fire = r < p` is always False), so falling through instead of
        # short-circuiting is still correct.
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(4).uniform(-np.pi, np.pi, n_params))
        e_ideal, _ = energy_fn(theta, h_matrix)

        @jax.jit
        def run(p):
            noise = de.NoiseSpec(model='depolarizing', p=p, jax_key=jax.random.PRNGKey(1))
            e, _ = energy_fn(theta, h_matrix, None, noise)
            return e

        e_zero = run(0.0)
        assert float(e_zero) == pytest.approx(float(e_ideal))

    def test_composable_with_jax_jit(self):
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(5).uniform(-np.pi, np.pi, n_params))
        noise = de.NoiseSpec(model='depolarizing', p=0.1, jax_key=jax.random.PRNGKey(3))

        e_eager, _ = energy_fn(theta, h_matrix, None, noise)
        e_jit, _ = jax.jit(energy_fn)(theta, h_matrix, None, noise)
        assert float(e_eager) == pytest.approx(float(e_jit))

    def test_composable_with_jax_grad_through_noise(self):
        # gradient w.r.t. theta must flow through the noisy pipeline,
        # not just the ideal one
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(6).uniform(-np.pi, np.pi, n_params))
        noise = de.NoiseSpec(model='depolarizing', p=0.1, jax_key=jax.random.PRNGKey(9))

        def energy(t):
            e, _ = energy_fn(t, h_matrix, None, noise)
            return e

        grad = jax.grad(energy)(theta)
        assert float(jnp.linalg.norm(grad)) > 0.0

    def test_composable_with_jax_vmap_over_keys(self):
        # a batch of independent, reproducible noise realizations,
        # natively -- no external Python loop managing keys
        #
        # p=0.9 (not 0.15): post-fix, apply_to_sv draws one real
        # Bernoulli(p) decision per qubit per shot instead of the old
        # buggy 2**(n_qubits-1) independent per-branch decisions, so a
        # low p has a real, non-negligible chance every key in a small
        # batch independently draws zero errors on both qubits --
        # verified directly: p=0.15 with this exact PRNGKey(0)/6-key
        # split gave all 6 keys the same (zero-error) outcome. p=0.9
        # keeps that chance below ~1% per key.
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(8).uniform(-np.pi, np.pi, n_params))
        keys = jax.random.split(jax.random.PRNGKey(0), 6)

        def run(k):
            noise = de.NoiseSpec(model='depolarizing', p=0.9, jax_key=k)
            e, _ = energy_fn(theta, h_matrix, None, noise)
            return e

        batch = jax.vmap(run)(keys)
        assert batch.shape == (6,)
        # independent keys must not all collapse to the same value
        assert float(jnp.std(batch)) > 0.0

    def test_qubits_subset_still_works(self):
        circ = de.QASMParser().parse(VQE_QASM)
        energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
        h_matrix = _random_hamiltonian(circ.n_qubits)
        theta = jnp.asarray(np.random.default_rng(9).uniform(-np.pi, np.pi, n_params))
        noise = de.NoiseSpec(model='bitflip', p=0.5, jax_key=jax.random.PRNGKey(2), qubits=[0])
        e, sv = energy_fn(theta, h_matrix, None, noise)
        assert sv.shape == (2 ** circ.n_qubits,)


class TestImportSafety:

    def test_root_import_never_fails(self):
        assert hasattr(de, 'circuit_to_energy_fn')

    def test_missing_jax_raises_clear_importerror(self, monkeypatch):
        monkeypatch.setattr(autodiff, 'HAS_JAX', False)
        with pytest.raises(ImportError, match='JAX'):
            autodiff.circuit_to_energy_fn(None, 1)
