"""
End-to-end integration test spanning multiple modules (parser, compiler,
simulator, registry) wired together, rather than any one module in
isolation -- kept in its own file instead of forced into a single-module
test file for that reason.

Converted from dense_evolution/test2.py and dense_evolution/stress_test.py
(audit finding #5): two byte-identical, assertion-free print-and-eyeball
debug scripts that shipped inside every `pip install dense-evolution`
(via the package-data "*.py" glob), were 0% covered, and never ran in CI.
The one real signal they checked -- parser -> transpiler -> simulator ->
noise model wired together end to end, and Kraus noise application being
genuinely stochastic across independent runs -- is preserved here as a
real, CI-enforced test; both original scripts have been deleted.
"""
import numpy as np
import jax
import jax.numpy as jnp

from dense_evolution import DenseSVSimulator, NoiseModel, QASMParser, QuantumTranspiler


class TestFullPipelineIntegration:

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
        sim_ideale.run_circuit_jit(tuples)
        prob_ideale = sim_ideale.get_probabilities()
        assert abs(float(np.sum(prob_ideale)) - 1.0) < 1e-9

        # Kraus noise must be genuinely stochastic: independent applications
        # of the same channel to the same clean state must not all produce
        # identical output. Regression note: a single pair of draws used to
        # be enough to show this (pre-fix, `amplitude_damping` fired its
        # decay branch with a flat probability independent of the qubit's
        # actual |1> population -- see NoiseModel.apply_to_sv's docstring).
        # After that Born-rule fix, THIS specific circuit's post-rotation
        # state turns out to be sparse (only 4 of the 64 basis states have
        # any amplitude at all -- H+CX chain gives a 2-branch cat state,
        # and RX/RY only ever double the branch count, never spread over
        # the full space), so most of the per-qubit random draws compare
        # against an exactly-zero decay probability and can never fire --
        # only ~12 of the 192 draws across all 6 qubits are "live" at all.
        # Measured directly (500 trials, gamma=0.15): the single most
        # common outcome (no visible decay anywhere) occurs ~72% of the
        # time, so a bare 2-trial comparison collides more than half the
        # time -- a real, now-fixed CI flake, not a hypothetical one (it
        # reproduced on this exact seed-free path in GitHub Actions CI).
        # N=80 independent draws, asserting they are not ALL identical,
        # keeps the same physical check with a false-flake probability of
        # roughly 0.72**80 ~ 1e-12 -- functionally never, without needing
        # to change gamma or the circuit's structural assertions above.
        prob_trials = []
        for trial_seed in range(80):
            sv_trial = NoiseModel.apply_to_sv(
                sim_ideale.sv, n_qubits, model='amplitude_damping', p=0.15,
                jax_key=jax.random.PRNGKey(trial_seed),
            )
            prob_trials.append(np.asarray(jnp.abs(sv_trial) ** 2))

        all_identical = all(
            np.allclose(prob_trials[0], p, atol=1e-12) for p in prob_trials[1:]
        )
        assert not all_identical
