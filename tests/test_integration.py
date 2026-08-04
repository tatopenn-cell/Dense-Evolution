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

        sim_noisy1 = DenseSVSimulator(n_qubits)
        sim_noisy1.run_circuit_jit(tuples)
        sim_noisy1.sv = NoiseModel.apply_to_sv(sim_noisy1.sv, n_qubits, model='amplitude_damping', p=0.15)
        prob_noisy1 = sim_noisy1.get_probabilities()

        sim_noisy2 = DenseSVSimulator(n_qubits)
        sim_noisy2.run_circuit_jit(tuples)
        sim_noisy2.sv = NoiseModel.apply_to_sv(sim_noisy2.sv, n_qubits, model='amplitude_damping', p=0.15)
        prob_noisy2 = sim_noisy2.get_probabilities()

        # Kraus noise must be genuinely stochastic: two independent
        # applications of the same channel to the same clean state must
        # not produce identical output.
        stochastic_spread = float(np.linalg.norm(prob_noisy1 - prob_noisy2))
        assert stochastic_spread > 1e-12
