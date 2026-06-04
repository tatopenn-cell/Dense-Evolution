import time
import numpy as np

# Importiamo direttamente dai file che sono nella stessa cartella!
from .simulator import DenseSVSimulator
from .parser import QASMParser
from .compiler import QuantumTranspiler
from .registry import NoiseModel

print("====================================================")
print("🔬 ADVANCED PRODUCTION STRESS-TEST: KRAUS & JIT SIMULATION")
print("====================================================")

try:
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
    
    print("1. Parsing e Transpilazione del circuito QASM...")
    parser = QASMParser()
    circ = parser.parse(qasm_bench)
    tuples = QuantumTranspiler.transpile(circ.to_tuples())
    n_qubits = circ.n_qubits
    print(f"✓ Parsing completato. Qubit: {n_qubits}, Gate transpilati: {len(tuples)}")

    print("\n2. Validazione Core JIT in modalità ideale (simulator.py)...")
    t0 = time.perf_counter()
    sim_ideale = DenseSVSimulator(n_qubits)
    sim_ideale.run_circuit_jit_beast_mode(tuples)
    t_ideale = time.perf_counter() - t0
    prob_ideale = sim_ideale.get_probabilities()
    print(f"✓ Successo ideale in {t_ideale:.4f} s | Dimensione stato: {len(prob_ideale)}")

    print("\n3. Validazione NoiseModel (registry.py): Canale stocastico Amplitude Damping...")
    t0 = time.perf_counter()
    sim_noisy1 = DenseSVSimulator(n_qubits)
    sim_noisy1.run_circuit_jit_beast_mode(tuples)
    sim_noisy1.sv = NoiseModel.apply_to_sv(sim_noisy1.sv, n_qubits, model='amplitude_damping', p=0.15)
    prob_noisy1 = sim_noisy1.get_probabilities()
    t_noisy1 = time.perf_counter() - t0
    
    sim_noisy2 = DenseSVSimulator(n_qubits)
    sim_noisy2.run_circuit_jit_beast_mode(tuples)
    sim_noisy2.sv = NoiseModel.apply_to_sv(sim_noisy2.sv, n_qubits, model='amplitude_damping', p=0.15)
    prob_noisy2 = sim_noisy2.get_probabilities()
    
    print(f"✓ Successo noisy in {t_noisy1:.4f} s")

    print("\n4. Analisi statistica delle fluttuazioni (Shot Noise Reale)...")
    scostamento_stocastico = np.linalg.norm(prob_noisy1 - prob_noisy2)
    scostamento_ideale = np.linalg.norm(prob_ideale - prob_noisy1)
    
    print(f"🔬 Distanza ideale-noisy: {scostamento_ideale:.6f}")
    print(f"🔬 Fluttuazione quantistica indipendente (Shot Noise): {scostamento_stocastico:.6f}")

    if scostamento_stocastico > 1e-12:
        print("\n✅ TEST SUPERATO: Moduli collegati, PRNG JAX funzionante e rumore autenticamente non-deterministico!")
    else:
        print("\n❌ FALLIMENTO: Il rumore è ancora lineare o statico. Le due run hanno prodotto stati identici.")

    print("====================================================")

except Exception as e:
    print(f"\n❌ CRASH DURANTE L'ELABORAZIONE HARDWARE: {str(e)}")
    print("====================================================")
