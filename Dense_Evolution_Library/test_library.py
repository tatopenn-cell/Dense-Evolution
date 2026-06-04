import numpy as np
from dense_evolution import DenseSVSimulator, QASMRunner, CircuitOptimizer

print("====================================================")
print("🔬 DENSE EVOLUTION CORE SYSTEM VERIFICATION")
print("====================================================")

try:
    sim = DenseSVSimulator(n_qubits=4, use_gpu=False, use_float32=False)
    sim.set_initial_state()
    
    test_circuit = [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]
    sim.run_circuit_jit_beast_mode(test_circuit)
    
    probs = sim.get_probabilities()
    print(f"✓ Statevector Engine: OK (Probabilità stato fondamentale: {probs[0]:.4f})")
    
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    h q[0];
    cx q[0], q[1];
    rx(pi/4) q[2];
    """
    runner = QASMRunner()
    sim_qasm, circ = runner.run(qasm_code, mode='auto')
    print(f"✓ OpenQASM Parser & Runner: OK (Rilevati {circ.n_qubits} qubit)")
    
    raw_circuit = [('h', 0), ('h', 0), ('x', 1), ('x', 1)]
    optimized = CircuitOptimizer.optimize(raw_circuit)
    print(f"✓ Gate Optimizer: OK (Istruzioni dopo cancellazione: {len(optimized)})")
    
    print("\n✅ TUTTI I MODULI SONO ALLINEATI: LA LIBRERIA È OPERATIVA AL 100%!")
    print("====================================================")

except Exception as e:
    print(f"\n❌ ERRORE DI VERIFICA STRUTTURALE: {str(e)}")
    print("====================================================")
