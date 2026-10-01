"""
QASM circuit library and MPS-engine gate dispatch.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import numpy as np

import dense_evolution.gates as _mps_gates_module
from dense_evolution.mps import MPSSimulator


def _run_on_mps(comandi_beast_mode, n_qubits, max_bond=64, jsd_budget=1e-5):
    """Executes comandi_beast_mode (the same [name, *args] list the dense
    beast-mode path consumes) through MPSSimulator, gate by gate, via
    dense_evolution.gates' GATES/PARAMETRIC_GATES tables (converted from
    JAX to numpy -- MPSSimulator's einsum core is plain numpy).

    u2/u3 are skipped with a warning: comandi_beast_mode only carries a
    single parameter slot per gate (see the parsing loop in
    _run_simulation_body), which can't represent u2's 2 or u3's 3
    independent parameters -- the same pre-existing limitation the dense
    beast-mode path already has for these two gates, not something new
    introduced here."""
    mps = MPSSimulator(n_qubits=n_qubits, max_bond=max_bond, jsd_budget=jsd_budget)
    for cmd in comandi_beast_mode:
        nome_porta = cmd[0]
        if nome_porta in ('h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg'):
            gate = np.array(_mps_gates_module.GATES[nome_porta])
            mps.apply_gate_1q(gate, cmd[1])
        elif nome_porta in ('rx', 'ry', 'rz', 'u1', 'p'):
            gate = np.array(_mps_gates_module.PARAMETRIC_GATES[nome_porta](cmd[2]))
            mps.apply_gate_1q(gate, cmd[1])
        elif nome_porta in ('u2', 'u3'):
            print(f"Warning: motore MPS salta la porta '{nome_porta}' (richiede piu' "
                  "di un parametro, non rappresentabile nel formato beast-mode corrente).")
        elif nome_porta == 'cx':
            mps.apply_cx(cmd[1], cmd[2])
        elif nome_porta == 'cz':
            mps.apply_cz(cmd[1], cmd[2])
        elif nome_porta == 'swap':
            mps.apply_swap(cmd[1], cmd[2])
        elif nome_porta == 'cy':
            gate = np.array(_mps_gates_module.GATES['cy']).reshape(2, 2, 2, 2)
            mps.apply_gate_2q(gate, cmd[1], cmd[2])
        elif nome_porta in ('cp', 'crz'):
            gate = np.array(_mps_gates_module.PARAMETRIC_GATES[nome_porta](cmd[3])).reshape(2, 2, 2, 2)
            mps.apply_gate_2q(gate, cmd[1], cmd[2])
        elif nome_porta in ('ccx', 'toffoli'):
            mps.apply_ccx(cmd[1], cmd[2], cmd[3])
        else:
            print(f"Warning: motore MPS salta la porta non gestita: {nome_porta}")
    return mps


QASM_LIBRARY = {
    'Bell |Φ+⟩': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;',
    'QFT 4 qubit': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; ry(pi/4) q[0]; ry(pi/4) q[2]; h q[3]; cp(pi/2) q[2],q[3]; cp(pi/4) q[1],q[3]; cp(pi/8) q[0],q[3]; h q[2]; cp(pi/2) q[1],q[2]; cp(pi/4) q[0],q[2]; h q[1]; cp(pi/2) q[0],q[1]; h q[0]; swap q[0],q[3]; swap q[1],q[2]; barrier q; measure q -> c;',
    'Simon_Algorithm_4q_s11':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; '
    '// Simon: f(x)=f(x+11), q[0:1]=input, q[2:3]=output\n'
    'h q[0]; h q[1]; barrier q; '
    'cx q[0],q[2]; cx q[1],q[3]; cx q[0],q[3]; barrier q; '
    'h q[0]; h q[1]; '
    'measure q[0] -> c[0]; measure q[1] -> c[1]; '
    'measure q[2] -> c[2]; measure q[3] -> c[3];',
    'Grover_3q_Oracle_111':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[3]; '
    '// Init\n'
    'h q[0]; h q[1]; h q[2]; barrier q; '
    '// Oracle CCZ on |111>\n'
    'h q[2]; ccx q[0],q[1],q[2]; h q[2]; barrier q; '
    '// Diffuser\n'
    'h q[0]; h q[1]; h q[2]; '
    'x q[0]; x q[1]; x q[2]; '
    'h q[2]; ccx q[0],q[1],q[2]; h q[2]; '
    'x q[0]; x q[1]; x q[2]; '
    'h q[0]; h q[1]; h q[2]; '
    'barrier q; measure q -> c;',
    'Dicke_State_D42':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; '
    '// Dicke |D(4,2)> approx via SCS\n'
    'h q[0]; h q[1]; h q[2]; h q[3]; '
    'cx q[0],q[1]; cx q[2],q[3]; '
    'rz(1.5708) q[1]; rz(1.5708) q[3]; '
    'cx q[0],q[1]; cx q[2],q[3]; '
    'ry(0.9553) q[0]; ry(0.9553) q[2]; '
    'cx q[0],q[2]; '
    'barrier q; measure q -> c;',
    'MultiControlled_Z_5q':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[5]; '
    '// 4-controlled Z via T-gate decomposition\n'
    'h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; '
    'ccx q[0],q[1],q[3]; '
    'ccx q[2],q[3],q[4]; '
    't q[0]; t q[1]; t q[2]; tdg q[3]; tdg q[4]; '
    'cx q[0],q[1]; cx q[2],q[3]; '
    'tdg q[1]; t q[3]; '
    'cx q[0],q[1]; cx q[2],q[3]; '
    'ccx q[1],q[2],q[4]; '
    'h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; '
    'barrier q; measure q -> c;',
    'Anyonic_Braiding_Fibonacci_6q':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[6]; creg c[6]; '
    '// Fibonacci anyon braiding: sigma_1 sigma_2 sequence\n'
    'h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; h q[5]; '
    'cz q[0],q[1]; ry(1.2566) q[1]; '
    'cz q[1],q[2]; ry(1.2566) q[2]; '
    'cz q[0],q[2]; ry(0.9425) q[0]; '
    'cz q[2],q[3]; ry(1.2566) q[3]; '
    'cz q[3],q[4]; ry(1.2566) q[4]; '
    'cz q[2],q[4]; '
    'rz(3.0718) q[1]; rz(3.0718) q[3]; '
    'cx q[0],q[5]; cx q[2],q[5]; cx q[4],q[5]; '
    'rz(0.7) q[0]; rz(0.7) q[2]; rz(0.7) q[4]; '
    'barrier q; measure q -> c;',
    'Peptide_Furin_RRAR_8q':
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; '
    '// SARS-CoV-2 furin site: RRAR|S PDF-encoded\n'
    'ry(0.8727) q[0]; ry(0.5236) q[1]; ry(0.8727) q[2]; ry(0.8727) q[3]; '
    'ry(0.3927) q[4]; ry(0.2094) q[5]; ry(0.7854) q[6]; ry(0.4712) q[7]; '
    'rz(1.9106) q[0]; rz(1.9106) q[1]; rz(1.9106) q[2]; rz(1.9106) q[3]; '
    'rz(1.9106) q[4]; rz(1.9106) q[5]; rz(1.9106) q[6]; rz(1.9106) q[7]; '
    'cx q[0],q[1]; cx q[2],q[3]; cx q[4],q[5]; cx q[6],q[7]; '
    'cx q[1],q[2]; cx q[3],q[4]; cx q[5],q[6]; '
    'rz(0.7) q[0]; rz(0.7) q[1]; rz(0.7) q[2]; rz(0.7) q[3]; '
    'rz(0.7) q[4]; rz(0.7) q[5]; rz(0.7) q[6]; rz(0.7) q[7]; '
    'barrier q; measure q -> c;',
    'Grover Motif Search (0011)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; h q[2]; h q[3]; x q[2]; x q[3]; h q[3]; ccx q[0],q[2],q[3]; cx q[1],q[3]; h q[3]; x q[2]; x q[3]; h q[0]; h q[1]; h q[2]; h q[3]; x q[0]; x q[1]; x q[2]; x q[3]; h q[3]; ccx q[0],q[1],q[3]; cx q[2],q[3]; h q[3]; x q[0]; x q[1]; x q[2]; x q[3]; h q[0]; h q[1]; h q[2]; h q[3]; measure q -> c;',
    'Quantum Neural Neuron': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; ry(pi/4) q[0]; ry(pi/4) q[1]; ry(pi/4) q[2]; ry(pi/4) q[3]; barrier q; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; u3(0.1,0,0) q[0]; u3(0.5,0,0) q[1]; u3(-0.3,0,0) q[2]; u3(0.8,0,0) q[3]; measure q -> c;',
    'Approx QFT (Optimized)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[3]; cp(pi/2) q[2],q[3]; cp(pi/4) q[1],q[3]; h q[2]; cp(pi/2) q[1],q[2]; h q[1]; cp(pi/2) q[0],q[1]; h q[0]; barrier q; measure q -> c;',
    'Quantum Neural Layer': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; rz(pi/4) q[0]; rz(pi/4) q[1]; rz(pi/4) q[2]; rz(pi/4) q[3]; barrier q; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; cx q[3],q[0]; ry(0.5) q[0]; ry(0.5) q[1]; ry(0.5) q[2]; ry(0.5) q[3]; measure q -> c;',
    'Quantum Game Theory': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; cx q[0],q[1]; h q[0]; x q[1]; u3(pi/2,0,pi/2) q[0]; u3(pi/2,0,pi/2) q[1]; h q[0]; x q[1]; cx q[0],q[1]; measure q -> c;',
    'Quantum Teleportation': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[2]; h q[1]; cx q[1],q[2]; cx q[0],q[1]; h q[0]; measure q[0] -> c[0]; measure q[1] -> c[1]; x q[2] if(c[1]==1); z q[2] if(c[0]==1);',
    'Pixel Encoder (Phase)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[1]; h q[0]; h q[1]; cu1(pi/4) q[0],q[1]; measure q[1] -> c[0];',
    'Hardware Stress Test': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; h q[0]; barrier q[0]; id q[0]; id q[0]; id q[0]; measure q[0] -> c[0];',
    'Universal Swap Test (3q)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[1]; h q[0]; cswap q[0],q[1],q[2]; h q[0]; measure q[0] -> c[0];',
    'Bio-Sequence Matcher (8q)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; h q[5]; h q[6]; h q[7]; cx q[0],q[4]; cx q[1],q[5]; cp(pi/4) q[0],q[4]; cp(pi/2) q[1],q[5]; cx q[2],q[6]; cx q[3],q[7]; cp(pi/8) q[2],q[6]; x q[4]; x q[5]; x q[6]; x q[7]; h q[7]; ccx q[4],q[5],q[7]; ccx q[6],q[7],q[3]; h q[7]; x q[4]; x q[5]; x q[6]; x q[7]; h q[0]; h q[1]; h q[2]; h q[3]; x q[0]; x q[1]; x q[2]; x q[3]; h q[3]; ccx q[0],q[1],q[3]; h q[3]; x q[0]; x q[1]; x q[2]; x q[3]; h q[0]; h q[1]; h q[2]; h q[3]; measure q -> c;',
    'Grover AA Lys Search (K=01011)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[5]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; x q[0]; x q[2]; x q[4]; h q[4]; ccx q[0],q[1],q[4]; cx q[2],q[4]; cx q[3],q[4]; h q[4]; x q[0]; x q[2]; x q[4]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; x q[0]; x q[1]; x q[2]; x q[3]; x q[4]; h q[4]; ccx q[0],q[2],q[4]; cx q[1],q[4]; cx q[3],q[4]; h q[4]; x q[0]; x q[1]; x q[2]; x q[3]; x q[4]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; measure q -> c;',
    'Alpha-helix Detector (5q)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[5]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; cx q[0],q[2]; cx q[1],q[3]; rz(0.5) q[2]; rz(-0.25) q[3]; cx q[0],q[2]; cx q[1],q[3]; cx q[2],q[4]; cx q[3],q[4]; rz(0.785) q[4]; cx q[3],q[4]; cx q[2],q[4]; h q[0]; h q[1]; h q[2]; h q[3]; measure q -> c;',
    'HP Lattice 5-mer (10q)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[10]; creg c[10]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; h q[5]; h q[6]; h q[7]; h q[8]; h q[9]; cx q[0],q[2]; cx q[1],q[3]; cx q[4],q[6]; cx q[5],q[7]; rz(0.785) q[2]; rz(0.392) q[3]; rz(-0.392) q[6]; rz(-0.785) q[7]; cx q[0],q[2]; cx q[1],q[3]; cx q[4],q[6]; cx q[5],q[7]; cx q[2],q[4]; cx q[3],q[5]; cx q[6],q[8]; cx q[7],q[9]; rz(0.5) q[4]; rz(-0.5) q[5]; rz(0.25) q[8]; rz(-0.25) q[9]; cx q[2],q[4]; cx q[3],q[5]; cx q[6],q[8]; cx q[7],q[9]; measure q -> c;',
    'VQE BeH2 (8q-UCCSD)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; x q[0]; x q[1]; x q[2]; h q[0]; h q[1]; h q[2]; h q[3]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; cx q[3],q[4]; rz(0.18) q[4]; rz(0.09) q[6]; cx q[3],q[4]; cx q[2],q[3]; cx q[1],q[2]; cx q[0],q[1]; h q[4]; cx q[4],q[5]; cx q[5],q[6]; cx q[6],q[7]; rz(-0.18) q[7]; cx q[6],q[7]; cx q[5],q[6]; cx q[4],q[5]; h q[4]; measure q -> c;',
    'Bernstein-Vazirani (101)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; h q[2]; x q[3]; h q[3]; cx q[0],q[3]; cx q[2],q[3]; h q[0]; h q[1]; h q[2]; measure q -> c;',
    'Deutsch-Jozsa bilanciata': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[3]; h q[0]; h q[1]; x q[2]; h q[2]; cx q[0],q[2]; cx q[1],q[2]; h q[0]; h q[1]; measure q -> c;',
    'Toffoli (CCX)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[3]; h q[2]; cx q[1],q[2]; tdg q[2]; cx q[0],q[2]; t q[2]; cx q[1],q[2]; tdg q[2]; cx q[0],q[2]; t q[1]; t q[2]; h q[2]; cx q[0],q[1]; t q[0]; tdg q[1]; cx q[0],q[1]; measure q -> c;',
    'Grover 3q target 101': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[3]; h q[0]; h q[1]; h q[2]; x q[0]; x q[1]; h q[2]; ccx q[0],q[1],q[2]; h q[2]; x q[0]; x q[1]; h q[0]; h q[1]; x q[0]; x q[1]; h q[1]; cx q[0],q[1]; h q[1]; x q[0]; x q[1]; h q[0]; h q[1]; measure q->c;',
    'VQE ansatz H₂': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; ry(0.5) q[0]; rx(0.5) q[1]; cx q[0],q[1]; rz(0.2) q[1]; cx q[0],q[1]; ry(0.5) q[0]; rx(0.5) q[1]; measure q -> c;',
    'Adder 2-bit': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[5]; cx q[0],q[3]; cx q[1],q[3]; ccx q[0],q[1],q[4]; cx q[2],q[4]; cx q[3],q[4]; measure q -> c;',
    'Quantum Supremacy (toy)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[5]; h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; cx q[3],q[4]; rx(pi/4) q[0]; ry(pi/4) q[1]; rz(pi/4) q[2]; rx(pi/4) q[3]; ry(pi/4) q[4]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; cx q[3],q[4]; measure q -> c;',
    'Random Entanglement': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; h q[2]; h q[3]; cx q[0],q[1]; rz(0.3) q[1]; cx q[2],q[3]; ry(0.7) q[3]; cx q[1],q[2]; measure q -> c;',
    'HP Lattice 3-mer (6q)': '''include "qelib1.inc";
// HP Lattice 3-mer: HHP
// 2 turn angles * 2 qubit + 2 ancilla
qreg q[6];
creg c[6];
h q[0]; h q[1]; h q[2]; h q[3];
cx q[0],q[1]; cx q[2],q[3];
rz(0.785) q[1]; rz(-0.392) q[3];
cx q[0],q[1]; cx q[2],q[3];
cx q[1],q[4]; cx q[3],q[5];
rz(0.5) q[4]; rz(-0.25) q[5];
cx q[1],q[4]; cx q[3],q[5];
h q[4]; h q[5];
measure q->c;''',
    'NH3 Complex (4q)': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
// Complex NH3 Ansatz (4-qubit)
// Layer 1
ry(pi/2) q[0];
rz(pi/4) q[1];
ry(pi/2) q[2];
rz(pi/4) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];

// Layer 2
ry(0.6) q[0];
rz(0.3) q[1];
ry(0.9) q[2];
rz(0.5) q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];

// Layer 3
ry(0.1) q[0];
rz(0.8) q[1];
ry(0.4) q[2];
rz(0.7) q[3];
cx q[0],q[2];
cx q[1],q[3];

// Layer 4 (Final Rotations)
ry(0.2) q[0];
rz(0.5) q[1];
ry(0.3) q[2];
rz(0.6) q[3];

measure q -> c;''',
    'Amplitude Estimation (Finance)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[3]; h q[0]; h q[1]; h q[2]; ry(pi/4) q[3]; cp(pi/2) q[2],q[3]; cp(pi) q[1],q[3]; h q[0]; cp(-pi/2) q[0],q[1]; h q[1]; cp(-pi/4) q[0],q[2]; cp(-pi/2) q[1],q[2]; h q[2]; measure q[0:2] -> c;',
    'VQE Water (FORCE)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[10]; creg c[10]; h q[0]; h q[1]; h q[2]; h q[3]; ry(0.5) q[0]; ry(1.0) q[1]; ry(1.5) q[2]; ry(2.0) q[3]; cx q[0],q[1]; cx q[2],q[3]; measure q -> c;',
    'VQE LiH Ansatz': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; x q[0]; x q[1]; ry(0.15) q[0]; ry(0.15) q[1]; cx q[0],q[2]; cx q[1],q[3]; ry(0.05) q[2]; ry(0.05) q[3]; measure q -> c;',
    'QFT 8q Safe-Scan': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; h q[0]; h q[1]; h q[2]; h q[3]; cp(pi/2) q[0],q[1]; cp(pi/4) q[1],q[2]; cp(pi/8) q[2],q[3]; h q[4]; h q[5]; h q[6]; h q[7]; measure q -> c;',
    'HHL Matrix Inversion': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[2]; h q[1]; h q[2]; cp(pi/2) q[1],q[3]; cp(pi) q[2],q[3]; ch q[1],q[0]; ch q[2],q[0]; measure q[1:2] -> c;',
    'QAOA Max-Cut 4q (Pro)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; h q[2]; h q[3]; cx q[0],q[1]; rz(1.57) q[1]; cx q[0],q[1]; cx q[2],q[3]; rz(1.57) q[3]; cx q[2],q[3]; rx(0.78) q[0]; rx(0.78) q[1]; rx(0.78) q[2]; rx(0.78) q[3]; measure q -> c;',
    'QML ZZ-FeatureMap': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; h q[1]; rz(0.5) q[0]; rz(1.2) q[1]; cx q[0],q[1]; rz(0.6) q[1]; cx q[0],q[1]; ry(0.2) q[0]; ry(0.2) q[1]; measure q -> c;',
    'QPE Precision 5q': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; creg c[4]; h q[0]; h q[1]; h q[2]; h q[3]; x q[4]; cp(pi/8) q[0],q[4]; cp(pi/4) q[1],q[4]; cp(pi/2) q[2],q[4]; cp(pi) q[3],q[4]; h q[0]; cp(-pi/2) q[0],q[1]; h q[1]; measure q[0:3] -> c;',
    'Interference Stress Test': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; ry(0.9) q[1]; cx q[0],q[1]; rz(1.3) q[1]; h q[0]; measure q -> c;',
    'Shor 15 (Simplified)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; h q[0:3]; x q[4]; cx q[2],q[5]; cx q[2],q[6]; h q[0:3]; measure q -> c;',
    'Ising Model Simulation': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; rz(0.5) q[0]; cx q[0],q[1]; rz(0.3) q[1]; cx q[1],q[0]; measure q -> c;',
    'Grover 4-item Search': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; h q[1]; x q[1]; h q[1]; cx q[0],q[1]; h q[1]; x q[1]; h q[0]; h q[1]; x q[0]; x q[1]; h q[1]; cx q[0],q[1]; h q[1]; x q[0]; x q[1]; h q[0]; h q[1]; measure q -> c;',
    'VQE Water (H2O) 6q': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[6]; creg c[6]; x q[0]; x q[1]; ry(0.1) q[0]; ry(0.1) q[1]; cx q[0],q[2]; cx q[1],q[3]; cx q[2],q[4]; cx q[3],q[5]; measure q -> c;',
    'Quantum Key Distribution': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; h q[0]; h q[1]; measure q -> c;',
    'QFT 8 qubit High-Res': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[8]; creg c[8]; h q[7]; cp(pi/2) q[6],q[7]; cp(pi/4) q[5],q[7]; cp(pi/8) q[4],q[7]; h q[6]; cp(pi/2) q[5],q[6]; measure q -> c;',
    'Quantum Walk FORCE': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; h q[1]; h q[2]; cx q[0],q[1]; ccx q[1],q[2],q[3]; rz(1.5) q[3]; measure q -> c;',
    'Error Mitigation (Real-Stress)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[15]; creg c[15]; h q[0]; cx q[0],q[1]; rz(0.45) q[1]; cx q[0],q[1]; rz(0.45) q[1]; cx q[0],q[1]; h q[0]; measure q -> c;',
    'Thermalizer VQT': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; ry(0.4) q[0]; ry(0.8) q[1]; ry(1.2) q[2]; ry(1.6) q[3]; cx q[0],q[1]; cx q[2],q[3]; measure q -> c;',
    'Ising Spectrum (Multi-Color)': 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; h q[0]; ry(0.8) q[1]; cx q[0],q[1]; rz(pi/4) q[0]; h q[2]; cx q[1],q[2]; measure q -> c;',
    'NH3 Simplified (4q)': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
// Simplified NH3 VQE-like ansatz
ry(pi/2) q[0];
rz(pi/4) q[1];
ry(pi/2) q[2];
rz(pi/4) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
ry(0.5) q[0];
rz(0.3) q[1];
ry(0.7) q[2];
rz(0.4) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
measure q -> c;''',
    'Beta-sheet pattern detector (5q)': '''OPENQASM 2.0;
include "qelib1.inc";
// Beta-sheet pattern detector
// Sheet residues: V(19),I(9),F(13),Y(18),W(17)
qreg q[5];
creg c[5];
h q[0]; h q[1]; h q[2]; h q[3]; h q[4];
x q[0]; x q[2];
cx q[0],q[4]; cx q[2],q[4];
rz(0.785) q[4];
cx q[0],q[4]; cx q[2],q[4];
x q[0]; x q[2];
h q[0]; h q[1]; h q[2]; h q[3]; h q[4];
x q[0]; x q[1]; x q[2]; x q[3]; x q[4];
cx q[0],q[4]; cx q[1],q[4];
x q[0]; x q[1]; x q[2]; x q[3]; x q[4];
h q[0]; h q[1]; h q[2]; h q[3]; h q[4];
measure q->c;''',
}


def infer_qubit_count_from_qasm(qasm_text: str):
    """Cheap qubit-count guess from raw QASM text (regex on `qreg q[N]`),
    used only to pre-filter the Hamiltonian-compatibility dropdown in the UI
    before a circuit actually runs. The real, authoritative qubit count
    always comes from run_simulation()'s res['n_qubits'] (full QASM parse);
    this is not used for execution. Returns None if no qreg declaration is found."""
    import re
    match = re.search(r'qreg\s+\w+\s*\[\s*(\d+)\s*\]', qasm_text or '')
    return int(match.group(1)) if match else None
