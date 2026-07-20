"""
dashboard_core.py — compute layer for app_dashboard.py's Quantum Simulator tab.

Adapted from dash.py (the ipywidgets/Colab notebook export). dash.py is not a
clean importable module (it runs a subprocess pip-install check, builds a full
ipywidgets UI, and does an unconditional `from google.colab import files` on
import), so its logic is extracted and adapted here into plain functions that
take explicit values instead of ipywidgets objects, with no import-time side
effects.
"""

import hashlib
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go

import dense_evolution as de
# Private but same ecosystem: DenseSVSimulator.run_parametric_batch_jit already
# calls this internally for the exact "inject a parameter without severing the
# JAX trace" pattern the real VQE gradient below reuses (see _vqe_energy_fn) —
# there's no public wrapper that takes a pre-built ops array directly.
from dense_evolution.compiler import _compile_and_run_circuit_jit


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

QM_MM_HEAVY_QUBIT_THRESHOLD = 12


LIBRERIA_HAMILTONIANE = {
    # ---------------------------------------------------------------------
    # --- GRUPPO 1: BENCHMARK CHIMICI REALI (2 QUBIT / DIM=4 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H2 (Idrogeno) - R = 0.50 Å [Compressione]": [-0.51, -0.12, 0.35, 0.85],
    "H2 (Idrogeno) - R = 0.74 Å [Equilibrio Reale]": [-1.13, -0.45, 0.12, 0.64],
    "H2 (Idrogeno) - R = 1.20 Å [Dissociazione]": [-0.92, -0.68, -0.15, 0.22],
    "HeH+ (Idruro di Elio) - R = 0.93 Å [Equilibrio]": [-1.41, -0.82, -0.22, 0.45],
    "H2 (Idrogeno) - R = 1.50 Å [Asintoto Dissoc.]": [-0.78, -0.65, -0.31, 0.11],
    "HeH+ (Idruro di Elio) - R = 0.50 Å [Stallo Interno]": [-0.22, 0.14, 0.76, 1.48],
    "HeH+ (Idruro di Elio) - R = 1.60 Å [Limite Ionico]": [-1.05, -0.91, -0.44, 0.02],
    "LiH (Idruro di Litio) - STO-3G (Sotto-spazio 2q)": [-1.62, -0.98, -0.11, 0.54],

    # ---------------------------------------------------------------------
    # --- GRUPPO 2: CHIMICA ED ENTANGLEMENT AVANZATO (3 QUBIT / DIM=8 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H3+ (Ione Triidrogeno Linear) - R = 0.85 Å": [-1.28, -0.94, -0.51, -0.08, 0.33, 0.76, 1.18, 1.55],
    "H3+ (Ione Triidrogeno Triang) - R = 0.90 Å": [-1.34, -1.01, -0.62, -0.14, 0.28, 0.69, 1.09, 1.42],
    "Modello di Lipkin (Fisica Nucleare 3q Baseline)": [-2.00, -1.41, -0.82, -0.22, 0.35, 0.91, 1.54, 2.11],
    "Catena di Heisenberg XXX (Antiferromagnetica 3q)": [-1.82, -1.22, -0.61, -0.11, 0.42, 0.93, 1.34, 1.85],

    # ---------------------------------------------------------------------
    # --- GRUPPO 3: MASSA MOLECOLARE INTERMEDIA (4 QUBIT / DIM=16 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "Modello Ising Lineare (4 Qubit Baseline)": [-1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5],
    "LiH (Idruro di Litio) - R = 1.40 Å [Minimo]": [-2.31, -2.01, -1.65, -1.22, -0.85, -0.41, 0.02, 0.44, 0.88, 1.25, 1.61, 1.98, 2.34, 2.71, 3.05, 3.42],
    "LiH (Idruro di Litio) - R = 2.20 Å [Torsione]": [-1.89, -1.62, -1.31, -0.98, -0.62, -0.22, 0.15, 0.52, 0.91, 1.28, 1.64, 1.99, 2.33, 2.68, 3.01, 3.35],
    "BH3 (Borano Parziale) - R = 1.15 Å": [-2.85, -2.42, -2.01, -1.55, -1.11, -0.65, -0.21, 0.22, 0.64, 1.05, 1.47, 1.88, 2.29, 2.69, 3.08, 3.49],
    "H4 (Catena di Idrogeno Quadrata) - R = 1.00 Å": [-2.14, -1.82, -1.44, -1.02, -0.61, -0.18, 0.24, 0.65, 1.08, 1.49, 1.91, 2.32, 2.72, 3.11, 3.51, 3.92],
    "H4 (Catena di Idrogeno Lineare) - R = 1.25 Å": [-2.45, -2.11, -1.72, -1.34, -0.92, -0.51, -0.08, 0.34, 0.76, 1.17, 1.58, 1.99, 2.38, 2.78, 3.16, 3.55],
    "Modello Hubbard (Sito 2x2, Half-Filling 4q)": [-3.21, -2.75, -2.24, -1.72, -1.18, -0.64, -0.11, 0.42, 0.95, 1.48, 2.01, 2.54, 3.06, 3.58, 4.11, 4.64],
    "Interazione Cooper (Superconduttività BC 4q)": [-1.95, -1.68, -1.41, -1.12, -0.84, -0.55, -0.24, 0.05, 0.36, 0.68, 0.98, 1.29, 1.58, 1.88, 2.19, 2.48],

    # ---------------------------------------------------------------------
    # --- GRUPPO 4: MACROMOLECOLE PRE-CALCOLATE (5-6 QUBIT / DIM=32-64 SPRAZIO DENSE)
    # ---------------------------------------------------------------------
    "H2O (Acqua Embedding Core) - R = 0.96 Å [32 Val]": [
        -4.12, -3.79, -3.47, -3.15, -2.83, -2.51, -2.19, -1.86, -1.54, -1.22, -0.90, -0.58, -0.26, 0.06, 0.38, 0.70,
         1.02,  1.34,  1.67,  1.99,  2.31,  2.63,  2.95,  3.27,  3.59,  3.91,  4.24,  4.56,  4.88, 5.20, 5.52, 5.85
    ],
    "NH3 (Ammoniaca Sotto-guscio) - R = 1.01 Å [32 Val]": [
        -4.85, -4.49, -4.13, -3.78, -3.42, -3.06, -2.71, -2.35, -2.00, -1.64, -1.28, -0.93, -0.57, -0.21, 0.14, 0.50,
         0.85,  1.21,  1.56,  1.92,  2.28,  2.63,  2.99,  3.35,  3.70,  4.06,  4.41,  4.77,  5.13, 5.48, 5.84, 6.22
    ],
    "CH4 (Metano Orbitale Ibrido) - R = 1.09 Å [32 Val]": [
        -5.12, -4.71, -4.31, -3.91, -3.51, -3.11, -2.71, -2.31, -1.90, -1.50, -1.10, -0.70, -0.30, 0.10, 0.50, 0.90,
         1.31,  1.71,  2.11,  2.51,  2.91,  3.31,  3.71,  4.11,  4.52,  4.92,  5.32,  5.72,  6.12, 6.52, 6.92, 7.34
    ],
    "BeH2 (Idruro di Berillio Active Space) [64 Val]": [
        -6.42, -6.19, -5.96, -5.73, -5.50, -5.27, -5.04, -4.81, -4.58, -4.35, -4.12, -3.89, -3.66, -3.43, -3.20, -2.97,
        -2.74, -2.51, -2.28, -2.05, -1.82, -1.59, -1.36, -1.13, -0.90, -0.67, -0.44, -0.21,  0.02,  0.25,  0.48,  0.71,
         0.94,  1.17,  1.40,  1.63,  1.86,  2.09,  2.32,  2.55,  2.78,  3.01,  3.24,  3.47,  3.70,  3.93,  4.16,  4.39,
         4.62,  4.85,  5.08,  5.31,  5.54,  5.77,  6.00,  6.23,  6.46,  6.69,  6.92,  7.15,  7.38,  7.61,  7.84,  8.11
    ],
    "N2 (Azoto Molecolare Singlet-State) [64 Val]": [
        -8.95, -8.63, -8.31, -7.99, -7.67, -7.35, -7.03, -6.71, -6.39, -6.07, -5.75, -5.43, -5.11, -4.79, -4.47, -4.15,
        -3.83, -3.51, -3.19, -2.87, -2.55, -2.23, -1.91, -1.59, -1.27, -0.95, -0.63, -0.31,  0.01,  0.33,  0.65,  0.97,
         1.29,  1.61,  1.93,  2.25,  2.57,  2.89,  3.21,  3.53,  3.85,  4.17,  4.49,  4.81,  5.13,  5.45,  5.77,  6.09,
         6.41,  6.73,  7.05,  7.37,  7.69,  8.01,  8.33,  8.65,  8.97,  9.29,  9.61,  9.93, 10.25, 10.57, 10.89, 11.24
    ],
    "HF (Acido Fluoridrico Valence Space) [64 Val]": [
        -7.14, -6.87, -6.61, -6.34, -6.08, -5.81, -5.54, -5.28, -5.01, -4.75, -4.48, -4.21, -3.95, -3.68, -3.42, -3.15,
        -2.88, -2.62, -2.35, -2.09, -1.82, -1.55, -1.29, -1.02, -0.76, -0.49, -0.22,  0.04,  0.31,  0.57,  0.84,  1.10,
         1.37,  1.64,  1.90,  2.17,  2.43,  2.70,  2.96,  3.23,  3.50,  3.76,  4.03,  4.29,  4.56,  4.82,  5.09,  5.35,
         5.62,  5.89,  6.15,  6.42,  6.68,  6.95,  7.21,  7.48,  7.75,  8.01,  8.28,  8.54,  8.81,  9.07,  9.34,  9.65
    ],

    "Spettro Uniforme Classico (Baseline linspace)": None,
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


def get_compatible_hamiltonians(n_qubits, library=None):
    """Filters a Hamiltonian library down to entries whose diagonal length
    matches 2**n_qubits. Adapted from update_hamiltonian_options_and_state's
    filter (dash.py:2825) — `values is not None and len(values) == expected_dim`,
    which is also why "Spettro Uniforme Classico" (value None) never actually
    appears as selectable in the original either; ported faithfully."""
    library = library if library is not None else LIBRERIA_HAMILTONIANE
    if n_qubits is None or n_qubits <= 0:
        return {}
    expected_dim = 2 ** n_qubits
    return {name: values for name, values in library.items()
            if values is not None and len(values) == expected_dim}


def save_custom_hamiltonian(library, name, values_json_str):
    """Validates and inserts a custom Hamiltonian into `library` (mutated in
    place). Adapted from core_trigger_save_hamiltonian (dash.py:2920), minus
    the blocking input() call for the name (replaced by an explicit `name`
    param — the UI layer collects it via st.text_input instead).
    Returns (success: bool, message: str)."""
    import json as _json

    if not name:
        return False, "Nome dell'Hamiltoniana non valido."
    if name in library:
        return False, f"Un'Hamiltoniana con il nome '{name}' esiste già. Scegli un nome diverso."
    try:
        values = _json.loads(values_json_str)
    except _json.JSONDecodeError:
        return False, "Errore di parsing JSON: assicurati che l'input sia un array JSON valido."
    if not isinstance(values, list) or not all(isinstance(x, (int, float)) for x in values):
        return False, "L'input deve essere un array JSON di numeri (es. [1.0, 2.0, 3.0])."

    library[name] = values
    return True, f"Hamiltoniana '{name}' salvata con successo!"


def estrai_valore_puro(elemento):
    if elemento is None:
        return 0
    tipo_str = str(type(elemento)).lower()
    if "builtin" in tipo_str or "method" in tipo_str or "function" in tipo_str:
        try:
            return estrai_valore_puro(elemento())
        except Exception:
            return 0
    if callable(elemento):
        try:
            return estrai_valore_puro(elemento())
        except Exception:
            pass
    if hasattr(elemento, 'index'):
        val = getattr(elemento, 'index')
        val_tipo = str(type(val)).lower()
        if "builtin" in val_tipo or "method" in val_tipo or callable(val):
            try: val = val()
            except Exception: pass
        if val is not elemento:
            return estrai_valore_puro(val)
    if hasattr(elemento, 'value'):
        val = getattr(elemento, 'value')
        val_tipo = str(type(val)).lower()
        if "builtin" in val_tipo or "method" in val_tipo or callable(val):
            try: val = val()
            except Exception: pass
        if val is not elemento:
            return estrai_valore_puro(val)
    if isinstance(elemento, str):
        try:
            if '.' in elemento:
                return float(elemento)
            return int(elemento)
        except ValueError:
            return elemento
    if isinstance(elemento, (int, float, np.integer, np.floating)):
        return elemento
    return elemento


def run_simulation(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32=True):
    """Adapted from core_calcolo_quantistico (dash.py:3356, canonical/later definition)."""
    previous_x64_state = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', not use_float32)
    try:
        return _run_simulation_body(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32)
    finally:
        # jax_enable_x64 is a process-wide flag: without restoring it here, a float32 run here
        # silently downgrades precision for unrelated code (e.g. the Vector Healing page) that
        # runs later in the same process and never sets its own precision.
        jax.config.update('jax_enable_x64', previous_x64_state)


def _run_simulation_body(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32):
    if source_mode == 'Libreria Built-in' and QASM_LIBRARY:
        qasm_string = QASM_LIBRARY[circuit_name]
        nome_circuito = circuit_name
    else:
        qasm_string = qasm_text
        nome_circuito = 'Custom Workspace'

    parser = de.QASMParser()
    parsed_circuit = parser.parse(qasm_string)
    comandi_originali = parsed_circuit.ops

    try:
        n_qubits = int(parsed_circuit.n_qubits)
    except Exception:
        n_qubits = 0

    if n_qubits <= 2 or n_qubits > 34:
        max_qubit_idx = -1
        for cmd in comandi_originali:
            if isinstance(cmd, dict) and 'qubits' in cmd:
                q_list = cmd['qubits']
                if isinstance(q_list, (list, tuple, np.ndarray)):
                    for q_item in q_list:
                        val_puro = estrai_valore_puro(q_item)
                        try:
                            idx_check = int(val_puro)
                            if idx_check > max_qubit_idx and idx_check < 40:
                                max_qubit_idx = idx_check
                        except Exception:
                            pass
                else:
                    try:
                        idx_check = int(estrai_valore_puro(q_list))
                        if idx_check > max_qubit_idx and idx_check < 40:
                            max_qubit_idx = idx_check
                    except Exception:
                        pass
        n_qubits = max_qubit_idx + 1 if max_qubit_idx != -1 else 4

    for token in nome_circuito.replace('_', ' ').split():
        if 'q' in token.lower() and token.lower().replace('q', '').isdigit():
            n_qubits = int(token.lower().replace('q', ''))
        elif 'd' in token.lower() and token.lower().replace('d', '').isdigit():
            n_qubits = int(token.lower().replace('d', ''))

    sim = de.DenseSVSimulator(n_qubits=n_qubits, use_gpu=False, use_float32=use_float32)

    comandi_beast_mode = []
    for cmd in comandi_originali:
        if not isinstance(cmd, dict) or 'name' not in cmd:
            continue
        nome_porta = str(cmd['name']).lower().strip()
        qubits_grezzi = cmd.get('qubits', [])
        params_grezzi = cmd.get('params', [])

        if nome_porta in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']:
            try:
                target = int(estrai_valore_puro(qubits_grezzi[0]))
                if target < n_qubits:
                    comandi_beast_mode.append([nome_porta, target, -1])
            except Exception: pass
        elif nome_porta in ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'p']:
            try:
                param = float(estrai_valore_puro(params_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[0]))
                if target < n_qubits:
                    comandi_beast_mode.append([nome_porta, target, param])
            except Exception: pass
        elif nome_porta in ['cx', 'cy', 'cz', 'swap']:
            try:
                control = int(estrai_valore_puro(qubits_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[1]))
                if control < n_qubits and target < n_qubits:
                    # compiler.py's documented tuple contract is (gate, control, target)
                    # for 2-qubit gates — this used to be reversed (see audit finding #1).
                    comandi_beast_mode.append([nome_porta, control, target])
            except Exception: pass
        elif nome_porta in ['ccx', 'toffoli']:
            try:
                c1 = int(estrai_valore_puro(qubits_grezzi[0]))
                c2 = int(estrai_valore_puro(qubits_grezzi[1]))
                t = int(estrai_valore_puro(qubits_grezzi[2]))
                if c1 < n_qubits and c2 < n_qubits and t < n_qubits:
                    comandi_beast_mode.append([nome_porta, c1, c2, t])
            except Exception: pass
        elif nome_porta in ['cp', 'crz']:
            try:
                param = float(estrai_valore_puro(params_grezzi[0]))
                control = int(estrai_valore_puro(qubits_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[1]))
                if control < n_qubits and target < n_qubits:
                    comandi_beast_mode.append([nome_porta, control, target, param])
            except Exception: pass

    start_time = time.perf_counter()

    if noise_model == 'ideal':
        sim.run_circuit_jit_beast_mode(comandi_beast_mode)
        prob_ideal = sim.get_probabilities()
        prob_noisy = prob_ideal
    else:
        np.random.seed(seed)
        sim_ideal = de.DenseSVSimulator(n_qubits=n_qubits, use_float32=use_float32)
        sim_ideal.run_circuit_jit_beast_mode(comandi_beast_mode)
        prob_ideal = sim_ideal.get_probabilities()

        sim.run_circuit_jit_beast_mode(comandi_beast_mode)
        if noise_p > 0:
            sim.sv = de.NoiseModel.apply_to_sv(
                sv=sim.sv, n=n_qubits, model=noise_model, p=float(noise_p)
            )
        prob_noisy = sim.get_probabilities()

    t_elapsed = time.perf_counter() - start_time

    if use_float32:
        prob = np.array(prob_noisy, dtype=np.float32)
        prob_id = np.array(prob_ideal, dtype=np.float32)
    else:
        prob = np.array(prob_noisy, dtype=np.float64)
        prob_id = np.array(prob_ideal, dtype=np.float64)

    shannon_entropy = -np.sum(prob * np.log2(prob + 1e-10))
    idx_max = np.argmax(prob)
    stato_dominante = format(idx_max, '0' + str(n_qubits) + 'b')

    fidelity_value = float(np.sum(np.sqrt(prob * prob_id)))
    noise_factor_curve = np.array([fidelity_value * (1.0 - (i * float(noise_p) / 20.0)) for i in range(100)])
    noise_factor_curve = np.clip(noise_factor_curve, 0.0, 1.0)

    shots_data = np.random.choice(len(prob), p=prob, size=shots)

    return {
        'prob': prob,
        'prob_ideal': prob_ideal,
        'noise_factor': noise_factor_curve,
        'fidelity': fidelity_value,
        'n_qubits': n_qubits,
        'entropy': shannon_entropy,
        'idx_max': idx_max,
        'stato_dominante': stato_dominante,
        'tempo': t_elapsed,
        'ram': sim.memory_mb(),
        'nome': nome_circuito,
        'porte_count': len(comandi_beast_mode),
        'shots_data': shots_data,
        'sim': sim,
        'parser': parser,
    }


class QMMMForceEngine:
    """Hellmann-Feynman QM/MM force engine via JAX autodiff. Adapted from dash.py:1455."""

    def __init__(self, simulator_instance):
        self.sim = simulator_instance
        self.dim = simulator_instance.dim
        self.n_qubits = simulator_instance.n

    def build_loss_function(self):
        def qm_mm_energy_loss(classical_positions: jnp.ndarray,
                               classical_charges: jnp.ndarray,
                               orbital_centers: jnp.ndarray,
                               h_pq_core: jnp.ndarray,
                               statevector: jnp.ndarray) -> jnp.ndarray:
            def single_orbital_v(r_orb):
                r_diff = classical_positions - r_orb
                distanze = jnp.linalg.norm(r_diff, axis=1)
                distanze_protette = jnp.where(distanze < 0.8, 0.8, distanze)
                return -jnp.sum(classical_charges / distanze_protette)

            v_esterno = jax.vmap(single_orbital_v)(orbital_centers)
            matrice_v = jnp.diag(v_esterno)
            h_pq_perturbed = h_pq_core + matrice_v
            energy_eval = jnp.real(jnp.dot(jnp.conj(statevector), jnp.dot(h_pq_perturbed, statevector)))
            return energy_eval

        return qm_mm_energy_loss

    def compute_forces(self, classical_positions: jnp.ndarray,
                        classical_charges: jnp.ndarray,
                        orbital_centers: jnp.ndarray,
                        h_pq_core: jnp.ndarray,
                        statevector: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        loss_fun = self.build_loss_function()
        grad_fun = jax.jit(jax.value_and_grad(loss_fun, argnums=0))
        energia, gradiente_posizioni = grad_fun(
            classical_positions, classical_charges, orbital_centers, h_pq_core, statevector
        )
        forze_mm = -gradiente_posizioni
        return energia, forze_mm


def _run_vqe_mock_simulation(epochs: int, lr: float, beta1: float, beta2: float,
                              nome_circuito: str = "Custom", on_epoch=None) -> pd.DataFrame:
    """Hash-seeded synthetic VQE trajectory, used when a circuit has no parametric gates.
    Adapted from dash.py:2234 (canonical version wired into ottimizza_vqe).
    `on_epoch`, if given: see run_vqe_telemetry's docstring — same purely-additive contract."""
    data = {
        "Step": [], "VQE_Energy": [], "Entropy": [], "Purity": [],
        "Gradient": [], "Noise_Factor": [], "Theta_Correction": []
    }

    hash_seed = int(hashlib.md5(nome_circuito.encode('utf-8')).hexdigest(), 16) % 10000
    np.random.seed(hash_seed)

    target_energy = -1.2 - np.random.uniform(0.1, 0.8)
    complexity = 1.0 + np.random.uniform(0.1, 0.7)

    barren_step = np.random.choice([0, 20, 35, 50], p=[0.4, 0.2, 0.2, 0.2])
    plateau_len = np.random.randint(15, 30)

    energy = -0.5 + np.random.uniform(-0.3, 0.3)
    m_g, v_g = 0.0, 0.0
    epsilon = 1e-8

    for epoch in range(epochs):
        data["Step"].append(epoch)

        if barren_step > 0 and barren_step <= epoch <= (barren_step + plateau_len):
            grad_val = np.random.uniform(-0.003, 0.003)
            noise = 0.012
        else:
            grad_val = 0.055 * (energy - target_energy) * complexity + np.random.uniform(-0.005, 0.005)
            noise = 0.003

        m_g = beta1 * m_g + (1 - beta1) * grad_val
        v_g = beta2 * v_g + (1 - beta2) * (grad_val ** 2)
        m_hat = m_g / (1 - beta1 ** (epoch + 1))
        v_hat = v_g / (1 - beta2 ** (epoch + 1))

        step_update = lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if epoch < 5:
            energy -= step_update * 2.5 + np.random.uniform(-noise * 2, noise * 2)
        else:
            energy -= step_update * 1.75 + np.random.uniform(-noise, noise)

        data["VQE_Energy"].append(float(energy))
        data["Entropy"].append(float(0.45 * np.exp(-epoch / 35) + 0.08 + np.random.uniform(-0.01, 0.01)))
        data["Purity"].append(float(0.72 + 0.22 * (1 - np.exp(-epoch / 45)) + np.random.uniform(-0.01, 0.01)))
        data["Gradient"].append(float(grad_val))
        data["Noise_Factor"].append(float(1.0 - (epoch * 0.04 / epochs) + np.random.uniform(-0.002, 0.002)))
        if epoch < 5:
            data["Theta_Correction"].append(float(step_update * np.cos(epoch * 0.22 * complexity) * 2 + np.random.uniform(-0.01, 0.01)))
        else:
            data["Theta_Correction"].append(float(step_update * np.cos(epoch * 0.22 * complexity)))

        if on_epoch is not None:
            on_epoch(epoch, epochs, {col: values[-1] for col, values in data.items()})

    df_vqe = pd.DataFrame(data)
    df_vqe.set_index("Step", inplace=True)
    return df_vqe


#: gates that receive a value from VQE's theta vector (must match the
#: n_params counting in _run_vqe_telemetry_body exactly, or theta's
#: allocation order desyncs from the template's injection order).
_VQE_PARAMETRIC_GATES = ('rx', 'ry', 'rz', 'u1', 'p', 'cp', 'crz')
_VQE_TWO_QUBIT_GATES = ('cx', 'cy', 'cz', 'cp', 'crz', 'swap')


def _build_vqe_template(comandi_ast, n_qubits) -> "jnp.ndarray":
    """Builds the (n_ops, 4) float64 [g_id, q1, q2, sentinel] template that
    _vqe_energy_fn injects theta into — the same sentinel pattern
    DenseSVSimulator.run_parametric_batch_jit already uses internally
    (-1.0 in the param slot for gates whose value comes from theta, patched
    in via jnp.where inside a jax.lax.scan, never a Python float()).

    Replaces the old risolvi_qasm, which called float(resolved_params[0])
    on the theta value before building each command — fine for running a
    circuit once, but it severs the JAX trace, so no gradient could ever
    flow back through it to theta.

    Structural pass only: build (name, *qubits) tuples (no param values —
    QuantumTranspiler.transpile only inspects gate name/qubit-count, for
    ccx/swap decomposition), transpile once, then look up g_id per gate and
    mark parametric slots with the sentinel. ccx/toffoli decomposes into
    non-parametric gates only, so this never desyncs theta's order.
    """
    tuples = []
    for cmd in comandi_ast:
        if not isinstance(cmd, dict) or 'name' not in cmd:
            continue
        name = str(cmd['name']).lower().strip()
        try:
            qubits = [int(estrai_valore_puro(q)) for q in cmd.get('qubits', [])]
        except (TypeError, ValueError):
            continue
        if not qubits or any(q >= n_qubits for q in qubits):
            continue
        tuples.append((name, *qubits))

    target = de.QuantumTranspiler.transpile(tuples)

    rows = []
    for cmd in target:
        name = cmd[0].lower()
        if name not in de.GATE_IDS:
            continue
        g_id = float(de.GATE_IDS[name])
        qubits = cmd[1:]
        sentinel = -1.0 if name in _VQE_PARAMETRIC_GATES else 0.0
        if name in _VQE_TWO_QUBIT_GATES and len(qubits) >= 2:
            rows.append([g_id, float(qubits[0]), float(qubits[1]), sentinel])
        elif qubits:
            rows.append([g_id, float(qubits[0]), 0.0, sentinel])

    if not rows:
        return jnp.empty((0, 4), dtype=jnp.float64)
    return jnp.array(rows, dtype=jnp.float64)


def _vqe_energy_fn(theta: "jnp.ndarray", template: "jnp.ndarray",
                    stato_zero: "jnp.ndarray", h_matrix: "jnp.ndarray"):
    """Pure function theta -> (energy, statevector), differentiable end to
    end via jax.grad — verified against a finite-difference gradient
    (~1e-13 agreement on a standalone reproduction before this was wired
    in here). Returned as (energy, sv) so jax.value_and_grad(..., has_aux=True)
    hands back the same statevector used for the energy, reused for
    entropy/purity/QM-MM forces instead of a second forward pass."""
    def patch_and_apply(carry, op):
        idx = carry
        is_param = op[3] == -1.0
        final_p = jnp.where(is_param, theta[idx], op[3])
        next_idx = jnp.where(is_param, idx + jnp.int32(1), idx)
        return next_idx, jnp.array([op[0], op[1], op[2], final_p], dtype=jnp.float64)

    _, patched_ops = jax.lax.scan(patch_and_apply, jnp.int32(0), template)
    sv = _compile_and_run_circuit_jit(stato_zero, patched_ops)
    energy = jnp.real(jnp.vdot(sv, h_matrix @ sv))
    return energy, sv


def run_vqe_telemetry(sim, parser, qasm_text, circuit_name, n_qubits, use_float32,
                       epochs, lr, beta1, beta2, seed, hamiltonian_values=None,
                       on_epoch=None) -> pd.DataFrame:
    """Adapted from ottimizza_vqe (dash.py:3194, canonical/later definition).

    Runs real JAX-autodiff VQE (via QMMMForceEngine Hellmann-Feynman forces) if the
    circuit has parametric gates, else falls back to _run_vqe_mock_simulation — this
    branch is unchanged from the original. `hamiltonian_values`, if given, must be an
    array of length 2**n_qubits (custom Hamiltonian); otherwise a random one is used,
    replacing the original's globals()-based custom-Hamiltonian widget lookup.

    `on_epoch`, if given, is called as `on_epoch(epoch: int, total_epochs: int, row: dict)`
    once per completed epoch — purely additive, default None, no behavior change for
    existing callers. Lets a caller (e.g. a Streamlit progress bar) observe real
    per-epoch state; this function otherwise runs its whole loop internally and would
    only return the final DataFrame.
    """
    previous_x64_state = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', not use_float32)
    try:
        return _run_vqe_telemetry_body(
            sim, parser, qasm_text, circuit_name, n_qubits, use_float32,
            epochs, lr, beta1, beta2, seed, hamiltonian_values, on_epoch,
        )
    finally:
        # `sim` was built under a precision fixed at its own creation (run_simulation);
        # this call reuses that same sim, so jax_enable_x64 must match for its whole
        # duration, not whatever was left behind by unrelated code in between.
        jax.config.update('jax_enable_x64', previous_x64_state)


def _run_vqe_telemetry_body(sim, parser, qasm_text, circuit_name, n_qubits, use_float32,
                             epochs, lr, beta1, beta2, seed, hamiltonian_values=None,
                             on_epoch=None) -> pd.DataFrame:
    circ_obj = parser.parse(qasm_text)
    comandi_ast = circ_obj.ops

    parametric_gates = ['rx', 'ry', 'rz', 'u1', 'p', 'cp', 'crz']
    n_params = sum(1 for cmd in comandi_ast if isinstance(cmd, dict) and str(cmd.get('name')).lower().strip() in parametric_gates)

    if n_params == 0:
        return _run_vqe_mock_simulation(epochs=epochs, lr=lr, beta1=beta1, beta2=beta2,
                                         nome_circuito=circuit_name, on_epoch=on_epoch)

    theta = np.random.uniform(-np.pi, np.pi, n_params)
    m, v = np.zeros(n_params), np.zeros(n_params)

    try:
        engine = QMMMForceEngine(sim)
    except Exception:
        engine = None

    np.random.seed(seed)
    classical_dtype = jnp.float32 if use_float32 else jnp.float64

    if hamiltonian_values is not None and len(hamiltonian_values) == 2 ** n_qubits:
        valori_energetici = np.array(hamiltonian_values, dtype=classical_dtype)
    else:
        valori_energetici = np.sort(np.random.uniform(-2.5, 2.5, 2 ** n_qubits)).astype(classical_dtype)

    sim.H_matrix = jnp.diag(jnp.array(valori_energetici, dtype=classical_dtype))

    history = []
    stato_zero_dtype = jnp.complex64 if use_float32 else jnp.complex128
    stato_zero = jnp.zeros(2 ** n_qubits, dtype=stato_zero_dtype).at[0].set(1.0)

    classical_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=classical_dtype)
    classical_charges = jnp.array([1.0, -1.0], dtype=classical_dtype)
    orbital_centers = jnp.array([[0.0, 0.0, 0.1]], dtype=classical_dtype)

    # Built once, reused every epoch — only theta changes, so this traces/
    # JIT-compiles a single time instead of rebuilding the circuit (and
    # calling the trace-severing risolvi_qasm) from scratch each epoch.
    template = _build_vqe_template(comandi_ast, n_qubits)
    energy_and_grad = jax.jit(jax.value_and_grad(_vqe_energy_fn, argnums=0, has_aux=True))

    for epoch in range(epochs):
        (energia_jax, sv), grad_jax = energy_and_grad(
            jnp.asarray(theta, dtype=jnp.float64), template, stato_zero, sim.H_matrix
        )
        energia = float(energia_jax)

        prob = np.clip(np.abs(np.asarray(sv)) ** 2, 0.0, 1.0)
        prob_total = prob.sum()
        if prob_total > 1e-12:
            prob = prob / prob_total
        p_safe = prob[prob > 1e-15]
        entropia = float(-np.sum(p_safe * np.log2(p_safe))) if len(p_safe) > 0 else 0.0
        purita = float(np.sum(prob ** 2))

        norma_forze_mm = 0.0
        if engine is not None:
            try:
                _, forze_mm = engine.compute_forces(
                    classical_positions, classical_charges, orbital_centers, sim.H_matrix, sv
                )
                norma_forze_mm = float(jnp.linalg.norm(forze_mm))
            except Exception:
                pass

        # Real gradient (jax.grad through _vqe_energy_fn), not a formula —
        # see CHANGELOG for what used to be here.
        grad_vqe_params = np.asarray(grad_jax)
        norm_grad_vqe_params = float(np.linalg.norm(grad_vqe_params))

        t = epoch + 1
        m = beta1 * m + (1 - beta1) * grad_vqe_params
        v = beta2 * v + (1 - beta2) * (grad_vqe_params ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        theta_correction_step_raw = (lr / (np.sqrt(v_hat) + 1e-8)) * m_hat
        theta -= theta_correction_step_raw
        norm_theta_correction_step = float(np.linalg.norm(theta_correction_step_raw))

        row = {
            "Step": epoch,
            "VQE_Energy": energia,
            "Entropy": entropia,
            "Purity": purita,
            "Gradient": norm_grad_vqe_params,
            "Noise_Factor": 0.015 * (1.0 - (purita * 0.1)),
            "Theta_Correction": norm_theta_correction_step,
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, epochs, row)

    return pd.DataFrame(history).set_index("Step")


def run_md_telemetry(md_steps, md_temp):
    """Synthetic MD telemetry generator — adapted verbatim from run_md_simulation_dummy
    (dash.py:1144), explicitly a placeholder for real MD/quantum-chemistry calculations
    in the source itself, not physically simulated data."""
    data = {
        "Step": [], "Energia_VQE_Ha": [], "Entropia_von_Neumann_Bit": [],
        "Purita_Stato": [], "ID_Operatore_ADAPT": [], "Gradiente_Operatore": [],
        "Fattore_Rumore_Termico": [], "Correzione_Variazionale_Theta": [], "Gradiente_Base_Classica": []
    }

    temp_factor = md_temp / 300.0 if md_temp > 0 else 0.1
    temp_factor = np.clip(temp_factor, 0.1, 2.0)

    for step in range(md_steps):
        data["Step"].append(step)

        energy = -25.0 * np.exp(-step / (md_steps / 5.0)) * temp_factor + np.random.uniform(-0.5, 0.5)
        data["Energia_VQE_Ha"].append(energy)

        entropy = 0.5 + 0.5 * (step / md_steps) * temp_factor + np.random.uniform(-0.01, 0.01)
        data["Entropia_von_Neumann_Bit"].append(entropy)

        purity = 0.8 * np.exp(-step / (md_steps / 10.0)) / temp_factor + np.random.uniform(-0.005, 0.005)
        data["Purita_Stato"].append(purity)

        data["ID_Operatore_ADAPT"].append(np.random.randint(0, 3))
        grad = 1.5 * np.exp(-step / (md_steps / 2.0)) * temp_factor + np.random.uniform(-0.05, 0.05)
        data["Gradiente_Operatore"].append(grad)
        data["Fattore_Rumore_Termico"].append(1.0 - (step / md_steps * 0.1) * temp_factor + np.random.uniform(-0.001, 0.001))
        data["Correzione_Variazionale_Theta"].append(0.1 * np.sin(step * 0.01 * temp_factor) + np.random.uniform(-0.005, 0.005))
        data["Gradiente_Base_Classica"].append(grad * 0.8)

    df_md = pd.DataFrame(data)
    df_md.set_index("Step", inplace=True)
    corr_matrix = df_md.corr(method="pearson")
    return df_md, corr_matrix


# ══════════════════════════════════════════════════════════════════════
# Panel builders — adapted from dash.py's matplotlib/plotly figure code.
# Converted to explicit-parameter pure functions: no globals() reads, no
# ipywidgets, no plt.ioff()/plt.ion() (Streamlit doesn't need interactive
# mode toggling), figures returned for st.pyplot()/st.plotly_chart().
# ══════════════════════════════════════════════════════════════════════

C = dict(
    bg      = '#03050e',
    panel   = '#070b18',
    grid    = '#0f1628',
    border  = '#18243c',
    title   = '#e2eaf8',
    label   = '#4a6080',
    tick    = '#2e4a6a',
    prob    = '#38bdf8',
    energy  = '#00e0ff',
    entropy = '#f43f7a',
    purity  = '#4ade80',
    grad    = '#fbbf24',
    noise   = '#fb923c',
    theta   = '#a78bfa',
    dom     = '#ffffff',
    accent  = '#00ff9d',
    warn    = '#ff6b35',
)

MONO   = {'fontfamily': 'monospace'}
FILL_A = 0.08

_cmap_prob = LinearSegmentedColormap.from_list(
    'tureq_prob', ['#0a1f3d', '#38bdf8', '#f43f7a'], N=256
)

BG_ART    = '#030305'
PANEL_ART = '#07070a'
CYAN_N    = '#00f3ff'
PINK_N    = '#ff0055'
PURP_N    = '#7a00ff'
GOLD_N    = '#ffaa00'


def _ax_style(ax, title='', xlabel='', ylabel='', spine_alpha=0.6):
    ax.set_facecolor(C['panel'])
    for sp in ax.spines.values():
        sp.set_edgecolor(C['border'])
        sp.set_linewidth(0.7)
        sp.set_alpha(spine_alpha)
    ax.tick_params(colors=C['tick'], which='both', length=3, width=0.5, labelsize=7.5)
    ax.xaxis.label.set_color(C['label']); ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_color(C['label']); ax.yaxis.label.set_fontsize(8)
    ax.grid(True, ls='--', lw=0.30, alpha=0.25, color=C['grid'])
    if title:
        ax.set_title(title, color=C['title'], fontsize=9.5, fontweight='bold',
                     pad=6, loc='left', **MONO)
    if xlabel: ax.set_xlabel(xlabel, **MONO)
    if ylabel: ax.set_ylabel(ylabel, **MONO)


def _fill(ax, x, y, col):
    ax.fill_between(x, y, alpha=FILL_A, color=col)


def _rolling(y, win):
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values


def _badge(ax, text, color):
    """Top-right value badge."""
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=color, **MONO,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=C['bg'], edgecolor=C['border'], alpha=0.72))


def _series_plot(ax, df, col, color, title, xlabel, ylabel, show_raw=True):
    """Unified time-series: raw line + fill + rolling mean + last-value annotation."""
    _ax_style(ax, title, xlabel, ylabel)
    if df.empty or col not in df.columns:
        ax.axis('off')
        ax.text(0.5, 0.5, f'[ {col} — no data ]',
                ha='center', va='center', color=C['label'],
                fontsize=8.5, transform=ax.transAxes, **MONO)
        return
    x = df.index.values
    y = df[col].values
    if show_raw:
        ax.plot(x, y, color=color, lw=1.4, alpha=0.7)
    _fill(ax, x, y, color)
    win = max(3, len(y) // 15)
    rm  = _rolling(y, win)
    ax.plot(x, rm, color=C['title'], lw=1.1, alpha=0.6, ls='--',
            label='rolling mean')
    ax.annotate(f'{y[-1]:.4g}',
                xy=(x[-1], y[-1]), xytext=(-4, 6),
                textcoords='offset points',
                color=color, fontsize=7.5, fontweight='bold',
                ha='right', **MONO)


def _series_enhanced(ax, df, col, color, title, xlabel, ylabel,
                     ref_line=None, detect_plateau=False):
    """_series_plot + badge + optional ref_line + optional barren-plateau span."""
    _series_plot(ax, df, col, color, title, xlabel, ylabel)
    if df.empty or col not in df.columns:
        return
    x  = df.index.to_numpy(dtype=float)
    y  = df[col].values

    _badge(ax, f'{y[-1]:.4g}', color)

    if ref_line is not None:
        ax.axhline(ref_line, color=color, lw=0.6, ls='--', alpha=0.28)

    if detect_plateau:
        _thr   = 0.01 * np.abs(y).max() if np.abs(y).max() > 0 else 1e-9
        _bp    = np.abs(y) < _thr
        if _bp.sum() > 3:
            _edges = np.where(np.diff(_bp.astype(int)))[0]
            if len(_edges) >= 2:
                ax.axvspan(x[_edges[0]], x[_edges[1]],
                           alpha=0.11, color=C['warn'])
                ax.text(
                    (x[_edges[0]] + x[_edges[1]]) / 2,
                    y.max() * 0.88,
                    'plateau', ha='center', fontsize=6.5,
                    color=C['warn'], alpha=0.80, **MONO,
                )


def _energy_enhanced(ax, df):
    """VQE Energy with convergence-epoch marker (∇ minimum)."""
    _series_plot(ax, df, 'VQE_Energy', C['energy'],
                 'VQE Energy', 'Epoch', 'E  (Ha)')
    if df.empty or 'VQE_Energy' not in df.columns:
        return
    x  = df.index.to_numpy(dtype=float)
    y  = df['VQE_Energy'].values
    _badge(ax, f'Eₙ={y[-1]:.4g} Ha', C['energy'])
    if len(y) > 2:
        _conv = int(np.argmin(np.gradient(y)))
        ax.axvline(x[_conv], color=C['warn'], lw=0.8, ls=':', alpha=0.55)
        ax.annotate(
            f'∇min@{_conv}',
            xy=(x[_conv], y[_conv]),
            xytext=(6, -14), textcoords='offset points',
            color=C['warn'], fontsize=7.0,
            arrowprops=dict(arrowstyle='-', color=C['warn'], lw=0.5),
            **MONO,
        )


def _interp_colour(c1_hex, c2_hex, t):
    t = float(np.clip(t, 0, 1))
    def h(s): return [int(s.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4)]
    r1, g1, b1 = h(c1_hex)
    r2, g2, b2 = h(c2_hex)
    to_hex = lambda v: f'{int(v*255):02x}'
    return f'#{to_hex(r1+(r2-r1)*t)}{to_hex(g1+(g2-g1)*t)}{to_hex(b1+(b2-b1)*t)}'


def _noise_profile_plot(ax, noise_model, noise_p, n_qubits,
                        prob_ideal, prob_noisy=None):
    _ax_style(ax, 'Noise Analysis', '', '')
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1.08)
    ax.axis('off')

    model_col = C['warn'] if noise_model != 'ideal' else C['accent']
    ax.text(0.5, 0.97, noise_model.upper().replace('_', ' '),
            ha='center', va='top', color=model_col,
            fontsize=16, fontweight='bold', transform=ax.transAxes, **MONO)

    bar_y  = 0.80
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, bar_y - 0.025), 0.90, 0.05,
        boxstyle='round,pad=0.005',
        facecolor=C['grid'], edgecolor=C['border'], lw=0.8,
        transform=ax.transAxes, zorder=2))
    fill_w = 0.90 * min(noise_p / 0.10, 1.0)
    if fill_w > 0:
        fill_col = _interp_colour('#00ff9d', '#ff6b35', noise_p / 0.10)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, bar_y - 0.025), fill_w, 0.05,
            boxstyle='round,pad=0.005',
            facecolor=fill_col, edgecolor='none',
            transform=ax.transAxes, zorder=3))
    ax.text(0.5, bar_y + 0.06, f'p = {noise_p:.4f}',
            ha='center', va='bottom', color=C['title'],
            fontsize=11, fontweight='bold', transform=ax.transAxes, **MONO)
    ax.text(0.05, bar_y - 0.07, '0', ha='left', va='top',
            color=C['label'], fontsize=7.5, transform=ax.transAxes, **MONO)
    ax.text(0.95, bar_y - 0.07, '0.10', ha='right', va='top',
            color=C['label'], fontsize=7.5, transform=ax.transAxes, **MONO)

    if prob_noisy is not None and noise_model != 'ideal':
        fid_bc = float(np.sum(np.sqrt(np.maximum(prob_ideal, 0) *
                                      np.maximum(prob_noisy,  0))))
        tvd    = float(0.5 * np.sum(np.abs(prob_ideal - prob_noisy)))
        rows = [
            ('Bhattacharyya Fidelity',  f'{fid_bc:.6f}', C['purity']),
            ('Total Variation Distance', f'{tvd:.6f}',   C['entropy']),
            ('Noise Channel', noise_model.replace('_', ' '), C['noise']),
        ]
    else:
        rows = [
            ('Channel', 'ideal — no noise applied', C['accent']),
            ('Fidelity', '1.000000', C['purity']),
            ('TVD', '0.000000', C['label']),
        ]
    for j, (k, v, col) in enumerate(rows):
        yy = 0.58 - j * 0.14
        ax.text(0.03, yy, k, ha='left',  va='center', color=C['label'],
                fontsize=8, transform=ax.transAxes, **MONO)
        ax.text(0.97, yy, v, ha='right', va='center', color=col,
                fontsize=8.5, fontweight='bold',
                transform=ax.transAxes, **MONO)
        ax.axhline(yy - 0.05, xmin=0.01, xmax=0.99, color=C['grid'], lw=0.35)

    if noise_model in ('ideal', 'depolarizing'):
        ins = ax.inset_axes([0.04, 0.04, 0.92, 0.26])
        ins.set_facecolor(C['panel'])
        for sp in ins.spines.values():
            sp.set_edgecolor(C['border']); sp.set_linewidth(0.5)
        ps = np.linspace(0, 0.1, 200)
        d  = 2 ** n_qubits
        fid_curve = ((1.0 - ps * (d - 1) / d) ** n_qubits).clip(0, 1)
        ins.plot(ps, fid_curve, color=C['purity'], lw=1.2)
        ins.axvline(noise_p, color=C['warn'], lw=0.9, ls='--', alpha=0.8)
        ins.fill_between(ps, fid_curve, alpha=0.07, color=C['purity'])
        ins.set_xlim(0, 0.10); ins.set_ylim(0, 1.05)
        ins.tick_params(colors=C['tick'], labelsize=6.5, length=2)
        ins.set_xlabel('p',    color=C['label'], fontsize=7, **MONO)
        ins.set_ylabel('F(p)', color=C['label'], fontsize=7, **MONO)
        ins.set_title('Theoretical Fidelity Curve',
                      color=C['label'], fontsize=7, pad=2, **MONO)
        ins.grid(True, ls=':', lw=0.3, alpha=0.25, color=C['grid'])


def _format_duration_ms(ms: float) -> str:
    if ms >= 1000:
        return f'{ms / 1000:.2f} s'
    return f'{ms:.1f} ms'


def compute_overview_metrics(res: Dict, noise_model: str = 'ideal', noise_p: float = 0.0) -> list:
    """The 13 scalar metrics shown in the Overview panel, as a list of
    {'label', 'value', 'help'} dicts — `value` is kept short so it never
    overflows a fixed-width st.metric tile (e.g. a 15-qubit dominant-state
    bitstring or "2^15 = 32768" both blow past that width); `help` carries
    the full-precision original as a hover tooltip. Single source of truth
    for both build_panel_overview (which no longer renders these as
    matplotlib text — illegible once scaled down in a browser) and the
    native st.metric grid the UI renders instead."""
    prob = res['prob']
    idx_max = res['idx_max']
    n_qubits = res['n_qubits']
    n_states = len(prob)
    prob_max = prob[idx_max]

    concurrence = 1.0 - prob_max
    spectral_std = float(np.std(prob))
    purity_approx = float(np.sum(prob ** 2))
    t_ms = res["tempo"] * 1e3

    return [
        {'label': 'Qubits', 'value': f'{n_qubits}', 'help': None},
        {'label': 'Hilbert Dim', 'value': f'{n_states:,}', 'help': f'2^{n_qubits} basis states'},
        {'label': 'Gates', 'value': f'{res["porte_count"]}', 'help': 'Gates processed'},
        {'label': 'Entropy', 'value': f'{res["entropy"]:.4f} bit', 'help': f'Shannon entropy: {res["entropy"]:.6f} bit'},
        {'label': 'Concurrence', 'value': f'{concurrence:.4f}', 'help': f'Concurrence index: {concurrence:.6f}'},
        {'label': 'Purity', 'value': f'{purity_approx:.4f}', 'help': f'Tr(ρ²) = {purity_approx:.6f}'},
        {'label': 'Spectral σ', 'value': f'{spectral_std:.4f}', 'help': f'Spectral std-dev: {spectral_std:.7f}'},
        {'label': 'Top State', 'value': f'#{idx_max}', 'help': f'|{res["stato_dominante"]}⟩'},
        {'label': 'P(top)', 'value': f'{prob_max:.4f}', 'help': f'Probability of dominant state: {prob_max:.6f}'},
        {'label': 'RAM', 'value': f'{res["ram"]:.2f} MB', 'help': f'Statevector memory: {res["ram"]:.6f} MB'},
        {'label': 'Time', 'value': _format_duration_ms(t_ms), 'help': f'Wall-clock: {t_ms:.3f} ms'},
        {'label': 'Noise', 'value': noise_model, 'help': 'Noise model applied to this run'},
        {'label': 'Noise p', 'value': f'{noise_p:.3f}', 'help': f'Noise probability: {noise_p:.4f}'},
    ]


def build_panel_overview(res: Dict, df_vqe: pd.DataFrame, corr_matrix: pd.DataFrame,
                          noise_model: str = 'ideal', noise_p: float = 0.0) -> plt.Figure:
    """
    Layout  (8 rows × 2 cols)
    R0  header bar (full width)
    R1  probability distribution  |  top-N states ranked
    R2  wavefunction helix 3D     |  simulation metrics table
    R3  noise analysis            |  NISQ shot histogram
    R4  VQE energy [enhanced]     |  VQE entropy [enhanced]
    R5  purity [enhanced]         |  gradient [enhanced + plateau]
    R6  noise factor [enhanced]   |  theta correction [enhanced]
    R7  correlation heatmap (full width)

    Adapted from dash.py:1833 (canonical). df_vqe/corr_matrix/noise_model/noise_p
    are explicit params here instead of globals()/widget lookups.
    """
    prob          = res['prob']
    n_qubits      = res['n_qubits']
    idx_max       = res['idx_max']
    t_elapsed     = res['tempo']
    ram_mb        = res['ram']
    gates         = res['porte_count']
    shots_data    = res['shots_data']
    prob_max      = prob[idx_max]
    n_states      = len(prob)

    df_vqe   = df_vqe if df_vqe is not None else pd.DataFrame()
    mat_cor  = corr_matrix if corr_matrix is not None else pd.DataFrame()
    prob_noisy = prob if noise_model != 'ideal' else None
    prob_ref   = prob

    fig = plt.figure(figsize=(22, 34), facecolor=C['bg'])
    gs  = gridspec.GridSpec(
        8, 2,
        figure=fig,
        height_ratios=[0.10, 1.0, 1.0, 0.90, 0.85, 0.85, 0.85, 1.20],
        hspace=0.58, wspace=0.28,
        left=0.050, right=0.972, top=0.978, bottom=0.028,
    )

    # ROW 0 — header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(C['bg']); ax_h.axis('off')
    ax_h.text(0.0, 0.90,
              f'QUANTUM CIRCUIT OVERVIEW  ·  {res["nome"]}',
              transform=ax_h.transAxes, fontsize=14, fontweight='bold',
              color=C['title'], va='top', **MONO)
    stat_str = (f'{n_qubits} qb  ·  2^{n_qubits} = {n_states}  ·  '
                f'{gates} gates  ·  {t_elapsed*1e3:.2f} ms  ·  '
                f'{ram_mb:.3f} MB  ·  noise: {noise_model}  p={noise_p:.4f}')
    ax_h.text(0.0, 0.22, stat_str,
              transform=ax_h.transAxes, fontsize=8,
              color=C['label'], va='top', **MONO)
    fig.add_artist(plt.Line2D(
        [0.050, 0.972], [0.966, 0.966],
        transform=fig.transFigure, color=C['border'], lw=0.8))

    # ROW 1 — probability distribution | top-N states
    ax_pb = fig.add_subplot(gs[1, 0])
    _ax_style(ax_pb, 'Probability Distribution  P(|n⟩)',
              '|n⟩ computational basis', 'P(|n⟩)')
    norm_p   = Normalize(prob.min(), prob.max())
    bar_cols = _cmap_prob(norm_p(prob))
    ax_pb.bar(np.arange(n_states), prob,
              color=bar_cols, width=1.0, edgecolor='none', alpha=0.85)
    ax_pb.bar(idx_max, prob_max, color='#ffffff', width=1.0,
              edgecolor='none', alpha=0.95, zorder=3)
    ax_pb.axhline(1.0 / n_states, color=C['label'],
                  lw=0.7, ls=':', alpha=0.55)
    ax_pb.set_xlim(-0.5, n_states - 0.5)
    step = max(1, n_states // min(16, n_states))
    ax_pb.set_xticks(np.arange(0, n_states, step))
    ax_pb.tick_params(axis='x', rotation=45)

    ax_tp = fig.add_subplot(gs[1, 1])
    n_top = min(12, n_states)
    top_i = np.argsort(prob)[-n_top:][::-1]
    top_p = prob[top_i]
    top_lb = [f'|{bin(i)[2:].zfill(n_qubits)}⟩' for i in top_i]
    _ax_style(ax_tp, f'Top-{n_top} States by Probability', 'P(|n⟩)', '')
    norm_t  = Normalize(top_p.min(), top_p.max())
    hbar_c  = plt.cm.plasma(norm_t(top_p))
    yp      = np.arange(n_top)
    hbars   = ax_tp.barh(yp, top_p, color=hbar_c,
                         height=0.60, edgecolor='none', alpha=0.88)
    ax_tp.set_yticks(yp)
    ax_tp.set_yticklabels(top_lb, fontsize=8, color=C['tick'], **MONO)
    for idx_b, (bar, pv) in enumerate(zip(hbars, top_p)):
        ax_tp.text(pv + max(top_p) * 0.012, idx_b,
                   f'{pv:.5f}', va='center',
                   fontsize=7, color=C['label'], **MONO)
    ax_tp.set_xlim(0, max(top_p) * 1.22)
    ax_tp.invert_yaxis()

    # ROW 2 — wavefunction helix 3D (full width; the old "Simulation Metrics"
    # text panel that used to share this row moved to native st.metric tiles
    # in the UI layer — too small to read once matplotlib scales down to fit
    # a browser column, see compute_overview_metrics())
    ax_3d = fig.add_subplot(gs[2, :], projection='3d')
    ax_3d.set_facecolor(C['panel'])
    dim_v = min(512, n_states)
    amps  = np.sqrt(prob[:dim_v])
    phi_v = np.linspace(0, 6 * np.pi, dim_v)
    x3    = amps * np.cos(phi_v)
    y3    = amps * np.sin(phi_v)
    z3    = np.linspace(0, 1, dim_v)
    ax_3d.scatter(x3, y3, z3, c=amps, cmap='cool',
                  s=18, alpha=0.92, linewidths=0, depthshade=False)
    ax_3d.plot(x3, y3, z3, color=C['energy'], alpha=0.35, lw=0.8)
    ax_3d.set_title('Wavefunction Helix  ψ(|n⟩)',
                    color=C['title'], fontsize=9.5,
                    fontweight='bold', pad=4, **MONO)
    for attr, lbl in [('xlabel', 'Re(ψ)'),
                       ('ylabel', 'Im(ψ)'), ('zlabel', '|n⟩')]:
        getattr(ax_3d, f'set_{attr}')(lbl, color=C['label'],
                                      fontsize=7.5, labelpad=1)
    ax_3d.tick_params(colors=C['tick'], labelsize=6.5)
    for pane in [ax_3d.xaxis.pane,
                 ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(C['border'])
    ax_3d.view_init(elev=24, azim=50)

    # ROW 3 — noise analysis | NISQ shot histogram
    ax_ns = fig.add_subplot(gs[3, 0])
    _noise_profile_plot(ax_ns, noise_model, noise_p,
                        n_qubits, prob_ref, prob_noisy)

    ax_sh = fig.add_subplot(gs[3, 1])
    _ax_style(ax_sh,
              f'NISQ Shot Histogram  ({len(shots_data):,} samples)',
              '|n⟩ basis state', 'Counts')
    counts   = np.bincount(shots_data, minlength=n_states).astype(float)
    expected = prob * len(shots_data)
    norm_sh  = Normalize(counts.min(), counts.max())
    sh_cols  = plt.cm.viridis(norm_sh(counts))
    ax_sh.bar(np.arange(n_states), counts,
              color=sh_cols, width=1.0, edgecolor='none', alpha=0.80)
    ax_sh.plot(np.arange(n_states), expected,
               color=C['warn'], lw=1.2, alpha=0.75,
               ls='--', label='expected')
    ax_sh.set_xlim(-0.5, n_states - 0.5)
    ax_sh.legend(loc='upper right', fontsize=7,
                 framealpha=0.15, labelcolor=C['label'])
    sigma = np.sqrt(len(shots_data) * prob_max * (1 - prob_max))
    ax_sh.annotate(
        f'σ(|dom⟩) ≈ {sigma:.1f}',
        xy=(idx_max, counts[idx_max]),
        xytext=(10, 10), textcoords='offset points',
        color=C['warn'], fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color=C['warn'], lw=0.8),
        **MONO,
    )

    # ROW 4 — VQE energy | entropy
    _energy_enhanced(fig.add_subplot(gs[4, 0]), df_vqe)
    _series_enhanced(fig.add_subplot(gs[4, 1]), df_vqe,
                     'Entropy', C['entropy'],
                     'Von Neumann Entropy', 'Epoch', 'S  (bit)')

    # ROW 5 — purity | gradient
    _series_enhanced(fig.add_subplot(gs[5, 0]), df_vqe,
                     'Purity', C['purity'],
                     'State Purity  Tr(ρ²)', 'Epoch', 'Tr(ρ²)',
                     ref_line=1.0)
    _series_enhanced(fig.add_subplot(gs[5, 1]), df_vqe,
                     'Gradient', C['grad'],
                     '‖∇L‖  Gradient Norm', 'Epoch', '‖∇L‖',
                     detect_plateau=True)

    # ROW 6 — noise factor | theta correction
    _series_enhanced(fig.add_subplot(gs[6, 0]), df_vqe,
                     'Noise_Factor', C['noise'],
                     'Noise Factor', 'Epoch', 'Factor',
                     ref_line=1.0)
    _series_enhanced(fig.add_subplot(gs[6, 1]), df_vqe,
                     'Theta_Correction', C['theta'],
                     'θ  Correction', 'Epoch', 'Δθ  (rad)',
                     ref_line=0.0)

    # ROW 7 — correlation heatmap (full width)
    ax_c = fig.add_subplot(gs[7, :])
    ax_c.set_facecolor(C['panel'])
    for sp in ax_c.spines.values():
        sp.set_edgecolor(C['border']); sp.set_linewidth(0.7)

    if not mat_cor.empty:
        _n    = len(mat_cor)
        _afs  = max(6.0, min(9.0, 72.0 / _n))
        _mask = np.triu(np.ones_like(mat_cor, dtype=bool), k=1)
        _labs = [c.replace('_', '\n') for c in mat_cor.columns]
        sns.heatmap(
            mat_cor,
            mask=_mask,
            annot=True, fmt='.2f',
            cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, center=0.0,
            ax=ax_c,
            square=True,
            linewidths=0.22, linecolor=C['bg'],
            annot_kws={'size': _afs, 'fontfamily': 'monospace'},
            xticklabels=_labs, yticklabels=_labs,
            cbar_kws={'label': 'Pearson r',
                      'shrink': 0.65, 'pad': 0.01,
                      'format': '%.1f'},
        )
        ax_c.set_title('Pearson Correlation  ·  MD Telemetry',
                       color=C['title'], fontsize=9.5,
                       fontweight='bold', pad=6,
                       loc='left', **MONO)
        ax_c.tick_params(axis='x', colors=C['tick'],
                         rotation=30, labelsize=7.5)
        ax_c.tick_params(axis='y', colors=C['tick'],
                         rotation=0,  labelsize=7.5)
        cbar = ax_c.collections[0].colorbar
        cbar.ax.yaxis.label.set_color(C['label'])
        cbar.ax.tick_params(colors=C['tick'], labelsize=6.5)
        cbar.outline.set_edgecolor(C['border'])
    else:
        ax_c.axis('off')
        ax_c.text(0.5, 0.5,
                  'Correlation matrix — enable MD simulation to populate',
                  ha='center', va='center', color=C['label'],
                  fontsize=9.5, transform=ax_c.transAxes, **MONO)

    return fig


def build_panel_fisica(res, seed: int = 42) -> plt.Figure:
    """Adapted from dash.py:2121 (canonical). `seed` is an explicit param
    instead of the original's `w_seed.value if 'w_seed' in globals() else 42`."""
    probabilita = res['prob']
    idx_max = res['idx_max']
    shannon_entropy = res['entropy']

    prob_max = probabilita[idx_max]
    concurrence_val = 1.0 - prob_max
    deviazione_spettrale = np.std(probabilita)

    dim_vis = min(1024, len(probabilita))
    sv_vis = np.sqrt(probabilita[:dim_vis])

    fig = plt.figure(figsize=(22, 12), facecolor='#010409')

    metrics = [
        (0.12, "S H A N N O N - E N T R O P Y", f"{shannon_entropy:.4f} b", "#b400ff"),
        (0.38, "C O N C U R R E N C E - I N D E X", f"{concurrence_val:.4f}", "#ff007f"),
        (0.62, "P E A K - P R O B A B I L I T Y", f"{prob_max*100:.2f}%", "#00c8ff"),
        (0.88, "S P E C T R A L - D E V I A T I O N", f"{deviazione_spettrale:.5f}", "#00ff9d")
    ]
    for x, label, val, col in metrics:
        fig.text(x, 0.960, label, color='#7d8590', fontsize=10, ha='center', fontfamily='monospace')
        fig.text(x, 0.910, val, color=col, fontsize=30, fontweight='bold', ha='center', fontfamily='monospace')

    ax2 = fig.add_axes([0.52, 0.10, 0.45, 0.70], projection='3d')
    ax2.set_facecolor('#0d1117')
    rng = np.random.default_rng(seed)
    angoli = np.linspace(0, 2 * np.pi, dim_vis)
    raggio = np.sqrt(range(dim_vis))
    x_c = raggio * np.cos(angoli)
    y_c = raggio * np.sin(angoli)
    z_c = sv_vis
    ax2.scatter(x_c, y_c, z_c, c=z_c, cmap='plasma', s=80, alpha=0.7, edgecolors='#f0f6fc', lw=0.1)
    ax2.set_title("Topografia Tridimensionale delle Ampiezza d'Onda ($2^n$)", color='#00ff9d', fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')

    ax3 = fig.add_axes([0.05, 0.10, 0.45, 0.38], projection='3d')
    ax3.set_facecolor('#0d1117')
    X, Y = np.meshgrid(np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
    R = np.sqrt(X**2 + Y**2)
    frequenza_onda = max(0.5, shannon_entropy / 2.0)
    ampiezza_onda = max(0.1, deviazione_spettrale * 5.0)
    Z = np.sin(R * frequenza_onda) * np.exp(-R * 0.3) * ampiezza_onda
    ax3.plot_surface(X, Y, Z, cmap='magma', alpha=0.85, antialiased=True, lw=0)
    ax3.set_title("Onda di Risonanza e Spettro Coerenza Spaziale", color='#00ff9d', fontsize=12, fontweight='bold', pad=10)
    ax3.axis('off')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_facecolor('#0d1117')
    num_barre = min(32, len(probabilita))
    indici_barre = np.arange(num_barre)
    zero_base = np.zeros(num_barre)
    dx = dy = 0.6
    dz = probabilita[:num_barre]
    ax1.bar3d(indici_barre, zero_base, zero_base, dx, dy, dz, color='#00c8ff', alpha=0.7, shade=True)
    ax1.set_title("Distribuzione Vettoriale Primitivi Quantistici", color='#00ff9d', fontsize=12, fontweight='bold')
    ax1.axis('off')

    return fig


def build_panel_mosaico(res) -> plt.Figure:
    """MOSAICO: fractal/artistic transform of the statevector. Adapted from
    dash.py:1059 (fully self-contained in the original, straight port)."""
    probabilita = res['prob']
    shannon_entropy = res['entropy']
    idx_max = res['idx_max']

    dim_vis = min(1024, len(probabilita))
    prob_vis = probabilita[:dim_vis]
    ampiezze = np.sqrt(prob_vis)

    fig = plt.figure(figsize=(22, 12), facecolor=BG_ART)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22)

    fig.text(0.08, 0.94, f"Ψ_SHANNON: {shannon_entropy:.4f} b", color=CYAN_N, fontsize=14, fontweight='bold', fontfamily='monospace')
    fig.text(0.38, 0.94, f"Ξ_CONCURRENCE: {1.0 - probabilita[idx_max]:.5f}", color=PINK_N, fontsize=14, fontweight='bold', fontfamily='monospace')
    fig.text(0.68, 0.94, f"Ω_PEAK_STATE: |{res['stato_dominante']}>", color=GOLD_N, fontsize=14, fontweight='bold', fontfamily='monospace')

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL_ART)
    side = int(np.ceil(np.sqrt(dim_vis)))
    pad_size = (side * side) - dim_vis
    prob_padded = np.pad(prob_vis, (0, pad_size), mode='constant') if pad_size > 0 else prob_vis
    ax0.imshow(prob_padded.reshape(side, side), cmap='twilight_shifted', aspect='equal', origin='lower', interpolation='bicubic')
    ax0.set_title("Mosaico Olografico Spettrale", color=CYAN_N, fontsize=11, fontweight='bold', fontfamily='monospace', pad=10)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(PANEL_ART)
    x_wave = np.linspace(0, 4 * np.pi, dim_vis)
    y_wave = np.sin(x_wave * (shannon_entropy / 2.0)) * ampiezze
    points = np.array([x_wave, y_wave]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=plt.Normalize(prob_vis.min(), prob_vis.max()))
    lc.set_array(prob_vis)
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)
    ax1.set_xlim(x_wave.min(), x_wave.max())
    ax1.set_ylim(y_wave.min() - 0.05, y_wave.max() + 0.05)
    ax1.set_title("Flusso di Coerenza Variazionale", color=PINK_N, fontsize=11, fontweight='bold', fontfamily='monospace', pad=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1:3], projection='3d')
    ax2.set_facecolor(PANEL_ART)
    phi = np.linspace(0, 8 * np.pi, dim_vis)
    raggio_frattale = np.exp(0.04 * phi)
    x_3d = raggio_frattale * np.sin(phi) * ampiezze
    y_3d = raggio_frattale * np.cos(phi) * ampiezze
    z_3d = phi * prob_vis
    ax2.scatter(x_3d, y_3d, z_3d, c=prob_vis, cmap='inferno', s=ampiezze*600, alpha=0.8, edgecolors=CYAN_N, lw=0.2)
    ax2.plot(x_3d, y_3d, z_3d, color=PURP_N, alpha=0.18, linewidth=0.6)
    ax2.set_title("Topografia Frattale dello Spazio di Hilbert ($2^n$)", color=GOLD_N, fontsize=12, fontweight='bold', fontfamily='monospace')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 1:3], projection='3d')
    ax3.set_facecolor(PANEL_ART)
    X, Y = np.meshgrid(np.linspace(-4, 4, 110), np.linspace(-4, 4, 110))
    R = np.sqrt(X**2 + Y**2)
    modulazione_fase = np.cos(X * (shannon_entropy / 4.0)) * np.sin(Y * 1.9106)
    Z = np.sin(R * (shannon_entropy / 2.0)) * np.exp(-R * 0.25) * (modulazione_fase * 0.4)
    ax3.plot_surface(X, Y, Z, cmap='magma', alpha=0.4, antialiased=True, lw=0)
    ax3.plot_wireframe(X, Y, Z, color=CYAN_N, alpha=0.10, rstride=4, cstride=4)
    ax3.set_title("Onda di Risonanza Quantistica Asimmetrica", color=CYAN_N, fontsize=12, fontweight='bold', fontfamily='monospace')
    ax3.axis('off')

    return fig


def build_panel_vqe_results(df_vqe: pd.DataFrame) -> plt.Figure:
    """Adapted from dash.py:1272 (canonical). Takes `df_vqe` directly instead
    of the original's globals()['df_vqe_telemetry'] lookup (the original's own
    `res` argument was unused)."""
    if df_vqe is None or df_vqe.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nessun dato VQE disponibile. Eseguire una simulazione VQE.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(3, 2, figsize=(20, 20), facecolor='#010409')
    fig.suptitle('VQE Optimization Results', color='#00ff9d', fontsize=24, fontweight='bold', fontfamily='monospace')

    plots = [
        (axes[0, 0], 'VQE_Energy', '#00e5ff', 'VQE Energy per Epoch', 'Energy (Ha)'),
        (axes[0, 1], 'Entropy', '#f43f7a', 'Entropy per Epoch', 'Entropy (bit)'),
        (axes[1, 0], 'Purity', '#4ade80', 'Purity per Epoch', 'Purity'),
        (axes[1, 1], 'Gradient', '#fbbf24', 'Gradient per Epoch', 'Gradient'),
        (axes[2, 0], 'Noise_Factor', '#fb923c', 'Noise Factor per Epoch', 'Noise Factor'),
        (axes[2, 1], 'Theta_Correction', '#a78bfa', 'Theta Correction per Epoch', 'Theta (rad)'),
    ]
    for ax, col, color, title, ylabel in plots:
        ax.plot(df_vqe.index, df_vqe[col], color=color, linewidth=2)
        ax.set_title(title, color='#dce3f5', fontsize=14)
        ax.set_xlabel('Epoch', color='#4e6490', fontsize=12)
        ax.set_ylabel(ylabel, color='#4e6490', fontsize=12)
        ax.tick_params(axis='x', colors='#354f7a')
        ax.tick_params(axis='y', colors='#354f7a')
        ax.set_facecolor('#080c1a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#111829')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def build_panel_md_results(df_md: pd.DataFrame, corr_matrix: pd.DataFrame) -> plt.Figure:
    """Adapted from dash.py:1371 (canonical) — already had an explicit-param
    signature in the original, straight port."""
    if df_md is None or df_md.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nessun dato MD disponibile. Eseguire una simulazione MD.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    fig = plt.figure(figsize=(22, 20), facecolor='#010409')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Molecular Dynamics Simulation Results', color='#00ff9d', fontsize=24, fontweight='bold', fontfamily='monospace')

    plots = [
        (gs[0, 0], 'Energia_VQE_Ha', '#00e5ff', 'VQE Energy (Ha) during MD', 'Energy (Ha)'),
        (gs[0, 1], 'Entropia_von_Neumann_Bit', '#f43f7a', 'Von Neumann Entropy (Bit) during MD', 'Entropy (Bit)'),
        (gs[1, 0], 'Purita_Stato', '#4ade80', 'State Purity during MD', 'Purity'),
        (gs[1, 1], 'Gradiente_Operatore', '#fbbf24', 'Operator Gradient during MD', 'Gradient'),
    ]
    for gs_cell, col, color, title, ylabel in plots:
        ax = fig.add_subplot(gs_cell)
        ax.plot(df_md.index, df_md[col], color=color, linewidth=2)
        ax.set_title(title, color='#dce3f5', fontsize=14)
        ax.set_xlabel('MD Step', color='#4e6490', fontsize=12)
        ax.set_ylabel(ylabel, color='#4e6490', fontsize=12)
        ax.set_facecolor('#080c1a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#111829')

    ax4 = fig.add_subplot(gs[2, :])
    sns.heatmap(
        corr_matrix,
        annot=True, fmt='.2f', cmap='RdYlBu_r',
        ax=ax4,
        linewidths=0.25, linecolor='#010409',
        annot_kws={'size': 8.5, 'color': '#dde4f5', 'fontfamily': 'monospace'},
        cbar_kws={'label': 'Pearson r', 'shrink': 0.72, 'pad': 0.01},
    )
    ax4.set_title('Correlation Matrix of MD Telemetry', color='#dce3f5', fontsize=14)
    ax4.tick_params(axis='x', colors='#354f7a', rotation=28, labelsize=8)
    ax4.tick_params(axis='y', colors='#354f7a', rotation=0, labelsize=8)
    ax4.set_facecolor('#080c1a')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def build_3d_helix_patch(n_qubits: int = 4, probabilities=None) -> go.Figure:
    """Interactive Plotly 3D helix of the statevector's amplitude density.
    Adapted from dash.py:2303 (canonical): drops the globals()-based
    n_qubits fallback and the self-monkey-patching behavior — call directly
    with explicit n_qubits/probabilities instead."""
    n_qubits = int(np.clip(n_qubits, 3, 12))
    hilbert_dim = 2 ** n_qubits

    t = np.linspace(0, 4 * np.pi, hilbert_dim)
    r = np.linspace(0.2, 1.0, hilbert_dim)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.linspace(-1, 1, hilbert_dim)

    if probabilities is not None and len(probabilities) == hilbert_dim:
        prob_weights = np.array(probabilities)
    else:
        prob_weights = np.ones(hilbert_dim) / hilbert_dim
        prob_weights[0] = 0.4
        prob_weights[-1] = 0.3

    sizes = 3 + 25 * (prob_weights / np.max(prob_weights))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color='#8A2BE2', width=2),
        name='Quantum Coherence Spine'
    ))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=sizes, color=prob_weights, colorscale='Viridis',
            opacity=0.8, line=dict(color='#FFA500', width=1)
        ),
        name='Amplitude State Density'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#0D0E15',
        plot_bgcolor='#0D0E15',
        showlegend=False
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# Hamiltonian panel — genuinely new, not a port. dash.py's "Custom
# Hamiltonian" panel option (dash.py:3025-3029) just printed the raw H-matrix
# text to console; built here as an actual energy-spectrum chart instead.
# ══════════════════════════════════════════════════════════════════════

def build_panel_hamiltonian(hamiltonian_values, name: str = 'Custom') -> plt.Figure:
    """Bar chart of a Hamiltonian's diagonal energy spectrum (the `H_matrix`
    diagonal built in run_vqe_telemetry when hamiltonian_values is passed)."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C['bg'])

    if not hamiltonian_values:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Nessuna Hamiltoniana selezionata.',
                ha='center', va='center', color=C['label'], fontsize=11, transform=ax.transAxes, **MONO)
        return fig

    values = np.asarray(hamiltonian_values, dtype=float)
    _ax_style(ax, f'Spettro Energetico — {name}', 'Autostato |n⟩', 'Energia (Ha)')
    norm = Normalize(values.min(), values.max())
    colors = plt.cm.viridis(norm(values))
    ax.bar(np.arange(len(values)), values, color=colors, width=0.9, edgecolor='none', alpha=0.9)
    ax.axhline(0.0, color=C['label'], lw=0.6, ls=':', alpha=0.6)
    _badge(ax, f'dim={len(values)}  ·  E_min={values.min():.3f}  ·  E_max={values.max():.3f}', C['accent'])

    return fig


# ══════════════════════════════════════════════════════════════════════
# Performance panel — genuinely new, not a port. build_panel_performance
# in dash.py (lines 893 and 2904, both identical) was never implemented:
# it just prints a placeholder and returns None. Built here from data that
# already exists (res['tempo']/['ram']/..., run history) plus a real,
# explicitly-triggered benchmark scan (adapted from
# DiagnosticTools.core_trigger_benchmark, dash.py:2460, canonical version).
# ══════════════════════════════════════════════════════════════════════

def convert_numpy_types_to_python(obj):
    """Recursively converts numpy numeric types to plain Python types so a
    structure is JSON-serializable. Adapted from
    DiagnosticTools._convert_numpy_types_to_python (dash.py:2443)."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_to_python(elem) for elem in obj]
    else:
        return obj


def build_panel_performance(res: Dict, run_history: list) -> plt.Figure:
    """New panel (see module docstring above). Time/RAM per run from
    `run_history` (a list of dicts with at least 'nome', 'n_qubits',
    'tempo', 'ram' — the same fields already returned by run_simulation)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor='#010409')

    if not run_history:
        for ax in axes:
            ax.axis('off')
        fig.text(0.5, 0.5, 'Nessuno storico run disponibile. Esegui una simulazione.',
                  ha='center', va='center', color=C['label'], fontsize=12, **MONO)
        return fig

    names   = [r['nome'] for r in run_history]
    tempi   = [r['tempo'] * 1e3 for r in run_history]
    rams    = [r['ram'] for r in run_history]
    qubits  = [r['n_qubits'] for r in run_history]
    x       = np.arange(len(run_history))

    ax0 = axes[0]
    _ax_style(ax0, 'Wall-clock Time per Run', 'Run #', 'ms')
    bars0 = ax0.bar(x, tempi, color=C['energy'], alpha=0.85)
    for xi, q in zip(x, qubits):
        ax0.text(xi, 0, f'{q}q', ha='center', va='bottom', fontsize=7, color=C['label'], **MONO)

    ax1 = axes[1]
    _ax_style(ax1, 'Statevector RAM per Run', 'Run #', 'MB')
    ax1.bar(x, rams, color=C['purity'], alpha=0.85)

    fig.suptitle(f'Performance — {len(run_history)} run(s) in this session',
                 color=C['title'], fontsize=14, fontweight='bold', **MONO)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    return fig


def run_benchmark_scan(q_range=range(2, 15, 2)) -> pd.DataFrame:
    """Real hardware scaling scan: allocates a fresh DenseSVSimulator at each
    qubit count and times a Hadamard-on-every-qubit circuit. Adapted from
    DiagnosticTools.core_trigger_benchmark (dash.py:2460, canonical version).
    Genuinely slow (dense allocation up to 2**14) — caller should run this
    behind an explicit action + spinner, not automatically."""
    import psutil

    processo_os = psutil.Process()
    ram_iniziale_rss = processo_os.memory_info().rss / (1024 ** 2)

    rows = []
    for q in q_range:
        t0 = time.perf_counter()
        test_sim = de.DenseSVSimulator(n_qubits=q)
        circuito_stress = [["h", idx, -1] for idx in range(q)]
        test_sim.run_circuit_jit_beast_mode(circuito_stress)
        t_elapsed = time.perf_counter() - t0

        ram_corrente_rss = processo_os.memory_info().rss / (1024 ** 2)
        delta_ram_rss = max(0.0, ram_corrente_rss - ram_iniziale_rss)

        rows.append({
            'Qubits': q,
            'Hilbert_Dim': 2 ** q,
            'Time_s': t_elapsed,
            'RAM_Sim_MB': test_sim.memory_mb(),
            'Delta_RAM_RSS_MB': delta_ram_rss,
        })

    return pd.DataFrame(rows)


def build_provenance_json(run_history: list) -> bytes:
    """Provenance export (metadata + run history), SHA-256-signed.
    Adapted from DiagnosticTools.core_trigger_export_json (dash.py:2511) —
    returns bytes for st.download_button instead of google.colab.files.download()."""
    import json
    import platform
    import sys
    import psutil

    try:
        py_ver = sys.version.split()[0]
    except Exception:
        py_ver = "3.x-unknown"

    serializable_runs = convert_numpy_types_to_python(run_history)

    provenance_payload = {
        "metadata": {
            "software_signature": f"dense-evolution-{de.__version__}",
            "export_timestamp_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            "execution_environment": {
                "os": platform.system(),
                "architecture": platform.machine(),
                "python_version": py_ver,
                "hardware": {
                    "cpu_cores_logical": psutil.cpu_count(logical=True),
                    "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                },
            },
        },
        "records": serializable_runs,
    }

    raw_json_bytes = json.dumps(provenance_payload, sort_keys=True, indent=4).encode('utf-8')
    sha256_hash = hashlib.sha256(raw_json_bytes).hexdigest()
    provenance_payload["metadata"]["integrity_sha256"] = sha256_hash

    return json.dumps(provenance_payload, sort_keys=True, indent=4).encode('utf-8')
