import time
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de
from dense_evolution import DARK_BG, PANEL_BG, BORDER, ACC_G, ACC_B, MUTED, TEXT

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

print("="*80)
print("HIGH-DENSITY STRUCTURAL STRESS TEST: 16 QUBITS (65,536 COMPLEX AMPLITUDES)")
print("="*80)

n_qubits = 16
circuit = []

for q in range(n_qubits):
    circuit.append(('h', q))

for q in range(n_qubits):
    circuit.append(('rx', q, 0.432 + (q * 0.1)))
    circuit.append(('ry', q, 1.234 - (q * 0.05)))
    circuit.append(('rz', q, 0.987 + (q * 0.15)))

for q in range(n_qubits - 1):
    circuit.append(('cx', q, q + 1))

for q in range(0, n_qubits // 2):
    circuit.append(('cx', q, n_qubits - 1 - q))

for q in range(0, n_qubits, 2):
    circuit.append(('h', q))

print(f"Circuit Payload: {len(circuit)} structural primitive gates loaded.")

sim = de.DenseSVSimulator(n_qubits=n_qubits, use_gpu=False, use_float32=False)
sim.set_initial_state()

print("\nExecuting dense linear kernel computation...")
start_time = time.time()
sim.run_circuit(circuit)
statevector = sim.get_statevector()
execution_time = time.time() - start_time

print(f"Execution Completed in: {execution_time:.4f} seconds.")

probabilities = np.abs(statevector)**2
norma_l2 = np.sum(probabilities)

print(f"L2-Norm Conservation Drift: {norma_l2:.15f}")

sorted_indices = np.argsort(probabilities)[::-1]
top_indices = sorted_indices[:50]
top_probabilities = probabilities[top_indices]
top_amplitudes = statevector[top_indices]

print("\nGenerating structural visualization plots using Cell 2 native style...")
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Dense-Evolution Stress Test Matrix ({n_qubits} Qubits — 65,536 Amplitudes)', fontsize=14, fontweight='bold', color=ACC_G)

ax1.bar(range(50), top_probabilities, color=ACC_B, edgecolor=BORDER, alpha=0.8, label='State Probability')
ax1.set_title('Top 50 Computational States Peaks Distribution', fontsize=11, color=TEXT)
ax1.set_xlabel('Ranked States Indices (Highest to Lowest)', color=MUTED)
ax1.set_ylabel('Probability Magnitude |ψ|²', color=MUTED)
ax1.grid(True, linestyle='--', alpha=0.3, color=BORDER)
ax1.legend()

ax2.scatter(top_amplitudes.real, top_amplitudes.imag, c=top_probabilities, cmap='cool', edgecolors=BORDER, s=50, alpha=0.9, label='Quantum Amplitude')
ax2.axhline(0, color=BORDER, linestyle='-', alpha=0.5)
ax2.axvline(0, color=BORDER, linestyle='-', alpha=0.5)
ax2.set_title('Complex Amplitudes Phase Space Constellation (Real vs Imag)', fontsize=11, color=TEXT)
ax2.set_xlabel('Real Component Re(ψ)', color=MUTED)
ax2.set_ylabel('Imaginary Component Im(ψ)', color=MUTED)
ax2.grid(True, linestyle='--', alpha=0.3, color=BORDER)
ax2.legend()

info_text = f"Hardware Metrics:\nRuntime Time: {execution_time:.4f}s\nNorm L2: {norma_l2:.14f}\nGate Payloads: {len(circuit)}\nPrecision: float64/complex128"
props = dict(boxstyle='round', facecolor=PANEL_BG, edgecolor=BORDER, alpha=0.8)
ax1.text(0.55, 0.95, info_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=props, color=TEXT)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("COMPUTATIONAL WAVEFUNCTION PEAKS STATE LOG")
print("="*80)
for rank, idx in enumerate(top_indices[:10]):
    binary_state = format(idx, f'0{n_qubits}b')
    print(f"Rank {rank+1:02d} | State: |{binary_state}⟩ (Idx: {idx:5d}) | Amp: {statevector[idx].real:+.6f} {statevector[idx].imag:+.6f}j | Prob: {probabilities[idx]*100:6.3f}%")
print("="*80)

