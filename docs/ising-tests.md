# Dense-Evolution-Ising-Tests: The Full Research Log

The companion repository [**Dense-Evolution-Ising-Tests**](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests)
cross-validates this engine against PennyLane and closed-form references, and hosts a running,
honestly-reported research log of physics experiments built on top of it — including negative
results, reported in full because they carry the same scientific value as positive ones. This
page mirrors that repository's README in full, including every measured result and every plot.

This repository contains a rigorous empirical study, raw datasets, and quantum error mitigation protocols executed on **Dense Evolution (v8.1.21)**—a high-performance *Statevector* quantum simulator. Utilizing 64-bit double precision (`complex128`) and hardware-accelerated static compilation via the JAX XLA engine, this project maps the non-linear physics of the Transverse Field Ising Model (TFIM), Tight-Binding Fermionic dynamics, and semiconductor solid-state thermodynamics.

Jump straight to any result below and explore what catches your eye — every claim links to the exact script that produced it on GitHub, so you can run it yourself. Or start with the three newest, most rigorously validated additions:

---

## 🆕 Latest Results (start here)

All three of these found and fixed a *real* error in an AI-drafted physics claim, not just implemented what was proposed as-is:

- **[Loschmidt Echo](#17-loschmidt-echo-a-real-time-reversal-test-not-a-static-noise-channel)** — a claimed ZNE fidelity recovery turned out to come from dead code that never actually evolved anything. The real kicked-Ising forward/backward circuit recovers return fidelity from **0.7769 → 0.9965**.
- **[A Real VQE Optimization Loop](#18-topological-mott-isolator-a-real-optimization-loop-validated-against-exact-diagonalization)** — a reported "ground state" was actually the energy of one random, unoptimized guess. Real gradient-descent optimization, validated against exact diagonalization, closes nearly all of the gap.
- **[Real DFT-Derived Material Parameters](#19-from-a-made-up-u-to-a-real-one-dft-and-dielectric-screening-for-actual-gaas)** — a non-converged DFT calculation was used anyway. Fixed the convergence, caught a genuine wavefunction-stability trap along the way, then grounded the result in GaAs's real dielectric constant.

[![Loschmidt echo: raw vs. ZNE-corrected return fidelity](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/loschmidt_echo_zne.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/loschmidt_echo_zne.png)

---

## Repository Layout

```
Dense-Evolution-Ising-Tests/
├── scripts/     # 17 production scripts (see below) -- tracked in git
├── tests/       # pytest suite, run by CI on every push
├── data/        # CSV outputs -- NOT tracked (.gitignore); populated by running scripts/tests
├── images/      # PNG outputs -- NOT tracked (.gitignore); populated by running scripts/tests
└── README.md
```

`git clone` gives you exactly the scripts and tests, nothing pre-generated -- run anything and `data/`/`images/` fill up with fresh output, so there's never any ambiguity about whether what you're looking at is old or new. Pre-made results for browsing without running anything live as attachments on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) instead (also what every image embedded in this README below links to).

## Repository Architecture & Ecosystem

- **`scripts/scan_ising.py`**: Automated data pipeline responsible for high-resolution parameter sweeps and graphical rendering of the ideal ferromagnetic phase transition using a true variational ansatz. Produces `data/transizione_fase_ising.csv`.
- **`scripts/plot_ising.py`**: Computes the first-order numerical derivative (quantum susceptibility) from the CSV dataset to locate the exact critical phase boundary. Produces `images/curva_transizione_ising.png`.
- **`scripts/zne_mitigation.py`**: Mathematical implementation of a stochastic Richardson Zero-Noise Extrapolation (ZNE) protocol over discrete Pauli-Z phase dephasing channels with 2,000 hardware shot sampling. Produces `data/dati_mitigazione_zne.csv` and `images/transizione_ising_mitigata.png`.
- **`scripts/vqe_gradient.py`**: Exact numerical finite-difference gradient tracker (`h = 1e-5`) mapping the variational energy landscape and locating stationary points. Produces `data/vqe_gradient_landscape.csv` and `images/vqe_gradient_landscape.png`.
- **`scripts/vqe_jax_grad.py`**: Advanced VQE gradient execution computing the exact Parameter-Shift Rule gate-by-gate via the chain rule over a massively parallel 73,500-track JAX batch array. Produces `data/vqe_jax_gradient.csv` and `images/vqe_jax_gradient.png`.
- **`scripts/quantum_defect_scanner.py`**: Isotropic resilience topology mapper evaluating node-by-node quantum coherence under a localized parametric RZ dephasing rotation via `run_parametric_batch_jit()`. Produces `data/mappa_difetti_silicio.csv` and `images/mappa_difetti_silicio.png`.
- **`scripts/next_gen_silicon.py`**: Solid-state bandstructure designer tracking continuous dispersion shifts induced by 5% mechanical lattice tensile strain via Harrison's hopping law. Produces `data/bande_nuovo_silicio.csv` and `images/confronto_nuovo_silicio.png`.
- **`scripts/manufacturing_thermodynamics.py`**: Quantum lattice thermodynamics simulator modeling electron-phonon scattering and decoherence via Bose-Einstein statistical distributions over a 10–400 K temperature sweep. Produces `data/validazione_fabbricazione_silicio.csv` and `images/validazione_fabbricazione.png`.
- **`scripts/vqe_silicon_molecular.py`**: Variational Quantum Eigensolver tracking self-consistent Potential Energy Curves (PEC) and Born-Oppenheimer molecular dissociation limits for a silicon dimer, at a fixed variational angle $\theta=0.38$ rad. Produces `data/vqe_molecola_silicio.csv` and `images/curva_potenziale_silicio.png`.
- **`scripts/vqe_silicon_molecular_optimized.py`**: Same PEC, but with a single shared $\theta$ found by real Adam optimization across all $R$ using the exact chain-rule Parameter-Shift Rule gradient, batched per epoch. Produces `data/vqe_molecola_silicio_ottimizzata.csv` and `images/curva_potenziale_silicio_ottimizzata.png`. See Section 9b.
- **`scripts/vqe_silicon_molecular_optimized_per_bond.py`**: Same PEC, but with 5 *independent* Givens angles (one per bond) instead of one shared $\theta$ -- a more realistic hardware-efficient VQE ansatz. Produces `data/vqe_molecola_silicio_ottimizzata_per_legame.csv` and `images/curva_potenziale_silicio_ottimizzata_per_legame.png`. See Section 9c.
- **`scripts/vqe_extreme_geometries.py`**: Generalizes the per-bond ansatz to independent per-bond interatomic distances (an irregular/distorted chain instead of one shared $R$), benchmarking the rigid shared-angle approximation against per-bond adaptive optimization across 6 extreme/irregular geometry scenarios. Produces `data/vqe_extreme_geometries.csv` and `images/vqe_extreme_geometries.png`. See Section 9e.
- **`scripts/vqe_extreme_geometries_deep.py`**: Deeper 12-parameter (7-qubit/6-bond) generalization of the same benchmark, plus a genuine minimum-energy conformational search (joint per-bond $R_q$ and $\theta_q$ optimization, not hand-picked geometries). Produces `data/vqe_extreme_geometries_deep.csv`, `data/vqe_extreme_geometries_deep_conformazioni.csv`, and `images/vqe_extreme_geometries_deep.png`. See Section 9f.
- **`scripts/zne_stabilized_psr_gradient.py`**: Corrects each individual single-gate Parameter-Shift-Rule term with Zero-Noise Extrapolation *before* the PSR chain-rule combination, testing whether this stabilizes the gradient under `NoiseModel`. Produces `data/zne_stabilized_psr_gradient.csv` and `images/zne_stabilized_psr_gradient.png`. See Section 12.
- **`scripts/zne_adaptive_psr_gradient.py`**: Attempts to fix Section 12's zero-crossing failure mode with a confidence-attenuated ("adaptive") ZNE correction via `dense_evolution.healing.calculate_delta_preemp`. Honest negative result. Produces `data/zne_adaptive_psr_gradient.csv` and `images/zne_adaptive_psr_gradient.png`. See Section 13.
- **`scripts/zne_snr_adaptive_psr_gradient.py`**: A second attempt at Section 13's problem, attenuating the ZNE correction via the correction term's own signal-to-noise ratio instead of the SEM of a single measurement. Also an honest negative result. Produces `data/zne_snr_adaptive_psr_gradient.csv` and `images/zne_snr_adaptive_psr_gradient.png`. See Section 14.
- **`scripts/sophia_reflection.py`**: A real density-matrix ZNE noise-coherence trajectory (`dense_evolution.mitigation.zne_density_matrix`/`uhlmann_fidelity`, distinct from `zne_mitigation.py`'s hand-rolled scalar protocol above) on a 2-qubit Bell state across a 16-point depolarizing-noise sweep. Produces `data/sophia_reflection.csv` and `images/sophia_reflection.png`. See [`SOPHIA_REFLECTION.md`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/SOPHIA_REFLECTION.md) for the real measured trajectory and an LLM's own reflection on it, written after seeing the actual output.
- **`scripts/channel_order_noncommutativity.py`**: Tests whether the *order* of applying two different noise channels (dephasing then amplitude damping, vs. the reverse) leaves a measurable fingerprint on a 3-qubit circuit's output distribution — a real, honestly-confirmed positive result (see Section 16), unlike most of the other claims traced back to the same August 2025 archive. Produces `data/channel_order_noncommutativity.csv` and `images/channel_order_noncommutativity.png`.
- **`scripts/loschmidt_echo_zne.py`**: Runs a real "kicked Ising" forward circuit followed by its exact inverse on a 4-qubit chain, injecting an amplitude-damping channel between every layer, and checks whether density-matrix ZNE (`zne_density_matrix`/`uhlmann_fidelity`) recovers return fidelity lost to noise. A noiseless self-check (fidelity must be exactly 1.0) gates the noisy results. Produces `data/loschmidt_echo_zne.csv` and `images/loschmidt_echo_zne.png`.
- **`scripts/vqe_tmi_material_design.py`**: Real Adam optimization (exact JAX autodiff via `circuit_to_energy_fn`, multi-start, batched with `jax.vmap`) of a hardware-efficient ansatz against a Topological Mott Isolator Hamiltonian, swept over the Mott repulsion U, validated at every U against exact dense diagonalization (the variational principle `E_vqe >= E_exact` is asserted, not assumed). Also runs the same pipeline at a real GaAs point (DFT-derived hopping, dielectrically-screened on-site Coulomb repulsion — see Section 19). Produces `data/vqe_tmi_material_design.csv`, `data/vqe_tmi_material_design_gaas_real.csv`, and `images/vqe_tmi_material_design.png`.
- **`tests/test_pennylane_comparison.py`**: Automated cross-validation suite integrating PennyLane as a baseline verification engine. It programmatically contrasts the JAX/XLA statevector predictions generated by Dense Evolution against PennyLane's analytical execution to enforce strict regression boundaries in the CI pipeline.
- **`tests/test_analytical.py`**: Built-in mathematical validation suite executing 5 zero-external-dependency tests. It verifies Potential Energy Curve (PEC) physical boundaries, exact Parameter-Shift Rule (PSR) gradients on $RY+\langle Z \rangle$, Harrison's strain-hopping ratios, and time-reversal dispersion symmetries under machine-precision tolerances ($\le 10^{-10}$).
- **`tests/test_integration_smoke.py`**: Imports and executes the REAL functions from `scripts/vqe_gradient.py`, `scripts/zne_mitigation.py`, `scripts/scan_ising.py`, `scripts/next_gen_silicon.py` (not hand-derived copies), cross-validated against PennyLane or closed-form references.



---

## Scientific Discoveries & Empirical Evidence

### 1. Quantum Phase Transition & Order Parameters

We present a rigorous physical validation of the longitudinal spin-correlation order parameter $\langle H_{zz} \rangle$ governed by the 1D Transverse Field Ising Model Hamiltonian:

$$H = -\sum_{i} Z_i Z_{i+1} - g\sum_{i} X_i$$

As the transverse field coupling strength $g$ sweeps from $0.0$ to $2.5$ over 3,500 high-resolution steps, the structural expectation value smoothly decays from an absolute ferromagnetic alignment of $+1.0000$ down to $+0.0050$. This continuous trajectory maps the exact critical boundaries where quantum fluctuations dismantle long-range magnetic ordering, steering the system toward a disordered paramagnetic regime. The critical phase transition boundary is resolved via quantum susceptibility metrics at exactly **$g = 1.309$** with a maximum peak susceptibility of **$1.0000$** under zero-drift conditions.

The ansatz deploys alternating CX–RZ–CX entangling blocks across all 11 nearest-neighbor qubit pairs on a 12-qubit chain, followed by parametric RX rotations scaled to the transverse field strength ($\theta = 0.6 \cdot g$). The `<H_zz>` order parameter is computed analytically from the statevector probability distribution via bitwise parity extraction.

[![Quantum Ising Phase Scan and Susceptibility](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_transizione_ising.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_transizione_ising.png)

---

### 2. Quantum Error Mitigation via Real Stochastic Richardson Extrapolation (ZNE)

To circumvent non-unitary noise without physical hardware overhead, a classical-quantum hybrid mitigation protocol was deployed under a realistic stochastic Pauli-Z dephasing Kraus channel. By scaling the noise density via stretching coefficients ($\lambda_1 = 1.0, \lambda_2 = 2.0$) over $2,000$ discrete hardware shots, a linear Richardson extrapolation was computed:

$$E(0) = 2E(\lambda_1) - E(\lambda_2)$$

The protocol operates on Bloch wavevector states $|\psi(k)\rangle = \frac{1}{\sqrt{N}} \sum_q e^{iqk} |1_q\rangle$ injected over 25 k-points spanning the full Brillouin zone $[-\pi, \pi]$. The base dephasing probability per qubit is $p = 0.06 \cdot \lambda$, applied stochastically via per-shot Kraus channel sampling with controlled seeds.

The ZNE protocol successfully reconstructed the unperturbed, zero-noise ideal target trajectory, forcing the corrupted noisy minimum at $k=0$ (degraded up to $-3.3155\text{ eV}$) back to its true analytic target value of **$-4.2467\text{ eV}$** without introducing non-linear artifacts.

[![Stochastic Zero-Noise Extrapolation Results](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_transizione_noisy.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_transizione_noisy.png)

---

### 3. Numerical Finite-Difference Gradient Mapping (VQE Energy Landscape)

A brute-force numerical gradient sweep over the full VQE variational energy landscape was executed using a centered finite-difference scheme with step $h = 10^{-5}$ radians:

$$\frac{\partial E}{\partial \theta} \approx \frac{E(\theta + h) - E(\theta - h)}{2h}$$

The ansatz uses Givens rotation excitation-preserving blocks (CX–RY–CX–RY–CX chains) initialized from a single-excitation Fock state $|100000\rangle$, preserving strict particle-number conservation throughout. 3,500 continuous $\theta$ values spanning $[0, 2\pi]$ are evaluated over a 6-qubit tight-binding Hamiltonian with $t_{hopping} = 2.11$ eV.

The gradient landscape confirms the exact analytic minimum bound at:

$$E_{ground} = -2 \cdot t_{hopping} = -4.22 \text{ eV}$$

with all stationary points and gradient zero-crossings fully resolved, and no vanishing gradient plateaus present under the compact excitation-preserving ansatz.

> **Note:** This script (`vqe_gradient.py`) uses classical finite-difference differentiation. For exact quantum-native analytical gradients via Parameter-Shift Rule, see Section 6 (`vqe_jax_grad.py`).

#### 3b. Closed Form: E(θ) Without Simulating a Circuit at All

This ansatz shares one $\theta$ across every bond instead of the independent per-bond angles of Section 9c/9d — so instead of landing anywhere on the single-excitation manifold, sweeping $\theta$ traces one 1-parameter curve through it. That curve has a closed form, using the same amplitude-cascade recursion behind Section 9d's discovery:

$$c_0(\theta) = \cos^{N-1}(\theta), \qquad c_q(\theta) = \sin(\theta)\cos^{N-1-q}(\theta) \quad (q=1,\ldots,N-1)$$

`calcola_energia_vqe`'s kinetic sum is **periodic** ($q_{\text{next}} = (q+1) \bmod N$, all $N=6$ bonds including the wraparound $5\to0$ — not the $N{-}1$ open-chain bonds used by the molecular PEC scripts), giving:

$$E(\theta) = -2\,t_{hopping}\sum_{q=0}^{N-1} c_q(\theta)\, c_{(q+1)\bmod N}(\theta)$$

Verified exact (machine precision, $\sim 10^{-15}$) against `calcola_energia_vqe` across the full sweep, including at the printed checkpoints — e.g. $\theta=0.4471\text{ rad} \to E=-3.9489\text{ eV}$, gradient $-8.440000$ at $\theta=0$ — with **no circuit simulation** needed to evaluate it, `scripts/vqe_gradient.py`'s `energia_forma_chiusa()`. `tests/test_integration_smoke.py::test_vqe_gradient_closed_form_matches_real_circuit_exactly` checks the identity at 7 points across the range.

---

### 4. Parallel Quantum Defect Mapping via JAX Parallel Batching

Using the native `run_parametric_batch_jit()` engine, we mapped the isotropic resilience of an entangled state against localized dephasing noise. A 12-qubit entangled chain is prepared by uniform RY($\pi/4$) rotations followed by a full CX entangling ladder. A parametric RZ dephasing gate is injected node-by-node on the diagonal of the batch parameter grid, resulting in 12 concurrent independent execution tracks compiled in a single JAX XLA macro-cycle.

The evaluation maps the systematic loss of $\langle X \rangle$ single-qubit coherence:

$$\langle X_q \rangle = \text{Re}\left[\sum_i \psi_i^* \psi_{i \oplus 2^q}\right]$$

> **Correction 1 (audit finding, dense-evolution 8.1.21):** `run_parametric_batch_jit()` assigns one `parameter_batch` column per rotation gate *in the order the gates appear*, even when a gate is given a literal float instead of a string placeholder — the literal is silently discarded. The batch grid used to have only 12 columns while the circuit has 24 rotation slots (12 fixed RY(π/4) + 12 varying RZ), so the RY gates absorbed the intended RZ values and the true RZ columns ran out of bounds (silently clipped by JAX instead of raising). Fixed by supplying all 24 slots explicitly.
>
> **Correction 2 (found the same day, while trying to explain Correction 1's numbers):** `DenseSVSimulator` uses MSB-first indexing internally ($\text{phys} = N_Q{-}1{-}\text{qubit}$, see `_cx_numpy`/`apply_cx` in `dense_evolution/simulator.py`) — gate-qubit $q$ lives at physical array bit $N_Q{-}1{-}q$. The script's coherence measurement used `1 << local_qubit` directly on the *gate*-qubit index, reading a **different physical qubit** than the one that actually received that row's dephasing. The "70.71% / 50% / 43.88%" pattern originally reported here (and the "interplay between the RY layer and the CX ladder" explanation) was this indexing artifact, not real physics.

With both fixed, the true pattern is much simpler: **11 of the 12 nodes give an identical residual coherence, $43.88\%$**, and only the very last node in the chain differs, at $62.05\%$. We don't have a fully derived analytic explanation for *why* exactly one node is different (an attempted derivation via "the CNOT control's coherence is preserved" turned out to rest on a false premise — that invariance holds for Z-basis populations, not X-basis coherence — so it's reported here as a verified empirical fact, not a proven mechanism).

[![True Quantum Defect Mapping Graph](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/mappa_difetti_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/mappa_difetti_silicio.png)

---

### 5. Rigorous 1D Crystalline Lattice Dispersion

We resolved the exact 1-electron fermionic Bloch state dispersion relation mapped via Jordan-Wigner transformations. By evaluating the pure exchange interactions ($\langle X_i X_{i+1} + Y_i Y_{i+1} \rangle$) and applying strict periodic boundary conditions (PBC), the engine resolves the full, continuous single-band cosine energy spectrum:

$$E(k) = -2t \cos(k)$$

This eliminates artificial scaling factors and rigid offsets, delivering an honest statevector simulation of tight-binding quantum dynamics under strict 1-fermion subspace conservation. The Bloch states are analytically constructed as $|\psi(k)\rangle = \frac{1}{\sqrt{N}} \sum_q e^{iqk} |1_q\rangle$ over 8 qubits, with the kinetic energy expectation evaluated via tensor-product bitwise XY-operator matrix elements.

[![Rigorous Quantum Tight-Binding Dispersion](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/bande_silicio_ibrido.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/bande_silicio_ibrido.png)

---

### 6. Analytical Gradients via Parallel Parameter-Shift Rule

> **Correction (audit finding, dense-evolution 8.1.21):** the original implementation shifted the *shared* variational parameter $t$ by $\pm\pi/2$ and read the resulting energies straight off the batch. The textbook Parameter-Shift Rule is only exact when a **single gate's own parameter** is shifted while every other gate is held fixed — here each bond applies *two* `ry` gates both driven by $t$ (`param_vqe = t`, `param_vqe_inv = -t`), so the shared-shift reading conflated their contributions. Verified against an independent finite-difference reference: the old heuristic could disagree with the true $dE/dt$ by 100%, including the wrong sign.

The corrected, mathematically exact gradient follows from the chain rule over every gate parameter individually:

$$\frac{dE}{dt} = \sum_{q} \left[ \frac{\partial E}{\partial \theta_{A,q}} \cdot \frac{d\theta_{A,q}}{dt} + \frac{\partial E}{\partial \theta_{B,q}} \cdot \frac{d\theta_{B,q}}{dt} \right], \qquad \frac{\partial E}{\partial \theta} = \frac{1}{2}\left[E(\theta+\tfrac{\pi}{2}) - E(\theta-\tfrac{\pi}{2})\right]$$

where each $\partial E/\partial \theta$ is a genuine single-gate Parameter-Shift Rule evaluation (only that one gate shifted, all 21 others held at their base value), and $d\theta_A/dt = 1$, $d\theta_B/dt = -1$ are the known chain-rule coefficients. Verified against finite differences: agreement to $\sim 10^{-9}$, limited by finite-difference truncation rather than by the PSR itself.

By packing every shifted configuration concurrently into `run_parametric_batch_jit()`, JAX XLA processed **73,500 continuous configurations** (3,500 $\theta$ values $\times$ 21 tracks each: 1 center point + 2 shifts $\times$ 10 gate parameters) in a single chunked macro-batch execution completed in **570 seconds** on CPU — substantially more expensive than the old (incorrect) shortcut, since an exact gradient over a shared parameter genuinely requires one PSR evaluation per gate it drives, not one shift of the shared variable.

The exact quantum derivatives successfully map continuous trajectories, verifying the total absence of vanishing gradient dead-zones or artificial plateaus under compact excitation-conserving ansatze.

[![Exact Parameter-Shift Rule Gradients](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_jax_gradient.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_jax_gradient.png)

---

### 7. Strained Silicon Bandstructure Engineering (3,500-Point Sweep)

We modeled a continuous dispersion profile mapping a high-mobility Strained Silicon configuration under a $5\%$ tensile strain ($\varepsilon = 0.05$). By perturbing the atomic equilibrium distances, the physical Hamiltonian undergoes an exponential inter-orbital hopping decay dictated by Harrison's law:

$$t(\varepsilon) = \frac{t_0}{(1 + \varepsilon)^2}$$

The high-resolution 3,500-point k-space parameter sweep executed via JAX maps the physical contraction of the modal hopping energy from the standard $\pm 4.2200\text{ eV}$ limits down to the accurate engineered boundary of **$\pm 3.8277\text{ eV}$** across the Brillouin zone. The simulation uses 8-qubit fermionic Bloch states with Jordan-Wigner XY exchange operators evaluated over all 8 bonds under periodic boundary conditions.

[![Strained Silicon Next-Gen Bandstructure](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_nuovo_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_nuovo_silicio.png)

---

### 8. Quantum Lattice Thermodynamics: Phonon Scattering & Decoherence

A quantum-statistical simulation of electron-phonon scattering decoherence was executed over a 10–400 K temperature sweep at 3,500 discrete points, modeling the thermal degradation of coherent electronic hopping in a silicon lattice.

The Debye-Bose-Einstein phonon occupancy is computed as:

$$\bar{n}(\omega, T) = \frac{1}{\exp\!\left(\frac{\hbar\omega}{k_B T}\right) - 1}$$

with $\hbar\omega = 32\text{ meV}$ (silicon optical phonon branch). The effective hopping integral degrades with phonon bath population according to:

$$t_{\text{eff}}(T) = t_0 \left(1 - 0.15 \cdot \bar{n}(\omega, T)\right)$$

This captures the physical mechanism by which thermally-activated phonon scattering reduces long-range electronic coherence. A fixed Bloch state $|\psi(k = \pi/4)\rangle$ is used as the probe state on 8 qubits; the coherent kinetic energy $E(k, T)$ is evaluated via XY-operator matrix elements at each temperature step, tracking the monotonic energy suppression from cryogenic to room temperature.

[![Quantum Lattice Thermodynamics: Phonon Decoherence](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/validazione_fabbricazione.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/validazione_fabbricazione.png)

---

### 9. Molecular VQE and Potential Energy Dissociation Curves

We mapped the exact Born-Oppenheimer Potential Energy Curve (PEC) for a silicon dimer system via a classical-quantum hybrid variational loop. The effective Hamiltonian tracks electronic hopping integrals $t(R)$ alongside nuclear Coulomb repulsion fields $V_{rep}(R)$ decaying over the interatomic coordinate:

$$t(R) = t_0 \, e^{-\beta(R - R_0)}, \qquad V_{rep}(R) = V_0 \, e^{-\gamma(R - R_0)}$$

with $t_0 = 2.11$ eV, $\beta = 1.5$ Å$^{-1}$, $R_0 = 2.35$ Å, $V_0 = 5.4$ eV, $\gamma = 3.0$ Å$^{-1}$. The ansatz is a 6-qubit excitation-preserving Givens circuit initialized from the single-fermion Fock state $|100000\rangle$, with a fixed variational angle $\theta = 0.38$ rad.

#### 9b. Adam-Optimized PEC with the Exact Chain-Rule Gradient

`vqe_silicon_molecular_optimized.py` replaces the fixed $\theta = 0.38$ with a real per-$R$ Adam optimization, using the same exact chain-rule Parameter-Shift Rule gradient validated in Section 6 (agreement with finite differences to $\sim 10^{-9}$). All $R$ points are optimized in parallel — every Adam epoch batches every $R$'s current $\theta$ into a single `run_parametric_batch_jit()` call, rather than looping epochs inside a per-$R$ loop.

The result is a genuine physical insight, not just a better number: $E(R,\theta) = -\frac{t(R)}{2}\,K(\theta) + V_{rep}(R)$, where $K(\theta)$ is the ansatz's kinetic term. Since $t(R) > 0$ everywhere and $V_{rep}(R)$ doesn't depend on $\theta$, the location of the energy minimum over $\theta$ — $\arg\max_\theta K(\theta)$ — **does not depend on $R$ at all**. Optimizing independently at 200 $R$ points converges to the same $\theta^\star \approx 0.613$ rad everywhere (confirmed against an independent fine-grained scan of $K(\theta)$, whose true maximum sits at $\theta = 0.6129$), not a curve. This is a property of the model (the $R$-dependence enters only as a positive multiplicative prefactor on a $\theta$-only kinetic term), not a bug in the optimizer.

Because $\theta = 0.38$ was not that optimum, using the correct $\theta^\star$ deepens the binding well substantially: the global minimum moves from $-0.302\text{ eV}$ at $R \approx 3.32\text{ Å}$ (fixed $\theta$) to $-0.4615\text{ eV}$ at $R \approx 3.17\text{ Å}$ (optimized), an improvement that grows to over $3\text{ eV}$ at short range where the kinetic term dominates.

[![Adam-Optimized Silicon Dimer PEC](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata.png)

The 3,500-point variational sweep over $R \in [1.2, 4.5]$ Å cleanly resolves the stable binding landscape, isolating the exact molecular equilibrium bond length and asymptotic dissociation limits without numerical instabilities.

[![Silicon Dimer Potential Energy Curve](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio.png)

#### 9c. Per-Bond Optimized PEC — 5 Independent Givens Angles

`vqe_silicon_molecular_optimized_per_bond.py` asks a sharper question: Section 9b's shared $\theta$ is $R$-independent *by construction*, since $R$ only ever enters as a scalar prefactor on a $\theta$-only kinetic term — true no matter how that single $\theta$ is chosen. A genuinely richer test is whether a **multi-parameter** ansatz (one independent Givens angle $\theta_q$ per bond, $q=0..4$, same particle-number-conserving structure) settles on a uniform value across bonds, or differentiates.

It differentiates, and cleanly: the 5 bonds converge to $\theta_0,...,\theta_4 \approx 0.234, 0.439, 0.632, 0.818, 1.024$ rad — an almost perfectly even $\approx 0.2$ rad spacing, not noise. This value set is itself still $R$-independent (same underlying reason as 9b — the argmax of a multi-variable $K(\vec\theta)$ under a positive scalar prefactor doesn't move with $R$ either), but it is **not** the "all bonds equal" point: the shared-$\theta$ ansatz was leaving real variational power on the table by forcing symmetry across bonds that the true optimum doesn't have. The minimum deepens further, from $-0.4615\text{ eV}$ (shared, optimized) to $-0.6685\text{ eV}$ (per-bond, optimized).

[![Per-Bond Optimized Silicon Dimer PEC](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata_per_legame.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata_per_legame.png)

#### 9d. Closed Form: the Optimizer Rediscovers the Tight-Binding Ground State

Section 9c's near-even $\approx 0.2$ rad spacing between bond angles isn't the real story — it's a side effect of *what* the optimizer actually converges to. The sequential Givens-rotation ansatz (CX–RY–CX–RY–CX per bond, starting from $|100\ldots0\rangle$) can prepare **any** normalized single-excitation state exactly — it's a universal staircase state-preparation circuit for that Hilbert-space sector. So maximizing the total hopping energy $K(\vec\theta)$ over this ansatz is *unconstrained*: it finds the true maximum of $K$ over every possible single-excitation state, which is exactly the top-eigenvalue problem of an open tight-binding chain — the same math as a **particle in a box**. The optimizer, with no physics told to it beyond "maximize this energy," rediscovers the box's ground state on its own.

The amplitude at site $q$ (0-indexed, $N$ sites) settles on the first sine mode:

$$c_q \propto \sin\!\left(\frac{(q+1)\pi}{N+1}\right), \qquad K_{max} = 4\cos\!\left(\frac{\pi}{N+1}\right)$$

and the per-bond Givens angle that *prepares* this profile via the sequential construction has its own closed form ($r_q$ = the tail norm $\sqrt{\sum_{k=q}^{N-1} c_k^2}$):

$$\theta_q = \arcsin\!\left(\frac{c_q}{r_q}\right)$$

No optimizer needed: plugging this formula directly into the circuit reproduces the numerically Adam-optimized result to **machine precision** ($\sim 10^{-15}$), and holds for every chain length tested (4, 5, 6, 7, 8, 10 qubits) — not a coincidence specific to this 6-qubit example. `scripts/vqe_silicon_molecular_optimized_per_bond.py` implements this as `theta_ground_state_closed_form()` / `kinetic_max_closed_form()`, and `tests/test_vqe_molecular_per_bond.py` verifies the identity exactly (`test_closed_form_ground_state_matches_script_kinetic_maximum`, `test_closed_form_generalizes_across_chain_lengths`).

#### 9e. Extreme/Irregular Geometry Benchmark — When Does a Rigid Shared Angle Actually Fail?

Sections 9-9d all sweep one interatomic distance $R$ shared by the whole chain. `vqe_extreme_geometries.py` breaks that symmetry: each of the 5 bonds gets its own distance $R_q$, modeling an irregular or distorted chain (extreme compression, near-dissociation stretch, a localized "mutated" bond, or several at once) instead of a smooth uniform-$R$ sweep. Hopping $t_q(R_q)$ stays strictly local (bond-by-bond), while the steric/electrostatic repulsion keeps Section 9's exact single-formula shape, evaluated at the geometry's mean bond length — so the model collapses **exactly** onto Section 9's original scalar energy in the uniform-$R$, uniform-$\theta$ limit (`tests/test_vqe_extreme_geometries.py::test_energy_matches_reference_scalar_formula_for_uniform_geometry`).

A naive comparison against the fixed $\theta=0.38$ baseline is scale-confounded: every uniform-$R$ geometry already shows a large "improvement" from per-bond optimization, purely because the per-bond ansatz can reach the true sine-mode kinetic maximum (Section 9d) that a single shared angle structurally cannot — regardless of whether the geometry is "extreme." The metric that actually isolates the effect of geometry *irregularity* is scale-normalized:

$$\text{deficit\_fraction} = \frac{K_{\text{per-bond}}^{\star} - K_{\text{shared}}^{\star}}{K_{\text{per-bond}}^{\star}}$$

comparing per-bond adaptive optimization against the **best achievable single shared angle** — re-optimized per geometry, not the fixed $0.38$ — which cancels the overall $t(R)$ energy scale (verified scale-invariant across uniform geometries to $< 5\times10^{-3}$).

Across 6 hand-picked scenarios, the result is not "any distortion is worse":

| Scenario | `deficit_fraction` |
|---|---|
| Uniform equilibrium / compressed / dissociated | $0.169$ (identical across all three — scale-invariant) |
| Single localized mutated bond | $0.103$ (**below** the uniform baseline) |
| Alternating compressed/stretched pattern | $0.027$ (**below** the uniform baseline) |
| Two mutated bonds at opposite chain ends | $\mathbf{0.500}$ (nearly 3$\times$ the uniform baseline) |

A single shared angle is not simply "worse under any distortion" — it specifically struggles when the geometry forces it to reconcile two strongly-weighted but topologically distant bonds (opposite ends of the chain) at once, something one scalar parameter cannot do but per-bond adaptation can. This is a tight-binding / single-excitation hopping toy model, not ab-initio electronic structure: "rigid angle fails" here means concretely that a single shared $\theta$ cannot simultaneously satisfy 5 different per-bond stationarity conditions once the $R_q$ differ, leaving variational energy on the table that per-bond optimization recovers — nothing here claims to model real electron correlation or nuclear quantum repulsion.

[![Extreme/Irregular Geometry Benchmark](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_extreme_geometries.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_extreme_geometries.png)

#### 9f. Deeper Ansatz (12 Parameters) + a Genuine Minimum-Energy Conformational Search

`vqe_extreme_geometries_deep.py` generalizes Section 9e's benchmark from $N_Q=6$ (5 bonds, 10 parameters) to $N_Q=7$ (6 bonds, 12 parameters) — the same model, one more bond in the chain, everything already parametric in $N_Q$. The qualitative `deficit_fraction` pattern does **not** carry over unchanged: at 12 parameters, `mutazione_localizzata` (0.188) and `distorsione_alternata` (0.264) now sit *above* the uniform baseline (0.163), whereas at 10 parameters both sat *below* it (0.103 / 0.027 vs. 0.169). Only `mutazioni_congiunte_estremi` is a robust standout across both depths (0.464 at 12 parameters vs. 0.500 at 10 — still, by far, the worst case).

[![12-Parameter Extreme-Geometry Benchmark](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/vqe_extreme_geometries_deep.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/vqe_extreme_geometries_deep.png)

Beyond the hand-picked geometries, `optimize_geometry_and_theta_jointly()` searches for a genuine minimum-energy conformation by optimizing the bond distances $R_q$ *jointly* with the per-bond angles (classical analytic gradient for $R$ — it only enters through $t_q(R_q)$ and the repulsion term, no PSR needed). This required a **per-bond** repulsion term, not the mean-based one used by the fixed-geometry benchmark above: under free $R$ optimization, mean-based repulsion dilutes the repulsive cost by $1/N_{\text{bonds}}$, letting a single bond collapse almost without limit — caught during development when every tested starting point drove one bond straight to an artificial clip boundary with a suspiciously large negative energy. With per-bond repulsion (every bond gets its own local repulsive wall), three different starting geometries converge to distinct, physically reasonable conformations ($R^\star$ in the 3.4–5.3 Å range, no boundary artifacts), with energies $-0.171$ / $-0.153$ / $-0.052$ eV — suggesting genuinely different local minima depending on the starting geometry, not full global convergence at the epoch budget used (reported honestly, not oversold).

---

## Additional Investigation: Hunting Quantum Many-Body Scars

`scripts/quantum_scar_investigation/` contains a self-contained, honestly-reported investigation into whether a "quantum many-body scar" (the non-thermalizing phenomenon first observed in 2017 Rydberg-atom experiments) shows up in Dense Evolution's frustrated Ising simulations. Full writeup: [`report_indagine_scar.md`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/scripts/quantum_scar_investigation/report_indagine_scar.md) (Italian).

**Short version**: an initial-looking scar signature on a 4x4 frustrated TFIM grid did **not** survive rigorous verification (entanglement entropy, Trotter convergence, and a systematic 25-combination parameter scan) — it turned out to be the wrong observable (energy instead of entanglement entropy) plus a gauge-equivalence coincidence between sign patterns. The verification pipeline was then validated against the PXP model (Rydberg blockade), where scars are known to genuinely exist — confirmed via fidelity revivals and the characteristic "tower" of low-entanglement eigenstates in the exact spectrum. Using Dense Evolution's own `NoiseModel.apply_to_sv` (real stochastic Kraus channel, averaged over 30 quantum trajectories), the PXP scars turned out to be extremely fragile: **a 0.5-1% per-site depolarizing error rate destroys almost the entire revival signal.** Projecting the noisy state back onto the exact 13-state scar tower recovers ~31x of the lost revival amplitude — an idealized theoretical bound (not a realizable hardware protocol as-is) showing the protection target exists.

Open for anyone who wants to pick it up: translating the PXP dynamics into an actual circuit and testing revival + a physically realizable protection protocol (e.g. constraint-postselection instead of exact-eigenstate projection) on real quantum hardware.

---

## Technical Stack

| Component | Version / Detail |
|---|---|
| Simulator | Dense Evolution v8.1.21 |
| Backend | DenseSVSimulator (Statevector) |
| Precision | `complex128` (64-bit double) |
| Compilation | JAX XLA JIT static compilation |
| Parallelism | `run_parametric_batch_jit()` — up to 73,500 tracks/cycle |
| Gradient engine | Exact chain-rule Parameter-Shift Rule + finite-difference |
| Noise model | Stochastic Pauli-Z Kraus dephasing channel |
| Phonon model | Bose-Einstein / Debye |
| Bandstructure | Jordan-Wigner XY tight-binding, Harrison's law strain |
| Python deps | `jax`, `jaxlib`, `numpy`, `pandas`, `matplotlib` |

---
### 10. Automated CI Cross-Validation (Dense Evolution vs. PennyLane)

To guarantee the mathematical stability and absolute physical accuracy of the simulated quantum dynamics, the repository includes a strict continuous integration (CI) pipeline executed via GitHub Actions (`ci.yml`). 

The test suite (`test_pennylane_comparison.py`) establishes an automated cross-validation layer by mirroring the statevector computations on two completely independent software architectures:
- **Target Simulator:** Dense Evolution (v8.1.21) accelerated via JAX XLA.
- **Baseline Reference:** PennyLane.

The pipeline runs on every code splotch or pull request, evaluating the numerical consistency of the 1D Transverse Field Ising Model (TFIM) expectation values, variational gradients, and Bloch state rotations. By testing the outputs across both engines, the CI automatically flags floating-point drift or algebraic regressions exceeding machine-epsilon tolerances.

---
### 11. Zero-Dependency Analytical Validation Suite

To ensure absolute core-level stability without relying on third-party frameworks, the repository features a dedicated self-contained validation layer (`test_analytical.py`). This suite runs directly against exact mathematical identities and physics boundaries under machine-precision tolerances ($\le 10^{-10}$), keeping execution times strictly below 20 seconds on standard GitHub Actions CPU runners.

The suite enforces verification across five distinct physical and algorithmic benchmarks:

1. **Potential Energy Curve (PEC) Topography (`test_pec_shape`)**: Validates the qualitative Born-Oppenheimer energy landscape of molecular Silicon systems. It guarantees that the simulation resolves the correct three-region behavior: a steep repulsive wall at short range ($R = 1.4\text{ Å}, E > 0$), a stable binding well at intermediate distance ($R = 3.3\text{ Å}, E < 0$), and asymptotic stabilization near the dissociation limit ($R = 7.0\text{ Å}, |E| < 0.01\text{ eV}$).
2. **Bound-State Existence (`test_pec_minimum_is_negative`)**: Scans the well core ($R \in [3.0, 4.5]\text{ Å}$, empirically confirmed bound at every sampled point) and asserts $E < -0.01\text{ eV}$ at each one, proving a genuine stable ground state rather than merely finite output — tightened during the dense-evolution 8.1.21 audit, when this test was found asserting only `np.isfinite` despite its docstring's stronger claim.
3. **Exact Parameter-Shift Rule (`test_psr_exactness_ry_z`)**: Mathematically benchmarks the single-gate PSR primitive underlying the VQE gradient engine (`vqe_jax_grad.py`). By tracking an $RY(\theta)|0\rangle$ state followed by a $\langle Z \rangle$ measurement, it verifies that the computed gradient perfectly mirrors the exact analytical identity $\frac{dE}{d\theta} = -\sin(\theta)$. `tests/test_vqe_jax_gradient.py` extends this same exactness check to the full multi-gate, chain-rule PSR gradient used in production.
4. **Harrison's Hopping Law (`test_harrison_strain_ratio`)**: Verifies the bandstructure deformation engine under mechanical stress (`next_gen_silicon.py`). It enforces that the exact ratio of strained to unstrained tight-binding energies follows Harrison's solid-state scaling law, $t(\varepsilon) = \frac{t_0}{(1+\varepsilon)^2}$, at every non-trivial $k$-point across the Brillouin zone.
5. **Time-Reversal Dispersion Symmetry (`test_dispersion_time_reversal_symmetry`)**: Checks the underlying algebraic symmetry of the tight-binding Bloch states, ensuring that the dispersion relation satisfies the strict time-reversal constraint $E(k) \equiv E(-k)$ to isolate and prevent unphysical symmetry-breaking artifacts.

---

### 12. ZNE-Before-PSR: Correcting Each Gradient Term Before the Chain Rule, Not After

Every gradient in this repo is computed via the exact Parameter-Shift Rule (PSR): shift a single gate's own parameter by $\pm\pi/2$, held fixed against every other gate, and take $\frac{1}{2}[E(\theta+\tfrac{\pi}{2}) - E(\theta-\tfrac{\pi}{2})]$ (Section 6). `zne_stabilized_psr_gradient.py` asks what happens to that gradient under a real stochastic noise channel (`dense_evolution.registry.NoiseModel.apply_to_sv`, always applied post-hoc to an already-computed clean statevector — verified by reading the simulator's source in full, no noise anywhere in the unitary simulation path itself): does correcting each individual single-gate shifted evaluation with Zero-Noise Extrapolation — the same static 2-point Richardson formula already validated in Section 2, $E_{\text{zne}} = 2E(\lambda{=}1) - E(\lambda{=}2)$ — **before** combining them via the chain rule, stabilize the resulting gradient?

> **Correction (found during development):** an early version of this script shifted the shared *scalar* $\theta$ by $\pm\pi/2$ directly inside the circuit — exactly the mistake already documented and fixed in Section 6. A finite-difference test caught it immediately (a completely different gradient, wrong magnitude *and* sign region). Fixed by shifting each of the $2\times5=10$ individual gate parameters one at a time and recombining via the chain rule, exactly as Section 6 already does.

The honest finding is **not** what a naive "ZNE = more stable" intuition predicts, and is not uniform across $\theta$ either (40 trials, 200 shots, measured RMSE against the exact gradient):

| $\theta$ | exact | naive RMSE | ZNE-pre-PSR RMSE |
|---|---|---|---|
| 0.20 | +7.354 | 3.118 | 1.394 |
| 0.38 | +5.071 | 2.167 | 1.041 |
| 0.62 | −0.127 | 0.115 | 0.261 |
| 1.00 | −3.323 | 1.414 | 0.716 |

Away from a gradient zero-crossing ($\theta=0.20, 0.38, 1.00$), ZNE-pre-PSR cuts bias roughly in half to a third and — despite *increasing* the trial-to-trial standard deviation by roughly 2x (the textbook Richardson bias/variance tradeoff: $2E_1-E_2$ amplifies statistical noise in exchange for cancelling the leading systematic error) — nets a clearly lower RMSE, roughly 2–2.2x better. But at $\theta=0.62$, where the exact gradient is itself near zero, ZNE-pre-PSR is worse on **every** axis: there is very little systematic bias left to correct, so Richardson's variance amplification just adds noise. "ZNE stabilizes the gradient" is therefore a regime-dependent claim, not a universal property of ZNE-pre-PSR.

[![ZNE-Before-PSR Gradient Stabilization Study](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/zne_stabilized_psr_gradient.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/zne_stabilized_psr_gradient.png)

---

### 13. Adaptive ZNE-Before-PSR via Predictive Healing — An Honest Negative Result

Section 12's static correction actively *hurts* near a gradient zero-crossing ($\theta=0.62$). `zne_adaptive_psr_gradient.py` asks whether **attenuating** the correction when per-shift confidence is low — via `dense_evolution.healing.calculate_delta_preemp`, previously only prototyped against a synthetic "coherence" proxy in scratch code, here fed a **real, measured** standard error of the mean (SEM) of each shifted evaluation — can recover $\theta=0.62$ without giving up Section 12's wins elsewhere.

**It does not.** No `(target_sigma_ideal, k_sensitivity)` calibration explored Pareto-dominates; the shipped default is a documented compromise point, not a solution. At full budget (40 trials, 200 shots), the adaptive correction sits strictly **between** naive and static at every tested $\theta$ and never wins outright anywhere:

| $\theta$ | naive RMSE | static RMSE | adaptive RMSE |
|---|---|---|---|
| 0.20 | 3.131 | 1.425 | 2.501 |
| 0.38 | 2.180 | 0.961 | 2.170 |
| 0.62 | 0.106 | 0.248 | 0.175 |
| 1.00 | 1.406 | 0.684 | 1.139 |

**Root cause:** the measured SEM sits in the same narrow band ($\sim 0.016$–$0.025$) regardless of $\theta$ — it's driven by shot count and the noise probability, not by proximity to a gradient zero-crossing, so it carries no information about the quantity that actually determines whether Richardson correction helps or hurts (the size of the systematic bias being corrected, relative to noise). A single scalar SEM threshold can't distinguish "large bias, correction pays off" from "near-zero bias, correction just adds noise." A more promising unexplored direction: a confidence signal built from the correction term's own signal-to-noise ratio ($|E_1-E_2|$ relative to their combined SEM), rather than the noise of a single measurement in isolation.

This negative result is reported in full because, done rigorously, it carries the same scientific value as a positive one: it saves the next person from re-walking this exact path.

[![Adaptive ZNE-Before-PSR: Negative Result](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/zne_adaptive_psr_gradient.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.2.0/zne_adaptive_psr_gradient.png)

---

### 14. A Second Adaptive-ZNE Attempt via the Correction Term's Own SNR — Hypothesis Rejected

Section 13's SEM-based confidence signal failed because SEM doesn't correlate with proximity to a gradient zero-crossing. `zne_snr_adaptive_psr_gradient.py` tries the more principled signal flagged there: the correction term's own signal-to-noise ratio, $|E_1-E_2|$ relative to their combined standard error, reusing `calculate_delta_preemp` honestly via a clamp so confidence correctly *grows* with SNR (the opposite direction from Section 13's SEM case).

**Hypothesis rejected again, with a more interesting failure mode.** At the calibrated default ($\text{SNR}_{\text{target}}=3.0$, 40 trials, 200 shots):

| $\theta$ | naive RMSE | static RMSE | SNR-adaptive RMSE |
|---|---|---|---|
| 0.20 | 3.151 | 1.380 | 1.421 |
| 0.38 | 2.178 | 0.920 | **0.862 (beats static)** |
| 0.62 | 0.107 | 0.263 | **0.307 (regresses)** |
| 1.00 | 1.427 | 0.673 | 0.665 |

At $\theta=0.38$ the correction beats the static one outright — something Section 13's SEM-based model never achieved anywhere. But at $\theta=0.62$, the exact case this was meant to fix, it makes things *worse* instead of better. Raising $\text{SNR}_{\text{target}}$ to 6.0 does not selectively fix that case either — it makes every $\theta$ worse, disproving the "a higher threshold isolates the zero-crossing" hypothesis: the algorithm is hyper-sensitive to this one calibration knob, not robust to it.

**Root cause:** SNR here is computed from a difference between two *noise scales* at the same gate-shift ($E_1$ at $p$, $E_2$ at $2p$). The PSR gradient's zero-crossing comes from a near-cancellation between two *gate-shifts* at the same noise scale ($E_{+\pi/2} - E_{-\pi/2}$). These are independent quantities with no causal link — and $|E_1-E_2|$ is itself a biased estimator with a noise floor (its expectation doesn't vanish even when the true difference does), which is why SNR measured a misleadingly high $\sim 3.3$–$6.6$ at *every* tested $\theta$, including 0.62. Development on this direction is halted; future adaptive-ZNE work returns to refining Section 13's SEM-based model, which specifically improved on static at the critical $\theta=0.62$ case, something this attempt did not reproduce.

> **Correction (reproducibility bug found during this study):** Python's built-in `hash()` is not stable across process invocations for tuples containing strings (hash randomization on by default) — the `hash((tag, theta, trial)) % 2**32` seeding pattern used for trial data in this study and in Sections 12–13 silently drew different noise realizations on every run despite looking deterministic. A test asserting a specific RMSE comparison passed once and failed on an immediate re-run as a direct result. Fixed with `zlib.crc32`-based stable seeding across all three ZNE-PSR studies; verified identical results across two separate process invocations after the fix. None of the reported findings above changed as a result — only the seeding mechanism did.

[![SNR-Adaptive ZNE-Before-PSR: Hypothesis Rejected](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.3.0/zne_snr_adaptive_psr_gradient.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.3.0/zne_snr_adaptive_psr_gradient.png)

---

### 15. Sophia Reflection: A Real ZNE Trajectory, Not an Invented One

`scripts/sophia_reflection.py` runs the density-matrix extension of ZNE (`dense_evolution.mitigation.zne_density_matrix`/`uhlmann_fidelity` — the matrix-valued form, distinct from the scalar Richardson protocol Section 2 above hand-rolls) across a 16-point depolarizing-noise sweep on a 2-qubit Bell state. All 16 points improve fidelity (mean delta +0.182, range [+0.025, +0.366]); the gain isn't monotonic in noise strength — it grows through the low-to-mid range, peaks around $p\approx0.21$, then tapers as extrapolation itself gets less reliable at high noise, consistent with the Richardson-noise-amplification finding already documented in dense-evolution's own changelog.

This one has a second half. It's a direct answer to an August 2025 personal notebook that modeled subjective experience as invented Hilbert-space vectors and fed them to an LLM ("Sophia") for reflection — with real measured data instead of synthesized states, and a reflection an LLM (Claude) actually wrote after seeing this script's real output, not before. See [`SOPHIA_REFLECTION.md`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/SOPHIA_REFLECTION.md) for the full trajectory and the reflection itself.

---

### 16. Channel-Order Non-Commutativity: An Honest Positive Result from the Same Archive

The same August-September 2025 archive that produced Sophia's origin also built up a numbered "rule book" of ~120 empirical claims about noise behavior on small quantum circuits (a "Teoria della Riorganizzazione Entropica Coerente", TREC). Batch-testing 8 of its explicit, unambiguous rules against a fair same-size GHZ baseline (same noise channels, same intensities, same statistics) found **zero** rules showing a real, un-confounded difference from generic noise physics — every claimed "structural resilience" reduced to either ordinary noise physics or a methodological artifact (e.g. a baseline that happened to include the amplitude-damping ground state itself, discovered and corrected during this same investigation).

One claim was different: that the **order** of applying two distinct noise channels leaves a measurable fingerprint — the archive's own "Attrattori Crono-Topologici" idea, claimed as a blanket rule across Regole 100-109 for *any* channel pair. `scripts/channel_order_noncommutativity.py` tests it directly on the Regola 16 circuit (`GHZ(3q) -> X(Q0) -> Z(Q1) -> X(Q2) -> CNOT(Q0,Q2)`) on two different channel pairs, same intensities (p=0.3 each), 8192 Monte Carlo trajectories per order:

- **dephasing → amplitude damping vs. the reverse: real, not noise.** State `|000⟩` lands at 17.4% under dephasing→AD vs. 13.3% under AD→dephasing. Jensen-Shannon divergence 0.00174 against a permutation null topping out at 0.00054 (p=0.0033).
- **dephasing → bit-flip vs. the reverse: no signal.** Jensen-Shannon divergence 0.00027, well inside the permutation null (p=0.24).

The dividing line isn't "any two channels" — it's **Pauli vs. non-Pauli**. Phaseflip and bit-flip are both Pauli channels, and Pauli channels commute with each other as superoperators (they're simultaneously diagonal in the Pauli-transfer-matrix basis), so their order genuinely doesn't matter. Amplitude damping is non-unital — it has a preferred fixed point, `|0⟩` — and isn't a Pauli channel, which is exactly why composing it with a Pauli channel *is* order-dependent. So Regole 100-109's blanket "order always matters" is an overgeneralization from one true instance: the precise, verified rule is that channel order matters **iff at least one of the two channels is non-Pauli**. Quantum channels genuinely don't commute in general, but not universally either — this is the one claim from the whole archive that survived contact with a fair, statistically rigorous test, refined into its correct, narrower form.

---

### 17. Loschmidt Echo: A Real Time-Reversal Test, Not a Static Noise Channel

In a closed quantum system, evolving forward in time under a chaotic unitary $U$ and then backward under $U^{-1}$ reconstructs the initial state exactly ($F=1.0$). Coupling to an environment along the way breaks that time-reversal symmetry — the **Loschmidt echo** fidelity

$$F = \left|\langle\psi_0|\,U^{-1}\,\mathcal{N}\,U\,|\psi_0\rangle\right|^2$$

decays below 1 as noise $\mathcal{N}$ is injected mid-evolution. A draft proposed for the main Dense-Evolution README's benchmarks section ("Eco di Loschmidt", authored by Gemini) claimed ZNE could recover this fidelity — but its own `run_noisy_eco()` built `forward_ops`/`backward_ops`/`h_fields` and never applied any of them; it only ever called `NoiseModel.apply_to_sv` on one static state, so $U$ and $U^{-1}$ were never actually run. There was no time-reversal, and nothing was being echoed — the reported "recovery" was an artifact of that dead code.

**The real model**, one Trotter step of a "kicked Ising" chain:

$$U_{\text{step}} = \left(\prod_{i} \text{CX}_{i,i+1}\right)\left(\prod_i RX_i(\pi/4)\right)\left(\prod_i RZ_i(h_i)\right), \qquad h_i \sim \mathcal{U}(-2, 2)$$

— a fixed transverse "kick" ($RX$), a fresh random longitudinal disorder field per step ($RZ$), and nearest-neighbor coupling ($CX$), the standard toy model for quantum chaos. `scripts/loschmidt_echo_zne.py` runs this circuit forward for 4 steps, then its **exact** inverse backward ($RZ(\theta)^{-1}=RZ(-\theta)$, $RX(\theta)^{-1}=RX(-\theta)$, $CX^{-1}=CX$, gates in reverse order), with an amplitude-damping channel injected between *every single layer* — forward and backward — and reinjected into the simulator via `set_initial_state` so each subsequent layer acts on the actually-noisy state, not a copy that gets thrown away. A noiseless self-check ($p=0$ must return fidelity exactly $1.0$) gates the noisy results before they're trusted — a broken inverse would otherwise still look like a plausible echo.

| Quantity | Value |
|---|---|
| Qubits / Trotter steps | 4 / 4 |
| Kick angle ($RX$) | $\pi/4$ |
| Disorder field | $h_i \sim \mathcal{U}(-2, 2)$ rad, resampled every step, per qubit |
| Noise channel | amplitude damping, injected between every layer |
| ZNE noise scales | $1.0\lambda,\ 1.5\lambda,\ 2.0\lambda$ ($\lambda = 0.015$) |
| Monte Carlo trajectories per scale | 300 |
| **Noiseless self-check fidelity** | **1.000000000000** (exact) |
| Raw noisy return fidelity ($\lambda=1.0$) | **0.7769** |
| ZNE-corrected return fidelity | **0.9965** |
| Net fidelity gain | **+0.2195** |

[![Loschmidt echo: raw vs. ZNE-corrected return fidelity](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/loschmidt_echo_zne.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/loschmidt_echo_zne.png)

---

### 18. Topological Mott Isolator: A Real Optimization Loop, Validated Against Exact Diagonalization

A related draft ("New Silicon" Haldane-Hubbard material design, same origin as Section 17) built a Topological Mott Isolator Hamiltonian and an RY-CX-RZ ansatz via `de.circuit_to_energy_fn`, then evaluated the energy at a single fixed random `theta` and called the result "the ideal state" — no optimization ever ran, so the reported energies had no relationship to the material's actual ground state.

**The Hamiltonian**, built directly on computational basis states (site $A$ = qubits $0,1$; site $B$ = qubits $2,3$):

$$H(U) = \frac{U}{2}\sum_{s\in\{A,B\}} n_s(n_s-1)\; -\; t_1\!\!\sum_{\langle i,j\rangle}\!\!\left(c_i^\dagger c_j + \text{h.c.}\right)\; -\; t_2\!\!\sum_{\langle\langle i,j\rangle\rangle}\!\!\left(e^{i\phi} c_i^\dagger c_j + \text{h.c.}\right)$$

— on-site Mott repulsion $U$, real nearest-neighbor hopping $t_1$, and a complex next-nearest-neighbor "Haldane phase" hopping $t_2 e^{i\phi}$. `build_tmi_hamiltonian` constructs this by touching each unordered basis-state pair exactly once and setting both conjugate entries together, so it's Hermitian by construction, not by luck (`test_hamiltonian_is_hermitian`).

**The fix**: `scripts/vqe_tmi_material_design.py` runs a real VQE loop — exact JAX autodiff (`jax.value_and_grad` straight through `circuit_to_energy_fn`, no finite differences, no dead parameters) drives Adam over an 8-parameter RY-CX-RZ ansatz, with 8 random restarts per Mott-repulsion value $U$ to avoid a bad local minimum, all batched into one `jax.vmap`'d update per epoch. Every result is checked against an independent reference: direct dense diagonalization of the same Hamiltonian, which fixes the true ground energy and the variational bound $E_{\text{VQE}} \geq E_{\text{exact}}$ that any correct implementation must respect — not assumed, asserted.

| $U$ (eV) | Exact ground state | VQE-optimized | Fixed random $\theta$ (original draft) | Gap |
|---|---|---|---|---|
| 0.00 | −3.3451 | −3.3451 | −0.0424 | **+0.0000** |
| 0.55 | −3.1231 | −3.0984 | −1.0628 | +0.0248 |
| 1.64 | −2.8580 | −2.7145 | +1.5875 | +0.1435 |
| 3.27 | −2.6608 | −2.3180 | +0.4666 | +0.3428 |
| 4.91 | −2.5581 | −2.0602 | +4.7166 | +0.4979 |
| 6.00 | −2.5136 | −2.0000 | +2.4801 | **+0.5136** |

The optimizer respects the bound at every single point in the full 12-point sweep (no violations) and reaches the exact ground state at $U=0$. The gap grows with $U$ — an honest ansatz-expressivity limit in the strongly-correlated regime, not an optimizer bug: multi-start restarts converge to the *same* plateau there rather than scattering, which is what a genuine expressivity ceiling looks like, not under-training. Against the original draft's unoptimized baseline — a single random $\theta$, which drifts as high as $+4.72$ eV, nowhere near any ground state — the fix is a difference in kind, not degree.

[![Topological Mott Isolator: real optimization vs. exact diagonalization](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/vqe_tmi_material_design.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.4.0/vqe_tmi_material_design.png)

---

### 19. From a Made-Up U to a Real One: DFT and Dielectric Screening for Actual GaAs

Section 18's $U$ sweep uses arbitrary units — it explores a hypothetical design space, not a specific material. A companion draft tried to ground $t$/$U$ in real chemistry via a PySCF DFT calculation on a GaAs dimer (PBE/STO-3G, the real Ga-As zinc-blende nearest-neighbor bond length of 2.44 Å), but that SCF never converged, and the draft extracted HOMO/LUMO integrals from the non-converged density anyway — numbers with no defined physical meaning.

**Getting to a trustworthy SCF took two separate real problems, not one:**

| Stage | Method | SCF energy (Ha) | Converged? | Stable? |
|---|---|---|---|---|
| Original draft | plain CDIIS, 200 cycles | −4111.4032 | ✗ | — |
| This fix, stage 1 | level-shift (0.3) + ADIIS pre-step | −4111.1869 | ✗ | — |
| This fix, stage 2 (first attempt) | Newton-Raphson (SOSCF), seeded from stage 1 | −4111.9696 | ✓ | ✗ (genuine saddle point, Hessian eigenvalues `[-2.37, -2.37, -2.30]`) |
| This fix, stage 3 (restart) | Newton-Raphson, reseeded from `mf.stability()`'s lower-energy orbitals | −4111.9696 | ✓ | ✓ (Hessian eigenvalues `[~0, +0.0108, +0.0359]`) |

The final stable point was reproduced independently from a second, differently-seeded optimization path to 6 significant figures — real convergence to the same physical state looks exactly like that. Along the way, a cruder diagnostic (comparing raw occupied/virtual orbital energies) flagged a "HOMO above LUMO" ordering on this same converged, stable solution; that turned out to be a red herring specific to Kohn-Sham DFT, where virtual orbitals see the same $N$-electron potential as occupied ones rather than an $N{+}1$-electron one, so they aren't required to sit above the HOMO the way Hartree-Fock intuition expects. The rigorous test is the stability Hessian, which passed.

**From a converged calculation to a real material parameter:**

| Parameter | Value | Source |
|---|---|---|
| $t$ (hopping) | **7.9170 eV** | DFT, real 2.44 Å Ga-As bond |
| $U$ (bare) | 38.3847 eV | DFT, gas-phase / unscreened |
| GaAs static dielectric constant $\varepsilon$ | 12.9 | Sze, literature value |
| $U$ (screened) | **2.9756 eV** | $U_{\text{bare}} / \varepsilon$ |
| $U/t$ (screened) | **0.376** | weakly-correlated regime |

GaAs isn't conventionally modeled as a Hubbard material in the first place, so there's no literature $U$ to check the bare DFT value against directly — but a bare two-atom-in-vacuum Coulomb integral is expected to badly overestimate a solid's actual on-site repulsion, which the surrounding crystal's dielectric response screens. Dividing by GaAs's real static dielectric constant lands $U/t = 0.376$: deep in the weakly-correlated regime. That's itself a meaningful cross-check — GaAs is a conventional band semiconductor, not a Mott insulator, and the physically-corrected parameter lands exactly where real materials physics says it should, while the naive unscreened value ($U/t=4.85$) would have implied, wrongly, a strongly-correlated material.

`scripts/vqe_tmi_material_design.py`'s `run_real_gaas_point()` runs the same VQE-vs-diagonalization pipeline from Section 18 at both the bare and screened points (right panel of the plot above). The variational bound holds at both (gap `+3.9060` bare, `+0.1028` screened) — the larger gap at the bare point simply reflects that a much larger $U$ pushes further outside this fixed ansatz's expressivity, the same honest limitation Section 18 already documents, not a new problem.

---

## Reproducing the Results

```bash
# Clone and install
git clone https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests.git
cd Dense-Evolution-Ising-Tests
pip install -r requirements-ci.txt

# Run the test suite (fast, ~2 minutes, no CSV/PNG needed)
pytest tests/ -v

# Or run the full experiments -- each creates data/*.csv and/or images/*.png,
# safe to run from the repo root regardless of your current directory:
python scripts/scan_ising.py
python scripts/plot_ising.py
python scripts/zne_mitigation.py
python scripts/vqe_gradient.py
python scripts/vqe_jax_grad.py
python scripts/quantum_defect_scanner.py
python scripts/next_gen_silicon.py
python scripts/manufacturing_thermodynamics.py
python scripts/vqe_silicon_molecular.py
python scripts/vqe_silicon_molecular_optimized.py
python scripts/vqe_silicon_molecular_optimized_per_bond.py
python scripts/vqe_extreme_geometries.py
python scripts/vqe_extreme_geometries_deep.py
python scripts/zne_stabilized_psr_gradient.py
python scripts/zne_adaptive_psr_gradient.py
python scripts/zne_snr_adaptive_psr_gradient.py
python scripts/sophia_reflection.py
python scripts/channel_order_noncommutativity.py
python scripts/loschmidt_echo_zne.py
python scripts/vqe_tmi_material_design.py
```

`data/` and `images/` are gitignored -- they exist only after you run something, so it's always unambiguous whether what you're looking at is fresh. Pre-made results are on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) instead.

> **Hardware note:** All benchmarks were executed on CPU. The JAX XLA engine will automatically utilize GPU acceleration if available via `use_gpu=True` in the simulator constructor.

---

## Output Datasets

All produced under `data/` when you run the corresponding script (see [Repository Layout](#repository-layout)); also downloadable from the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) without running anything.

| CSV File | Description | Rows |
|---|---|---|
| `transizione_fase_ising.csv` | TFIM order parameter vs transverse field g | 3,500 |
| `dati_mitigazione_zne.csv` | ZNE ideal / noisy / mitigated energies vs k | 25 |
| `vqe_gradient_landscape.csv` | VQE energy and finite-diff gradient vs θ | 3,500 |
| `vqe_jax_gradient.csv` | VQE energy and PSR gradient vs θ (JAX batch) | 3,500 |
| `mappa_difetti_silicio.csv` | Residual qubit coherence vs defect node position | 12 |
| `bande_nuovo_silicio.csv` | Strained Si valence/conduction bands vs k | 3,500 |
| `validazione_fabbricazione_silicio.csv` | Phonon occupancy and hopping energy vs temperature | 3,500 |
| `vqe_molecola_silicio.csv` | Born-Oppenheimer PEC vs interatomic distance R (fixed θ=0.38) | 3,500 |
| `vqe_molecola_silicio_ottimizzata.csv` | Adam-optimized PEC: shared θ*(R), E*(R), final gradient | 200 |
| `vqe_molecola_silicio_ottimizzata_per_legame.csv` | Adam-optimized PEC: 5 independent θ*(R) per bond, E*(R) | 200 |
| `vqe_extreme_geometries.csv` | Rigid vs. per-bond adaptive energy and deficit_fraction across 6 extreme/irregular chain geometries | 6 |
| `vqe_extreme_geometries_deep.csv` | Same benchmark at 12 parameters (7 qubits / 6 bonds) | 6 |
| `vqe_extreme_geometries_deep_conformazioni.csv` | Minimum-energy conformational search: R*, θ*, E_min from 3 starting geometries | 3 |
| `zne_stabilized_psr_gradient.csv` | Naive vs. ZNE-pre-PSR gradient bias/std/RMSE vs. θ | 4 |
| `zne_adaptive_psr_gradient.csv` | Naive vs. static vs. adaptive ZNE-pre-PSR gradient bias/std/RMSE vs. θ | 4 |
| `zne_snr_adaptive_psr_gradient.csv` | Naive vs. static vs. SNR-adaptive ZNE-pre-PSR gradient RMSE vs. θ | 4 |
| `sophia_reflection.csv` | Real density-matrix ZNE fidelity trajectory (raw/corrected/delta) vs. depolarizing noise probability | 16 |
| `channel_order_noncommutativity.csv` | Regola 16 outcome distribution under dephasing→AD vs. AD→dephasing, per basis state | 8 |
| `loschmidt_echo_zne.csv` | Kicked-Ising forward/backward echo: raw vs. ZNE-corrected return fidelity, net gain | 1 |
| `vqe_tmi_material_design.csv` | Exact ground energy vs. VQE-optimized vs. unoptimized random-theta baseline, per Mott repulsion U | 12 |
| `vqe_tmi_material_design_gaas_real.csv` | Same, at the real GaAs point: DFT-derived t1, bare vs. dielectrically-screened U | 2 |

---

## License

MIT License — © 2026 Salvatore Pennacchio (tatopenn-cell)
This repository depends on Dense Evolution, licensed under Business Source License 1.1. See https://github.com/tatopenn-cell/Dense-Evolution for license terms.

