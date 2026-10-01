"""
VQE / QM-MM telemetry: real JAX-autodiff VQE plus the synthetic mock
fallback used when a circuit has no parametric gates.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import hashlib
from typing import Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import dense_evolution as de

QM_MM_HEAVY_QUBIT_THRESHOLD = 12


class QMMMForceEngine:
    """Hellmann-Feynman QM/MM force engine via JAX autodiff. Adapted from
    dash.py:1455 -- that version's generate_pauli_expectation_mpo docstring
    admitted it was "Placeholder tracciabile... In produzione si
    interfaccia con le stringhe generate da Jordan-Wigner". The mean-field
    external-potential approximation here (single_orbital_v below) is
    still a real simplification, not full electronic structure -- but as
    of dashboard_core.hamiltonians.build_molecular_hamiltonian, h_pq_core
    is now a genuine Jordan-Wigner-mapped molecular Hamiltonian when a
    catalog molecule is selected, not a fake one, and the classical
    positions/charges it's perturbed by (see _run_vqe_telemetry_body) are
    that same molecule's real equilibrium geometry, not a fixed stand-in."""

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


def run_vqe_telemetry(sim, parser, qasm_text, circuit_name, n_qubits, use_float32,
                       epochs, lr, beta1, beta2, seed, hamiltonian_values=None,
                       on_epoch=None, classical_geometry=None) -> pd.DataFrame:
    """Adapted from ottimizza_vqe (dash.py:3194, canonical/later definition).

    Runs real JAX-autodiff VQE (via QMMMForceEngine Hellmann-Feynman forces) if the
    circuit has parametric gates, else falls back to _run_vqe_mock_simulation — this
    branch is unchanged from the original.

    `hamiltonian_values`, if given, is either a flat array of length 2**n_qubits
    (a diagonal Hamiltonian -- a toy model from LIBRERIA_HAMILTONIANE, or whatever a
    user pastes into the Custom JSON textarea) or a dense (2**n_qubits, 2**n_qubits)
    Hermitian matrix (a real molecular Hamiltonian from
    dashboard_core.hamiltonians.get_molecular_hamiltonian_matrix -- Jordan-Wigner maps
    electronic structure onto real X/Y/Z terms, so this is generally NOT diagonal).
    If not given, a random diagonal spectrum is used, replacing the original's
    globals()-based custom-Hamiltonian widget lookup.

    `classical_geometry`, if given, is a dict with 'positions' ((n_atoms, 3) Å),
    'charges' ((n_atoms,)) and optionally 'orbital_centers' -- the REAL geometry of
    whatever molecule hamiltonian_values came from, fed to QMMMForceEngine instead of
    a fixed placeholder. When None (a toy model or a Custom JSON Hamiltonian with no
    known geometry is selected), QM/MM force computation is skipped entirely rather
    than perturbed by a geometry that doesn't actually correspond to anything real —
    engine stays None, norma_forze_mm stays 0.0, same as any other QMMMForceEngine
    failure already handled below.

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
            classical_geometry,
        )
    finally:
        # `sim` was built under a precision fixed at its own creation (run_simulation);
        # this call reuses that same sim, so jax_enable_x64 must match for its whole
        # duration, not whatever was left behind by unrelated code in between.
        jax.config.update('jax_enable_x64', previous_x64_state)


def _run_vqe_telemetry_body(sim, parser, qasm_text, circuit_name, n_qubits, use_float32,
                             epochs, lr, beta1, beta2, seed, hamiltonian_values=None,
                             on_epoch=None, classical_geometry=None) -> pd.DataFrame:
    circ_obj = parser.parse(qasm_text)
    energy_fn, n_params = de.circuit_to_energy_fn(circ_obj, n_qubits)

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
    h_complex_dtype = jnp.complex64 if use_float32 else jnp.complex128

    ham_array = np.asarray(hamiltonian_values) if hamiltonian_values is not None else None
    if ham_array is not None and ham_array.ndim == 2 and ham_array.shape == (2 ** n_qubits, 2 ** n_qubits):
        # A real molecular Hamiltonian (dashboard_core.hamiltonians.
        # get_molecular_hamiltonian_matrix) -- Jordan-Wigner maps onto X/Y/Z
        # terms, so this is generally dense, not diagonal.
        sim.H_matrix = jnp.array(ham_array, dtype=h_complex_dtype)
    elif ham_array is not None and ham_array.ndim == 1 and len(ham_array) == 2 ** n_qubits:
        valori_energetici = ham_array.astype(classical_dtype)
        sim.H_matrix = jnp.diag(jnp.array(valori_energetici, dtype=classical_dtype)).astype(h_complex_dtype)
    else:
        valori_energetici = np.sort(np.random.uniform(-2.5, 2.5, 2 ** n_qubits)).astype(classical_dtype)
        sim.H_matrix = jnp.diag(jnp.array(valori_energetici, dtype=classical_dtype)).astype(h_complex_dtype)

    history = []
    stato_zero_dtype = jnp.complex64 if use_float32 else jnp.complex128
    stato_zero = jnp.zeros(2 ** n_qubits, dtype=stato_zero_dtype).at[0].set(1.0)

    if classical_geometry is not None:
        # Real geometry (dashboard_core.hamiltonians.MOLECULE_CATALOG),
        # matching whatever molecule hamiltonian_values above came from --
        # not a fixed stand-in used for every circuit regardless of what
        # it represents.
        classical_positions = jnp.array(classical_geometry['positions'], dtype=classical_dtype)
        classical_charges = jnp.array(classical_geometry['charges'], dtype=classical_dtype)
        orbital_centers = jnp.array(
            classical_geometry.get('orbital_centers', classical_geometry['positions']),
            dtype=classical_dtype,
        )
    else:
        # No known real geometry for the selected Hamiltonian (a toy model,
        # or a Custom JSON array with no associated molecule) -- skip QM/MM
        # force computation entirely below rather than perturb the energy
        # with a geometry that doesn't correspond to anything real.
        classical_positions = classical_charges = orbital_centers = None

    # Built once, reused every epoch — only theta changes, so this traces/
    # JIT-compiles a single time instead of rebuilding the circuit (and
    # calling the trace-severing risolvi_qasm) from scratch each epoch.
    energy_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))

    for epoch in range(epochs):
        (energia_jax, sv), grad_jax = energy_and_grad(
            jnp.asarray(theta, dtype=jnp.float64), sim.H_matrix, stato_zero
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
        if engine is not None and classical_positions is not None:
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
