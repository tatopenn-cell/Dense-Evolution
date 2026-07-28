"""
Molecular-dynamics-style telemetry.

Two modes, chosen automatically by run_md_telemetry() based on what the
caller passes in:

  REAL mode  -- a valid Hamiltonian (from LIBRERIA_HAMILTONIANE or a custom
                one, diagonal, length 2**n_qubits -- see hamiltonians.py)
                and the circuit's current statevector are both given, and
                their sizes agree. The state is evolved in real time under
                that Hamiltonian and a physically-grounded thermal-noise
                channel; every column is a real, computed quantity.

  MOCK mode  -- no compatible Hamiltonian/statevector given. Falls back to
                the original synthetic generator (adapted verbatim from
                run_md_simulation_dummy, dash.py:1152 -- explicitly a
                placeholder in the source since it was first written, never
                real MD/quantum-chemistry data).

Both modes return the same (df_md, corr_matrix) shape the rest of the
dashboard already expects. Which mode actually ran is recorded in
df_md.attrs['is_real'] / df_md.attrs['note'] (pandas' own metadata slot --
survives on the DataFrame itself, does not require changing every call
site's return-value unpacking) so the UI can label the result honestly
instead of presenting a mock run as if it were real.
"""

import numpy as np
import pandas as pd


def run_md_telemetry(md_steps, md_temp, hamiltonian_values=None, sv=None, n_qubits=None, seed=None):
    is_real, mismatch_note = _check_real_mode_inputs(hamiltonian_values, sv, n_qubits)
    if is_real:
        df_md, corr_matrix = _run_md_telemetry_real(md_steps, md_temp, hamiltonian_values, sv, int(n_qubits), seed)
        df_md.attrs['is_real'] = True
        df_md.attrs['note'] = 'Dinamica reale: evoluzione temporale sotto l\'Hamiltoniana selezionata.'
    else:
        df_md, corr_matrix = _run_md_telemetry_mock(md_steps, md_temp)
        df_md.attrs['is_real'] = False
        df_md.attrs['note'] = mismatch_note
    return df_md, corr_matrix


def _check_real_mode_inputs(hamiltonian_values, sv, n_qubits):
    if hamiltonian_values is None or sv is None or n_qubits is None:
        return False, 'Dati dimostrativi -- nessuna Hamiltoniana reale attiva.'
    expected_dim = 2 ** int(n_qubits)
    if len(hamiltonian_values) != expected_dim or len(sv) != expected_dim:
        return False, (
            f'Dati dimostrativi -- l\'Hamiltoniana selezionata (dim={len(hamiltonian_values)}) '
            f'non e\' compatibile con questo circuito (dim richiesta={expected_dim}).'
        )
    return True, None


def _run_md_telemetry_mock(md_steps, md_temp):
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


def _run_md_telemetry_real(md_steps, md_temp, hamiltonian_values, sv, n_qubits, seed):
    """Real-time evolution of the actual circuit state under the actual
    (diagonal) Hamiltonian selected by the user, plus a physically-grounded
    thermal-noise channel reusing dense_evolution's own NoiseModel (not a
    reinvented noise formula).

    Every LIBRERIA_HAMILTONIANE entry is a diagonal Hamiltonian (a flat list
    of 2**n_qubits eigenvalues, see hamiltonians.py) -- so time evolution
    under it is *exact*, not a Trotter approximation: each basis-state
    amplitude just picks up a phase exp(-i * eigenvalue * dt) per step.

    DT is a fixed, unitless step size (these Hamiltonians are unitless
    benchmark energies, not calibrated to a real physical time unit -- there
    is no "correct" DT to derive, this one is chosen so md_steps=20-500
    sweeps a visually reasonable range of phase evolution, nothing more).

    "Temperature" drives the probability of dense_evolution's own
    NoiseModel.apply_to_sv (depolarizing channel) applied once per step --
    real decoherence from real, already-tested code, not a fabricated
    thermal-noise curve.

    Purity is computed exactly (not approximated) from an ensemble-averaged
    density matrix: ENSEMBLE independent noisy realizations of the same
    step are drawn, their outer products averaged into rho_avg, and
    Purita_Stato = Tr(rho_avg^2). This is exact given rho_avg; the only
    approximation is that ENSEMBLE=8 samples estimate the true
    noise-channel-averaged state (more samples would converge tighter).
    Every LIBRERIA_HAMILTONIANE entry is <=6 qubits (dim<=64), so
    materializing dim x dim density matrices per step is cheap.
    """
    import jax.numpy as jnp
    import dense_evolution as de

    DT = 0.15
    ENSEMBLE = 8
    dim = 2 ** n_qubits

    h_diag = np.asarray(hamiltonian_values, dtype=np.float64)
    psi = np.asarray(sv, dtype=np.complex128).copy()
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi = psi / norm

    temp_factor = float(np.clip(md_temp / 300.0 if md_temp > 0 else 0.1, 0.1, 2.0))
    # 300K (temp_factor=1.0) -> p_thermal ~= 0.142; scales linearly with temp_factor
    # in [0.1, 2.0] -> p_thermal in [0.0, 0.30]. An explicit design choice
    # (no calibrated real-world mapping exists here), not a physical constant.
    p_thermal = float(np.clip((temp_factor - 0.1) / 1.9 * 0.30, 0.0, 0.30))

    rng = np.random.default_rng(seed if seed is not None else 0)

    data = {
        "Step": [], "Energia_VQE_Ha": [], "Entropia_von_Neumann_Bit": [],
        "Purita_Stato": [], "Gradiente_Operatore": [], "Fattore_Rumore_Termico": [],
    }

    energy_prev = None
    for step in range(md_steps):
        phase = np.exp(-1j * h_diag * DT)
        psi = psi * phase
        psi = psi / np.linalg.norm(psi)

        if p_thermal > 0:
            realizations = []
            for _ in range(ENSEMBLE):
                sub_seed = int(rng.integers(0, 2**31 - 1))
                noisy = de.NoiseModel.apply_to_sv(
                    sv=np.array(psi, copy=True), n=n_qubits, model='depolarizing',
                    p=p_thermal, rng=np.random.default_rng(sub_seed),
                )
                realizations.append(np.asarray(noisy, dtype=np.complex128))
        else:
            realizations = [psi]

        rho_avg = sum(np.outer(v, v.conj()) for v in realizations) / len(realizations)
        prob_t = np.real(np.diag(rho_avg))
        purity_t = float(np.real(np.trace(rho_avg @ rho_avg)))

        # True von Neumann entropy of rho_avg (matches the column's name,
        # unlike the Shannon-entropy-of-the-diagonal used elsewhere in the
        # dashboard for "entropy") -- cheap here since rho_avg is already
        # materialized (dim<=64) and Hermitian, so eigvalsh is exact and fast.
        eigvals = np.linalg.eigvalsh(rho_avg)
        eig_safe = eigvals[eigvals > 1e-12]
        entropy_t = float(-np.sum(eig_safe * np.log2(eig_safe))) if len(eig_safe) else 0.0

        energy_t = float(np.sum(prob_t * h_diag))
        drift_t = 0.0 if energy_prev is None else abs(energy_t - energy_prev)
        energy_prev = energy_t

        data["Step"].append(step)
        data["Energia_VQE_Ha"].append(energy_t)
        data["Entropia_von_Neumann_Bit"].append(entropy_t)
        data["Purita_Stato"].append(purity_t)
        data["Gradiente_Operatore"].append(drift_t)
        data["Fattore_Rumore_Termico"].append(p_thermal)

        # carry the *last* noisy realization forward as the next state (one
        # MD trajectory, not a density-matrix simulation) -- the other
        # ENSEMBLE-1 realizations exist only to estimate purity at this step
        psi = realizations[-1] / np.linalg.norm(realizations[-1])

    df_md = pd.DataFrame(data)
    df_md.set_index("Step", inplace=True)
    corr_matrix = df_md.corr(method="pearson")
    return df_md, corr_matrix
