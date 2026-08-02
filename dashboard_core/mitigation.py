"""
Real Zero-Noise Extrapolation (ZNE) error mitigation, wired to the same
engine and real noise channels (dense_evolution.NoiseModel) the rest of
the Composer uses. Noise is scaled by running the real channel at
noise_p, 2*noise_p, 3*noise_p and Richardson-extrapolating the measured
Pauli expectation back to zero noise via dense_evolution's own
zero_noise_extrapolation -- not a fabricated mitigated curve.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

import dense_evolution as de

__all__ = ['MitigationResult', 'run_zne_mitigation', 'DensityMatrixZNEResult', 'run_density_matrix_zne']

_DEFAULT_NOISE_FACTORS = (1.0, 2.0, 3.0)


@dataclass
class MitigationResult:
    n_qubits: int
    pauli_string: str
    ideal_expectation: float
    noise_factors: list
    noisy_expectations: list
    zne_extrapolated: float


def run_zne_mitigation(
    qasm_text: str,
    pauli_string: str,
    noise_model: str,
    noise_p: float,
    seed: Optional[int] = None,
    noise_factors=_DEFAULT_NOISE_FACTORS,
    n_trials: int = 200,
) -> MitigationResult:
    """Real ZNE: <P> is measured at the real ideal state and at the real
    channel applied at noise_p * each factor, then Richardson-
    extrapolated to zero noise.

    NoiseModel.apply_to_sv is a *stochastic single-shot* Kraus draw (one
    random outcome per call, not the channel's averaged/ensemble
    behavior) -- feeding a single draw straight into ZNE gives a
    discontinuous, meaningless curve (verified: bitflip/depolarizing
    jumped 1.0 -> 0.0 -> 0.0 across noise scales instead of decaying
    smoothly). <P> under a Kraus channel is Tr(rho P) = mean over the
    channel's trajectories, so each scale here averages n_trials
    independent stochastic draws -- the actual expectation value ZNE is
    defined against, not one random sample of it.

    pauli_string uses dense_evolution's own qubit-0-is-position-0
    convention (same as pauli_expectation), independent of Qiskit's
    little-endian display convention used elsewhere on this page --
    this function never touches a Qiskit-ordered array.
    """
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_text)
    n_qubits = qiskit_circuit.num_qubits
    if len(pauli_string) != n_qubits:
        raise ValueError(f"pauli_string length {len(pauli_string)} != n_qubits {n_qubits}")
    if noise_model not in de.NoiseModel.MODELS:
        raise ValueError(f"unknown noise model {noise_model!r}, must be one of {de.NoiseModel.MODELS}")

    sim, _ = de.run_qiskit_circuit(qiskit_circuit, use_float32=False)
    sv_ideal = np.asarray(sim.sv)
    ideal_expectation = float(np.real(de.pauli_expectation(sv_ideal, pauli_string)))

    rng = np.random.default_rng(seed)
    noisy_expectations = []
    for factor in noise_factors:
        scaled_p = min(noise_p * factor, 1.0)
        trial_values = np.empty(n_trials)
        for i in range(n_trials):
            sv_noisy = de.NoiseModel.apply_to_sv(
                sv_ideal.copy(), n_qubits, noise_model, scaled_p, rng=rng,
            )
            trial_values[i] = np.real(de.pauli_expectation(sv_noisy, pauli_string))
        noisy_expectations.append(float(trial_values.mean()))

    zne_value = float(de.zero_noise_extrapolation(noisy_expectations, list(noise_factors)))

    return MitigationResult(
        n_qubits=n_qubits,
        pauli_string=pauli_string,
        ideal_expectation=ideal_expectation,
        noise_factors=list(noise_factors),
        noisy_expectations=noisy_expectations,
        zne_extrapolated=zne_value,
    )


@dataclass
class DensityMatrixZNEResult:
    n_qubits: int
    noise_factors: list
    fidelity_raw: float          # Uhlmann fidelity: ideal vs raw noisy (base scale) rho
    fidelity_corrected: float    # Uhlmann fidelity: ideal vs ZNE-corrected rho


def run_density_matrix_zne(
    qasm_text: str,
    noise_model: str,
    noise_p: float,
    seed: Optional[int] = None,
    noise_factors=_DEFAULT_NOISE_FACTORS,
    n_trials: int = 200,
) -> DensityMatrixZNEResult:
    """Density-matrix ZNE (dense_evolution.zne_density_matrix): builds a
    real Monte-Carlo density-matrix estimate at each noise scale (the
    mean of n_trials |psi_k><psi_k| projectors from independent
    NoiseModel draws -- a real ensemble reconstruction, not a single
    trajectory), extrapolates to zero noise, and projects onto the
    nearest physical (PSD, trace-1) density matrix internally.

    Graded (never fed back into the extrapolation) against the true
    ideal density matrix via dense_evolution.uhlmann_fidelity, so the
    reported improvement is an honest measurement of whether the
    correction actually helped -- matches the pattern in
    zne_density_matrix's own docstring (experiments/matrix_healing_zne.py:
    raw ~0.865, corrected ~0.947 on a 2-qubit Bell state).
    """
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_text)
    n_qubits = qiskit_circuit.num_qubits
    if noise_model not in de.NoiseModel.MODELS:
        raise ValueError(f"unknown noise model {noise_model!r}, must be one of {de.NoiseModel.MODELS}")

    sim, _ = de.run_qiskit_circuit(qiskit_circuit, use_float32=False)
    sv_ideal = np.asarray(sim.sv)
    rho_ideal = np.outer(sv_ideal, sv_ideal.conj())

    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    rhos_at_scales = []
    for factor in noise_factors:
        scaled_p = min(noise_p * factor, 1.0)
        rho_acc = np.zeros((dim, dim), dtype=complex)
        for _ in range(n_trials):
            sv_noisy = de.NoiseModel.apply_to_sv(
                sv_ideal.copy(), n_qubits, noise_model, scaled_p, rng=rng,
            )
            rho_acc += np.outer(sv_noisy, sv_noisy.conj())
        rho_acc /= n_trials
        rhos_at_scales.append(rho_acc)

    rho_corrected = np.asarray(de.zne_density_matrix(rhos_at_scales, list(noise_factors)))

    fidelity_raw = float(de.uhlmann_fidelity(rho_ideal, rhos_at_scales[0]))
    fidelity_corrected = float(de.uhlmann_fidelity(rho_ideal, rho_corrected))

    return DensityMatrixZNEResult(
        n_qubits=n_qubits,
        noise_factors=list(noise_factors),
        fidelity_raw=fidelity_raw,
        fidelity_corrected=fidelity_corrected,
    )
